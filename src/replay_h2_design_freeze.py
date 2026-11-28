"""Reproduce procurement decisions and replay archived single-path utility."""

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from h2_cost_scenarios import save_csv, simulate


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def ordering(keys, pool, values):
    return {i: (values[keys[i]], keys[i]) for i in pool}


def random_order(keys, pool, salt):
    return {i: (sha256(f"{salt}/{keys[i]}".encode()).hexdigest(), keys[i]) for i in pool}


def methods_for(keys, pool, refs, bvse, spread, soap, salt):
    return {
        "Random": random_order(keys, pool, salt),
        "BVSE": ordering(keys, pool, bvse),
        "D0 coverage": {i: (float(soap[i, refs].max()), keys[i]) for i in pool},
        "SOAP": None,
        "SOAP+Activation": None,
        "Model Disagreement": ordering(keys, pool, {k: -v for k, v in spread.items()}),
    }


def stress_multiplier(path, draw, fraction):
    token = sha256(f"h2-cost-stress-v1/{draw}/{path}".encode()).digest()[:8]
    unit = int.from_bytes(token, "big") / 2**64
    return 1 + fraction * (2 * unit - 1)


def same_set(a, b):
    return set(a) == set(b)


def run_stress(replay, base_cost, budgets, fractions, draws):
    results = []
    for budget in budgets:
        nominal = replay(base_cost, budget)
        for fraction in fractions:
            cases = {
                f"common_minus_{fraction:.0%}": [{p: c*(1-fraction) for p, c in base_cost.items()}],
                f"common_plus_{fraction:.0%}": [{p: c*(1+fraction) for p, c in base_cost.items()}],
                f"candidate_specific_{fraction:.0%}": [
                    {p: c*stress_multiplier(p, draw, fraction) for p, c in base_cost.items()}
                    for draw in range(draws)
                ],
            }
            for name, cost_vectors in cases.items():
                sets = [replay(cost, budget) for cost in cost_vectors]
                for method, base in nominal.items():
                    selections = [run[method] for run in sets]
                    results.append(dict(
                        budget_h=budget, stress=name, method=method, runs=len(sets),
                        changed=sum(not same_set(base, pick) for pick in selections),
                        min_paths=min(map(len, selections)), max_paths=max(map(len, selections)),
                        empty=sum(not pick for pick in selections),
                        distinct_from_soap_plus_activation=sum(
                            not same_set(run[method], run["SOAP+Activation"]) for run in sets
                        ),
                    ))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.root
    results = root / "results"
    inventory = {
        "protocol": results / "phase_h2_design_freeze/protocol.json",
        "costs": results / "phase_h2_boundary_cost/candidate_cost_scenarios.csv",
        "original": results / "phase_h2_boundary_cost/budget_sensitivity.csv",
        "order": results / "phase_b5_r/independent_candidate_ranking.csv",
        "index": results / "phase_b5_r/kernel_matrices/matrix_index.csv",
        "kernels": results / "phase_b5_r/kernel_matrices/mean.npz",
        "prospective_spread": results / "phase_h2_boundary_cost/candidate_model_disagreement.csv",
        "historical_spread": results / "phase_b3_a_ensemble/per_seed_barriers.csv",
        "historical_geometry": root / "experiments/phase1/candidate_path_features.csv",
        "historical_labels": results / "phase_b3_a/path_utility_labels.csv",
    }
    protocol = json.loads(inventory["protocol"].read_bytes())
    index = rows(inventory["index"])
    keys = [row["edge_id"] for row in index]
    refs = [i for i, row in enumerate(index) if row["cohort"] == "reference"]
    with np.load(inventory["kernels"], allow_pickle=False) as data:
        soap, activation = data["soap_full"], data["activation_full"]
    if (len(set(keys)) != len(keys) or not refs or soap.shape != (len(keys), len(keys))
            or activation.shape != soap.shape or not np.isfinite(soap).all() or not np.isfinite(activation).all()):
        raise ValueError("Invalid matrix or index")
    kernels = {"SOAP": soap, "SOAP+Activation": (soap+activation)/2}
    cost_rows = {x["edge_id"]: x for x in rows(inventory["costs"])}
    eligible = {k for k, x in cost_rows.items() if x["eligible_after_H1_filter"] == "True"}
    pool = [i for i, row in enumerate(index) if row["cohort"] == "candidate" and keys[i] in eligible]
    if {keys[i] for i in pool} != eligible or len(pool) != protocol["prospective_expected_paths"]:
        raise ValueError("Changed prospective pool")
    bvse = {x["edge_id"]: float(x["source_BVSE_eV"]) for x in rows(inventory["order"])
            if x["aggregation"] == "mean" and x["method"] == "BVSE"}
    spread = {x["edge_id"]: float(x["barrier_sd_eV"]) for x in rows(inventory["prospective_spread"])}
    methods = methods_for(keys, pool, refs, bvse, spread, soap, protocol["primary_random_order_salt"])
    if list(methods) != protocol["methods"]:
        raise ValueError("Changed methods")

    def replay(cost, budget):
        if any(not math.isfinite(v) or v <= 0 for v in cost.values()):
            raise ValueError("Nonpositive or nonfinite scenario cost")
        picks = simulate(pool, methods, cost, budget, keys, refs, kernels)
        if any(sum(cost[edge] for edge in chosen) > budget + 1e-9 for chosen in picks.values()):
            raise ValueError("Scenario budget exceeded")
        return picks

    original = rows(inventory["original"])
    budgets = sorted({float(row["budget_h"]) for row in original})
    costs = {scenario: {edge: float(cost_rows[edge][f"{scenario}_scenario_total_h"]) for edge in eligible}
             for scenario in ("low", "middle", "high")}
    recovered = {(b, s): replay(cost, b) for b in budgets for s, cost in costs.items()}
    for row in original:
        scenario = row["scenario"]
        actual = recovered[float(row["budget_h"]), scenario][row["method"]]
        expected = row["edge_ids"].split("|") if row["edge_ids"] else []
        spend = sum(costs[scenario][path] for path in actual)
        if (actual != expected or len(actual) != int(row["paths"])
                or not math.isclose(spend, float(row["scenario_spend_h"]), abs_tol=1e-9, rel_tol=0)):
            raise ValueError(f"Changed archived procurement decision for {row}")
    if len({(r["budget_h"], r["scenario"], r["method"]) for r in original}) != len(original) or len(original) != 54:
        raise ValueError("Expected 54 archived procurement decisions")
    stress = run_stress(replay, costs["middle"], budgets, protocol["cost_disturbance_fractions"],
                        protocol["candidate_specific_stress_draws"])

    historic = [i for i, row in enumerate(index) if row["cohort"] == "historical"]
    historic_edges = {keys[i] for i in historic}
    if len(historic) != protocol["historical_expected_paths"]:
        raise ValueError("Changed historical feature pool")
    bvse_hist = {x["edge_id"]: float(x["bvse_forward_eV"]) for x in rows(inventory["historical_geometry"])
                 if x["edge_id"] in historic_edges}
    seed_barriers = {key: {} for key in historic_edges}
    for row in rows(inventory["historical_spread"]):
        key, seed = row["edge_id"], int(row["seed"])
        if key in historic_edges:
            if seed in seed_barriers[key]:
                raise ValueError(f"Duplicate seed score for {key}")
            seed_barriers[key][seed] = float(row["forward_barrier_meV"])
    if any(set(v) != set(protocol["training_seeds"]) for v in seed_barriers.values()):
        raise ValueError("Cannot replay disagreement without the same three seeds")
    disagreement = {key: stdev(v.values()) for key, v in seed_barriers.items()}
    choices = methods_for(keys, historic, refs, bvse_hist, disagreement, soap,
                          protocol["primary_random_order_salt"])
    count = protocol["historical_paths_per_policy"]
    unit_cost = dict.fromkeys(historic_edges, 1.0)
    selected = simulate(historic, choices, unit_cost, count, keys, refs, kernels)
    random_picks = {
        i: keys[min(historic, key=random_order(keys, historic, f"h2-random-{i:02d}").__getitem__)]
        for i in range(1, protocol["random_order_draws"]+1)
    }
    # Only now attach utility to choices that have already been made.
    labels = {x["edge_id"]: x for x in rows(inventory["historical_labels"])}
    if (set(labels) != historic_edges or
            len({(x["evaluation_panel"], x["cohort"], x["seeds"]) for x in labels.values()}) != 1 or count != 1):
        raise ValueError("Expected comparable utility labels for the single-path replay")

    def observed(path):
        row = labels[path]
        return {k: float(row[k]) for k in ("Delta_abs", "Delta_net")}

    historical = [dict(method=method, edge_id=paths[0], **observed(paths[0]))
                  for method, paths in selected.items()]
    random_rows = [dict(draw=i, edge_id=path, **observed(path)) for i, path in random_picks.items()]
    random_summary = {k: dict(
        pool_mean=mean(float(row[k]) for row in labels.values()),
        draw_mean=mean(row[k] for row in random_rows),
        draw_P05=float(np.quantile([row[k] for row in random_rows], .05)),
        draw_P95=float(np.quantile([row[k] for row in random_rows], .95)),
    ) for k in ("Delta_abs", "Delta_net")}
    args.output.mkdir(parents=True, exist_ok=True)
    save_csv(args.output / "cost_stress.csv", stress)
    save_csv(args.output / "historical_single_path_replay.csv", historical)
    save_csv(args.output / "historical_random_draws.csv", random_rows)
    meta = dict(
        verified_archived_decisions=len(original), prospective_pool=sorted(eligible),
        historical_distinct_selections=len({paths[0] for paths in selected.values()}),
        random_summary=random_summary, input_sha256={k: sha256(p.read_bytes()).hexdigest()
                                                     for k, p in inventory.items()},
    )
    with (args.output / "replay_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"historical": historical, "random": random_summary}, indent=2))


if __name__ == "__main__":
    main()
