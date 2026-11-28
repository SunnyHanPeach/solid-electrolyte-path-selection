"""Prioritize cost measurements and size a future independent barrier panel."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from h2_cost_scenarios import save_csv, simulate
from replay_h2_design_freeze import methods_for, rows


def budget_scan(task):
    model, budget, base, pool, methods, keys, refs, kernels = task

    def decision(cost):
        sets = {method: frozenset(chosen) for method, chosen in
                simulate(pool, methods, cost, budget, keys, refs, kernels).items()}
        soap, combined = sets["SOAP"], sets["SOAP+Activation"]
        return sets, bool(soap and combined and soap != combined)

    baseline, contrast = decision(base)
    records = []
    for edge, hours in sorted(base.items()):
        costs = base.copy()
        counters = Counter()
        flips = {}
        for step in range(-60, 61):
            if step == 0:
                continue
            percent = step / 2
            costs[edge] = hours * (1 + percent / 100)
            result, present = decision(costs)
            counters["any_strategy_changed"] += result != baseline
            counters["contrast_different"] += present != contrast
            if present != contrast:
                flips[percent] = result
        first_flip = min(flips, key=abs) if flips else None
        nearest_sets = {} if first_flip is None else flips[first_flip]
        records.append(dict(
            model=model, budget_h=budget, edge_id=edge, baseline_cost_h=hours,
            baseline_soap_contrast=contrast, scenario_count=120,
            any_strategy_changed=counters["any_strategy_changed"],
            contrast_changed=counters["contrast_different"],
            nearest_contrast_flip_pct="" if first_flip is None else first_flip,
            nearest_contrast_flip_cost_h="" if first_flip is None else hours*(1+first_flip/100),
            nearest_lower_cost_flip_pct=max((x for x in flips if x < 0), default=""),
            nearest_higher_cost_flip_pct=min((x for x in flips if x > 0), default=""),
            soap_at_nearest_flip="|".join(sorted(nearest_sets.get("SOAP", []))),
            combined_at_nearest_flip="|".join(sorted(nearest_sets.get("SOAP+Activation", []))),
        ))
    return records


def panel_plan(task):
    contrast, family_seed_values, sizes, effects, margin, trials, inner, rng_seed = task
    rng = np.random.default_rng(rng_seed)
    residuals = family_seed_values - family_seed_values.mean(axis=0)
    records = []
    for n in sizes:
        hits = np.zeros(len(effects), dtype=int)
        for first in range(0, trials, 8):
            m = min(8, trials - first)
            values = residuals[rng.integers(len(residuals), size=(m, n))]
            seed_means = values.mean(axis=1)
            family_macro = values.mean(axis=2)
            indices = rng.integers(n, size=(m, inner, n), dtype=np.int32)
            bootstrap = family_macro[np.arange(m)[:, None, None], indices].mean(axis=2)
            lower = np.quantile(bootstrap, 0.025, axis=1)
            for i, effect in enumerate(effects):
                hits[i] += np.count_nonzero((lower + effect > margin) &
                                           np.all(seed_means + effect > 0, axis=1))
        for effect, hit in zip(effects, hits):
            probability = int(hit) / trials
            records.append(dict(
                dispersion_source=contrast, independent_families=n, assumed_gain_meV=effect,
                practical_margin_meV=margin, trials=trials, family_bootstraps_per_trial=inner,
                probability_pass=probability, monte_carlo_se=(probability*(1-probability)/trials)**.5,
                source_sd_family_macro_meV=float(family_seed_values.mean(axis=1).std(ddof=1)),
            ))
    return records


def plot(output, scans, plans):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 3.3), constrained_layout=True)
    selected = [r for r in scans if 133 < r["budget_h"] < 134 and r["nearest_contrast_flip_pct"] != ""]
    if selected:
        ys = sorted({r["edge_id"] for r in selected})
        for j, model in enumerate(("H2 middle", "Public grid proxy")):
            specific = [r for r in selected if r["model"] == model]
            points = [(r[field], ys.index(r["edge_id"]) + (j-.5)*.16)
                      for r in specific for field in ("nearest_lower_cost_flip_pct", "nearest_higher_cost_flip_pct")
                      if r[field] != ""]
            left.scatter([p[0] for p in points], [p[1] for p in points],
                         label=model, marker=("o", "s")[j], color=("#175676", "#BB4E20")[j])
        left.set_yticks(range(len(ys)), ys)
        left.axvline(0, color="0.6", lw=.8)
        left.legend(frameon=False, loc="best", fontsize=8)
    left.set_title("A  Cost shifts that change the SOAP contrast", loc="left")
    left.set_xlabel("Single-path cost shift (%) at 133.67 h")
    for label, color in (("absolute", "#175676"), ("net", "#BB4E20")):
        subset = [r for r in plans if r["dispersion_source"] == label and r["assumed_gain_meV"] == .5]
        right.plot([r["independent_families"] for r in subset], [r["probability_pass"] for r in subset],
                   "o-", label=f"Historical {label} dispersion", color=color, ms=3)
    right.axhline(.80, color="0.6", lw=.8, linestyle=":")
    right.set(ylim=(0, 1), xlabel="Independent evaluation families", ylabel="Simulated pass probability")
    right.set_title("B  Planning scenario for a +0.50 meV gain", loc="left")
    right.legend(frameon=False, loc="lower right", fontsize=8)
    fig.savefig(output / "planning_figure.png", dpi=220)
    fig.savefig(output / "planning_figure.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    results = args.results
    h2 = results / "phase_h2_boundary_cost"
    b5 = results / "phase_b5_r"
    grid_file = results / "phase_h3_public_cost/candidate_cost_overlay.csv"
    family_file = results / "public_neb_panel/mp753143_effects/family_gains.csv"
    protocol_file = results / "phase_h2_design_freeze/protocol.json"
    protocol = json.loads(protocol_file.read_bytes())
    budgets = json.loads((h2 / "cost_scenario_manifest.json").read_bytes())["budgets_reference_scenario_h"]
    keys_info = rows(b5 / "kernel_matrices/matrix_index.csv")
    keys = [r["edge_id"] for r in keys_info]
    refs = [i for i, row in enumerate(keys_info) if row["cohort"] == "reference"]
    models = {"H2 middle": {}, "Public grid proxy": {}}
    for row in rows(grid_file):
        models["H2 middle"][row["edge_id"]] = float(row["H2_middle_estimate_h"])
        models["Public grid proxy"][row["edge_id"]] = float(row["public_grid_proxy_estimate_h"])
    pool = [i for i, row in enumerate(keys_info) if row["cohort"] == "candidate" and keys[i] in models["H2 middle"]]
    if len(pool) != protocol["prospective_expected_paths"] or set(keys[i] for i in pool) != models["H2 middle"].keys():
        raise ValueError("The frozen candidate pool has changed")
    bvse = {r["edge_id"]: float(r["source_BVSE_eV"]) for r in rows(b5 / "independent_candidate_ranking.csv")
            if r["aggregation"] == "mean" and r["method"] == "BVSE"}
    spreads = {r["edge_id"]: float(r["barrier_sd_eV"]) for r in rows(h2 / "candidate_model_disagreement.csv")}
    with np.load(b5 / "kernel_matrices/mean.npz", allow_pickle=False) as f:
        soap = f["soap_full"]
        kernels = {"SOAP": soap, "SOAP+Activation": (soap + f["activation_full"])/2}
    methods = methods_for(keys, pool, refs, bvse, spreads, soap, protocol["primary_random_order_salt"])
    tasks = [(name, budget, costs, pool, methods, keys, refs, kernels)
             for name, costs in models.items() for budget in budgets]

    history = rows(family_file)
    seeds = protocol["training_seeds"]
    sizes = [12, 20, 32, 48, 64, 96, 128]
    effects = [0.40, 0.50, 0.75, 1.00]
    panels = []
    for j, contrast in enumerate(("absolute", "net")):
        cohort = [r for r in history if r["contrast"] == contrast]
        if len(cohort) != 42 or len({r["family"] for r in cohort}) != 42:
            raise ValueError("The archived family-effect cohort has changed")
        values = np.array([[float(r[f"seed{s}_gain_meV"]) for s in seeds] for r in cohort])
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite historic gain")
        panels.append((contrast, values, sizes, effects, protocol["practical_margin_meV"],
                       768, 384, 20261129+j))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        cost_jobs = [executor.submit(budget_scan, task) for task in tasks]
        power_jobs = [executor.submit(panel_plan, task) for task in panels]
        scans = [row for job in cost_jobs for row in job.result()]
        plans = [row for job in power_jobs for row in job.result()]
    args.output.mkdir(parents=True, exist_ok=True)
    save_csv(args.output / "single_candidate_cost_thresholds.csv", scans)
    save_csv(args.output / "independent_panel_scenarios.csv", plans)
    plot(args.output, scans, plans)
    inputs = [grid_file, family_file, protocol_file, h2 / "cost_scenario_manifest.json",
              h2 / "candidate_model_disagreement.csv", b5 / "independent_candidate_ranking.csv",
              b5 / "kernel_matrices/matrix_index.csv", b5 / "kernel_matrices/mean.npz"]
    (args.output / "manifest.json").write_text(json.dumps(
        dict(cost_rows=len(scans), planning_rows=len(plans), random_seeds=[20261129, 20261130],
             source_sha256={str(p.relative_to(results)): sha256(p.read_bytes()).hexdigest() for p in inputs}),
        indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
