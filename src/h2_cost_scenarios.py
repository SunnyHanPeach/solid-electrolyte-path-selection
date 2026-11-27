"""Compare diagnostic budget scenarios without any new DFT or model training."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from zipfile import ZipFile

import numpy as np

from audit_phase_h2_evaluation import digest, rows
from phase_b1_inference import load_calculators
from run_force_weighting import score_profiles


def save_csv(path, values):
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(values[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def calibration(archive):
    jobs = archive["jobs"]
    anchors = [r["solver_wall_s"] for r in jobs if r["material"] == "mp-1191903"
               and r["hardware"] == "RTX 3090" and r["grid"] == "2x2x2" and r["cutoff_Ry"] == "50/400"]
    using = {r["role"]: r["solver_wall_s"] for r in jobs
             if (r["phase"], r["role"]) in {("fixed_geometry_precision", "middle_k222"),
                                            ("fixed_geometry_precision", "middle_k333")}
             or (r["phase"], r["role"]) == ("grid_pair", "endpoint")}
    endpoint_222 = next(r["solver_wall_s"] for r in jobs if r["phase"] == "io_k222")
    ratios = sorted([using["middle_k333"] / using["middle_k222"], using["endpoint"] / endpoint_222])
    powers = [np.log(r) / np.log(27/8) for r in (min(ratios), median(ratios), max(ratios))]
    settings = zip(("low", "middle", "high"), (min(anchors), median(anchors), max(anchors)),
                   (1., 1.5, 2.), powers, (8, 16, 32), (6, 12, 24), (3, 8, 16),
                   (.25, .50, .85), (.00, .10, .25))
    return {
        name: dict(anchor_seconds=t, atom_power=a, grid_power=k, endpoint_steps_per_side=e,
                   neb_updates=n, ci_updates=c, warm_band_fraction=w, retry_fraction=r)
        for name, t, a, k, e, n, c, w, r in settings
    }, ratios


def compute_cost(atoms, k, images, config, scenario):
    atom_ratio = atoms / 87.
    a = config["atom_power"]
    if atom_ratio < 1 and scenario != "middle":
        a = 3 - a
    atom_only = config["anchor_seconds"] * atom_ratio**a
    with_grid = atom_only * (k/8.)**config["grid_power"]
    image_h = with_grid / 3600
    internals = images - 2
    endpoint = 2 * config["endpoint_steps_per_side"] * image_h
    neb = internals * (1 + (config["neb_updates"]-1)*config["warm_band_fraction"]) * image_h
    ci = internals * config["ci_updates"] * config["warm_band_fraction"] * image_h
    retry = (endpoint + neb + ci) * config["retry_fraction"]
    return dict(atom_only_scf_h=atom_only/3600, atom_and_grid_scf_h=image_h,
                endpoints_h=endpoint, neb_h=neb, ci_h=ci, retry_h=retry,
                scenario_total_h=endpoint + neb + ci + retry)


def disagreement(results, geometry, weights, threads):
    from ase.io import read
    from mace import data
    import torch

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    b5 = json.loads((results / "phase_b5_r/manifest.json").read_bytes())
    if digest(geometry) != b5["geometry_sha256"]["candidate_images.xyz"]:
        raise ValueError("Candidate initial geometries changed")
    provenance = json.loads((results / "phase_b3_a_ensemble/run_manifest.json").read_bytes())
    if digest(weights) != provenance["checkpoint_archive_sha256"]:
        raise ValueError("B0 seed-spread archive changed")
    models, weight_ids = {}, {}
    with ZipFile(weights) as archive:
        for row in provenance["checkpoints"]:
            seed = int(row["checkpoint_id"].split("-")[-1])
            raw = archive.read(row["member"])
            if sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("Checkpoint differs")
            models[seed] = (raw, None)
            weight_ids[seed] = row["sha256"]
    calculators = load_calculators(models, torch)
    atoms = read(geometry, ":", format="extxyz")
    species = data.KeySpecification()
    calc = calculators[17]
    graphs, frames = {}, []
    for i, frame in enumerate(atoms):
        config = data.config_from_atoms(frame, key_specification=species, head_name=calc.head)
        graphs[i] = data.AtomicData.from_config(
            config, z_table=calc.z_table, cutoff=calc.r_max, heads=calc.available_heads)
        frames.append(dict(edge_id=frame.info["edge_id"], image=int(frame.info["image"])))
    results_by_edge = {}
    for seed, calculator in calculators.items():
        pred = score_profiles(calculator.models[0], graphs, frames, "cpu", torch)
        curves = {}
        for row in pred:
            curves.setdefault(row["edge_id"], {})[row["image"]] = row["energy_eV"]
        for key, values in curves.items():
            if set(values) != set(range(len(values))):
                raise ValueError("Unordered initial trajectory")
            results_by_edge.setdefault(key, {})[seed] = max(values.values()) - values[0]
        print(f"Seed {seed}: {len(pred)} unlabelled candidate images scored.", flush=True)
    spreads = [dict(edge_id=key, **{f"seed{seed}_barrier_eV": v[seed] for seed in sorted(v)},
                    barrier_sd_eV=float(np.std(list(v.values()), ddof=1)))
               for key, v in sorted(results_by_edge.items())]
    return spreads, weight_ids


def simulate(pool, methods, costs, budget, keys, refs, kernels):
    completed = {}
    for method in methods:
        order = methods[method]
        left = budget
        chosen = []
        while True:
            affordable = [i for i in pool if i not in chosen and costs[keys[i]] <= left + 1e-9]
            if not affordable:
                break
            if method in kernels:
                k = kernels[method]
                anchors = refs + chosen
                pick = min(affordable, key=lambda i: (float(k[i, anchors].max()), keys[i]))
            else:
                pick = min(affordable, key=order.__getitem__)
            chosen.append(pick)
            left -= costs[keys[pick]]
        completed[method] = [keys[i] for i in chosen]
    return completed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "geometry", "weights", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    r = args.results
    history = r / "phase_h2_cost_reuse/recorded_costs.json"
    candidate_source = r / "phase_b4_r2_preflight/h1c_cost_inputs.csv"
    audit = r / "phase_h1/structure_output/candidate_ranking_and_audit.csv"
    b5 = r / "phase_b5_r/independent_candidate_ranking.csv"
    kernel_index = r / "phase_b5_r/kernel_matrices/matrix_index.csv"
    kernel_data = r / "phase_b5_r/kernel_matrices/mean.npz"
    scenarios, ratios = calibration(json.loads(history.read_bytes()))
    eligible = {row["edge_id"]: not row["structural_exclusion"] for row in rows(audit)}
    data, costs = [], {name: {} for name in scenarios}
    for row in rows(candidate_source):
        key = row["edge_id"]
        record = dict(edge_id=key, eligible_after_H1_filter=eligible[key], atoms=int(row["atoms"]),
                      nominal_kpoints=int(row["nominal_k_points"]), images=int(row["initial_images"]),
                      unbenchmarked_pseudos=row["elements_without_previously_benchmarked_upf"])
        for name, config in scenarios.items():
            values = compute_cost(record["atoms"], record["nominal_kpoints"], record["images"], config, name)
            costs[name][key] = values["scenario_total_h"]
            record.update({f"{name}_{k}": v for k, v in values.items()})
        if not costs["low"][key] <= costs["middle"][key] <= costs["high"][key]:
            raise ValueError("Scenarios are not ordered")
        data.append(record)

    spreads, hashes = disagreement(r, args.geometry, args.weights, args.threads)
    sd = {row["edge_id"]: row["barrier_sd_eV"] for row in spreads}
    if set(sd) != set(eligible) or len(eligible) != 24:
        raise ValueError("Missing candidate scores or geometries")
    bvse = {row["edge_id"]: float(row["source_BVSE_eV"]) for row in rows(b5)
            if row["aggregation"] == "mean" and row["method"] == "BVSE"}
    index = rows(kernel_index)
    keys = [row["edge_id"] for row in index]
    refs = [i for i, row in enumerate(index) if row["cohort"] == "reference"]
    pool = [i for i, row in enumerate(index) if row["cohort"] == "candidate" and eligible[keys[i]]]
    with np.load(kernel_data, allow_pickle=False) as z:
        soap, activation = z["soap_full"], z["activation_full"]
    methods = {
        "Random": {i: (sha256(f"h2-random-01/{keys[i]}".encode()).hexdigest(), keys[i]) for i in pool},
        "BVSE": {i: (bvse[keys[i]], keys[i]) for i in pool},
        "D0 coverage": {i: (float(soap[i, refs].max()), keys[i]) for i in pool},
        "SOAP": None,
        "SOAP+Activation": None,
        "Model Disagreement": {i: (-sd[keys[i]], keys[i]) for i in pool},
    }
    reference = compute_cost(87, 8, 5, scenarios["middle"], "middle")["scenario_total_h"]
    budgets = [reference * n for n in (1, 2, 4)]
    sets, contrasts = [], []
    for budget in budgets:
        for scenario in scenarios:
            acquired = simulate(pool, methods, costs[scenario], budget, keys, refs,
                                {"SOAP": soap, "SOAP+Activation": (soap+activation)/2})
            for method, subset in acquired.items():
                spent = sum(costs[scenario][path] for path in subset)
                if spent > budget + 1e-9:
                    raise ValueError("Budget overspend")
                sets.append(dict(budget_h=budget, scenario=scenario, method=method, paths=len(subset),
                                 scenario_spend_h=spent, edge_ids="|".join(subset)))
            a, b = set(acquired["SOAP"]), set(acquired["SOAP+Activation"])
            contrasts.append(dict(budget_h=budget, scenario=scenario, same_set=a == b,
                                  soap_paths=len(a), combined_paths=len(b), shared_paths=len(a & b)))
    args.output.mkdir(parents=True, exist_ok=True)
    save_csv(args.output / "candidate_cost_scenarios.csv", data)
    save_csv(args.output / "candidate_model_disagreement.csv", spreads)
    save_csv(args.output / "budget_sensitivity.csv", sets)
    meta = dict(
        inputs_sha256={file.name: digest(file) for file in
                       (history, candidate_source, audit, b5, kernel_index, kernel_data, args.geometry)},
        seed_spread_model_hashes=hashes, candidate_paths=len(data), currently_eligible=len(pool),
        k333_over_k222_same_material_ratios=ratios, design_assumptions=scenarios,
        budgets_reference_scenario_h=budgets, activation_sets=contrasts,
        scope="Illustrative candidate acquisition under assumed solver hours; actual caps remain unset.",
    )
    with (args.output / "cost_scenario_manifest.json").open("x", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(json.dumps(contrasts), flush=True)


if __name__ == "__main__":
    main()
