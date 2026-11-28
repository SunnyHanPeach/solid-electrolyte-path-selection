"""Replay the frozen H2 choice rules with a public grid-cost sensitivity model."""

import argparse
import gzip
from hashlib import md5, sha256
import json
import math
from pathlib import Path
import re
import tarfile

import numpy as np

from h2_cost_scenarios import compute_cost, save_csv, simulate
from replay_h2_design_freeze import methods_for, rows


def interpolate(point, published):
    lower, upper = next(
        (a, b) for a, b in zip(published, published[1:]) if a[0] <= point <= b[0]
    )
    fraction = math.log(point / lower[0]) / math.log(upper[0] / lower[0])
    return lower[1] * (upper[1] / lower[1]) ** fraction


def extract_neb(path):
    if md5(path.read_bytes()).hexdigest() != "1f8bcff0655e09b6fbd072993b9f2c3c":
        raise ValueError("Public VASP source archive differs")
    result = []
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            match = re.fullmatch(r"1\.surface/1\.diffusion/([^/]+)/(\d+)/OUTCAR\.gz", member.name)
            if not match or not member.isfile():
                continue
            text = gzip.decompress(archive.extractfile(member).read()).decode("ascii", errors="replace")
            elapsed = re.findall(r"Elapsed time \(sec\):\s*([0-9.]+)", text)
            ranks_text = next((line.strip() for line in text.splitlines()
                               if re.search(r"running (?:on|\d)", line, re.I)), "")
            result.append(dict(
                public_path=match[1], image=int(match[2]), elapsed_seconds=elapsed[-1] if elapsed else "",
                electronic_loop_count=len(re.findall(r"^\s*LOOP:", text, re.M)),
                ionic_loop_count=len(re.findall(r"^\s*LOOP\+:", text, re.M)),
                vasp_parallel_header=ranks_text,
                input_member=member.name,
            ))
    return sorted(result, key=lambda r: (r["public_path"], r["image"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "public_vasp_archive", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    results = args.results
    stage = results / "phase_h3_public_cost"
    config = json.loads((stage / "protocol.json").read_bytes())
    public_source = stage / "figure2_walltime.csv"
    published = sorted((float(x["k_spacing_inv_A"]), float(x["relative_wall_time"]))
                       for x in rows(public_source))
    eval_spec = config["eval06_fixed_geometry"]
    factor = (interpolate(eval_spec["dense_spacing_inv_A"], published)
              / interpolate(eval_spec["reference_spacing_inv_A"], published))
    predictions = [
        dict(anchor=name, baseline_wall_seconds=seconds, public_predicted_ratio=factor,
             public_predicted_dense_wall_seconds=seconds*factor)
        for name, seconds in eval_spec["reference_wall_seconds"].items()
    ]

    h2 = results / "phase_h2_boundary_cost"
    old = json.loads((h2 / "cost_scenario_manifest.json").read_bytes())
    original = rows(h2 / "budget_sensitivity.csv")
    eligible = {x["edge_id"]: x for x in rows(h2 / "candidate_cost_scenarios.csv")
                if x["eligible_after_H1_filter"] == "True"}
    p = results / "phase_b5_r/kernel_matrices"
    index = rows(p / "matrix_index.csv")
    keys = [r["edge_id"] for r in index]
    refs = [i for i, row in enumerate(index) if row["cohort"] == "reference"]
    pool = [i for i, row in enumerate(index) if row["cohort"] == "candidate" and keys[i] in eligible]
    if {keys[i] for i in pool} != eligible.keys():
        raise ValueError("The prospective pool changed")
    with np.load(p / "mean.npz", allow_pickle=False) as data:
        soap, activation = data["soap_full"], data["activation_full"]
    bvse = {x["edge_id"]: float(x["source_BVSE_eV"])
            for x in rows(results / "phase_b5_r/independent_candidate_ranking.csv")
            if x["aggregation"] == "mean" and x["method"] == "BVSE"}
    spread = {x["edge_id"]: float(x["barrier_sd_eV"]) for x in rows(h2 / "candidate_model_disagreement.csv")}
    methods = methods_for(keys, pool, refs, bvse, spread, soap, "h2-random-01")
    table = dict(published)
    exponent = math.log(table[.20] / table[.30]) / (3 * math.log(.30 / .20))
    public_config = dict(old["design_assumptions"]["middle"], grid_power=exponent)
    comparison = []
    costs = {}
    for edge, item in sorted(eligible.items()):
        hours = compute_cost(int(item["atoms"]), int(item["nominal_kpoints"]), int(item["images"]),
                             public_config, "middle")["scenario_total_h"]
        costs[edge] = hours
        comparison.append(dict(
            edge_id=edge, nominal_kpoints=int(item["nominal_kpoints"]),
            H2_middle_estimate_h=float(item["middle_scenario_total_h"]), public_grid_proxy_estimate_h=hours,
        ))
    replay = []
    for budget in old["budgets_reference_scenario_h"]:
        picks = simulate(pool, methods, costs, budget, keys, refs, {"SOAP": soap, "SOAP+Activation": (soap+activation)/2})
        for method, paths in picks.items():
            prior = next(x for x in original if float(x["budget_h"]) == budget and x["scenario"] == "middle"
                         and x["method"] == method)
            spent = sum(costs[path] for path in paths)
            if spent > budget + 1e-9:
                raise ValueError("The cost simulation overspent")
            replay.append(dict(
                budget_h=budget, method=method, old_edges=prior["edge_ids"], public_proxy_edges="|".join(paths),
                same_set=set(paths) == {p for p in prior["edge_ids"].split("|") if p},
                public_proxy_spend_h=spent,
            ))
    args.output.mkdir(parents=True, exist_ok=True)
    save_csv(args.output / "eval06_walltime_predictions.csv", predictions)
    save_csv(args.output / "candidate_cost_overlay.csv", comparison)
    save_csv(args.output / "procurement_replay.csv", replay)
    save_csv(args.output / "public_neb_outcar_timings.csv", extract_neb(args.public_vasp_archive))
    print(json.dumps({
        "figure2_walltime_sha256": sha256(public_source.read_bytes()).hexdigest(),
        "public_spacing_ratio": factor, "prospective_grid_exponent_proxy": exponent,
        "changed_procurement_cases": sum(not row["same_set"] for row in replay),
    }, indent=2))


if __name__ == "__main__":
    main()
