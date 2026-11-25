"""Compare archived path gains under the same seeds and decompose the replay effect."""
import argparse
import csv
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from audit_signed_barriers import file_sha
from run_force_weighting import atomic_json
from validate_phase1_materials import bootstrap_weights, estimates

ANCHOR = "mp-753143"
INFLUENTIAL = "mp-1277814"
COMMON_SEEDS = (17, 29, 43)
ALL_SEEDS = (*COMMON_SEEDS, 53, 67)
METRICS = ("net", "absolute")
TRANSFERS = ("bi_li_s", "ag_f_li", "co_f_li_o", "cr_ga_li_o")


def read_gains(path, default_candidate=None):
    gains = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            candidate = row.get("candidate", row.get("path", default_candidate)).split("_", 1)[0]
            material = row.get("material_id", row.get("material"))
            key = (candidate, material, row["chemsys"], int(row["seed"]), row["metric"])
            value = float(row["gain_meV"])
            if key in gains or not np.isfinite(value):
                raise ValueError(f"Duplicate or invalid gain: {key}")
            gains[key] = value
    return gains


def build_panel(records, paths, seeds):
    materials = sorted({(mat, system) for _, mat, system, _, _ in records})
    if len({mat for mat, _ in materials}) != len(materials):
        raise ValueError("Expected a single chemistry per evaluation material")
    expected = {(path, mat, sys, seed, metric) for path in paths for mat, sys in materials
                for seed in seeds for metric in METRICS}
    if records.keys() != expected:
        raise ValueError("Path cohorts or seed coverage differ")
    data = {path: {metric: np.array([[records[path, mat, sys, seed, metric] for mat, sys in materials]
                                    for seed in seeds]) for metric in METRICS} for path in paths}
    return materials, data


def weights(materials):
    return bootstrap_weights([{"chemsys": system} for _, system in materials])[1]


def ordering(original, replication, metric):
    means = [{path: float(values[metric].mean()) for path, values in panel.items()}
             for panel in (original, replication)]
    ranked = [sorted(m, key=lambda p: (-m[p], p)) for m in means]
    differences = [(a,b) for a,b in combinations(ranked[0], 2)
                   if means[0][a] != means[0][b] and means[1][a] != means[1][b]]
    agreed = sum(means[1][a] > means[1][b] for a,b in differences)
    flips = [p for p in ranked[0] if means[0][p] * means[1][p] < 0]
    return dict(original=means[0], replication=means[1], ranking=ranked,
                concordant_path_pairs=agreed, comparable_pairs=len(differences), sign_flips=flips)


def describe_anchor(data, materials, mask=None):
    if mask is not None:
        data = {metric: a[:,mask] for metric,a in data.items()}
        materials = [m for m, use in zip(materials, mask) if use]
    w = weights(materials)
    # Gain versus replay = gain versus pure D0 + extra error of replay versus pure D0.
    replay_penalty = data["net"] - data["absolute"]
    penalty = estimates(replay_penalty, w)
    return dict(net=estimates(data["net"], w), absolute=estimates(data["absolute"], w),
                replay_minus_pure_error={k: penalty[k] for k in ("mean_meV", "interval95_meV")})


def compare_anchor(panel, materials):
    w = weights(materials)
    comparisons = {}
    for path, gains in panel.items():
        if path == ANCHOR:
            continue
        delta = panel[ANCHOR]["absolute"] - gains["absolute"]
        stat = estimates(delta, w)
        comparisons[path] = {k:stat[k] for k in ("mean_meV", "interval95_meV", "seed_means_meV")}
    return comparisons


def repeated_inference(root, materials, one, many):
    replication = json.loads((root/"material_replication"/"summary.json").read_bytes())["proof"]
    comparison = json.loads((root/"candidate_comparison"/"summary.json").read_bytes())["predictions"]
    if any(replication[k] != comparison[k] for k in ("bundle", "panel", "torch", "hardware")):
        raise ValueError("Inference inputs differ")
    case = next(k.split("/")[1] for k in comparison["weights"] if k.startswith(f"17/{ANCHOR}_"))
    for seed in COMMON_SEEDS:
        for original, expanded in (("pure", "pure"), ("path3", f"{case}/path"), ("replay3", f"{case}/replay")):
            if replication["weights"][f"{seed}/{original}"] != comparison["weights"][f"{seed}/{expanded}"]:
                raise ValueError("Repeated inference used different model weights")
    w = weights(materials)
    outcome = {}
    for metric in METRICS:
        a, b = one[metric][:len(COMMON_SEEDS)], many[metric]
        diff = np.abs(a-b)
        outcome[metric] = dict(max_difference_meV=float(diff.max()),
                               mean_absolute_difference_meV=float(diff.mean()),
                               entries_differing=int(np.count_nonzero(diff)), total_entries=diff.size,
                               original=estimates(a, w), repeated=estimates(b, w))
    return outcome


def run(root, output):
    pilot = root/"paired_materials.csv"
    new = root/"candidate_comparison"/"paired_materials.csv"
    confirm = root/"confirmation"/"paired_materials.csv"
    replication = root/"material_replication"/"paired_materials.csv"
    sources = [pilot, new, confirm, replication, root/"material_replication"/"summary.json",
               root/"candidate_comparison"/"summary.json"]
    original_records = read_gains(pilot)
    for case in TRANSFERS:
        directory = root/"transfer"/case
        summary = directory/"summary.json"
        table = directory/"paired_materials.csv"
        sources.extend([summary, table])
        edge = json.loads(summary.read_bytes())["path"]["edge_id"]
        values = read_gains(table, edge)
        if original_records.keys() & values.keys():
            raise ValueError("Repeated original records")
        original_records.update(values)
    paths = sorted({key[0] for key in original_records})
    original_mat, original = build_panel(original_records, paths, COMMON_SEEDS)
    new_records = read_gains(new)
    new_mat, new_panel = build_panel(new_records, paths, COMMON_SEEDS)
    if {m for m,_ in original_mat} & {m for m,_ in new_mat}:
        raise ValueError("Material overlap across panels")
    old_anchor = {key:value for key,value in original_records.items() if key[0] == ANCHOR}
    old_anchor.update(read_gains(confirm, ANCHOR))
    old_five_mat, old_five = build_panel(old_anchor, [ANCHOR], ALL_SEEDS)
    new_anchor = read_gains(replication, ANCHOR)
    new_five_mat, new_five = build_panel(new_anchor, [ANCHOR], ALL_SEEDS)
    if original_mat != old_five_mat or new_mat != new_five_mat:
        raise ValueError("Shared-seed and five-seed cohorts differ")
    omit = np.asarray([m != INFLUENTIAL for m,_ in new_mat])
    if int((~omit).sum()) != 1:
        raise ValueError("Influence material must occur once")
    i = list(omit).index(False)
    influence = {path: {metric: float(a[:,i].mean()) for metric,a in values.items()}
                 for path,values in new_panel.items()}
    atomic_json(output, dict(
        scope="Descriptive reuse of archived results. Comparisons are conditional on the trained seeds.",
        inputs_sha256={p.relative_to(root).as_posix(): file_sha(p) for p in sources},
        paths=len(paths), panels={"original": len(original_mat), "replication": len(new_mat)},
        ranking={metric: ordering(original, new_panel, metric) for metric in METRICS},
        anchor_five_seeds=dict(original=describe_anchor(old_five[ANCHOR], original_mat),
                               replication=describe_anchor(new_five[ANCHOR], new_mat),
                               replication_without_influential=describe_anchor(new_five[ANCHOR], new_mat, omit)),
        anchor_vs_existing_paths_absolute_gain=dict(original=compare_anchor(original, original_mat),
                                                     replication=compare_anchor(new_panel, new_mat)),
        anchor_repeat_inference=repeated_inference(root, new_mat, new_five[ANCHOR], new_panel[ANCHOR]),
        influential_material=dict(material=INFLUENTIAL, gains=influence),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase1_directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.phase1_directory, args.output)
