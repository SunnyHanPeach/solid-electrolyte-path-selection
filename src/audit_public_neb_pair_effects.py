"""Decompose the archived FP-DeErr pair gains without extra inference."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

from audit_phase_h2_evaluation import digest, rows
from audit_public_neb_reuse import FP_HASH, checked
from phase_h1_shortlist import write


def profiles(predictions, paths, seeds, arms):
    curves = defaultdict(dict)
    for row in rows(predictions):
        key = (row["path_id"], int(row["seed"]), row["arm"])
        image = int(row["image"])
        value = float(row["energy_eV"])
        if image in curves[key] or not np.isfinite(value):
            raise ValueError(f"Invalid image prediction: {key}/{image}")
        curves[key][image] = value
    if set(curves) != {(key, seed, arm) for key in paths for seed in seeds for arm in arms}:
        raise ValueError("Incomplete matched predictions")
    return {key: relative(curve) for key, curve in curves.items()}


def relative(curve):
    if set(curve) != set(range(7)):
        raise ValueError("Unexpected image numbering")
    absolute = np.array([curve[i] for i in range(7)], dtype=float)
    return (absolute - absolute[0]) * 1000


def score_pair(reference, path, control):
    dft_peak, path_peak, control_peak = map(np.argmax, (reference, path, control))
    target = reference[dft_peak]
    gain = float(abs(control[control_peak] - target) - abs(path[path_peak] - target))
    fixed = float(abs(control[dft_peak] - target) - abs(path[dft_peak] - target))
    image_gains = abs(control - reference) - abs(path - reference)
    other_internal = [image_gains[i] for i in range(1, 6) if i != dft_peak]
    return dict(
        dft_peak_image=int(dft_peak), path_peak_image=int(path_peak), control_peak_image=int(control_peak),
        barrier_reference_meV=float(target), barrier_gain_meV=gain, fixed_DFT_peak_gain_meV=fixed,
        peak_selection_residual_meV=gain-fixed,
        final_relative_energy_gain_meV=float(image_gains[-1]),
        other_internal_relative_energy_gain_meV=float(np.mean(other_internal)),
    )


METRICS = ("barrier_gain_meV", "fixed_DFT_peak_gain_meV", "peak_selection_residual_meV",
           "final_relative_energy_gain_meV", "other_internal_relative_energy_gain_meV")


def audit_families(paired, seeds):
    by_family = defaultdict(list)
    for row in paired:
        by_family[row["family"]].append(row)
    families = []
    for key, subset in sorted(by_family.items()):
        seed_values = {f"seed{seed}_gain_meV": float(np.mean(
            [row["barrier_gain_meV"] for row in subset if row["seed"] == seed])) for seed in seeds}
        families.append(dict(
            family=key, paths=len({row["path_id"] for row in subset}), **seed_values,
            positive_seeds=sum(value > 0 for value in seed_values.values()),
            negative_seeds=sum(value < 0 for value in seed_values.values()),
            **{name: float(np.mean([row[name] for row in subset])) for name in METRICS},
        ))
    return families


def summarize(paired, families, seeds):
    ranked = sorted(families, key=lambda row: row["barrier_gain_meV"], reverse=True)
    values = np.array([row["barrier_gain_meV"] for row in ranked])
    total = float(values.sum())
    return dict(
        family_macro={name: float(np.mean([row[name] for row in families])) for name in METRICS},
        all_seed_positive_families=sum(row["positive_seeds"] == len(seeds) for row in families),
        all_seed_negative_families=sum(row["negative_seeds"] == len(seeds) for row in families),
        top_3_net_contribution_percent=float(values[:3].sum() / total * 100) if total > 0 else None,
        top_3_share_of_positive_contributions_percent=float(
            values[:3].sum() / values[values > 0].sum() * 100) if (values[:3] > 0).all() else None,
        gain_excluding_top_3_families_meV=float(values[3:].mean()),
        top_3_families=[row["family"] for row in ranked[:3]],
        path_seed_cells=len(paired),
        peak_changes_between_arms=sum(row["path_peak_image"] != row["control_peak_image"] for row in paired),
        dft_peaks_internal_path_count=len({row["path_id"] for row in paired if 0 < row["dft_peak_image"] < 6}),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "paired", "reference", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.paired / "summary.json").read_bytes())
    if digest(args.panel) != manifest["panel_sha256"] or manifest["reference_sha256"] != FP_HASH:
        raise ValueError("Original panel or reference provenance differs")
    predictions = args.paired / "image_predictions.csv"
    source = json.loads(gzip.decompress(checked(args.reference, FP_HASH)))["pathways"]
    excluded = set(manifest["excluded_for_training_framework_overlap"])
    panel = {row["path_id"]: row for row in rows(args.panel)
             if row["use_for_external_scoring"] == "True" and row["path_id"] not in excluded}
    seeds = manifest["seeds"]
    controls = {"net": "replay1", "absolute": "pure"}
    curves = profiles(predictions, panel, seeds, ["path1", *controls.values()])
    refs = {key: relative({image["image_index"]: image["energy_total_eV"]
                           for image in source[key]["dft_neb_reference"]["images"]}) for key in panel}
    scores, all_families, summaries = [], [], {}
    for label, arm in controls.items():
        paired = [dict(path_id=key, family=row["screened_framework_family"], chemsys=row["chemsys"],
                       seed=seed, contrast=label,
                       **score_pair(refs[key], curves[key, seed, "path1"], curves[key, seed, arm]))
                  for key, row in panel.items() for seed in seeds]
        families = audit_families(paired, seeds)
        stats = summarize(paired, families, seeds)
        if len(families) != manifest["n_families"] or len(panel) != manifest["n_paths"]:
            raise ValueError("Path or family count differs from original scoring")
        averages = stats["family_macro"]
        np.testing.assert_allclose(averages["barrier_gain_meV"], manifest["gains"][label]["mean_meV"],
                                   atol=1e-9, rtol=0)
        np.testing.assert_allclose(averages["fixed_DFT_peak_gain_meV"] +
                                   averages["peak_selection_residual_meV"], averages["barrier_gain_meV"],
                                   atol=1e-9, rtol=0)
        scores.extend(paired)
        all_families.extend(dict(contrast=label, **row) for row in families)
        summaries[label] = stats
    args.output.mkdir(exist_ok=False, parents=True)
    write(args.output / "family_gains.csv", all_families)
    write(args.output / "image_region_effects.csv", scores)
    summary = dict(reference_sha256=FP_HASH, panel_sha256=digest(args.panel),
                   predictions_sha256=digest(predictions), script_sha256=digest(Path(__file__)),
                   trained_seeds=seeds, descriptive_only=True, results=summaries)
    data = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    (args.output / "summary.json").write_text(data, encoding="utf-8", newline="\n")
    print(data)


if __name__ == "__main__":
    main()
