"""Compare four-arm endpoint force errors and integrate paired approach forces."""
import argparse
from collections import defaultdict
import io
import json
from pathlib import Path

import numpy as np
from ase.geometry import find_mic
from ase.io import read
from scipy.stats import spearmanr

from audit_project_data import write_csv, write_json
from audit_signed_barriers import PANELS, SEEDS, file_sha, rows
from run_force_weighting import load_bundle
from score_phase1_forces import references
from score_step200_branches import CONTRASTS
from step200_protocol import BRANCHES


def score(args):
    _, files = load_bundle(args.bundle)
    index = rows(files["evaluation_frames.csv"])
    forces, offsets, paths, _, excluded = references(args, files)
    blind = read(io.StringIO(files["evaluation.xyz"].decode()), index=":", format="extxyz")
    gpu_manifest = args.inference / "manifest.json"
    manifest = json.loads(gpu_manifest.read_bytes())
    expected = dict(line.split()[::-1] for line in args.checkpoint_hashes.read_text().splitlines())
    if manifest["bundle_sha256"] != file_sha(args.bundle):
        raise ValueError("Different geometry bundle")
    predictions = {}
    for row in manifest["models"]:
        seed, arm = row["seed"], row["arm"]
        p = args.inference / row["file"]
        if (row["checkpoint_sha256"] != expected[f"full/seed{seed}/{arm}/resume.pt"]
                or file_sha(p) != row["file_sha256"] or row["frames"] != len(index)):
            raise ValueError("Wrong force prediction provenance")
        values = np.load(p, allow_pickle=False).astype(np.float64)
        if values.shape != forces.shape or not np.isfinite(values).all():
            raise ValueError("Invalid force array")
        predictions[seed, arm] = values
    if set(predictions) != {(s, a) for s in SEEDS for a in BRANCHES}:
        raise ValueError("Incomplete four-arm forces")
    dft_peak_drift = {(r["seed"], r["panel"], r["edge_id"], r["contrast"]): float(r["fixed_peak_signed_error"])
                      for r in rows(args.energy_contrasts.read_bytes()) if r["step"] == "400"}
    groups = defaultdict(list)
    for i, row in enumerate(index):
        groups[row["panel"], row["edge_id"], row["material_id"]].append((int(row["frame"]), i))
    force_rows, work_rows = [], []
    force_groups, work_groups = defaultdict(list), defaultdict(list)
    residuals = {k: np.square(p-forces) for k, p in predictions.items()}
    for key, members in sorted(groups.items()):
        panel, edge, material = key
        li, regions = paths[edge]
        trajectory = [i for _, i in sorted(members)]
        peak = trajectory.index(regions["high_interior"][0])
        slices = {i: slice(offsets[i], offsets[i+1]) for i in trajectory}
        segments = []
        for i, j in zip(trajectory[:peak], trajectory[1:peak+1]):
            if not (np.array_equal(blind[i].numbers, blind[j].numbers)
                    and np.allclose(blind[i].cell, blind[j].cell, atol=1e-8, rtol=0)):
                raise ValueError(f"Atom order or cell differs: {edge}")
            displacement, _ = find_mic(blind[j].positions - blind[i].positions, blind[i].cell, blind[i].pbc)
            segments.append((slices[i], slices[j], displacement))
        for left, right in CONTRASTS:
            name = f"{left}_minus_{right}"
            for seed in SEEDS:
                label = dict(panel=panel, material_id=material, edge_id=edge, contrast=name, seed=seed)
                a, b = residuals[seed, left], residuals[seed, right]
                for region, indices in regions.items():
                    for target in ("all_atoms", "migrating_li"):
                        if not indices or (target == "migrating_li" and li is None):
                            continue
                        indices_at_target = [slices[i] if target == "all_atoms" else offsets[i]+li for i in indices]
                        aa, bb = (1000*np.sqrt(np.mean([f[i] for i in indices_at_target])) for f in (a, b))
                        force_rows.append(dict(**label, target=target, region=region,
                                               left_rmse_meV_per_angstrom=float(aa), right_rmse_meV_per_angstrom=float(bb),
                                               change_meV_per_angstrom=float(aa-bb)))
                        force_groups[panel, name, target, region, material].append(float(aa-bb))
                left_force, right_force = predictions[seed, left], predictions[seed, right]
                total, mover = 0., 0.
                for i, j, displacement in segments:
                    difference = (left_force[i]-right_force[i]) + (left_force[j]-right_force[j])
                    contributions = -500*displacement*difference
                    total += float(contributions.sum())
                    if li is not None:
                        mover += float(contributions[li].sum())
                exact = dft_peak_drift[str(seed), panel, edge, name]
                work_rows.append(dict(**label, total_work_meV=total, Li_work_meV=mover if li is not None else None,
                                      other_work_meV=total-mover if li is not None else None,
                                      fixed_peak_drift_meV=exact))
                work_groups[panel, name, material].append((total, exact))
    rng = np.random.default_rng(20261127)
    rmse, work = [], []
    rmse_data, work_data = defaultdict(list), defaultdict(list)
    for (panel, name, target, region, _), value in force_groups.items():
        rmse_data[panel, name, target, region].append(value)
    for (panel, name, _), value in work_groups.items():
        work_data[panel, name].append(value)
    for (panel, name, target, region), items in rmse_data.items():
        vals = np.asarray(items)
        means = vals.mean(axis=1)
        samples = rng.integers(len(vals), size=(10_000,len(vals)))
        low, high = np.quantile(means[samples].mean(axis=1), [.025, .975])
        rmse.append(dict(panel=panel, contrast=name, target=target, region=region, materials=len(vals),
                         mean_meV_per_angstrom=float(means.mean()), lo95=float(low), hi95=float(high),
                         seed17=float(vals[:,0].mean()), seed29=float(vals[:,1].mean())))
    for (panel, name), items in work_data.items():
        estimate, actual = np.asarray(items).mean(axis=1).T
        work.append(dict(panel=panel, contrast=name, materials=len(items), work_meV=float(estimate.mean()),
                         exact_meV=float(actual.mean()), approximation_mae_meV=float(np.abs(estimate-actual).mean()),
                         spearman=float(spearmanr(estimate, actual).statistic)))
    args.output.mkdir()
    write_csv(args.output / "material_force_rmse.csv", force_rows)
    write_csv(args.output / "force_rmse_summary.csv", rmse)
    write_csv(args.output / "material_force_work.csv", work_rows)
    write_csv(args.output / "force_work_summary.csv", work)
    write_json(args.output / "scoring_manifest.json", dict(
        gpu_manifest_sha256=file_sha(gpu_manifest), energy_contrasts_sha256=file_sha(args.energy_contrasts),
        source_sha256=file_sha(Path(__file__)), excluded_for_migrating_Li=excluded,
        panels=PANELS, bootstrap="10,000 paired material resamples conditional on fixed seeds 17 and 29"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "project", "inference", "checkpoint_hashes", "energy_contrasts", "output"):
        parser.add_argument(name, type=Path)
    score(parser.parse_args())
