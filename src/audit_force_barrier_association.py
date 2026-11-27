"""Compare signed-model barrier degradation with paired force changes."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from audit_project_data import read_csv, write_csv


def association(path, barriers, output):
    barrier = {}
    for row in read_csv(barriers):
        key = (row["panel"], int(row["step"]), int(row["seed"]), row["edge_id"])
        barrier[key] = abs(float(row["augmented_signed_error_meV"])) - abs(float(row["replay_signed_error_meV"]))
    groups = defaultdict(lambda: defaultdict(list))
    for row in read_csv(path):
        step, seed = int(row["step"]), int(row["seed"])
        panel, path_id = row["panel"], row["edge_id"]
        force = float(row["augmented_minus_replay_eV_A"])
        change = barrier[panel, step, seed, path_id]
        groups[step, panel, row["region"], row["target"]][path_id].append((seed, force, change))
    out = []
    for (step, panel, region, target), materials in sorted(groups.items()):
        usable = np.asarray([sorted(arm) for arm in materials.values()])
        if usable.shape != (len(materials), 2, 3):
            raise ValueError("Incomplete paired perturbation data")
        x, y = usable[:, :, 1:].mean(axis=1).T
        rho = float(spearmanr(x, y).statistic)
        leave_one = [float(spearmanr(np.delete(x, i), np.delete(y, i)).statistic) for i in range(len(x))]
        by_seed = [spearmanr(usable[:, j, 1], usable[:, j, 2]).statistic for j in range(2)]
        out.append(dict(step=step, panel=panel, region=region, target=target, materials=len(materials),
                        spearman_material_mean=rho, seed17=float(by_seed[0]), seed29=float(by_seed[1]),
                        leave_one_min=min(leave_one), leave_one_max=max(leave_one)))
    write_csv(output, out)


def region_contrast(path, output):
    cells = defaultdict(dict)
    for row in read_csv(path):
        key = (int(row["step"]), row["panel"], row["target"], row["edge_id"], int(row["seed"]))
        cells[key][row["region"]] = float(row["augmented_minus_replay_eV_A"])
    diffs = defaultdict(lambda: defaultdict(dict))
    for (step, panel, target, edge, seed), regions in cells.items():
        if "high_interior" in regions and "other_interior" in regions:
            diffs[step, panel, target][edge][seed] = regions["other_interior"]-regions["high_interior"]
    out = []
    rng = np.random.default_rng(20261127)
    for (step, panel, target), materials in sorted(diffs.items()):
        values = np.asarray([[s[17], s[29]] for s in materials.values()])
        avg = values.mean(axis=1)
        indices = rng.integers(len(avg), size=(10000, len(avg)))
        lo, hi = np.quantile(avg[indices].mean(axis=1), [.025, .975])
        out.append(dict(step=step, panel=panel, target=target, materials=len(avg),
                        other_minus_high_eV_A=float(avg.mean()), lo95=float(lo), hi95=float(hi),
                        seed17=float(values[:, 0].mean()), seed29=float(values[:, 1].mean())))
    write_csv(output, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("paired_forces", "barrier_delta", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--region-contrast", type=Path)
    args = parser.parse_args()
    association(args.paired_forces, args.barrier_delta, args.output)
    if args.region_contrast:
        region_contrast(args.paired_forces, args.region_contrast)
