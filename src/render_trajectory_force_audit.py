"""Render paired force-error contrasts at the protected trajectory checkpoints."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from audit_project_data import read_csv
from render_project_audit import BLUE, GRAY, RUST, label, save, style


def plot(table, output):
    records = read_csv(table)
    style()
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.15))
    fig.subplots_adjust(left=.11, right=.975, top=.90, bottom=.15, hspace=.69, wspace=.42)
    for rowidx, (axis_row, target) in enumerate(zip(axes, ("moving_Li", "all_atoms"))):
        for col, (ax, panel, title) in enumerate(zip(axis_row, ("primary91", "historical_support"),
                                                    ("Main panel", "Auxiliary panel"))):
            for position, (region, display, color, marker) in enumerate((
                    ("high_interior", "High-energy interior", RUST, "o"),
                    ("other_interior", "Other interior", BLUE, "^"),
                    ("endpoints", "Endpoints", GRAY, "s"))):
                rows = sorted((r for r in records if (r["panel"], r["target"], r["region"]) ==
                               (panel, target, region)), key=lambda r: int(r["step"]))
                if [int(r["step"]) for r in rows] != [8, 200, 400]:
                    raise ValueError("Trajectory step or region differs")
                arr = 1000 * np.array([[float(r[k]) for k in ("mean", "lo95", "hi95")] for r in rows])
                center, lo, hi = arr.T
                x = np.arange(3) + (position - 1)*.09
                ax.errorbar(x, center, yerr=np.vstack([center-lo, hi-center]), fmt=marker + "-",
                            lw=.85, markersize=4., capsize=2., label=display, color=color)
            ax.axhline(0., color="#727b85", linewidth=.8, linestyle="--", zorder=0)
            ax.grid(axis="y")
            ax.set_xticks(range(3), ("8", "200", "400"))
            ax.set_xlabel("Training update")
            ax.set_ylabel("Δ force RMSE (meV/Å)")
            label(ax, "abcd"[2*rowidx + col], f"{title} · {'migrating Li' if rowidx == 0 else 'all atoms'}")
    h, titles = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, titles, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(.55, .035))
    save(fig, output, "protected_trajectory_force_rmse")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("table", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    plot(args.table, args.output)
