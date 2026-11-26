"""Plot fixed-geometry QE numerical controls against the frozen acceptance limits."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from audit_project_data import read_json
from render_project_audit import BLUE, RUST, TEAL, label, save, style


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("grid", "cutoff", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    grid = read_json(args.grid)
    cutoff = read_json(args.cutoff)
    expected = [(grid, k) for k in ("peak:peak_k333", "middle_k222:middle_k333")] + [(cutoff, "peak:peak_cut125")]
    if not all(record["verdict"][key]["pass_gate"] for record, key in expected):
        raise ValueError("This figure requires the observed two grid and one cutoff gates")
    shifts = [abs(record["verdict"][k]["relative_energy_shift_meV"]) for record, k in expected]
    style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.25, 3.08))
    fig.subplots_adjust(left=.12, right=.97, bottom=.29, top=.84, wspace=.45)
    x = np.arange(3)
    left.scatter(x, shifts, c=[BLUE, BLUE, TEAL], s=25, zorder=3)
    for i, value in enumerate(shifts):
        left.annotate(f"{value:.3g}", (i, value), xytext=(0, 6), textcoords="offset points",
                      ha="center", fontsize=7.4)
    left.axhline(10., color=RUST, linestyle="--", lw=.9)
    left.text(1.14, 11.7, "10 meV limit", color=RUST, fontsize=7.3)
    left.set_yscale("log")
    left.set_ylim(.013, 30)
    left.set_xlim(-.48, 2.48)
    left.set_xticks(x, ["Grid:\npeak", "Grid:\nnon-peak", "Cutoff:\npeak"])
    left.set_ylabel("Absolute relative-energy change (meV)")
    left.grid(axis="y", which="major")
    label(left, "a", "Endpoint-referenced energy")
    roles = ["Endpoint", "Peak", "Non-peak"]
    comparisons = (("endpoint:endpoint_k333", "peak:peak_k333", "middle_k222:middle_k333"),
                   ("endpoint:endpoint_cut125", "peak:peak_cut125"))
    for offset, records, keys, color, title in (
        (-.08, grid, comparisons[0], BLUE, "Grid control"),
        (.08, cutoff, comparisons[1], TEAL, "Cutoff control"),
    ):
        vals = [records["comparisons"][k]["force_component_rmse_eV_A"] for k in keys]
        right.scatter(x[:len(vals)]+offset, vals, color=color, s=23, label=title, zorder=3)
    right.axhline(.020, color=RUST, linestyle="--", lw=.9)
    right.axhline(.010, color=RUST, linestyle=":", lw=.8)
    right.text(1.08, .021, "Acceptance limit", color=RUST, fontsize=7.2)
    right.text(1.08, .011, "Reporting line", color=RUST, fontsize=7.2)
    right.set_ylim(0, .0245)
    right.set_xlim(-.5, 2.65)
    right.set_xticks(x, roles)
    right.set_ylabel("Force component RMSE (eV/Å)")
    right.grid(axis="y")
    right.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.52, -.17))
    label(right, "b", "All-atom force agreement")
    save(fig, args.output, "qe_fixed_geometry_precision")


if __name__ == "__main__":
    main()
