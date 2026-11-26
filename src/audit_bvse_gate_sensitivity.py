"""Measure how the cheap BVSE geometry gate responds to threshold choices."""
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from audit_project_data import read_csv, write_csv, write_json
from render_project_audit import label, save, style


CONTACTS = (1.2, 1.3, 1.4, 1.5, 1.6, 1.7)
PROMINENCE = (0., .01, .02, .05, .10)


def run(table, frozen, output):
    candidates = read_csv(table)
    h1 = {row["edge_id"] for row in read_csv(frozen)}
    ids = {row["edge_id"] for row in candidates}
    if not h1.issubset(ids):
        raise ValueError("The frozen candidate set is not fully represented")
    grouped = defaultdict(list)
    for row in candidates:
        grouped[row["initial_stratum"]].append(row)
    fractions = {}
    results = []
    for key, rows in sorted(grouped.items()):
        material_indices = {name: offset for offset, name in enumerate(sorted({r["material_id"] for r in rows}))}
        groups = np.array([material_indices[r["material_id"]] for r in rows])
        population = np.bincount(groups)
        good = np.array([r["verified"] == "True" and r["solver_status"] == "converged" for r in rows])
        contacts = np.array([float(r["verified_min_contact_A"] or "-inf") for r in rows])
        peaks = np.array([float(r["verified_peak_eV"] or "-inf") for r in rows])
        is_h1 = np.array([r["edge_id"] in h1 for r in rows])
        matrix = np.zeros((len(PROMINENCE), len(CONTACTS)))
        for i, peak in enumerate(PROMINENCE):
            for j, contact in enumerate(CONTACTS):
                passed = good & (contacts >= contact) & (peaks >= peak)
                rate = float(np.mean(np.bincount(groups[passed], minlength=len(population)) / population))
                matrix[i, j] = rate
                results.append(dict(initial_stratum=key, final_min_contact_A=contact, peak_margin_eV=peak,
                                    screened_paths=len(rows), passing_paths=int(passed.sum()),
                                    material_weighted_pass=rate, frozen_candidates=int(is_h1.sum()),
                                    frozen_candidates_passing=int((passed & is_h1).sum())))
        fractions[key] = matrix
    original = [r for r in results if r["final_min_contact_A"] == 1.4 and r["peak_margin_eV"] == .02]
    precomputed = sum(row["screen_thresholds_passed"] == "True" for row in candidates)
    if sum(r["passing_paths"] for r in original) != precomputed:
        raise ValueError("Original thresholds do not reproduce the frozen screen")
    output.mkdir(exist_ok=False, parents=True)
    write_csv(output / "gate_sensitivity.csv", results)
    write_json(output / "summary.json", dict(paths=len(candidates), frozen_candidates=len(h1), original_gate=original,
                                              convergence_limit_eV_A=0.1, thresholds_changed_in_frozen_screen=False))
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.2))
    fig.subplots_adjust(left=.10, bottom=.18, right=.84, top=.86, wspace=.42)
    for ax, letter, key, title in zip(axes, "ab", ("short", "ordinary"), ("Short contacts", "Other paths")):
        m = fractions[key] * 100
        image = ax.imshow(m, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=100)
        ax.set_xticks(range(len(CONTACTS)), [str(v) for v in CONTACTS])
        ax.set_yticks(range(len(PROMINENCE)), [f"{v:.2f}" for v in PROMINENCE])
        ax.set_xlabel("Minimum final Li contact (Å)")
        ax.set_ylabel("Peak prominence (eV)")
        label(ax, letter, title)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i,j]:.0f}", ha="center", va="center", fontsize=7,
                        color="#111923" if m[i,j] > 53 else "white")
        ax.add_patch(plt.Rectangle((1.5, 1.5), 1, 1, fill=False, edgecolor="white", linewidth=1.3))
    scale = fig.colorbar(image, ax=axes.tolist(), fraction=.048, pad=.045)
    scale.set_label("Material-weighted pass (%)")
    save(fig, output, "bvse_gate_sensitivity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("verified_geometry", "candidate_readiness", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.verified_geometry, args.candidate_readiness, args.output)


if __name__ == "__main__":
    main()
