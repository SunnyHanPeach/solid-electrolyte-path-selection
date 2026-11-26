"""Produce journal-style figures from the frozen CPU audit results."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from audit_project_data import read_csv, read_json


BLACK, GRAY, LIGHT = "#20252a", "#737e86", "#e2e7ea"
BLUE, TEAL, RUST = "#244e6e", "#087b73", "#aa512b"
COLORS = {"positive": TEAL, "negative": RUST, "unresolved": GRAY, "equivalent": BLUE}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.3, "text.color": BLACK,
        "axes.labelcolor": BLACK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .65, "axes.edgecolor": "#a7b0b6", "axes.labelsize": 8.3,
        "xtick.color": BLACK, "ytick.color": BLACK, "xtick.major.width": .6,
        "ytick.major.width": .6, "grid.color": LIGHT, "grid.linewidth": .55,
        "legend.fontsize": 7.6, "savefig.facecolor": "white", "pdf.fonttype": 42,
    })


def save(fig, destination, stem):
    for suffix in ("png", "pdf"):
        with (destination / f"{stem}.{suffix}").open("xb") as target:
            fig.savefig(target, format=suffix, dpi=300)
    plt.close(fig)


def label(ax, letter, title):
    ax.set_title(f"({letter})  {title}", loc="left", fontsize=9.2, pad=7, fontweight="semibold")


def utility(path, output):
    records = sorted(read_csv(path), key=lambda row: float(row["Delta_net"]))
    if len(records) != 14 or any(row["audit_status"] != "verified" for row in records):
        raise ValueError("Forest plot requires the verified B2 regular cohort")
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 4.3), sharey=True)
    fig.subplots_adjust(left=.156, right=.98, bottom=.21, top=.88, wspace=.16)
    keys = ("Delta_abs", "Delta_net")
    low = min(float(row[f"{k}_low95"]) for row in records for k in keys)
    high = max(float(row[f"{k}_high95"]) for row in records for k in keys)
    pad = (high-low)*.05
    for ax, key, letter, title in zip(axes, keys, "ab", ("Gain versus pure D₀", "Gain versus matched replay")):
        label(ax, letter, title)
        ax.axvspan(-.25, .25, color="#d8e4ec", zorder=0)
        ax.axvline(0, color="#8a959d", lw=.7)
        ax.grid(axis="x", zorder=0)
        for i, row in enumerate(records):
            mid, lo, hi = (float(row[key+s]) for s in ("", "_low95", "_high95"))
            ax.errorbar(mid, i, xerr=[[mid-lo], [hi-mid]], fmt="o", color=COLORS[row[key+"_class"]],
                        markersize=3.8, elinewidth=.9, capsize=2.0, zorder=3)
        ax.set_xlim(low-pad, high+pad)
        ax.set_ylim(-.65, len(records)-.35)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.set_yticks(range(len(records)), [r["edge_id"].split("_", 1)[0] for r in records])
        ax.tick_params(axis="y", length=0)
    fig.text(.57, .118, "Energy-barrier MAE gain (meV)", ha="center")
    fig.legend([Line2D([0], [0], marker="o", ls="", color=COLORS[k], markersize=4.2)
                for k in ("positive", "negative", "unresolved")],
               ["Positive", "Negative", "Unresolved"], frameon=False, ncol=3,
               loc="lower center", bbox_to_anchor=(.57, .024))
    save(fig, output, "fig01_path_utility")


def qe(grid_data, ledger, output):
    endpoint = grid_data["comparisons"]["endpoint:endpoint_k333"]
    peak = grid_data["comparisons"]["peak:peak_k333"]
    shift = abs(grid_data["verdict"]["peak:peak_k333"]["relative_energy_shift_meV"])
    fractions = [shift/10, endpoint["force_component_rmse_eV_A"]/.020, peak["force_component_rmse_eV_A"]/.020]
    number = [f"{shift:.3f} meV",
              f"{endpoint['force_component_rmse_eV_A']:.5f} eV/Å",
              f"{peak['force_component_rmse_eV_A']:.5f} eV/Å"]
    measures = ["Relative gap change", "Endpoint force RMSE", "Peak force RMSE"]
    fig, (ax, timing) = plt.subplots(1, 2, figsize=(7.25, 3.02), gridspec_kw={"width_ratios": [1.13, 1]})
    fig.subplots_adjust(left=.20, right=.976, bottom=.24, top=.85, wspace=.40)
    label(ax, "a", "Change from 2×2×2 to 3×3×3")
    y = np.arange(3)
    ax.scatter(fractions, y, c=TEAL, edgecolors="white", linewidths=.6, s=27, zorder=3)
    for i, (x, text) in enumerate(zip(fractions, number)):
        ax.annotate(text, (x, i), xytext=(6, 6), textcoords="offset points", fontsize=7.1)
    ax.axvline(1, ls="--", color=GRAY, lw=.9)
    ax.set_xscale("log")
    ax.set_xlim(.0013, 1.5)
    ax.set_ylim(-.45, 2.50)
    ax.invert_yaxis()
    ax.set_yticks(y, measures)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Observed change / acceptance threshold")
    ax.grid(axis="x")
    label(timing, "b", "Observed solver time")
    roles = ["endpoint", "peak"]
    recovered = []
    for role in roles:
        [source] = [r for r in ledger if r["material"] == "mp-1191903" and r["phase"] == "mesh"
                    and r["role"] == role and r["hardware"] == "RTX 3090"]
        recovered.append(float(source["solver_wall_s"])/60)
    vals = (recovered, [float(r["test"]["elapsed_seconds"])/60 for r in (endpoint, peak)])
    x = np.arange(2)
    for i, (name, color) in enumerate((("2×2×2", BLUE), ("3×3×3", TEAL))):
        bars = timing.bar(x+(i-.5)*.35, vals[i], .31, color=color, label=name, zorder=2)
        timing.bar_label(bars, fmt="%.0f", fontsize=7.3, padding=2)
    timing.set_ylim(0, 115)
    timing.grid(axis="y")
    timing.set_axisbelow(True)
    timing.set_ylabel("Wall time (min)")
    timing.set_xticks(x, ["Endpoint", "Peak"])
    timing.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.56, -.19))
    save(fig, output, "fig02_qe_grid_precision_cost")


def evidence(evaluation, output):
    method = evaluation["method_audit"]
    names = ["PBE, path linkage pending", "PW91 or PBEsol", "Hubbard U unresolved", "Functional unresolved", "Migrating ion unresolved"]
    keys = ["pbe_path_unverified", "incompatible_xc", "pbe_correction_unresolved", "mixed_xc_unresolved",
            "migrating_species_unresolved"]
    counts = [method[k] for k in keys]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.25, 3.02), gridspec_kw={"width_ratios": [1, .8]})
    fig.subplots_adjust(left=.245, right=.963, bottom=.20, top=.85, wspace=.70)
    label(left, "a", "Protocol metadata")
    bars = left.barh(names, counts, height=.57, color=[BLUE, RUST, GRAY, GRAY, GRAY])
    left.bar_label(bars, fontsize=8, padding=3)
    left.invert_yaxis()
    left.set_xlim(0, 21)
    left.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    left.set_xlabel("Public records")
    left.tick_params(axis="y", length=0, labelsize=7.5)
    left.grid(axis="x")
    left.set_axisbelow(True)
    label(right, "b", "Check outcomes")
    available = [evaluation["geometry_mapping_verified"],
                 evaluation["overlap"]["historical_parent"]["affected_public_records"],
                 evaluation["primary_released"]]
    bars = right.barh(["Mapped endpoints", "Framework flag", "Benchmark eligible"], available,
                      color=[BLUE, RUST, TEAL], height=.43)
    right.bar_label(bars, padding=3, fontsize=8)
    right.set_xlim(0, 47)
    right.invert_yaxis()
    right.set_xlabel("Public records")
    right.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    right.tick_params(axis="y", length=0, labelsize=7.6)
    right.grid(axis="x")
    right.set_axisbelow(True)
    save(fig, output, "fig03_evaluation_evidence")


def geometric_pool(paths, output):
    data = read_csv(paths)
    valid = [row for row in data if row["status"] == "geometry_verified"]
    if len(data) != 16155:
        raise ValueError("Geometry figure requires the complete frozen pool")
    a = np.array([(float(r["hop_length_A"]), float(r["minimum_interpolation_contact_A"]), int(r["atoms"]))
                  for r in valid])
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.25, 3.04), gridspec_kw={"width_ratios": [1.38, 1]})
    fig.subplots_adjust(left=.108, right=.963, top=.85, bottom=.19, wspace=.48)
    label(left, "a", "Straight-interpolation contact")
    hb = left.hexbin(a[:, 0], a[:, 1], gridsize=46, mincnt=1, bins="log", cmap="cividis",
                    linewidths=.03)
    left.axhline(1.4, color=RUST, linestyle="--", lw=1.0, label="1.4 Å screening threshold")
    left.set_xlabel("Encoded Li hop length (Å)")
    left.set_ylabel("Minimum Li contact (Å)")
    left.legend(loc="upper right", frameon=False, labelcolor=RUST, fontsize=7.0)
    scale = fig.colorbar(hb, ax=left, fraction=.048, pad=.014)
    scale.set_label("Paths per bin", size=7.3)
    scale.ax.tick_params(labelsize=6.6)
    label(right, "b", "Vacancy supercell size")
    step = 10
    bins = np.arange(0, a[:, 2].max()+step*2, step)
    right.hist(a[:, 2], bins=bins, color=BLUE, edgecolor="white", linewidth=.3)
    right.axvline(180, color=RUST, linestyle="--", lw=1.0)
    right.set_xlabel("Atoms per image")
    right.set_ylabel("Paths")
    right.grid(axis="y")
    right.set_axisbelow(True)
    save(fig, output, "fig04_candidate_geometry")


def family_overlap(path, output):
    data = read_csv(path)
    if len(data) != 2647:
        raise ValueError("Family comparison requires the complete candidate material pool")
    categories = ("match", "none_detected", "no_stoichiometry_comparison")
    matrix = [[sum(row[key] == status for row in data) for status in categories]
              for key in ("parent_match_status", "framework_match_status")]
    parent = np.array([row["historical_parent_flag"] == "True" for row in data])
    framework = np.array([row["historical_framework_flag"] == "True" for row in data])
    combos = [int(x.sum()) for x in (parent & framework, ~parent & framework, parent & ~framework,
                                    ~parent & ~framework)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.25, 3.08), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=.143, right=.982, top=.84, bottom=.285, wspace=.70)
    label(left, "a", "Historical structure matches")
    seen = np.zeros(2)
    for i, (title, hue) in enumerate((("Match", RUST), ("Compared, no match", BLUE), ("No comparable stoichiometry", "#d5dde2"))):
        width = np.asarray(matrix)[:, i]
        bars = left.barh([0, 1], width, left=seen, color=hue, height=.53, label=title)
        left.bar_label(bars, label_type="center", fmt="%.0f", fontsize=7.5, color=BLACK if i==2 else "white")
        seen += width
    left.set_yticks([0, 1], ["Full parent", "Non-Li framework"])
    left.invert_yaxis()
    left.set_xlim(0, len(data))
    left.set_xlabel("Candidate materials")
    left.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    left.tick_params(axis="y", length=0, labelsize=7.6)
    handles, titles = left.get_legend_handles_labels()
    fig.legend(handles, titles, frameon=False, loc="lower center", bbox_to_anchor=(.55, .025),
               ncol=3, columnspacing=1.6, fontsize=7)
    label(right, "b", "Overlap of match flags")
    bars = right.barh(["Both", "Framework only", "Parent only", "Neither"], combos,
                      color=[RUST, TEAL, BLUE, "#aab6bf"], height=.55)
    right.bar_label(bars, fmt="%d", fontsize=7.7, padding=2)
    right.set_xlim(0, max(combos)*1.12)
    right.invert_yaxis()
    right.set_xlabel("Candidate materials")
    right.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    right.tick_params(axis="y", length=0, labelsize=7.7)
    right.grid(axis="x")
    right.set_axisbelow(True)
    save(fig, output, "fig05_historical_structure_overlap")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "audit", "evaluation", "geometry", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--family-index", type=Path)
    args = p.parse_args()
    style()
    args.output.mkdir()
    utility(args.audit / "path_utility_audit.csv", args.output)
    qe(read_json(args.repo / "results/phase_b4_r2/grid_pair/scores.json"),
       read_csv(args.audit / "dft_cost_ledger.csv"), args.output)
    evidence(read_json(args.evaluation / "manifest.json"), args.output)
    geometric_pool(args.geometry, args.output)
    if args.family_index:
        family_overlap(args.family_index, args.output)


if __name__ == "__main__":
    main()
