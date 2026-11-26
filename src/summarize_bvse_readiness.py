"""Summarize the BVSE screen at material level and annotate the frozen H1 panel."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from audit_project_data import read_csv, write_csv, write_json
from render_project_audit import BLUE, RUST, TEAL, label, save, style


def flags(row):
    return row["parent_family_flag"] == "True" or row["framework_family_flag"] == "True"


def interval(values, rng):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"materials": 0, "mean": None, "interval_95": [None, None]}
    draws = np.concatenate([
        values[rng.integers(len(values), size=(250, len(values)))].mean(axis=1)
        for _ in range(20)
    ])
    return {"materials": len(values), "mean": float(values.mean()),
            "interval_95": np.quantile(draws, [.025, .975]).tolist()}


def draw(rows, metrics, dest):
    style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.25, 3.0))
    fig.subplots_adjust(left=.115, right=.975, top=.84, bottom=.245, wspace=.43)
    strata = ("short", "ordinary")
    names = ["Short contact", "Other paths"]
    count = [sum(r["initial_stratum"] == x for r in rows) for x in strata]
    passing = [sum(r["initial_stratum"] == x and r["screen_thresholds_passed"] == "True" for r in rows) for x in strata]
    verified = [sum(r["initial_stratum"] == x and r["verified"] == "True" for r in rows) for x in strata]
    x = np.arange(2)
    left.bar(x, passing, color=TEAL, label="Passed")
    left.bar(x, np.asarray(verified) - passing, bottom=passing, color=BLUE, label="Below screen threshold")
    left.bar(x, np.asarray(count) - verified, bottom=verified, color="#c1cbd1", label="Unable to verify")
    left.set_xticks(x, names)
    left.set_ylabel("Directed paths")
    left.set_ylim(0, 1.13 * max(count))
    left.grid(axis="y")
    left.set_axisbelow(True)
    handles, titles = left.get_legend_handles_labels()
    fig.legend(handles, titles, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(.53, .055))
    label(left, "a", "Path screening outcomes")
    for i, key in enumerate(strata):
        m = metrics[key]
        if m["materials"] == 0:
            continue
        y = m["mean"] * 100
        lo, hi = np.asarray(m["interval_95"]) * 100
        right.errorbar(i, y, yerr=[[y - lo], [hi - y]], fmt="o", capsize=3,
                       markersize=4, color=RUST if key == "short" else TEAL, zorder=3)
        right.annotate(f'n = {m["materials"]:,}', (i, y), xytext=(0, 9),
                       textcoords="offset points", ha="center", fontsize=7.4)
    right.set_xticks(x, names)
    right.set_xlim(-.46, 1.46)
    right.set_ylim(0, 100)
    right.set_ylabel("Material-weighted pass rate (%)")
    right.grid(axis="y")
    right.set_axisbelow(True)
    label(right, "b", "Material-weighted pass rate")
    save(fig, dest, "bvse_geometry_prescreen")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("verified", "frozen_h1", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--readiness-table", type=Path)
    args = parser.parse_args()
    out = args.output
    out.mkdir(exist_ok=False, parents=True)
    verified = read_csv(args.verified)
    if not verified or any(row["screen_thresholds_passed"] == "True" and row["verified"] != "True" for row in verified):
        raise ValueError("The input screen has an unverified acceptance")
    by_edge = {r["edge_id"]: r for r in verified}
    if len(by_edge) != len(verified):
        raise ValueError("Repeated directed-path identity")
    by_material = defaultdict(Counter)
    for row in verified:
        key = row["initial_stratum"]
        counts = by_material[row["material_id"]]
        counts[key] += 1
        counts[key + "_pass"] += row["screen_thresholds_passed"] == "True"
    rates = {key: {mat: counts[key + "_pass"]/counts[key] for mat, counts in sorted(by_material.items()) if counts[key]}
             for key in ("short", "ordinary")}
    rng = np.random.default_rng(4817)
    stats = {key: interval(list(materials.values()), rng) for key, materials in rates.items()}
    shared = sorted(rates["short"].keys() & rates["ordinary"].keys())
    stats["short_minus_ordinary_paired_materials"] = interval(
        [rates["short"][mat] - rates["ordinary"][mat] for mat in shared], rng)
    h1 = []
    for orig in read_csv(args.frozen_h1):
        matched = by_edge.get(orig["edge_id"])
        h1.append(dict(edge_id=orig["edge_id"], material_id=orig["material_id"], h1_role=orig["panel_role"],
                       h1_position=orig["shortlist_position"], geometry_record_found=matched is not None,
                       screen_pass=matched["screen_thresholds_passed"] if matched else "",
                       historical_family_flag=flags(matched) if matched else "",
                       geometry_note=matched["detail"] if matched else "not included in the screened pool"))
    passed = [row for row in verified if row["screen_thresholds_passed"] == "True"]
    write_json(out / "material_statistics.json", {
        "attempted_paths": len(verified),
        "accepted_direction_pairs": len({r["canonical_direction_pair"] for r in passed}),
        "accepted_materials": len({r["material_id"] for r in passed}),
        "passed_and_no_historical_family_flag": len({r["canonical_direction_pair"] for r in passed if not flags(r)}),
        "bootstrap": {"replicates": 5000, "seed": 4817, "unit": "material", "metrics": stats},
        "h1_rankings": "frozen and unchanged",
    })
    write_csv(out / "h1_geometry_annotations.csv", h1)
    if args.readiness_table:
        readiness = []
        for orig in read_csv(args.readiness_table):
            matched = by_edge.get(orig["edge_id"])
            readiness.append({
                **orig, "bvse_geometry_audited": matched is not None,
                "bvse_geometry_pass": matched["screen_thresholds_passed"] if matched else "",
                "bvse_min_contact_A": matched["verified_min_contact_A"] if matched else "",
                "historical_parent_match_flag": matched["parent_family_flag"] if matched else "",
                "historical_framework_match_flag": matched["framework_family_flag"] if matched else "",
            })
        write_csv(out / "candidate_readiness_with_bvse.csv", readiness)
    draw(verified, stats, out)
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
