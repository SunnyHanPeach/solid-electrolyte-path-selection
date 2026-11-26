"""Investigate failed centroid-to-parent mappings and index the recovered pool."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
import json
from pathlib import Path

from ase.build import make_supercell
from ase.geometry import find_mic
from ase.io import read as ase_read
import numpy as np
from scipy.optimize import linear_sum_assignment

from audit_project_data import read_csv, write_csv, write_json
from run_phase_b4_source_pilot import MATCH_TOL, archive_geometry


def diagnose(row, parent_blob, centroid_blob):
    parent, centroid = (ase_read(StringIO(raw.decode()), format="extxyz") for raw in (parent_blob, centroid_blob))
    transform = np.rint(centroid.cell.array @ np.linalg.inv(parent.cell.array)).astype(int)
    supercell = make_supercell(parent, transform, wrap=False)
    _, start, finish, *image = row["edge_id"].split("_")
    initial_li = parent.positions[int(start)]
    final_li = parent.positions[int(finish)] + np.array(image, dtype=int) @ parent.cell.array
    marker = centroid.positions[centroid.numbers == 0][0]
    expected_origin = marker - (initial_li+final_li)/2

    groups = []
    for z in np.unique(supercell.numbers):
        lhs = supercell.positions[supercell.numbers == z]
        rhs = centroid.positions[centroid.numbers == z]
        if len(lhs) != len(rhs):
            raise ValueError(f"Framework species counts changed for {row['edge_id']}")
        groups.append((lhs, rhs))
    anchors, targets = min(groups, key=lambda g: len(g[0]))

    def max_match_error(origin):
        worst = 0.
        for lhs, rhs in groups:
            delta = lhs[:, None] + origin - rhs
            _, distance = find_mic(delta.reshape(-1, 3), centroid.cell, True)
            distance = distance.reshape(len(lhs), len(rhs))
            i, j = linear_sum_assignment(distance)
            worst = max(worst, float(distance[i, j].max()))
        return worst

    offsets = targets - anchors[0]
    old_error = max_match_error(expected_origin)
    matches = [(max_match_error(offset), offset) for offset in offsets]
    best_error, best = min(matches, key=lambda x: x[0])
    _, offset = find_mic(best-expected_origin, centroid.cell, True)
    return dict(edge_id=row["edge_id"], material_id=row["material_id"], chemsys=row["chemsys"],
                baseline_max_periodic_match_error_A=old_error, searched_anchor_origins=len(offsets),
                best_alternative_framework_error_A=best_error,
                minimal_shift_from_published_centroid_A=float(offset),
                status="framework_maps_after_origin_search" if best_error <= MATCH_TOL else "unresolved_framework_difference",
                released_as_path=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parents", "centroids", "path_audit", "family_audit", "h1", "output"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    args.output.mkdir()
    paths = read_csv(args.path_audit)
    families = {r["material_id"]: r for r in read_csv(args.family_audit)}
    h1 = {r["material_id"]: r for r in read_csv(args.h1)}
    flagged = [r for r in paths if r["status"] != "geometry_verified"]
    mats, edges = {r["material_id"] for r in flagged}, {r["edge_id"] for r in flagged}
    with ThreadPoolExecutor(max_workers=2) as scouts:
        p = scouts.submit(archive_geometry, args.parents, "BVEL13k", "material_id", mats, mats)
        c = scouts.submit(archive_geometry, args.centroids, "nebBVSE122k", "edge_id", edges, edges)
        _, parents, _ = p.result()
        _, centroids, _ = c.result()
    with ThreadPoolExecutor(max_workers=8) as workers:
        root_causes = list(workers.map(lambda r: diagnose(r, parents[r["material_id"]][1],
                                                         centroids[r["edge_id"]][1]), flagged))
    write_csv(args.output / "geometry_issue_diagnostics.csv", root_causes)

    material_paths = defaultdict(list)
    for path in paths:
        material_paths[path["material_id"]].append(path)
    if set(families) != set(material_paths) or not set(h1) <= set(families):
        raise ValueError("Completed path and family audits have different membership")
    result = []
    for mid, family in sorted(families.items()):
        rows = material_paths[mid]
        valid = [r for r in rows if r["status"] == "geometry_verified"]
        clean = [r for r in valid if r["contact_below_1_4_A"] == "False" and r["more_than_180_atoms"] == "False"]
        result.append(dict(material_id=mid, chemsys=family["chemsys"], all_paths=len(rows),
                           mapped_paths=len(valid), initial_contact_and_size_pass=len(clean),
                           historical_parent_flag=family["parent_status"] == "match",
                           historical_framework_flag=family["framework_status"] == "match",
                           parent_match_status=family["parent_status"], framework_match_status=family["framework_status"],
                           h1_slot=h1.get(mid, {}).get("panel_role", ""),
                           h1_source_candidate=h1.get(mid, {}).get("edge_id", "")))
    write_csv(args.output / "candidate_material_index.csv", result)
    summary = dict(
        path_mapping_issues=len(flagged), affected_materials=len(mats),
        origin_search_status=dict(Counter(r["status"] for r in root_causes)), materials=len(result),
        at_least_one_path_with_initial_contact_and_size_pass=sum(r["initial_contact_and_size_pass"] > 0 for r in result),
        historical_parent_flags=sum(r["historical_parent_flag"] for r in result),
        historical_framework_flags=sum(r["historical_framework_flag"] for r in result),
        both_flags=sum(r["historical_parent_flag"] and r["historical_framework_flag"] for r in result),
        h1_materials=len(h1),
        h1_framework_flags=sum(bool(r["h1_slot"]) and r["historical_framework_flag"] for r in result),
    )
    write_json(args.output / "manifest.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
