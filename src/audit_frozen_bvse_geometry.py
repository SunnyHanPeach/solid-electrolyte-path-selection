"""Verify that the frozen B5 geometries match the geometry-only BVSE pool."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from ase.geometry import find_mic
from ase.io import read
import numpy as np

from audit_project_data import hash_file, read_csv, write_csv, write_json
from summarize_bvse_readiness import flags


def main(args):
    frozen = args.repo / "results/phase_b5_r"
    manifest = json.loads((frozen / "manifest.json").read_bytes())
    readiness = read_csv(args.repo / "results/project_data_audit_20261127/bvse_prescreen/candidate_readiness_with_bvse.csv")
    geometry = args.b5 / "candidate_images.xyz"
    sources = args.b5 / "candidate_manifest.json"
    for path, expected in ((geometry, manifest["geometry_sha256"]["candidate_images.xyz"]),
                           (sources, manifest["frozen_candidate_manifest_sha256"])):
        if hash_file(path) != expected:
            raise ValueError(f"Frozen input does not match: {path.name}")
    entries = {row["edge_id"]: row for row in json.loads(sources.read_bytes())["rows"] if row["status"] == "included"}
    frames = defaultdict(list)
    for atoms in read(geometry, index=":"):
        frames[atoms.info["edge_id"]].append(atoms)
    paths = {row["edge_id"]: row for row in read_csv(args.screen / "verified/verified_geometry.csv")}
    profiles = {}
    with (args.screen / "profiles.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["edge_id"] in entries:
                profiles[row["edge_id"]] = row
    if len(entries) != manifest["candidate_paths"] or not set(entries) == set(frames) == set(profiles) == {r["edge_id"] for r in readiness}:
        raise ValueError("Frozen pool membership differs between assets")
    report = []
    for edge, entry in entries.items():
        old_file = args.b5 / "scratch" / f"{edge}.extxyz"
        new_file = args.screen / "preconditioned" / f"{edge}.extxyz"
        old_hash, new_hash = hash_file(old_file), hash_file(new_file)
        if old_hash != entry["trajectory_sha256"]:
            raise ValueError(f"Original path checksum differs: {edge}")
        profile = profiles[edge]
        assets_match = all(entry[k] == profile[k] for k in
                           ("parent_member", "parent_sha256", "centroid_member", "centroid_sha256",
                            "migrating_Li_0based", "atoms", "images"))
        frozen_order = sorted(frames[edge], key=lambda a: a.info["image"])
        reordered = read(new_file, index=":")
        shifts = []
        if len(frozen_order) != len(reordered) or len(reordered) != entry["images"]:
            raise ValueError(f"Wrong image count for {edge}")
        for i, (a, b) in enumerate(zip(frozen_order, reordered)):
            expected_mover = entry["migrating_Li_0based"]
            if not (np.array_equal(a.numbers, b.numbers) and np.allclose(a.cell, b.cell, atol=1e-9, rtol=0)
                    and (a.pbc == b.pbc).all() and a.info["image"] == i
                    and a.info["moving"] == expected_mover and a.numbers[expected_mover] == 3):
                raise ValueError(f"Image {i} integrity failed for {edge}")
            shifts.append(float(find_mic(a.positions-b.positions, a.cell, a.pbc)[1].max()))
        row = paths[edge]
        report.append(dict(edge_id=edge, images=len(shifts), atoms=entry["atoms"], input_sources_match=assets_match,
                           initial_min_contact_A=min(i["min_Li_contact_A"] for i in profile["initial"]),
                           final_min_contact_A=min(i["min_Li_contact_A"] for i in profile["final"]),
                           max_periodic_coordinate_residual_A=max(shifts), trajectory_coordinates_match=max(shifts) <= 1e-7,
                           preconditioned_byte_match=old_hash == new_hash,
                           geometry_pass=row["screen_thresholds_passed"] == "True", historical_family_flag=flags(row),
                           original_trajectory_sha256=old_hash, new_trajectory_sha256=new_hash))
    args.output.mkdir()
    write_csv(args.output / "frozen_candidate_provenance.csv", report)
    write_json(args.output / "candidate_provenance_summary.json", dict(
        all_screened_paths=len(paths), audited_candidates=len(report), audited_images=sum(row["images"] for row in report),
        initial_short_contact=sum(row["initial_min_contact_A"] < 1.4 for row in report),
        geometry_pass=sum(row["geometry_pass"] for row in report),
        geometry_matches=sum(row["trajectory_coordinates_match"] for row in report),
        input_matches=sum(row["input_sources_match"] for row in report),
        byte_matches=sum(row["preconditioned_byte_match"] for row in report),
        historical_family_flags=sum(row["historical_family_flag"] for row in report),
        maximum_periodic_residual_A=max(row["max_periodic_coordinate_residual_A"] for row in report),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "b5", "screen", "output"):
        parser.add_argument(name, type=Path)
    main(parser.parse_args())
