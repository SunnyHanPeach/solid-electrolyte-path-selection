"""Freeze and reconstruct the B5-R path pool from geometrical metadata."""
import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import io
import json
from math import gcd
from pathlib import Path
from time import time
from zipfile import ZipFile

from ase import Atoms
from ase.io import read, write
import numpy as np
import spglib

from phase_a0_geometry import parse_path, reaction_coordinate
from run_phase_b4_source_pilot import archive_geometry, pilot, read_csv


def digest(raw):
    return sha256(raw).hexdigest()


def clean(atoms, **info):
    return Atoms(numbers=atoms.numbers, positions=atoms.positions, cell=atoms.cell, pbc=atoms.pbc, info=info)


def parent_family(raw):
    atoms = read(io.StringIO(raw.decode()), format="extxyz")
    symmetry = spglib.get_symmetry_dataset(
        (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers), symprec=1e-3
    )
    elements = Counter(atoms.get_chemical_symbols())
    divisor = gcd(*elements.values())
    stoichiometry = tuple(sorted(n // divisor for n in elements.values()))
    anions = tuple(sorted(elements.keys() & {"O", "S", "Se", "N", "F", "Cl", "Br", "I"}))
    return (int(symmetry.number) if symmetry is not None else 0, stoichiometry, anions)


def deterministic_pool(low, historic, species, spec):
    materials = {r["material_id"] for r in historic}
    chemsys = {r["chemsys"] for r in historic}
    eligible = [r for r in low if r["material_id"] not in materials and r["chemsys"] not in chemsys]
    if len({r["material_id"] for r in eligible}) != 2647 or len(eligible) != 16155:
        raise ValueError("B4 source pool identity changed")
    blocked = {"mp-1177164", "mp-1110579", "mp-1191903", "mp-5012"}
    by_family = defaultdict(list)
    for row in eligible:
        if row["material_id"] not in blocked and set(row["chemsys"].split("-")) <= species:
            by_family[row["材料家族"]].append(row)
    token = lambda row: digest(f'{spec["seed"]}|{row["edge_id"]}'.encode())
    queues = {k: deque(sorted(rows, key=token)) for k, rows in by_family.items()}
    selected, used_material, used_system = [], set(), set()
    while any(queues.values()) and len(selected) < spec["attempts"]:
        for family in sorted(queues):
            stack = queues[family]
            while stack:
                row = stack.popleft()
                if row["material_id"] in used_material or row["chemsys"] in used_system:
                    continue
                selected.append(row)
                used_material.add(row["material_id"])
                used_system.add(row["chemsys"])
                break
            if len(selected) >= spec["attempts"]:
                break
    return selected


def prepare(args):
    repo, out = args.repo, args.output
    spec = json.loads((repo / "results/phase_b5_r/protocol.json").read_bytes())["candidate_pool"]
    historical = read_csv(args.history)
    candidates = read_csv(repo / "results/b2_overnight_01/six_hour_candidates.csv")
    a1 = json.loads((repo / "results/phase_a1/manifest.json").read_bytes())
    lookup = {r["edge_id"]: r for r in historical}
    selected = deterministic_pool(read_csv(args.low), historical, set(a1["species"]), spec)
    out.mkdir()
    (out / "scratch").mkdir()
    parents = {r["material_id"] for r in selected}
    historical_parents = {r["material_id"] for r in historical}
    edges = {r["edge_id"] for r in selected}
    with ThreadPoolExecutor(max_workers=2) as executor:
        p = executor.submit(archive_geometry, args.bvel, "BVEL13k", "material_id",
                            parents | historical_parents, parents | historical_parents)
        c = executor.submit(archive_geometry, args.bvse, "nebBVSE122k", "edge_id", edges, edges)
        _, parent_blobs, _ = p.result()
        _, centroid_blobs, _ = c.result()
    have_historical = {i: parent_blobs[i][1] for i in historical_parents if i in parent_blobs}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        old_families = set(pool.map(parent_family, have_historical.values()))
    del have_historical

    def evaluate(row):
        edge, mat = row["edge_id"], row["material_id"]
        summary = {k: row[k] for k in ("edge_id", "material_id", "chemsys", "em", "_split", "材料家族")}
        if time() > args.stop:
            return dict(summary, status="budget_limit")
        if mat not in parent_blobs or edge not in centroid_blobs:
            return dict(summary, status="missing_source")
        try:
            family = parent_family(parent_blobs[mat][1])
            summary["family"] = json.dumps(family, separators=(",", ":"))
            if family[0] == 0 or family in old_families:
                return dict(summary, status="family_proxy_overlap_or_unresolved")
            job = pilot(row, parent_blobs[mat], centroid_blobs[edge], out / "scratch")
            summary.update(job)
            if job["status"] != "converged":
                return summary
            profile = job["final"]
            if job["atoms"] > spec["atoms_max"] or min(r["min_Li_contact_A"] for r in profile) < spec["li_contact_floor_A"]:
                summary["status"] = "geometry_filter"
                return summary
            bump = max(r["bvse_energy_eV"] for r in profile[1:-1]) - max(profile[0]["bvse_energy_eV"], profile[-1]["bvse_energy_eV"])
            if bump < spec["minimum_internal_peak_above_both_ends_eV"]:
                summary["status"] = "no_resolved_internal_peak"
            return summary
        except Exception as exc:
            return dict(summary, status="reconstruction_error", exception=f"{type(exc).__name__}: {exc}")

    rows, used, new_images = [], set(), []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, len(selected), args.workers):
            for row in pool.map(evaluate, selected[start:start + args.workers]):
                if row["status"] == "converged":
                    if row["family"] in used:
                        row["status"] = "pilot_family_proxy_overlap"
                    elif len(used) >= spec["maximum_selected"]:
                        row["status"] = "quota"
                    else:
                        used.add(row["family"])
                        path_file = out / "scratch" / f"{row['edge_id']}.extxyz"
                        row["trajectory_sha256"] = digest(path_file.read_bytes())
                        row["status"] = "included"
                        frames = read(path_file, ":", format="extxyz")
                        for i, a in enumerate(frames):
                            new_images.append(clean(a, cohort="candidate", edge_id=row["edge_id"], image=i,
                                                    moving=row["migrating_Li_0based"],
                                                    s=row["final"][i]["reaction_coordinate"]))
                rows.append(row)
            if len(used) >= spec["maximum_selected"] or time() > args.stop:
                break
    for row in selected[len(rows):]:
        rows.append({k: row[k] for k in ("edge_id", "material_id", "chemsys", "em", "_split", "材料家族")} |
                    {"status": "not_needed_at_quota" if len(used) >= spec["maximum_selected"] else "budget_limit"})
    (out / "candidate_manifest.json").write_text(json.dumps(
        {"historical_parents_with_metadata": sum(i in parent_blobs for i in historical_parents),
         "historical_parent_total": len(historical_parents), "reference_proxy_family_count": len(old_families),
         "selected": len(used), "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if new_images:
        write(out / "candidate_images.xyz", new_images, format="extxyz", write_results=False)
    if digest(args.neb.read_bytes()) != "b7a99d89337902e9e1da319f57547170fdb132bf15bf6ffef03a0140e2207d7f":
        raise ValueError("Historical source ZIP differs from B3 history")
    image_rows = []
    with ZipFile(args.neb) as z:
        for c in candidates:
            edge = c["edge_id"]
            moving = int(lookup[edge]["moving_Li_row"]) - 1
            _, source, images = parse_path(z, edge, "init", moving)
            if source != c["init_sha256"] or len(images) != int(c["images"]):
                raise ValueError(f"Historical path {edge} differs from B2")
            _, _, s = reaction_coordinate([a.positions[moving] for a in images], images[0].cell, images[0].pbc)
            for i, a in enumerate(images):
                image_rows.append(clean(a, cohort="historical", edge_id=edge, image=i, moving=moving, s=float(s[i])))
    write(out / "historical_images.xyz", image_rows, format="extxyz", write_results=False)
    print(f"{len(used)} new paths, {len(image_rows)} historical images, {len(parent_blobs)} matched parent IDs")


def prepare_d0_reference(bundle, weights, output):
    from audit_phase_b0 import frame_prefix
    from phase_a0_geometry import atoms_from_bytes, csv_rows

    if digest(bundle.read_bytes()) != "6bfd1a816609771b8a5124b6e3000262eb5258969b6756243997b8d632126862":
        raise ValueError("Canonical training bundle checksum differs")
    with ZipFile(bundle) as archive:
        raw = frame_prefix(archive.read("training.xyz"), 278)
        if digest(raw) != "dda468e74c0dbd46c3d5b07121b900184a917dd319415e826af6e25b56ee7081":
            raise ValueError("Canonical D0 prefix differs")
        metadata = csv_rows(archive.read("training_frames.csv"))[:278]
        frames = atoms_from_bytes(raw)
    reference, seen = [], set()
    for i, (frame, meta) in enumerate(zip(frames, metadata)):
        material = meta["material_id"]
        if material in seen:
            continue
        if 3 not in frame.numbers:
            raise ValueError(f"No lithium in D0 reference {material}")
        seen.add(material)
        reference.append(clean(frame, cohort="reference", edge_id=material, image=i, moving=-1, s=0.0))
    if len(reference) != 50:
        raise ValueError("Expected 50 canonical reference materials")
    with ZipFile(weights) as archive:
        checkpoint = archive.read("D0_17.pt")
    if digest(checkpoint) != "c0ed6bed7c168f24b0c9392dd2d786c10f209072493858aa04dbcc1ab3c4e69a":
        raise ValueError("Canonical D0_17 checkpoint differs")
    write(output / "reference_images.xyz", reference, format="extxyz", write_results=False)
    (output / "D0_17.pt").write_bytes(checkpoint)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "low", "history", "bvel", "bvse", "neb", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--stop", type=float, default=float("inf"))
    p.add_argument("--reference-bundle", type=Path)
    p.add_argument("--weights", type=Path)
    args = p.parse_args()
    if bool(args.reference_bundle) != bool(args.weights):
        p.error("--reference-bundle and --weights are required together")
    prepare(args)
    if args.reference_bundle:
        prepare_d0_reference(args.reference_bundle, args.weights, args.output)


if __name__ == "__main__":
    main()
