"""Audit full LiTraj geometry coverage and run a bounded, CPU-only BVSE pilot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import re
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from ase.build import make_supercell
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from ase.geometry import find_mic
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import FIRE
from scipy.optimize import linear_sum_assignment


FIELD = {
    key: re.compile(rb"(?:^| )" + key.encode() + rb"=([^\s]+)")
    for key in ("material_id", "edge_id")
}
SPLITS = ("train", "val", "test")
STEPS = 60
FMAX = 0.1
IMAGE_COUNT = 5
MATCH_TOL = 1e-5
DIFFERENCE_STEP = 1e-5


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def archive_geometry(path, kind, id_key, keep_ids, store_ids):
    """Check complete XYZ framing and collect matched IDs without loading all atoms."""
    matched = set()
    selected = {}
    counts = {}
    with ZipFile(path) as archive:
        for split in SPLITS:
            name = f"{kind}/{kind}_{split}.xyz"
            seen = 0
            with archive.open(name) as stream:
                while count_row := stream.readline():
                    if not count_row.strip():
                        continue
                    count = int(count_row)
                    header = stream.readline()
                    match = FIELD[id_key].search(header)
                    if count <= 0 or match is None:
                        raise ValueError(f"Invalid frame {seen} in {name}")
                    key = match[1].decode("ascii")
                    lines = [stream.readline() for _ in range(count)]
                    if not all(line.endswith(b"\n") for line in lines):
                        raise ValueError(f"Truncated frame {seen} in {name}")
                    if key in keep_ids:
                        if key in matched:
                            raise ValueError(f"Duplicate eligible ID {key}")
                        matched.add(key)
                    if key in store_ids:
                        blob = b"".join((count_row, header, *lines))
                        selected[key] = (name, blob)
                    seen += 1
            counts[split] = seen
    return matched, selected, counts


def periodic_distance(vectors, atoms):
    return find_mic(vectors, atoms.cell, atoms.pbc)[1]


def reconstruct(edge_id, parent, centroid):
    _, source, target, *shift = edge_id.split("_")
    source, target = int(source), int(target)
    if parent.numbers[source] != 3 or parent.numbers[target] != 3:
        raise ValueError("Both encoded endpoint rows must contain Li")
    if not (parent.pbc.all() and centroid.pbc.all() and np.isfinite(parent.positions).all()
            and np.isfinite(centroid.positions).all()):
        raise ValueError("Invalid periodic geometry")
    transform = centroid.cell.array @ np.linalg.inv(parent.cell.array)
    matrix = np.rint(transform).astype(int)
    if not np.allclose(matrix, transform, atol=2e-7, rtol=0):
        raise ValueError("Centroid cell is not an integer supercell")
    supercell = make_supercell(parent, matrix, wrap=False)

    start = parent.positions[source]
    end = parent.positions[target] + np.asarray(shift, dtype=int) @ parent.cell.array
    marker = centroid.positions[centroid.numbers == 0]
    if len(marker) != 1 or len(supercell) + 1 != len(centroid):
        raise ValueError("Unexpected centroid marker or site count")
    translation = marker[0] - (start + end) / 2

    max_error = 0.
    for element in np.unique(supercell.numbers):
        p = supercell.positions[supercell.numbers == element] + translation
        q = centroid.positions[centroid.numbers == element]
        if len(p) != len(q):
            raise ValueError("Element count disagrees with the centroid structure")
        distances = periodic_distance((p[:, None] - q).reshape(-1, 3), centroid).reshape(len(p), len(q))
        left, right = linear_sum_assignment(distances)
        max_error = max(max_error, float(distances[left, right].max()))
    if max_error > MATCH_TOL:
        raise ValueError(f"Framework periodic match error {max_error:g} angstrom")

    lithium = np.flatnonzero(supercell.numbers == 3)
    source_sites = lithium[periodic_distance(supercell.positions[lithium] - start, supercell) < MATCH_TOL]
    target_sites = lithium[periodic_distance(supercell.positions[lithium] - end, supercell) < MATCH_TOL]
    if len(source_sites) != 1 or len(target_sites) != 1 or source_sites[0] == target_sites[0]:
        raise ValueError("Periodic endpoint identity is ambiguous")
    moving, vacancy = int(source_sites[0]), int(target_sites[0])
    del supercell[vacancy]
    moving -= int(vacancy < moving)
    supercell.positions += translation
    supercell.wrap()
    constraint = FixAtoms(indices=np.flatnonzero(np.arange(len(supercell)) != moving))

    images = []
    for position in np.linspace(start, end, IMAGE_COUNT):
        image = supercell.copy()
        image.positions[moving] = position + translation
        image.set_constraint(constraint)
        images.append(image)
    return images, moving, max_error


def image_profile(images, moving):
    profile = []
    lo = np.arange(len(images[0])) != moving
    previous = None
    length = 0.
    atoms0 = images[0]
    for i, image in enumerate(images):
        if not (np.array_equal(image.numbers, atoms0.numbers) and np.array_equal(image.cell, atoms0.cell)):
            raise ValueError("Atom order or cell changed during the pilot")
        if i:
            step, distance = find_mic(image.positions[moving] - previous, image.cell, image.pbc)
            if not np.allclose(step, image.positions[moving] - previous, atol=MATCH_TOL, rtol=0):
                raise ValueError("Optimized hop has an ambiguous adjacent periodic displacement")
            length += float(distance)
        previous = image.positions[moving].copy()
        frozen_displacement = periodic_distance(image.positions[lo] - atoms0.positions[lo], image).max()
        if frozen_displacement > MATCH_TOL:
            raise ValueError("Frozen framework moved")
        contact = periodic_distance(image.positions[lo] - image.positions[moving], image).min()
        energy = image.get_potential_energy()
        force = np.linalg.norm(image.get_forces()[moving])
        if not np.isfinite([length, contact, energy, force]).all():
            raise ValueError("Nonfinite profile")
        profile.append(dict(image=i, arc_length_A=length, min_Li_contact_A=float(contact),
                            bvse_energy_eV=float(energy), mobile_force_eV_A=float(force)))
    for row in profile:
        row["reaction_coordinate"] = row["arc_length_A"] / length
    return profile


def energy_difference_force(atoms, moving, calculate_energy, step):
    trial = atoms.copy()
    force = np.empty(3)
    for axis in range(3):
        coord = atoms.positions[moving, axis]
        trial.positions[moving, axis] = coord + step
        plus = calculate_energy(trial)
        trial.positions[moving, axis] = coord - step
        minus = calculate_energy(trial)
        trial.positions[moving, axis] = coord
        force[axis] = -(plus - minus) / (2 * step)
    return force


class BVSEEnergyGradient(Calculator):
    """Use the upstream BVSE energy and derive its single mobile-site force."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, energy_calculator, moving):
        super().__init__()
        self.energy_calculator = energy_calculator
        self.moving = moving

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        energy = self.energy_calculator.get_potential_energy
        forces = np.zeros_like(self.atoms.positions)
        self.results["energy"] = energy(self.atoms)
        forces[self.moving] = energy_difference_force(self.atoms, self.moving, energy, DIFFERENCE_STEP)
        self.results["forces"] = forces


def bvse_short(images, moving):
    from ions.calculators.bvse_calculator import BVSECalculator
    from ions.decorate import assign_bvse_params

    upstream = []
    for image in images:
        assign_bvse_params(image, "Li", 1)
        upstream.append(BVSECalculator(sites=[moving]))
    mid = images[len(images) // 2]
    native = upstream[len(images) // 2]
    direct = native.get_forces(mid)[moving]
    documented_energy = native.get_potential_energy
    derivative = energy_difference_force(mid, moving, documented_energy, DIFFERENCE_STEP)
    refined = energy_difference_force(mid, moving, documented_energy, DIFFERENCE_STEP / 2)
    force_error = float(np.max(np.abs(direct - derivative)))
    numerical_error = float(np.max(np.abs(refined - derivative)))
    if not np.isfinite([force_error, numerical_error]).all() or numerical_error > 5e-3:
        raise ValueError(f"BVSE energy derivative step-size check fails by {numerical_error:g} eV/A")
    for image, calculator in zip(images, upstream):
        image.calc = BVSEEnergyGradient(calculator, moving)
    neb = NEB(images, method="improvedtangent")
    initial = image_profile(images, moving)
    residual0 = float(np.linalg.norm(neb.get_forces(), axis=1).max())
    optimizer = FIRE(neb, logfile=None, maxstep=0.12)
    optimizer.run(fmax=FMAX, steps=STEPS)
    residual1 = float(np.linalg.norm(neb.get_forces(), axis=1).max())
    final = image_profile(images, moving)
    return {
        "initial_neb_max_force_eV_A": residual0,
        "final_neb_max_force_eV_A": residual1,
        "upstream_force_vs_energy_gradient_error_eV_A": force_error,
        "half_step_finite_difference_error_eV_A": numerical_error,
        "steps": optimizer.nsteps,
        "converged": bool(residual1 <= FMAX),
        "initial": initial,
        "final": final,
    }


def pilot(cand, parent_record, centroid_record, output):
    import spglib
    from ions.decorate import assign_oxidation_states

    mark = perf_counter()
    parent_member, parent_blob = parent_record
    centroid_member, centroid_blob = centroid_record
    edge = cand["edge_id"]
    parent = read(io.StringIO(parent_blob.decode()), format="extxyz")
    centroid = read(io.StringIO(centroid_blob.decode()), format="extxyz")
    sym = spglib.get_symmetry_dataset((parent.cell.array, parent.get_scaled_positions(), parent.numbers), symprec=1e-3)
    result = {
        "edge_id": edge, "chemsys": cand["chemsys"], "parent_member": parent_member,
        "centroid_member": centroid_member, "parent_sha256": hashlib.sha256(parent_blob).hexdigest(),
        "centroid_sha256": hashlib.sha256(centroid_blob).hexdigest(),
        "spacegroup_number": int(sym.number) if sym else None,
    }
    try:
        # Oxidation states refer to the neutral parent before creating the Li vacancy.
        assign_oxidation_states(parent, max_permutations=10000)
        states = sorted({(symbol, float(q)) for symbol, q in zip(parent.symbols, parent.arrays["oxi_states"])})
        result["oxidation_states"] = states
        if abs(float(parent.arrays["oxi_states"].sum())) > 1e-5:
            raise ValueError("Parent oxidation states are not charge-neutral")
        imgs, moving, error = reconstruct(edge, parent, centroid)
        if imgs[0].arrays["oxi_states"][moving] != 1:
            raise ValueError("Mobile Li did not receive oxidation state +1")
        result.update(atoms=len(imgs[0]), images=len(imgs), migrating_Li_0based=moving,
                      periodic_mapping_error_A=error)
        result.update(bvse_short(imgs, moving))
        result["status"] = "converged" if result["converged"] else "step_limit"
        clean = []
        for image in imgs:
            # Publish these disposable coordinates only to the local scratch directory.
            geom = image.copy()
            geom.set_constraint()
            geom.wrap()
            clean.append(geom)
        write(output / f"{edge}.extxyz", clean, format="extxyz")
    except (ValueError, KeyError, RuntimeError, TypeError) as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    result["seconds"] = perf_counter() - mark
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("parents", "centroids", "bvse_index", "history_index", "candidates", "scratch"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--workers", type=int, default=3,
                        help="Concurrent archive scans and independent BVSE paths")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.scratch.mkdir(exist_ok=False)
    historical = read_csv(args.history_index)
    old_materials = {r["material_id"] for r in historical}
    old_systems = {r["chemsys"] for r in historical}
    eligible = {r["edge_id"]: r["material_id"] for r in read_csv(args.bvse_index)
                if r["material_id"] not in old_materials and r["chemsys"] not in old_systems}
    chosen = read_csv(args.candidates)
    candidate_ids = {r["material_id"] for r in chosen}
    candidate_edges = {r["edge_id"] for r in chosen}
    if (not 1 <= len(chosen) <= 3 or len(candidate_ids) != len(chosen)
            or any(eligible.get(r["edge_id"]) != r["material_id"] for r in chosen)):
        raise ValueError("Pilot roster violates the frozen selection")
    start = perf_counter()
    with ThreadPoolExecutor(max_workers=min(2, args.workers)) as scans:
        parent_scan = scans.submit(archive_geometry, args.parents, "BVEL13k", "material_id",
                                   set(eligible.values()), candidate_ids)
        centroid_scan = scans.submit(archive_geometry, args.centroids, "nebBVSE122k", "edge_id",
                                     set(eligible), candidate_edges)
        parents, p, p_counts = parent_scan.result()
        edges, c, c_counts = centroid_scan.result()
    coverage = {
        "initial_eligible_paths": len(eligible), "initial_eligible_materials": len(set(eligible.values())),
        "centroid_path_records_found": len(edges),
        "materials_with_centroid_and_parent": len({eligible[edge] for edge in edges} & parents),
        "paths_with_centroid_and_parent": sum(eligible[e] in parents for e in edges),
        "parent_frames_by_split": p_counts, "centroid_frames_by_split": c_counts,
        "full_archive_scan_seconds": perf_counter() - start,
    }
    (args.scratch / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(coverage), flush=True)
    def run_candidate(entry):
        mat, edge = entry["material_id"], entry["edge_id"]
        if mat not in p or edge not in c:
            return {"edge_id": edge, "status": "missing_source"}
        return pilot(entry, p[mat], c[edge], args.scratch)

    workers = min(args.workers, len(chosen))
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_candidate, chosen))
    solver_wall_seconds = perf_counter() - started
    for result in results:
        print(json.dumps({k: v for k, v in result.items() if k not in ("initial", "final")}), flush=True)
    output = {
        "optimizer": "ASE FIRE", "neb_method": "improvedtangent", "mobile_sites": 1,
        "step_limit": STEPS, "fmax_eV_A": FMAX, "image_count": IMAGE_COUNT,
        "force_source": "central difference of ions BVSE energy",
        "difference_step_A": DIFFERENCE_STEP, "device": "cpu",
        "archive_scan_workers": min(2, args.workers), "path_workers": workers,
        "parallel_pilot_seconds": solver_wall_seconds,
        "versions": {k: version(k) for k in ("ase", "ions", "pymatgen", "numpy", "scipy", "spglib")},
        "results": results,
    }
    (args.scratch / "bvse_pilot.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
