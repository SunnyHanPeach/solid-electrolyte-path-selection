"""Prepare local defect and spin checks for two H3 pilot geometries, without any solver."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path

import numpy as np
from ase.data import atomic_numbers
from ase.geometry import find_mic, minkowski_reduce
from ase.io import read

from audit_phase_h2_evaluation import digest
from audit_public_neb_reuse import checked
from prepare_h2_eval_geometry_panel import save_json
from run_phase_b4_source_pilot import archive_geometry, reconstruct


K_SPACING_INV_A = 0.30
MAX_BASELINE_ATOMIC_NUMBER = 80


def select(pilots):
    eligible = [r for r in pilots if not (r["pending_upf"] or r["magnetic_element_screen"]
                or r["local_overlap_flags"] or r["spin_orbit_upf"]
                or r["spin_orbit_header_unresolved"])
                and all(atomic_numbers[s] < MAX_BASELINE_ATOMIC_NUMBER for s in r["stoichiometry"])]
    ranked = sorted(eligible, key=lambda r: (r["atoms"], r["neutral_valence_electron_count"], r["slot"]))
    first = ranked[0]
    backup = next(r for r in ranked if r["stratum"] != first["stratum"])
    return first, backup


def load_profiles(path, ids):
    records = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            edge = record["edge_id"]
            if edge in ids:
                if edge in records:
                    raise ValueError(f"Duplicated archived path {edge}")
                records[edge] = record
    if set(records) != ids:
        raise ValueError("Selected path absent from the archived screen")
    return records


def audit_geometry(row, geometry_row, profiles, parents, centroids, base):
    edge = geometry_row["edge_id"]
    profile = profiles[edge]
    parent_member, parent_bytes = parents[geometry_row["material_id"]]
    centroid_member, centroid_bytes = centroids[edge]
    if (parent_member != profile["parent_member"] or centroid_member != profile["centroid_member"]
            or sha256(parent_bytes).hexdigest() != geometry_row["parent_source_sha256"]
            or sha256(parent_bytes).hexdigest() != profile["parent_sha256"]
            or sha256(centroid_bytes).hexdigest() != profile["centroid_sha256"]):
        raise ValueError(f"Frozen structural provenance changed for {row['slot']}")
    parent = read(StringIO(parent_bytes.decode()), format="extxyz")
    marker = read(StringIO(centroid_bytes.decode()), format="extxyz")
    original, mover, mapping = reconstruct(edge, parent, marker)
    frozen_file = base / "seed_paths" / (row["slot"] + ".extxyz")
    checked(frozen_file, geometry_row["seed_sha256"])
    frozen = read(frozen_file, ":")
    if mover != geometry_row["moving_Li_index_zero_based"] or mover != profile["migrating_Li_0based"]:
        raise ValueError(f"Moving Li differs for {row['slot']}")
    errors = []
    for expected, actual in ((original[0], frozen[0]), (original[-1], frozen[-1])):
        if not np.array_equal(actual.numbers, expected.numbers) or not np.allclose(
                actual.cell, expected.cell, atol=1e-8, rtol=0):
            raise ValueError(f"Endpoint order or cell differs for {row['slot']}")
        errors.append(float(find_mic(actual.positions - expected.positions, actual.cell, True)[1].max()))
    if max(errors) > 1e-5:
        raise ValueError(f"Frozen endpoints no longer reconstruct for {row['slot']}")
    ox = dict(profile["oxidation_states"])
    if len(ox) != len(profile["oxidation_states"]) or ox.get("Li") != 1:
        raise ValueError(f"Ambiguous archived formal valences for {row['slot']}")
    parent_counts = Counter(s for s in marker.get_chemical_symbols() if s != "X")
    defect_counts = Counter(frozen[0].get_chemical_symbols())
    if (parent_counts != defect_counts + Counter({"Li": 1}) or defect_counts != row["stoichiometry"]
            or sum(parent_counts[e] * ox[e] for e in parent_counts) != 0):
        raise ValueError(f"Expected neutral formal parent and one Li vacancy for {row['slot']}")
    top_anion = min((s for s in ox if ox[s] < 0), key=lambda s: ox[s])
    z = row["neutral_valence_electron_count"]
    if not z % 2:
        raise ValueError(f"Expected an odd-electron neutral model for {row['slot']}")
    images = [0, 1 + int(np.argmin([abs(a.info["s"] - .5) for a in frozen[1:-1]])), len(frozen) - 1]
    reciprocal = np.linalg.norm(2 * np.pi * frozen[0].cell.reciprocal(), axis=1)
    k = np.ceil(reciprocal / K_SPACING_INV_A).astype(int)
    reduced, _ = minkowski_reduce(frozen[0].cell, pbc=True)
    a, b = frozen[0], frozen[-1]
    anion_mask = np.array(a.get_chemical_symbols()) == top_anion
    vacancy_distance = find_mic(a.positions[anion_mask] - b.positions[mover], a.cell, True)[1]
    seed_anion = int(np.flatnonzero(anion_mask)[np.argmin(vacancy_distance)])
    return dict(
        slot=row["slot"], parent_counts=dict(parent_counts), defect_counts=dict(defect_counts),
        formal_parent_oxidation_states=ox, removed_Li_count=1, formal_ionic_charge_after_removal=-1,
        neutral_valence_electrons=z, negative_valence_electrons=z+1,
        parent_vacancy_fraction_Li_sites=1/parent_counts["Li"],
        endpoint_max_MIC_error_A=max(errors), source_framework_mapping_error_A=mapping,
        shortest_supercell_translation_A=float(np.linalg.norm(reduced, axis=1).min()),
        moving_Li_index_zero_based=mover, anion_for_initial_spin_seed=top_anion,
        nearest_anion_to_A_vacancy_index_zero_based=seed_anion,
        starting_kmesh=k.tolist(), tighter_kmesh=(k+1).tolist(),
        start_cutoff_Ry=[row["cutoff_reference_wfc_Ry"], row["cutoff_reference_rho_Ry"]],
        geometry_anchors=dict(zip(("A", "middle", "B"), images)),
        anchor_s={str(i): float(frozen[i].info["s"]) for i in images},
        min_Li_contact_A=geometry_row["min_Li_contact_A"],
        source_geometry_sha256=geometry_row["seed_sha256"],
        pseudopotential_status="reference_headers_verified", calculation_launched=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("geometry", "qualification", "profiles", "parents", "centroids", "local_output"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    manifest = args.geometry / "geometry_manifest.json"
    register = args.qualification / "qualification_register.json"
    geometries = {r["slot"]: r for r in json.loads(manifest.read_bytes())}
    pilots = [r for r in json.loads(register.read_bytes()) if r["protocol_role"] == "Protocol Qualification Set"]
    if len(pilots) != 12 or any(geometries[r["slot"]]["role"] != "pilot" for r in pilots):
        raise ValueError("Pilot cohort changed")
    chosen = select(pilots)
    selected_geometries = [geometries[r["slot"]] for r in chosen]
    materials = {r["material_id"] for r in selected_geometries}
    edges = {r["edge_id"] for r in selected_geometries}
    with ThreadPoolExecutor(max_workers=3) as pool:
        parents = pool.submit(archive_geometry, args.parents, "BVEL13k", "material_id", materials, materials)
        centroids = pool.submit(archive_geometry, args.centroids, "nebBVSE122k", "edge_id", edges, edges)
        profiles = pool.submit(load_profiles, args.profiles, edges)
        parent_records, centroid_records, profile_records = parents.result()[1], centroids.result()[1], profiles.result()
    audits = [audit_geometry(q, g, profile_records, parent_records, centroid_records, args.geometry)
              for q, g in zip(chosen, selected_geometries)]
    args.local_output.mkdir(parents=True)
    save_json(args.local_output / "geometry_and_charge_audit.json", dict(
        primary=audits[0], backup=audits[1], geometry_manifest_sha256=digest(manifest),
        qualification_register_sha256=digest(register), reserved_labels_read=0))
    for field, record in zip(("primary", "backup"), audits):
        print(field, len(record["defect_counts"]), record["endpoint_max_MIC_error_A"],
              record["starting_kmesh"], record["tighter_kmesh"],
              record["anchor_s"], record["shortest_supercell_translation_A"])


if __name__ == "__main__":
    main()
