"""Verify public NEB assets and screen FP-DeErr Li paths for framework overlap."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import gzip
from io import StringIO
import json
import math
from pathlib import Path
from zipfile import ZipFile

from ase.geometry import find_mic
from ase.io import iread
import numpy as np
from pymatgen.core import Structure

from audit_phase_h2_evaluation import endpoint, fit_pair, framework, rows
from phase_h1_shortlist import anonymous_formula, read_parents, write


FP_HASH = "e211dc4a88fe5739218d6b1e47404624225c5a0d787f4dcce3ef61a5574087c4"
CATALOGUE_HASH = "1fae0e22f837e33cf1384ef5624a920d383565d9100714ca8b42e7bade974e2b"
PROTECTED_HASH = "e37516df396b723498618a601b2a6333ea65320c7691a8763af77d9e18d8978b"
PILOT_MATERIAL = "mp-1191903"
PILOT_ELEMENTS = {"In", "Li", "S", "Si"}


def checked(path, expected):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f"Source SHA256 differs: {path.name}")
    return raw


def classify_mobile(start, end):
    if start.species != end.species or not np.allclose(start.lattice.matrix, end.lattice.matrix,
                                                       atol=1e-6, rtol=0):
        return "unmapped"
    _, distance = find_mic(end.cart_coords-start.cart_coords, start.lattice.matrix, True)
    mover = int(distance.argmax())
    if distance[mover] < .75 or np.delete(distance, mover).max() > min(.60, distance[mover]/2):
        return "ambiguous"
    return start.species[mover].symbol


def run(args):
    data = json.loads(gzip.decompress(checked(args.fp_reference, FP_HASH)))
    catalogue = json.loads(checked(args.literature, CATALOGUE_HASH))
    history = rows(args.history)
    candidates = rows(args.results / "phase_h1/structure_output/candidate_ranking_and_audit.csv")
    candidate_ids = {r["material_id"] for r in candidates}
    history_ids = {r["material_id"] for r in history}
    if (len(data["common_pathway_keys"]), len(catalogue), len(history_ids), len(candidate_ids)) != (154, 621, 867, 24):
        raise ValueError("Unexpected source or cohort sizes")
    structures, parents_sha = read_parents(args.parents, history_ids | candidate_ids)
    groups = defaultdict(list)

    def add(cohort, key, struct):
        fw = framework(struct)
        groups[anonymous_formula(fw)].append((cohort, key, fw))

    for key, (struct, _) in structures.items():
        if key in history_ids:
            add("historical_parent", sha256(key.encode()).hexdigest()[:16], struct)
        else:
            add("candidate_parent", key, struct)

    with ZipFile(args.protected) as bundle:
        plan = json.loads(bundle.read("plan.json"))
        raw = bundle.read("training.xyz")
        if sha256(raw).hexdigest() != plan["sha256"]["training.xyz"]:
            raise ValueError("Protected training geometry hash differs")
    checked(args.protected, PROTECTED_HASH)
    d0_count = 0
    for i, frame in enumerate(iread(StringIO(raw.decode()), "0:278", format="extxyz")):
        add("protected_D0", f"D0-{i:03d}", Structure(frame.cell.array, frame.get_chemical_symbols(),
                                                   frame.positions, coords_are_cartesian=True))
        d0_count += 1
    if d0_count != 278:
        raise ValueError("Protected D0 size differs")

    # The literature catalogue is used here for same-element preliminary overlap.
    literature_groups = defaultdict(list)
    for row in catalogue:
        start = row["structure_ini"]
        f = framework(endpoint(start))
        literature_groups[tuple(sorted(f.symbol_set)), anonymous_formula(f)].append((str(row["jid"]), f))

    records, tasks = {}, []
    for key in data["common_pathway_keys"]:
        row = data["pathways"][key]
        source = row["full_fp_neb_input"]
        start = Structure.from_dict(source["source_start_endpoint_structure"])
        end = Structure.from_dict(source["source_end_endpoint_structure"])
        mobile = classify_mobile(start, end)
        orient = row["endpoint_correspondence"]["orientation_relation"]
        images = row["dft_neb_reference"]["images"]
        finite = len(images) == 7 and [
            (im["image_index"], im["endpoint_role"]) for im in images
        ] == list(enumerate(["initial", *["intermediate"]*5, "final"]))
        for im in images:
            force = np.asarray(im["forces_eV_per_angstrom"])
            finite &= (force.shape == (len(im["structure"]["sites"]), 3)
                       and np.isfinite(force).all() and math.isfinite(im["energy_total_eV"]))
        records[key] = dict(path_id=key, icsd_id=row["identifiers"]["icsd_id"],
                            mobile=mobile, source_orientation=orient, images=len(images),
                            complete_DFT_images_energies_forces=bool(finite),
                            chemsys="-".join(sorted(start.symbol_set)))
        if mobile != "Li" or orient != "direct" or not finite:
            continue
        fw = framework(start)
        anon = anonymous_formula(fw)
        tasks.extend((cohort, key, ref, fw, target) for cohort, ref, target in groups[anon])
        tasks.extend(("literature_same_elements", key, ref, fw, target)
                     for ref, target in literature_groups[tuple(sorted(fw.symbol_set)), anon])
    print(f"Screening {len(tasks)} framework pairs", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pair_results = list(pool.map(fit_pair, tasks, chunksize=12))
    matches = [r for r in pair_results if r["anonymous_framework_fit"]]
    external = {"historical_parent", "protected_D0", "candidate_parent"}
    for key, row in records.items():
        found = [match for match in matches if match["public_id"] == key]
        row["project_exact_hits"] = sum(m["cohort"] in external and m["exact_framework_fit"] for m in found)
        row["project_anonymous_hits_including_exact"] = sum(m["cohort"] in external for m in found)
        row["literature_exact_hits"] = sum(m["cohort"] == "literature_same_elements" and
                                          m["exact_framework_fit"] for m in found)
    eligible = [r for r in records.values() if r["mobile"] == "Li" and r["source_orientation"] == "direct"
                and r["complete_DFT_images_energies_forces"]]
    cohorts = {}
    for cohort in sorted(external | {"literature_same_elements"}):
        selection = [r for r in matches if r["cohort"] == cohort]
        cohorts[cohort] = dict(
            pairs=sum(item[0] == cohort for item in tasks),
            anonymous_hit_paths=len({r["public_id"] for r in selection}),
            exact_hit_paths=len({r["public_id"] for r in selection if r["exact_framework_fit"]}),
        )
    target_bvse = [r["edge_id"] for r in rows(args.bvse) if r["material_id"] == PILOT_MATERIAL]
    summary = dict(
        fp_sha256=FP_HASH, literature_sha256=CATALOGUE_HASH, protected_D0_bundle_sha256=PROTECTED_HASH,
        parent_archive_sha256=parents_sha, FP_total=len(records),
        complete_DFT_paths=sum(r["complete_DFT_images_energies_forces"] for r in records.values()),
        mobile=dict(Counter(r["mobile"] for r in records.values())), eligible_direct_Li=len(eligible),
        eligible_direct_Li_icsd=len({r["icsd_id"] for r in eligible}),
        no_exact_overlap_paths=sum(not r["project_exact_hits"] for r in eligible),
        no_anonymous_overlap_paths=sum(not r["project_anonymous_hits_including_exact"] for r in eligible),
        no_anonymous_overlap_icsd=len({r["icsd_id"] for r in eligible
                                      if not r["project_anonymous_hits_including_exact"]}),
        overlap_screen=cohorts, pilot_BVSE_edges=target_bvse,
        pilot_dft2k_edges=[r["edge_id"] for r in history if r["material_id"] == PILOT_MATERIAL],
        pilot_fp_same_chemical_system=sum(r["chemsys"] == "-".join(sorted(PILOT_ELEMENTS))
                                          for r in records.values()),
        pilot_literature_same_chemical_system=sum(set(r["structure_ini"]["elements"]) == PILOT_ELEMENTS
                                                for r in catalogue),
        FP_reference_method_metadata=data["dft_calculation_metadata"],
    )
    args.output.mkdir()
    write(args.output/"fp_paths.csv", list(records.values()))
    write(args.output/"framework_hits.csv", matches)
    (args.output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("fp_reference", "literature", "history", "bvse", "parents", "protected", "results", "output"):
        parser.add_argument(option, type=Path)
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args())
