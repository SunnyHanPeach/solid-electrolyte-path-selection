"""Audit public migration references without using barrier values or private labels."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from hashlib import md5, sha256
from itertools import combinations
from io import StringIO
import json
from pathlib import Path
import re
from time import perf_counter
from zipfile import ZipFile

from ase.geometry import find_mic
from ase.io import iread
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from phase_h1_shortlist import anonymous_formula, read_parents, write


UPSTREAM_SHA256 = "fe8347abbbdb6d941805f645b7d5651eb919ecd3649c12f44535b948a737d8cd"
PUBLISHED_MD5 = "878646be7126dd30175576a494b726b4"
MATCHER = dict(ltol=.15, stol=.20, angle_tol=5., primitive_cell=True, scale=True)
D0_BUNDLE_SHA256 = "6bfd1a816609771b8a5124b6e3000262eb5258969b6756243997b8d632126862"


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def geometry_key(row):
    keys = ("sys_name", "structure_ini", "structure_fin")
    return json.dumps([row[key] for key in keys], sort_keys=True, separators=(",", ":"))


def endpoint(raw):
    return Structure(raw["lattice_mat"], raw["elements"], raw["coords"],
                     coords_are_cartesian=raw["cartesian"])


def inspect_direction(start, finish):
    if start.species != finish.species or not np.allclose(start.lattice.matrix, finish.lattice.matrix, atol=1e-6, rtol=0):
        raise ValueError("Curated ordered endpoint mapping differs")
    vector, distance = find_mic(finish.cart_coords - start.cart_coords, start.lattice.matrix, True)
    moving = int(distance.argmax())
    next_move = float(np.delete(distance, moving).max())
    if start.species[moving].symbol != "Li" or distance[moving] < .75 or next_move > min(.60, distance[moving] / 2):
        raise ValueError("Largest isolated Li displacement could not be reproduced")
    return moving, vector[moving], float(distance[moving]), next_move


def framework(structure):
    result = structure.copy()
    result.remove_species(["Li"])
    if len(result) == 0:
        raise ValueError("Cannot compare a framework containing only Li")
    return result


def fit_pair(pair):
    cohort, qid, key, left, right = pair
    matcher = StructureMatcher(**MATCHER)
    same_species = set(left.symbol_set) == set(right.symbol_set)
    exact = same_species and matcher.fit(left, right)
    anonymous = exact or matcher.fit_anonymous(left, right)
    return dict(cohort=cohort, public_id=qid, reference_key=key,
                same_framework_elements=same_species, exact_framework_fit=bool(exact),
                anonymous_framework_fit=bool(anonymous))


def read_catalogues(args):
    old_bytes = args.upstream.read_bytes()
    published_bytes = args.published.read_bytes()
    if sha256(old_bytes).hexdigest() != UPSTREAM_SHA256 or md5(published_bytes).hexdigest() != PUBLISHED_MD5:
        raise ValueError("Public catalogue digest mismatch")
    old, published = json.loads(old_bytes), json.loads(published_bytes)
    frozen = json.loads((args.results / "phase_b5_r/evaluation_asset_audit.json").read_bytes())
    if frozen["local_dataset_sha256"] != UPSTREAM_SHA256 or len(old) != frozen["public_records"]:
        raise ValueError("Public record set differs from B5-R")
    selected = {row["public_id"]: row["chemsys"] for row in frozen["disjoint_candidate_metadata"]}
    if len(selected) != 42:
        raise ValueError("B5-R expected exactly 42 audited pointers")
    lookup = defaultdict(list)
    for row in published:
        lookup[geometry_key(row)].append(row)
    records, public = [], {}
    sources = {r["source_key"]: r for r in rows(args.sources)}
    for row in old:
        pid = row["jid"]
        if pid not in selected:
            continue
        match, = lookup[geometry_key(row)]
        src_key = re.search(r"@\w+\{([^,]+),", row["bibtex"])[1]
        published_key = re.search(r"@\w+\{([^,]+),", match["bibtex"])[1]
        if src_key != published_key:
            raise ValueError(f"Public journal reference changed for {pid}")
        src = sources[src_key]
        if not isinstance(match["calc_metadata"], dict):
            raise ValueError(f"Missing finalized computation metadata for {pid}")
        start, end = endpoint(row["structure_ini"]), endpoint(row["structure_fin"])
        chemsys = "-".join(sorted(start.symbol_set))
        if chemsys != selected[pid] or row["XC"] != "GGA":
            raise ValueError(f"Frozen chemistry differs for {pid}")
        moving, vector, hop, other = inspect_direction(start, end)
        public[pid] = framework(start)
        pp = str(match["calc_metadata"]["PP_INFO"])
        records.append(dict(
            public_id=pid, published_id=match["jid"], system=row["sys_name"], chemsys=chemsys,
            source_key=src_key, coarse_xc=row["XC"], method_audit=src["method_audit"],
            software_reported=match["calc_metadata"]["SOFTWARE_INFO"],
            pp_mentions_pbe=bool(re.search(r"\bPBE\b", pp, re.I)),
            migrating_Li_index_0=moving, hop_A=hop, maximum_other_A=other,
            displacement_x_A=float(vector[0]), displacement_y_A=float(vector[1]), displacement_z_A=float(vector[2]),
            direction_status="curated_A_to_B_geometry_only",
            published_catalogue_has_full_neb=False,
            primary_full_neb_status=src["primary_full_neb_status"],
            original_directed_barrier_linkage_verified=False,
            primary_panel_eligible=False, blocker="missing_verified_full_path_and_directed_label",
        ))
    if set(public) != set(selected) or {r["source_key"] for r in records} != set(sources):
        raise ValueError("Public records or source inventory is incomplete")
    return records, public, dict(original_count=len(old), published_count=len(published),
                                 original_sha256=UPSTREAM_SHA256, published_md5=PUBLISHED_MD5,
                                 published_sha256=sha256(published_bytes).hexdigest())


def build_comparisons(args, public):
    history = rows(args.history)
    historical_ids = {r["material_id"] for r in history}
    candidate_table = rows(args.results / "phase_h1/structure_output/candidate_ranking_and_audit.csv")
    candidate_ids = {r["material_id"] for r in candidate_table}
    shortlist = {r["material_id"] for r in rows(args.results / "phase_h1/structure_output/prospective_review_panel.csv")}
    if len(historical_ids) != 867 or len(candidate_ids) != 24 or len(shortlist) != 10 or historical_ids & candidate_ids:
        raise ValueError("Unexpected history or H1 cohort counts")
    parents, parents_sha = read_parents(args.parent_geometry, historical_ids | candidate_ids)
    b5 = json.loads((args.results / "phase_b5_r/manifest.json").read_bytes())
    reference_sha = digest(args.d0_geometry)
    if reference_sha != b5["geometry_sha256"]["reference_images.xyz"]:
        raise ValueError("D0 reference geometry differs from the frozen asset")
    groups = defaultdict(list)
    for mid, (structure, _) in parents.items():
        reduced = framework(structure)
        cohort = "historical_parent" if mid in historical_ids else "H1_shortlist" if mid in shortlist else "B5_other_candidate"
        key = sha256(mid.encode()).hexdigest()[:16] if mid in historical_ids else mid
        groups[anonymous_formula(reduced)].append((cohort, key, reduced))
    if args.d0_bundle:
        if digest(args.d0_bundle) != D0_BUNDLE_SHA256:
            raise ValueError("Canonical D0 bundle digest changed")
        with ZipFile(args.d0_bundle) as archive:
            plan = json.loads(archive.read("plan.json"))
            training = archive.read("training.xyz")
        if sha256(training).hexdigest() != plan["sha256"]["training.xyz"]:
            raise ValueError("Canonical D0 training member digest changed")
        frames = iread(StringIO(training.decode()), "0:278", format="extxyz")
        d0_scope = "canonical_first_278_training_frames"
        expected_d0 = 278
    else:
        frames = iread(args.d0_geometry, ":", format="extxyz")
        d0_scope = "B5_reference_geometries"
        expected_d0 = b5["references"]
    d0_count = 0
    for frame in frames:
        reduced = framework(Structure(frame.cell.array, frame.get_chemical_symbols(),
                                      frame.positions, coords_are_cartesian=True))
        # These frames contain geometry only. Keep the reference identity local to the audit.
        groups[anonymous_formula(reduced)].append(("D0_geometry", f"D0-frame-{d0_count:03}", reduced))
        d0_count += 1
    if d0_count != expected_d0:
        raise ValueError("D0 reference frame count differs")
    work = [(cohort, pid, key, left, right)
            for pid, left in public.items()
            for cohort, key, right in groups[anonymous_formula(left)]]
    work.extend(("public_internal", first, str(second), public[first], public[second])
                for first, second in combinations(public, 2)
                if anonymous_formula(public[first]) == anonymous_formula(public[second]))
    return work, dict(history_materials=len(historical_ids), b5_parent_materials=len(candidate_ids),
                      h1_review_materials=len(shortlist), d0_frames=d0_count, d0_scope=d0_scope,
                      history_metadata_sha256=digest(args.history), parent_geometry_sha256=parents_sha,
                      b5_reference_geometry_sha256=reference_sha,
                      d0_bundle_sha256=D0_BUNDLE_SHA256 if args.d0_bundle else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "published", "results", "sources", "history", "parent_geometry", "d0_geometry", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--d0-bundle", type=Path)
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be positive")
    args.output.mkdir()
    start = perf_counter()
    records, public, catalogue = read_catalogues(args)
    work, assets = build_comparisons(args, public)
    print(f"Reconciled {len(records)} public records; testing {len(work)} framework pairs using {args.workers} workers.", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        comparisons = list(pool.map(fit_pair, work, chunksize=8))
    hits = Counter(row["public_id"] for row in comparisons
                   if row["cohort"] != "public_internal" and row["anonymous_framework_fit"])
    for row in records:
        row["external_anonymous_framework_matches"] = hits[row["public_id"]]
    write(args.output / "record_audit.csv", records)
    write(args.output / "framework_overlap.csv", comparisons)
    fields = list(records[0])
    write(args.output / "primary_panel_release.csv", [r for r in records if r["primary_panel_eligible"]], fields)
    by_cohort = {}
    for name in ("historical_parent", "D0_geometry", "H1_shortlist", "B5_other_candidate", "public_internal"):
        group = [r for r in comparisons if r["cohort"] == name]
        matched = {r["public_id"] for r in group if r["anonymous_framework_fit"]}
        if name == "public_internal":
            matched |= {int(r["reference_key"]) for r in group if r["anonymous_framework_fit"]}
        by_cohort[name] = dict(attempted=len(group),
                               anonymous_matches=sum(r["anonymous_framework_fit"] for r in group),
                               exact_matches=sum(r["exact_framework_fit"] for r in group),
                               affected_public_records=len(matched))
    summary = dict(catalogues=catalogue, references=assets, sources=len({r["source_key"] for r in records}),
                   rows_audited=len(records), method_audit=dict(Counter(r["method_audit"] for r in records)),
                   geometry_mapping_verified=len(records), directed_source_label_verified=0,
                   primary_released=0, overlap=by_cohort, matcher=MATCHER,
                   barrier_values_read_for_analysis=0, reserved_dft_labels_read=0,
                   workers=args.workers, elapsed_seconds=round(perf_counter() - start, 3))
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
