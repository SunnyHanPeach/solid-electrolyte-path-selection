"""Prepare a geometry-only independent evaluation shortlist from existing BVSE paths."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
from hashlib import sha256
from io import StringIO
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import stdev
from zipfile import ZipFile

from ase import Atoms
from ase.geometry import find_mic
from ase.io import iread, read, write as write_ase
import numpy as np
from pymatgen.core import Structure

from audit_phase_h2_evaluation import MATCHER, digest, fit_pair, framework, rows
from audit_public_neb_panel import groups
from audit_public_neb_reuse import FP_HASH, checked
from phase_h1_shortlist import anonymous_formula, read_parents


SALT = "h2-independent-evaluation-geometry-v1"
STRATA = ("氧化物", "硫化物", "卤化物", "复合/其他")
SCAN_PER_STRATUM = 40
TRACKS = {"pilot": 3, "reserve": 2}
ENGINEERING = {"mp-1191903", "mp-1177164", "mp-1110579", "mp-5012"}
PREVIOUSLY_USED_QE_ELEMENTS = {"Li", "O", "Si", "In", "S"}


def stable_rank(key):
    return sha256(f"{SALT}|{key}".encode()).digest()


def source_sample(screen, catalog, h1, history, seed_directory):
    new_index = {row["edge_id"]: row for row in rows(catalog)}
    used_ids = {row["material_id"] for row in h1 + history} | ENGINEERING
    used_chemistry = {row["chemsys"] for row in h1 + history}
    one_per_material = {}
    counts = Counter()
    for row in rows(screen):
        key, mid = row["edge_id"], row["material_id"]
        if (row["verified"] != "True"
                or row["screen_thresholds_passed"] != "True"
                or row["parent_family_flag"] != "False"
                or row["framework_family_flag"] != "False"):
            continue
        counts["passed_previous_geometry_and_family_checks"] += 1
        if mid in used_ids or row["chemsys"] in used_chemistry:
            counts["blocked_material_or_chemical_system"] += 1
            continue
        meta = new_index[key]
        if mid != meta["material_id"] or row["chemsys"] != meta["chemsys"]:
            raise ValueError(f"Source ID disagreement for {key}")
        if meta["迁移离子"] != "Li" or meta["材料家族"] not in STRATA:
            continue
        if not (seed_directory / f"{key}.extxyz").is_file():
            raise ValueError(f"Missing preconditioned geometry for {key}")
        old = one_per_material.get(mid)
        if old is None or stable_rank(key) < stable_rank(old["edge_id"]):
            one_per_material[mid] = {**row, "stratum": meta["材料家族"]}
    strata = {k: sorted([r for r in one_per_material.values() if r["stratum"] == k],
                        key=lambda r: stable_rank(r["material_id"])) for k in STRATA}
    sample = [r for pool in strata.values() for r in pool[:SCAN_PER_STRATUM]]
    return sample, {k: len(pool) for k, pool in strata.items()}, dict(counts)


def framework_tasks(sample, args, history, h1, fp_panel):
    mids = {r["material_id"] for r in history + h1 + sample}
    parents, parent_sha = read_parents(args.parents, mids)
    candidates = {r["material_id"]: framework(parents[r["material_id"]][0]) for r in sample}
    ref = defaultdict(list)

    def add(cohort, key, s):
        host = framework(s)
        ref[anonymous_formula(host)].append((cohort, key, host))

    for cohort, panel in (("historical", history), ("training_candidates", h1)):
        for mid in sorted({r["material_id"] for r in panel}):
            add(cohort, f"{cohort}|{mid}", parents[mid][0])
    with ZipFile(args.d0) as z:
        data = z.read("training.xyz")
        plan = json.loads(z.read("plan.json"))
    if sha256(data).hexdigest() != plan["sha256"]["training.xyz"]:
        raise ValueError("Frozen D0 training member changed")
    n_d0 = 0
    for i, frame in enumerate(iread(StringIO(data.decode()), "0:278", format="extxyz")):
        add("D0", f"D0|{i}", Structure(frame.cell.array, frame.get_chemical_symbols(),
                                     frame.positions, coords_are_cartesian=True))
        n_d0 += 1
    if n_d0 != 278:
        raise ValueError("Unexpected canonical D0 frame count")
    fp = json.loads(gzip.decompress(checked(args.fp_reference, FP_HASH)))["pathways"]
    for row in fp_panel:
        key = row["path_id"]
        reference = fp[key]["full_fp_neb_input"]["source_start_endpoint_structure"]
        add("FP-DeErr", f"FP-DeErr|{key}", Structure.from_dict(reference))
    tasks = [(kind, mid, key, host, reference)
             for mid, host in candidates.items()
             for kind, key, reference in ref[anonymous_formula(host)]]
    tasks.extend(("within_panel_candidates", a, b, x, y)
                 for (a, x), (b, y) in combinations(candidates.items(), 2)
                 if anonymous_formula(x) == anonymous_formula(y))
    return tasks, parents, parent_sha


def select(sample, matches):
    mids = {r["material_id"] for r in sample}
    within = [(r["public_id"], r["reference_key"]) for r in matches
              if r["cohort"] == "within_panel_candidates"]
    cluster = groups(mids, within)
    blocked = {cluster[r["public_id"]] for r in matches if r["cohort"] != "within_panel_candidates"}
    seen = set()
    formulas = set()
    result = []
    for role, count in TRACKS.items():
        for stratum in STRATA:
            cohort = [r for r in sample if r["stratum"] == stratum]
            chosen = 0
            for row in cohort:
                family = cluster[row["material_id"]]
                if family in blocked or family in seen or row["chemsys"] in formulas:
                    continue
                result.append({**row, "role": role, "framework_group": family})
                seen.add(family)
                formulas.add(row["chemsys"])
                chosen += 1
                if chosen == count:
                    break
    return result, blocked


def export_seed(row, source, target):
    frames = read(source, ":", format="extxyz")
    if len(frames) != 5 or frames[0].numbers.tolist().count(3) < 1:
        raise ValueError(f"Expected five Li migration geometries: {source.name}")
    first = frames[0]
    for a in frames:
        if not (np.array_equal(first.numbers, a.numbers)
                and np.allclose(first.cell, a.cell, rtol=0, atol=1e-8)
                and a.pbc.all()):
            raise ValueError(f"Inconsistent image order or cell: {source.name}")
    all_displacements = np.stack([find_mic(a.positions - first.positions, first.cell, True)[1]
                                  for a in frames[1:]])
    moving = np.flatnonzero(all_displacements.max(axis=0) > 1e-5)
    if len(moving) != 1 or first.numbers[moving[0]] != 3:
        raise ValueError(f"Expected a single moving Li and fixed framework: {source.name}")
    mover = int(moving[0])
    steps = np.diff([a.positions[mover] for a in frames], axis=0)
    _, distance = find_mic(steps, first.cell, True)
    cumulative = np.r_[0., np.cumsum(distance)]
    if np.any(distance <= 1e-7) or abs(cumulative[-1] - float(row["verified_final_length_A"])) > 1e-4:
        raise ValueError(f"Archived reaction coordinate changed: {source.name}")
    keep = np.arange(len(first)) != mover
    shortest = min(float(find_mic(a.positions[keep] - a.positions[mover], a.cell, True)[1].min())
                   for a in frames)
    if shortest < 1.4 or abs(shortest - float(row["verified_min_contact_A"])) > 1e-4:
        raise ValueError(f"Archived contact screen changed: {source.name}")
    clean = [Atoms(numbers=a.numbers, positions=a.positions, cell=a.cell, pbc=a.pbc,
                   info=dict(image=i, s=float(s / cumulative[-1]), moving_Li_index=mover))
             for i, (a, s) in enumerate(zip(frames, cumulative))]
    write_ase(target, clean, format="extxyz")
    return dict(atoms=len(first), images=len(frames), moving_Li_index_zero_based=mover,
                trajectory_length_A=float(cumulative[-1]), min_Li_contact_A=shortest,
                seed_source_sha256=digest(source), seed_sha256=digest(target))


def precision_plan(path):
    summary = {}
    for contrast in ("net", "absolute"):
        values = [float(r["barrier_gain_meV"]) for r in rows(path) if r["contrast"] == contrast]
        sigma = stdev(values)
        summary[contrast] = dict(families=len(values), sd_meV=sigma,
                                 idealized_half_width_with_12_families_meV=1.96*sigma/math.sqrt(12),
                                 idealized_n_for_half_width_0_25_meV=math.ceil((1.96*sigma/.25)**2))
    return summary


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("results", "screen", "catalog", "history", "parents", "d0", "fp_reference",
                "seed_directory", "local_output", "public_output"):
        p.add_argument(key, type=Path)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    r = args.results
    h1_file = r / "phase_h1/structure_output/candidate_ranking_and_audit.csv"
    fp_panel_file = r / "public_neb_panel/panel_paths.csv"
    families = r / "public_neb_panel/mp753143_effects/family_gains.csv"
    history, h1, fp_panel = rows(args.history), rows(h1_file), rows(fp_panel_file)
    if (len({row["material_id"] for row in history}), len(h1), len(fp_panel)) != (867, 24, 62):
        raise ValueError("Frozen references differ from the H2 audit")
    sample, source_counts, filters = source_sample(args.screen, args.catalog, h1, history, args.seed_directory)
    tasks, parents, parent_sha = framework_tasks(sample, args, history, h1, fp_panel)
    print(f"Comparing {len(sample)} evaluation candidates across {len(tasks)} structure pairs.", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        matches = [task for task in pool.map(fit_pair, tasks, chunksize=8)
                   if task["anonymous_framework_fit"]]
    selected, blocked = select(sample, matches)
    args.local_output.mkdir(parents=True)
    args.public_output.mkdir(parents=True, exist_ok=True)
    (args.local_output / "seed_paths").mkdir()
    detail = []
    for i, row in enumerate(selected, start=1):
        key = row["edge_id"]
        slot = f"eval-{i:02d}"
        geometry = export_seed(row, args.seed_directory / f"{key}.extxyz",
                               args.local_output / f"seed_paths/{slot}.extxyz")
        detail.append(dict(slot=slot, edge_id=key, material_id=row["material_id"], chemsys=row["chemsys"],
                           role=row["role"], stratum=row["stratum"], framework_group=row["framework_group"],
                           additional_QE_element_review=sorted(set(row["chemsys"].split("-")) -
                                                               PREVIOUSLY_USED_QE_ELEMENTS),
                           parent_source_sha256=parents[row["material_id"]][1], **geometry))
    save_json(args.local_output / "geometry_manifest.json", detail)
    save_json(args.local_output / "geometry_match_hits.json", matches)
    save_json(args.public_output / "summary.json", dict(
        rule=dict(salt=SALT, representatives_per_stratum_screened=SCAN_PER_STRATUM,
                  requested_per_stratum=TRACKS, matching=MATCHER, selection_uses_DFT_labels=False),
        prior_filters=filters, eligible_materials_by_stratum=source_counts,
        material_count_screened=len(sample), matching_pairs=len(tasks),
        strata={role: dict(Counter(row["stratum"] for row in detail if row["role"] == role))
                for role in TRACKS},
        matching_hits_by_scope=dict(Counter(row["cohort"] for row in matches)),
        candidate_clusters_blocked_by_reference=len(blocked),
        independent_geometry_candidates=len(detail),
        verified_seed_images=sum(row["images"] for row in detail),
        all_elements_previously_used_in_QE=sum(not row["additional_QE_element_review"] for row in detail),
        strict_primary_panel_labeled_paths=0, reserved_DFT_labels_read=0,
        precision_planning_from_existing_external_case=precision_plan(families),
        sha256={path.name: digest(path) for path in
                (args.screen, args.catalog, args.history, args.d0, h1_file, fp_panel_file, families,
                 args.local_output / "geometry_manifest.json", args.local_output / "geometry_match_hits.json")},
        BVEL_parent_archive_sha256=parent_sha, FP_DFT_source_archive_sha256=FP_HASH))


if __name__ == "__main__":
    main()
