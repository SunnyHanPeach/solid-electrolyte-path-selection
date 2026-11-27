"""Group screened FP-DeErr references before any project model scoring."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import platform
from importlib.metadata import version
from time import perf_counter

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from audit_phase_h2_evaluation import MATCHER, digest, fit_pair, framework, inspect_direction, rows
from audit_public_neb_reuse import FP_HASH, checked
from phase_h1_shortlist import anonymous_formula, write


def geometry_hash(images):
    payload = json.dumps([image["structure"] for image in images], sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()


def marker_structure(start, finish, moving, reverse=False):
    structure = start.copy()
    beginning = start[moving].frac_coords
    ending = finish[moving].frac_coords
    structure.remove_sites([moving])
    structure.append("Xa", ending if reverse else beginning)
    structure.append("Xb", beginning if reverse else ending)
    return structure


def compare(job):
    p, q, left, right, same_icsd = job
    f1, forward, _, hash1 = left
    f2, other, other_reverse, hash2 = right
    match = fit_pair(("panel", p, q, f1, f2))
    direct = reverse = False
    if match["exact_framework_fit"]:
        # Markers test the unrelaxed source hop endpoints and occupied Li sublattice.
        # A shared endpoint pair need not have the same relaxed migration channel.
        matcher = StructureMatcher(**MATCHER)
        if anonymous_formula(forward) == anonymous_formula(other):
            direct = matcher.fit(forward, other)
            reverse = matcher.fit(forward, other_reverse)
    return dict(path_a=p, path_b=q, same_icsd=same_icsd,
                anonymous_framework_fit=match["anonymous_framework_fit"],
                same_element_framework_fit=match["exact_framework_fit"],
                source_hop_forward_fit=bool(direct), source_hop_reverse_fit=bool(reverse),
                identical_ordered_DFT_geometries=hash1 == hash2)


def groups(keys, links):
    parent = dict(zip(keys, keys))

    def root(key):
        while parent[key] != key:
            key = parent[key]
        return key

    for first, second in links:
        left, right = sorted((root(first), root(second)))
        parent[right] = left
    return {key: root(key) for key in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "prior_audit", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    started = perf_counter()
    source = json.loads(gzip.decompress(checked(args.reference, FP_HASH)))
    prior_screen = args.prior_audit / "fp_paths.csv"
    prior_hits = args.prior_audit / "framework_hits.csv"
    screened = {r["path_id"]: r for r in rows(prior_screen)}
    retained = {key: row for key, row in sorted(screened.items())
                if row["mobile"] == "Li" and row["source_orientation"] == "direct"
                and row["complete_DFT_images_energies_forces"] == "True"
                and int(row["project_anonymous_hits_including_exact"]) == 0}
    if (len(retained), len({row["icsd_id"] for row in retained.values()})) != (62, 45):
        raise ValueError("The input no longer reproduces the screened 62 paths and 45 ICSD IDs")

    assets, records = {}, {}
    for key, row in retained.items():
        record = source["pathways"][key]
        raw = record["full_fp_neb_input"]
        start = Structure.from_dict(raw["source_start_endpoint_structure"])
        finish = Structure.from_dict(raw["source_end_endpoint_structure"])
        mover, _, displacement, other_displacement = inspect_direction(start, finish)
        images = record["dft_neb_reference"]["images"]
        geometry_sha = geometry_hash(images)
        assets[key] = (framework(start), marker_structure(start, finish, mover),
                       marker_structure(start, finish, mover, reverse=True), geometry_sha)
        records[key] = dict(path_id=key, icsd_id=row["icsd_id"], chemsys=row["chemsys"],
                            dft_images=len(images), source_orientation=row["source_orientation"],
                            source_mobile_Li_index_zero_based=mover,
                            source_mobile_Li_MIC_A=round(displacement, 8),
                            source_other_max_MIC_A=round(other_displacement, 8),
                            ordered_DFT_geometry_sha256=geometry_sha)

    jobs = []
    for p, q in combinations(retained, 2):
        left, right = assets[p], assets[q]
        same_icsd = retained[p]["icsd_id"] == retained[q]["icsd_id"]
        if (same_icsd or anonymous_formula(left[0]) == anonymous_formula(right[0])
                or left[3] == right[3]):
            jobs.append((p, q, left, right, same_icsd))
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
        comparisons = list(executor.map(compare, jobs, chunksize=4))
    families = groups(retained, [(r["path_a"], r["path_b"]) for r in comparisons
                                 if r["same_icsd"] or r["anonymous_framework_fit"]
                                 or r["identical_ordered_DFT_geometries"]])
    exact_groups = groups(retained, [(r["path_a"], r["path_b"]) for r in comparisons
                                     if r["identical_ordered_DFT_geometries"]])
    family_sizes = Counter(families[key] for key in retained if key == exact_groups[key])
    for key, row in records.items():
        row["screened_framework_family"] = families[key]
        row["identical_geometry_group"] = exact_groups[key]
        row["use_for_external_scoring"] = key == exact_groups[key]
        row["family_weight_within_panel"] = (
            1 / (len(family_sizes) * family_sizes[families[key]])
            if row["use_for_external_scoring"] else 0
        )

    literature = []
    for match in rows(prior_hits):
        if match["cohort"] != "literature_same_elements":
            continue
        row = screened[match["public_id"]]
        literature.append(dict(path_id=row["path_id"], literature_jid=match["reference_key"],
                               in_screened_panel=row["path_id"] in retained,
                               project_framework_hits=row["project_anonymous_hits_including_exact"],
                               evidence="same-element framework match only"))

    args.output.mkdir(exist_ok=False, parents=True)
    write(args.output / "panel_paths.csv", list(records.values()))
    write(args.output / "pair_audit.csv", comparisons)
    write(args.output / "literature_overlap.csv", literature)
    summary = dict(
        source_sha256=FP_HASH,
        prior_path_audit_sha256=digest(prior_screen),
        prior_overlap_audit_sha256=digest(prior_hits),
        script_sha256=digest(Path(__file__)),
        matcher=MATCHER,
        n_paths=len(records),
        n_icsd_ids=len({row["icsd_id"] for row in records.values()}),
        n_screened_framework_families=len(family_sizes),
        n_family_link_pairs=sum(r["same_icsd"] or r["anonymous_framework_fit"] for r in comparisons),
        n_anonymous_framework_matches=sum(r["anonymous_framework_fit"] for r in comparisons),
        n_source_hop_match_pairs=sum(r["source_hop_forward_fit"] or r["source_hop_reverse_fit"]
                                     for r in comparisons),
        n_identical_geometry_omissions=sum(key != exact_groups[key] for key in retained),
        n_literature_overlap_paths=len({row["path_id"] for row in literature}),
        n_literature_overlap_paths_in_panel=len({row["path_id"] for row in literature
                                                if row["in_screened_panel"]}),
        dft_barrier_values_used_for_panel_selection=0,
        model_scores_used_for_panel_selection=0,
        software={name: version(name) for name in ("ase", "numpy", "pymatgen")},
        python=platform.python_version(),
        elapsed_seconds=round(perf_counter() - started, 3),
    )
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n",
                                              encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
