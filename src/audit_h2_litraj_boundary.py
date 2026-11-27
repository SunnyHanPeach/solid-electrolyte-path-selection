"""Audit LiTraj split boundaries while protecting the four sealed materials."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from itertools import combinations
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from audit_h2_cost_and_reuse import archived_energies
from audit_phase_h2_evaluation import MATCHER, digest, fit_pair, framework, rows
from audit_public_neb_panel import groups
from phase_h1_shortlist import anonymous_formula, read_parents

SEALED_MEMBER = "strict_original_test/manifest.csv"
SEALED_SHA = "a47442e74f8851e8bf2eba898dc28382c28c803bb199142ada3381f3a1f687e4"
SPLITS = ("train", "val", "test")


def read_index(archive, member):
    return list(csv.DictReader(archive.read(member).decode("utf-8-sig").splitlines()))


def sealed_materials(path):
    from hashlib import sha256

    with ZipFile(path) as archive:
        table = archive.read(SEALED_MEMBER)
    if sha256(table).hexdigest() != SEALED_SHA:
        raise ValueError("Label-free sealed manifest does not match the project record")
    materials = {r["material_id"] for r in csv.DictReader(table.decode("utf-8-sig").splitlines())}
    if len(materials) != 4:
        raise ValueError("Expected four sealed materials")
    return materials


def profile_counts(archive, index, protected):
    output = {}
    for split in ("val", "test"):
        n = forward = span = 0
        maximum = 0.
        for row in index:
            if row["_split"] != split or row["material_id"] in protected:
                continue
            e = archived_energies(archive, f"nebDFT2k/{row['edge_id']}_relaxed.xyz")
            index_value = float(row["em_dft"])
            error = abs(e.max() - e[0] - index_value)
            forward += error <= .001
            span += abs(e.max() - e.min() - index_value) <= .001
            maximum = max(maximum, error)
            n += 1
        output[split] = dict(paths_recomputed=n, forward_matches_index_within_1meV=int(forward),
                             full_profile_range_matches_index_within_1meV=int(span),
                             max_forward_difference_meV=float(maximum) * 1000)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "neb", "sealed", "d0", "parents", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    protected = sealed_materials(args.sealed)
    with ZipFile(args.neb) as z:
        index = read_index(z, "nebDFT2k/nebDFT2k_index.csv")
        by_split = {name: {r["material_id"] for r in index if r["_split"] == name}
                    for name in SPLITS}
        if not protected <= by_split["test"] or any(protected & by_split[k] for k in SPLITS[:2]):
            raise ValueError("Sealed material IDs are outside the expected Test split")
        energy_summary = profile_counts(z, index, protected)
    with ZipFile(args.d0) as z:
        training = read_index(z, "training_frames.csv")[:278]
    exact_d0 = {r["material_id"] for r in training}
    edges_d0 = {r["edge_id"] for r in training}
    all_materials = set.union(*by_split.values())
    if not exact_d0 <= all_materials:
        raise ValueError("D0 lineage differs from the expected source generations")
    source, source_sha = read_parents(args.parents, all_materials)
    buckets = defaultdict(list)
    for key, (structure, _) in sorted(source.items()):
        host = framework(structure)
        buckets[anonymous_formula(host)].append((key, host))
    tasks = [("LiTraj", a, b, left, right) for bucket in buckets.values()
             for (a, left), (b, right) in combinations(bucket, 2)]
    print(f"Checking {len(tasks)} unique parent pairs with {args.workers} processes.", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        compared = list(executor.map(fit_pair, tasks, chunksize=12))
    links = [(r["public_id"], r["reference_key"]) for r in compared if r["anonymous_framework_fit"]]
    cluster = groups(all_materials, links)
    family_sets = {name: {cluster[key] for key in keys} for name, keys in by_split.items()}
    train_families = family_sets["train"]
    d0_families = {cluster[key] for key in exact_d0}
    fp = rows(args.results / "public_neb_reuse/fp_paths.csv")
    used_fp = {row["path_id"] for row in rows(args.results / "public_neb_panel/panel_paths.csv")}
    overlap_prior = json.loads((args.results / "public_neb_reuse/summary.json").read_bytes())
    if overlap_prior["parent_archive_sha256"] != source_sha or len(used_fp) != 62:
        raise ValueError("Prior FP overlap does not refer to these parent structures")
    fp_hits = sum(int(r["project_anonymous_hits_including_exact"]) > 0
                  for r in fp if r["path_id"] in used_fp)
    scope_summary = {}
    for name, ids in by_split.items():
        overlapping = {key for key in ids if cluster[key] in train_families}
        scope_summary[name] = dict(
            distinct_materials=len(ids), paths=sum(r["_split"] == name for r in index),
            structural_families=len(family_sets[name]), exact_D0_materials=len(ids & exact_d0),
            exact_D0_paths=sum(r["_split"] == name and r["edge_id"] in edges_d0 for r in index),
            materials_in_train_families=len(overlapping), reserved_materials=len(ids & protected),
            structural_families_shared_with_D0=len(family_sets[name] & d0_families),
            nonsealed_materials_in_D0_families=sum(cluster[key] in d0_families
                                                  for key in ids - protected),
        )
    summary = dict(
        source_DFT_archive_sha256=digest(args.neb), parent_archive_sha256=source_sha,
        d0_source_sha256=digest(args.d0), sealed_geometry_manifest_sha256=SEALED_SHA,
        matching=MATCHER, parent_pairs=len(compared), anonymous_match_pairs=len(links),
        split_summary=scope_summary,
        cross_split={f"{a}__{b}": dict(shared_materials=len(by_split[a] & by_split[b]),
                                     shared_structural_families=len(family_sets[a] & family_sets[b]))
                     for a, b in combinations(SPLITS, 2)},
        nonsealed_label_audit=energy_summary,
        fp_panel_structural_overlap_matches_from_prior_full_parent_audit=fp_hits,
        prior_project_registry_new_val_test_materials=0,
        prior_registry_source="experiments/phase1/absolute_gain_readiness.md",
        prior_registry_source_sha256=digest(args.results.parent / "experiments/phase1/absolute_gain_readiness.md"),
        new_primary_panel_released=0, reserved_dft_labels_read=0,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    with (args.output / "litraj_boundary.json").open("x", encoding="utf-8", newline="\n") as f:
        f.write(blob)
    print(blob, flush=True)


if __name__ == "__main__":
    main()
