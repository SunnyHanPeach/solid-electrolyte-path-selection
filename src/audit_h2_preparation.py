"""Build the H2 source trace and replay the recorded H1 subset prefixes."""

import argparse
import csv
import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, records):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def trace(records, sources):
    tracked = []
    for row in records:
        if row["method_audit"] != "pbe_path_unverified":
            continue
        source = sources[row["source_key"]]
        tracked.append(dict(
            published_id=row["published_id"], system=row["system"], source_key=row["source_key"],
            original_source_url=source["original_source_url"], checked_asset_url=source["checked_asset_url"],
            source_checked=source["source_checked"],
            historical_framework_hits=int(row["external_anonymous_framework_matches"]),
            full_ordered_neb_verified=False, original_direction_verified=False,
            primary_released=False,
            next_action=source["next_action"],
        ))
    if len(tracked) != 18 or {r["source_key"] for r in tracked} != set(sources):
        raise ValueError("PBE cohort no longer matches the archived 18 entries and seven sources")
    return tracked


def prefix_overlap(rankings, candidates):
    rules = ("SOAP", "SOAP+Activation")
    pools = {"all_H1_candidates": set(candidates),
             "after_H1_structural_exclusions": {key for key, row in candidates.items()
                                                if not row["structural_exclusion"]}}
    order = defaultdict(list)
    for row in sorted(rankings, key=lambda row: int(row["rank"])):
        if row["aggregation"] == "mean" and row["method"] in rules:
            order[row["method"]].append(row["edge_id"])
    if any(len(order[method]) != len(candidates) or set(order[method]) != set(candidates) for method in rules):
        raise ValueError("B5-R Mean rankings are not a permutation of the H1 pool")
    output = []
    for scope, keys in pools.items():
        soap, combined = ([key for key in order[method] if key in keys] for method in rules)
        for k in range(1, len(keys) + 1):
            a, b = set(soap[:k]), set(combined[:k])
            output.append(dict(scope=scope, k=k, shared=len(a & b), equal_sets=a == b,
                               soap_only="|".join(sorted(a - b)), combined_only="|".join(sorted(b - a))))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    r, out = args.results, args.output
    inputs = {
        "records": r / "phase_h2_evaluation/record_audit.csv",
        "sources": out / "source_followup.csv",
        "rankings": r / "phase_b5_r/independent_candidate_ranking.csv",
        "candidates": r / "phase_h1/structure_output/candidate_ranking_and_audit.csv",
    }
    records, sources, rankings, candidates = (read(path) for path in inputs.values())
    traced = trace(records, {row["source_key"]: row for row in sources})
    prefixes = prefix_overlap(rankings, {row["edge_id"]: row for row in candidates})
    write(out / "pbe_trace.csv", traced)
    write(out / "h1_prefix_overlap.csv", prefixes)
    metadata = dict(
        pbe_rows=len(traced), sources=len(sources),
        without_prior_framework_hits=sum(row["historical_framework_hits"] == 0 for row in traced),
        newly_released=sum(row["primary_released"] for row in traced),
        top_k_replay="fixed ordering, equal item count, no budget or utility estimate",
        equal_prefix_k={scope: [r["k"] for r in prefixes if r["scope"] == scope and r["equal_sets"]]
                        for scope in {r["scope"] for r in prefixes}},
        input_sha256={name: sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()},
    )
    with (out / "audit.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
