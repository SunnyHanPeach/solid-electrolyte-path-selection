"""Screen recovered candidate parents for periodic framework overlap with history."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from pymatgen.analysis.structure_matcher import StructureMatcher

from audit_phase_h2_evaluation import MATCHER, framework
from phase_h1_shortlist import anonymous_formula, read, read_parents


REFERENCE_GROUPS = {}


def initialize(groups):
    global REFERENCE_GROUPS
    REFERENCE_GROUPS = groups


def screen(job):
    uid, chemistry, n_paths, parent = job
    examiner = StructureMatcher(**MATCHER)
    result = dict(material_id=uid, chemsys=chemistry, geometry_verified_paths=n_paths)
    for tag, geom in (("parent", parent), ("framework", framework(parent))):
        eligible = REFERENCE_GROUPS[tag].get(anonymous_formula(geom), ())
        match = ""
        compared = 0
        try:
            for key, candidate in eligible:
                compared += 1
                if examiner.fit_anonymous(geom, candidate):
                    match = key
                    break
            status = "match" if match else "none_detected" if eligible else "no_stoichiometry_comparison"
        except (ValueError, TypeError, RuntimeError) as exc:
            status = "matcher_error"
            result[f"{tag}_detail"] = f"{type(exc).__name__}: {exc}"
        result.update({
            f"{tag}_candidate_references": len(eligible), f"{tag}_tested": compared,
            f"{tag}_status": status, f"{tag}_first_match_hash": match,
        })
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "history", "geometry_audit", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--deadline", required=True, type=datetime.fromisoformat)
    args = p.parse_args()
    if args.deadline.tzinfo is None:
        p.error("Deadline must include timezone")
    args.output.mkdir()
    started = perf_counter()
    history = {r["material_id"] for r in read(args.history)}
    geometry = read(args.geometry_audit)
    candidates = {r["material_id"]: r["chemsys"] for r in geometry}
    verified = Counter(r["material_id"] for r in geometry if r["status"] == "geometry_verified")
    if len(history) != 867 or len(candidates) != 2647 or len(geometry) != 16155 or set(candidates) & history:
        raise ValueError("Candidate or historical identity differs from the completed audit")
    parents, digest = read_parents(args.archive, history | set(candidates))
    groups = {"parent": defaultdict(list), "framework": defaultdict(list)}
    for mid in sorted(history):
        parent = parents[mid][0]
        key = sha256(mid.encode()).hexdigest()[:16]
        for tag, geom in (("parent", parent), ("framework", framework(parent))):
            groups[tag][anonymous_formula(geom)].append((key, geom))
    jobs = [(mid, chemistry, verified[mid], parents[mid][0]) for mid, chemistry in sorted(candidates.items())]
    fields = ["material_id", "chemsys", "geometry_verified_paths"]
    fields.extend(f"{tag}_{name}" for tag in ("parent", "framework")
                  for name in ("candidate_references", "tested", "status", "first_match_hash", "detail"))
    totals = Counter()
    comparisons = Counter()
    print(f"Comparing {len(jobs)} candidate parents against {len(history)} historical parents with {args.workers} CPU processes.",
          flush=True)
    with (args.output / "candidate_parent_overlap.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(groups,)) as workers:
            for start in range(0, len(jobs), 48):
                if datetime.now(timezone.utc) >= args.deadline:
                    break
                for row in workers.map(screen, jobs[start:start+48], chunksize=1):
                    writer.writerow(row)
                    totals["audited_materials"] += 1
                    for tag in ("parent", "framework"):
                        totals[f"{tag}_{row[f'{tag}_status']}"] += 1
                        comparisons[tag] += row[f"{tag}_tested"]
                f.flush()
                if start % 240 == 0 or start + 48 >= len(jobs):
                    print(f"{totals['audited_materials']}/{len(jobs)} candidate parents audited", flush=True)
    summary = dict(records=dict(totals), actual_periodic_comparisons=dict(comparisons),
                   matcher=MATCHER, parent_archive_sha256=digest, workers=args.workers,
                   deadline=args.deadline.isoformat(), elapsed_seconds=perf_counter() - started)
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
