"""Run the audited CPU BVSE preconditioner on a deterministic candidate queue."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter, time

from run_phase_b4_source_pilot import archive_geometry, pilot, read_csv


def prioritized(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["status"] == "geometry_verified" and int(row["atoms"]) <= 180:
            grouped[row["material_id"], row["contact_below_1_4_A"]].append(row)
    ranked = []
    for values in grouped.values():
        ordered = sorted(values, key=lambda row: order(row["edge_id"]))
        ranked.extend((i, order(row["edge_id"]), row) for i, row in enumerate(ordered))
    return [row for _, _, row in sorted(ranked)]


def order(edge):
    return hashlib.sha256(f"geometry-audit-20261127:{edge}".encode()).hexdigest()


def run_path(args):
    row, parent, centroid, output, deadline = args
    if time() >= deadline:
        return {"edge_id": row["edge_id"], "status": "deferred"}
    try:
        return pilot(row, parent, centroid, output)
    except (ValueError, KeyError, RuntimeError, TypeError, IndexError, ArithmeticError) as exc:
        return {"edge_id": row["edge_id"], "status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def classify(item):
    if "final" not in item:
        return None, None, None
    final_min = min(row["min_Li_contact_A"] for row in item["final"])
    energies = [row["bvse_energy_eV"] for row in item["final"]]
    prominence = max(energies[1:-1]) - max(energies[0], energies[-1])
    return final_min, prominence, item["converged"] and final_min >= 1.4 and prominence >= .02


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("geometry_audit", "parents", "centroids", "scratch"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--exclude-results", type=Path, action="append", default=[])
    args = parser.parse_args()
    start = perf_counter()
    rows = prioritized(read_csv(args.geometry_audit))
    if args.limit is not None:
        rows = rows[:args.limit]
    already_completed = {
        row["edge_id"]
        for table in args.exclude_results
        for row in read_csv(table)
        if row["edge_id"] and row["status"] in {"converged", "step_limit", "failed"}
        and (row["seconds"] or row["error"])
    }
    rows = [row for row in rows if row["edge_id"] not in already_completed]
    geometry = args.scratch / "preconditioned"
    geometry.mkdir(parents=True, exist_ok=False)
    material_ids = {row["material_id"] for row in rows}
    edge_ids = {row["edge_id"] for row in rows}
    with ThreadPoolExecutor(max_workers=2) as scans:
        parents = scans.submit(archive_geometry, args.parents, "BVEL13k", "material_id", material_ids, material_ids)
        centroids = scans.submit(archive_geometry, args.centroids, "nebBVSE122k", "edge_id", edge_ids, edge_ids)
        _, parents, _ = parents.result()
        _, centroids, _ = centroids.result()
    metadata = {
        "eligible_paths": len(rows), "materials": len(material_ids), "workers": args.workers,
        "previously_completed_paths": len(already_completed),
        "start_deadline_epoch": args.deadline, "priority": "one path per material and initial-contact stratum before repeats, then SHA256",
        "input_geometry_audit_sha256": hashlib.sha256(args.geometry_audit.read_bytes()).hexdigest(),
        "versions": {name: version(name) for name in ("ase", "ions", "pymatgen", "numpy", "scipy", "spglib")},
    }
    (args.scratch / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    stats = defaultdict(Counter)
    fields = ("edge_id", "material_id", "chemsys", "atoms", "initial_contact_A", "final_contact_A",
              "peak_above_endpoints_eV", "converged", "geometry_screen_pass", "status", "seconds", "error")
    with (args.scratch / "per_path.csv").open("w", encoding="utf-8", newline="") as stream, \
            (args.scratch / "profiles.jsonl").open("w", encoding="utf-8", newline="\n") as profiles, \
            ProcessPoolExecutor(max_workers=args.workers) as pool:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        batch_size = args.workers * 4
        for offset in range(0, len(rows), batch_size):
            if time() >= args.deadline:
                break
            batch = rows[offset:offset + batch_size]
            tasks = [(row, parents[row["material_id"]], centroids[row["edge_id"]], geometry, args.deadline)
                     for row in batch]
            for row, item in zip(batch, pool.map(run_path, tasks)):
                if item["status"] == "deferred":
                    continue
                contact, prominence, passed = classify(item)
                writer.writerow({
                    **{k: row[k] for k in fields[:4]},
                    "initial_contact_A": row["minimum_interpolation_contact_A"], "final_contact_A": contact,
                    "peak_above_endpoints_eV": prominence, "converged": item.get("converged"),
                    "geometry_screen_pass": passed, "status": item["status"], "seconds": item.get("seconds"),
                    "error": item.get("error"),
                })
                profiles.write(json.dumps(item) + "\n")
                group = stats[row["contact_below_1_4_A"]]
                group["evaluated"] += 1
                group[item["status"]] += 1
                group["geometry_screen_pass"] += bool(passed)
            stream.flush()
            profiles.flush()
            print(json.dumps(dict(attempted=sum(group["evaluated"] for group in stats.values()),
                                  elapsed_seconds=round(perf_counter() - start, 1), strata=stats)), flush=True)
    (args.scratch / "summary.json").write_text(
        json.dumps({**metadata, "elapsed_seconds": perf_counter() - start, "strata": stats}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
