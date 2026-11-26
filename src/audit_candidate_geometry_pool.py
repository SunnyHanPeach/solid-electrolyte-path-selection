"""Check periodic reconstruction of the frozen 16,155-path BVSE source pool."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from ase.geometry import find_mic
from ase.io import read

from run_phase_b4_source_pilot import archive_geometry, read_csv, reconstruct


FIELDS = ("edge_id", "material_id", "chemsys", "atoms", "initial_images", "hop_length_A",
          "minimum_interpolation_contact_A", "periodic_mapping_error_A", "contact_below_1_4_A",
          "more_than_180_atoms", "status", "error")


def inspect(work):
    row, parent, centroid = work
    result = {k: row[k] for k in ("edge_id", "material_id", "chemsys")}
    try:
        a, b = (read(StringIO(raw.decode()), format="extxyz") for raw in (parent, centroid))
        frames, mover, error = reconstruct(row["edge_id"], a, b)
        lo = np.arange(len(frames[0])) != mover
        pts = np.array([frame.positions[mover] for frame in frames])
        cell = frames[0].cell
        _, contacts = find_mic((frames[0].positions[lo][None] - pts[:, None]).reshape(-1, 3), cell, True)
        hop = float(np.linalg.norm(pts[-1] - pts[0]))
        step = np.diff(pts, axis=0)
        mapped, _ = find_mic(step, cell, True)
        if not np.allclose(mapped, step, atol=1e-5, rtol=0):
            raise ValueError("A neighboring interpolation step has a periodic alias")
        result.update(atoms=len(frames[0]), initial_images=len(frames), hop_length_A=hop,
                      minimum_interpolation_contact_A=float(contacts.min()), periodic_mapping_error_A=error,
                      contact_below_1_4_A=bool(contacts.min() < 1.4),
                      more_than_180_atoms=len(frames[0]) > 180, status="geometry_verified")
    except (TypeError, KeyError, ValueError, IndexError, RuntimeError, np.linalg.LinAlgError) as exc:
        result.update(status="mapping_issue", error=f"{type(exc).__name__}: {exc}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent_archive", "centroid_archive", "bvse_index", "history_index", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--deadline", type=datetime.fromisoformat, required=True)
    args = parser.parse_args()
    if args.deadline.tzinfo is None:
        parser.error("Deadline must include timezone")
    args.output.mkdir()
    start = perf_counter()
    history = read_csv(args.history_index)
    old_material = {r["material_id"] for r in history}
    old_system = {r["chemsys"] for r in history}
    pool = sorted((r for r in read_csv(args.bvse_index)
                   if r["material_id"] not in old_material and r["chemsys"] not in old_system),
                  key=lambda r: (r["material_id"], r["edge_id"]))
    mat = {r["material_id"] for r in pool}
    edges = {r["edge_id"] for r in pool}
    if len(pool) != len(edges) or len(edges) != 16155 or len(mat) != 2647:
        raise ValueError("Frozen B4 candidate pool has changed")
    with ThreadPoolExecutor(max_workers=2) as threads:
        parent = threads.submit(archive_geometry, args.parent_archive, "BVEL13k", "material_id", mat, mat)
        centroid = threads.submit(archive_geometry, args.centroid_archive, "nebBVSE122k", "edge_id", edges, edges)
        _, parents, _ = parent.result()
        _, centroids, _ = centroid.result()
    if set(parents) != mat or set(centroids) != edges:
        raise ValueError("A source geometry is missing")
    counts = Counter()
    risks = Counter()
    print(f"Checking {len(pool)} paths on {args.workers} CPU processes.", flush=True)
    with (args.output / "initial_geometry_audit.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as processes:
            for begin in range(0, len(pool), 120):
                if datetime.now(timezone.utc) >= args.deadline:
                    break
                batch = pool[begin:begin + 120]
                payload = ((r, parents[r["material_id"]][1], centroids[r["edge_id"]][1]) for r in batch)
                for result in processes.map(inspect, payload, chunksize=4):
                    writer.writerow(result)
                    counts[result["status"]] += 1
                    risks["short_contact"] += int(result.get("contact_below_1_4_A", False))
                    risks["large_supercell"] += int(result.get("more_than_180_atoms", False))
                f.flush()
                if begin % 1200 == 0 or begin+len(batch) == len(pool):
                    print(f"{sum(counts.values())}/{len(pool)} complete", flush=True)
    inputs = {p.name: sha256(p.read_bytes()).hexdigest()
              for p in (args.parent_archive, args.centroid_archive, args.bvse_index, args.history_index)}
    summary = dict(path_count=len(pool), material_count=len(mat), audited=sum(counts.values()),
                   status_counts=dict(counts), geometry_risks=dict(risks), workers=args.workers,
                   source_hashes=inputs, deadline=args.deadline.isoformat(), elapsed_seconds=perf_counter() - start,
                   geometry_only=True)
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
