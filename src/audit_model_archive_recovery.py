"""Locate B2 checkpoints by binary SHA256 across existing local snapshots."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import tarfile
from zipfile import BadZipFile, ZipFile

from audit_project_data import hash_file, member_hash, read_csv, write_csv, write_json


def is_checkpoint(name):
    return Path(name).suffix.lower() in (".pt", ".pth", ".model", ".safetensors")


def scan(path, root, wanted, deadline):
    checked = 0
    found = []

    def consume(name, stream):
        nonlocal checked
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError("Local archive audit deadline")
        digest = member_hash(stream)
        checked += 1
        if digest in wanted:
            found.append(dict(sha256=digest, local_source=path.relative_to(root).as_posix(), archive_member=name))

    try:
        if path.suffix.lower() == ".zip":
            with ZipFile(path) as archive:
                for entry in archive.infolist():
                    if not entry.is_dir() and is_checkpoint(entry.filename):
                        with archive.open(entry) as f:
                            consume(entry.filename, f)
        elif is_checkpoint(path.name):
            with path.open("rb") as f:
                consume("", f)
        else:
            with tarfile.open(path, "r|*") as archive:
                for entry in archive:
                    if datetime.now(timezone.utc) >= deadline:
                        raise TimeoutError("Local archive audit deadline")
                    if entry.isfile() and is_checkpoint(entry.name):
                        with archive.extractfile(entry) as f:
                            consume(entry.name, f)
        state = "checked"
        error = ""
    except (OSError, RuntimeError, tarfile.TarError, BadZipFile, EOFError) as exc:
        state = "partial_or_unreadable"
        error = f"{type(exc).__name__}: {exc}"
    return dict(local_source=path.relative_to(root).as_posix(), checkpoint_members_hashed=checked,
                matching_members=len(found), state=state, detail=error), found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "b2_checkpoint_hashes", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--deadline", type=datetime.fromisoformat, required=True)
    args = parser.parse_args()
    args.output.mkdir()
    expected = read_csv(args.b2_checkpoint_hashes)
    wanted = {r["checkpoint_sha256"] for r in expected}
    if len(expected) != 93:
        raise ValueError("Expected the frozen 93 B2 prediction identities")

    # Restrict the search to curated project snapshots and known training outputs.
    paths = [p for p in args.assets.glob("MACE_*")
             if p.is_file() and p.name.endswith((".zip", ".tar", ".tar.gz", ".tar.xz"))]
    for folder in ("phase1-p0", "b5-r-work", "phase-b0-verification", "b2-overnight-01", "b3-a-label-check"):
        root = args.assets / folder
        if root.is_dir():
            paths.extend(p for p in root.rglob("*") if p.is_file()
                         and (is_checkpoint(p.name) or p.name in ("compact.tar.gz", "forces.zip")))
    paths = sorted(set(paths))
    print(f"Checking {len(paths)} curated local snapshots and weight files against {len(wanted)} B2 hashes.", flush=True)
    inventory, matches = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for meta, rows in pool.map(lambda p: scan(p, args.assets, wanted, args.deadline), paths):
            inventory.append(meta)
            matches.extend(rows)
    by_hash = defaultdict(list)
    for r in matches:
        by_hash[r["sha256"]].append(r)
    outcomes = []
    for r in expected:
        sources = by_hash.get(r["checkpoint_sha256"], ())
        outcomes.append(dict(arm=r["arm"], seed=r["seed"], checkpoint_sha256=r["checkpoint_sha256"],
                             verified_local_copy_count=len(sources), local_recovery_status="hash_matched" if sources else "not_found_in_scanned_local_assets"))
    write_csv(args.output / "checkpoint_recovery.csv", outcomes)
    write_csv(args.output / "checkpoint_locations.csv", matches, ("sha256", "local_source", "archive_member"))
    write_csv(args.output / "checkpoint_archive_scan.csv", inventory)
    summary = dict(prediction_identities=len(expected), unique_checkpoint_hashes=len(wanted),
                   archive_or_weight_files_scanned=len(inventory), readable=sum(r["state"] == "checked" for r in inventory),
                   checkpoint_members_hashed=sum(r["checkpoint_members_hashed"] for r in inventory),
                   candidate_arms_and_seeds_found=sum(r["verified_local_copy_count"] > 0 for r in outcomes),
                   unique_expected_hashes_found=len(by_hash), source_ledger_sha256=hash_file(args.b2_checkpoint_hashes))
    write_json(args.output / "checkpoint_recovery_summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
