"""Verify archived project evidence in parallel and assemble a reusable data index."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
from hashlib import sha256
from io import TextIOWrapper, BytesIO
import json
import math
from pathlib import Path
import re
from statistics import fmean
import subprocess
import tarfile
from time import perf_counter


def read_json(path):
    return json.loads(path.read_bytes())


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def write_csv(path, records, fields=()):
    fields = list(dict.fromkeys([*fields, *(k for r in records for k in r)]))
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(records)


def hash_file(path):
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def time_left(args):
    if datetime.now(timezone.utc) >= args.deadline:
        raise TimeoutError("Audit deadline reached")


def index_results(args):
    paths = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "results", "experiments", "docs"], cwd=args.repo
    ).decode("utf-8").split("\0")
    records = []
    for ref in sorted(filter(None, paths)):
        time_left(args)
        # The reserved labels are outside this audit. No need to read locked files or metadata.
        if any(part in ref.lower() for part in ("sealed", "locked", "test_labels", "holdout_label")):
            continue
        path = args.repo / ref
        scope = ref.split("/")
        kind = {".csv": "table", ".json": "metadata", ".jsonl": "metadata", ".md": "report",
                ".png": "figure", ".pdf": "figure", ".npz": "features", ".log": "log"}.get(path.suffix, "asset")
        records.append(dict(path=ref, area="/".join(scope[:2]), kind=kind, bytes=path.stat().st_size,
                            sha256=hash_file(path)))
    write_csv(args.output / "project_file_index.csv", records)
    return dict(tracked_files=len(records), types=dict(Counter(r["kind"] for r in records)))


def audit_training(args):
    b2 = args.repo / "results/b2_overnight_01"
    scored = b2 / "six_hour_results"
    b3 = args.repo / "results/phase_b3_a"
    expected = {(r["arm"], int(r["seed"])): r for r in read_csv(b2 / "six_hour_tasks.csv")}
    weight_hashes = {(r["arm"], int(r["seed"])): r for r in read_csv(scored / "checkpoint_hashes.csv")}
    fit_rows = []
    for (arm, seed), task in expected.items():
        time_left(args)
        relative = f"results/b2_overnight_01/six_hour_results/fits/{arm}/seed{seed}/done.json"
        path = args.repo / relative
        if not path.is_file():
            fit_rows.append(dict(arm=arm, seed=seed, path=relative, status="missing_done_record"))
            continue
        done = read_json(path)
        h = weight_hashes[(arm, seed)]["checkpoint_sha256"]
        t = done.get("files", {})
        training = path.with_name("training.csv")
        valid = (done["seed"] == seed and done["arm"] == arm
                 and done["steps"] == int(task["steps"]) == done["frozen_steps"]
                 and done["schedule_sha256"] == task["order_sha256"]
                 and t.get("model.pt") == h
                 and training.is_file() and hash_file(training) == t.get("training.csv"))
        fit_rows.append(dict(arm=arm, seed=seed, path=relative, edge_id=task["edge_id"], checkpoint_sha256=h,
                             steps=done["steps"], status="metadata_verified" if valid else "identity_or_log_mismatch"))
    write_csv(args.output / "training_fit_audit.csv", fit_rows)

    data = read_csv(b3 / "paired_material_gains.csv")
    regular = defaultdict(list)
    seen = set()
    max_deviation = 0.0
    for r in data:
        time_left(args)
        key = (r["analysis"], r["candidate_edge"], r["seed"], r["material_id"])
        if key in seen:
            raise ValueError(f"Repeated analysis/pair/material key: {key}")
        seen.add(key)
        target = float(r["dft_forward_barrier_meV"])
        pure, path, replay = (abs(float(r[k + "_prediction_meV"]) - target) for k in ("pure", "path", "replay"))
        absolute, net = pure - path, replay - path
        max_deviation = max(max_deviation, abs(absolute - float(r["absolute_gain_meV"])),
                            abs(net - float(r["net_gain_meV"])))
        if r["analysis"] == "regular":
            regular[r["candidate_edge"]].append((int(r["seed"]), r["material_id"], absolute, net))
    labels = read_csv(b3 / "path_utility_labels.csv")
    utility = []
    for label in labels:
        observed = regular[label["edge_id"]]
        record = dict(label)
        count = Counter(seed for seed, _, _, _ in observed)
        panels = {seed: {r[1] for r in observed if r[0] == seed} for seed in count}
        expected_panel = int(label["materials"])
        abs_gain, net_gain = (fmean(r[i] for r in observed) for i in (2, 3))
        abs_deviation = abs(abs_gain - float(label["Delta_abs"]))
        net_deviation = abs(net_gain - float(label["Delta_net"]))
        correct = count == {17: expected_panel, 29: expected_panel, 43: expected_panel}
        correct = correct and panels[17] == panels[29] == panels[43]
        correct = correct and max(abs_deviation, net_deviation, max_deviation) < 1e-7
        record.update(audit_status="verified" if correct else "mismatch", paired_rows=len(observed),
                      recomputed_abs_meV=abs_gain, recomputed_net_meV=net_gain,
                      maximum_abs_gain_deviation_meV=abs_deviation,
                      maximum_net_gain_deviation_meV=net_deviation)
        utility.append(record)
    write_csv(args.output / "path_utility_audit.csv", utility)
    return dict(fits=len(fit_rows), fit_metadata_verified=sum(r["status"] == "metadata_verified" for r in fit_rows),
                paired_rows_all_analyses=len(data), regular_rows=sum(map(len, regular.values())),
                outcomes_verified=sum(r["audit_status"] == "verified" for r in utility),
                max_formula_deviation_meV=max_deviation, abs_classes=dict(Counter(r["Delta_abs_class"] for r in utility)),
                net_classes=dict(Counter(r["Delta_net_class"] for r in utility)))


def member_hash(f):
    h = sha256()
    for block in iter(lambda: f.read(2**20), b""):
        h.update(block)
    return h.hexdigest()


def audit_archives(args):
    results = args.repo / "results"
    b2 = results / "b2_overnight_01/six_hour_results"
    checkpoint = read_json(results / "protected_trajectory/checkpoint_verification.json")
    expected = {f"results/seed{r['seed']}/{r['arm']}/step{r['step']:04d}/checkpoint.pt": r["sha256"]
                for r in checkpoint["records"]}
    b2_verification = read_json(b2 / "verification.json")
    predictions = {"predictions/" + r["prediction_file"]: r["prediction_sha256"]
                   for r in read_csv(b2 / "checkpoint_hashes.csv")}
    jobs = [
        ("trajectory_24", args.assets / "trajectory_replay/trajectory_checkpoints.tar.xz", checkpoint["sha256"], expected),
        ("B2_predictions", args.assets / "b2-overnight-01/compact.tar.gz", b2_verification["compact_sha256"], predictions),
    ]
    out, bits = [], []
    for name, archive, wanted, members in jobs:
        time_left(args)
        digest = hash_file(archive)
        checked = {}
        n_model_files = 0
        with tarfile.open(archive, mode="r|*") as tar:
            for item in tar:
                time_left(args)
                key = item.name.removeprefix("./")
                if key.endswith(("/checkpoint.pt", "/model.pt", "/weights.pt")) and item.isfile():
                    n_model_files += 1
                if key in members:
                    with tar.extractfile(item) as f:
                        checked[key] = member_hash(f)
        for key, h in members.items():
            bits.append(dict(archive=name, member=key, expected_sha256=h, found_sha256=checked.get(key, ""),
                             status="verified" if checked.get(key) == h else "missing_or_changed"))
        out.append(dict(asset=name, local_subpath=archive.relative_to(args.assets).as_posix(), bytes=archive.stat().st_size,
                        archive_sha256=digest, archive_hash_verified=digest == wanted, member_count=len(members),
                        verified_members=sum(checked.get(k) == h for k, h in members.items()), raw_weight_members=n_model_files))
    write_csv(args.output / "archive_member_audit.csv", bits)
    write_csv(args.output / "archive_inventory.csv", out)
    return {r["asset"]: r for r in out}


def cost_events(args):
    r = args.repo / "results"
    out = []

    def add(path, phase, material, role, hardware, seconds, status, grid="", cutoff="", gpu_count=1, rank_seconds=None):
        time_left(args)
        out.append(dict(material=material, phase=phase, role=role, hardware=hardware, grid=grid, cutoff_Ry=cutoff,
                        solver_wall_s=float(seconds), gpu_occupied_s=float(seconds) * gpu_count if gpu_count else "",
                        assigned_rank_s=rank_seconds, outcome=status, source=path))

    k = "phase_b4_bridge01/bridge_scores.json"
    for row in read_json(r / k)["images"]:
        if row["solver_status"] == "completed":
            add(k, "fixed_geometry_bridge", "mp-35822", row["image"], "RTX 3080 Ti", row["gpu_seconds"], "complete", "Gamma", "50/400")
    k = "phase_b4_bridge01/numeric15_scores.json"
    for row in read_json(r / k)["runs"]:
        add(k, "bridge_" + row["variant"], "mp-35822", row["image"], "RTX 3080 Ti", row["gpu_wall_seconds"], "complete",
            "Gamma", "62.5/500" if row["variant"] == "cutoff" else "50/400")
    k = "phase_b4_spin_pilot/scores.json"
    for row in read_json(r / k)["jobs"]:
        add(k, "spin_" + row["spin"], "mp-1191903", row["role"], "RTX 3080 Ti", row["elapsed_seconds"], "complete", "Gamma", "50/400")
    k = "phase_b4_numeric_grid_recovered/scores.json"
    grid = read_json(r / k)["comparisons"]["k222"]
    for row in grid["jobs"]:
        add(k, "mesh", "mp-1191903", row["role"], "RTX 3090", row["elapsed_seconds"],
            "SCF_and_forces_verified_with_partial_wavefunction_output" if row.get("wavefunction_save_incomplete") else "complete",
            "2x2x2", "50/400")
    for row in grid["incomplete_attempts"]:
        if row["failure_stage"] == "USPP force (addusforce_g)":
            add(k, "mesh", "mp-1191903", "endpoint", "RTX 3080 Ti", row["elapsed_seconds"], "GPU_OOM", "2x2x2", "50/400")
    for stage, grid_name in (("io_k222", "2x2x2"), ("grid_pair", "3x3x3")):
        k = f"phase_b4_r2/{stage}/scores.json"
        for comp, row in read_json(r / k)["comparisons"].items():
            add(k, stage, "mp-1191903", comp.split(":")[0], "RTX 3090", row["test"]["elapsed_seconds"], "complete",
                grid_name, "50/400")
    k = "phase_b4_bridge01/run_manifest.json"
    for row in read_json(r / k)["accounting"]["stopped_local_cpu_runs"]:
        add(k, "bridge_cpu", "mp-35822", row["image"], "CPU", row["wall_seconds"], "stopped",
            "Gamma", "50/400", gpu_count=0, rank_seconds=row["allocated_rank_seconds"])
    for filename, state in (("cost.json", "CI_partial"), ("initial_launch_cost.json", "CI_mpi_preflight_exit")):
        k = f"dft/mp5012_ci/{filename}"
        row = read_json(r / k)
        add(k, "CI_pilot", "mp-5012", "band", "CPU", row["wall_seconds"], state,
            gpu_count=0, rank_seconds=row["allocated_mpi_rank_seconds"])
    write_csv(args.output / "dft_cost_ledger.csv", out)
    return dict(events=len(out), gpu_solver_seconds=sum(r["gpu_occupied_s"] for r in out if r["gpu_occupied_s"] != ""),
                assigned_cpu_rank_seconds=sum(r["assigned_rank_s"] or 0 for r in out),
                estimated_costs_filled=0, cost_scope="observed events in cited sources, not billing or complete programme totals")


def compile_candidates(args):
    time_left(args)
    r = args.repo / "results"
    audit = read_json(args.evaluation / "manifest.json")
    rows = read_csv(r / "phase_b4_r2_preflight/h1c_cost_inputs.csv")
    if len(rows) != 24:
        raise ValueError("Expected 24 H1 candidate paths")
    for row in rows:
        row["has_assets_for_previous_upf_tests"] = not bool(row["elements_without_previously_benchmarked_upf"])
        row["evaluation_paths_released"] = audit["primary_released"]
        row["readiness"] = "needs_pseudo_audit_and_eval_panel" if row["elements_without_previously_benchmarked_upf"] else "needs_protocol_and_eval_panel"
    write_csv(args.output / "candidate_readiness.csv", rows)
    return dict(candidates=len(rows), h1_review=sum(bool(r["h1_panel_role"]) for r in rows),
                all_element_precedent=sum(r["has_assets_for_previous_upf_tests"] for r in rows),
                evaluation_primary_released=audit["primary_released"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "assets", "evaluation", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--deadline", required=True, type=datetime.fromisoformat)
    args = p.parse_args()
    if args.deadline.tzinfo is None:
        p.error("Use a timezone-aware deadline")
    args.output.mkdir()
    tasks = (index_results, audit_training, audit_archives, cost_events, compile_candidates)
    summary = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {task.__name__: pool.submit(task, args) for task in tasks}
        for name, job in futures.items():
            try:
                summary[name] = dict(status="complete", **job.result())
            except Exception as exc:
                summary[name] = dict(status="failed", detail=f"{type(exc).__name__}: {exc}")
            print(name, summary[name]["status"], flush=True)
    summary["audit"] = dict(deadline=args.deadline.isoformat(), finished=datetime.now().astimezone().isoformat(),
                            input_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip(),
                            workers=len(tasks), reserved_evaluation_label_files_opened=0)
    write_json(args.output / "audit_summary.json", summary)
    if any(summary[task.__name__]["status"] != "complete" for task in tasks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
