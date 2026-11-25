""" Compare the seven completed Phase 1 paths on the same held-out materials. """
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from audit_archived_net_path import blind_panel
from audit_signed_barriers import file_sha, rows, sha
from freeze_net_path_panel import csv_bytes
from run_force_weighting import atomic_json
from validate_phase1_materials import bootstrap_weights, estimates, predict_checkpoints, read_barrier_errors

SEEDS = (17, 29, 43)
PILOT = "results"
TRANSFER_CASES = (
    ("transfer", "transfer_delta.zip"),
    ("next_block/new_anion", "next-block/new_anion.zip"),
    ("next_block/near_geometry", "next-block/near_geometry.zip"),
    ("next_block/feature_gap", "next-block/feature_gap.zip"),
)
INFLUENTIAL = "mp-1277814"


def freeze(args):
    with ZipFile(args.bundle) as archive:
        pilot = json.loads(archive.read("plan.json"))
        pilot_jobs = {job["seed"]: job["jobs"] for job in json.loads(archive.read("jobs.json"))}
    bundle = file_sha(args.bundle)
    _, panel = blind_panel(args.panel)
    roots = [(edge, PILOT, f"path{i}", f"replay{i}", pilot_jobs, None)
             for i, edge in enumerate(pilot["selected"], 1)]
    for remote, filename in TRANSFER_CASES:
        source = args.runs / filename
        with ZipFile(source) as archive:
            delta = json.loads(archive.read("plan.json"))
        if delta["base_bundle"] != bundle:
            raise ValueError(f"Unexpected training bundle: {source}")
        order = {int(seed): arms for seed, arms in delta["orders"].items()}
        roots.append((delta["selected"]["edge_id"], remote, "path", "replay", order, file_sha(source)))
    paths = sorted(item[0] for item in roots)
    if len(paths) != 7 or len(set(paths)) != len(paths):
        raise ValueError("Expected seven distinct archived paths")
    if {edge.split("_", 1)[0] for edge in paths} & {r["material_id"] for r in rows(panel["panel.csv"])}:
        raise ValueError("Training candidate overlaps evaluation materials")
    jobs, environments = {}, set()
    for edge, remote, path, replay, order, delta_sha in [
        *roots, ("", PILOT, "pure", None, pilot_jobs, None)
    ]:
        for arm, label in ((path, f"{edge}/path" if edge else "pure"), (replay, f"{edge}/replay")):
            if arm is None:
                continue
            for seed in SEEDS:
                schedule = order[seed][arm]
                step = len(schedule)
                relative = Path(remote) / f"seed{seed}" / arm
                local = (args.runs / relative.relative_to(PILOT) if remote == PILOT
                         else args.runs / "received_transfers" / relative)
                marker = json.loads((local / f"done_{step}.json").read_bytes())
                expected = (bundle, seed, arm, step, sha(json.dumps(schedule).encode()), delta_sha)
                actual = tuple(marker.get(k) for k in (
                    "bundle", "seed", "arm", "optimizer_steps", "schedule", "delta"))
                if expected != actual:
                    raise ValueError(f"Training identity mismatch: {local}")
                model = f"model_{step}.pt"
                jobs[f"{seed}/{label}"] = dict(path=(relative/model).as_posix(),
                                              sha256=marker["files"][model])
                environments.add(json.dumps(marker["environment"], sort_keys=True))
    if len(jobs) != 45 or len(environments) != 1:
        raise ValueError("Missing models or training environment mismatch")
    atomic_json(args.output, dict(
        bundle=bundle, panel=file_sha(args.panel), frame_manifest=sha(panel["frames.csv"]),
        candidates=paths, seeds=list(SEEDS), jobs=jobs,
        question="Do other completed Phase 1 paths improve the same 26-material panel?",
        scope="Secondary comparison on a panel already scored for mp-753143. Individual intervals are exploratory.",
        metric="Control minus path forward barrier MAE in meV. Report both matched replay and pure D0.",
        sensitivity=f"Repeat estimates omitting {INFLUENTIAL} for every path, and report every leave-one-system-out mean range.",
        decision="Use the results to identify replication candidates. A new positive claim needs another independent material panel.",
    ))


def inputs(args):
    lock = json.loads(args.lock.read_bytes())
    if (lock["bundle"], lock["panel"]) != (file_sha(args.bundle), file_sha(args.panel)):
        raise ValueError("Frozen input identity mismatch")
    return lock


def predict(args):
    lock = inputs(args)
    jobs = {k: (args.runs/v["path"], v["sha256"]) for k, v in lock["jobs"].items()}
    predict_checkpoints(args.bundle, args.panel, args.output, jobs)


def score(args):
    lock = inputs(args)
    _, panel = blind_panel(args.panel)
    with ZipFile(args.predictions) as archive:
        raw = archive.read("predictions.csv")
        proof = json.loads(archive.read("proof.json"))
    if (proof["bundle"], proof["panel"], proof["predictions"], proof["weights"]) != (
        lock["bundle"], lock["panel"], sha(raw), {k:v["sha256"] for k,v in lock["jobs"].items()}
    ):
        raise ValueError("Predictions differ from frozen inputs")
    material, errors = read_barrier_errors(args.project, panel, raw, lock["jobs"])
    edges = [r["edge_id"] for r in material]
    keep = np.asarray([r["material_id"] != INFLUENTIAL for r in material])
    if int((~keep).sum()) != 1:
        raise ValueError("Influence material must occur exactly once")
    systems, w = bootstrap_weights(material)
    _, w_without = bootstrap_weights([r for r,k in zip(material,keep) if k])
    chemsys = np.asarray([r["chemsys"] for r in material])
    outcomes, table = {}, []
    for path in lock["candidates"]:
        outcomes[path] = {}
        for metric, control in (("net", f"{path}/replay"), ("absolute", "pure")):
            delta = np.asarray([[errors[e,s,control]-errors[e,s,f"{path}/path"] for e in edges] for s in SEEDS])
            mean = delta.mean(axis=0)
            leave_system_out = [float(mean[chemsys != system].mean()) for system in systems]
            result = estimates(delta, w)
            result["without_influential_material"] = estimates(delta[:,keep], w_without)
            result["leave_one_system_out_mean_meV"] = [min(leave_system_out), max(leave_system_out)]
            outcomes[path][metric] = result
            for seed, values in zip(SEEDS,delta):
                table.extend(dict(r, candidate=path, seed=seed, metric=metric, gain_meV=float(x))
                             for r,x in zip(material,values))
    args.output.mkdir(exist_ok=True)
    atomic_json(args.output/"summary.json", dict(
        materials=len(edges), exact_chemistries=len(systems), lock_sha256=file_sha(args.lock),
        predictions=proof, results=outcomes))
    (args.output/"paired_materials.csv").write_bytes(csv_bytes(table))
    for path, metrics in outcomes.items():
        print(path, {m: (round(r["mean_meV"], 2), [round(q,2) for q in r["interval95_meV"]],
                           round(r["without_influential_material"]["mean_meV"],2))
                     for m,r in metrics.items()}, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(required=True)
    for function, names in (
        (freeze, ("bundle","panel","runs","output")),
        (predict, ("bundle","panel","lock","runs","output")),
        (score, ("project","bundle","panel","lock","predictions","output"))
    ):
        mode = sub.add_parser(function.__name__)
        for name in names:
            mode.add_argument(name, type=Path)
        mode.set_defaults(function=function)
    args = cli.parse_args()
    args.function(args)
