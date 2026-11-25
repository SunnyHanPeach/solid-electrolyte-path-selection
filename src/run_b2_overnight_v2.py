"""Extend the running B2 queue, preserving completed fits and the original time window."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from audit_signed_barriers import file_sha, rows, sha
from run_force_weighting import atomic_json, save_csv
import run_b2_overnight_01 as v1

MAIN_SEEDS = (17, 29, 43)
EXTRA_SEEDS = (53, 67)


def instrumented_fit(*args, **kwargs):
    from mace.modules import loss

    energy_fn = loss.weighted_mean_squared_error_energy
    force_fn = loss.mean_squared_error_forces
    energy, force = [], []

    def remember(func, trace):
        def wrapped(*a, **k):
            value = func(*a, **k)
            trace.append(float(value.detach()))
            return value
        return wrapped

    config = args[4]
    with (patch.object(loss, "weighted_mean_squared_error_energy", remember(energy_fn, energy)),
          patch.object(loss, "mean_squared_error_forces", remember(force_fn, force))):
        result = v1_fit(*args, **kwargs)
    trace = result[1]
    if len(energy) != len(trace) or len(force) != len(trace):
        raise ValueError("Loss components do not match the optimizer trace")
    for row, e, f in zip(trace, energy, force):
        w_e, w_f = e * config["energy_weight"], f * config["forces_weight"]
        if not math.isclose(row["loss"], w_e + w_f, rel_tol=2e-7, abs_tol=1e-8):
            raise ValueError("Logged components disagree with frozen loss")
        row.update(energy_loss_unweighted=e, force_loss_unweighted=f,
                   energy_loss_weighted=w_e, force_loss_weighted=w_f,
                   gradient_clipped=int(row["gradient_norm"] > config["clip_norm"]))
    return result


v1_fit = v1.fit


def pure_weights_for_revision(args, extras, jobs, plan):
    paths = {}
    for seed, record in json.loads(extras["baselines.json"]).items():
        root = args.baselines if int(seed) in MAIN_SEEDS else args.confirm_baselines
        run = root/f"seed{seed}/pure"
        target = run/"model_710.pt"
        expected = record["files"]["model_710.pt"]
        if (json.loads((run/"done_710.json").read_bytes()) != record
                or record["schedule"] != sha(json.dumps(jobs[seed]["pure"]).encode())
                or record["optimizer_steps"] != 710 or file_sha(target) != expected):
            raise ValueError(f"Invalid archived pure D0: seed{seed}")
        paths[seed] = (target, expected, record)
    return paths


def checked_reused_run(folder, identity):
    saved = json.loads((folder/"done.json").read_bytes())
    if saved["matrix_sha256"] != identity["matrix_sha256"]:
        wanted = saved["matrix_sha256"]
        legacy = json.loads(os.environ["B2_COMPATIBLE_MATRICES"])
        slot = int(saved["arm"].removeprefix("path").removeprefix("replay"))
        if slot > legacy.get(wanted, 0):
            raise ValueError(f"Unknown matrix provenance: {folder}")
        identity = dict(identity, matrix_sha256=wanted)
    return v1_verify(folder, identity)


v1_verify = v1.verify_run


def freeze_previous_dispatcher(args):
    """Hand over after the already-started pair exits. Never resubmit its finished jobs."""
    pid = int(args.old_pid.read_text().strip())
    process = Path(f"/proc/{pid}")
    if not process.exists():
        return
    if (process/"stat").read_text().split(") ", 1)[1][0] in ("Z", "X"):
        return
    cmd = (process/"cmdline").read_bytes()
    earlier = (b"run_b2_overnight_01.py\x00launch\x00",
               b"run_b2_overnight_v2_fixed.py\x00launch\x00")
    if not any(name in cmd for name in earlier):
        raise ValueError("Earlier coordinator identity does not match")
    os.kill(pid, signal.SIGSTOP)
    child_list = process/f"task/{pid}/children"
    while process.exists():
        active = []
        for child in child_list.read_text().split():
            target = Path(f"/proc/{child}/stat")
            if target.exists() and target.read_text().split(") ", 1)[1][0] not in ("Z", "X"):
                active.append(child)
        if not active:
            os.kill(pid, signal.SIGKILL)
            return
        time.sleep(3)


def child(args, name, seed, gpu, smoke=False):
    output = args.output/"instrumentation_smoke" if smoke else args.shared_fits
    output.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, "-u", str(Path(__file__).resolve()), "fit", str(args.base),
            str(args.matrix), str(args.baselines), str(output), "--arm", name, "--seed", str(seed)]
    if smoke:
        argv.append("--smoke")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               PYTHONDONTWRITEBYTECODE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    with (output/f"{name}_seed{seed}.log").open("ab") as f:
        return subprocess.Popen(argv, env=env, stdout=f, stderr=subprocess.STDOUT)


def existing(args, plan, jobs, arm, seed):
    path = args.shared_fits/arm/f"seed{seed}"
    if not (path/"done.json").is_file():
        return None
    return checked_reused_run(path, v1.task_identity(seed, arm, args, plan, jobs[str(seed)][arm]))


def missing_for(args, plan, jobs, slot, seeds):
    return [(arm, seed) for seed in seeds for arm in (f"path{slot}", f"replay{slot}")
            if existing(args, plan, jobs, arm, seed) is None]


def run_pairs(args, pending, visible, cutoff):
    for start in range(0, len(pending), 2):
        group = pending[start:start+2]
        processes = [child(args, a, s, i % visible) for i,(a,s) in enumerate(group)]
        try:
            codes = [p.wait(timeout=max(1., cutoff - time.time())) for p in processes]
        except subprocess.TimeoutExpired:
            for p in processes:
                if p.poll() is None:
                    p.terminate()
            for p in processes:
                p.wait(timeout=30)
            return False
        if any(codes):
            failed = [pair for pair, code in zip(group, codes) if code]
            retries = [child(args, a, s, i % visible) for i,(a,s) in enumerate(failed)]
            if any(p.wait() for p in retries):
                raise RuntimeError(f"Revised B2 failed again: {failed}")
    return True


def summarize_metrics(args, model_records, panel, completed):
    _, _, extras, _, _ = v1.assets(args)
    reference = rows(extras["reference_barriers.csv"])
    truth = {r["edge_id"]: r for r in reference}
    all_runs = {(r["arm"], r["seed"]): r for r in model_records}
    errors = {}
    barriers = {}
    for key, r in all_runs.items():
        raw = (args.output/"predictions"/r["prediction_file"]).read_bytes()
        if sha(raw) != r["prediction_sha256"]:
            raise ValueError("Inference output changed")
        curves = defaultdict(dict)
        for a in rows(raw):
            edge, frame = a["edge_id"], int(a["frame"])
            if a["material_id"] != truth[edge]["material_id"] or frame in curves[edge]:
                raise ValueError("Duplicate or mismatched evaluation key")
            curves[edge][frame] = float(a["energy_eV"])
        if curves.keys() != truth.keys():
            raise ValueError("Evaluation paths changed")
        for edge, e in curves.items():
            if set(e) != set(range(1, int(truth[edge]["images"])+1)) or not all(map(math.isfinite, e.values())):
                raise ValueError("Missing or invalid evaluation images")
            b = 1000 * (max(e.values())-e[1])
            barriers[key + (edge,)] = b
            errors[key + (edge,)] = abs(b - float(truth[edge]["dft_forward_barrier_meV"]))
        r["validation_barrier_mae_meV"] = sum(errors[key+(q,)] for q in truth) / len(truth)
    chemistry = sorted({v["chemsys"] for v in reference})
    n = len(chemistry)
    sampling = np.random.default_rng(20261126).multinomial(n,np.full(n,1/n),size=10000)
    weights = sampling[:, [chemistry.index(r["chemsys"]) for r in reference]]
    weights = weights / weights.sum(axis=1, keepdims=True)
    scores, materials = [], []
    for slot, seeds, label in completed:
        candidate = panel["candidates"][slot-1]
        ids = ("pure", f"path{slot}", f"replay{slot}")
        es = [np.array([[errors[a,s,r["edge_id"]] for r in reference] for s in seeds]) for a in ids]
        for i, seed in enumerate(seeds):
            for j, ref in enumerate(reference):
                edge = ref["edge_id"]
                materials.append(dict(analysis=label, slot=slot, candidate_edge=candidate["edge_id"], seed=seed,
                    material_id=ref["material_id"], evaluation_edge=edge, chemsys=ref["chemsys"],
                    dft_forward_barrier_meV=ref["dft_forward_barrier_meV"],
                    pure_prediction_meV=barriers["pure",seed,edge],
                    path_prediction_meV=barriers[ids[1],seed,edge], replay_prediction_meV=barriers[ids[2],seed,edge],
                    pure_error_meV=float(es[0][i,j]), path_error_meV=float(es[1][i,j]),
                    replay_error_meV=float(es[2][i,j]), absolute_gain_meV=float(es[0][i,j]-es[1][i,j]),
                    net_gain_meV=float(es[2][i,j]-es[1][i,j])))
        for name, control in (("absolute",0),("net",2)):
            g = es[control] - es[1]
            seed_means = g.mean(axis=1)
            low, high = np.quantile(weights @ g.mean(axis=0), [.025,.975])
            margin = .25
            state = ("positive" if low > margin and seed_means.min() > 0 else
                     "negative" if high < -margin and seed_means.max() < 0 else
                     "near_zero" if low >= -margin and high <= margin else "unresolved")
            scores.append(dict(analysis=label, slot=slot, candidate_edge=candidate["edge_id"], metric=name,
                seeds=";".join(map(str,seeds)), seed_gains_meV=";".join(map(str,seed_means)),
                mean_meV=float(g.mean()), low95_meV=float(low), high95_meV=float(high), classification=state,
                pure_MAE_meV=float(es[0].mean()), path_MAE_meV=float(es[1].mean()), replay_MAE_meV=float(es[2].mean()),
                materials=len(reference), chemistry_groups=n))
    return scores, materials


def finalize(args, plan, jobs, slots, extra, status):
    args.output.joinpath("fits").symlink_to(args.shared_fits.resolve(), target_is_directory=True)
    v1.verify_run = checked_reused_run
    v1.pure_weights = pure_weights_for_revision
    v1.inference(args, slots)
    original = json.loads((args.output/"inference_manifest.json").read_bytes())
    records = original["records"]
    measured = [(i, MAIN_SEEDS, "regular") for i in slots]
    if extra:
        sub = args.output/"replication"
        sub.mkdir(exist_ok=True)
        sub.joinpath("fits").symlink_to(args.shared_fits.resolve(), target_is_directory=True)
        v1.SEEDS = EXTRA_SEEDS
        try:
            v1.inference(SimpleNamespace(**(vars(args) | {"output":sub})), (1,))
        finally:
            v1.SEEDS = MAIN_SEEDS
        r = json.loads((sub/"inference_manifest.json").read_bytes())
        for row in r["records"]:
            source = sub/"predictions"/row["prediction_file"]
            (args.output/"predictions"/source.name).hardlink_to(source)
        records += r["records"]
        measured += [(1, EXTRA_SEEDS, "extra_only"), (1, MAIN_SEEDS+EXTRA_SEEDS, "five_seed")]
    scores, materials = summarize_metrics(args, records, plan, measured)
    save_csv(args.output/"intervals.csv", scores)
    save_csv(args.output/"material_gains.csv", materials)
    save_csv(args.output/"checkpoint_hashes.csv", records)
    history = []
    for r in records:
        if r["arm"] == "pure":
            continue
        arm, seed = r["arm"],r["seed"]
        marker = existing(args,plan,jobs,arm,seed)
        trace = rows((args.shared_fits/arm/f"seed{seed}/training.csv").read_bytes())
        components = "energy_loss_weighted" in trace[0]
        history.append(dict(arm=arm, seed=seed, steps=marker["steps"], fit_seconds=marker["fit_seconds"],
              setup_seconds=marker["setup_seconds"], total_seconds=marker["total_seconds"],
              gradient_clipping_frequency=sum(float(row["gradient_norm"])>plan["training"]["clip_norm"]
                                              for row in trace)/len(trace),
              mean_total_loss=float(np.mean([float(row["loss"]) for row in trace])),
              mean_energy_loss_weighted=float(np.mean([float(row["energy_loss_weighted"]) for row in trace]))
                  if components else None,
              mean_force_loss_weighted=float(np.mean([float(row["force_loss_weighted"]) for row in trace]))
                  if components else None,
              component_tracking=components, validation_barrier_mae_meV=r["validation_barrier_mae_meV"],
              origin_matrix_sha256=marker["matrix_sha256"], checkpoint_sha256=r["checkpoint_sha256"]))
    save_csv(args.output/"training_diagnostics.csv", history)
    atomic_json(args.output/"inference_manifest.json", dict(original, records=records,
                                                        extra_replication_scored=extra))
    status.update(state="complete", ended_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    atomic_json(args.output/"status.json", status)
    archive = args.output/"compact.tar.gz"
    part = archive.with_suffix(".partial")
    with tarfile.open(part,"w:gz") as z:
        for path in sorted(args.output.glob("*.csv"))+sorted(args.output.glob("*.json")):
            z.add(path,arcname=path.name)
        for path in sorted((args.output/"predictions").glob("*.csv")):
            z.add(path,arcname=f"predictions/{path.name}")
        for path in sorted((args.output/"instrumentation_smoke").rglob("*")):
            if path.is_file():
                z.add(path,arcname=path.relative_to(args.output).as_posix())
        for row in history:
            arm, seed = row["arm"], row["seed"]
            for suffix in ("training.csv","done.json"):
                entry=f"{arm}/seed{seed}/{suffix}"
                z.add(args.shared_fits/entry,arcname=f"fits/{entry}")
            z.add(args.shared_fits/f"{arm}_seed{seed}.log", arcname=f"fits/{arm}_seed{seed}.log")
    part.replace(archive)
    (args.output/"compact.sha256").write_text(file_sha(archive)+"\n")


def launch(args):
    args.output.mkdir(parents=True,exist_ok=False)
    old = json.loads(args.old_status.read_bytes())
    start = datetime.fromisoformat(old["started_utc"].replace("Z","+00:00")).timestamp()
    deadline = start + 7*3600
    train_cutoff = deadline - 1800
    plan, _, extras, _, jobs = v1.assets(args)
    info = dict(state="handoff", planned=plan["planned_fits"], parallel_workers=2, prior_start_utc=old["started_utc"],
                deadline_utc=datetime.fromtimestamp(deadline,timezone.utc).isoformat(), completed_paths=[],
                completed_fits=0, extra_replicated=False, skipped=[], cuda_allocator="expandable_segments:True")
    place = args.output/"status.json"
    atomic_json(place,info)
    try:
        legacy = dict(plan.get("compatible_matrix_sha256_slots", {}))
        legacy.setdefault(plan["earlier_matrix_sha256"],6)
        os.environ["B2_COMPATIBLE_MATRICES"] = json.dumps(legacy)
        pure_weights_for_revision(args,extras,jobs,plan)
        freeze_previous_dispatcher(args)
        visible = old.get("visible_gpu_count", 1)
        if args.reuse_benchmark:
            shutil.copytree(args.reuse_benchmark,args.output/"instrumentation_smoke")
        else:
            smoke_a = child(args,"path3",17,0,smoke=True)
            smoke_b = child(args,"replay3",17,1 % visible,smoke=True)
            if any([smoke_a.wait(), smoke_b.wait()]):
                raise RuntimeError("Dual-process instrumentation smoke failed")
        smoke = [json.loads((args.output/f"instrumentation_smoke/{arm}/seed17/smoke.json").read_bytes())
                 for arm in ("path3","replay3")]
        if any(r["matrix_sha256"] not in (file_sha(args.matrix), *legacy) or r["steps"]!=36
               or r["smoke_force_gradient"] <= 0 for r in smoke):
            raise ValueError("Missing compatible parallel instrumented acceptance")
        completed_speeds = []
        for slot in [r["slot"] for r in plan["candidates"]]:
            for seed in MAIN_SEEDS:
                for arm in (f"path{slot}",f"replay{slot}"):
                    record = existing(args,plan,jobs,arm,seed)
                    if record is not None:
                        completed_speeds.append(record["seconds_per_step"])
        speed = max([r["seconds_per_step"] for r in smoke] + completed_speeds)
        setup = max(r["setup_seconds"] for r in smoke)+10
        pending_est = sum(
            math.ceil(len(missing_for(args,plan,jobs,i,seeds))/2)*(plan["candidates"][i-1]["optimizer_steps"]*speed+setup)
            for i,seeds in [*((r["slot"],MAIN_SEEDS) for r in plan["candidates"]),(1,EXTRA_SEEDS)])
        info.update(state="training", predicted_remaining_fit_seconds=pending_est)
        atomic_json(place,info)
        for slot,seeds in [*((r["slot"],MAIN_SEEDS) for r in plan["candidates"]),(1,EXTRA_SEEDS)]:
            candidate = plan["candidates"][slot-1]
            pending = missing_for(args,plan,jobs,slot,seeds)
            if pending and time.time()+1.35*math.ceil(len(pending)/2)*(candidate["optimizer_steps"]*speed+setup)>train_cutoff:
                info["skipped"].append({"slot":slot,"seeds":seeds,"reason":"30-minute reserve"})
                if seeds == MAIN_SEEDS:
                    continue
                break
            info.update(state="training", current_slot=slot, seeds=seeds, candidate=candidate["edge_id"])
            atomic_json(place,info)
            if not run_pairs(args,pending,visible,train_cutoff):
                info["skipped"].append({"slot":slot,"seeds":seeds,"reason":"training cutoff"})
                break
            for seed in seeds:
                for prefix in ("path","replay"):
                    r = existing(args,plan,jobs,f"{prefix}{slot}",seed)
                    if r is None:
                        raise ValueError("Partial block")
                    speed=max(speed,r["seconds_per_step"])
            if seeds==MAIN_SEEDS:
                info["completed_paths"].append(slot)
            else:
                info["extra_replicated"]=True
            info["completed_fits"]=6*len(info["completed_paths"])+4*info["extra_replicated"]
            atomic_json(place,info)
        if not info["completed_paths"]:
            raise RuntimeError("No complete regular path")
        info.update(state="inference", scoring_started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
        atomic_json(place,info)
        finalize(args,plan,jobs,info["completed_paths"],info["extra_replicated"],info)
    except BaseException as error:
        info.update(state="error", error=f"{type(error).__name__}: {error}")
        atomic_json(place,info)
        raise


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("launch","fit"))
    for name in ("base","matrix","baselines","output"):
        p.add_argument(name,type=Path)
    for name in ("confirm-baselines","old-status","old-pid","shared-fits","reuse-benchmark"):
        p.add_argument("--"+name,type=Path)
    p.add_argument("--arm")
    p.add_argument("--seed",type=int)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    if a.stage=="fit":
        v1.fit=instrumented_fit
        v1.fit_task(a)
    else:
        launch(a)
