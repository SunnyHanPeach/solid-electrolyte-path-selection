"""Run B2 on the frozen Phase 1 recipe, then score all complete paths on one panel."""
import argparse
from collections import defaultdict
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from zipfile import ZipFile

import numpy as np

from audit_phase_b0 import frame_prefix
from audit_signed_barriers import file_sha, rows, sha
from run_ef import fit, make_graphs
from run_force_weighting import atomic_json, load_bundle, save_csv, score_profiles

SEEDS = (17, 29, 43)
THREADS = 4
SMOKE_STEPS = 36
MARGIN_MEV = .25


def assets(args):
    plan, base = load_bundle(args.base)
    with ZipFile(args.matrix) as z:
        lock = json.loads(z.read("plan.json"))
        extras = {name: z.read(name) for name in lock["sha256"]}
    if (file_sha(args.base) != lock["base_bundle_sha256"]
            or any(sha(v) != lock["sha256"][k] for k, v in extras.items())
            or lock["training"] != plan["training"]
            or sha(base["foundation.model"]) != lock["foundation_sha256"]):
        raise ValueError("B2 frozen inputs changed")
    base["training.xyz"] = frame_prefix(base["training.xyz"], 278)
    if sha(base["training.xyz"]) != lock["d0_sha256"]:
        raise ValueError("B2 D0 identity changed")
    base["training.xyz"] += extras["added.xyz"]
    frame_rows = rows(base["training_frames.csv"])[:278] + rows(extras["added_frames.csv"])
    return lock, base, extras, frame_rows, json.loads(extras["orders.json"])


def torch_environment():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    import ase
    import mace
    import torch

    if (str(torch.__version__), mace.__version__, ase.__version__, np.__version__) != (
            "2.8.0+cu128", "0.3.14", "3.29.0", "2.2.6"):
        raise RuntimeError("Historical MACE, Torch, ASE or NumPy version changed")
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU visible")
    torch.set_num_threads(THREADS)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    gpu = torch.cuda.get_device_properties(0)
    return torch, dict(gpu=gpu.name, vram_bytes=gpu.total_memory, capability=[gpu.major, gpu.minor],
                       torch=str(torch.__version__), mace=mace.__version__, ase=ase.__version__,
                       numpy=np.__version__, threads=THREADS, dtype="float32", allow_tf32=False,
                       deterministic=True, python=sys.version.split()[0], cuda=torch.version.cuda)


def foundation(data, torch):
    from mace.calculators import MACECalculator
    model = torch.load(io.BytesIO(data["foundation.model"]), map_location="cpu", weights_only=False)
    return MACECalculator(models=model, device="cuda", default_dtype="float32")


def fit_task(args):
    from ase.io import read
    torch, environment = torch_environment()
    lock, base, _, frame_rows, jobs = assets(args)
    order = jobs[str(args.seed)][args.arm]
    frozen_steps = len(order)
    schedule = sha(json.dumps(order).encode())
    if args.smoke:
        sizes = [int(r["n_atoms"]) for r in frame_rows]
        heavy = max(order, key=lambda batch: sum(sizes[i] for i in batch))
        order = [heavy, *order[:SMOKE_STEPS - 1]]
    folder = args.output/args.arm/f"seed{args.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    identity = dict(seed=args.seed, arm=args.arm, schedule_sha256=schedule, frozen_steps=frozen_steps,
                    matrix_sha256=file_sha(args.matrix), source_base_sha256=lock["base_bundle_sha256"],
                    environment=environment)
    marker = folder/"done.json"
    if marker.exists() and not args.smoke:
        verify_run(folder, identity)
        return
    start = time.monotonic()
    calc = foundation(base, torch)
    atoms = read(io.StringIO(base["training.xyz"].decode()), index=":", format="extxyz")
    graphs = make_graphs(atoms, frame_rows, sorted({i for b in order for i in b}), calc)
    for g in graphs.values():
        if any(g[k].numel() != 1 or g[k].item() != 1. for k in ("weight", "energy_weight", "forces_weight")):
            raise ValueError("Nonunit graph loss weight")
    setup = time.monotonic() - start
    torch.cuda.reset_peak_memory_stats()
    model, trace, _, grad, _, elapsed = fit(
        calc.models[0].eval(), graphs, list(range(len(frame_rows))), order,
        lock["training"], args.seed, "cuda", args.smoke)
    torch.cuda.synchronize()
    timings = dict(fit_seconds=elapsed, setup_seconds=setup, total_seconds=time.monotonic() - start,
                   seconds_per_step=elapsed / len(order), steps=len(order),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(), smoke_force_gradient=grad)
    if args.smoke:
        atomic_json(folder/"smoke.json", dict(identity, **timings))
    else:
        stage = folder/"model.partial"
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, stage)
        stage.replace(folder/"model.pt")
        save_csv(folder/"training.csv", trace)
        atomic_json(marker, dict(identity, **timings, files={
            name: file_sha(folder/name) for name in ("model.pt", "training.csv")}))
    print(json.dumps(timings), flush=True)


def verify_run(folder, identity):
    record = json.loads((folder/"done.json").read_bytes())
    if any(record[k] != v for k, v in identity.items()) or record["steps"] != record["frozen_steps"]:
        raise ValueError(f"Wrong run identity: {folder}")
    if any(file_sha(folder/k) != v for k,v in record["files"].items()):
        raise ValueError(f"Model or log checksum differs: {folder}")
    trace = rows((folder/"training.csv").read_bytes())
    if [int(r["step"]) for r in trace] != list(range(1, record["steps"] + 1)) or not all(
            math.isfinite(float(r[k])) for r in trace for k in ("loss", "gradient_norm")):
        raise ValueError(f"Incomplete optimizer trace: {folder}")
    return record


def pure_weights(args, extras, jobs, plan):
    weights = {}
    for s, record in json.loads(extras["baselines.json"]).items():
        folder = args.baselines/f"seed{s}"/"pure"
        if (json.loads((folder/"done_710.json").read_bytes()) != record
                or record["steps" if "steps" in record else "optimizer_steps"] != 710
                or record["bundle"] != plan["base_bundle_sha256"]
                or record["schedule"] != sha(json.dumps(jobs[s]["pure"]).encode())):
            raise ValueError(f"Archived pure D0 run does not match seed {s}")
        path = folder/"model_710.pt"
        expected = record["files"]["model_710.pt"]
        if file_sha(path) != expected:
            raise ValueError(f"Pure D0 weight changed for seed {s}")
        weights[s] = (path, expected, record)
    return weights


def task_identity(seed, arm, args, plan, job):
    return dict(seed=seed, arm=arm, frozen_steps=len(job), schedule_sha256=sha(json.dumps(job).encode()),
                matrix_sha256=file_sha(args.matrix), source_base_sha256=plan["base_bundle_sha256"])


def inference(args, slots):
    from ase.io import read
    from mace import data
    torch, environment = torch_environment()
    plan, base, extras, _, jobs = assets(args)
    pure = pure_weights(args, extras, jobs, plan)
    calc = foundation(base, torch)
    model = calc.models[0].eval().requires_grad_(False)
    frames = rows(base["evaluation_frames.csv"])
    atoms = read(io.StringIO(base["evaluation.xyz"].decode()), index=":", format="extxyz")
    if len(frames) != len(atoms) or any(len(a) != int(r["n_atoms"]) or a.info or a.calc or
                                       set(a.arrays) != {"numbers", "positions"} for a,r in zip(atoms, frames)):
        raise ValueError("Evaluation frames differ or contain labels")
    graphs = {}
    spec = data.KeySpecification()
    for i,a in enumerate(atoms):
        conf = data.config_from_atoms(a, key_specification=spec, head_name=calc.head)
        graphs[i] = data.AtomicData.from_config(conf, z_table=calc.z_table, cutoff=calc.r_max,
                                                heads=calc.available_heads)
    output = args.output/"predictions"
    output.mkdir(exist_ok=True)
    records = []
    for arm in ("pure", *(f"{prefix}{slot}" for slot in slots for prefix in ("path", "replay"))):
        for seed in SEEDS:
            if arm == "pure":
                path, expected, source = pure[str(seed)]
                training_env = source["environment"]
                origin = "historical pure D0; re-inferred in B2 runtime"
            else:
                folder = args.output/"fits"/arm/f"seed{seed}"
                info = verify_run(folder, task_identity(seed, arm, args, plan, jobs[str(seed)][arm]))
                path, expected, training_env = folder/"model.pt", info["files"]["model.pt"], info["environment"]
                origin = "B2 protected Path/Replay"
            start = time.monotonic()
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
            table = score_profiles(model, graphs, frames, "cuda", torch)
            name = f"{arm}_seed{seed}.csv"
            save_csv(output/name, table)
            records.append(dict(arm=arm, seed=seed, prediction_file=name, prediction_sha256=file_sha(output/name),
                                checkpoint_sha256=expected, count=len(table), origin=origin,
                                training_hardware=training_env.get("gpu", "unavailable"),
                                scoring_seconds=time.monotonic() - start))
            print(f"scored {arm} seed{seed} {records[-1]['scoring_seconds']:.1f}s", flush=True)
    atomic_json(args.output/"inference_manifest.json",
                dict(inference_environment=environment, matrix_sha256=file_sha(args.matrix), records=records))


def score(args, slots):
    plan, _, extras, _, jobs = assets(args)
    reference = rows(extras["reference_barriers.csv"])
    truth = {r["edge_id"]: r for r in reference}
    # Group resampling preserves material-level paired predictions within an exact chemistry.
    chemicals = sorted({r["chemsys"] for r in reference})
    indexes = [chemicals.index(r["chemsys"]) for r in reference]
    n = len(chemicals)
    counts = np.random.default_rng(20261126).multinomial(n, np.full(n, 1/n), size=10000)[:, indexes]
    draws = counts / counts.sum(axis=1, keepdims=True)
    expected_keys = {(r["edge_id"], r["material_id"], i) for r in reference
                     for i in range(1, int(r["images"])+1)}
    manifest = json.loads((args.output/"inference_manifest.json").read_bytes())
    if manifest["matrix_sha256"] != file_sha(args.matrix):
        raise ValueError("Inference matrix mismatch")
    expected_arms = ("pure", *(f"{a}{i}" for i in slots for a in ("path", "replay")))
    records = manifest["records"]
    if {(r["arm"], r["seed"]) for r in records} != {(a, s) for a in expected_arms for s in SEEDS} or len(
            records) != len(expected_arms) * len(SEEDS):
        raise ValueError("Incomplete inference grid")
    predictions = {}
    errors = {}
    for info in records:
        raw = (args.output/"predictions"/info["prediction_file"]).read_bytes()
        table = rows(raw)
        if (sha(raw) != info["prediction_sha256"] or len(table) != len(expected_keys) or
                {(r["edge_id"], r["material_id"], int(r["frame"])) for r in table} != expected_keys):
            raise ValueError("Evaluation key or hash mismatch")
        curves = defaultdict(dict)
        for r in table:
            curves[r["edge_id"]][int(r["frame"])] = float(r["energy_eV"])
        for edge, v in curves.items():
            barrier = 1000. * (max(v.values()) - v[1])
            key = info["arm"], info["seed"], edge
            predictions[key] = barrier
            errors[key] = abs(barrier - float(truth[edge]["dft_forward_barrier_meV"]))
    path_results, material_results, intervals = [], [], []
    all_metrics = []
    for slot in slots:
        candidate = plan["candidates"][slot-1]
        arms = ("pure", f"path{slot}", f"replay{slot}")
        blocks = {arm: np.array([[errors[arm, seed, r["edge_id"]] for r in reference] for seed in SEEDS])
                  for arm in arms}
        path_error = blocks[arms[1]]
        fit_envs = {
            json.dumps(verify_run(args.output/"fits"/arm/f"seed{s}",
                                 task_identity(s, arm, args, plan, jobs[str(s)][arm]))["environment"], sort_keys=True)
            for arm in arms[1:] for s in SEEDS
        }
        if len(fit_envs) != 1:
            raise ValueError("Mixed Path/Replay training environments")
        path_results.append(dict(candidate, pure_D0_MAE_meV=float(blocks["pure"].mean()),
            path_MAE_meV=float(path_error.mean()), replay_MAE_meV=float(blocks[arms[2]].mean()),
            absolute_gain_meV=float((blocks["pure"]-path_error).mean()),
            net_gain_meV=float((blocks[arms[2]]-path_error).mean())))
        for i, seed in enumerate(SEEDS):
            for j, r in enumerate(reference):
                e = r["edge_id"]
                material_results.append(dict(slot=slot, candidate_edge=candidate["edge_id"], seed=seed,
                    material_id=r["material_id"], evaluation_edge=e, chemsys=r["chemsys"],
                    dft_forward_barrier_meV=r["dft_forward_barrier_meV"],
                    pure_prediction_meV=predictions["pure", seed, e],
                    path_prediction_meV=predictions[arms[1], seed, e],
                    replay_prediction_meV=predictions[arms[2], seed, e],
                    pure_error_meV=float(blocks["pure"][i,j]), path_error_meV=float(path_error[i,j]),
                    replay_error_meV=float(blocks[arms[2]][i,j]),
                    absolute_gain_meV=float(blocks["pure"][i,j]-path_error[i,j]),
                    net_gain_meV=float(blocks[arms[2]][i,j]-path_error[i,j])))
        for metric, control in (("absolute", "pure"), ("net", arms[2])):
            diff = blocks[control] - path_error
            avg = diff.mean(axis=0)
            low, high = np.quantile(draws @ avg, [.025, .975])
            seed_values = diff.mean(axis=1)
            state = ("positive" if low > MARGIN_MEV and seed_values.min() > 0 else
                     "negative" if high < -MARGIN_MEV and seed_values.max() < 0 else
                     "near_zero" if low >= -MARGIN_MEV and high <= MARGIN_MEV else "unresolved")
            intervals.append(dict(slot=slot, candidate_edge=candidate["edge_id"], metric=metric, mean_meV=float(avg.mean()),
                low95_meV=float(low), high95_meV=float(high), seed17_meV=float(seed_values[0]),
                seed29_meV=float(seed_values[1]), seed43_meV=float(seed_values[2]),
                materials=len(reference), chemistry_groups=n, margin_meV=MARGIN_MEV, status=state))
        all_metrics.append(path_results[-1])
    if not path_results:
        raise ValueError("No complete Path/Replay blocks")
    save_csv(args.output/"path_results.csv", path_results)
    save_csv(args.output/"material_gains.csv", material_results)
    save_csv(args.output/"intervals.csv", intervals)
    save_csv(args.output/"checkpoint_hashes.csv", records)
    return all_metrics


def child(args, arm, seed, directory, gpu=0, smoke=False):
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "fit", str(args.base),
               str(args.matrix), str(args.baselines), str(directory), "--arm", arm, "--seed", str(seed)]
    if smoke:
        command.append("--smoke")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS=str(THREADS),
               MKL_NUM_THREADS=str(THREADS), PYTHONDONTWRITEBYTECODE="1")
    with (directory/f"{arm}_seed{seed}.log").open("ab") as log:
        return subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)


def wait_children(children):
    return [process.wait() for process in children]


def smoke_timing(args, plan, gpu_count):
    slot = max(plan["candidates"], key=lambda r: int(r["atoms"]))["slot"]
    first, second = f"path{slot}", f"replay{slot}"
    smoke = args.output/"smoke"
    solo = smoke/"single"
    start = time.monotonic()
    if child(args, first, 17, solo, smoke=True).wait():
        raise RuntimeError("Single process smoke failed")
    one = json.loads((solo/first/"seed17/smoke.json").read_bytes())
    solo_seconds = time.monotonic() - start
    limit = one["environment"]["vram_bytes"]
    try_dual = gpu_count >= 2 or 2 * one["peak_reserved_bytes"] < .86 * limit
    result = dict(single=one, single_worker_wall_seconds=solo_seconds, attempted_dual=try_dual,
                  visible_gpu_count=gpu_count, parallel_workers=1, predicted_seconds_per_36=0)
    if try_dual:
        dual = smoke/"concurrent"
        start = time.monotonic()
        rc = wait_children([child(args, arm, 17, dual, i % gpu_count, smoke=True)
                            for i, arm in enumerate((first, second))])
        wall = time.monotonic() - start
        result["dual_exit_codes"] = rc
        result["dual_worker_wall_seconds"] = wall
        if not any(rc):
            fits = [json.loads((dual/arm/"seed17/smoke.json").read_bytes()) for arm in (first, second)]
            has_space = gpu_count >= 2 or sum(f["peak_reserved_bytes"] for f in fits) < .92 * limit
            if has_space and wall < 2 * solo_seconds and max(f["seconds_per_step"] for f in fits) < (
                    2 * one["seconds_per_step"]):
                result["parallel_workers"] = 2
            result["dual"] = fits
    active = result["dual"] if result["parallel_workers"] == 2 else [one]
    result["fit_seconds_per_step"] = max(f["seconds_per_step"] for f in active)
    result["startup_seconds"] = max(f["setup_seconds"] for f in active) + 10.
    result["predicted_seconds_per_36"] = sum(
        (c["optimizer_steps"] * result["fit_seconds_per_step"] + result["startup_seconds"])
        * (6 // result["parallel_workers"]) for c in plan["candidates"])
    atomic_json(args.output/"timing_smoke.json", result)
    print(json.dumps({k:result[k] for k in ("visible_gpu_count", "parallel_workers",
                     "fit_seconds_per_step", "predicted_seconds_per_36")}), flush=True)
    return result


def launch(args):
    start = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    plan, _, extras, _, jobs = assets(args)
    pure_weights(args, extras, jobs, plan)
    visible = int(subprocess.check_output(
        [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"], text=True).strip().splitlines()[-1])
    if visible < 1:
        raise ValueError("GPU device unavailable")
    deadline = start + plan["max_window_seconds"] - args.elapsed_setup_seconds
    train_cutoff = deadline - plan["reserved_end_seconds"]
    slots, timings, skipped = [], [], []
    status_path = args.output/"status.json"
    status = dict(started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start)),
                  deadline_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(deadline)),
                  planned=36, completed_paths=slots, skipped=skipped, state="smoke")
    atomic_json(status_path, status)
    try:
        smoke = smoke_timing(args, plan, visible)
        workers = smoke["parallel_workers"]
        slow_step = smoke["fit_seconds_per_step"]
        status["workers"] = workers
        status["estimated_training_seconds"] = smoke["predicted_seconds_per_36"]
        for candidate in plan["candidates"]:
            slot = candidate["slot"]
            required = (6 // workers) * (candidate["optimizer_steps"] * slow_step + smoke["startup_seconds"])
            if time.time() + 1.40 * required > train_cutoff:
                skipped.extend([r["slot"] for r in plan["candidates"] if r["slot"] >= slot])
                break
            status.update(state="training", current_slot=slot, candidate=candidate["edge_id"], workers=workers)
            atomic_json(status_path, status)
            began = time.monotonic()
            for seed in SEEDS:
                arms = [f"path{slot}", f"replay{slot}"]
                if workers == 2:
                    rc = wait_children([child(args, arm, seed, args.output/"fits", i % visible)
                                        for i, arm in enumerate(arms)])
                    if any(rc):
                        workers = 1
                        arms = [arm for arm, code in zip(arms, rc) if code]
                    else:
                        arms = []
                for arm in arms:
                    if child(args, arm, seed, args.output/"fits").wait():
                        raise RuntimeError(f"Training failed twice or in single-worker mode: {arm} seed{seed}")
                for arm in (f"path{slot}", f"replay{slot}"):
                    run = verify_run(args.output/"fits"/arm/f"seed{seed}",
                                     task_identity(seed, arm, args, plan, jobs[str(seed)][arm]))
                    slow_step = max(slow_step, run["seconds_per_step"])
            slots.append(slot)
            timings.append(dict(slot=slot, edge=candidate["edge_id"], seconds=time.monotonic() - began,
                                steps=candidate["optimizer_steps"], workers=workers))
            atomic_json(args.output/"training_time.json", dict(smoke=smoke, blocks=timings))
        if not slots:
            raise RuntimeError("Training budget did not admit a complete three-seed path")
        status.update(state="inference", complete_fits=len(slots)*6)
        atomic_json(status_path, status)
        stamp = time.monotonic()
        inference(args, slots)
        score(args, slots)
        atomic_json(args.output/"training_time.json", dict(smoke=smoke, blocks=timings,
                      inference_and_scoring_seconds=time.monotonic()-stamp, elapsed_seconds=time.time()-start))
        status.update(state="complete", completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        atomic_json(status_path, status)
        with tarfile.open(args.output/"compact.tar.gz.partial", "w:gz") as z:
            for path in sorted(args.output.rglob("*")):
                if path.is_file() and path.suffix not in (".pt", ".partial", ".tar", ".gz"):
                    z.add(path, arcname=path.relative_to(args.output).as_posix())
        temp = args.output/"compact.tar.gz.partial"
        temp.replace(args.output/"compact.tar.gz")
        (args.output/"compact.sha256").write_text(file_sha(args.output/"compact.tar.gz") + "\n")
    except BaseException as error:
        status.update(state="error", error=f"{type(error).__name__}: {error}")
        atomic_json(status_path, status)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("launch", "fit"))
    for name in ("base", "matrix", "baselines", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--arm")
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--elapsed-setup-seconds", type=int, default=600)
    opts = p.parse_args()
    (launch if opts.stage == "launch" else fit_task)(opts)
