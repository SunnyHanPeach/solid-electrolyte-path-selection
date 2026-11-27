"""Resume the protected trajectory with crossed model parameters and Adam histories."""
import argparse
from copy import deepcopy
import io
import json
import os
from pathlib import Path
from importlib.metadata import version
import random
from shutil import copyfile
import subprocess
from time import monotonic
from zipfile import ZipFile

import numpy as np

from audit_signed_barriers import ARMS, SEEDS, file_sha, rows, sha
from run_ef import build_batch, make_graphs, write_csv
from run_force_weighting import atomic_json, frozen_schedules, load_bundle, score_profiles
from step200_protocol import BRANCHES, branch_design


FACTORS = ("replay", "added")
STEPS = (200, 201, 210, 250, 300, 400)


def record_prediction(folder, step, model, graphs, eval_rows, device, torch):
    target = folder / f"predictions_{step}.csv"
    pending = target.with_suffix(".partial")
    values = score_profiles(model, graphs, eval_rows, device, torch)
    write_csv(pending, values)
    pending.replace(target)
    return file_sha(target)


def step_model(model, optimizer, loss, batch, clip, torch):
    optimizer.zero_grad(set_to_none=True)
    pred = model(batch.to_dict(), training=True, compute_force=True)
    value = loss(batch, pred)
    value.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    if not torch.isfinite(value) or not torch.isfinite(norm):
        raise RuntimeError("Nonfinite gradient or loss")
    optimizer.step()
    return {"loss": float(value.detach()), "preclip_grad_norm": float(norm), "clipped": bool(norm > clip)}


def main(args):
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import ase
    from ase.io import read
    import mace
    from mace import data
    from mace.calculators import MACECalculator
    from mace.modules.loss import WeightedEnergyForcesLoss
    import torch

    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device unavailable")
    if (mace.__version__, ase.__version__, np.__version__) != ("0.3.14", "3.29.0", "2.2.6"):
        raise RuntimeError("The concrete data and model library versions differ from the trajectory")
    plan, files = load_bundle(args.bundle)
    index = rows(files["training_frames.csv"])
    replay, _augmented, _added = frozen_schedules(index, files, args.seed)
    order = replay[200:201 if args.smoke else 400]
    if args.branch_protocol:
        replay_branch, augmented_branch, weights, smoke_slot = branch_design(
            args.branch_protocol, args.bundle, args.states, index, files, args.seed)
        orders = {name: replay_branch if name == "replay" else augmented_branch for name in BRANCHES}
        if args.smoke:
            orders = {name: [batch[smoke_slot]] for name, batch in orders.items()}
        names = BRANCHES
    else:
        names = tuple(f"{model_name}-{adam_name}" for model_name in FACTORS for adam_name in FACTORS)
        orders = dict.fromkeys(names, order)
        weights = dict.fromkeys(names, {})
    selected = args.arm or names
    if not set(selected).issubset(names) or len(set(selected)) != len(selected):
        raise ValueError(f"Expected distinct arms from {names}")
    with ZipFile(args.states) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        provenance = next(r for r in manifest["start_states"] if r["seed"] == args.seed)
        if (manifest["bundle_sha256"] != file_sha(args.bundle) or
                provenance["replay_steps_201_400_sha256"] != sha(json.dumps(replay[200:400]).encode())):
            raise ValueError("Frozen training schedule differs")
        origin = {}
        for name in FACTORS:
            raw = archive.read(f"seed{args.seed}/{name}.pt")
            key = "replay_hash" if name == "replay" else "augmentation_hash"
            if sha(raw) != provenance[key]:
                raise ValueError("Input checkpoint hash differs")
            origin[name] = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    output = args.output / f"seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    foundation = output / "foundation.model"
    if not foundation.exists():
        with foundation.open("xb") as f:
            f.write(files["foundation.model"])
    elif file_sha(foundation) != plan["sha256"]["foundation.model"]:
        raise ValueError("Foundation hash differs")
    calc = MACECalculator(model_paths=str(foundation), device=args.device, default_dtype="float32")
    model = calc.models[0].train().requires_grad_(True)
    train_atoms = read(io.StringIO(files["training.xyz"].decode()), index=":", format="extxyz")
    graphs = make_graphs(train_atoms, index, sorted({i for name in selected for ids in orders[name] for i in ids}), calc)
    eval_graphs = {}
    eval_rows = rows(files["evaluation_frames.csv"])
    blind = read(io.StringIO(files["evaluation.xyz"].decode()), index=":", format="extxyz")
    if len(blind) != len(eval_rows) or len(eval_rows) != 997:
        raise ValueError("Evaluation geometry count differs")
    if args.smoke:
        eval_rows, blind = eval_rows[:5], blind[:5]
    spec = data.KeySpecification()
    for i, (a, row) in enumerate(zip(blind, eval_rows)):
        if a.info or a.calc or set(a.arrays) != {"numbers", "positions"} or len(a) != int(row["n_atoms"]):
            raise ValueError("Invalid blind evaluation geometry")
        c = data.config_from_atoms(a, key_specification=spec, head_name=calc.head)
        eval_graphs[i] = data.AtomicData.from_config(c, z_table=calc.z_table, cutoff=calc.r_max, heads=calc.available_heads)
    recipe = plan["training"]
    loss = WeightedEnergyForcesLoss(recipe["energy_weight"], recipe["forces_weight"]).to(args.device)

    runtime = dict(torch=str(torch.__version__), device=args.device, threads=args.threads,
                   hardware=torch.cuda.get_device_name() if args.device == "cuda" else "cpu")
    if args.branch_protocol:
        runtime.update(ase=ase.__version__, mace=mace.__version__, numpy=np.__version__,
                       e3nn=version("e3nn"), cuda=torch.version.cuda, python=os.sys.version)
        if args.device == "cuda":
            runtime["gpu_driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True).splitlines()[0].strip()
    initial_predictions = {}
    for name in selected:
        t0 = monotonic()
        parameters, moments = ("added", "added") if args.branch_protocol else name.split("-")
        identity = dict(seed=args.seed, weights=parameters, adam=moments,
                        bundle=manifest["bundle_sha256"], states=file_sha(args.states),
                        schedule=provenance["replay_steps_201_400_sha256"],
                        runtime=runtime,
                        smoke=args.smoke, source_script_sha256=file_sha(Path(__file__)))
        if args.branch_protocol:
            identity.update(arm=name, protocol_sha256=file_sha(args.branch_protocol),
                            start_state_sha256=provenance["augmentation_hash"],
                            rng_state_sha256=provenance["replay_hash"],
                            schedule=sha(json.dumps(orders[name]).encode()),
                            smoke_source_step=201 + smoke_slot if args.smoke else None)
            random.seed(args.seed)
            np.random.seed(args.seed)
        folder = output / name
        folder.mkdir(exist_ok=True)
        model.load_state_dict(origin[parameters]["model"], strict=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=recipe["lr"])
        optimizer.load_state_dict(deepcopy(origin[moments]["adam"]))
        for group in optimizer.param_groups:
            if group["lr"] != recipe["lr"]:
                raise ValueError("History changed the learning rate")
            for parameter in group["params"]:
                state = optimizer.state[parameter]
                if state["exp_avg"].shape != parameter.shape or state["exp_avg_sq"].shape != parameter.shape:
                    raise ValueError("Adam state does not match its named model parameter")
        torch.set_rng_state(origin["replay"]["cpu_rng"])
        if args.device == "cuda":
            torch.cuda.set_rng_state(origin["replay"]["cuda_rng"])
            torch.cuda.reset_peak_memory_stats()
        history, predictions = [], {}
        resume_file = folder / "resume.pt"
        start = 200
        if resume_file.exists():
            state = torch.load(resume_file, map_location="cpu", weights_only=True)
            if state["identity"] != identity:
                raise ValueError(f"Resume identity mismatch in {folder}")
            model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["cpu_rng"])
            if args.device == "cuda":
                torch.cuda.set_rng_state(state["cuda_rng"])
            start, history, predictions = state["step"], state["history"], state["predictions"]
            for step, digest in predictions.items():
                if file_sha(folder / f"predictions_{step}.csv") != digest:
                    raise ValueError(f"Prediction hash differs in {folder}")
            initial = initial_predictions.setdefault(parameters, folder / "predictions_200.csv")
            if file_sha(initial) != predictions[200]:
                raise ValueError(f"Shared step-200 prediction differs in {folder}")
        else:
            if parameters not in initial_predictions:
                predictions[200] = record_prediction(folder, 200, model, eval_graphs, eval_rows, args.device, torch)
                initial_predictions[parameters] = folder / "predictions_200.csv"
            else:
                target = folder / "predictions_200.csv"
                temporary = target.with_suffix(".partial")
                copyfile(initial_predictions[parameters], temporary)
                temporary.replace(target)
                predictions[200] = file_sha(target)
        for step, ids in enumerate(orders[name], 201):
            if step <= start:
                continue
            batch = build_batch(graphs, ids, args.device)
            multipliers = [weights[name].get(i, 1.) for i in ids]
            if weights[name]:
                batch.forces_weight.mul_(batch.forces_weight.new_tensor(multipliers).reshape_as(batch.forces_weight))
            result = step_model(model, optimizer, loss, batch, recipe["clip_norm"], torch)
            if args.branch_protocol:
                result.update(frame_indices=ids, force_multipliers=multipliers)
            history.append(dict(step=step, **result))
            if step in STEPS:
                if any(v["step"].item() != step for v in optimizer.state.values()):
                    raise ValueError("Unexpected optimizer step after update")
                predictions[step] = record_prediction(folder, step, model, eval_graphs, eval_rows, args.device, torch)
                pending = resume_file.with_suffix(".partial")
                torch.save(dict(identity=identity, step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state() if args.device == "cuda" else None,
                                history=history, predictions=predictions), pending)
                pending.replace(resume_file)
                print(f"seed={args.seed} {name} checkpoint={step} elapsed={monotonic()-t0:.1f}s", flush=True)
        atomic_json(folder / "done.json", dict(
            identity=identity, predictions=predictions, training=history, seconds=monotonic()-t0,
            peak_memory_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ("bundle", "states", "output"):
        parser.add_argument(n, type=Path)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--arm", action="append")
    parser.add_argument("--branch-protocol", type=Path)
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args())
