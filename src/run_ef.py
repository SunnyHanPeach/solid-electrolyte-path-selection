"""Fine-tune paired MACE models on fixed energy/force data with common batch orders."""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from prepare import digest, rows, write_csv


def load_design(inputs, checkpoint):
    design = json.loads((inputs / "design.json").read_bytes())
    for name, sha in design["input_sha256"].items():
        if digest(inputs / name) != sha:
            raise ValueError(f"Changed frozen input: {name}")
    if digest(checkpoint) != design["checkpoint_sha256"]:
        raise ValueError("Unexpected checkpoint SHA-256")
    return design


def make_graphs(atoms_list, index, wanted, calc):
    from mace import data

    spec = data.KeySpecification(info_keys={"energy": "REF_energy"}, arrays_keys={"forces": "REF_forces"})
    graphs = {}
    for i in wanted:
        atoms, row = atoms_list[i], index[i]
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces(apply_constraint=False)
        if (len(atoms) != int(row["n_atoms"]) or forces.shape != (len(atoms), 3)
                or not np.isfinite(forces).all() or abs(energy - float(row["energy_eV"])) > 1e-8):
            raise ValueError(f"Energy or force labels do not match the index at {i}")
        atoms = atoms.copy()
        atoms.info["REF_energy"] = energy
        atoms.arrays["REF_forces"] = forces
        config = data.config_from_atoms(atoms, key_specification=spec, head_name=calc.head)
        graphs[i] = data.AtomicData.from_config(config, z_table=calc.z_table,
                                                 cutoff=calc.r_max, heads=calc.available_heads)
    return graphs


def schedule(n_frames, config, seed):
    rng = np.random.default_rng(seed)
    size = config["batch_size"]
    for _ in range(config["epochs"]):
        epoch = rng.permutation(n_frames)
        for i in range(0, n_frames, size):
            yield epoch[i:i + size].tolist()


def build_batch(graphs, indices, device):
    from mace.tools.torch_geometric.batch import Batch

    return Batch.from_data_list([graphs[i] for i in indices]).to(device)


def check_energy_gradient(model, batch, energy, loss, energy_weight):
    """Verify the shortcut against the original forward path at zero force weight."""
    import torch
    from mace.modules.loss import WeightedEnergyForcesLoss

    full = model(batch.to_dict(), training=True, compute_force=True)
    full_loss = WeightedEnergyForcesLoss(energy_weight, 0.0).to(energy.device)(batch, full)
    torch.testing.assert_close(energy, full["energy"], atol=0, rtol=0)
    torch.testing.assert_close(loss, full_loss, atol=0, rtol=0)
    params = tuple(model.parameters())
    fast = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    reference = torch.autograd.grad(full_loss, params, allow_unused=True)
    for param, fast_grad, expected_grad in zip(params, fast, reference):
        if fast_grad is None:
            fast_grad = torch.zeros_like(param)
        if expected_grad is None:
            expected_grad = torch.zeros_like(param)
        torch.testing.assert_close(fast_grad, expected_grad, rtol=1e-5, atol=1e-8)


def fit(foundation, graphs, indices, order, config, seed, device, smoke, after_step=None):
    import torch
    from mace.modules.loss import WeightedEnergyForcesLoss, weighted_mean_squared_error_energy

    torch.manual_seed(seed)
    model = deepcopy(foundation).train()
    model.requires_grad_(True)
    energy_weight, force_weight = config["energy_weight"], config["forces_weight"]
    supervise_forces = force_weight > 0
    if supervise_forces:
        loss_fn = WeightedEnergyForcesLoss(energy_weight, force_weight).to(device)
    else:
        def loss_fn(batch, prediction):
            return energy_weight * weighted_mean_squared_error_energy(batch, prediction)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    first_energy = None
    force_gradient_max = None
    energy_gradient_check = None
    trace = []
    start = time.monotonic()
    for step, slots in enumerate(order, 1):
        frame_ids = [indices[slot] for slot in slots]
        batch = build_batch(graphs, frame_ids, device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch.to_dict(), training=True, compute_force=supervise_forces)
        loss = loss_fn(batch, pred)
        if first_energy is None:
            first_energy = pred["energy"].detach().cpu().tolist()
            if smoke and supervise_forces:
                force_loss = (pred["forces"] - batch.forces).square().mean()
                gradients = torch.autograd.grad(force_loss, tuple(model.parameters()),
                                                retain_graph=True, allow_unused=True)
                force_gradient_max = max(float(g.detach().abs().max()) for g in gradients if g is not None)
                if not np.isfinite(force_gradient_max) or force_gradient_max <= 0:
                    raise ValueError("The force loss did not produce a finite nonzero parameter gradient")
            elif smoke:
                full_batch = build_batch(graphs, frame_ids, device)
                check_energy_gradient(model, full_batch, pred["energy"], loss, energy_weight)
                energy_gradient_check = True
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_norm"], error_if_nonfinite=True)
        value, gradient = float(loss.detach()), float(norm)
        if not np.isfinite(value) or gradient <= 0:
            raise ValueError(f"Invalid E/F loss or gradient at step {step}")
        trace.append(dict(step=step, loss=value, gradient_norm=gradient))
        optimizer.step()
        if after_step is not None:
            after_step(step, model)
        if step % 70 == 0:
            print(f"step {step}/{len(order)}, {'E/F' if supervise_forces else 'E'} loss {value:.6g}", flush=True)
    moved = any(not torch.equal(param.detach(), original.detach()) for param, original in
                zip(model.parameters(), foundation.parameters()))
    if not moved:
        raise ValueError("No model parameters changed")
    model.eval().requires_grad_(False)
    return model, trace, first_energy, force_gradient_max, energy_gradient_check, time.monotonic() - start


def score(model, graphs, frame_rows, validation_ids, references, device):
    import torch

    prediction = []
    force_sse, force_count = 0.0, 0
    grouped = defaultdict(list)
    for start in range(0, len(validation_ids), 2):
        ids = validation_ids[start:start + 2]
        batch = build_batch(graphs, ids, device)
        with torch.enable_grad():
            pred = model(batch.to_dict(), training=False, compute_force=True)
        if not torch.isfinite(pred["forces"]).all() or not torch.isfinite(pred["energy"]).all():
            raise ValueError("Validation predictions contain nonfinite values")
        diff = (pred["forces"].detach() - batch.forces).to(torch.float64)
        force_sse += float(diff.square().sum())
        force_count += diff.numel()
        for i, value in zip(ids, pred["energy"].detach().cpu().tolist()):
            row = frame_rows[i]
            edge = row["edge_id"]
            prediction.append(dict(edge_id=edge, frame=row["frame"], energy_eV=value))
            grouped[edge].append(value)
    if grouped.keys() != references.keys():
        raise ValueError("Validation paths differ from the locked references")
    by_path = []
    for edge, values in grouped.items():
        ref = references[edge]
        if len(values) != int(ref["images"]):
            raise ValueError(f"Incomplete validation path: {edge}")
        by_path.append(dict(edge_id=edge, material_id=ref["material_id"],
                            forward_error_eV=abs(max(values) - values[0] - float(ref["forward_eV"])),
                            span_error_eV=abs(max(values) - min(values) - float(ref["published_span_eV"]))))
    return prediction, by_path, dict(forward_mae_eV=float(np.mean([r["forward_error_eV"] for r in by_path])),
                                     span_mae_eV=float(np.mean([r["span_error_eV"] for r in by_path])),
                                     force_rmse_eV_A=float(np.sqrt(force_sse / force_count)))


def run(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from ase import __version__ as ase_version
    from ase.io import read
    with torch.serialization.safe_globals([slice]):
        from e3nn import o3  # noqa: F401
    import mace
    from mace.calculators import MACECalculator

    if mace.__version__ != "0.3.14":
        raise RuntimeError(f"Expected MACE 0.3.14, found {mace.__version__}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("This Python environment has no CUDA-capable PyTorch device")
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    try:
        design = load_design(args.inputs, args.checkpoint)
        config = dict(design["settings"])
        if args.energy_only:
            config["forces_weight"] = 0.0
        if args.seed not in config["seeds"] or args.case not in design["cases"]:
            raise ValueError("Case or seed was not frozen")
        case = design["cases"][args.case]
        frames = rows(args.inputs / "frames.csv")
        refs = {r["edge_id"]: r for r in rows(args.inputs / "validation_paths.csv")}
        validation = [i for i, r in enumerate(frames) if r["role"] == "validation"]
        baseline = design["d0"]
        if args.smoke:
            baseline = baseline[:2]
            keys = list(refs)[:2]
            refs = {key: refs[key] for key in keys}
            validation = [i for i in validation if frames[i]["edge_id"] in refs]
        variants = {arm: baseline + case[arm] for arm in ("replay", "candidate")}
        replay_plan_sha = None
        if args.replay_plan is not None:
            plan = json.loads(args.replay_plan.read_bytes())
            if (args.energy_only or plan["ef_design_sha256"] != digest(args.inputs / "design.json")
                    or args.seed not in plan["seeds"]):
                raise ValueError("Replay check does not match the frozen E/F protocol")
            extra = plan["cases"][args.case]["new_replays"][args.replay_draw]
            if not set(extra).issubset(design["d0"]):
                raise ValueError("New replay contains a frame outside D0")
            variants["replay"] = baseline + extra
            replay_plan_sha = digest(args.replay_plan)
        order = list(schedule(len(variants["replay"]), config, args.seed))
        if args.smoke:
            order = order[:3]
        if (len(variants["replay"]) != len(variants["candidate"])
                or sum(int(frames[i]["n_atoms"]) for i in variants["replay"])
                != sum(int(frames[i]["n_atoms"]) for i in variants["candidate"])):
            raise ValueError("Training arms have different exposure")
        if replay_plan_sha is not None:
            del variants["candidate"]
        if args.candidate_only:
            del variants["replay"]
        atoms = read(str(args.inputs / "geometries.extxyz"), index=":", format="extxyz")
        wanted = sorted(set(validation).union(*variants.values()))
        calc = MACECalculator(model_paths=str(args.checkpoint), device=args.device, default_dtype="float32")
        foundation = calc.models[0].eval()
        graphs = make_graphs(atoms, frames, wanted, calc)
        del atoms
        output = args.output
        output.mkdir(parents=True, exist_ok=False)
        run_info = dict(seed=args.seed, edge_id=args.case, smoke=args.smoke, steps=len(order),
                        training_energy_weight=config["energy_weight"], training_force_weight=config["forces_weight"],
                        order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),
                        input_design_sha256=digest(args.inputs / "design.json"), device=args.device,
                        hardware=torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
                        torch=torch.__version__, mace=mace.__version__, ase=ase_version,
                        cuda=torch.version.cuda, training_frames=len(next(iter(variants.values()))),
                        training_atom_images=sum(int(frames[i]["n_atoms"]) for i in next(iter(variants.values()))),
                        validation_paths=len(refs), arms={})
        if replay_plan_sha is not None:
            run_info.update(replay_plan_sha256=replay_plan_sha, replay_draw=int(args.replay_draw))
        for arm, indices in variants.items():
            print(f"{args.case} seed {args.seed} {arm}: {len(order)} optimization steps", flush=True)
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            model, trace, energies, force_gradient, energy_gradient, seconds = fit(
                foundation, graphs, indices, order, config, args.seed, args.device, args.smoke)
            pred, by_path, metrics = score(model, graphs, frames, validation, refs, args.device)
            write_csv(output / f"{arm}_training.csv", trace)
            write_csv(output / f"{arm}_predictions.csv", pred)
            write_csv(output / f"{arm}_path_errors.csv", by_path)
            if not args.smoke:
                torch.save(model.cpu(), output / f"{arm}.model")
            run_info["arms"][arm] = dict(metrics, training_seconds=seconds, first_batch_energies_eV=energies,
                                         first_force_only_gradient_max=force_gradient,
                                         energy_gradient_matches_full_forward=energy_gradient,
                                         peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None)
        result_name = ("control_result" if replay_plan_sha is not None else
                       "candidate_result" if args.candidate_only else "paired_result")
        destination = output / f"{result_name}.json"
        temporary = output / f"{result_name}.partial.json"
        temporary.write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)
        print(json.dumps(run_info, indent=2), flush=True)
    finally:
        torch.use_deterministic_algorithms(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--energy-only", action="store_true")
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--replay-plan", type=Path)
    parser.add_argument("--replay-draw")
    args = parser.parse_args()
    if (args.replay_plan is None) != (args.replay_draw is None):
        parser.error("--replay-plan and --replay-draw must be passed together")
    if args.candidate_only and args.replay_plan is not None:
        parser.error("A candidate-only run cannot also train a replay")
    run(args)
