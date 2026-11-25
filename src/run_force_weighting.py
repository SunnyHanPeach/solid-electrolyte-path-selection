"""Test energy-dependent force weighting on the frozen eight-path schedules."""
import argparse
from collections import defaultdict
from copy import deepcopy
import io
import json
import math
import os
from pathlib import Path
import tempfile
import time
from zipfile import ZipFile

import numpy as np

from audit_signed_barriers import ARMS, SEEDS, file_sha, load_frozen_data, rows, sha
from run_ef import build_batch, fit, make_graphs, write_csv


TREATMENTS = ("full", "energy_weighted", "uniform_weighted")
MILESTONES = (400, 807)
BATCH2_STRENGTH = 0.75
BATCH2_ARM = "protected8_batch2_random"


def load_bundle(path):
    with ZipFile(path) as archive:
        plan = json.loads(archive.read("plan.json"))
        names = ("training_frames.csv", "training.xyz", "evaluation_frames.csv",
                 "evaluation.xyz", "jobs.json", "foundation.model")
        files = {name: archive.read(name) for name in names}
    for name, raw in files.items():
        if sha(raw) != plan["sha256"][name]:
            raise ValueError(f"Frozen file changed: {name}")
    return plan, files


def force_weights(frames, order, added, strength=0.5):
    """Match the force coefficient per path after batch atom-count reduction."""
    edges = {frames[i]["edge_id"] for i in added}
    paths = defaultdict(list)
    for i, frame in enumerate(frames):
        if frame["edge_id"] in edges:
            paths[frame["edge_id"]].append(i)
    atoms = [int(frame["n_atoms"]) for frame in frames]
    exposures = defaultdict(float)
    for batch in order:
        count = sum(atoms[i] for i in batch)
        for i in batch:
            if i in added:
                exposures[i] += atoms[i] / count
    energy, uniform, details = {}, {}, {}
    for edge, ids in sorted(paths.items()):
        ids.sort(key=lambda i: int(frames[i]["frame"]))
        if [int(frames[i]["frame"]) for i in ids] != list(range(1, len(ids) + 1)):
            raise ValueError(f"Incomplete frozen training path {edge}")
        if not set(ids).issubset(added) or len(ids) < 3:
            raise ValueError(f"Expected a complete newly added path: {edge}")
        e = [float(frames[i]["energy_eV"]) for i in ids]
        excess = [max(0.0, v - min(e[0], e[-1])) for v in e[1:-1]]
        maximum = max(excess)
        interior = ids[1:-1]
        energy.update({i: 1.0 - strength * v / maximum if maximum else 1.0
                       for i, v in zip(interior, excess)})
        total = sum(exposures[i] for i in interior)
        weighted = sum(exposures[i] * energy[i] for i in interior)
        if not total:
            raise ValueError(f"No interior presentations for {edge}")
        uniform.update(dict.fromkeys(interior, weighted / total))
        if not math.isclose(sum(exposures[i] * uniform[i] for i in interior), weighted,
                            abs_tol=1e-12, rel_tol=0):
            raise ValueError(f"Force exposure mismatch for {edge}")
        details[edge] = dict(frames=interior, energy_multipliers=[energy[i] for i in interior],
                             uniform_multiplier=weighted / total,
                             force_coefficient_before=total, force_coefficient_after=weighted)
    return {"full": {}, "energy_weighted": energy, "uniform_weighted": uniform}, details


def frozen_schedules(frames, files, seed, augmented_arm=ARMS[1]):
    jobs = {cell["seed"]: cell["jobs"] for cell in json.loads(files["jobs.json"])}[seed]
    replay, augmented = jobs[ARMS[0]], jobs[augmented_arm]
    changes = [step for step, (a, b) in enumerate(zip(replay, augmented), 1) if a != b]
    atoms = [int(row["n_atoms"]) for row in frames]
    if (len(replay) != len(augmented) or len(augmented) != 807 or len(changes) != 97
            or changes[0] != 8):
        raise ValueError("Unexpected frozen schedule")
    for a, b in zip(replay, augmented):
        if len(a) != 2 or len(b) != 2 or [atoms[i] for i in a] != [atoms[i] for i in b]:
            raise ValueError("Frozen replay does not match batch atom counts")
    original = {i for batch in replay for i in batch}
    added = {i for batch in augmented for i in batch} - original
    if len({frames[i]["edge_id"] for i in added}) != 8:
        raise ValueError("The augmented schedule must add the original eight paths")
    return replay, augmented, added


def frozen_experiment(frames, files, seed, strength=0.5):
    _, augmented, added = frozen_schedules(frames, files, seed)
    weights, details = force_weights(frames, augmented, added, strength)
    return augmented, weights, details


def batch2_experiment(frames, files, seed):
    replay, augmented, added = frozen_schedules(frames, files, seed, BATCH2_ARM)
    _, _, first_added = frozen_schedules(frames, files, seed)
    if {frames[i]["material_id"] for i in added} & {frames[i]["material_id"] for i in first_added}:
        raise ValueError("Batch 2 overlaps the materials used for strength selection")
    weights, details = force_weights(frames, augmented, added, BATCH2_STRENGTH)
    matrix = {}
    for arm in (*TREATMENTS, "replay"):
        order = replay if arm == "replay" else augmented
        matrix[arm] = (order, weights.get(arm, {}), dict(
            cohort=BATCH2_ARM, strength=BATCH2_STRENGTH, schedule=sha(json.dumps(order).encode()),
            weights=details if arm in ("energy_weighted", "uniform_weighted") else {},
        ))
    return matrix


def omit_frames(frames, files, seed, region):
    """Replace equal numbers of presentations per path with existing matched replay."""
    replay, order, added = frozen_schedules(frames, files, seed)
    paths = defaultdict(list)
    for i in added:
        paths[frames[i]["edge_id"]].append(i)
    rng = np.random.default_rng(seed)
    edited = [batch.copy() for batch in order]
    manifest = {}
    atoms = [int(frame["n_atoms"]) for frame in frames]
    for edge, ids in sorted(paths.items()):
        ids.sort(key=lambda i: int(frames[i]["frame"]))
        interior = ids[1:-1]
        ranked = sorted(interior, key=lambda i: (-float(frames[i]["energy_eV"]), i))
        high = set(ranked[:math.ceil(len(interior) / 4)])
        other = set(interior) - high
        strata = defaultdict(lambda: ([], []))
        for step, batch in enumerate(order):
            for slot, i in enumerate(batch):
                if i in high or i in other:
                    strata[atoms[i], atoms[batch[1-slot]]][int(i in other)].append((step, slot))
        replacements = []
        which = 0 if region == "high" else 1
        for groups in strata.values():
            count = min(map(len, groups))
            selected = [rng.permutation(len(group))[:count] for group in groups]
            for k in selected[which]:
                step, slot = groups[which][k]
                replacement = replay[step][slot]
                if replacement in added:
                    raise ValueError(f"Replacement was not drawn from D0: {edge}")
                edited[step][slot] = replacement
                replacements.append([step + 1, slot, order[step][slot], replacement])
        if not replacements:
            raise ValueError(f"Path has no matching interior presentations: {edge}")
        manifest[edge] = dict(presentations=len(replacements), replacements=replacements)
    if any(len(set(batch)) != 2 for batch in edited):
        raise ValueError("Omission produced a duplicate batch member")
    return edited, {f"omit_{region}": {}}, manifest


def atomic_json(path, value):
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def score_profiles(model, graphs, eval_rows, device, torch):
    predictions = []
    mode = model.training
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, len(eval_rows), 2):
                ids = list(range(start, min(start + 2, len(eval_rows))))
                pred = model(build_batch(graphs, ids, device).to_dict(),
                             training=False, compute_force=False)["energy"]
                if not torch.isfinite(pred).all():
                    raise ValueError("Nonfinite held-out energy")
                predictions.extend(dict(eval_rows[i], energy_eV=value)
                                   for i, value in zip(ids, pred.cpu().tolist()))
    finally:
        model.train(mode)
    return predictions


def train(args, files, frames, config, order, weights, identity, milestones=MILESTONES):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    import ase
    from ase.io import read
    import torch
    import mace
    from mace import data
    from mace.calculators import MACECalculator

    if (mace.__version__, ase.__version__, np.__version__) != ("0.3.14", "3.29.0", "2.2.6"):
        raise RuntimeError("Expected MACE 0.3.14, ASE 3.29.0 and NumPy 2.2.6")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    environment = dict(torch=str(torch.__version__), mace=mace.__version__, ase=ase.__version__,
                       numpy=np.__version__, device=args.device, threads=args.threads)
    if args.device == "cuda":
        environment["gpu"] = torch.cuda.get_device_name()
    identity = dict(identity, environment=environment)
    max_step = args.smoke or args.until
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with tempfile.TemporaryDirectory(dir=args.output) as work:
        foundation_path = Path(work) / "foundation.model"
        foundation_path.write_bytes(files["foundation.model"])
        calc = MACECalculator(model_paths=str(foundation_path), device=args.device, default_dtype="float32")
        model = calc.models[0].eval()
        train_atoms = read(io.StringIO(files["training.xyz"].decode()), index=":", format="extxyz")
        graphs = make_graphs(train_atoms, frames, sorted({i for b in order[:max_step] for i in b}), calc)
        for graph in graphs.values():
            for key in ("weight", "energy_weight", "forces_weight"):
                if graph[key].numel() != 1 or graph[key].item() != 1:
                    raise ValueError(f"Expected original unit graph coefficient for {key}")
        eval_rows = rows(files["evaluation_frames.csv"])
        eval_graphs = {}
        if not args.smoke:
            blind = read(io.StringIO(files["evaluation.xyz"].decode()), index=":", format="extxyz")
            if len(blind) != len(eval_rows):
                raise ValueError("Held-out geometry count changed")
            spec = data.KeySpecification()
            for i, (atoms, row) in enumerate(zip(blind, eval_rows)):
                if (atoms.info or atoms.calc or set(atoms.arrays) != {"positions", "numbers"}
                        or len(atoms) != int(row["n_atoms"])):
                    raise ValueError("Invalid blind geometry")
                conf = data.config_from_atoms(atoms, key_specification=spec, head_name=calc.head)
                eval_graphs[i] = data.AtomicData.from_config(
                    conf, z_table=calc.z_table, cutoff=calc.r_max, heads=calc.available_heads
                )
        print(f"Prepared {len(graphs)} training graphs and {len(eval_graphs)} evaluation graphs "
              f"in {time.monotonic()-start:.1f}s", flush=True)
        for arm in args.arm or weights:
            task = dict(identity, arm=arm, optimizer_steps=max_step)
            folder = args.output / f"seed{args.seed}" / arm
            marker = folder / f"done_{max_step}.json"
            if marker.exists() and not args.smoke:
                old = json.loads(marker.read_bytes())
                if any(old[k] != v for k, v in task.items()) or any(
                    file_sha(folder / k) != h for k, h in old["files"].items()
                ):
                    raise ValueError(f"Completed run differs from its saved identity: {folder}")
                continue
            weighted_graphs = dict(graphs)
            for i, factor in weights[arm].items():
                if i in graphs:
                    weighted_graphs[i] = deepcopy(graphs[i])
                    weighted_graphs[i].forces_weight.mul_(factor)

            def after_step(step, current):
                if step not in milestones:
                    return
                prediction = score_profiles(current, eval_graphs, eval_rows, args.device, torch)
                save_csv(folder / f"predictions_{step}.csv", prediction)
                target = folder / f"model_{step}.pt"
                stage = target.with_suffix(".partial")
                torch.save({key: value.cpu() for key, value in current.state_dict().items()}, stage)
                stage.replace(target)
                print(f"seed {args.seed} {arm} step {step} scored", flush=True)

            folder.mkdir(parents=True, exist_ok=True)
            _, trace, _, grad, _, elapsed = fit(
                model, weighted_graphs, list(range(len(frames))), order[:max_step], config,
                args.seed, args.device, bool(args.smoke), None if args.smoke else after_step
            )
            print(json.dumps(dict(seed=args.seed, arm=arm, updates=max_step, seconds=elapsed,
                                  seconds_per_step=elapsed / max_step, smoke_force_gradient=grad)),
                  flush=True)
            if not args.smoke:
                save_csv(folder / f"training_{max_step}.csv", trace)
                names = [f"training_{max_step}.csv"] + [
                    f"{kind}_{step}.{suffix}" for step in milestones if step <= max_step
                    for kind, suffix in (("predictions", "csv"), ("model", "pt"))
                ]
                atomic_json(marker, dict(task, seconds=elapsed, files={
                    name: file_sha(folder / name) for name in names
                }))


def save_csv(path, records):
    stage = path.with_suffix(".partial")
    write_csv(stage, records)
    stage.replace(path)


def matches_batch2_design(marker, design):
    if any(marker.get(k) != design[k] for k in ("cohort", "strength", "schedule")):
        return False
    observed = marker.get("weights", {})
    expected = design["weights"]
    if observed.keys() != expected.keys():
        return False
    # Python 3.10 and 3.12 may sum exposure coefficients with different rounding.
    for edge, target in expected.items():
        record = observed[edge]
        if record.keys() != target.keys() or record["frames"] != target["frames"]:
            return False
        if not np.array_equal(np.asarray(record["energy_multipliers"], dtype=np.float32),
                              np.asarray(target["energy_multipliers"], dtype=np.float32)):
            return False
        if np.float32(record["uniform_multiplier"]) != np.float32(target["uniform_multiplier"]):
            return False
        if any(not math.isclose(record[k], target[k], rel_tol=0, abs_tol=1e-12)
               for k in ("force_coefficient_before", "force_coefficient_after")):
            return False
    return True


def score(args):
    eval_rows, _, truth = load_frozen_data(args.bundle, args.project)
    bundle_hash = file_sha(args.bundle)
    expected = {(r["panel"], r["edge_id"], int(r["frame"])) for r in eval_rows}
    materials = {(r["panel"], r["edge_id"]): r["material_id"] for r in eval_rows}
    if len(materials.values()) != len(set(materials.values())):
        raise ValueError("Multiple paths per evaluation material require grouped resampling")
    implementations = defaultdict(set)
    environments = set()
    experiments = {arm: (args.output, arm) for arm in TREATMENTS}
    if args.replicate_batch2:
        experiments["replay"] = (args.output, "replay")
        _, files = load_bundle(args.bundle)
        frames = rows(files["training_frames.csv"])
        designs = {seed: batch2_experiment(frames, files, seed) for seed in SEEDS}
    for item in args.compare:
        label, sep, job = item.partition("=")
        directory, tag, source_arm = job.rpartition("@")
        if not label or not sep or not directory or not tag or label in experiments:
            raise ValueError("Additional arms must use the unique syntax LABEL=DIRECTORY@SOURCE_ARM")
        experiments[label] = Path(directory), source_arm
    results = []
    for arm, (location, source_arm) in experiments.items():
        for seed in SEEDS:
            folder = location / f"seed{seed}" / source_arm
            done = folder / "done_807.json"
            if not done.exists():
                done = folder / f"done_{args.until}.json"
            marker = json.loads(done.read_bytes())
            if (marker["arm"] != source_arm or marker["seed"] != seed or marker["bundle"] != bundle_hash
                    or marker["optimizer_steps"] < args.until):
                raise ValueError(f"Unexpected run identity: {folder}")
            if args.replicate_batch2 and not matches_batch2_design(marker, designs[seed][arm][2]):
                raise ValueError(f"Changed batch 2 design: {folder}")
            implementations[arm].add(marker["implementation"])
            environments.add(json.dumps(marker["environment"], sort_keys=True))
            path = folder / f"predictions_{args.until}.csv"
            if file_sha(path) != marker["files"][path.name]:
                raise ValueError(f"Changed predictions: {path}")
            data = defaultdict(dict)
            records = rows(path.read_bytes())
            found = {(r["panel"], r["edge_id"], int(r["frame"])) for r in records}
            if found != expected or len(records) != len(expected):
                raise ValueError(f"Changed evaluation frames: {path}")
            for row in records:
                key = row["panel"], row["edge_id"]
                if materials[key] != row["material_id"]:
                    raise ValueError(f"Changed material ID: {key}")
                data[key][int(row["frame"])] = float(row["energy_eV"])
            for key, ref in truth.items():
                energies = np.array([data[key][i] for i in range(1, len(ref) + 1)])
                if not np.isfinite(energies).all():
                    raise ValueError(f"Nonfinite evaluation energy: {key}")
                pred = energies - energies[0]
                error = float(1000 * (pred.max() - ref.max()))
                high = sorted(range(1, len(ref) - 1), key=lambda i: (-ref[i], i))
                high = high[:math.ceil(len(high) / 4)]
                results.append(dict(panel=key[0], edge=key[1], material_id=materials[key],
                                    seed=seed, arm=arm, barrier_signed_error_meV=error,
                                    barrier_abs_error_meV=abs(error),
                                    high_profile_mae_meV=float(1000 * abs(pred-ref)[high].mean())))
    if len(environments) != 1 or any(len(h) != 1 for h in implementations.values()):
        raise ValueError("Treatment matrix contains mixed execution environments or code within an arm")
    same_code = experiments if args.replicate_batch2 else TREATMENTS
    if len({next(iter(implementations[arm])) for arm in same_code}) != 1:
        raise ValueError("Original matrix contains mixed code")
    metrics = ("barrier_abs_error_meV", "barrier_signed_error_meV", "high_profile_mae_meV")
    cells = {(r["panel"], r["edge"], r["seed"], r["arm"]): r for r in results}
    comparisons = {}
    arm_metrics = {}
    contrasts = [("energy_weighted", "full"), ("uniform_weighted", "full"),
                 ("energy_weighted", "uniform_weighted")]
    if args.replicate_batch2:
        contrasts += [(arm, "replay") for arm in TREATMENTS]
    for arm in sorted(experiments.keys() - {*TREATMENTS, "replay"}):
        contrasts.extend([(arm, "full"), (arm, "energy_weighted")])
    if {"omit_high", "omit_other"} <= experiments.keys():
        contrasts.append(("omit_high", "omit_other"))
    for requested in args.pair:
        test, sep, control = requested.partition("=")
        if not sep or test == control or not {test, control} <= experiments.keys():
            raise ValueError(f"Unknown or unpaired contrast: {requested}")
        if (test, control) not in contrasts:
            contrasts.append((test, control))
    for panel in sorted({key[0] for key in truth}):
        edges = sorted(edge for p, edge in truth if p == panel)
        draws = np.random.default_rng(20261125).integers(len(edges), size=(10000, len(edges)))
        panels = comparisons[panel] = {}
        arm_metrics[panel] = {
            arm: {metric: float(np.mean([cells[panel, edge, seed, arm][metric]
                                         for edge in edges for seed in SEEDS]))
                  for metric in metrics}
            for arm in experiments
        }
        for test, control in contrasts:
            summary = panels[f"{test}_minus_{control}"] = {}
            for metric in metrics:
                diff = np.array([
                    [cells[panel, edge, seed, test][metric] - cells[panel, edge, seed, control][metric]
                     for edge in edges] for seed in SEEDS
                ])
                summary[metric] = dict(mean=float(diff.mean()), seed_means=diff.mean(axis=1).tolist(),
                                       interval95=np.quantile(diff.mean(axis=0)[draws].mean(axis=1),
                                                              [.025, .975]).tolist())
    out = args.score_output or args.output
    out.mkdir(parents=True, exist_ok=True)
    save_csv(out / f"per_material_{args.until}.csv", results)
    report = dict(step=args.until, seeds=SEEDS, arms=arm_metrics, contrasts=comparisons,
                  implementations={arm: next(iter(h)) for arm, h in implementations.items()},
                  environment=json.loads(environments.pop()),
                  interval="Paired material bootstrap conditional on the two seeds")
    if args.replicate_batch2:
        report.update(cohort=BATCH2_ARM, strength=BATCH2_STRENGTH)
    atomic_json(out / f"summary_{args.until}.json", report)
    print(json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--seed", type=int, choices=SEEDS, default=17)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--arm", action="append", choices=(*TREATMENTS, "replay"))
    p.add_argument("--replicate-batch2", action="store_true",
                   help="Use the locked second eight-path block plus matched replay at strength 0.75")
    p.add_argument("--strength", type=float, default=0.5, help="Force-weight reduction at the peak")
    p.add_argument("--omit", choices=("high", "other"), help="Omit matched interior presentations")
    p.add_argument("--compare", action="append", default=[], metavar="LABEL=DIRECTORY@SOURCE_ARM")
    p.add_argument("--pair", action="append", default=[], metavar="TREATMENT=CONTROL")
    p.add_argument("--score-output", type=Path)
    p.add_argument("--until", type=int, choices=MILESTONES, default=807)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--smoke", type=int, metavar="STEPS")
    modes.add_argument("--project", type=Path, help="Score completed arms against local labels")
    args = p.parse_args()
    if args.replicate_batch2 and (args.omit or args.strength != 0.5 or args.compare):
        p.error("batch 2 replication has its own fixed strength, treatments and controls")
    if not args.replicate_batch2 and args.arm and "replay" in args.arm:
        p.error("replay is an arm of the batch 2 replication")
    if args.project:
        score(args)
        return
    if not 0 <= args.strength < 1:
        p.error("strength must be in [0, 1), preserving nonzero force weights")
    if args.smoke is not None and not 8 <= args.smoke <= 200:
        p.error("smoke must include the first path insertion and use 8 to 200 steps")
    plan, files = load_bundle(args.bundle)
    frames = rows(files["training_frames.csv"])
    if args.replicate_batch2:
        matrix = batch2_experiment(frames, files, args.seed)
        identity = dict(seed=args.seed, bundle=file_sha(args.bundle),
                        implementation=file_sha(Path(__file__)))
        if args.preflight:
            print(json.dumps({arm: details for arm, (_, _, details) in matrix.items()},
                             indent=2, allow_nan=False))
            return
        chosen = args.arm or tuple(matrix)
        # Each arm has one immutable batch schedule and one force-weight map.
        for arm in chosen:
            order, weights, details = matrix[arm]
            args.arm = (arm,)
            train(args, files, frames, plan["training"], order, {arm: weights}, identity | details)
        return
    if args.omit:
        if args.arm or args.strength != 0.5:
            p.error("omission uses original force weights and its own treatment name")
        order, weights, details = omit_frames(frames, files, args.seed, args.omit)
    else:
        order, weights, details = frozen_experiment(frames, files, args.seed, args.strength)
    identity = dict(seed=args.seed, bundle=file_sha(args.bundle),
                    schedule=sha(json.dumps(order).encode()), implementation=file_sha(Path(__file__)),
                    weights=details, strength=None if args.omit else args.strength)
    if args.preflight:
        print(json.dumps(identity, indent=2, allow_nan=False))
        return
    train(args, files, frames, plan["training"], order, weights, identity)


if __name__ == "__main__":
    main()
