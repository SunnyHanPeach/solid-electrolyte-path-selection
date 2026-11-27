"""Score the protected eight-path trajectory's fixed-geometry forces on CPU."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import io
import json
import os
from pathlib import Path
from tarfile import open as open_tar
from time import monotonic

import numpy as np

from audit_project_data import write_csv, write_json
from audit_signed_barriers import ARMS, SEEDS, file_sha, rows, sha
from run_force_weighting import load_bundle
from score_phase1_forces import references


STEPS = (8, 200, 400)


def infer(task):
    bundle, archive, output, needed, graphs_threads, frames_subset, models = task
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    import ase
    from ase.io import read
    import mace
    from mace import data
    from mace.calculators import MACECalculator
    import torch
    from run_ef import build_batch

    torch.set_num_threads(graphs_threads)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    _, files = load_bundle(bundle)
    check_bundle = file_sha(bundle)
    schedules = {row["seed"]: row["jobs"] for row in json.loads(files["jobs.json"])}
    frames = rows(files["evaluation_frames.csv"])
    atoms = read(io.StringIO(files["evaluation.xyz"].decode()), index=":", format="extxyz")
    if len(frames) != 997 or len(atoms) != len(frames):
        raise ValueError("Expected the frozen 997-frame blind bundle")
    calc = MACECalculator(model_paths=str(output / "foundation.model"), device="cpu", default_dtype="float32")
    model = calc.models[0].eval().requires_grad_(False)
    spec = data.KeySpecification()
    graphs = {}
    for i in frames_subset:
        a = atoms[i]
        if a.info or a.calc or set(a.arrays) != {"positions", "numbers"} or len(a) != int(frames[i]["n_atoms"]):
            raise ValueError("Blind evaluation input did not retain its original format")
        config = data.config_from_atoms(a, key_specification=spec, head_name=calc.head)
        graphs[i] = data.AtomicData.from_config(config, z_table=calc.z_table, cutoff=calc.r_max, heads=calc.available_heads)
    records = []
    with open_tar(archive, "r:") as source:
        for step, seed, arm in models:
            t0 = monotonic()
            stem = f"results/seed{seed}/{arm}/step{step:04d}"
            marker = json.loads(source.extractfile(f"{stem}/result.json").read())
            checkpoint = source.extractfile(f"{stem}/checkpoint.pt").read()
            digest = sha(checkpoint)
            if ((marker["step"], marker["seed"], marker["arm"], marker["bundle_sha256"],
                 marker["schedule_sha256"], marker["files"]["checkpoint.pt"]) !=
                    (step, seed, arm, check_bundle, sha(json.dumps(schedules[seed][arm]).encode()), digest)
                    or needed[step, seed, arm] != digest):
                raise ValueError(f"Checkpoint or schedule identity mismatch: {stem}")
            state = torch.load(io.BytesIO(checkpoint), map_location="cpu", weights_only=True)
            model.load_state_dict(state["model"], strict=True)
            blocks = []
            with torch.enable_grad():
                for start in range(0, len(frames_subset), 4):
                    ids = frames_subset[start:start+4]
                    batch = build_batch(graphs, ids, "cpu")
                    values = model(batch.to_dict(), training=False, compute_force=True)["forces"]
                    blocks.append(values.detach().cpu().numpy())
            force = np.concatenate(blocks, axis=0)
            if force.shape != (sum(int(frames[i]["n_atoms"]) for i in frames_subset), 3) or not np.isfinite(force).all():
                raise ValueError(f"Invalid force output: {stem}")
            name = f"step{step:04d}-seed{seed}-{arm}.npy"
            with (output / name).open("xb") as handle:
                np.save(handle, force, allow_pickle=False)
            record = dict(step=step, seed=seed, arm=arm, checkpoint_sha256=digest, file=name,
                          file_sha256=file_sha(output / name), seconds=monotonic() - t0, frames=len(frames_subset),
                          torch=str(torch.__version__), mace=mace.__version__, ase=ase.__version__, threads=graphs_threads)
            print(json.dumps(record), flush=True)
            records.append(record)
    return records


def summarize(args, files, predicted):
    frames = rows(files["evaluation_frames.csv"])
    truth, offsets, paths, _, excluded = references(args, files)
    panel = {row["edge_id"]: row["panel"] for row in frames}
    by_model = {(r["step"], r["seed"], r["arm"]): r for r in predicted}
    paired = defaultdict(dict)
    for step in STEPS:
        for seed in SEEDS:
            errors = []
            for arm in ARMS:
                metadata = by_model[step, seed, arm]
                arr = np.load(args.output / metadata["file"], allow_pickle=False)
                errors.append(np.square(arr - truth))
            for edge, (li, regions) in paths.items():
                for region, indices in regions.items():
                    if not indices:
                        continue
                    for target in ("all_atoms", "moving_Li"):
                        if target == "moving_Li" and li is None:
                            continue
                        vals = [np.sqrt(np.mean([err[offsets[i]:offsets[i+1]] if target == "all_atoms"
                                                 else err[offsets[i] + li] for i in indices])) for err in errors]
                        paired[step, panel[edge], edge, region, target][seed] = vals
    by_group = defaultdict(list)
    paired_rows = []
    for (step, panel_name, edge, region, target), cell in sorted(paired.items()):
        delta = [float(cell[s][1] - cell[s][0]) for s in SEEDS]
        by_group[step, panel_name, region, target].append(delta)
        for seed, diff in zip(SEEDS, delta):
            paired_rows.append(dict(step=step, panel=panel_name, edge_id=edge, region=region, target=target,
                                    seed=seed, rmse_replay_eV_A=cell[seed][0], rmse_augmented_eV_A=cell[seed][1],
                                    augmented_minus_replay_eV_A=diff))
    rng = np.random.default_rng(20261127)
    summaries = []
    for group, vals in sorted(by_group.items()):
        step, panel_name, region, target = group
        arr = np.asarray(vals)
        means = arr.mean(axis=1)
        draws = rng.integers(len(arr), size=(10000, len(arr)))
        lo, hi = np.quantile(means[draws].mean(axis=1), [.025, .975])
        summaries.append(dict(step=step, panel=panel_name, region=region, target=target, materials=len(arr),
                              mean=float(means.mean()), lo95=float(lo), hi95=float(hi),
                              seed17=float(arr[:, 0].mean()), seed29=float(arr[:, 1].mean())))
    write_csv(args.output / "paired_force_rmse.csv", paired_rows)
    write_csv(args.output / "force_rmse_summary.csv", summaries)
    write_json(args.output / "inference_manifest.json", dict(bundle_sha256=file_sha(args.bundle), frames=len(frames),
               excluded_for_moving_li=excluded, models=predicted, device="cpu"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "archive", "verification", "project", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    _, files = load_bundle(args.bundle)
    frozen = json.loads(args.verification.read_bytes())
    needed = {(m["step"], m["seed"], m["arm"]): m["sha256"] for m in frozen["records"]}
    models = [(step, seed, arm) for step in STEPS for seed in SEEDS for arm in ARMS]
    subset = list(range(len(rows(files["evaluation_frames.csv"]))))
    if args.smoke:
        models = [(200, 17, arm) for arm in ARMS]
        subset = subset[:5]
    args.output.mkdir()
    (args.output / "foundation.model").write_bytes(files["foundation.model"])
    n = 1 if args.smoke else min(args.workers, len(models))
    tasks = [(args.bundle, args.archive, args.output, needed, args.threads, subset, models[i::n]) for i in range(n)]
    if args.smoke:
        for task in tasks:
            infer(task)
    else:
        if len(needed) != 24:
            raise ValueError("Missing verified checkpoint metadata")
        with ProcessPoolExecutor(max_workers=n) as pool:
            records = [item for group in pool.map(infer, tasks) for item in group]
        summarize(args, files, sorted(records, key=lambda m: (m["step"], m["seed"], m["arm"])))


if __name__ == "__main__":
    main()
