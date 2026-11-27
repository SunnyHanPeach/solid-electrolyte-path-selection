"""Compare archived B2 mp-753143 training arms on the public fixed-geometry panel."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import gzip
import io
from importlib.metadata import version
import json
from pathlib import Path
import tarfile
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from ase import Atoms
from ase.io import read
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

from audit_phase_b0 import FOUNDATION
from audit_phase_h2_evaluation import MATCHER, digest, framework, rows
from audit_public_neb_panel import geometry_hash
from audit_public_neb_reuse import FP_HASH, checked
from phase_h1_shortlist import anonymous_formula, write
from run_force_weighting import score_profiles
from validate_phase1_materials import SEEDS, checkpoint

ARMS = ("pure", "replay1", "path1")
EDGE = "mp-753143_5_13_0_0_0"
GATE_MEV = 0.25
PAIR_REPRESENTATIVE = "127400|8"
PAIR_SECOND = "147920|10"


def verified_member(archive, plan, name):
    raw = archive.read(name)
    if sha256(raw).hexdigest() != plan["sha256"][name]:
        raise ValueError(f"Training bundle member changed: {name}")
    return raw


def compare_training_framework(task):
    key, host, comparisons = task
    matcher = StructureMatcher(**MATCHER)
    hits = [name for name, ref in comparisons if matcher.fit_anonymous(host, ref)]
    return dict(path_id=key, comparisons=len(comparisons),
                training_framework_matches=len(hits), matches="|".join(hits), scored=not hits)


def training_scope(args, source, panel):
    b2_plan_hash = json.loads((args.b2 / "inference_manifest.json").read_bytes())["matrix_sha256"]
    if digest(args.matrix) != b2_plan_hash:
        raise ValueError("B2 matrix identity differs")
    with ZipFile(args.matrix) as matrix:
        plan = json.loads(matrix.read("plan.json"))
        if (plan["candidates"][0]["edge_id"], plan["foundation_sha256"]) != (EDGE, FOUNDATION):
            raise ValueError("B2 path identity changed")
        if digest(args.base) != plan["base_bundle_sha256"]:
            raise ValueError("Wrong B2 base training bundle")
        extra_frames = read(io.StringIO(verified_member(matrix, plan, "added.xyz").decode()),
                            ":", format="extxyz")
        extra_keys = verified_member(matrix, plan, "added_frames.csv").decode().splitlines()
        orders = json.loads(verified_member(matrix, plan, "orders.json"))
    import csv

    keys = list(csv.DictReader(extra_keys))
    if len(keys) != len(extra_frames):
        raise ValueError("Added-frame identity table differs")
    added = [frame for frame, row in zip(extra_frames, keys) if row["edge_id"] == EDGE]
    with ZipFile(args.base) as base:
        base_plan = json.loads(base.read("plan.json"))
        d0 = read(io.StringIO(verified_member(base, base_plan, "training.xyz").decode()),
                  "0:278", format="extxyz")
        foundation = verified_member(base, base_plan, "foundation.model")
    if (len(d0), len(added), sha256(foundation).hexdigest()) != (278, 5, FOUNDATION):
        raise ValueError("Unexpected B2 training geometry or initialization")

    groups = defaultdict(list)
    for prefix, frames in (("pure", d0), ("added", added)):
        for index, frame in enumerate(frames):
            host = framework(Structure(frame.cell.array, frame.get_chemical_symbols(), frame.positions,
                                       coords_are_cartesian=True))
            groups[anonymous_formula(host)].append((f"{prefix}-{index:03d}", host))
    tasks = []
    for row in panel:
        entry = source["pathways"][row["path_id"]]
        start = Structure.from_dict(entry["full_fp_neb_input"]["source_start_endpoint_structure"])
        host = framework(start)
        tasks.append((row["path_id"], host, groups[anonymous_formula(host)]))
    print(f"Checking {sum(len(task[2]) for task in tasks)} training-framework pairs.", flush=True)
    with ProcessPoolExecutor(max_workers=max(2, args.workers * 2)) as pool:
        overlap = list(pool.map(compare_training_framework, tasks, chunksize=1))
    return foundation, overlap, plan, orders


def weights_for(args, plan, orders):
    expected = {(r["arm"], int(r["seed"])): r for r in rows(args.b2 / "checkpoint_hashes.csv")}
    locations = {r["sha256"]: r for r in rows(args.recovery / "checkpoint_locations.csv")}
    expected_base = plan["base_bundle_sha256"]
    accepted_matrices = {digest(args.matrix), plan["earlier_matrix_sha256"],
                         *plan["compatible_matrix_sha256_slots"]}
    evidence, models = [], {}
    with (tarfile.open(args.weights / "b2_verified_pure.tar") as pure,
          tarfile.open(args.weights / "b2_verified_weights.tar") as b2):
        for seed in SEEDS:
            pair = []
            for arm in ARMS:
                row = expected[arm, seed]
                expected_sha = row["checkpoint_sha256"]
                location = locations[expected_sha]
                is_pure = arm == "pure"
                archive = pure if is_pure else b2
                if Path(archive.name).name != location["local_source"]:
                    raise ValueError("Wrong weight archive")
                if is_pure:
                    _, historical_sha = checkpoint(args.base.parent, seed, arm, expected_base)
                    if historical_sha != expected_sha:
                        raise ValueError("Pure D0 checkpoint differs from Phase 1")
                    schedule = None
                    training_matrix = None
                else:
                    stamp = json.loads((args.b2 / f"fits/{arm}/seed{seed}/done.json").read_bytes())
                    if ((stamp["arm"], stamp["seed"], stamp["frozen_steps"], stamp["steps"],
                         stamp["source_base_sha256"], stamp["files"]["model.pt"],
                         stamp["environment"]["gpu"]) !=
                        (arm, seed, 735, 735, expected_base, expected_sha, row["training_hardware"])
                        or stamp["matrix_sha256"] not in accepted_matrices):
                        raise ValueError(f"Mismatched training identity for {seed}/{arm}")
                    schedule = sha256(json.dumps(orders[str(seed)][arm]).encode()).hexdigest()
                    if stamp["schedule_sha256"] != schedule:
                        raise ValueError(f"Mismatched frozen schedule for {seed}/{arm}")
                    training_matrix = stamp["matrix_sha256"]
                    pair.append(stamp["environment"])
                with archive.extractfile(location["archive_member"]) as stream:
                    weights = stream.read()
                if sha256(weights).hexdigest() != expected_sha:
                    raise ValueError(f"Corrupt checked-in model identity for {seed}/{arm}")
                models[seed, arm] = weights
                evidence.append(dict(seed=seed, arm=arm, steps=710 if is_pure else 735,
                                     checkpoint_sha256=expected_sha, schedule_sha256=schedule,
                                     training_matrix_sha256=training_matrix,
                                     training_hardware=row["training_hardware"],
                                     source_archive=Path(archive.name).name,
                                     archive_member=location["archive_member"]))
            if pair[0] != pair[1]:
                raise ValueError(f"Path and Replay differ in provenance for seed {seed}")
    return models, evidence


def initialize_worker(foundation, geometries, threads):
    import torch
    from mace import data
    from mace.calculators import MACECalculator

    global MODEL, GRAPHS, FRAMES
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    model = torch.load(io.BytesIO(foundation), map_location="cpu", weights_only=False)
    calc = MACECalculator(models=model, device="cpu", default_dtype="float64")
    MODEL = calc.models[0].eval().requires_grad_(False)
    spec = data.KeySpecification()
    GRAPHS, FRAMES = {}, []
    for index, (path, frame, symbols, positions, cell) in enumerate(geometries):
        atoms = Atoms(symbols, positions=positions, cell=cell, pbc=True)
        config = data.config_from_atoms(atoms, key_specification=spec, head_name=calc.head)
        GRAPHS[index] = data.AtomicData.from_config(
            config, z_table=calc.z_table, cutoff=calc.r_max, heads=calc.available_heads)
        FRAMES.append(dict(path_id=path, image=frame))


def score_seed(job):
    import torch

    seed, models = job
    scored = []
    started = perf_counter()
    for arm, raw in models:
        MODEL.load_state_dict(torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True),
                              strict=True)
        pred = score_profiles(MODEL, GRAPHS, FRAMES, "cpu", torch)
        smoke_refs = FRAMES[:7]
        first = score_profiles(MODEL, GRAPHS, smoke_refs, "cpu", torch)
        np.testing.assert_allclose([row["energy_eV"] for row in pred[:7]],
                                   [row["energy_eV"] for row in first], atol=1e-8, rtol=0)
        scored.extend(dict(row, seed=seed, arm=arm) for row in pred)
        print(f"Seed {seed} {arm}: {len(pred)} images, {perf_counter()-started:.1f} s since seed start",
              flush=True)
    return scored


def bootstrap_gains(gains, panel, omit=()):
    selected = [r for r in panel if r["path_id"] not in omit]
    family_ids = sorted({r["screened_framework_family"] for r in selected})
    paths = defaultdict(list)
    for row in selected:
        paths[row["screened_framework_family"]].append(row["path_id"])
    matrix = np.asarray([[np.mean([gains[seed, path] for path in paths[family]])
                          for family in family_ids] for seed in SEEDS])
    per_seed = matrix.mean(axis=1)
    family_mean = matrix.mean(axis=0)
    counts = np.random.default_rng(20261127).multinomial(
        len(family_ids), np.full(len(family_ids), 1/len(family_ids)), size=20000)
    lo, hi = np.quantile(counts @ family_mean / len(family_ids), [.025, .975])
    loo = (family_mean.sum() - family_mean) / (len(family_mean) - 1)
    return dict(mean_meV=float(family_mean.mean()), interval95_meV=[float(lo), float(hi)],
                seed_means_meV=dict(zip(SEEDS, per_seed.tolist())),
                favorable_families=int((family_mean > 0).sum()),
                favorable_paths=int(sum(np.mean([gains[seed, r["path_id"]] for seed in SEEDS]) > 0
                                        for r in selected)),
                leave_one_family_out_meV=[float(loo.min()), float(loo.max())],
                practical_positive=bool(lo > GATE_MEV and per_seed.min() > 0 and loo.min() > 0))


def summarize(predictions, truth, panel):
    curves = defaultdict(dict)
    for r in predictions:
        key = (r["seed"], r["path_id"], r["arm"])
        if r["image"] in curves[key]:
            raise ValueError(f"Repeated image: {key}")
        curves[key][r["image"]] = r["energy_eV"]
    scored, family_maes = [], defaultdict(list)
    paired = {"net": {}, "absolute": {}}
    for seed in SEEDS:
        for row in panel:
            key, family = row["path_id"], row["screened_framework_family"]
            target, errors = truth[key], {}
            for arm in ARMS:
                curve = curves[seed, key, arm]
                if set(curve) != set(range(7)):
                    raise ValueError(f"Incomplete scored curve: {seed}/{key}/{arm}")
                barrier = max(curve.values()) - curve[0]
                errors[arm] = abs(barrier-target) * 1000
                family_maes[seed, arm, family].append(errors[arm])
                scored.append(dict(path_id=key, icsd_id=row["icsd_id"], family=family,
                                   seed=seed, arm=arm, forward_barrier_eV=barrier,
                                   reference_eV=target, absolute_error_meV=errors[arm]))
            paired["net"][seed, key] = errors["replay1"] - errors["path1"]
            paired["absolute"][seed, key] = errors["pure"] - errors["path1"]
    results = {name: bootstrap_gains(gains, panel) for name, gains in paired.items()}
    if {PAIR_REPRESENTATIVE, PAIR_SECOND}.issubset(truth):
        sensitivities = {name: bootstrap_gains(gains, panel, (PAIR_SECOND,))
                         for name, gains in paired.items()}
    else:
        sensitivities = {}
    arm_mae = {arm: float(np.mean([np.mean(values) for (_, a, _), values in family_maes.items()
                                  if a == arm])) for arm in ARMS}
    return scored, results, sensitivities, arm_mae


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("reference", "panel", "base", "matrix", "b2", "recovery", "weights", "output"):
        p.add_argument(key, type=Path)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=3)
    args = p.parse_args()
    started = perf_counter()
    args.output.mkdir(parents=True)
    source = json.loads(gzip.decompress(checked(args.reference, FP_HASH)))
    panel_sha = digest(args.panel)
    baseline = json.loads((args.panel.parent / "d0_seed17/scoring_manifest.json").read_bytes())
    if panel_sha != baseline["panel_sha256"]:
        raise ValueError("Panel membership differs from previous fixed-geometry scoring")
    panel = [row for row in rows(args.panel) if row["use_for_external_scoring"] == "True"]
    foundation, overlap, plan, orders = training_scope(args, source, panel)
    write(args.output / "training_overlap.csv", overlap)
    excluded = [r["path_id"] for r in overlap if not r["scored"]]
    panel = [row for row in panel if row["path_id"] not in excluded]
    models, evidence = weights_for(args, plan, orders)
    write(args.output / "checkpoint_identity.csv", evidence)
    geometries, truth = [], {}
    for row in panel:
        key = row["path_id"]
        images = source["pathways"][key]["dft_neb_reference"]["images"]
        if len(images) != 7 or geometry_hash(images) != row["ordered_DFT_geometry_sha256"]:
            raise ValueError(f"Invalid public evaluation geometries: {key}")
        energies = []
        for index, image in enumerate(images):
            if image["image_index"] != index:
                raise ValueError(f"Unordered public geometry: {key}")
            s = Structure.from_dict(image["structure"])
            geometries.append((key, index, [site.specie.symbol for site in s], s.cart_coords, s.lattice.matrix))
            energies.append(float(image["energy_total_eV"]))
        truth[key] = max(energies) - energies[0]
    payload = [(seed, [(arm, models[seed, arm]) for arm in ARMS]) for seed in SEEDS]
    print(f"Verified {len(evidence)} checkpoints. Scoring {len(panel)} paths with {args.workers} processes.",
          flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                             initargs=(foundation, geometries, args.threads)) as pool:
        predictions = [r for batch in pool.map(score_seed, payload) for r in batch]
    scored, outcome, sens, arm_mae = summarize(predictions, truth, panel)
    write(args.output / "image_predictions.csv", predictions)
    write(args.output / "path_scores.csv", scored)
    summary = dict(
        candidate=EDGE, seeds=list(SEEDS), n_paths=len(panel),
        n_families=len({r["screened_framework_family"] for r in panel}),
        images_per_model=len(geometries), excluded_for_training_framework_overlap=excluded,
        reference_sha256=FP_HASH, panel_sha256=panel_sha, base_sha256=plan["base_bundle_sha256"],
        matrix_sha256=digest(args.matrix), script_sha256=digest(Path(__file__)),
        source_checkpoint_ledger_sha256=digest(args.b2 / "checkpoint_hashes.csv"),
        cpu_workers=args.workers, cpu_threads_per_worker=args.threads, inference_dtype="float64",
        practical_margin_meV=GATE_MEV, family_macro_MAE_meV=arm_mae, gains=outcome,
        near_duplicate_sensitivity=sens, elapsed_seconds=round(perf_counter()-started, 2),
        software={name: version(name) for name in ("mace-torch", "torch", "ase", "pymatgen")},
    )
    data = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    (args.output / "summary.json").write_text(data, encoding="utf-8", newline="\n")
    print(data, flush=True)


if __name__ == "__main__":
    main()
