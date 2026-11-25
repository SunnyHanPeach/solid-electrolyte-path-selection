"""Replicate the confirmed Phase 1 case on new materials using archived weights."""
import argparse
from collections import defaultdict
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from audit_archived_net_path import blind_panel
from audit_signed_barriers import REFERENCE_SHA256, file_sha, rows, sha
from freeze_net_path_panel import csv_bytes
from run_force_weighting import atomic_json, load_bundle, score_profiles

SEEDS = (17, 29, 43, 53, 67)
ARMS = ("pure", "replay3", "path3")


def checkpoint(runs, seed, arm, bundle_hash):
    root = runs / ("results" if seed in SEEDS[:3] else "confirm")
    if seed in SEEDS[:3] and not root.is_dir():
        root = runs
    step = 710 if arm == "pure" else 735
    folder = root / f"seed{seed}" / arm
    record = json.loads((folder / f"done_{step}.json").read_bytes())
    if (record["arm"], record["seed"], record["bundle"], record["optimizer_steps"]) != (
        arm, seed, bundle_hash, step
    ):
        raise ValueError(f"Checkpoint identity failed: {folder}")
    name = f"model_{step}.pt"
    return folder / name, record["files"][name]


def predict(args):
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    import torch
    from ase.io import read
    from mace import data
    from mace.calculators import MACECalculator

    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    _, files = load_bundle(args.bundle)
    bundle_hash = file_sha(args.bundle)
    _, panel = blind_panel(args.panel)
    refs = rows(panel["frames.csv"])
    atoms = read(io.StringIO(panel["blind.xyz"].decode()), index=":", format="extxyz")
    if len(atoms) != len(refs):
        raise ValueError("Wrong panel shape")
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        foundation = Path(tmp) / "foundation.model"
        foundation.write_bytes(files["foundation.model"])
        calc = MACECalculator(model_paths=str(foundation), device="cuda", default_dtype="float32")
        model = calc.models[0].eval().requires_grad_(False)
        spec = data.KeySpecification()
        graphs = {}
        for i, (a, r) in enumerate(zip(atoms, refs)):
            if len(a) != int(r["n_atoms"]) or a.calc or a.info or set(a.arrays) != {"positions", "numbers"}:
                raise ValueError("Evaluation geometry has unexpected metadata")
            config = data.config_from_atoms(a, key_specification=spec, head_name=calc.head)
            graphs[i] = data.AtomicData.from_config(
                config, z_table=calc.z_table, cutoff=calc.r_max, heads=calc.available_heads)
        evidence, predictions = {}, []
        for seed in SEEDS:
            for arm in ARMS:
                weight, digest = checkpoint(args.runs, seed, arm, bundle_hash)
                if file_sha(weight) != digest:
                    raise ValueError(f"Weight checksum failed: {weight}")
                model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True), strict=True)
                values = score_profiles(model, graphs, refs, "cuda", torch)
                predictions.extend(dict(r, arm=arm, seed=seed) for r in values)
                evidence[f"{seed}/{arm}"] = digest
                print(f"{seed}/{arm}: {len(values)} predictions", flush=True)
        raw = csv_bytes(predictions)
        with ZipFile(args.output, "x", compression=ZIP_DEFLATED) as archive:
            archive.writestr("predictions.csv", raw)
            archive.writestr("proof.json", json.dumps(dict(
                bundle=bundle_hash, panel=file_sha(args.panel), weights=evidence,
                predictions=sha(raw), hardware=torch.cuda.get_device_name(), torch=str(torch.__version__),
            ), indent=2))


def score(args):
    sys.path.insert(0, str(args.project / "experiment2"))
    from prepare_data import NEB, metadata, raw_frames

    lock, panel = blind_panel(args.panel)
    bundle_hash = file_sha(args.bundle)
    known = {(r["edge_id"], int(r["frame"])): r["material_id"] for r in rows(panel["frames.csv"])}
    with ZipFile(args.predictions) as archive:
        raw = archive.read("predictions.csv")
        proof = json.loads(archive.read("proof.json"))
    if (proof["bundle"], proof["panel"], proof["predictions"]) != (
        bundle_hash, file_sha(args.panel), sha(raw)
    ):
        raise ValueError("Predictions differ from their frozen sources")
    expected_weights = {f"{seed}/{arm}": checkpoint(args.runs, seed, arm, bundle_hash)[1]
                        for seed in SEEDS for arm in ARMS}
    if proof["weights"] != expected_weights:
        raise ValueError("Predictions differ from the previous training checkpoints")
    curves = defaultdict(dict)
    for r in rows(raw):
        edge, i, arm, seed = r["edge_id"], int(r["frame"]), r["arm"], int(r["seed"])
        value = float(r["energy_eV"])
        c = curves[edge, seed, arm]
        if r["material_id"] != known.get((edge, i)) or i in c or not np.isfinite(value):
            raise ValueError("Invalid prediction")
        c[i] = value
    materials = rows(panel["panel.csv"])
    edges = [r["edge_id"] for r in materials]
    if len(set(edges)) != len(edges) or set(curves) != {(e,s,a) for e in edges for s in SEEDS for a in ARMS}:
        raise ValueError("Missing models or materials")

    if file_sha(NEB) != REFERENCE_SHA256:
        raise ValueError("DFT archive changed")
    errors = {}
    with ZipFile(NEB) as archive:
        for edge in edges:
            ref = {i: float(metadata(head)[b"energy"]) for i, _, _, head, _ in
                   raw_frames(archive, f"nebDFT2k/{edge}_relaxed.xyz")}
            if set(ref) != {i for e, i in known if e == edge}:
                raise ValueError("Reference images differ")
            truth = max(ref.values()) - ref[1]
            for seed in SEEDS:
                for arm in ARMS:
                    c = curves[edge, seed, arm]
                    if c.keys() != ref.keys():
                        raise ValueError("Incomplete predicted curve")
                    errors[edge, seed, arm] = 1000 * abs(max(c.values()) - c[1] - truth)

    systems = sorted({r["chemsys"] for r in materials})
    clusters = [systems.index(r["chemsys"]) for r in materials]
    w = np.random.default_rng(20261125).multinomial(
        len(systems), np.full(len(systems), 1/len(systems)), size=10000)[:, clusters]
    outcomes, table = {}, []
    for name, control in (("net", "replay3"), ("absolute", "pure")):
        delta = np.asarray([[errors[e,s,control]-errors[e,s,"path3"] for e in edges] for s in SEEDS])
        avg = delta.mean(axis=0)
        lo, hi = np.quantile((w @ avg)/w.sum(axis=1), [.025, .975])
        seeds = delta.mean(axis=1)
        loo = (avg.sum()-avg)/(len(avg)-1)
        outcomes[name] = dict(mean_meV=float(avg.mean()), interval95_meV=[float(lo), float(hi)],
            seed_means_meV=seeds.tolist(), favorable_materials=int((avg > 0).sum()),
            leave_one_out_meV=[float(loo.min()), float(loo.max())],
            practical_positive=bool(lo > .25 and seeds.min() > 0 and loo.min() > 0))
        for i, seed in enumerate(SEEDS):
            table.extend(dict(r, seed=seed, metric=name, gain_meV=float(x))
                         for r, x in zip(materials, delta[i]))
    args.output.mkdir(exist_ok=True)
    atomic_json(args.output / "summary.json", dict(lock=lock, proof=proof,
                materials=len(materials), exact_chemistries=len(systems), results=outcomes))
    (args.output / "paired_materials.csv").write_bytes(csv_bytes(table))
    print(json.dumps(outcomes,indent=2),flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(required=True)
    for mode, action, paths in (
        ("predict",predict,("bundle","panel","runs","output")),
        ("score",score,("project","bundle","panel","runs","predictions","output")),
    ):
        cmd = sub.add_parser(mode)
        for name in paths:
            cmd.add_argument(name,type=Path)
        cmd.set_defaults(function=action)
    a=p.parse_args()
    a.function(a)
