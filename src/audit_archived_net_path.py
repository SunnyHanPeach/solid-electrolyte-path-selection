"""Evaluate archived pure-D0 and complete-path models on the locked blind panel."""
import argparse
import io
import json
import os
from pathlib import Path
import sys
import tarfile
from zipfile import ZipFile

import numpy as np

from audit_signed_barriers import REFERENCE_SHA256, file_sha, rows, sha
from freeze_net_path_panel import csv_bytes
from run_force_weighting import atomic_json, score_profiles


ARMS = ("D0", "P", "Q")
SEEDS = (17, 29, 43)
CASE = "v2.0/pair_1"


def blind_panel(path):
    with ZipFile(path) as archive:
        lock = json.loads(archive.read("lock.json"))
        items = {name: archive.read(name) for name in lock["assets_sha256"]}
    if any(sha(blob) != lock["assets_sha256"][name] for name, blob in items.items()):
        raise ValueError("Changed blind panel input")
    return lock, items


def pack(args):
    history = args.history / "outputs"
    with ZipFile(args.foundation_bundle) as source:
        foundation = source.read("foundation.model")
    recipe = json.loads((history / "MACE_E0_clip_recipe_lock.json").read_bytes())
    if sha(foundation) != recipe["checkpoint_sha256"]:
        raise ValueError("Wrong foundation")
    with ZipFile(history / "MACE_106_panel_reconstruction_deploy.zip") as deploy:
        jobs = {(r["arm"], r["seed"]): r for r in rows(deploy.read("jobs.csv"))
                if r["pair_key"] == CASE}
    d0_archive = history / "MACE_E0_clip_factorial_archive.tar.gz"
    if file_sha(d0_archive) != recipe["evidence_sha256"][d0_archive.name]:
        raise ValueError("Wrong pure D0 archive")
    _, _ = blind_panel(args.panel)
    evidence = {}
    with ZipFile(args.output, "x") as out:
        out.write(args.panel, "panel_blind.zip")
        out.writestr("foundation.model", foundation)
        base_docs, base_hashes = {}, {}
        with tarfile.open(d0_archive, mode="r|gz") as archive:
            for member in archive:
                parts = member.name.split("/")
                if len(parts) != 3 or parts[0] != "runs" or parts[1] not in {f"A-{s}" for s in SEEDS}:
                    continue
                seed = int(parts[1][2:])
                if parts[2] == "result.json":
                    base_docs[seed] = json.load(archive.extractfile(member))
                if parts[2] == "model.pt":
                    blob = archive.extractfile(member).read()
                    base_hashes[seed] = sha(blob)
                    out.writestr(f"D0_{seed}.pt", blob)
        for seed in SEEDS:
            record = base_docs[seed]
            if (record["arm"], record["seed"], record["steps"], record["smoke"],
                record["protocol_sha256"], record["treatment"],
                record["output_sha256"]["model.pt"]) != (
                    "A", seed, 710, False, recipe["protocol_sha256"],
                    {"alignment": False, "clip": 10.0}, base_hashes[seed]):
                raise ValueError(f"Pure D0 archive provenance failed at seed {seed}")
            evidence[f"D0_{seed}.pt"] = dict(sha256=base_hashes[seed],
                                           order=record["order_sha256"], training=record["runtime"])
        with ZipFile(history / "MACE_106_cloud_reconstruction/weights_007-018.zip") as archive:
            for arm in ("P", "Q"):
                for seed in SEEDS:
                    prefix = f"{CASE}/{arm}/seed{seed}"
                    record = json.loads(archive.read(f"{prefix}/result.json"))
                    blob = archive.read(f"{prefix}/weights.pt")
                    job = jobs[arm, str(seed)]
                    if (record["pair_key"], record["arm"], record["seed"],
                        record["protocol_sha256"], record["indices_sha256"], record["order_sha256"],
                        record["files_sha256"]["weights.pt"]) != (
                            CASE, arm, str(seed), job["protocol_sha256"], job["frame_indices_sha256"],
                            job["seed_order_sha256"], sha(blob)):
                        raise ValueError(f"Wrong candidate checkpoint {prefix}")
                    name = f"{arm}_{seed}.pt"
                    evidence[name] = dict(sha256=sha(blob), order=record["order_sha256"],
                                          training=record["environment"])
                    out.writestr(name, blob)
        out.writestr("models.json", json.dumps(dict(
            foundation=sha(foundation), panel=file_sha(args.panel), weights=evidence
        ), indent=2))
    print(f"Locked {len(evidence)} archived checkpoints in {args.output}", flush=True)


def infer(args):
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    import torch
    from ase.io import read
    from mace import data
    from mace.calculators import MACECalculator
    import tempfile

    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    predictions = []
    with ZipFile(args.package) as archive, tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        weights = json.loads(archive.read("models.json"))
        model_path = directory / "foundation.model"
        panel_path = directory / "panel_blind.zip"
        model_path.write_bytes(archive.read("foundation.model"))
        panel_path.write_bytes(archive.read("panel_blind.zip"))
        if file_sha(model_path) != weights["foundation"] or file_sha(panel_path) != weights["panel"]:
            raise ValueError("Changed archived inference inputs")
        _, files = blind_panel(panel_path)
        calc = MACECalculator(model_paths=str(model_path), device=args.device, default_dtype="float32")
        refs = rows(files["frames.csv"])
        atoms = read(io.StringIO(files["blind.xyz"].decode()), index=":", format="extxyz")
        if len(atoms) != len(refs):
            raise ValueError("Wrong blind geometry count")
        specs = data.KeySpecification()
        graphs = {}
        for i, (image, row) in enumerate(zip(atoms, refs)):
            if len(image) != int(row["n_atoms"]) or image.calc or image.info or (
                set(image.arrays) != {"numbers", "positions"}
            ):
                raise ValueError("Evaluation XYZ contains unexpected metadata")
            config = data.config_from_atoms(image, key_specification=specs, head_name=calc.head)
            graphs[i] = data.AtomicData.from_config(
                config, z_table=calc.z_table, cutoff=calc.r_max, heads=calc.available_heads
            )
        for arm in ARMS:
            for seed in SEEDS:
                name = f"{arm}_{seed}.pt"
                raw = archive.read(name)
                if sha(raw) != weights["weights"][name]["sha256"]:
                    raise ValueError(f"Changed checkpoint {name}")
                state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=arm != "D0")
                model = calc.models[0].eval()
                model.load_state_dict(state.state_dict() if arm == "D0" else state, strict=True)
                result = score_profiles(model, graphs, refs, args.device, torch)
                predictions.extend(dict(r, arm=arm, seed=seed) for r in result)
                print(f"{arm}/seed{seed}: {len(result)} images", flush=True)
        csv = csv_bytes(predictions)
        with ZipFile(args.output, "x") as out:
            out.writestr("predictions.csv", csv)
            out.writestr("provenance.json", json.dumps(dict(
                panel=weights["panel"], weights=weights["weights"], prediction=sha(csv),
                device=args.device, torch=str(torch.__version__),
                hardware=torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
            ), indent=2))


def score(args):
    lock, files = blind_panel(args.panel)
    manifest = rows(files["panel.csv"])
    expected = {(r["edge_id"], int(r["frame"])):r["material_id"] for r in rows(files["frames.csv"])}
    with ZipFile(args.predictions) as archive:
        csv = archive.read("predictions.csv")
        provenance = json.loads(archive.read("provenance.json"))
    if sha(csv) != provenance["prediction"] or file_sha(args.panel) != provenance["panel"]:
        raise ValueError("Predictions do not match the frozen panel")
    seen = {}
    for row in rows(csv):
        image = row["edge_id"], int(row["frame"])
        key = row["arm"], int(row["seed"]), *image
        value = float(row["energy_eV"])
        if expected.get(image) != row["material_id"] or key in seen or not np.isfinite(value):
            raise ValueError(f"Invalid prediction {key}")
        seen[key] = value
    if seen.keys() != {(arm, seed, *image) for arm in ARMS for seed in SEEDS for image in expected}:
        raise ValueError("Missing blind predictions")
    sys.path.insert(0, str(args.project / "experiment2"))
    from prepare_data import NEB, metadata, raw_frames

    if file_sha(NEB) != REFERENCE_SHA256:
        raise ValueError("Reference archive changed")
    table = []
    with ZipFile(NEB) as archive:
        for row in manifest:
            edge = row["edge_id"]
            frames = {i: float(metadata(head)[b"energy"]) for i, _, _, head, _ in
                      raw_frames(archive, f"nebDFT2k/{edge}_relaxed.xyz")}
            if set(frames) != {i for e, i in expected if e == edge}:
                raise ValueError("Changed reference geometry coverage")
            barrier = max(frames.values()) - frames[1]
            for arm in ARMS:
                for seed in SEEDS:
                    curve = [seen[arm, seed, edge, i] for i in range(1, len(frames) + 1)]
                    error = 1000 * (max(curve) - curve[0] - barrier)
                    table.append(dict(row, arm=arm, seed=seed, signed_error_meV=error,
                                      abs_error_meV=abs(error)))
    comparisons = {}
    for group, subset in (("all", manifest), ("new_exact_chemistry", [
        r for r in manifest if r["novel_chemistry"] == "1"
    ])):
        edges = {r["edge_id"] for r in subset}
        if len(edges) < 2:
            continue
        errors = {(r["arm"], r["seed"], r["edge_id"]): r["abs_error_meV"]
                  for r in table if r["edge_id"] in edges}
        draws = np.random.default_rng(20261125).integers(len(edges), size=(10000, len(edges)))
        metrics = {}
        for test, control in (("P", "D0"), ("Q", "D0"), ("P", "Q")):
            delta = np.array([[errors[test, seed, edge] - errors[control, seed, edge]
                               for edge in sorted(edges)] for seed in SEEDS])
            mean = delta.mean(axis=0)
            lo, hi = np.quantile(mean[draws].mean(axis=1), [.025, .975])
            loo = (mean.sum() - mean) / (len(mean) - 1)
            seed_means = delta.mean(axis=1).tolist()
            metrics[f"{test}_minus_{control}"] = dict(
                mean_meV=float(mean.mean()), interval95=[float(lo), float(hi)], seed_means=seed_means,
                majority_improved=int(np.sum(mean < 0)), leave_one_out=[float(loo.min()),float(loo.max())],
                confirms=bool(hi < -0.25 and max(seed_means) < 0 and loo.max() < 0),
            )
        comparisons[group] = dict(materials=len(edges), contrasts=metrics)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "per_material.csv").write_bytes(csv_bytes(table))
    report = dict(protocol=lock, provenance=provenance, panels=comparisons)
    atomic_json(args.output / "summary.json", report)
    print(json.dumps(comparisons, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(required=True)
    for name, fn, paths in (
        ("pack", pack, ("history", "foundation_bundle", "panel", "output")),
        ("infer", infer, ("package", "output")),
        ("score", score, ("project", "panel", "predictions", "output")),
    ):
        sub = modes.add_parser(name)
        for key in paths:
            sub.add_argument(key, type=Path)
        if name == "infer":
            sub.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        sub.set_defaults(function=fn)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
