"""Infer forces on the 997 frozen geometries using the eight verified final states."""
import argparse
import io
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np

from audit_signed_barriers import SEEDS, file_sha, rows
from run_ef import build_batch
from run_force_weighting import atomic_json, load_bundle
from step200_protocol import BRANCHES


def infer(args):
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from ase.io import read
    from mace import data
    from mace.calculators import MACECalculator
    import torch

    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = args.run
    bundle = root / "frozen_bundle.zip"
    bundle_digest = file_sha(bundle)
    _, files = load_bundle(bundle)
    ids = rows(files["evaluation_frames.csv"])
    structures = read(io.StringIO(files["evaluation.xyz"].decode()), index=":", format="extxyz")
    if len(ids) != 997 or len(structures) != len(ids):
        raise ValueError("Frozen evaluation coverage changed")
    args.output.mkdir()
    foundation = root / "full" / "seed17" / "foundation.model"
    if foundation.read_bytes() != files["foundation.model"]:
        raise ValueError("Frozen foundation model changed")
    calc = MACECalculator(model_paths=str(foundation), device="cuda", default_dtype="float32")
    model = calc.models[0].eval().requires_grad_(False)
    spec = data.KeySpecification()
    graphs = {}
    for i, (atoms, row) in enumerate(zip(structures, ids)):
        if atoms.info or atoms.calc or set(atoms.arrays) != {"positions", "numbers"} or len(atoms) != int(row["n_atoms"]):
            raise ValueError("Invalid blind evaluation geometry")
        config = data.config_from_atoms(atoms, key_specification=spec, head_name=calc.head)
        graphs[i] = data.AtomicData.from_config(config, z_table=calc.z_table,
                                                cutoff=calc.r_max, heads=calc.available_heads)
    saved = dict(line.split()[::-1] for line in (root / "ending_checkpoints.sha256").read_text().splitlines())
    records = []
    for seed in SEEDS:
        for arm in BRANCHES:
            start = monotonic()
            relative = f"full/seed{seed}/{arm}/resume.pt"
            digest = file_sha(root / relative)
            if saved[relative] != digest:
                raise ValueError(f"Unexpected final checkpoint hash: {relative}")
            state = torch.load(root / relative, map_location="cpu", weights_only=True)
            who = state["identity"]
            if (state["step"], who["seed"], who["arm"], who["bundle"], who["smoke"]) != (
                    400, seed, arm, bundle_digest, False):
                raise ValueError(f"Unexpected checkpoint identity: {relative}")
            model.load_state_dict(state["model"], strict=True)
            blocks, energies = [], []
            torch.cuda.reset_peak_memory_stats()
            with torch.enable_grad():
                for first in range(0, len(graphs), args.batch_size):
                    indices = range(first, min(first + args.batch_size, len(graphs)))
                    batch = build_batch(graphs, indices, "cuda")
                    pred = model(batch.to_dict(), training=False, compute_force=True)
                    blocks.append(pred["forces"].detach().cpu().numpy())
                    energies.extend(pred["energy"].detach().cpu().tolist())
            forces = np.concatenate(blocks, axis=0)
            if forces.shape != (sum(len(a) for a in structures), 3) or not np.isfinite(forces).all():
                raise ValueError(f"Invalid forces: {relative}")
            archive_prediction = root / relative.replace("resume.pt", "predictions_400.csv")
            if file_sha(archive_prediction) != state["predictions"][400]:
                raise ValueError(f"Unexpected archived predictions: {relative}")
            previous = np.asarray([float(row["energy_eV"]) for row in rows(archive_prediction.read_bytes())])
            if len(energies) != len(previous) or not np.isfinite(energies).all():
                raise ValueError(f"Invalid energies: {relative}")
            name = f"seed{seed}-{arm}.npy"
            out = args.output / name
            with out.open("xb") as f:
                np.save(f, forces, allow_pickle=False)
            record = dict(seed=seed, arm=arm, frames=len(ids), atoms=len(forces), file=name,
                          file_sha256=file_sha(out), checkpoint_sha256=digest,
                          max_energy_difference_eV=float(np.max(np.abs(np.asarray(energies) - previous))),
                          seconds=monotonic() - start, peak_memory_bytes=torch.cuda.max_memory_allocated())
            records.append(record)
            atomic_json(args.output / "manifest.json", dict(
                bundle_sha256=bundle_digest, batch_size=args.batch_size, torch=str(torch.__version__),
                hardware=torch.cuda.get_device_name(), source_sha256=file_sha(Path(__file__)), models=records))
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    infer(parser.parse_args())
