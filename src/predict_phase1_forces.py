"""Evaluate forces from saved paired P0 checkpoints using only blind geometries."""
import argparse
import io
import json
import os
from pathlib import Path
import tempfile
import time
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from audit_signed_barriers import file_sha, rows
from run_ef import build_batch
from run_force_weighting import load_bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "runs", "output"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    import torch
    from ase.io import read
    from mace import data
    from mace.calculators import MACECalculator

    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    _, files = load_bundle(args.bundle)
    bundle_sha = file_sha(args.bundle)
    jobs = {r["seed"]:r["jobs"] for r in json.loads(files["jobs.json"])}
    frames = rows(files["evaluation_frames.csv"])
    atoms = read(io.StringIO(files["evaluation.xyz"].decode()), index=":", format="extxyz")
    if len(atoms) != len(frames):
        raise ValueError("Unexpected evaluation geometry")
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output) as scratch:
        foundation = Path(scratch)/"foundation.model"
        foundation.write_bytes(files["foundation.model"])
        calc = MACECalculator(model_paths=str(foundation), device="cuda", default_dtype="float32")
        model = calc.models[0].eval().requires_grad_(False)
        spec = data.KeySpecification()
        graphs = {}
        for i, (a,r) in enumerate(zip(atoms,frames)):
            if len(a)!=int(r["n_atoms"]) or a.calc or a.info or set(a.arrays)!={"positions","numbers"}:
                raise ValueError("Unexpected evaluation geometry metadata")
            config = data.config_from_atoms(a,key_specification=spec,head_name=calc.head)
            graphs[i] = data.AtomicData.from_config(
                config,z_table=calc.z_table,cutoff=calc.r_max,heads=calc.available_heads
            )
        with ZipFile(args.output/"forces.zip.partial","x",compression=ZIP_DEFLATED,compresslevel=1) as archive:
            hashes = {}
            for phase in ("final","early"):
                for arm in ("path3","replay3","path1","replay1","pure"):
                    for seed in (17,29,43):
                        start=time.monotonic()
                        final = len(jobs[seed][arm])
                        step = final if phase=="final" else 400
                        folder = args.runs/f"seed{seed}"/arm
                        report = json.loads((folder/f"done_{final}.json").read_bytes())
                        weight = folder/f"model_{step}.pt"
                        if report["bundle"]!=bundle_sha or report["arm"]!=arm or report["seed"]!=seed:
                            raise ValueError("Checkpoint identity mismatch")
                        digest=file_sha(weight)
                        if digest!=report["files"][weight.name]:
                            raise ValueError(f"Checkpoint checksum failed: {weight}")
                        model.load_state_dict(torch.load(weight,map_location="cpu",weights_only=True),strict=True)
                        blocks=[]
                        for j in range(0,len(frames),4):
                            batch = build_batch(graphs,list(range(j,min(j+4,len(frames)))),"cuda")
                            with torch.enable_grad():
                                f = model(batch.to_dict(),training=False,compute_force=True)["forces"]
                            blocks.append(f.detach().cpu().numpy())
                        force=np.concatenate(blocks,axis=0)
                        if force.shape != (sum(int(r["n_atoms"]) for r in frames),3) or not np.isfinite(force).all():
                            raise ValueError("Unexpected model force array")
                        name=f"{phase}/{arm}/seed{seed}.npy"
                        stream=io.BytesIO()
                        np.save(stream,force,allow_pickle=False)
                        archive.writestr(name,stream.getvalue())
                        hashes[name]=digest
                        print(f"{name}: {time.monotonic()-start:.1f}s",flush=True)
            archive.writestr("manifest.json",json.dumps(dict(
                bundle=bundle_sha,models=hashes,frame_count=len(frames),
                hardware=torch.cuda.get_device_name(),torch=str(torch.__version__),
            ),indent=2))
    (args.output/"forces.zip.partial").replace(args.output/"forces.zip")
    print("Phase 1 force predictions complete",flush=True)


if __name__=="__main__":
    main()
