"""Extract project SOAP and scalar MACE path features without force gradients."""
import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
from time import perf_counter

from ase.io import read
import numpy as np


D0_SHA = "c0ed6bed7c168f24b0c9392dd2d786c10f209072493858aa04dbcc1ab3c4e69a"


def frames_and_index(geometry):
    files = [geometry / f"{group}_images.xyz" for group in ("historical", "candidate", "reference")]
    frames, index, digests = [], [], {}
    for path in files:
        digests[path.name] = sha256(path.read_bytes()).hexdigest()
        for atoms in read(path, ":", format="extxyz"):
            index.append(dict(cohort=str(atoms.info["cohort"]), edge_id=str(atoms.info["edge_id"]),
                              image=int(atoms.info["image"]), moving=int(atoms.info["moving"]), s=float(atoms.info["s"])))
            atoms.info.clear()
            frames.append(atoms)
    return frames, index, digests


def soap(frames, index, species, workers):
    from concurrent.futures import ThreadPoolExecutor
    from dscribe.descriptors import SOAP

    from phase_a1_descriptors import SOAP_SETTINGS

    options = dict(compression={"mode":"mu1nu1", "species_weighting":None}, **SOAP_SETTINGS)
    def extract(args):
        atoms, meta = args
        # This is the arithmetic average of the atom-resolved power spectra, i.e. outer averaging.
        descriptor = SOAP(species=species, **options)
        out = descriptor.create(atoms)
        li = out[int(meta["moving"])] if meta["cohort"] != "reference" else out[atoms.numbers == 3].mean(axis=0)
        return out.mean(axis=0), li

    with ThreadPoolExecutor(max_workers=workers) as pool:
        whole, local = zip(*pool.map(extract, zip(frames, index), chunksize=1))
    return {"soap_full":np.array(whole), "soap_li":np.array(local)}


def make_activation(checkpoint, device, threads, memory_fraction):
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    import torch

    torch.set_num_threads(threads)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device is unavailable")
        if memory_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
    torch.set_default_dtype(torch.float64)
    if sha256(checkpoint.read_bytes()).hexdigest() != D0_SHA:
        raise ValueError("D0_17 bytes differ from A1")
    from e3nn import o3
    from mace import data
    from mace.calculators import MACECalculator
    from mace.modules.utils import extract_invariant
    from mace.tools import torch_geometric

    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    calc = MACECalculator(models=model, device=device, default_dtype="float64")
    network = calc.models[0].eval()
    layers = int(network.num_interactions)
    channels = o3.Irreps(str(network.products[0].linear.irreps_out))
    n_scalar = channels.dim // (channels.lmax + 1) ** 2
    if (layers, n_scalar) != (2, 128):
        raise ValueError("Unexpected hidden representations for audited A1 model")
    keyspec = data.KeySpecification(info_keys=calc.info_keys, arrays_keys=calc.arrays_keys)
    def graphs(frames):
        return [data.AtomicData.from_config(
                data.config_from_atoms(atoms, key_specification=keyspec, head_name=calc.head),
                z_table=calc.z_table, cutoff=calc.r_max, heads=calc.available_heads,
            ) for atoms in frames]

    def forward(graphs, batch_size):
        loader = torch_geometric.dataloader.DataLoader(
            dataset=graphs, batch_size=batch_size, shuffle=False, drop_last=False)
        atom_features = []
        with torch.no_grad():
            for batch in loader:
                limits = batch.ptr.tolist()
                batch = batch.to(device)
                states = network(batch.to_dict(), compute_force=False, compute_stress=False)["node_feats"]
                scalar = extract_invariant(states, num_layers=layers, num_features=n_scalar, l_max=channels.lmax)
                result = scalar.cpu().numpy()
                atom_features.extend(result[a:b] for a,b in zip(limits, limits[1:]))
        return atom_features

    return calc, graphs, forward, torch


def check_close(a, b):
    if len(a) != len(b):
        raise ValueError("Output frame count differs")
    rel = [float(np.linalg.norm(x-y)/max(np.linalg.norm(x), 1e-12)) for x,y in zip(a,b)]
    if not rel or not np.isfinite(rel).all() or max(rel)>1e-6:
        raise ValueError(f"Feature batch changed output, maximum relative L2 {max(rel, default=None)}")
    return max(rel)


def summarize_feature(values, frames, index):
    whole, local = [], []
    for feat, atoms, meta in zip(values, frames, index):
        whole.append(feat.mean(axis=0))
        local.append(feat[int(meta["moving"])] if meta["cohort"] != "reference"
                     else feat[atoms.numbers == 3].mean(axis=0))
    return {"activation_full":np.array(whole), "activation_li":np.array(local)}


def activation(args, frames, index):
    calc, build_graphs, forward, torch = make_activation(args.checkpoint, args.device, args.threads, args.cap)
    if args.mode == "benchmark":
        selected = np.r_[np.arange(0, 8), np.flatnonzero([m["cohort"]=="candidate" for m in index])[:8]]
        frames = [frames[i] for i in selected]
        index = [index[i] for i in selected]
        start = perf_counter()
        graphs = build_graphs(frames)
        graph_seconds = perf_counter() - start
        warm = forward(graphs[:1], 1)
        legacy = [calc.get_descriptors(atoms=frames[0], invariants_only=True, num_layers=2)]
        check_close(legacy, warm)
        runs = []
        base = None
        for batch in (1,2,4,8):
            if args.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = perf_counter()
            try:
                values = forward(graphs, batch)
            except RuntimeError as exc:
                if args.device != "cuda" or "CUDA out of memory" not in str(exc):
                    raise
                runs.append(dict(device=args.device, batch=batch, images=len(frames),
                                 status="memory_limit", cpu_threads=args.threads))
                torch.cuda.empty_cache()
                continue
            if args.device == "cuda":
                torch.cuda.synchronize()
            elapsed = perf_counter() - start
            base = values if base is None else base
            runs.append(dict(device=args.device, batch=batch, images=len(frames), status="ok", seconds=elapsed,
                             images_per_second=len(frames)/elapsed, graph_build_seconds=graph_seconds,
                             relative_L2_vs_batch1=check_close(base,values), cpu_threads=args.threads,
                             peak_allocated_MiB=(torch.cuda.max_memory_allocated()/2**20 if args.device=="cuda" else None),
                             peak_reserved_MiB=(torch.cuda.max_memory_reserved()/2**20 if args.device=="cuda" else None)))
        return frames, index, summarize_feature(base, frames,index), runs
    graphs = build_graphs(frames)
    values = forward(graphs, args.batch)
    return frames, index, summarize_feature(values, frames,index), []


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("soap","benchmark","activation"))
    p.add_argument("geometry", type=Path)
    p.add_argument("config", type=Path)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--cap", type=float)
    args = p.parse_args()
    frames, index, digests = frames_and_index(args.geometry)
    before = perf_counter()
    if args.mode == "soap":
        species = json.loads(args.config.read_bytes())["species"]
        data = soap(frames,index,species,args.threads)
        bench = []
    else:
        frames, index, data, bench = activation(args,frames,index)
    if any(len(v)!=len(index) or not np.isfinite(v).all() for v in data.values()):
        raise ValueError("Feature matrix has invalid rows")
    args.output.mkdir()
    np.savez_compressed(args.output/"features.npz", **data)
    record = dict(checkpoint_sha256=D0_SHA if args.mode!="soap" else None,
                  image_index=index, geometry_sha256=digests, elapsed=perf_counter()-before,
                  versions={p:version(p) for p in ("ase","numpy", "dscribe") if p!="dscribe" or args.mode=="soap"},
                  benchmark=bench)
    if args.mode!="soap":
        import torch
        record["versions"].update(torch=torch.__version__,mace=version("mace-torch"))
        if args.device=="cuda":
            record["device_name"] = torch.cuda.get_device_name()
    (args.output/"features.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print({name: value.shape for name,value in data.items()}, round(record["elapsed"],3), bench)


if __name__=="__main__":
    main()
