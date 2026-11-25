"""Run only the three verified D0 models on Phase A0's initial Li geometries."""
import argparse
from hashlib import sha256
import io
from importlib.metadata import version
from itertools import product
import json
import os
from pathlib import Path
import platform
import subprocess
import tarfile
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from ase import Atoms

from audit_phase_b0 import FOUNDATION, SEEDS, checkpoint_type, frame_prefix
from phase_a0_geometry import (
    EXPECTED, PATHS, csv_rows, exact_sha, export_csv, parse_path, reaction_coordinate,
    validate_sequence,
)

D0_SHA = "dda468e74c0dbd46c3d5b07121b900184a917dd319415e826af6e25b56ee7081"
MODEL_CLASS = "mace.modules.models.ScaleShiftMACE"
MODEL_SETTINGS = dict(seeds=list(SEEDS), epochs=5, batch_size=2, lr=.0001,
                      energy_weight=1., forces_weight=100., clip_norm=10.)
REPEAT_ATOL = 1e-9
PERIODIC_ATOL = 1e-8


def file_sha(path):
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def rows(path):
    return csv_rows(path.read_bytes())


def strict_check(condition, message):
    if not condition:
        raise ValueError(message)


def frozen_models(args):
    inv = {r["checkpoint_id"]: r for r in rows(args.repo/"results/phase_b0/checkpoint_inventory.csv")}
    prov = {r["checkpoint_id"]: r for r in rows(args.repo/"results/phase_b0/checkpoint_provenance.csv")}
    with ZipFile(args.protocol) as z:
        protocol = json.loads(z.read("protocol.json"))["protocol"]
        protocol_sha = z.read("protocol.sha256").decode().split()[0]
    strict_check(protocol["settings"] == MODEL_SETTINGS and protocol["steps"] == 710
                 and protocol["checkpoint_sha256"] == FOUNDATION, "Unexpected training protocol")
    targets = {f"runs/A-{s}/result.json": s for s in SEEDS}
    markers = {}
    with tarfile.open(args.training_archive, "r|gz") as z:
        for member in z:
            if member.name in targets:
                markers[targets[member.name]] = json.load(z.extractfile(member))
    models = {}
    with ZipFile(args.weights) as z:
        meta = json.loads(z.read("models.json"))["weights"]
        strict_check(exact_sha(z.read("foundation.model")) == FOUNDATION, "Wrong initialization")
        for seed in SEEDS:
            ident, member = f"e0-A-{seed}", f"D0_{seed}.pt"
            a, p, m = inv[ident], prov[ident], markers[seed]
            expected = a["sha256_recorded"]
            blob = z.read(member)
            strict_check(
                a["protocol_group"] == "factorial-A" and a["class_id"] == "A"
                and a["ensemble_eligibility"] == "primary seed-spread group"
                and expected == exact_sha(blob) == meta[member]["sha256"] == m["output_sha256"]["model.pt"]
                and checkpoint_type(blob) == a["architecture"] == MODEL_CLASS
                and p["canonical_D0_xyz_sha256"] == D0_SHA
                and p["initialization_sha256"] == FOUNDATION and p["optimizer"] == "Adam"
                and p["seed"] == str(seed) and m["seed"] == seed
                and m["arm"] == "A" and m["treatment"] == {"alignment": False, "clip": 10.}
                and not m["smoke"] and m["steps"] == int(p["optimizer_steps"]) == 710
                and m["protocol_sha256"] == protocol_sha
                and m["order_sha256"] == p["order_sha256"] == meta[member]["order"]
                and m["runtime"] == meta[member]["training"]
                and p["hardware"] == m["runtime"]["hardware"]
                and p["driver"] == m["runtime"]["driver"]
                and p["torch_version"] == m["runtime"]["torch"]
                and p["mace_version"] == m["runtime"]["mace"]
                and all(float(p[field]) == MODEL_SETTINGS[key] for field, key in (
                    ("learning_rate", "lr"), ("energy_weight", "energy_weight"),
                    ("forces_weight", "forces_weight"), ("batch_size", "batch_size"),
                    ("clip_norm", "clip_norm"))), f"Unexpected checkpoint or provenance for {seed}",
            )
            models[seed] = (blob, dict(checkpoint_id=ident, member=member, sha256=expected,
                                       order_sha256=p["order_sha256"], training_runtime=m["runtime"]))
    strict_check(len({m[1]["sha256"] for m in models.values()}) == 3
                 and len({m[1]["order_sha256"] for m in models.values()}) == 3, "Duplicate model or order")
    strict_check(all(m[1]["training_runtime"] == models[17][1]["training_runtime"]
                     for m in models.values()), "Mixed training environments")
    return models, dict(settings=MODEL_SETTINGS, optimizer="Adam", original_e0=True, steps=710,
                        initialization_sha256=FOUNDATION, D0_sha256=D0_SHA, D0_frames=278,
                        architecture=MODEL_CLASS, source_protocol_sha256=protocol_sha)


def tangent_vectors(unwrapped):
    steps = np.diff(unwrapped, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    strict_check((lengths > 1e-10).all(), "Repeated Li positions do not define a tangent")
    directions = steps / lengths[:, None]
    local = np.vstack((directions[0], directions[:-1] + directions[1:], directions[-1]))
    norm = np.linalg.norm(local, axis=1)
    strict_check((norm > 1e-10).all(), "180-degree path reversal does not define a tangent")
    return local / norm[:, None]


def check_mic(positions, unwrapped, cell):
    """Independently check hops by enumerating lattice images about the fractional shift."""
    shifts = np.array(list(product(range(-2, 3), repeat=3)))
    for raw, actual in zip(np.diff(positions, axis=0), np.diff(unwrapped, axis=0)):
        fractional = np.linalg.solve(cell.T, raw)
        candidates = (fractional - np.rint(fractional) - shifts) @ cell
        shortest = candidates[np.argmin(np.linalg.norm(candidates, axis=1))]
        np.testing.assert_allclose(actual, shortest, rtol=0, atol=1e-10)


def prepare_inputs(args):
    sources = dict(neb=args.neb, index=args.index, bundle=args.bundle)
    checksums = {name: file_sha(path) for name, path in sources.items()}
    strict_check(checksums == EXPECTED, "Phase A0 source checksums changed")
    with ZipFile(args.bundle) as z:
        strict_check(exact_sha(frame_prefix(z.read("training.xyz"), 278)) == D0_SHA,
                     "D0 geometry checksum changed")
    a0 = args.repo/"results/phase_a0"
    a1 = args.repo/"results/phase_a1"
    a2 = args.repo/"results/phase_a2"
    manifest = {(r["edge_id"], r["stage"]): r for r in rows(a0/"geometry_manifest.csv")}
    index = {r["edge_id"]: r for r in rows(args.index)}
    cached = {(r["edge_id"], int(r["image"])): r for r in rows(a1/"feature_index.csv")
              if r["stage"] == "init" and r["DFT_preselectable"] == "1"}
    images = []
    members = {}
    with ZipFile(args.neb) as z:
        for edge in PATHS:
            mover = int(index[edge]["moving_Li_row"]) - 1
            member, digest, trajectory = parse_path(z, edge, "init", mover)
            m = manifest[edge, "init"]
            strict_check((member, digest, len(trajectory), mover + 1) ==
                         (m["source_member"], m["member_sha256"], int(m["images"]),
                          int(m["moving_Li_1based"])), f"A0 manifest mismatch for {edge}")
            alternatives, _ = validate_sequence(trajectory, mover, "init")
            strict_check(not any(alternatives), f"Alternative mapping in {edge}")
            moving = np.stack([atoms.positions[mover] for atoms in trajectory])
            cell = trajectory[0].cell.array
            unwrapped, arc, s = reaction_coordinate(moving, cell, trajectory[0].pbc)
            check_mic(moving, unwrapped, cell)
            tangent = tangent_vectors(unwrapped)
            members[member] = digest
            for i, atoms in enumerate(trajectory):
                ref = cached[edge, i + 1]
                strict_check(int(ref["Li_1based"]) == mover + 1
                             and abs(float(ref["reaction_coordinate"]) - s[i]) < 1e-12,
                             f"A1 feature row mismatch for {edge} {i+1}")
                clean = Atoms(numbers=atoms.numbers, positions=atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
                key = dict(edge_id=edge, stage="init", image=i + 1, feature_row_0based=int(ref["feature_row_0based"]),
                           n_atoms=len(clean), Li_1based=mover + 1, arc_A=float(arc[i]), reaction_coordinate=float(s[i]),
                           **{f"Li_unwrapped_{v}_A": float(x) for v, x in zip("xyz", unwrapped[i])},
                           **{f"tangent_{v}": float(x) for v, x in zip("xyz", tangent[i])})
                images.append((clean, key))
    strict_check(len(images) == len(cached) == 17, "Expected 17 initial images")
    cached_files = (a0/"geometry_manifest.csv", a1/"manifest.json", a1/"feature_index.csv",
                    a1/"novelty_profiles.csv", a1/"path_features.npz", a2/"manifest.json",
                    a2/"reference_sensitivity.csv", args.repo/"results/phase_b0/checkpoint_inventory.csv",
                    args.repo/"results/phase_b0/checkpoint_provenance.csv")
    checksums.update({path.relative_to(args.repo).as_posix(): file_sha(path) for path in cached_files})
    checksums.update(weights_archive=file_sha(args.weights), training_archive=file_sha(args.training_archive),
                     training_protocol_archive=file_sha(args.protocol))
    return images, checksums, members


def load_calculators(models, torch):
    from mace.calculators import MACECalculator

    result = {}
    for seed, (blob, _) in models.items():
        model = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
        strict_check(f"{type(model).__module__}.{type(model).__name__}" == MODEL_CLASS,
                     f"Deserialized class mismatch for {seed}")
        model.eval()
        result[seed] = MACECalculator(models=model, device="cpu", default_dtype="float64")
    return result


def predict(calculator, atoms):
    start = perf_counter()
    batch = calculator._atoms_to_batch(atoms)
    output = calculator.models[0](batch.to_dict(), compute_force=True, compute_stress=False, training=False)
    total = float(output["energy"].detach().item())
    node = output["node_energy"].detach().numpy().copy()
    force = output["forces"].detach().numpy().copy()
    elapsed = perf_counter() - start
    strict_check(node.shape == (len(atoms),) and force.shape == (len(atoms), 3)
                 and node.dtype == force.dtype == np.float64
                 and np.isfinite(force).all() and np.isfinite(node).all() and np.isfinite(total),
                 "Missing, wrong dtype, or invalid predictions")
    np.testing.assert_allclose(node.sum(), total, rtol=0, atol=REPEAT_ATOL)
    return total, node, force, elapsed


def compare(a, b):
    return {name: float(np.max(np.abs(np.subtract(x, y))))
            for name, x, y in zip(("energy_eV", "atomic_energy_eV", "force_eV_per_A"), a[:3], b[:3])}


def smoke(calculators, images, torch):
    atoms, key = next(item for item in images if item[1]["edge_id"] == PATHS[0] and item[1]["image"] == 3)
    threads = [t for t in (1, 2, 4, 8, 12, 16) if t <= (os.cpu_count() or 1)]
    timings = []
    reference = predict(calculators[17], atoms)
    for count in threads:
        torch.set_num_threads(count)
        predict(calculators[17], atoms)
        observed = [predict(calculators[17], atoms) for _ in range(3)]
        max_diff = max(v for p in observed for v in compare(reference, p).values())
        strict_check(max_diff <= REPEAT_ATOL, f"Thread-dependent output beyond tolerance at {count}")
        timings.append(dict(threads=count, seconds=[p[3] for p in observed],
                            median_seconds=float(np.median([p[3] for p in observed])), max_abs_diff=max_diff))
    best = min(timings, key=lambda r: r["median_seconds"])
    torch.set_num_threads(best["threads"])
    shifted = atoms.copy()
    shifted.positions += np.array([.70, 1.20, -.40])
    shifted.wrap()
    checks = []
    for seed, calc in calculators.items():
        first = predict(calc, atoms)
        repeat = predict(calc, atoms)
        periodic = predict(calc, shifted)
        d1, d2 = compare(first, repeat), compare(first, periodic)
        strict_check(max(d1.values()) <= REPEAT_ATOL and max(d2.values()) <= PERIODIC_ATOL,
                     f"Repeat or PBC invariance failed for seed {seed}")
        checks.append(dict(seed=seed, energy_eV=first[0], n_force_vectors=len(first[2]),
                           moving_Li_force=first[2][key["Li_1based"] - 1].tolist(), repeat_abs_errors=d1,
                           translate_wrap_abs_errors=d2, repeated_seconds=repeat[3]))
    return dict(status="pass", smoke_image=key, threads_selected=best["threads"], timings=timings, checks=checks,
                repeat_absolute_tolerance=REPEAT_ATOL, periodic_absolute_tolerance=PERIODIC_ATOL)


def predictions(calculators, images, models):
    images_out, atomic_out = [], []
    for atoms, key in images:
        for seed, calc in calculators.items():
            total, nodes, force, seconds = predict(calc, atoms)
            li = force[key["Li_1based"] - 1]
            ident = dict(edge_id=key["edge_id"], stage="init", image=key["image"], seed=seed)
            images_out.append(dict(ident, checkpoint_id=models[seed][1]["checkpoint_id"],
                                   checkpoint_sha256=models[seed][1]["sha256"], n_atoms=len(atoms),
                                   Li_1based=key["Li_1based"], total_energy_eV=total,
                                   mean_energy_per_atom_eV=total / len(atoms), inference_seconds=seconds,
                                   **{f"Li_F{v}_eV_per_A": float(x) for v, x in zip("xyz", li)}))
            for i, (symbol, energy, f) in enumerate(zip(atoms.get_chemical_symbols(), nodes, force), 1):
                atomic_out.append(dict(ident, atom_1based=i, element=symbol,
                                       is_moving_Li=int(i == key["Li_1based"]), atomic_model_energy_eV=float(energy),
                                       **{f"F{v}_eV_per_A": float(x) for v, x in zip("xyz", f)}))
    return images_out, atomic_out


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    strict_check(torch.version.cuda is None, "B1 requires the CPU build of PyTorch")
    models, protocol = frozen_models(args)
    images, hashes, members = prepare_inputs(args)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()
    scripts = {name: file_sha(args.repo/"src"/name)
               for name in ("phase_b1_inference.py", "phase_a0_geometry.py", "audit_phase_b0.py")}
    provenance = dict(source_commit=commit, script_sha256=scripts, inputs_sha256=hashes,
                      trajectory_member_sha256=members, training_protocol=protocol,
                      checkpoints=[m for _, m in models.values()])
    runtime = dict(device="cpu", dtype="float64", torch_interop_threads=1, logical_cpus=os.cpu_count(),
                   platform=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                   packages={n: version(n) for n in ("mace-torch", "ase", "numpy", "scipy", "e3nn")})
    if args.stage == "smoke":
        report = smoke(load_calculators(models, torch), images, torch)
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output/"smoke.json", dict(report, provenance=provenance, runtime=runtime))
        print(json.dumps(report), flush=True)
        return
    prior = read_json(args.output/"smoke.json")
    strict_check(prior["status"] == "pass" and prior["provenance"] == provenance and prior["runtime"] == runtime,
                 "Missing matching single-image acceptance")
    for name in ("image_geometry.csv", "raw_image_predictions.csv", "raw_atomic_predictions.csv", "run_manifest.json"):
        strict_check(not (args.output/name).exists(), f"Output already exists: {name}")
    torch.set_num_threads(prior["threads_selected"])
    start = perf_counter()
    im, at = predictions(load_calculators(models, torch), images, models)
    elapsed = perf_counter() - start
    export_csv(args.output/"image_geometry.csv", [key for _, key in images])
    export_csv(args.output/"raw_image_predictions.csv", im)
    export_csv(args.output/"raw_atomic_predictions.csv", at)
    write_json(args.output/"run_manifest.json", dict(provenance, runtime=dict(runtime, threads=prior["threads_selected"]),
               elapsed_seconds=elapsed, images=len(images), runs=len(im), atomic_rows=len(at),
               smoke_sha256=file_sha(args.output/"smoke.json"),
               output_sha256={n: file_sha(args.output/n) for n in
                              ("image_geometry.csv", "raw_image_predictions.csv", "raw_atomic_predictions.csv")}))
    print(f"CPU inference: {len(im)} predictions, {len(at)} atomic rows, {elapsed:.2f} s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "run"))
    for arg in ("repo", "neb", "index", "bundle", "weights", "training_archive", "protocol", "output"):
        parser.add_argument(arg, type=Path)
    main(parser.parse_args())
