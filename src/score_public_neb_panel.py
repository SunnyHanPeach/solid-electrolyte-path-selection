"""Score the audited D0 seed on frozen public geometries, without model fitting."""

import argparse
from collections import defaultdict
import gzip
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from ase import Atoms
from pymatgen.core import Structure

from audit_phase_h2_evaluation import rows
from audit_public_neb_panel import geometry_hash
from audit_public_neb_reuse import FP_HASH, checked
from phase_b1_inference import file_sha, load_calculators, predict, write_json
from phase_h1_shortlist import write


def score_path(calculator, record, images):
    if geometry_hash(images) != record["ordered_DFT_geometry_sha256"]:
        raise ValueError(f"DFT geometry differs from the frozen panel: {record['path_id']}")
    predictions, references, force_squared_errors = [], [], []
    for index, image in enumerate(images):
        if image["image_index"] != index:
            raise ValueError(f"Unordered DFT frames: {record['path_id']}")
        structure = Structure.from_dict(image["structure"])
        atoms = Atoms([site.specie.symbol for site in structure], positions=structure.cart_coords,
                      cell=structure.lattice.matrix, pbc=True)
        predicted, _, forces, _ = predict(calculator, atoms)
        if not predictions:
            symbols = atoms.get_chemical_symbols()
            cell = atoms.cell.array.copy()
        if atoms.get_chemical_symbols() != symbols or not np.allclose(atoms.cell.array, cell, rtol=0, atol=1e-6):
            raise ValueError(f"Unmatched ordered images: {record['path_id']}")
        reference_force = np.asarray(image["forces_eV_per_angstrom"], dtype=float)
        force_squared_errors.append((forces - reference_force) ** 2)
        predictions.append(predicted)
        references.append(float(image["energy_total_eV"]))
    model_barrier = max(predictions) - predictions[0]
    dft_barrier = max(references) - references[0]
    return dict(path_id=record["path_id"], icsd_id=record["icsd_id"],
                family=record["screened_framework_family"], n_images=len(images),
                family_weight=float(record["family_weight_within_panel"]),
                model_forward_barrier_eV=model_barrier, dft_forward_barrier_eV=dft_barrier,
                signed_barrier_error_meV=1000 * (model_barrier - dft_barrier),
                absolute_barrier_error_meV=1000 * abs(model_barrier - dft_barrier),
                force_component_RMSE_eV_A=float(np.sqrt(np.mean(force_squared_errors))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "panel", "checkpoint", "b5_manifest", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    expected = json.loads(args.b5_manifest.read_bytes())["checkpoint_sha256"]
    weights = checked(args.checkpoint, expected)
    source = json.loads(gzip.decompress(checked(args.reference, FP_HASH)))
    panel = [row for row in rows(args.panel) if row["use_for_external_scoring"] == "True"]
    if len(panel) != 62 or len({row["screened_framework_family"] for row in panel}) != 42:
        raise ValueError("Unexpected panel membership or family count")
    np.testing.assert_allclose(sum(float(row["family_weight_within_panel"]) for row in panel),
                               1.0, atol=1e-12, rtol=0)
    calculator = load_calculators({17: (weights, None)}, torch)[17]
    args.output.mkdir(parents=True)
    scores = []
    for row in panel[:1] if args.smoke_only else panel:
        images = source["pathways"][row["path_id"]]["dft_neb_reference"]["images"]
        if len(images) != int(row["dft_images"]):
            raise ValueError(f"Incorrect number of DFT images: {row['path_id']}")
        result = score_path(calculator, row, images)
        if not scores:
            repeated = score_path(calculator, row, images)
            for key in ("model_forward_barrier_eV", "force_component_RMSE_eV_A"):
                np.testing.assert_allclose(result[key], repeated[key], atol=1e-9, rtol=0)
        scores.append(result)
        if len(scores) == 1 or len(scores) % 10 == 0:
            print(f"Scored {len(scores)} paths in {perf_counter() - started:.1f} seconds.", flush=True)
    write(args.output / "path_scores.csv", scores)
    summary = dict(
        assignment="D0-only baseline",
        metrics="fixed DFT geometries, listed forward endpoint, family macro average",
        smoke_only=args.smoke_only,
        checkpoint_sha256=expected,
        reference_sha256=FP_HASH,
        panel_sha256=file_sha(args.panel),
        script_sha256=file_sha(Path(__file__)),
        dtype="float64",
        device="cpu",
        seeds=[17],
        n_paths=len(scores),
        n_images=sum(row["n_images"] for row in scores),
        first_path_repeated=True,
        software={name: version(name) for name in ("mace-torch", "torch", "ase", "pymatgen")},
        cpu_threads=args.threads,
        elapsed_seconds=round(perf_counter()-started, 3),
    )
    families = defaultdict(list)
    for row in scores:
        families[row["family"]].append(row["absolute_barrier_error_meV"])
    if not args.smoke_only:
        summary["n_families"] = len(families)
        summary["path_mean_barrier_MAE_meV"] = float(np.mean([row["absolute_barrier_error_meV"]
                                                             for row in scores]))
        summary["family_macro_barrier_MAE_meV"] = float(np.mean([np.mean(errors)
                                                               for errors in families.values()]))
    write_json(args.output / "scoring_manifest.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
