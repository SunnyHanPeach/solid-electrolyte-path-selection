"""Tabulate verifiable H1-C cost inputs without assigning uncalibrated runtimes."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from ase.io import iread


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_xyz", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prior = json.loads((args.results / "phase_b5_r/manifest.json").read_bytes())
    if hashlib.sha256(args.candidate_xyz.read_bytes()).hexdigest() != prior["geometry_sha256"]["candidate_images.xyz"]:
        raise ValueError("The candidate geometry does not match the frozen B5-R dataset")
    with (args.results / "phase_h1/structure_output/candidate_ranking_and_audit.csv").open(newline="", encoding="utf-8") as handle:
        h1 = {row["edge_id"]: row for row in csv.DictReader(handle)}
    with (args.results / "phase_h1/structure_output/prospective_review_panel.csv").open(newline="", encoding="utf-8") as handle:
        panel = {row["edge_id"]: row["panel_role"] for row in csv.DictReader(handle)}
    spin = json.loads((args.results / "phase_b4_spin_preparation/manifest.json").read_bytes())
    bridge = json.loads((args.results / "phase_b4_bridge01/run_manifest.json").read_bytes())
    tested_species = set(spin["pseudopotentials"]) | set(bridge["pseudo_sha256"])

    frames = {}
    for frame in iread(args.candidate_xyz, ":", format="extxyz"):
        frames.setdefault(frame.info["edge_id"], []).append(frame)
    if set(frames) != set(h1) or len(frames) != prior["candidate_paths"]:
        raise ValueError("H1 and B5-R path memberships differ")
    fields = ("edge_id", "chemsys", "anion_family", "h1_panel_role", "atoms", "initial_images",
              "b1_A_inverse", "b2_A_inverse", "b3_A_inverse", "gamma_centered_mesh_delta_0_25",
              "nominal_k_points", "elements_without_previously_benchmarked_upf",
              "actual_qe_k_points", "single_image_scf_seconds", "neb_iterations",
              "gpu_occupied_hours", "failure_retry_fraction", "estimated_total_cost", "cost_status")
    with args.output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for edge, images in frames.items():
            first = images[0]
            row = h1[edge]
            if (len(images) != int(row["images"]) or len(first) != int(row["atoms"])
                    or sorted(int(img.info["image"]) for img in images) != list(range(len(images)))
                    or any(not np.allclose(first.cell.array, img.cell.array, atol=1e-9, rtol=0)
                           or not np.array_equal(first.numbers, img.numbers) for img in images)):
                raise ValueError(f"Geometry grouping, atom mapping or cell differs for {edge}")
            reciprocal = 2 * np.pi * np.linalg.norm(first.cell.reciprocal(), axis=1)
            mesh = np.ceil(reciprocal / .25).astype(int)
            missing = set(first.get_chemical_symbols()) - tested_species
            writer.writerow(dict(
                edge_id=edge, chemsys=row["chemsys"], anion_family=row["anion_family"],
                h1_panel_role=panel.get(edge, ""), atoms=len(first), initial_images=len(images),
                b1_A_inverse=f"{reciprocal[0]:.8f}", b2_A_inverse=f"{reciprocal[1]:.8f}",
                b3_A_inverse=f"{reciprocal[2]:.8f}", gamma_centered_mesh_delta_0_25="x".join(map(str, mesh)),
                nominal_k_points=int(np.prod(mesh)),
                elements_without_previously_benchmarked_upf=";".join(sorted(missing)),
                cost_status="awaiting_matched_qe_scf_and_neb_measurements",
            ))


if __name__ == "__main__":
    main()
