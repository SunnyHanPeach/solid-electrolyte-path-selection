"""Validate the completed H3 baseline and compare the first denser-grid pair."""

import argparse
import json
from pathlib import Path

from ase.io import read
import numpy as np

from prepare_h3_panel_anchor_checks import collect_branch, digest, write_json


def archive_branch(source, name, atoms, electrons, manifest, archive):
    if digest(source / name / "pw.in") != manifest["branches"][name]["input_sha256"]:
        raise ValueError(f"Unexpected input in completed branch {name}")
    return collect_branch(source, archive, name, electrons, atoms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("baseline", "dense-pair"))
    for field in ("case", "audit", "seed", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--middle-source", type=Path)
    parser.add_argument("--dense-source", type=Path)
    args = parser.parse_args()
    case = args.case
    manifest = json.loads((case / "input_manifest.json").read_bytes())
    primary = json.loads(args.audit.read_bytes())["primary"]
    if (manifest["slot"] != primary["slot"] or digest(args.seed) != primary["source_geometry_sha256"]
            or manifest["source_geometry_sha256"] != primary["source_geometry_sha256"]):
        raise ValueError("The frozen geometry or slot changed")
    atoms = {name: read(args.seed, index) for name, index in primary["geometry_anchors"].items()}
    electrons = primary["neutral_valence_electrons"]
    args.output.mkdir()
    if args.mode == "baseline":
        if args.middle_source is None:
            parser.error("--middle-source is required for baseline")
        old = case / "A_archive"
        reference = json.loads((old / "A_results.json").read_bytes())["branches"]["neutral_fixed"]
        if any(digest(old / "neutral_fixed" / name) != value for name, value in reference["archive_sha256"].items()):
            raise ValueError("The archived A branch differs from its receipt")
        branches = {
            "A": reference,
            "middle": archive_branch(args.middle_source, "base_middle", atoms["middle"],
                                     electrons, manifest, args.output),
            "B": archive_branch(case, "base_b", atoms["B"], electrons, manifest, args.output),
        }
        report = {"branches": branches}
        for reference_anchor in ("A", "B"):
            ref = branches[reference_anchor]["energy_eV"]
            report[f"relative_to_{reference_anchor}_meV"] = {
                name: 1000 * (item["energy_eV"] - ref) for name, item in branches.items()
            }
    else:
        if args.dense_source is None:
            parser.error("--dense-source is required for dense-pair")
        old = json.loads((case / "baseline_validation/summary.json").read_bytes())["branches"]
        a = archive_branch(args.dense_source, "k_dense_a", atoms["A"], electrons, manifest, args.output)
        middle = archive_branch(args.dense_source, "k_dense_middle", atoms["middle"],
                                electrons, manifest, args.output)
        change = 1000 * ((middle["energy_eV"] - a["energy_eV"])
                         - (old["middle"]["energy_eV"] - old["A"]["energy_eV"]))
        limits = manifest["acceptance"]
        li = manifest["moving_Li_index_zero_based"]
        force_differences = {}
        for name, row in (("A", a), ("middle", middle)):
            diff = np.asarray(row["forces_eV_per_A"]) - np.asarray(old[name]["forces_eV_per_A"])
            force_differences[name] = dict(
                component_RMSE_eV_per_A=float(np.sqrt(np.mean(diff**2))),
                moving_Li_vector_difference_eV_per_A=float(np.linalg.norm(diff[li])),
            )
        passed = abs(change) <= limits["relative_energy_meV"] and all(
            row["component_RMSE_eV_per_A"] <= limits["force_component_RMSE_eV_per_A"]
            and row["moving_Li_vector_difference_eV_per_A"] <= limits["moving_Li_force_vector_difference_eV_per_A"]
            for row in force_differences.values()
        )
        report = dict(branches={"A": a, "middle": middle}, numerical_pair_pass=passed,
                      delta_middle_minus_A_meV=change, force_differences=force_differences)
    if any(row["magnetization_muB"]["total"] != 1.0 for row in report["branches"].values()):
        raise ValueError("A branch differs from the fixed 1 μB spin setting")
    write_json(args.output / "summary.json", report)


if __name__ == "__main__":
    main()
