"""Prepare and score the mp-1191903 fixed-geometry QE production controls."""

import argparse
import csv
import json
from pathlib import Path
import shutil
from xml.etree import ElementTree as ET

import numpy as np
from ase.io import read

from prepare_phase_b4_spin_pilot import prepare_input
from score_phase_b4_numeric_pilot import sha, verify_grid_xml


def write_json(path, record):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")


def prepare(args):
    spin_bytes = (args.spin / "manifest.json").read_bytes()
    spin = json.loads(spin_bytes)
    previous = json.loads(args.previous.read_bytes())
    records = {entry["role"]: entry for entry in previous["comparisons"]["k222"]["jobs"]}
    if previous["edge_id"] != spin["edge_id"] or sha(spin_bytes) != previous["baseline_manifest_sha256"]:
        raise ValueError("Prior k222 and spin manifests differ")
    if sha(args.trajectory.read_bytes()) != spin["geometry_source_sha256"]:
        raise ValueError("Audited geometry hash differs")
    frames = read(args.trajectory, ":", format="extxyz")
    if len(frames) != 5 or spin["selected_images"] != {"endpoint": 0, "bvse_peak": 1}:
        raise ValueError("The expected five-image trajectory or reference roles changed")
    reciprocal = 2 * np.pi * np.linalg.norm(frames[0].cell.reciprocal(), axis=1)
    dense = tuple(np.ceil(reciprocal / .25).astype(int).tolist())
    if dense != (3, 3, 3):
        raise ValueError("Unexpected grid at the frozen reciprocal-space spacing")

    args.output.mkdir()
    baseline = args.output / "baseline"
    baseline.mkdir()
    pseudos = args.output / "pseudo"
    pseudos.mkdir()
    for entry in spin["pseudopotentials"].values():
        source = args.spin / "pseudo" / entry["filename"]
        if sha(source.read_bytes()) != entry["sha256"]:
            raise ValueError(f"Wrong pseudopotential: {source.name}")
        shutil.copyfile(source, pseudos / source.name)

    jobs = {}

    def add_job(name, role, image, text, mesh):
        target = args.output / f"{name}.in"
        target.write_bytes(text)
        geometry = read(target, format="espresso-in")
        expected = frames[image]
        if (len(geometry) != 87 or not np.array_equal(geometry.numbers, expected.numbers)
                or not np.allclose(geometry.positions, expected.positions, atol=1e-8, rtol=0)
                or not np.allclose(geometry.cell, expected.cell, atol=1e-8, rtol=0)):
            raise ValueError(f"Geometry mismatch for {name}")
        jobs[name] = dict(role=role, image=image, grid=list(mesh), sha256=sha(text))

    for role, image in (("endpoint", 0), ("peak", 1)):
        orig = (args.grid / f"{role}_k222.in").read_bytes()
        warm = orig.replace(b"&ELECTRONS\n", b"&ELECTRONS\n   startingpot      = 'file'\n", 1)
        reference = records[role]
        if sha(warm) != reference["input_sha256"]:
            raise ValueError(f"Previous warm input hash mismatch for {role}")
        folder = args.recovered / f"{role}_k222_warm"
        xml = args.recovered / f"{role}_schema.xml"
        if (sha(xml.read_bytes()) != reference["xml_sha256"]
                or sha((folder / "pw.out").read_bytes()) != reference["log_sha256"]):
            raise ValueError(f"Previous output hash mismatch for {role}")
        slot = baseline / role
        slot.mkdir()
        (slot / "input.in").write_bytes(warm)
        shutil.copyfile(xml, slot / "schema.xml")
        shutil.copyfile(folder / "pw.out", slot / "pw.out")
        verify_grid_xml(slot, role, slot / "schema.xml", slot / "input.in", spin, {})

        if role == "endpoint":
            for name, text in (("io_k222", warm), ("io_k222_cold", orig)):
                add_job(name, role, image, text.replace(b"disk_io = 'medium'", b"disk_io = 'nowf'", 1), (2, 2, 2))
        for suffix, text in (("", warm), ("_cold", orig)):
            add_job(f"{role}_k333{suffix}", role, image,
                    text.replace(b"disk_io = 'medium'", b"disk_io = 'nowf'", 1)
                    .replace(b"2 2 2 0 0 0", b"3 3 3 0 0 0", 1), dense)

    # Freeze an interior image before observing the denser-grid forces.
    mid = 2
    filenames = {key: entry["filename"] for key, entry in spin["pseudopotentials"].items()}
    template = prepare_input(frames[mid], "middle", mid, "spin1muB", filenames, spin["chosen_cutoffs_Ry"], args.output)
    path = args.output / template["input_file"]
    middle = path.read_bytes()
    path.unlink()
    for mesh, tag in (((2, 2, 2), "222"), (dense, "333")):
        card = ("K_POINTS automatic\n" + " ".join(map(str, mesh)) + " 0 0 0").encode()
        add_job(f"middle_k{tag}", "middle", mid,
                middle.replace(b"&CONTROL\n", b"&CONTROL\n   disk_io = 'nowf'\n", 1)
                .replace(b"K_POINTS gamma", card, 1), mesh)

    original_seed = (args.recovered / "endpoint_k222_warm" / "starting_density.sha256").read_text().split()[0]
    write_json(args.output / "manifest.json", dict(
        edge_id=spin["edge_id"], jobs=jobs, reciprocal_norms_A_inverse=list(reciprocal),
        prior_input_seed_sha256=original_seed, old_scores_sha256=sha(args.previous.read_bytes()),
        spin=spin, io_reproduction=dict(total_energy_meV=.05, max_atomic_force_norm_eV_A=.0001),
        k_point_screen=dict(relative_energy_meV=10, force_component_rmse_eV_A=.02,
                            fine_force_component_rmse_eV_A=.01, migrating_li_force_norm_eV_A=.02),
    ))
    for name in ("run_phase_b4_numeric_gpu.sh", "run_phase_b4_r2_job.sh"):
        shutil.copyfile(args.scripts / name, args.output / name)
    with (args.output / "checksums.sha256").open("x", encoding="ascii", newline="\n") as handle:
        for path in sorted(args.output.rglob("*")):
            if path.is_file() and path != args.output / "checksums.sha256":
                handle.write(f"{sha(path.read_bytes())}  {path.relative_to(args.output).as_posix()}\n")


def read_result(bundle, manifest, name, run):
    if name in ("endpoint", "peak"):
        folder = bundle / "baseline" / name
        xml, inp = folder / "schema.xml", folder / "input.in"
        grid = [2, 2, 2]
        details = {}
    else:
        folder = run / name
        inp = bundle / f"{name}.in"
        xml = folder / "data-file-schema.xml"
        slot = manifest["jobs"][name]
        grid = slot["grid"]
        if not (folder / "converged").is_file():
            raise ValueError(f"Solver did not complete successfully for {name}")
        if (sha(inp.read_bytes()) != slot["sha256"]
                or (folder / "input.sha256").read_text().split()[0] != slot["sha256"]):
            raise ValueError(f"Executed input hash mismatch for {name}")
        details = dict(line.split("=", 1) for line in (folder / "runtime.txt").read_text().splitlines())
        with (folder / "gpu_samples.csv").open(newline="") as handle:
            details["sampled_peak_device_memory_MiB"] = max(
                int(row[" memory.used [MiB]"].split()[0]) for row in csv.DictReader(handle)
            )
        if (folder / "starting_density.sha256").is_file():
            details["starting_density_sha256"] = (folder / "starting_density.sha256").read_text().split()[0]
    root = ET.parse(xml).getroot()
    if root.find("closed") is None:
        raise ValueError(f"Unclosed QE output for {name}")
    mesh = root.find("input/k_points_IBZ/monkhorst_pack").attrib
    if [int(mesh[f"nk{i}"]) for i in (1, 2, 3)] != grid or any(int(mesh[f"k{i}"]) for i in (1, 2, 3)):
        raise ValueError(f"Wrong grid or shift in {name}")
    return verify_grid_xml(folder, name, xml, inp, manifest["spin"], details,
                           expected_nks=8 if grid == [2, 2, 2] else None)


def score(args):
    manifest = json.loads((args.bundle / "manifest.json").read_bytes())
    args.output.mkdir()
    comparisons = {}
    fields = ["comparison", "atom_index_0", "element", "is_migrating_Li",
              "reference_fx", "reference_fy", "reference_fz",
              "test_fx", "test_fy", "test_fz", "delta_fx", "delta_fy", "delta_fz"]
    with (args.output / "atomic_forces_eV_A.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for pair in args.pairs:
            old, new = pair.split(":", 1)
            reference, ref_force = read_result(args.bundle, manifest, old, args.run)
            test, new_force = read_result(args.bundle, manifest, new, args.run)
            d = new_force - ref_force
            norms = np.linalg.norm(d, axis=1)
            moving = manifest["spin"]["migrating_Li_0based"]
            position = int(norms.argmax())
            inp = args.bundle / f"{new}.in"
            if manifest["jobs"][new]["role"] != (old if old in ("endpoint", "peak") else manifest["jobs"][old]["role"]):
                raise ValueError(f"Cannot compare different images: {pair}")
            elements = read(inp, format="espresso-in").get_chemical_symbols()
            for i, element in enumerate(elements):
                writer.writerow([pair, i, element, i == moving, *ref_force[i], *new_force[i], *d[i]])
            comparisons[pair] = dict(
                reference=reference, test=test,
                total_energy_shift_meV=1000 * (test["total_energy_eV"] - reference["total_energy_eV"]),
                force_component_rmse_eV_A=float(np.sqrt(np.mean(d ** 2))),
                max_atomic_force_diff_eV_A=float(norms[position]),
                max_diff_atom_index_0=position, max_diff_element=elements[position],
                migrating_li_diff_eV_A=d[moving].tolist(), migrating_li_norm_eV_A=float(norms[moving]),
            )
    verdict = {}
    for pair, row in comparisons.items():
        if pair.startswith("endpoint:io_"):
            limits = manifest["io_reproduction"]
            verdict["io_pass"] = (abs(row["total_energy_shift_meV"]) <= limits["total_energy_meV"]
                                  and row["max_atomic_force_diff_eV_A"] <= limits["max_atomic_force_norm_eV_A"])
            verdict["io_exact_start_density"] = (
                row["test"].get("starting_density_sha256") == manifest["prior_input_seed_sha256"]
            )
    endpoint = next((row for name, row in comparisons.items() if name.startswith("endpoint:endpoint_k333")), None)
    limits = manifest["k_point_screen"]
    if endpoint:
        for pair, row in comparisons.items():
            if pair.startswith(("peak:peak_k333", "middle_k222:middle_k333")):
                shift = row["total_energy_shift_meV"] - endpoint["total_energy_shift_meV"]
                checked = (endpoint, row)
                verdict[pair] = dict(
                    relative_energy_shift_meV=shift,
                    pass_gate=abs(shift) <= limits["relative_energy_meV"] and all(
                        m["force_component_rmse_eV_A"] <= limits["force_component_rmse_eV_A"]
                        and m["migrating_li_norm_eV_A"] <= limits["migrating_li_force_norm_eV_A"] for m in checked),
                    all_force_rmse_below_0_01=all(
                        m["force_component_rmse_eV_A"] <= limits["fine_force_component_rmse_eV_A"] for m in checked),
                )
    write_json(args.output / "scores.json", dict(comparisons=comparisons, verdict=verdict))
    print(json.dumps(verdict, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    prep = commands.add_parser("prepare")
    for name in ("spin", "grid", "recovered", "previous", "trajectory", "scripts", "output"):
        prep.add_argument(name, type=Path)
    prep.set_defaults(action=prepare)
    assess = commands.add_parser("score")
    for name in ("bundle", "run", "output"):
        assess.add_argument(name, type=Path)
    assess.add_argument("pairs", nargs="+")
    assess.set_defaults(action=score)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
