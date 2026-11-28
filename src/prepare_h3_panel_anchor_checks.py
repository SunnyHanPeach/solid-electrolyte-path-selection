"""Archive the completed A checks and stage the frozen B and midpoint comparisons."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from shutil import copy2
from xml.etree import ElementTree

from ase.io import read
import numpy as np

from prepare_h3_panel_scf import BRANCHES, prepare_input
from prepare_phase_b4_spin_pilot import RY_EV


HARTREE_EV = 2 * RY_EV
BOHR_ANGSTROM = 0.529177210903


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def collect_branch(source, archive, name, electrons, atoms):
    folder = source / name
    xml = folder / "scratch" / f"{name}.save" / "data-file-schema.xml"
    root = ElementTree.parse(xml).getroot()
    output = root.find("output")
    scf = output.find("convergence_info/scf_conv")
    log = (folder / "pw.log").read_text()
    timing = dict(line.split("=", 1) for line in (folder / "end.txt").read_text().splitlines())
    if (scf.findtext("convergence_achieved") != "true" or root.find("closed") is None
            or timing["solver_exit"] != "0" or "JOB DONE." not in log):
        raise ValueError(f"Missing converged QE receipt for {name}")

    structure = output.find("atomic_structure")
    qe_atoms = list(structure.find("atomic_positions"))
    xyz = np.array([[float(x) for x in atom.text.split()] for atom in qe_atoms])
    cell = np.array([[float(x) for x in vector.text.split()] for vector in structure.find("cell")])
    if ([atom.get("name").rstrip("0123456789") for atom in qe_atoms] != atoms.get_chemical_symbols()
            or not np.allclose(xyz * BOHR_ANGSTROM, atoms.positions, atol=1e-7, rtol=0)
            or not np.allclose(cell * BOHR_ANGSTROM, atoms.cell, atol=1e-7, rtol=0)):
        raise ValueError(f"Unexpected geometry in {name} XML")

    forces = output.find("forces")
    if forces.get("dims").split() != ["3", str(len(atoms))]:
        raise ValueError(f"Malformed force array for {name}")
    values = np.array([float(x) for x in forces.text.split()]).reshape(len(atoms), 3)
    values *= HARTREE_EV / BOHR_ANGSTROM
    energy = float(output.findtext("total_energy/etot")) * HARTREE_EV
    scf_error = float(scf.findtext("scf_error")) * 2
    bands = output.find("band_structure")
    if (float(bands.findtext("nelec")) != electrons or not np.isfinite(values).all()
            or not np.isfinite(energy) or not 0 <= scf_error <= 1e-8):
        raise ValueError(f"Invalid energy, forces or electronic convergence for {name}")
    magnetization = {}
    for key in ("total", "absolute"):
        moments = re.findall(rf"^\s+{key} magnetization\s*=\s*([+-]?[\d.]+)", log, re.M)
        magnetization[key] = float(moments[-1]) if moments else None
    fermi_Ha = bands.findtext("fermi_energy")
    if fermi_Ha is not None:
        fermi_eV = HARTREE_EV * float(fermi_Ha)
        levels_eV = np.fromstring(
            " ".join(band.text for band in bands.findall("ks_energies/eigenvalues")), sep=" "
        ) * HARTREE_EV
        near_fermi_eV = float(np.min(np.abs(levels_eV - fermi_eV)))
    else:
        near_fermi_eV = None
    destination = archive / name
    destination.mkdir()
    files = [folder / item for item in ("pw.in", "pw.log", "start.txt", "end.txt", "runtime.txt")]
    files.append(xml)
    hashes = {}
    for path in files:
        target = Path(copy2(path, destination / path.name))
        hashes[target.name] = digest(target)
        if hashes[target.name] != digest(path):
            raise ValueError(f"Archive differs from {path}")
    record = dict(
        electronic_converged=True, scf_error_Ry=scf_error, electrons=electrons, energy_eV=energy,
        forces_eV_per_A=values.tolist(), magnetization_muB=magnetization,
        min_level_distance_from_fermi_eV=near_fermi_eV,
        wall_seconds=int(timing["elapsed_seconds"]), archive_sha256=hashes,
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("audit", "seed", "completed", "output"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    primary = json.loads(args.audit.read_bytes())["primary"]
    source = args.completed
    manifest = json.loads((source / "input_manifest.json").read_bytes())
    if (manifest["slot"] != primary["slot"] or digest(args.seed) != primary["source_geometry_sha256"]
            or manifest["source_geometry_sha256"] != primary["source_geometry_sha256"]
            or manifest["kmesh"] != primary["starting_kmesh"]
            or manifest["cutoffs_Ry"] != primary["start_cutoff_Ry"]
            or "qualification_batch_exit=0" not in (source / "started/completion").read_text().splitlines()):
        raise ValueError("The completed A run and its frozen assets disagree")
    anchors = primary["geometry_anchors"]
    atoms = {key: read(args.seed, index=index) for key, index in anchors.items()}
    for key in ("middle", "B"):
        if (not np.array_equal(atoms["A"].numbers, atoms[key].numbers)
                or not np.array_equal(atoms["A"].cell, atoms[key].cell)
                or not np.array_equal(atoms["A"].pbc, atoms[key].pbc)):
            raise ValueError(f"Changing cell, atom order or periodicity at {key}")
    refs = manifest["pseudopotentials"]
    if sum(refs[s]["valence"] for s in atoms["A"].symbols) != primary["neutral_valence_electrons"]:
        raise ValueError("Frozen species and electron count disagree")
    settings = BRANCHES["neutral_fixed"]
    args.output.mkdir(parents=True)
    pseudo = args.output / "pseudo"
    pseudo.mkdir()
    for row in refs.values():
        original = source / "pseudo" / row["filename"]
        target = Path(copy2(original, pseudo / original.name))
        if digest(target) != row["sha256"]:
            raise ValueError(f"Incorrect UPF hash for {original.name}")

    archive = args.output / "A_archive"
    archive.mkdir()
    copy2(source / "input_manifest.json", archive / "input_manifest.json")
    copy2(source / "started/receipt", archive / "run_receipt.txt")
    copy2(source / "started/completion", archive / "completion.txt")
    records = {}
    for name, branch in manifest["branches"].items():
        expected = prepare_input(primary, atoms["A"], refs, name, BRANCHES[name]).encode("ascii")
        if (source / name / "pw.in").read_bytes() != expected or sha256(expected).hexdigest() != branch["input_sha256"]:
            raise ValueError(f"The regenerated A input differs for {name}")
        records[name] = collect_branch(source, archive, name, branch["electrons"], atoms["A"])
    fixed, local = records["neutral_fixed"], records["neutral_local"]
    forces = np.asarray(local["forces_eV_per_A"]) - np.asarray(fixed["forces_eV_per_A"])
    write_json(archive / "A_results.json", dict(
        branches=records,
        comparison=dict(neutral_local_minus_fixed_meV=1000*(local["energy_eV"]-fixed["energy_eV"]),
                        neutral_force_component_RMSE_eV_per_A=float(np.sqrt(np.mean(forces**2))),
                        neutral_max_force_vector_difference_eV_per_A=float(np.max(np.linalg.norm(forces, axis=1)))),
    ))

    branches = {}
    tighter = dict(primary, starting_kmesh=primary["tighter_kmesh"])
    more_cutoff = dict(primary, start_cutoff_Ry=[1.25*x for x in primary["start_cutoff_Ry"]])
    for variant, config in (("base", primary), ("k_dense", tighter), ("cutoff_up", more_cutoff)):
        for anchor, structure in atoms.items():
            if variant == "base" and anchor == "A":
                continue
            name = f"{variant}_{anchor.lower()}"
            folder = args.output / name
            folder.mkdir()
            raw = prepare_input(config, structure, refs, name, settings).encode("ascii")
            (folder / "pw.in").write_bytes(raw)
            branches[name] = dict(
                anchor=anchor, image_index=anchors[anchor],
                reaction_coordinate=primary["anchor_s"][str(anchors[anchor])],
                variant=variant, kmesh=config["starting_kmesh"], cutoffs_Ry=config["start_cutoff_Ry"],
                input_sha256=sha256(raw).hexdigest(),
            )
    write_json(args.output / "input_manifest.json", dict(
        slot=primary["slot"], source_geometry_sha256=digest(args.seed), pseudopotentials=refs,
        atoms=len(atoms["A"]), moving_Li_index_zero_based=primary["moving_Li_index_zero_based"],
        reference_settings=settings, branches=branches, baseline_A_archive="A_archive/neutral_fixed",
        first_run=["base_middle", "base_b"],
        acceptance=dict(relative_energy_meV=10, force_component_RMSE_eV_per_A=0.020,
                        moving_Li_force_vector_difference_eV_per_A=0.020),
    ))


if __name__ == "__main__":
    main()
