"""Stage the three approved fixed-geometry QE calculations for the H3 primary."""

import argparse
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import tarfile

from ase.data import atomic_masses, atomic_numbers
from ase.io import read
import numpy as np

from prepare_phase_b4_spin_pilot import RY_EV, SMEARING_EV


BRANCHES = {
    "neutral_fixed": dict(charge=0, nspin=2, constrained=True, seeded=False),
    "neutral_local": dict(charge=0, nspin=2, constrained=False, seeded=True),
    "negative": dict(charge=-1, nspin=1, constrained=False, seeded=False),
}


def prepare_input(primary, atoms, references, name, settings):
    labels = atoms.get_chemical_symbols()
    element_for_label = {element: element for element in sorted(set(labels))}
    if settings["seeded"]:
        seed = primary["nearest_anion_to_A_vacancy_index_zero_based"]
        element = primary["anion_for_initial_spin_seed"]
        if labels[seed] != element or len(element) > 2:
            raise ValueError("The frozen anion seed is invalid")
        label = element + "1"
        labels[seed] = label
        element_for_label[label] = element
    wfc, rho = primary["start_cutoff_Ry"]
    lines = [
        "&CONTROL", "  calculation = 'scf',", "  restart_mode = 'from_scratch',",
        f"  prefix = '{name}',", "  pseudo_dir = '../pseudo',", "  outdir = './scratch',",
        "  disk_io = 'medium',", "  verbosity = 'high',", "  tprnfor = .true.,", "/",
        "&SYSTEM", f"  ibrav = 0, nat = {len(atoms)}, ntyp = {len(element_for_label)},",
        f"  input_dft = 'PBE', ecutwfc = {wfc:g}, ecutrho = {rho:g},",
        f"  tot_charge = {settings['charge']}.0, nspin = {settings['nspin']},",
        "  nosym = .true., occupations = 'smearing', smearing = 'gaussian',",
        f"  degauss = {SMEARING_EV / RY_EV:.12f},",
    ]
    if settings["constrained"]:
        lines.append("  tot_magnetization = 1.0,")
    if settings["seeded"]:
        lines.append(f"  starting_magnetization({len(element_for_label)}) = 0.15,")
    lines += ["/", "&ELECTRONS", "  conv_thr = 1.0d-8, electron_maxstep = 80, mixing_beta = 0.3,",
              "/", "ATOMIC_SPECIES"]
    for label, element in element_for_label.items():
        lines.append(f"{label} {atomic_masses[atomic_numbers[element]]:.8f} {references[element]['filename']}")
    lines.append("ATOMIC_POSITIONS angstrom")
    lines.extend(label + " " + " ".join(f"{x:.12f}" for x in position)
                 for label, position in zip(labels, atoms.positions))
    lines.append("K_POINTS automatic")
    lines.append(" ".join(map(str, primary["starting_kmesh"])) + " 0 0 0")
    lines.append("CELL_PARAMETERS angstrom")
    lines.extend(" ".join(f"{x:.12f}" for x in vector) for vector in atoms.cell)
    text = "\n".join(lines) + "\n"
    imported = read(StringIO(text), format="espresso-in")
    if (not np.array_equal(imported.numbers, atoms.numbers)
            or not np.allclose(imported.cell, atoms.cell, atol=1e-10, rtol=0)
            or not np.allclose(imported.positions, atoms.positions, atol=1e-10, rtol=0)):
        raise ValueError("Prepared QE input differs from the frozen geometry")
    return text


def main():
    from audit_public_neb_reuse import checked
    from prepare_h2_eval_geometry_panel import save_json

    p = argparse.ArgumentParser(description=__doc__)
    for key in ("audit", "geometry", "upf_inventory", "archive", "output"):
        p.add_argument(key, type=Path)
    args = p.parse_args()
    primary = json.loads(args.audit.read_bytes())["primary"]
    seed = args.geometry / "seed_paths" / (primary["slot"] + ".extxyz")
    checked(seed, primary["source_geometry_sha256"])
    atoms = read(seed, primary["geometry_anchors"]["A"])
    inventory = json.loads(args.upf_inventory.read_bytes())
    refs = {}
    for element in sorted(set(atoms.symbols)):
        if inventory[element]["file_status"] != "verified_pbe_header":
            raise ValueError(f"Unverified pseudopotential for {element}")
        refs[element] = dict(inventory[element]["catalogue"], sha256=inventory[element]["file_sha256"],
                             valence=inventory[element]["z_valence"])
    if sum(refs[s]["valence"] for s in atoms.symbols) != primary["neutral_valence_electrons"]:
        raise ValueError("Electron count differs from the qualification audit")
    args.output.mkdir(parents=True)
    pseudo = args.output / "pseudo"
    pseudo.mkdir()
    with tarfile.open(args.archive, "r:gz") as archive:
        members = {Path(m.name).name: m for m in archive.getmembers() if m.isfile()}
        for row in refs.values():
            name = row["filename"]
            raw = archive.extractfile(members[name]).read()
            if sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("Pseudopotential differs from the audited file")
            (pseudo / name).write_bytes(raw)
    branches = {}
    for name, settings in BRANCHES.items():
        folder = args.output / name
        folder.mkdir()
        raw = prepare_input(primary, atoms, refs, name, settings).encode("ascii")
        (folder / "pw.in").write_bytes(raw)
        branches[name] = {**settings, "input_sha256": sha256(raw).hexdigest(),
                          "electrons": primary["neutral_valence_electrons"] - settings["charge"]}
    save_json(args.output / "input_manifest.json", dict(slot=primary["slot"], atoms=len(atoms),
              kmesh=primary["starting_kmesh"], cutoffs_Ry=primary["start_cutoff_Ry"],
              branches=branches, pseudopotentials=refs, source_geometry_sha256=primary["source_geometry_sha256"]))


if __name__ == "__main__":
    main()
