"""Validate relaxed endpoints and construct the fixed-cell 5- or 7-image CI-NEB seed."""
import argparse
from pathlib import Path
from xml.etree import ElementTree as ET

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.geometry import find_mic
from ase.io import read, write
import numpy as np


def qe_image(path):
    root = ET.parse(path).getroot()
    out = root.find("output")
    scf = out.find("convergence_info/scf_conv")
    if (root.find("closed") is None or scf.findtext("convergence_achieved") != "true"
            or float(scf.findtext("scf_error"))*2 > 1e-8):
        raise ValueError(f"Unconverged electronic force: {path}")
    structure = out.find("atomic_structure")
    pos = list(structure.find("atomic_positions"))
    bohr_A = .529177210903
    coords = np.asarray([np.fromstring(x.text, sep=" ") for x in pos])*bohr_A
    cell = np.asarray([np.fromstring(x.text, sep=" ") for x in structure.find("cell")])*bohr_A
    atoms = Atoms([x.attrib["name"] for x in pos], positions=coords, cell=cell, pbc=True)
    force = np.fromstring(out.findtext("forces"), sep=" ").reshape(-1,3)*51.42206748
    energy = float(out.findtext("total_energy/etot"))*27.211386245988
    if force.shape != (len(atoms),3) or not np.isfinite(force).all() or not np.isfinite(energy):
        raise ValueError(f"Invalid force or energy: {path}")
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=force)
    return atoms


def relax_endpoint(path, expected, other):
    endpoint = qe_image(path)
    if (not np.array_equal(endpoint.numbers, expected.numbers)
            or not np.allclose(endpoint.cell, expected.cell, rtol=0, atol=1e-6)):
        raise ValueError(f"Atom order or fixed cell changed in {path}")
    delta, distance = find_mic(endpoint.positions-expected.positions, expected.cell, True)
    _, to_other = find_mic(endpoint.positions[0]-other.positions[0], expected.cell, True)
    if distance[0] >= to_other or not np.isfinite(distance).all():
        raise ValueError(f"Migrating Li has left its intended endpoint basin: {path}")
    forces = endpoint.get_forces()
    if not np.isfinite(forces).all() or np.linalg.norm(forces, axis=1).max() > .05:
        raise ValueError(f"Endpoint force criterion failed: {path}")
    geom = expected.copy()
    geom.positions += delta
    geom.calc = SinglePointCalculator(geom, energy=endpoint.get_potential_energy(), forces=forces)
    return geom, delta


def five(seed, a_path, b_path):
    raw = read(seed, index=":", format="extxyz")
    if len(raw) != 5:
        raise ValueError("Expected original 5-image seed")
    a, da = relax_endpoint(a_path, raw[0], raw[-1])
    b, db = relax_endpoint(b_path, raw[-1], raw[0])
    if np.linalg.norm(find_mic(b.positions[0]-a.positions[0], a.cell, True)[0]) < 1:
        raise ValueError("Relaxed endpoints no longer define a distinct Li hop")
    segments = [find_mic(y.positions-x.positions, a.cell, True)[0] for x, y in zip(raw, raw[1:])]
    reaction = np.asarray([0, *np.cumsum([np.linalg.norm(x) for x in segments])])
    reaction /= reaction[-1]
    interior = []
    for j in range(1, len(raw)-1):
        atoms = raw[j].copy()
        q = reaction[j]
        atoms.positions = raw[0].positions + sum(segments[:j]) + (1-q)*da + q*db
        interior.append(atoms)
    return [a, *interior, b]


def seven(previous):
    images = read(previous, index=":")
    if len(images) != 5:
        raise ValueError("Expected one complete five-image CI result")
    peak = 1 + int(np.argmax([a.get_potential_energy() for a in images[1:-1]]))
    refined = [images[0]]
    for i, (a, b) in enumerate(zip(images, images[1:])):
        if i in (peak-1, peak):
            mid = a.copy()
            mid.positions += .5*find_mic(b.positions-a.positions, a.cell, True)[0]
            refined.append(mid)
        refined.append(b)
    return refined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    stages = parser.add_subparsers(dest="stage", required=True)
    base = stages.add_parser("five")
    for name in ("seed", "endpoint_a", "endpoint_b"):
        base.add_argument(name, type=Path)
    stages.add_parser("seven").add_argument("previous", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    images = five(args.seed, args.endpoint_a, args.endpoint_b) if args.stage == "five" else seven(args.previous)
    write(args.output, images, format="traj")
