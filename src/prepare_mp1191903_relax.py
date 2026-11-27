"""Build fixed-cell QE endpoint relaxations from the verified five-image seed."""
import argparse
import io
import json
from pathlib import Path
import re

import numpy as np
from ase.geometry import find_mic
from ase.io import read

from audit_signed_barriers import file_sha


def prepare(args):
    frames = read(args.geometry, index=":", format="extxyz")
    raw = args.qe_warm_input.read_text(encoding="ascii")
    reference = read(io.StringIO(raw), format="espresso-in")
    if len(frames) != 5 or any(
            len(a) != 87 or not a.pbc.all() or not np.array_equal(a.numbers, reference.numbers)
            or not np.allclose(a.cell, reference.cell, rtol=0, atol=1e-10) for a in frames):
        raise ValueError("The checked initial path must contain five consistent 87-atom images")
    if not np.allclose(frames[0].positions, reference.positions, rtol=0, atol=1e-10):
        raise ValueError("Warm starting density does not correspond to endpoint A")
    cumulative = np.zeros((87, 3))
    for a, b in zip(frames, frames[1:]):
        step, _ = find_mic(b.positions-a.positions, a.cell, a.pbc)
        cumulative += step
    movers = np.flatnonzero(np.linalg.norm(cumulative, axis=1) > 1e-6)
    if list(movers) != [0] or reference.numbers[0] != 3 or not 4.9 < np.linalg.norm(cumulative[0]) < 5.0:
        raise ValueError("The seed endpoints do not describe the frozen Li hop")
    header = raw.split("ATOMIC_POSITIONS angstrom")[0]
    if "2 2 2 0 0 0" not in header or "'file'" not in header:
        raise ValueError("Reference input is not the verified 2x2x2 warm endpoint")
    header = re.sub(r"(?m)^\s*disk_io\s*=\s*'nowf'", "\n   disk_io = 'medium'", header, count=1)
    header = re.sub(r"(?m)^\s*calculation\s*=\s*'scf'", "\n   calculation = 'relax'", header, count=1)
    header = header.replace("&CONTROL\n", "&CONTROL\n   restart_mode = 'from_scratch'\n"
                            "   nstep = 100\n   forc_conv_thr = 9.0d-4\n   etot_conv_thr = 1.0d-4\n"
                            "   max_seconds = 28800\n", 1)
    header = header.replace("ATOMIC_SPECIES", "&IONS\n   ion_dynamics = 'bfgs'\n/\nATOMIC_SPECIES", 1)
    args.output.mkdir()
    inputs = {}
    for name, atoms in (("endpoint_a", frames[0]), ("endpoint_b", frames[-1])):
        positions = "\n".join(f"{sym} " + " ".join(f"{v:.12f}" for v in pos)
                              for sym, pos in zip(atoms.symbols, atoms.positions))
        current = header if name == "endpoint_a" else re.sub(
            r"(?m)^\s*startingpot\s*=\s*'file'\s*\n", "\n", header, count=1)
        path = args.output / f"{name}.in"
        path.write_bytes((current + "ATOMIC_POSITIONS angstrom\n" + positions + "\n").encode("ascii"))
        check = read(path, format="espresso-in")
        if not (np.array_equal(check.numbers, atoms.numbers)
                and np.allclose(check.positions, atoms.positions, rtol=0, atol=1e-10)
                and np.allclose(check.cell, atoms.cell, rtol=0, atol=1e-10)):
            raise ValueError(f"Written endpoint input differs: {name}")
        inputs[path.name] = file_sha(path)
    (args.output / "geometry_receipt.json").write_bytes((json.dumps(dict(
        seed_sha256=file_sha(args.geometry), warm_input_sha256=file_sha(args.qe_warm_input),
        input_sha256=inputs, n_images=5, atoms=87, migrating_li_index_0=0,
        unwrapped_endpoint_distance_A=float(np.linalg.norm(cumulative[0])),
        purpose="fixed-cell ionic relax on seed endpoints, first branch 1 muB",
    ), indent=2) + "\n").encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("geometry", "qe_warm_input", "output"):
        parser.add_argument(name, type=Path)
    prepare(parser.parse_args())
