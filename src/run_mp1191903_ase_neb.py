"""Run ASE's improved-tangent path optimizer with the validated QE GPU engine."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from time import monotonic

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.geometry import find_mic
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import FIRE

from audit_signed_barriers import file_sha
from prepare_mp1191903_neb import qe_image


PREFIX = "b4_mp1191903_endpoint_spin1muB"
PW_HASH = "9d4510c4aff6c7340725bdb9244aef5b6011fdcd91f6a7197f526bdb1f0cc702"


class ImagePw(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, work, header, pw, environment):
        super().__init__()
        self.work, self.header, self.pw, self.environment = work, header, pw, environment
        self.work.mkdir(exist_ok=True)
        self.save = work / "scratch" / f"{PREFIX}.save"

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = "\n".join(f"{s} " + " ".join(f"{v:.12f}" for v in p)
                              for s, p in zip(atoms.symbols, atoms.positions))
        density = self.save / "charge-density.dat"
        header = self.header
        if density.is_file():
            header = header.replace("&ELECTRONS\n", "&ELECTRONS\n   startingpot = 'file'\n", 1)
        attempt = self.work / f"scf_{len(list(self.work.glob('scf_*.in'))):05d}"
        text = header + "ATOMIC_POSITIONS angstrom\n" + positions + "\n"
        attempt.with_suffix(".in").write_text(text, encoding="ascii")
        start = monotonic()
        with attempt.with_suffix(".log").open("xb") as handle:
            completed = subprocess.run([self.pw, "-in", str(attempt.with_suffix(".in"))],
                                       env=self.environment, cwd=self.work, stdout=handle, stderr=subprocess.STDOUT)
        attempt.with_suffix(".status.json").write_text(json.dumps(
            dict(seconds=monotonic()-start, returncode=completed.returncode, warm_start="startingpot" in header)))
        log = attempt.with_suffix(".log").read_text(errors="replace")
        if completed.returncode or "GPU acceleration is ACTIVE" not in log or "JOB DONE." not in log:
            raise RuntimeError(f"QE failed: {attempt}")
        xml = self.save / "data-file-schema.xml"
        durable = attempt.with_suffix(".xml")
        durable.write_bytes(xml.read_bytes())
        got = qe_image(durable)
        if (not np.array_equal(got.numbers, atoms.numbers)
                or not np.allclose(got.cell, atoms.cell, rtol=0, atol=1e-5)
                or np.max(find_mic(got.positions-atoms.positions, atoms.cell, True)[1]) > 1e-5):
            raise ValueError(f"QE output geometry differs: {attempt}")
        self.results.update(energy=got.get_potential_energy(), forces=got.get_forces())


def run(args):
    args.output = args.output.resolve()
    args.output.mkdir(exist_ok=True)
    runtime, pseudo = args.runtime.resolve(), args.pseudo.resolve()
    pw = runtime / "usr/local/qe/bin/pw.x"
    if file_sha(pw) != PW_HASH:
        raise ValueError("GPU QE runtime has changed")
    env = os.environ.copy()
    env.update(LD_LIBRARY_PATH=":".join(str(runtime / loc) for loc in (
        "opt/nvidia/hpc_sdk/Linux_x86_64/24.7/compilers/lib", "usr/local/openmpi/lib", "usr/local/ucx/lib",
        "usr/local/fftw/lib")) + ":/usr/local/cuda/lib64",
        OPAL_PREFIX=str(runtime / "usr/local/openmpi"), OMPI_MCA_ess_singleton_isolated="1", OMPI_MCA_pml="ob1",
        OMPI_MCA_btl="self", UCX_MEMTYPE_CACHE="n", HWLOC_COMPONENTS="-gl", OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1", NVCOMPILER_FIO_UNBUFFERED="1")
    header = args.template.read_text().split("ATOMIC_POSITIONS angstrom")[0]
    header = re.sub(r"(?m)^\s*startingpot.*\n", "\n", header, count=1)
    header = re.sub(r"(?m)(pseudo_dir\s*=\s*)'[^']+'", lambda m: m[1]+f"'{pseudo}'", header)
    identity = dict(seed=file_sha(args.path), template=file_sha(args.template), pw=PW_HASH, climb=args.climb,
                    fmax=args.fmax, spring_eV_A2=.1, optimizer="FIRE", method="improvedtangent")
    provenance = args.output / "protocol.json"
    if provenance.exists():
        if json.loads(provenance.read_bytes()) != identity:
            raise ValueError("The resumed path or method has changed")
    else:
        provenance.write_text(json.dumps(identity, indent=2)+"\n")
    latest = args.output / "latest.traj"
    images = read(latest if latest.exists() else args.path, index=":")
    if len(images) not in (5, 7):
        raise ValueError("The CI-NEB path must have five or seven images")
    for i, a in enumerate(images[1:-1], 1):
        previous = a.calc
        a.calc = ImagePw(args.output / f"image_{i:02d}", header, pw, env)
        if previous is not None and latest.exists():
            a.calc.atoms = a.copy()
            a.calc.results = {name: previous.get_property(name, a) for name in ("energy", "forces")}
    band = NEB(images, method="improvedtangent", k=.1, climb=args.climb)

    def snapshot():
        pending = latest.with_suffix(".partial")
        write(pending, images, format="traj")
        pending.replace(latest)

    opt = FIRE(band, logfile=str(args.output / "steps.log"), dt=.1, maxstep=.12)
    opt.attach(snapshot)
    done = opt.run(fmax=args.fmax, steps=args.steps)
    snapshot()
    energy = np.asarray([a.get_potential_energy() for a in images])
    residual = np.linalg.norm(band.get_forces(), axis=1).max()
    result = dict(converged=bool(done), fmax=float(residual), steps=opt.nsteps,
                  highest_image=int(energy.argmax()), forward_eV=float(energy.max()-energy[0]),
                  reverse_eV=float(energy.max()-energy[-1]), latest_sha256=file_sha(latest))
    (args.output / "block_result.json").write_text(json.dumps(result, indent=2)+"\n")
    if done and (not args.climb or 0 < result["highest_image"] < len(images)-1):
        (args.output / "passed").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("path", "template", "runtime", "pseudo", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--climb", action="store_true")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--fmax", type=float, required=True)
    run(parser.parse_args())
