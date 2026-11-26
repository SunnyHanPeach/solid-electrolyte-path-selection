"""Score a two-geometry QE spin screen from the archived raw output."""
import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np
from ase.units import Bohr, Ry

from score_phase_b4_bridge import FORCE_RE, SMEAR_RE, TOTAL_RE, fnum, sha


ELECTRONS = re.compile(r"number of electrons\s*=\s*(\S+)")
ITERATIONS = re.compile(r"convergence has been achieved in\s+(\d+)\s+iterations")
MAGNETIZATION = re.compile(r"^\s*total magnetization\s*=\s*(\S+)", re.M)
ABS_MAGNETIZATION = re.compile(r"^\s*absolute magnetization\s*=\s*(\S+)", re.M)
THREADS = re.compile(r"Threads/MPI process:\s*(\d+)")


def last_number(pattern, text):
    values = pattern.findall(text)
    return fnum(values[-1]) if values else None


def read_job(root, job, manifest):
    path = root / "runs" / f"{job['role']}_{job['spin']}"
    raw = (path / "pw.out").read_bytes()
    log = raw.decode()
    if "JOB DONE." not in log or "GPU acceleration is ACTIVE" not in log or not ITERATIONS.search(log):
        raise ValueError(f"{path.name} has no converged GPU result")

    electrons = last_number(ELECTRONS, log)
    magnetization = last_number(MAGNETIZATION, log)
    expected_magnetization = job["tot_magnetization_muB"]
    if electrons != manifest["qe_valence_electrons"]:
        raise ValueError(f"{path.name} has unexpected electron count {electrons}")
    if expected_magnetization is not None:
        if magnetization is None or abs(magnetization - expected_magnetization) > 0.01:
            raise ValueError(f"{path.name} has unexpected magnetization {magnetization}")

    count = manifest["first_atom_count"]
    rows = FORCE_RE.findall(log)[-count:]
    if [int(row[0]) for row in rows] != list(range(1, count + 1)):
        raise ValueError(f"{path.name} has missing or reordered force rows")
    forces = np.array([[fnum(x) for x in row[1:]] for row in rows]) * (Ry / Bohr)
    energy = last_number(TOTAL_RE, log)
    smearing = last_number(SMEAR_RE, log)
    if energy is None or smearing is None or not np.isfinite([energy, smearing]).all() or not np.isfinite(forces).all():
        raise ValueError(f"{path.name} has missing or invalid energy or force values")

    state = dict(line.split("=", 1) for line in (path / "runtime.txt").read_text().splitlines())
    if int(state["exit_status"]) != 0:
        raise ValueError(f"{path.name} reports exit status {state['exit_status']}")
    with (path / "gpu_samples.csv").open(newline="") as f:
        next(f)
        memory = [int(row[1].strip().split()[0]) for row in csv.reader(f) if len(row) == 3]

    return dict(
        role=job["role"], image=job["original_image"], spin=job["spin"],
        input_sha256=job["sha256"], log_sha256=sha(raw),
        electrons=electrons, total_magnetization_muB=magnetization,
        absolute_magnetization_muB=last_number(ABS_MAGNETIZATION, log),
        total_energy_eV=energy * Ry, smearing_term_eV=smearing * Ry,
        migrating_li_force_eV_A=forces[manifest["migrating_Li_0based"]].tolist(),
        iterations=int(last_number(ITERATIONS, log)), threads_per_rank=int(last_number(THREADS, log)),
        start_utc=state["start_utc"], elapsed_seconds=int(state["elapsed_seconds"]),
        sampled_peak_device_memory_MiB=max(memory, default=None),
    ), forces


def score(root, endpoint_only):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    roles = ("endpoint",) if endpoint_only else ("endpoint", "peak")
    jobs = [job for job in manifest["jobs"] if job["role"] in roles]
    runs = {(job["role"], job["spin"]): read_job(root, job, manifest) for job in jobs}
    branches = ("nspin1", "spin1muB")
    energy = {}
    forces = {}
    for role in roles:
        reference, constrained = (runs[role, branch] for branch in branches)
        delta = constrained[1] - reference[1]
        forces[role] = dict(
            force_component_rmse_eV_A=float(np.sqrt(np.mean(delta ** 2))),
            migrating_li_force_difference_norm_eV_A=float(np.linalg.norm(delta[manifest["migrating_Li_0based"]])),
        )
    result = dict(
        edge_id=manifest["edge_id"], input_manifest_sha256=sha(manifest_path.read_bytes()),
        energies="QE finite Gaussian smearing energy, with secondary E-0.5*(-TS) estimate",
        jobs=[run for run, _ in runs.values()],
        force_comparison=forces, thresholds=manifest["gate"],
        total_gpu_wall_seconds=sum(run["elapsed_seconds"] for run, _ in runs.values()),
    )
    if not endpoint_only:
        for branch in branches:
            endpoint, peak = (runs[role, branch][0] for role in roles)
            energy[branch] = dict(
                fixed_image_gap_meV=1000 * (peak["total_energy_eV"] - endpoint["total_energy_eV"]),
                smearing_difference_meV=1000 * (peak["smearing_term_eV"] - endpoint["smearing_term_eV"]),
            )
        gap_nspin1, gap_spin = (energy[b]["fixed_image_gap_meV"] for b in branches)
        smear_nspin1, smear_spin = (energy[b]["smearing_difference_meV"] for b in branches)
        shift = gap_spin - gap_nspin1
        result.update(geometric_gap_by_spin=energy, spin_minus_nonspin_gap_meV=shift,
                      gaussian_zero_smearing_gap_shift_meV=shift - .5 * (smear_spin - smear_nspin1),
                      spin_screen_pass=(abs(shift) <= manifest["gate"]["energy_screen_meV"] and
                          all(row["force_component_rmse_eV_A"] <= manifest["gate"]["force_component_rmse_screen_eV_A"] and
                              row["migrating_li_force_difference_norm_eV_A"] <= manifest["gate"]["migrating_li_force_difference_norm_screen_eV_A"]
                              for row in forces.values())))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--endpoint-only", action="store_true")
    args = parser.parse_args()
    result = score(args.bundle, args.endpoint_only)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
