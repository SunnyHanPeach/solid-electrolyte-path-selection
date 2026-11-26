"""Compare the two fixed-geometry numerical controls to the frozen Gamma results."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from score_phase_b4_bridge import FORCE_RE, TOTAL_RE, fnum, sha
from score_phase_b4_spin_pilot import ITERATIONS, read_job


def energy_gap(jobs, field):
    return 1000 * (jobs["peak"][0][field] - jobs["endpoint"][0][field])


def check_input(source, candidate, variant):
    text = source.read_bytes()
    if variant == "k222":
        expected = text.replace(b"K_POINTS gamma\n", b"K_POINTS automatic\n2 2 2 0 0 0\n")
        expected = expected.replace(b"&CONTROL\n", b"&CONTROL\n   disk_io = 'medium'\n")
    else:
        expected = text.replace(b"ecutwfc          = 50\n", b"ecutwfc          = 62.5\n")
        expected = expected.replace(b"ecutrho          = 400\n", b"ecutrho          = 500\n")
    raw = candidate.read_bytes()
    if text == expected or raw != expected:
        raise ValueError(f"{candidate} differs from the frozen input in unexpected fields")
    return sha(raw)


def describe_incomplete(folder, job_name, input_hash):
    log_path = folder / "pw.out"
    if not log_path.is_file():
        return "not_run", None
    raw = log_path.read_bytes()
    log = raw.decode()
    state = dict(line.split("=", 1) for line in (folder / "runtime.txt").read_text().splitlines())
    iterations = ITERATIONS.findall(log)
    energy = TOTAL_RE.findall(log)
    oom = "CUDA_ERROR_OUT_OF_MEMORY" in log
    complete_forces = len(FORCE_RE.findall(log)) == 87
    status = "scf_converged_force_oom" if iterations and oom and not complete_forces else "incomplete"
    with (folder / "gpu_samples.csv").open(newline="") as f:
        next(f)
        peak = max(int(row[1].strip().split()[0]) for row in csv.reader(f) if len(row) == 3)
    return status, dict(
        job=job_name, input_sha256=input_hash, log_sha256=sha(raw),
        elapsed_seconds=int(state["elapsed_seconds"]), exit_status=int(state["exit_status"]),
        converged_iterations=int(iterations[-1]) if iterations else None,
        provisional_total_energy_Ry=fnum(energy[-1]) if energy else None,
        force_rows=len(FORCE_RE.findall(log)), complete_forces=complete_forces,
        failure_stage="USPP force (addusforce_g)" if "addusforce_g" in log and oom else None,
        sampled_peak_device_memory_MiB=peak,
    )


def score(baseline, numeric):
    manifest_file = baseline / "manifest.json"
    manifest = json.loads(manifest_file.read_bytes())
    frozen = {job["role"]: job for job in manifest["jobs"] if job["spin"] == "spin1muB"}
    base = {role: read_job(baseline / "runs" / f"{role}_spin1muB", job, manifest)
            for role, job in frozen.items()}
    moving = manifest["migrating_Li_0based"]
    gate = manifest["gate"]
    rows = {}
    for variant in ("k222", "cut125"):
        jobs = {}
        status = {}
        comparisons = {}
        incomplete_attempts = []
        for role, old in frozen.items():
            name = f"{role}_{variant}"
            input_hash = check_input(baseline / old["input_file"], numeric / "inputs" / f"{name}.in", variant)
            folder = numeric / name
            if not (folder / "converged").is_file():
                status[role], attempt = describe_incomplete(folder, name, input_hash)
                if attempt is not None:
                    incomplete_attempts.append(attempt)
                continue
            run, new_forces = read_job(folder, dict(old, sha256=input_hash), manifest)
            run["numeric_variant"] = variant
            jobs[role] = (run, new_forces)
            status[role] = "converged"
            delta = new_forces - base[role][1]
            comparisons[role] = dict(force_component_rmse_eV_A=float(np.sqrt(np.mean(delta ** 2))),
                                     migrating_li_force_difference_norm_eV_A=float(np.linalg.norm(delta[moving])))
        rows[variant] = dict(status=status, jobs=[row for row, _ in jobs.values()], force_comparison=comparisons,
                             incomplete_attempts=incomplete_attempts)
        if len(jobs) != len(frozen):
            continue
        finite_smear_shift = energy_gap(jobs, "total_energy_eV") - energy_gap(base, "total_energy_eV")
        smear_term_shift = energy_gap(jobs, "smearing_term_eV") - energy_gap(base, "smearing_term_eV")
        rows[variant].update(
            gap_meV=energy_gap(jobs, "total_energy_eV"), gap_shift_meV=finite_smear_shift,
            gaussian_zero_smearing_gap_shift_meV=finite_smear_shift - 0.5 * smear_term_shift,
            screen_pass=abs(finite_smear_shift) <= gate["energy_screen_meV"] and
            all(row["force_component_rmse_eV_A"] <= gate["force_component_rmse_screen_eV_A"] and
                row["migrating_li_force_difference_norm_eV_A"] <= gate["migrating_li_force_difference_norm_screen_eV_A"]
                for row in comparisons.values()),
        )
    return dict(edge_id=manifest["edge_id"], fixed_total_magnetization_muB=1.0,
                baseline_manifest_sha256=sha(manifest_file.read_bytes()),
                baseline_gamma_50_400_gap_meV=energy_gap(base, "total_energy_eV"),
                thresholds=gate, comparisons=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("numerical_runs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = score(args.baseline, args.numerical_runs)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
