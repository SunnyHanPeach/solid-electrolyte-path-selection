"""Compare the two fixed-geometry numerical controls to the frozen Gamma results."""
import argparse
import csv
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from ase.io import read
from ase.units import Bohr, Hartree, Ry

from score_phase_b4_bridge import FORCE_RE, SMEAR_RE, TOTAL_RE, fnum, sha
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
        failure_stage=("USPP force (addusforce_g)" if "addusforce_g" in log and oom else
                      "wavefunction output (ENOSPC)" if "No space left on device" in log else None),
        sampled_peak_device_memory_MiB=peak,
    )


def recovery_entry(recovery, source, variant, role):
    if recovery is None:
        return None
    name = f"{role}_{variant}"
    old = source.read_bytes()
    if variant == "k222":
        name += "_warm"
        expected = old.replace(b"&ELECTRONS\n", b"&ELECTRONS\n   startingpot      = 'file'\n")
        archived = recovery / f"{name}.in"
        if archived.exists() and archived.read_bytes() != expected:
            raise ValueError(f"{name} used unexpected warm-start fields")
    else:
        expected = old
    folder = recovery / name
    if not folder.is_dir():
        return None
    digest_file = folder / "input.sha256"
    if not digest_file.is_file():
        raise ValueError(f"{name} is missing its executed input checksum")
    digest = digest_file.read_text().split()[0]
    if digest != sha(expected):
        raise ValueError(f"{name} input checksum differs")
    if variant == "k222" and (folder / "converged").is_file():
        if "The initial density is read from file" not in (folder / "pw.out").read_text():
            raise ValueError(f"{name} did not read the starting density")
    return folder, digest


def verify_grid_xml(folder, role, xml, expected_input, manifest, job, *, expected_nks=8):
    output = ET.parse(xml).getroot().find("output")
    log = (folder / "pw.out").read_text()
    tol = manifest["electronic_conv_thr_Ry"]
    state = output.find("convergence_info/scf_conv")
    if (state.findtext("convergence_achieved") != "true"
            or float(state.findtext("scf_error")) * 2 > tol
            or int(state.findtext("n_scf_steps")) != int(ITERATIONS.findall(log)[-1])
            or "GPU acceleration is ACTIVE" not in log):
        raise ValueError(f"Unverified electronic convergence in {xml.name}")
    charge = float(output.findtext("band_structure/nelec"))
    m = float(output.findtext("magnetization/total"))
    nks = int(output.findtext("band_structure/nks"))
    if (charge != manifest["qe_valence_electrons"] or abs(m - 1.0) > .01
            or (expected_nks is not None and nks != expected_nks)):
        raise ValueError(f"Unverified charge, spin or mesh in {xml.name}")

    origin = read(expected_input, format="espresso-in")
    geometry = output.find("atomic_structure")
    atoms = list(geometry.find("atomic_positions"))
    pos = np.array([np.fromstring(a.text, sep=" ") for a in atoms])
    cell = np.array([np.fromstring(a.text, sep=" ") for a in geometry.find("cell")])
    count = manifest["first_atom_count"]
    if ([a.attrib["name"] for a in atoms] != origin.get_chemical_symbols()
            or [int(a.attrib["index"]) for a in atoms] != list(range(1, count+1))
            or not np.allclose(pos * Bohr, origin.positions, atol=1e-7, rtol=0)
            or not np.allclose(cell * Bohr, origin.cell.array, atol=1e-7, rtol=0)):
        raise ValueError(f"XML atomic geometry differs from the frozen input in {xml.name}")

    node = output.find("forces")
    if tuple(map(int, node.attrib["dims"].split())) != (3, count):
        raise ValueError(f"Invalid XML force shape in {xml.name}")
    forces = np.fromstring(node.text, sep=" ").reshape(count, 3) * Hartree / Bohr
    energy = float(output.findtext("total_energy/etot")) * Hartree
    smear = float(output.findtext("total_energy/demet")) * Hartree
    partial = FORCE_RE.findall(log)
    if not partial or [int(row[0]) for row in partial] != list(range(1, len(partial)+1)):
        raise ValueError(f"Failed log force consistency check in {xml.name}")
    reference = np.array([[fnum(x) for x in row[1:]] for row in partial]) * Ry / Bohr
    max_diff = float(np.max(np.abs(forces[:len(partial)] - reference)))
    if (max_diff > 1.4e-7 or abs(energy - fnum(TOTAL_RE.findall(log)[-1]) * Ry) > 1e-7
            or abs(smear - fnum(SMEAR_RE.findall(log)[-1]) * Ry) > 1e-7
            or not np.isfinite(forces).all()):
        raise ValueError(f"XML does not agree with the stdout prefix in {xml.name}")
    job.update(electrons=charge, total_magnetization_muB=m, total_energy_eV=energy,
               absolute_magnetization_muB=float(output.findtext("magnetization/absolute")), smearing_term_eV=smear,
               migrating_li_force_eV_A=forces[manifest["migrating_Li_0based"]].tolist(),
               xml_sha256=sha(xml.read_bytes()), force_reference_log_rows=len(partial),
               xml_stdout_force_max_abs_difference_eV_A=max_diff, complete_xml_force_vectors=count,
               nks_ibz=nks, scf_error_Ry=float(state.findtext("scf_error")) * 2,
               energy_force_source="QE data-file-schema.xml, geometry and stdout cross-checked")
    return job, forces


def score(baseline, numeric, recovery=None):
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
            old_status = "converged"
            if not (folder / "converged").is_file():
                old_status, attempt = describe_incomplete(folder, name, input_hash)
                if attempt is not None:
                    incomplete_attempts.append(attempt)
            replacement = recovery_entry(recovery, numeric / "inputs" / f"{name}.in", variant, role)
            if replacement is not None:
                if old_status == "converged":
                    raise ValueError(f"Unexpected duplicate of the approved result for {name}")
                folder, input_hash = replacement
            xml = recovery / f"{role}_schema.xml" if replacement is not None and variant == "k222" else None
            if (folder / "converged").is_file():
                run, new_forces = read_job(folder, dict(old, sha256=input_hash), manifest)
                if xml is not None and xml.is_file():
                    run, new_forces = verify_grid_xml(folder, role, xml, numeric / "inputs" / f"{name}.in", manifest, run)
                status[role] = "converged"
            else:
                if replacement is None:
                    status[role] = old_status
                    continue
                status[role], attempt = describe_incomplete(folder, folder.name, input_hash)
                if attempt is not None:
                    incomplete_attempts.append(attempt)
                if xml is None or not xml.is_file() or attempt is None or attempt["failure_stage"] != "wavefunction output (ENOSPC)":
                    continue
                run, new_forces = verify_grid_xml(
                    folder, role, xml, numeric / "inputs" / f"{name}.in", manifest,
                    dict(role=role, image=old["original_image"], spin=old["spin"],
                         input_sha256=input_hash, log_sha256=attempt["log_sha256"], elapsed_seconds=attempt["elapsed_seconds"],
                         iterations=attempt["converged_iterations"], sampled_peak_device_memory_MiB=attempt["sampled_peak_device_memory_MiB"],
                         exit_status=attempt["exit_status"], wavefunction_save_incomplete=True))
                status[role] = "complete_scf_and_forces_from_xml_checkpoint_write_failed"
            run["numeric_variant"] = variant
            jobs[role] = (run, new_forces)
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
    parser.add_argument("--recovery", type=Path)
    args = parser.parse_args()
    result = score(args.baseline, args.numerical_runs, args.recovery)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
