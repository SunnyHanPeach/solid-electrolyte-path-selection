"""Write a read-only, timestamped acceptance snapshot for the QE pilot."""
import argparse
from datetime import datetime
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
import re
import shutil
import subprocess
from time import sleep


FORCE = re.compile(r"^\s*atom\s+(\d+)\s+type\s+\d+\s+force\s*=\s*(.+)$", re.MULTILINE)


def text(path):
    return path.read_text(errors="replace") if path.is_file() else ""


def hashes(folder, inventory):
    source = folder / inventory
    entries = [line.split(maxsplit=1) for line in text(source).splitlines() if line.strip()]
    checks = {}
    for expected, relative in entries:
        name = relative.strip()
        path = folder / name
        checks[name] = path.is_file() and sha256(path.read_bytes()).hexdigest() == expected
    return checks


def endpoint(root, role, farm, process_lines):
    folder = root / farm / role
    receipt = root / f"{role}_receipt"
    output = text(folder / "pw.out")
    iterations = re.findall(r"iteration #\s*(\d+)", output)
    scf_cycles = re.findall(r"convergence has been achieved in\s+(\d+)", output)
    block = {}
    maximum = None
    for match in FORCE.finditer(output):
        index = int(match[1])
        if index == 1:
            block = {}
        values = match[2].split()
        if len(values) == 3 and index not in block:
            block[index] = [float(v.replace("D", "E")) for v in values]
        if set(block) == set(range(1, 88)):
            maximum = max(sqrt(sum(v*v for v in row))*25.71103374 for row in block.values())
    runtime = [line for line in process_lines if f"/{role}.in" in line and "pw.x -in" in line]
    save = folder / "scratch" / "b4_mp1191903_endpoint_spin1muB.save"
    checks = hashes(receipt, "output.sha256")
    validated = (receipt / "validated").is_file() and bool(checks) and all(checks.values())
    errors = [line.strip() for line in output.splitlines() if any(key in line.lower() for key in
              ("error in routine", "out of memory", "no space left", "maximum cpu time exceeded"))]
    return dict(started=bool(output), live_process=runtime, completed_scf_cycles=len(scf_cycles),
                last_electronic_iteration=int(iterations[-1]) if iterations else None,
                latest_full_force_max_eV_A=maximum, validated=validated, output_checks=checks,
                persistent_files=dict(qe_xml=(save / "data-file-schema.xml").is_file(),
                                      density=(save / "charge-density.dat").is_file(),
                                      wavefunction_file_count=sum(1 for _ in save.glob("wfc*.dat"))),
                error_markers=errors[-6:])


def snapshot(root, scheduled):
    ps = subprocess.check_output(["ps", "-eo", "pid,ppid,etime,stat,args"], text=True).splitlines()
    gpu_query = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True)
    a, b = [endpoint(root, role, farm, ps) for role, farm in (
        ("endpoint_a", "work_a"), ("endpoint_b", "work_b"))]
    checks = {**hashes(root, "prepared_inputs.sha256"), **hashes(root, "path_seed.sha256")}
    queue = text(root / "endpoint_queue_status.txt")
    path_queue = text(root / "path_smoke_status.txt")
    valid_inputs = len(checks) == 3 and all(checks.values())
    probe_file = root / "path5_probe" / "block_result.json"
    probe = None
    if probe_file.is_file():
        try:
            probe = json.loads(text(probe_file))
        except json.JSONDecodeError:
            probe = {"status": "still_writing"}
    status = "in_progress"
    if (not valid_inputs or (queue and "exit_status=0\n" not in queue)
            or (path_queue and "exit_status=0\n" not in path_queue)):
        status = "attention_required"
    elif a["validated"] and b["validated"]:
        status = "endpoint_gate_passed"
        if probe and "exit_status=0\n" in path_queue:
            status = "path_timing_probe_passed"
    elif not (a["live_process"] or b["live_process"] or any(
            line.endswith("bash ./run_mp1191903_relax.sh") for line in ps)):
        status = "attention_required"
    cgroup = Path("/sys/fs/cgroup")
    return dict(scheduled_at=scheduled.isoformat(), observed_at=datetime.now().astimezone().isoformat(),
                status=status, input_checks=checks, endpoint_a=a, endpoint_b=b,
                endpoint_queue_exit=queue.strip(), path_probe_exit=path_queue.strip(), path_probe=probe,
                gpu=gpu_query.stdout.strip(), gpu_query_exit=gpu_query.returncode,
                free_disk_bytes=dict(system=shutil.disk_usage(root / "work_a").free,
                                     data=shutil.disk_usage(root).free),
                container_memory=dict(current_bytes=text(cgroup / "memory.current").strip(),
                                      limit_bytes=text(cgroup / "memory.max").strip(),
                                      events=text(cgroup / "memory.events").strip()),
                scope="endpoint and path timing acceptance; checkpoint presence alone does not certify restart")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--at", required=True, type=datetime.fromisoformat)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.at.tzinfo is None:
        parser.error("--at requires an explicit UTC offset")
    print(f"armed_for={args.at.isoformat()}", flush=True)
    while (remaining := (args.at - datetime.now().astimezone()).total_seconds()) > 0:
        sleep(min(60, remaining))
    outcome = snapshot(args.root, args.at)
    pending = args.output.with_suffix(".partial")
    with pending.open("x", encoding="utf-8") as file:
        json.dump(outcome, file, ensure_ascii=False, indent=2)
        file.write("\n")
    pending.replace(args.output)
    print(f"status={outcome['status']}", flush=True)
