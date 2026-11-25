"""Enforce the B2 wall-clock limit on the registered launcher and its own children."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
from time import sleep, time


def process_identity(proc):
    try:
        stat = (proc/"stat").read_text().rsplit(") ", 1)[1].split()
        return stat[19] if stat[0] not in ("Z", "X") else None
    except FileNotFoundError:
        return None


def family(pid):
    try:
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
    except FileNotFoundError:
        children = []
    return [int(pid), *(descendant for child in children for descendant in family(child))]


def guard(pid_file, status_path, launcher):
    pid = int(pid_file.read_text().strip())
    proc = Path(f"/proc/{pid}")
    marker = process_identity(proc)
    if marker is None:
        return
    cmd = (proc/"cmdline").read_bytes().split(b"\0")
    if launcher.encode() not in cmd or b"launch" not in cmd:
        raise ValueError("PID does not identify the B2 launcher")
    while not status_path.is_file():
        if process_identity(proc) != marker:
            return
        sleep(2)
    status = json.loads(status_path.read_bytes())
    deadline = datetime.fromisoformat(status["deadline_utc"]).timestamp()
    while time() < deadline:
        if process_identity(proc) != marker:
            return
        sleep(max(0., min(30., deadline-time())))
    if process_identity(proc) != marker:
        return
    target = {child:process_identity(Path(f"/proc/{child}")) for child in family(pid)}
    for child in reversed(target):
        if target[child] and process_identity(Path(f"/proc/{child}")) == target[child]:
            try:
                os.kill(child, signal.SIGTERM)
            except ProcessLookupError:
                pass
    evidence = dict(state="deadline_stopped", deadline_utc=status["deadline_utc"], terminated_processes=list(target),
                    recorded_utc=datetime.now(timezone.utc).isoformat())
    with (status_path.parent/"deadline_stop.json").open("x",encoding="utf-8") as f:
        json.dump(evidence,f,indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pid_file",type=Path)
    p.add_argument("status_path",type=Path)
    p.add_argument("launcher")
    guard(**vars(p.parse_args()))
