#!/usr/bin/env bash
# Refine the converged mp-5012 band and keep the previous checkpoint intact.
set -euo pipefail
if (( $# != 3 )); then
    echo "Usage: $0 SAVED_RUN RECORD_DIRECTORY BUDGET_SECONDS" >&2
    exit 2
fi
old=$1
record=$2
budget=$3
if (( budget < 300 )); then exit 2; fi
mkdir "$record"
record=$(cd "$record" && pwd)
exec 9>"$(dirname "$old")/mp-5012.lock"
flock -n 9
export TZ=Asia/Shanghai OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
work=$(mktemp -d "$(dirname "$old")/mp-5012-CI-XXXXXX")
printf '%s\n' "$work" > "$record/wsl_workdir"
status() { printf '%s\n' "$1" > "$record/state.tmp"; mv "$record/state.tmp" "$record/state"; }
timer=
trap 'code=$?; if [[ -n "$timer" ]]; then kill "$timer" 2>/dev/null || true; fi; if (( code )); then status "error:$code"; fi' EXIT
status preparing

python3 - "$old" "$work" "$record" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys

old, work, record = map(Path, sys.argv[1:])
assets = {name: (old / name).read_bytes() for name in (
    "resume_path.in", "lihop_neb.path", "lihop_neb.dat")}
text = assets["resume_path.in"].decode()
restart = assets["lihop_neb.path"].decode()
profile = [list(map(float,line.split())) for line in assets["lihop_neb.dat"].decode().splitlines()]
images = re.split(r"^Image:\s+\d+\s*$", restart, flags=re.M)[1:]
nstep = int(restart.splitlines()[1])
if len(images) != 5 or len(profile) != 5 or max(row[2] for row in profile) > .10:
    raise ValueError("A converged five-image standard NEB band is required")
peak = max(range(5), key=lambda i: profile[i][1])
if not 0 < peak < 4:
    raise ValueError("The converged path must have an interior peak")
blocks = list(re.finditer(r"ATOMIC_POSITIONS angstrom\s*\n((?:[^\n]+\n){95})", text))
if len(blocks) != 5:
    raise ValueError("Unexpected input geometry")
parts = [[line.split() for line in image.splitlines() if line.strip()] for image in images]
energies = [float(lines[0][0]) for lines in parts]
for j, values in enumerate(parts):
    if abs((energies[j]-energies[0])*27.211386245988-profile[j][1]) > 5e-5:
        raise ValueError(f"Image {j+1} has inconsistent saved energies")
bohr = .529177210903
for block, values in reversed(list(zip(blocks, parts))):
    symbols = [line.split()[0] for line in block[1].splitlines()]
    coords = [[bohr*float(x) for x in row[:3]] for row in values[1:96]]
    if len(coords) != 95 or not all(math.isfinite(x) for row in coords for x in row):
        raise ValueError("Unexpected saved positions")
    replacement = "".join(f"{symbol:3s} {x:.11f} {y:.11f} {z:.11f}\n"
                          for symbol, (x,y,z) in zip(symbols,coords))
    text = text[:block.start(1)] + replacement + text[block.end(1):]
if text.count("CI_scheme='no-CI'") != 1 or text.count("END_PATH_INPUT") != 1:
    raise ValueError("Wrong standard-NEB input template")
text = text.replace("CI_scheme='no-CI'", "CI_scheme='manual'")
text = text.replace("END_PATH_INPUT", f"CLIMBING_IMAGES\n{peak+1}\nEND_PATH_INPUT")
payload = text.encode()
(work / "ci.in").write_bytes(payload)
(record / "ci.in").write_bytes(payload)
shutil.copytree(old / "pseudo", work / "pseudo")
(work / "scratch").mkdir()
(record / "manifest.json").write_text(json.dumps(dict(
    path="mp-5012/P", source_steps=nstep, source_barrier_eV=profile[peak][1]-profile[0][1],
    source_max_residual_eV_A=max(row[2] for row in profile), climbing_image=peak+1,
    path_threshold_eV_A=.10, ranks=9, image_groups=3,
    settings="Original cell, pseudopotentials and DFT settings; fresh SCF from the saved path geometries.",
    source_hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in assets.items()},
    pseudo_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((work/"pseudo").iterdir())},
    input_sha256=hashlib.sha256(payload).hexdigest(),
),indent=2)+"\n")
PY

cd "$work"
date --iso-8601=seconds > "$record/started"
(
    sleep "$((budget - 120))"
    date --iso-8601=seconds > "$record/budget_stop_requested"
    touch lihop_neb.EXIT scratch/lihop_neb.EXIT
) &
timer=$!
status running
started=$(date +%s)
set +e
/usr/bin/time -v -o "$record/process.time" timeout --signal=TERM --kill-after=45s "${budget}s" \
    mpirun --bind-to none -np 9 /usr/bin/neb.x -ni 3 -inp ci.in < /dev/null > "$record/ci.log" 2>&1
code=$?
set -e
ended=$(date +%s)
date --iso-8601=seconds > "$record/ended"
printf '{"wall_seconds":%s,"allocated_mpi_rank_seconds":%s,"exit_code":%s}\n' \
    "$((ended-started))" "$((9*(ended-started)))" "$code" > "$record/cost.json"
for name in lihop_neb.path lihop_neb.dat lihop_neb.xyz lihop_neb.int; do
    if [[ -s $name ]]; then cp -- "$name" "$record/$name"; fi
done
if (( code == 0 )) && grep -qiE '^[[:space:]]*neb:[[:space:]]+convergence achieved' "$record/ci.log"; then
    status converged
elif [[ -s "$record/budget_stop_requested" && -s "$record/lihop_neb.path" ]]; then
    status budget_stopped_checkpoint_saved
elif (( code == 0 || code == 124 || code == 137 )); then
    status stopped_or_not_converged
else
    status "solver_error:$code"
fi
