"""Independently confirm the CPU screen and count distinct directional hops."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from math import isfinite
from pathlib import Path

from ase.geometry import find_mic
from ase.io import read
import numpy as np

from run_phase_b4_source_pilot import read_csv


def hop_key(edge):
    material, start, end, *shift = edge.split("_")
    forward = (int(start), int(end), *(int(v) for v in shift))
    reverse = (int(end), int(start), *(-int(v) for v in shift))
    return material + ":" + ",".join(map(str, min(forward, reverse)))


def verify(entry):
    row, profile, geometry_dir = entry
    edge = row["edge_id"]
    output = dict(edge_id=edge, material_id=row["material_id"], chemsys=row["chemsys"],
                  initial_stratum="short" if float(row["initial_contact_A"]) < 1.4 else "ordinary",
                  solver_status=row["status"], screen_thresholds_passed=False,
                  canonical_direction_pair=hop_key(edge), verified=False, detail="")
    if row["status"] == "failed":
        output["detail"] = "solver_failed"
        return output
    try:
        if profile is None:
            raise ValueError("profile unavailable")
        images = read(geometry_dir / f"{edge}.extxyz", index=":")
        index = int(profile["migrating_Li_0based"])
        final = profile["final"]
        if len(images) != len(final) or len(images) != 5:
            raise ValueError("unexpected image count")
        base = images[0]
        others = np.arange(len(base)) != index
        contacts = []
        lengths = [0.]
        previous = base.positions[index]
        for image in images:
            if not (image.pbc.all() and np.array_equal(image.numbers, base.numbers)
                    and np.array_equal(image.cell, base.cell) and image.numbers[index] == 3):
                raise ValueError("geometry or mobile-site mismatch")
            distances = find_mic(image.positions[others] - image.positions[index], image.cell, image.pbc)[1]
            framework = find_mic(image.positions[others] - base.positions[others], image.cell, image.pbc)[1]
            if framework.max() > 1e-5:
                raise ValueError("framework changed")
            contacts.append(float(distances.min()))
            step = float(find_mic(image.positions[index] - previous, image.cell, image.pbc)[1])
            if len(contacts) > 1:
                lengths.append(lengths[-1] + step)
            previous = image.positions[index]
        if not (np.allclose(contacts, [p["min_Li_contact_A"] for p in final], atol=1e-5, rtol=0)
                and np.allclose(lengths, [p["arc_length_A"] for p in final], atol=1e-5, rtol=0)):
            raise ValueError("cached distances disagree with periodic-coordinate recomputation")
        energies = [float(p["bvse_energy_eV"]) for p in final]
        force = float(profile["final_neb_max_force_eV_A"])
        if not all(isfinite(v) for v in [*energies, force, *contacts, *lengths]):
            raise ValueError("nonfinite BVSE profile")
        peak = max(energies[1:-1]) - max(energies[0], energies[-1])
        passed = force <= 0.1 and min(contacts) >= 1.4 and peak >= 0.02
        if passed != (row["geometry_screen_pass"] == "True"):
            raise ValueError("cache classification disagrees with verified values")
        output.update(verified=True, verified_min_contact_A=min(contacts),
                      verified_final_length_A=lengths[-1], verified_peak_eV=peak,
                      screen_thresholds_passed=passed)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        output["detail"] = f"{type(exc).__name__}: {exc}"
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("families", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    directory = args.output
    directory.mkdir(parents=True, exist_ok=False)
    family = {row["material_id"]: row for row in read_csv(args.families)}
    entries = {}
    for run in args.runs:
        with (run / "profiles.jsonl").open(encoding="utf-8") as stream:
            profiles = {record["edge_id"]: record for record in map(json.loads, stream)}
        for row in read_csv(run / "per_path.csv"):
            edge = row["edge_id"]
            if edge in entries:
                raise ValueError(f"Duplicate path result: {edge}")
            entries[edge] = (row, profiles.get(edge), run / "preconditioned")
    accepted = []
    counts = defaultdict(Counter)
    fields = ("edge_id", "material_id", "chemsys", "initial_stratum", "solver_status", "verified",
              "verified_min_contact_A", "verified_final_length_A", "verified_peak_eV",
              "screen_thresholds_passed", "canonical_direction_pair", "parent_family_flag",
              "framework_family_flag", "detail")
    with (directory / "verified_geometry.csv").open("w", encoding="utf-8", newline="") as stream, \
            ProcessPoolExecutor(max_workers=args.workers) as pool:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for i, row in enumerate(pool.map(verify, entries.values(), chunksize=20), 1):
            flags = family[row["material_id"]]
            row.update(parent_family_flag=flags["historical_parent_flag"],
                       framework_family_flag=flags["historical_framework_flag"])
            writer.writerow(row)
            status = counts[row["initial_stratum"]]
            status["attempted"] += 1
            status[row["solver_status"]] += 1
            status["verified_geometry"] += row["verified"]
            status["passed"] += row["screen_thresholds_passed"]
            if row["screen_thresholds_passed"]:
                accepted.append(row)
            if i % 100 == 0:
                stream.flush()
    summary = dict(
        attempted=len(entries), strata=counts, accepted_directed_paths=len(accepted),
        accepted_materials=len({row["material_id"] for row in accepted}),
        accepted_direction_pairs=len({row["canonical_direction_pair"] for row in accepted}),
        accepted_unflagged_direction_pairs=len({row["canonical_direction_pair"] for row in accepted
                                                if row["parent_family_flag"] != "True"
                                                and row["framework_family_flag"] != "True"}),
        interpretation="cheap geometry and BVSE screening only; family flags are not a proof of independent evaluation",
    )
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
