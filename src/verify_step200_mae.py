"""Independently reproduce final two-panel paired forward-barrier MAE point estimates."""
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import fmean
import sys

from audit_signed_barriers import load_frozen_data


def main(bundle, project, raw_runs, score, output):
    _, _, references = load_frozen_data(bundle, project)
    errors = {}
    for seed in (17, 29):
        for arm in ("replay", "full", "energy_weighted", "matched_uniform"):
            profiles = defaultdict(dict)
            with (raw_runs / f"seed{seed}" / arm / "predictions_400.csv").open(newline="") as file:
                for row in csv.DictReader(file):
                    profiles[row["panel"], row["edge_id"]][int(row["frame"])] = float(row["energy_eV"])
            for key, vals in profiles.items():
                pred = max(vals.values()) - vals[min(vals)]
                ref = max(references[key])
                errors[seed, arm, key] = abs(pred - ref) * 1000
    verified = []
    with (score / "contrasts.csv").open(newline="") as file:
        for row in csv.DictReader(file):
            if row["step"] != "400" or row["metric"] != "barrier_absolute_error":
                continue
            a, b = row["contrast"].split("_minus_")
            panel = row["panel"]
            group = [key for key in references if key[0] == panel]
            seed_means = [fmean(errors[seed, a, key] - errors[seed, b, key] for key in group)
                          for seed in (17, 29)]
            verified_mean = fmean(seed_means)
            candidates = [row["mean_meV"], row["seed17_meV"], row["seed29_meV"]]
            checks = [abs(x - float(y)) for x, y in zip([verified_mean, *seed_means], candidates)]
            if max(checks) > 1e-8:
                raise ValueError(f"Independent score differs: {panel}/{row['contrast']}")
            verified.append(dict(panel=panel, contrast=row["contrast"], mean_meV=verified_mean,
                                 material_count=len(group), max_discrepancy_meV=max(checks)))
    if len(verified) != 12:
        raise ValueError("Expected all 12 panel/contrast combinations")
    output.write_bytes((json.dumps(verified, indent=2) + "\n").encode())


if __name__ == "__main__":
    main(*(Path(value) for value in sys.argv[1:]))
