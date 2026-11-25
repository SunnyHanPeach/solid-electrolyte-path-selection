"""Independently reconstruct the B1 core summaries from archived raw tables."""
import argparse
from collections import defaultdict
import csv
import hashlib
from itertools import combinations
import json
from math import dist, fsum, isclose, sqrt
from pathlib import Path
from statistics import mean, stdev


def load(directory, filename):
    with (directory/filename).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def compare(actual, expected, context, gaps):
    if not isclose(actual, float(expected), rel_tol=0., abs_tol=1e-8):
        raise ValueError(f"Independent check failed: {context}")
    gaps[context] = max(gaps[context], abs(actual - float(expected)))


def pairwise_spread(vectors):
    count = len(vectors)
    return sqrt(fsum(dist(a, b)**2 for a, b in combinations(vectors, 2)) / (count * (count - 1)))


def verify(directory):
    raw = load(directory, "raw_image_predictions.csv")
    atoms = load(directory, "raw_atomic_predictions.csv")
    published = load(directory, "image_disagreement.csv")
    by_path = {r["edge_id"]: r for r in load(directory, "path_summary.csv")}
    per_seed = {(r["edge_id"], r["seed"]): r for r in load(directory, "per_seed_barriers.csv")}
    atom_groups, by_seed = defaultdict(list), defaultdict(list)
    for r in atoms:
        atom_groups[r["edge_id"], int(r["image"]), r["seed"]].append(r)
    for r in raw:
        by_seed[r["edge_id"], r["seed"]].append(r)
    gaps = defaultdict(float)
    series = {}
    forces = {}
    for (edge, seed), records in by_seed.items():
        records.sort(key=lambda r: int(r["image"]))
        values = []
        for i, row in enumerate(records, 1):
            assert int(row["image"]) == i
            group = atom_groups[edge, i, seed]
            assert len(group) == int(row["n_atoms"])
            assert {r["atom_1based"] for r in group} == {str(j) for j in range(1, len(group)+1)}
            moving, = [r for r in group if r["is_moving_Li"] == "1"]
            assert moving["atom_1based"] == row["Li_1based"] and moving["element"] == "Li"
            forces[edge, i, seed] = tuple(float(moving[f"F{v}_eV_per_A"]) for v in "xyz")
            value = fsum(float(r["atomic_model_energy_eV"]) for r in group)
            compare(value, row["total_energy_eV"], "node_to_total_eV", gaps)
            values.append(value)
        series[edge, seed] = values
        compare(1000. * (max(values) - values[0]), per_seed[edge, seed]["forward_barrier_meV"], "barrier_meV", gaps)
        assert values.index(max(values)) + 1 == int(per_seed[edge, seed]["peak_image"])
    seeds = ("17", "29", "43")
    for edge, path in by_path.items():
        values = [1000. * (max(series[edge, s]) - series[edge, s][0]) for s in seeds]
        compare(mean(values), path["forward_barrier_mean_meV"], "barrier_mean_meV", gaps)
        compare(stdev(values), path["forward_barrier_sd_meV"], "barrier_sd_meV", gaps)
    for row in published:
        edge, i = row["edge_id"], int(row["image"])
        t = tuple(float(row[f"tangent_{v}"]) for v in "xyz")
        vectors = [forces[edge, i, s] for s in seeds]
        parallel = [fsum(v * w for v, w in zip(t, f)) for f in vectors]
        normal = [tuple(v - p * w for v, w in zip(f, t)) for f, p in zip(vectors, parallel)]
        compare(pairwise_spread(vectors), row["Li_force_sd"], "Li_force_sd_eV_per_A", gaps)
        compare(stdev(parallel), row["Li_tangent_sd"], "Li_tangent_sd_eV_per_A", gaps)
        compare(pairwise_spread(normal), row["Li_normal_sd"], "Li_normal_sd_eV_per_A", gaps)
        e = [series[edge, s][i-1] for s in seeds]
        compare(1000. * stdev(e), row["energy_raw_sd"], "energy_raw_sd_meV", gaps)
        compare(1000. * stdev(v - mean(series[edge, s]) for v, s in zip(e, seeds)),
                row["energy_centered_sd"], "energy_centered_sd_meV", gaps)
        compare(1000. * stdev(v - series[edge, s][0] for v, s in zip(e, seeds)),
                row["energy_from_start_sd"], "energy_from_start_sd_meV", gaps)
    assert len(raw) == len(atom_groups) == 51 and len(published) == 17 and len(by_path) == 3
    manifests = (("run_manifest.json", "output_sha256"), ("analysis_manifest.json", "outputs"))
    for name, field in manifests:
        for output, expected in json.loads((directory/name).read_bytes())[field].items():
            assert hashlib.sha256((directory/output).read_bytes()).hexdigest() == expected
    return dict(status="pass", inputs=dict(predictions=len(raw), images=len(published), atomic_rows=len(atoms),
                     paths=len(by_path)), max_absolute_difference=dict(gaps),
                verification_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = verify(args.output)
    with (args.output/"verification.json").open("x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
