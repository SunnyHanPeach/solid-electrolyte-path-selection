"""Test whether force work on the approach tracks fixed-image energy drift."""
import argparse
from collections import defaultdict
import io
from pathlib import Path

from ase.geometry import find_mic
from ase.io import read
import numpy as np
from scipy.stats import spearmanr

from audit_project_data import read_json, write_csv
from audit_signed_barriers import ARMS, file_sha, rows, trajectory_profiles
from audit_protected_trajectory_forces import STEPS
from run_force_weighting import load_bundle


def analyze(args):
    _, files = load_bundle(args.bundle)
    frames = rows(files["evaluation_frames.csv"])
    atoms = read(io.StringIO(files["evaluation.xyz"].decode()), index=":", format="extxyz")
    offsets = np.cumsum([0] + [int(row["n_atoms"]) for row in frames])
    lookup = defaultdict(list)
    for idx, row in enumerate(frames):
        lookup[row["panel"], row["edge_id"]].append((int(row["frame"]), idx))
    identity = {(r["step"], r["seed"], r["arm"]): r for r in read_json(args.force_index)["models"]}
    work = []
    component_rows = []
    movers = {}
    if args.components:
        from score_phase1_forces import references as validate_forces

        _, _, definitions, _, _ = validate_forces(args, files)
        movers = {edge: record[0] for edge, record in definitions.items() if record[0] is not None}
    for step, seed, profiles, ids, references in trajectory_profiles(args.bundle, args.energy_archive, args.project):
        if step not in STEPS:
            continue
        force_files = [identity[step, seed, arm] for arm in ARMS]
        arrays = []
        for record in force_files:
            source = args.force_outputs / record["file"]
            if file_sha(source) != record["file_sha256"]:
                raise ValueError(f"Unexpected force archive checksum: {source}")
            arrays.append(np.load(source, allow_pickle=False))
        r, a = arrays
        if a.shape != r.shape or r.shape != (offsets[-1], 3):
            raise ValueError("Force-array identity differs")
        delta = a - r
        for key, reference in references.items():
            indices = [i for _, i in sorted(lookup[key])]
            highest = min(range(1, len(reference) - 1), key=lambda j: (-reference[j], j))
            approximate_energy = 0.
            mover = movers.get(key[1])
            pieces = np.zeros((2, 2))
            image_count = len(indices) if mover is not None else highest + 1
            for segment, (i, j) in enumerate(zip(indices[:image_count-1], indices[1:image_count])):
                if not (np.array_equal(atoms[i].numbers, atoms[j].numbers) and
                        np.allclose(atoms[i].cell, atoms[j].cell, atol=1e-8, rtol=0)):
                    raise ValueError(f"Atom order or cell differs along {key}")
                displacement, _ = find_mic(atoms[j].positions - atoms[i].positions, atoms[i].cell, atoms[i].pbc)
                grad_change = -.5 * (delta[offsets[i]:offsets[i+1]] + delta[offsets[j]:offsets[j+1]])
                products = grad_change * displacement
                total = float(np.sum(products))
                if segment < highest:
                    approximate_energy += total
                if mover is not None:
                    li = float(products[mover].sum())
                    pieces[int(segment >= highest)] += (li, total - li)
            old, new = (profiles[arm][key] for arm in ARMS)
            work.append(dict(step=step, seed=seed, panel=key[0], edge_id=key[1], material_id=ids[key],
                             highest_dft_interior_image=highest+1,
                             paired_force_work_meV=1000*approximate_energy,
                             exact_fixed_interior_energy_shift_meV=1000*(new[highest]-old[highest])))
            if mover is not None:
                if not np.isclose(pieces[0].sum(), approximate_energy, atol=1e-12, rtol=0):
                    raise ValueError("Migrating Li and remaining atom contributions do not close")
                for segment, (li, other), shift in zip(
                    ("initial_to_peak", "peak_to_final"), 1000*pieces,
                    (new[highest]-old[highest], (new[-1]-old[-1])-(new[highest]-old[highest])),
                ):
                    component_rows.append(dict(step=step, panel=key[0], seed=seed, edge_id=key[1],
                                               segment=segment, migrating_Li_meV=li, remaining_atoms_meV=other,
                                               total_meV=li+other, exact_energy_drift_meV=1000*shift))
    write_csv(args.output / "paired_force_work.csv", work)
    groups = defaultdict(dict)
    for r in work:
        key = r["step"], r["panel"], r["material_id"]
        groups[key][r["seed"]] = r["paired_force_work_meV"], r["exact_fixed_interior_energy_shift_meV"]
    data = defaultdict(list)
    for (step, panel, _), cells in groups.items():
        if set(cells) != {17, 29}:
            raise ValueError("Missing paired seeds")
        data[step, panel].append([cells[17], cells[29]])
    summary = []
    for (step, panel), values in sorted(data.items()):
        per_seed = np.asarray(values, dtype=float)
        pooled = per_seed.mean(axis=1)
        x, y = pooled.T
        summary.append(dict(step=step, panel=panel, materials=len(x), mean_force_work_meV=float(x.mean()),
                            mean_exact_energy_shift_meV=float(y.mean()), material_mae_meV=float(np.abs(x-y).mean()),
                            spearman_material_mean=float(spearmanr(x, y).statistic),
                            seed17=float(spearmanr(*per_seed[:, 0, :].T).statistic),
                            seed29=float(spearmanr(*per_seed[:, 1, :].T).statistic)))
    write_csv(args.output / "force_work_summary.csv", summary)
    if args.components:
        write_csv(args.output / "force_work_components.csv", component_rows)
        groups = defaultdict(dict)
        for row in component_rows:
            groups[row["step"], row["panel"], row["segment"]].setdefault(row["edge_id"], {})[row["seed"]] = [
                row[k] for k in ("migrating_Li_meV", "remaining_atoms_meV", "total_meV", "exact_energy_drift_meV")
            ]
        report = []
        for (step, panel, segment), paths in sorted(groups.items()):
            paired = np.asarray([[s[seed] for seed in (17, 29)] for s in paths.values()])
            means = paired.mean(axis=(0, 1))
            report.append(dict(step=step, panel=panel, segment=segment, materials=len(paths),
                               Li_mean_meV=float(means[0]), remaining_atoms_mean_meV=float(means[1]),
                               total_mean_meV=float(means[2]), exact_drift_mean_meV=float(means[3])))
        write_csv(args.output / "force_work_components_summary.csv", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "energy_archive", "project", "force_outputs", "force_index", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--components", action="store_true")
    analyze(parser.parse_args())
