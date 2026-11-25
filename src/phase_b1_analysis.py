"""Analyze B1 predictions and existing A1/A2 caches without loading models."""
import argparse
from collections import defaultdict
from importlib.metadata import version
from itertools import combinations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from phase_a0_geometry import PATHS, export_csv
from phase_b1_inference import SEEDS, file_sha, read_json, rows, strict_check, write_json

REPRESENTATIONS = dict(soap="soap_geometry", coordination_smooth="coordination_smooth",
                       coordination_hard="coordination_hard", activation="mace_invariants")
REFERENCES = ("deduplicated", "raw", "material_balanced", "exclude_target_material")
DESCRIPTORS = ("soap", "coordination_smooth", "activation")
FORCES = ("Li_force_sd", "Li_tangent_sd", "Li_normal_sd")
ENERGIES = ("energy_centered_sd", "energy_raw_sd", "energy_from_start_sd")
LABELS = dict(soap="SOAP distance", activation="MACE Activation distance",
              coordination_smooth="Smooth coordination distance", coordination_hard="Hard coordination distance",
              energy_centered_sd="Centered energy SD (meV)", energy_raw_sd="Total energy SD (meV)",
              energy_from_start_sd="Start-referenced energy SD (meV)", Li_force_sd="Li force vector SD (eV/Å)",
              Li_tangent_sd="Li signed tangent SD (eV/Å)", Li_normal_sd="Li normal vector SD (eV/Å)")
PROFILE_LABELS = ("SOAP distance", "Smooth CN distance", "Activation distance", "Energy SD (meV)",
                  "Li vector SD (eV/Å)", "Li tangent SD (eV/Å)", "Li normal SD (eV/Å)")
COLORS = ("#0072B2", "#D55E00", "#009E73")
SHORT = {key: key.split("_")[0] for key in PATHS}


def index(rows_in, keys):
    result = {tuple(r[k] for k in keys): r for r in rows_in}
    strict_check(len(result) == len(rows_in), f"Duplicated keys: {keys}")
    return result


def cache_columns(repo, out, manifest, geometry):
    for name, digest in manifest["inputs_sha256"].items():
        if name.startswith("results/"):
            strict_check(file_sha(repo/name) == digest, f"B1 input changed: {name}")
    for name, digest in manifest["output_sha256"].items():
        strict_check(file_sha(out/name) == digest, f"Inference data changed: {name}")
    strict_check(file_sha(out/"smoke.json") == manifest["smoke_sha256"], "Acceptance record changed")
    a1 = repo/"results/phase_a1"
    a2 = repo/"results/phase_a2"
    a1_manifest = read_json(a1/"manifest.json")
    strict_check(a1_manifest["checkpoint_sha256"] == manifest["checkpoints"][0]["sha256"],
                 "Activation model differs from B1 seed 17")
    names = REPRESENTATIONS.values()
    profiles = index([r for r in rows(a1/"novelty_profiles.csv")
                      if r["stage"] == "init" and r["representation"] in names],
                     ("edge_id", "image", "representation"))
    alternative = index([r for r in rows(a2/"reference_sensitivity.csv") if r["stage"] == "init"
                         and r["representation"] in names and r["variant"] == "baseline"
                         and r["reference"] in REFERENCES], ("edge_id", "image", "representation", "reference"))
    with np.load(a1/"path_features.npz", allow_pickle=False) as bank:
        for rep in names:
            selected = bank[rep][[int(row["feature_row_0based"]) for row in geometry]]
            strict_check(selected.shape[0] == 17 and np.isfinite(selected).all(), f"Invalid cached {rep}")
    result = {}
    for g in geometry:
        key = (g["edge_id"], g["image"])
        val = {}
        for label, rep in REPRESENTATIONS.items():
            raw = profiles[key + (rep,)]
            strict_check(all(raw[k] == g[k] for k in ("Li_1based", "feature_row_0based")),
                         f"A1 mapping mismatch: {key}")
            val[label] = float(raw["dedup_nearest"])
            for ref in REFERENCES:
                a2row = alternative[key + (rep, ref)]
                strict_check(a2row["Li_1based"] == g["Li_1based"], f"A2 mapping mismatch: {key}")
                val[f"{label}_{ref}"] = float(a2row["novelty"])
            for ref, field in (("deduplicated", "dedup_nearest"), ("raw", "raw_nearest")):
                np.testing.assert_allclose(val[f"{label}_{ref}"], float(raw[field]), rtol=0, atol=1e-8)
        result[key] = val
    return result


def force_components(raw, t):
    tangent = raw @ t
    normal = raw - tangent[:, None] * t[None, :]
    vector_sd = np.linalg.norm(np.std(raw, axis=0, ddof=1))
    return dict(Li_force_sd=float(vector_sd), Li_tangent_sd=float(np.std(tangent, ddof=1)),
                Li_normal_sd=float(np.linalg.norm(np.std(normal, axis=0, ddof=1))),
                Li_force_norm_sd=float(np.std(np.linalg.norm(raw, axis=1), ddof=1)))


def read_predictions(out, manifest):
    im = index(rows(out/"raw_image_predictions.csv"), ("edge_id", "image", "seed"))
    groups = defaultdict(list)
    for row in rows(out/"raw_atomic_predictions.csv"):
        groups[row["edge_id"], row["image"], row["seed"]].append(row)
    strict_check(len(im) == 51 and set(groups) == set(im) and all(r["stage"] == "init" for r in im.values())
                 and sum(map(len, groups.values())) == manifest["atomic_rows"], "Incomplete prediction records")
    weights = {r["checkpoint_id"]: r["sha256"] for r in manifest["checkpoints"]}
    all_forces, energies = {}, {}
    for key, r in im.items():
        group = groups[key]
        n, mover = int(r["n_atoms"]), int(r["Li_1based"])
        expected_id = f"e0-A-{key[2]}"
        strict_check(r["checkpoint_id"] == expected_id and r["checkpoint_sha256"] == weights[expected_id]
                     and list(range(1, n + 1)) == [int(x["atom_1based"]) for x in group]
                     and all(x["stage"] == "init" for x in group)
                     and [(int(x["atom_1based"]), x["element"]) for x in group if x["is_moving_Li"] == "1"]
                     == [(mover, "Li")], f"Invalid atom or model mapping: {key}")
        f = np.array([[float(x[f"F{v}_eV_per_A"]) for v in "xyz"] for x in group])
        e = np.array([float(x["atomic_model_energy_eV"]) for x in group])
        total = float(r["total_energy_eV"])
        strict_check(np.isfinite(f).all() and np.isfinite(e).all() and np.isfinite(total)
                     and float(r["inference_seconds"]) > 0, f"Invalid values: {key}")
        np.testing.assert_allclose([e.sum(), float(r["mean_energy_per_atom_eV"]) * n], total, rtol=0, atol=1e-9)
        np.testing.assert_array_equal(f[mover - 1], [float(r[f"Li_F{v}_eV_per_A"]) for v in "xyz"])
        energies[key], all_forces[key] = total, f
    return energies, all_forces


def summarize(geometry, features, energies, all_forces):
    images, paths, barrier_rows = [], [], []
    for edge in PATHS:
        path = [g for g in geometry if g["edge_id"] == edge]
        e = np.array([[energies[(edge, g["image"], str(s))] for s in SEEDS] for g in path])
        raw = e.std(axis=1, ddof=1)
        centered = (e - e.mean(axis=0)).std(axis=1, ddof=1)
        from_start = (e - e[0]).std(axis=1, ddof=1)
        barrier = e.max(axis=0) - e[0]
        peaks = e.argmax(axis=0) + 1
        for s, b, peak in zip(SEEDS, barrier, peaks):
            barrier_rows.append(dict(edge_id=edge, seed=s, forward_barrier_meV=float(b * 1000),
                                     peak_image=int(peak)))
        for g, v0, v1, v2 in zip(path, raw, centered, from_start):
            key = edge, g["image"]
            forces = np.stack([all_forces[key + (str(s),)] for s in SEEDS])
            mover = int(g["Li_1based"]) - 1
            li = forces[:, mover, :]
            t = np.array([float(g[f"tangent_{v}"]) for v in "xyz"])
            np.testing.assert_allclose(np.linalg.norm(t), 1, rtol=0, atol=1e-12)
            comp = force_components(li, t)
            np.testing.assert_allclose(comp["Li_force_sd"]**2, comp["Li_tangent_sd"]**2 + comp["Li_normal_sd"]**2,
                                       rtol=0, atol=1e-12)
            atom_sd = np.std(forces, axis=0, ddof=1)
            images.append(dict(g, **features[key], energy_raw_sd=float(v0 * 1000),
                               energy_centered_sd=float(v1 * 1000), energy_from_start_sd=float(v2 * 1000),
                               energy_per_atom_sd_meV=float(v0 * 1000 / int(g["n_atoms"])), **comp,
                               all_atom_vector_rms_sd_eV_per_A=float(np.sqrt(np.mean((atom_sd**2).sum(axis=1))))))
        block = images[-len(path):]
        summary = dict(edge_id=edge, n_images=len(path), Li_1based=path[0]["Li_1based"],
                       forward_barrier_mean_meV=float(barrier.mean() * 1000),
                       forward_barrier_sd_meV=float(barrier.std(ddof=1) * 1000),
                       peak_on_same_image=len(set(peaks)) == 1)
        for field in (*REPRESENTATIONS, *ENERGIES, *FORCES):
            summary[f"{field}_mean"] = float(np.mean([r[field] for r in block]))
            greatest = max(r[field] for r in block)
            summary[f"{field}_max"] = greatest
            summary[f"{field}_max_images"] = ";".join(r["image"] for r in block
                   if np.isclose(r[field], greatest, rtol=1e-10, atol=1e-12))
        paths.append(summary)
    return images, paths, barrier_rows


def spearman(x, y):
    if len(x) < 3 or np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        return None
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def relationship_table(images):
    tables, omit = [], []
    pairs = [(x, y, "primary") for x in DESCRIPTORS for y in (ENERGIES[0], *FORCES)]
    pairs += [(ENERGIES[0], y, "primary") for y in FORCES]
    pairs += [("coordination_hard", y, "coordination_form") for y in (ENERGIES[0], *FORCES)]
    pairs += [(x, y, "energy_definition") for x in (*DESCRIPTORS, *FORCES) for y in ENERGIES[1:]]
    pairs += [(f"{x}_{bank}", y, "reference_bank") for x in DESCRIPTORS for bank in REFERENCES[1:]
              for y in (ENERGIES[0], *FORCES)]
    pairs += [(x, y, "between_signals") for x, y in combinations(DESCRIPTORS, 2)]
    for x, y, analysis in pairs:
        include = [r for r in images if "energy_from_start_sd" not in (x, y) or int(r["image"]) > 1]
        coefficients = {}
        for edge in PATHS:
            block = [r for r in include if r["edge_id"] == edge]
            coefficients[edge] = spearman([r[x] for r in block], [r[y] for r in block])
        valid = [v for v in coefficients.values() if v is not None]
        full = float(np.mean(valid)) if valid else None
        key = dict(x=x, y=y, analysis=analysis)
        tables.append(dict(key, within_path_mean_rho=full,
                           **{f"rho_{SHORT[edge]}": coefficients[edge] for edge in PATHS},
                           n_paths_with_rho=len(valid), n_images=len(include),
                           pooled_rho_descriptive_only=spearman([r[x] for r in include], [r[y] for r in include])))
        for edge in PATHS:
            keep = [v for k, v in coefficients.items() if k != edge and v is not None]
            omit.append(dict(key, omitted_path=edge, n_paths_with_rho=len(keep),
                             within_path_mean_rho=float(np.mean(keep)) if keep else None,
                             full_mean_rho=full))
    return tables, omit


def high_images(block, field):
    ordered = sorted([r[field] for r in block], reverse=True)
    if np.ptp(ordered) <= 1e-12:
        return None
    cutoff = ordered[1]
    return {r["image"] for r in block if r[field] > cutoff or np.isclose(r[field], cutoff, rtol=1e-10, atol=1e-12)}


def overlap_table(images):
    signals = ("soap", "activation", ENERGIES[0], "Li_tangent_sd")
    output = []
    for edge in PATHS:
        block = [r for r in images if r["edge_id"] == edge]
        top = {field: high_images(block, field) for field in signals}
        for a, b in combinations(signals, 2):
            one, two = top[a], top[b]
            overlap = None if one is None or two is None else len(one & two) / len(one | two)
            output.append(dict(edge_id=edge, x=a, y=b, top_x=";".join(sorted(one or [])),
                               top_y=";".join(sorted(two or [])), jaccard=overlap))
    return output


def figures(images, summary, out):
    pairs = [(x, y) for x in ("soap", "activation") for y in (ENERGIES[0], "Li_force_sd", "Li_tangent_sd")]
    lookup = {(r["x"], r["y"]): r for r in summary}
    for filename, comparisons, width in (("descriptor_disagreement.png", pairs, 3),
        ("energy_force_disagreement.png", [(ENERGIES[0], f) for f in FORCES], 3)):
        fig, axes = plt.subplots(len(comparisons) // width, width, figsize=(11, 3.5 * len(comparisons) / width),
                                 squeeze=False, constrained_layout=True)
        for ax, (x, y) in zip(axes.flat, comparisons):
            for edge, c in zip(PATHS, COLORS):
                group = [r for r in images if r["edge_id"] == edge]
                ax.scatter([r[x] for r in group], [r[y] for r in group], color=c, s=27, label=SHORT[edge])
            rho = lookup[x, y]["within_path_mean_rho"]
            ax.set(title=f"Mean path Spearman ρ = {rho:.2f}" if rho is not None else "ρ undefined",
                   xlabel=LABELS[x], ylabel=LABELS[y])
            ax.grid(alpha=.22)
        axes.flat[0].legend(fontsize=8)
        fig.savefig(out/filename, dpi=170)
        plt.close(fig)
    metrics = (*DESCRIPTORS, ENERGIES[0], *FORCES)
    fig, axes = plt.subplots(len(metrics), len(PATHS), figsize=(12, 11), squeeze=False, sharex=True,
                             constrained_layout=True)
    for col, (edge, c) in enumerate(zip(PATHS, COLORS)):
        batch = [r for r in images if r["edge_id"] == edge]
        s = [float(r["reaction_coordinate"]) for r in batch]
        for i, field in enumerate(metrics):
            ax = axes[i, col]
            ax.plot(s, [r[field] for r in batch], marker="o", color=c, markersize=4)
            ax.grid(alpha=.25)
            if col == 0:
                ax.set_ylabel(PROFILE_LABELS[i], fontsize=8)
        axes[0, col].set_title(SHORT[edge])
        axes[-1, col].set_xlabel("Li reaction coordinate")
    fig.savefig(out/"path_profiles.png", dpi=170)
    plt.close(fig)


def analyze(repo, out):
    manifest = read_json(out/"run_manifest.json")
    geometry = rows(out/"image_geometry.csv")
    strict_check(len(geometry) == len(index(geometry, ("edge_id", "image"))) == 17
                 and all(r["stage"] == "init" for r in geometry), "Wrong image geometry")
    features = cache_columns(repo, out, manifest, geometry)
    energies, all_forces = read_predictions(out, manifest)
    image_data, path_data, barriers = summarize(geometry, features, energies, all_forces)
    rel, omit = relationship_table(image_data)
    names = ("image_disagreement.csv", "path_summary.csv", "per_seed_barriers.csv", "correlations.csv",
             "leave_one_path_out.csv", "top_image_overlap.csv")
    for name, values in zip(names, (image_data, path_data, barriers, rel, omit, overlap_table(image_data))):
        export_csv(out/name, values)
    figures(image_data, rel, out)
    write_json(out/"analysis_manifest.json", dict(input_manifest_sha256=file_sha(out/"run_manifest.json"),
               script_sha256=file_sha(repo/"src/phase_b1_analysis.py"),
               packages={n: version(n) for n in ("numpy", "scipy", "matplotlib")},
               feature_source="A1 nearest deduplicated; A2 raw, material-balanced, and target-excluded sensitivity",
               correlation="Spearman per path, equal-weight mean, no iid image p-values",
               outputs={n: file_sha(out/n) for n in (*names, "descriptor_disagreement.png",
                        "energy_force_disagreement.png", "path_profiles.png")}))
    print(f"Analyzed {len(image_data)} images, {len(path_data)} paths, {len(rel)} comparisons")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze(args.repo, args.output)
