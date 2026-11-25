"""Audit Li-centered geometry and reference coverage without modifying input files."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
from itertools import combinations
import json
from pathlib import Path
from zipfile import ZipFile

import ase
from ase.geometry import find_mic
from ase.io import read
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy import __version__ as scipy_version

PATHS = ("mp-753143_5_13_0_0_0", "mp-1177833_2_3_0_0_1", "mp-768690_0_0_1_0_0")
STAGES = ("init", "relaxed")
RADII = (2.8, 3.0, 3.2)
TOLERANCES = (.03, .10, .20)
SHELL_GAPS = (.25, .35, .45)
SHELL_LIMIT = 5.
D0_FRAMES = 278
EXPECTED = {
    "neb": "b7a99d89337902e9e1da319f57547170fdb132bf15bf6ffef03a0140e2207d7f",
    "index": "28342dda89b697f6b11ecc6c4d5b0233b77be49b68658ca7e9ea79858384e1c7",
    "bundle": "6bfd1a816609771b8a5124b6e3000262eb5258969b6756243997b8d632126862",
}


def minimum_vectors(vectors, cell, pbc):
    array = np.asarray(vectors, dtype=float)
    if not array.size:
        return array.copy()
    return find_mic(array.reshape(-1, 3), cell, pbc)[0].reshape(array.shape)


def reaction_coordinate(positions, cell, pbc):
    positions = np.asarray(positions, dtype=float)
    increments = minimum_vectors(np.diff(positions, axis=0), cell, pbc)
    arc = np.r_[0., np.cumsum(np.linalg.norm(increments, axis=1))]
    if arc[-1] <= 1e-10:
        raise ValueError("Cannot normalize a stationary migration trajectory")
    unwrapped = positions[0] + np.vstack([np.zeros(3), np.cumsum(increments, axis=0)])
    return unwrapped, arc, arc / arc[-1]


def assign_mapping(previous, current, fixed=None):
    """Assign by element and squared MIC displacement. Return the mapping and per-row residual."""
    if (len(previous) != len(current) or not np.array_equal(previous.pbc, current.pbc)
            or not np.allclose(previous.cell, current.cell, atol=1e-7, rtol=0)
            or Counter(previous.numbers) != Counter(current.numbers)):
        raise ValueError("Incompatible cell, periodicity or composition")
    mapping = np.empty(len(previous), dtype=int)
    residuals = np.empty(len(previous))
    if fixed is not None:
        if previous.numbers[fixed] != current.numbers[fixed]:
            raise ValueError("Fixed site changed element")
        mapping[fixed] = fixed
        residuals[fixed] = np.linalg.norm(minimum_vectors(
            current.positions[fixed] - previous.positions[fixed], previous.cell, previous.pbc))
    for element in sorted(set(previous.numbers)):
        a = np.flatnonzero(previous.numbers == element)
        b = np.flatnonzero(current.numbers == element)
        if fixed is not None:
            a, b = a[a != fixed], b[b != fixed]
        vectors = current.positions[b][None] - previous.positions[a][:, None]
        square = np.sum(minimum_vectors(vectors, previous.cell, previous.pbc)**2, axis=-1)
        i, j = linear_sum_assignment(square)
        mapping[a[i]] = b[j]
        residuals[a[i]] = np.sqrt(square[i, j])
    return mapping, residuals


def split_shells(distances, gap):
    distances = np.asarray(distances)
    order = np.argsort(distances, kind="stable")
    borders = np.flatnonzero(np.diff(distances[order]) > gap) + 1
    first, second, *_ = np.split(order, np.r_[borders, len(order)])
    return first, second


def exchange(previous, current):
    return sorted(current - previous), sorted(previous - current)


def signature_from_vectors(vectors, symbols, radius):
    all_radii = np.linalg.norm(vectors, axis=1)
    mask = (all_radii <= radius) & (all_radii > 1e-10)
    v, e, r = vectors[mask], symbols[mask], all_radii[mask]
    counts = tuple(sorted(Counter(e).items()))
    radial = np.concatenate([np.sort(r[e == symbol]) for symbol, _ in counts]) if counts else np.empty(0)
    i, j = np.triu_indices(len(v), 1)
    distances = np.linalg.norm(v[i] - v[j], axis=1)
    pair_keys = np.asarray(["-".join(sorted((a, b))) for a, b in zip(e[i], e[j])], dtype=str)
    pairs = np.concatenate([np.sort(distances[pair_keys == key]) for key in sorted(set(pair_keys))]) if distances.size else distances
    return counts, radial, pairs


def local_signature(positions, symbols, center, cell, pbc, radius):
    vectors = minimum_vectors(positions - positions[center], cell, pbc)
    return signature_from_vectors(vectors, np.asarray(symbols), radius)


def compare_signatures(query, reference):
    if not query[0] or query[0] != reference[0]:
        return None
    radial = np.sqrt(np.mean((query[1] - reference[1])**2))
    pair = np.sqrt(np.mean((query[2] - reference[2])**2)) if len(query[2]) else 0.
    return float(max(radial, pair))


def exact_sha(blob):
    return hashlib.sha256(blob).hexdigest()


def atoms_from_bytes(blob):
    return read(io.StringIO(blob.decode()), index=":", format="extxyz")


def csv_rows(blob):
    return list(csv.DictReader(io.StringIO(blob.decode("utf-8-sig"))))


def export_csv(path, records):
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def as_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def thresholds(distance):
    return {f"within_{int(round(t*100)):02d}cA": int(distance is not None and distance <= t)
            for t in TOLERANCES}


def load_reference(bundle):
    with ZipFile(bundle) as archive:
        lock = json.loads(archive.read("plan.json"))
        assets = {k: archive.read(k) for k in ("training.xyz", "training_frames.csv", "jobs.json")}
    if any(exact_sha(v) != lock["sha256"][k] for k, v in assets.items()):
        raise ValueError("D0 assets differ from the frozen checksums")
    schedule = json.loads(assets["jobs.json"])
    if any({i for b in job["jobs"]["pure"] for i in b} != set(range(D0_FRAMES)) for job in schedule):
        raise ValueError("Pure D0 schedules do not select the canonical frame range")
    frames = atoms_from_bytes(assets["training.xyz"])[:D0_FRAMES]
    rows = csv_rows(assets["training_frames.csv"])[:D0_FRAMES]
    if len(frames) != D0_FRAMES or any(len(a) != int(r["n_atoms"]) for a, r in zip(frames, rows)):
        raise ValueError("Incomplete D0 structures")
    references = {radius: defaultdict(list) for radius in RADII}
    total = 0
    for frame, (atoms, row) in enumerate(zip(frames, rows), 1):
        lithium = np.flatnonzero(atoms.numbers == 3)
        symbols = np.asarray(atoms.get_chemical_symbols())
        vectors = minimum_vectors(atoms.positions[None] - atoms.positions[lithium][:, None],
                                  atoms.cell, atoms.pbc)
        for center, v in zip(lithium, vectors):
            total += 1
            for radius in RADII:
                signature = signature_from_vectors(v, symbols, radius)
                references[radius][signature[0]].append((frame, int(center + 1), row, signature))
    return references, total, exact_sha(assets["training.xyz"]), exact_sha(assets["training_frames.csv"])


def validate_sequence(images, mover, stage):
    """Verify source IDs in Init and archived rows against possible MIC assignments."""
    if len(images) < 3 or images[0].numbers[mover] != 3:
        raise ValueError("Incomplete path or locked moving Li is not lithium")
    first = images[0]
    if stage == "init":
        ids = np.column_stack((first.arrays["index"], first.arrays["offset"]))
        if len(np.unique(ids, axis=0)) != len(first):
            raise ValueError("Nonunique Init site identities")
    alternative_counts, max_motion = [0], [0.]
    for previous, current in zip(images, images[1:]):
        if not np.array_equal(first.numbers, current.numbers):
            raise ValueError("The archived species row order changed")
        if stage == "init" and not np.array_equal(ids, np.column_stack(
                (current.arrays["index"], current.arrays["offset"]))):
            raise ValueError("Init site identity changed across images")
        mapping, residuals = assign_mapping(previous, current)
        alternative_counts.append(int(np.count_nonzero(mapping != np.arange(len(mapping)))))
        max_motion.append(float(residuals.max()))
    li = np.flatnonzero(first.numbers == 3)
    positions = np.stack([a.positions[li] for a in images])
    steps = minimum_vectors(np.diff(positions, axis=0), first.cell, first.pbc)
    arcs = np.linalg.norm(steps, axis=-1).sum(axis=0)
    if arcs[li == mover][0] < arcs.max() - 1e-8:
        raise ValueError("Locked Li does not have the largest lithium arc in the archived mapping")
    return alternative_counts, max_motion


def describe_shells(vectors, symbols, previous):
    distances = np.linalg.norm(vectors, axis=1)
    shells, reported, sensitivity = {}, {}, {}
    for element in sorted(set(symbols)):
        ids = np.flatnonzero((symbols == element) & (distances > 1e-10) & (distances <= SHELL_LIMIT))
        sites = [(ids[group]+1).tolist() for group in split_shells(distances[ids], .35)]
        shells[element] = [set(s) for s in sites]
        shifts = [exchange(previous[element][j], current) if previous else ([], [])
                  for j, current in enumerate(shells[element])]
        reported[element] = {
            "radial_A": np.sort(distances[ids]).tolist(),
            "shells": [{"atoms_1based": s, "count": len(s),
                        "distances_A": [float(distances[i-1]) for i in s],
                        "entered_1based": entered, "left_1based": left}
                       for s, (entered, left) in zip(sites, shifts)],
        }
        sensitivity[element] = {str(g): list(map(len, split_shells(distances[ids], g))) for g in SHELL_GAPS}
    return shells, reported, sensitivity


def environment_coverage(signature, candidates):
    best = None
    counts = np.zeros(len(TOLERANCES), dtype=int)
    frames = set()
    levels = np.asarray(TOLERANCES)
    for frame, li, row, other in candidates:
        distance = compare_signatures(signature, other)
        if distance is None:
            continue
        counts += distance <= levels
        if distance <= .10:
            frames.add(frame)
        if best is None or distance < best[0]:
            best = (distance, frame, li, row)
    return best, counts, len(frames)


def parse_path(archive, case, stage, mover):
    name = f"nebDFT2k/{case}_{stage}.xyz"
    raw = archive.read(name)
    images = atoms_from_bytes(raw)
    for atoms in images:
        if (not atoms.pbc.all() or not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell).all()
                or abs(atoms.cell.volume) <= 1e-10):
            raise ValueError(f"Invalid geometry: {name}")
        if stage == "init":
            free = np.flatnonzero(~atoms.arrays["freezed"].astype(bool))
            if free.tolist() != [mover]:
                raise ValueError(f"Frozen index conflicts with Init source: {name}")
        if stage == "relaxed":
            force = atoms.get_forces(apply_constraint=False)
            if force.shape != (len(atoms), 3) or not np.isfinite(force).all():
                raise ValueError(f"Incomplete DFT force labels: {name}")
    return name, exact_sha(raw), images


def analyze_stage(case, stage, images, mover, references):
    counts, motion = validate_sequence(images, mover, stage)
    unwrapped, arc, coordinate = reaction_coordinate(
        [a.positions[mover] for a in images], images[0].cell, images[0].pbc)
    radius_at_frame, coverage, rows = {}, [], []
    signatures = {}
    previous = None
    element_stats = defaultdict(list)
    shell_sensitive_images = 0
    for i, atoms in enumerate(images):
        key = dict(edge_id=case, stage=stage, image=i+1)
        symbols = np.asarray(atoms.get_chemical_symbols())
        vectors = minimum_vectors(atoms.positions - atoms.positions[mover], atoms.cell, atoms.pbc)
        previous, layers, sensitivity = describe_shells(vectors, symbols, previous)
        for element, info in layers.items():
            element_stats[element].append(info["shells"])
        shell_sensitive_images += any(len({tuple(pair) for pair in counts_at_gap.values()}) > 1
                                      for counts_at_gap in sensitivity.values())
        energy = (float(atoms.info.get("energy")) if stage == "init" and "energy" in atoms.info
                  else float(atoms.get_potential_energy()))
        if not np.isfinite(energy):
            raise ValueError(f"Invalid source energy: {key}")
        forces = np.linalg.norm(atoms.get_forces(apply_constraint=False), axis=1) if stage == "relaxed" else None
        rows.append(dict(key, reaction_coordinate=float(coordinate[i]), li_arc_A=float(arc[i]),
                         li_displacement_A=float(np.linalg.norm(unwrapped[i]-unwrapped[0])),
                         li_unwrapped_xyz_A=as_json(unwrapped[i].tolist()),
                         moving_Li_1based=mover+1, alternative_mapping_rows=counts[i],
                         mapped_max_step_A=motion[i],
                         shells_by_element=as_json(layers), shell_gap_counts=as_json(sensitivity),
                         source_energy_eV=energy, energy_kind="BVSE" if stage == "init" else "DFT",
                         dft_max_force_eV_per_A=float(forces.max()) if forces is not None else None,
                         dft_moving_Li_force_eV_per_A=float(forces[mover]) if forces is not None else None))
        for radius in RADII:
            signature = signature_from_vectors(vectors, symbols, radius)
            signatures[radius, i] = signature
            candidates = references[radius].get(signature[0], [])
            best, matches, ref_frames = environment_coverage(signature, candidates)
            distance, frame, ref_li, ref = best if best else (None, None, None, {})
            coverage.append(dict(key, radius_A=radius, n_neighbors=len(signature[1]),
                                 element_counts=as_json(dict(signature[0])), comparable_D0_Li=len(candidates),
                                 nearest_fingerprint_A=distance, nearest_D0_frame_1based=frame,
                                 nearest_D0_Li_1based=ref_li, nearest_D0_edge_id=ref.get("edge_id"),
                                 D0_Li_within_03cA=int(matches[0]), D0_Li_within_10cA=int(matches[1]),
                                 D0_Li_within_20cA=int(matches[2]), D0_frames_within_10cA=ref_frames))
            radius_at_frame[radius, i] = distance
    internal = []
    stage_stats = []
    for radius in RADII:
        comparisons = []
        for i, j in combinations(range(len(images)), 2):
            d = compare_signatures(signatures[radius, i], signatures[radius, j])
            pair = dict(edge_id=case, stage=stage, image_i=i+1, image_j=j+1, radius_A=radius,
                        comparable=int(d is not None), fingerprint_A=d, **thresholds(d))
            internal.append(pair)
            comparisons.append(pair)
        nearest = [radius_at_frame[radius, i] for i in range(len(images))]
        stage_stats.append(dict(radius_A=radius, internal_pairs=len(comparisons),
                                internal_comparable=sum(p["comparable"] for p in comparisons),
                                internal_matches=[sum(p[k] for p in comparisons) for k in thresholds(None)],
                                D0_comparable_images=sum(d is not None for d in nearest),
                                D0_covered_images=[sum(d is not None and d <= t for d in nearest)
                                                   for t in TOLERANCES]))
    key = dict(edge_id=case, stage=stage)
    summary = dict(key, images=len(images), li_path_length_A=float(arc[-1]),
                   li_end_displacement_A=rows[-1]["li_displacement_A"],
                   alternative_mapping_steps=sum(c > 0 for c in counts),
                   shell_gap_sensitive_images=shell_sensitive_images,
                   coordination_by_element=as_json(coordination_summary(element_stats)),
                   # Site-count caveat when the 5 A shell radius crosses half a lattice spacing.
                   shell_limit_A=SHELL_LIMIT, half_min_cell_height_A=float(.5 / np.linalg.norm(
                       np.linalg.inv(images[0].cell.array), axis=0).max()),
                   coverage_by_radius=as_json(stage_stats))
    return rows, internal, coverage, summary


def coordination_summary(element_stats):
    result = {}
    for element, frames in element_stats.items():
        result[element] = []
        for layer in zip(*frames):
            counts = [frame["count"] for frame in layer]
            result[element].append(dict(
                count_start=counts[0], count_end=counts[-1], count_range=[min(counts), max(counts)],
                changed_steps=sum(bool(frame["entered_1based"] or frame["left_1based"]) for frame in layer),
            ))
    return result


def cross_stage(case, init, relaxed, mover, init_rows, relaxed_rows):
    if len(init) != len(relaxed):
        raise ValueError(f"Archived image pairing differs: {case}")
    comparisons = []
    for i, (a, b) in enumerate(zip(init, relaxed)):
        mapping, residual = assign_mapping(a, b, fixed=mover)
        framework = np.delete(residual, mover)
        comparisons.append(dict(
            edge_id=case, image=i+1, init_s=init_rows[i]["reaction_coordinate"],
            relaxed_s=relaxed_rows[i]["reaction_coordinate"], Li_shift_A=float(residual[mover]),
            framework_RMS_A=float(np.sqrt(np.mean(framework**2))),
            framework_max_A=float(framework.max()),
            alternative_mapping_rows=int(np.count_nonzero(mapping != np.arange(len(mapping)))),
        ))
    return comparisons

def plot_case(output, case, stages, cross, anions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    arc_ax, shells_ax, radial_ax, cross_ax, ref_ax, pair_ax = axs.flat
    colors = dict(zip(anions, ("#0072B2", "#D55E00", "#009E73", "#CC79A7")))
    any_reference, any_pairs = False, False
    for stage, style, color_stage in zip(STAGES, ("-", "--"), ("#0072B2", "#D55E00")):
        images, internal, coverage, _ = stages[stage]
        s = [row["reaction_coordinate"] for row in images]
        arc_ax.plot(s, [row["li_arc_A"] for row in images], style, marker="o", label=f"{stage} arc")
        arc_ax.plot(s, [row["li_displacement_A"] for row in images], style, marker="x",
                    alpha=.55, label=f"{stage} displacement")
        layers = [json.loads(row["shells_by_element"]) for row in images]
        for element in anions:
            label = f"{element} {stage}"
            color = colors[element]
            for i, marker in enumerate(("o", "s")):
                shells_ax.plot(s, [x[element]["shells"][i]["count"] for x in layers], style,
                               marker=marker, color=color, alpha=(1., .5)[i],
                               label=f"{label} {i+1}")
            for rank in range(3):
                distances = [x[element]["radial_A"] for x in layers]
                radial_ax.plot(s, [d[rank] if len(d) > rank else np.nan for d in distances],
                               style, color=color, alpha=(1., .65, .4)[rank],
                               label=label if rank == 0 else None)
        distances = [row["nearest_fingerprint_A"] for row in coverage if row["radius_A"] == 3.]
        if any(d is not None for d in distances):
            any_reference = True
            ref_ax.plot(s, [d if d is not None else np.nan for d in distances], style, marker="o",
                        color=color_stage, label=stage)
        absent = [u for u,d in zip(s, distances) if d is None]
        if absent:
            ref_ax.plot(absent, [.965 if stage == "init" else .90] * len(absent), "x",
                        transform=ref_ax.get_xaxis_transform(), color=color_stage, label=f"{stage} no match")
        pairs = [r["fingerprint_A"] for r in internal if r["radius_A"] == 3. and r["comparable"]]
        if pairs:
            any_pairs = True
            pair_ax.scatter(pairs, [STAGES.index(stage)] * len(pairs), label=f"{stage} ({len(pairs)} comparable)",
                            marker=("o", "x")[STAGES.index(stage)], color=color_stage, alpha=.65)
    cross_ax.plot([r["image"] for r in cross], [r["Li_shift_A"] for r in cross], "-o", label="Migrating Li")
    cross_ax.plot([r["image"] for r in cross], [r["framework_RMS_A"] for r in cross], "-s", label="Framework RMS")
    if any_reference:
        ref_ax.axhline(.10, color="gray", lw=.8, ls=":")
    else:
        ref_ax.set_yticks([])
        ref_ax.text(.5, .55, "No comparable D₀ environments", ha="center", transform=ref_ax.transAxes)
    if not any_pairs:
        pair_ax.text(.5, .5, "No comparable image pairs", ha="center", transform=pair_ax.transAxes)
    for t in TOLERANCES:
        pair_ax.axvline(t, color="gray", lw=.8, ls=":")
    pair_ax.set_yticks([0, 1], STAGES)
    pair_ax.set_ylim(-.5, 1.5)
    axes = (
        (arc_ax, "Li trajectory", "Distance (Å)", "Reaction coordinate"),
        (shells_ax, "Anion shell counts", "Neighbors", "Reaction coordinate"),
        (radial_ax, "Nearest anions by element", "Li–anion distance (Å)", "Reaction coordinate"),
        (cross_ax, "Init vs Relaxed by image", "MIC shift (Å)", "Archived image"),
        (ref_ax, "Closest D₀ environment", "Fingerprint distance (Å)", "Reaction coordinate"),
        (pair_ax, "Comparable in-path pairs", "", "Fingerprint distance (Å)"),
    )
    for ax, title, ylabel, xlabel in axes:
        ax.set(title=title, ylabel=ylabel, xlabel=xlabel)
        ax.grid(alpha=.18)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=7, loc="best")
    fig.suptitle(case)
    fig.savefig(output / f"{case}.png", dpi=145)
    plt.close(fig)


def write_report(path, manifest, summaries, cross, n_reference):
    lines = [
        "# Phase A0 periodic geometry and reference coverage", "",
        f"The reference bank contains {n_reference:,} Li environments from {D0_FRAMES} canonical D₀ frames.",
        "Each path has separate Init and Relaxed results. Init carries BVSE energies. Relaxed carries DFT energies and force labels.",
        "All counts summarize cases. Images and Li sites within a frame are correlated observations.", "",
        "## Integrity", "",
        "| Path | Stage | Images | Li row | Arc (Å) | Steps with a different geometry assignment |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for m, s in zip(manifest, summaries):
        lines.append(f"| {m['edge_id']} | {m['stage']} | {m['images']} | {m['moving_Li_1based']} "
                     f"| {s['li_path_length_A']:.3f} | {s['alternative_mapping_steps']} |")
    lines += [
        "", "Init archives supply the atom index, offset and unique unfrozen site. The relaxed archive lacks",
        "these IDs. Relaxed rows are checked by species and adjacent minimum-image assignment; this is",
        "geometric evidence for the archived ordering. The indexed site has the longest Li arc in every stage.",
        "Init–Relaxed framework matching holds the migrating Li fixed.",
        "Cross-stage shifts use archived image pairing and record each side's independently normalized path coordinate.", "",
        "## Coordination and environment coverage", "",
        "Each atomic site contributes its closest periodic image. For each species, shell 1 starts at the nearest site.",
        "A gap above 0.35 Å starts a new radial group. Shell counts are also reported at 0.25 and 0.45 Å.",
        "The search extends to 5 Å. An empty second group or a shell near that boundary should be interpreted in that context.",
        "Wide groups can form when consecutive distances have small gaps. Reported group counts need not be bond counts.",
        "The image CSV records entrants and leavers by archived atom row. The path CSV records how many images",
        "change at least one element-specific count across the three shell gaps.", "",
        "Local fingerprints require matching element counts. Their distance is the larger of the radial and",
        "inter-neighbor distance RMS differences after sorting within the relevant species classes. Neighbor",
        "vectors are unwrapped about the center. This detects similarity in the represented distances;",
        "different environments can share a fingerprint. It is a screening measure, rather than an atom-wise match.",
        "A site outside the chosen radius changes the fingerprint membership. Results for three radii show this sensitivity.",
        "The 0.03, 0.10 and 0.20 Å tolerances are applied to the fingerprint distance, with 0.03 Å",
        "as a tight scale informed by prior project geometry checks. The metrics differ from the previous",
        "cell-level maximum residual. Matched environments and matched D₀ frames are counted separately.", "",
        "| Path | Stage | Radius (Å) | Comparable internal pairs | Within tolerance (0.03 / 0.10 / 0.20 Å) | Images with comparable D₀ environments | Images covered at tolerance |",
        "| --- | --- | ---: | ---: | --- | ---: | --- |",
    ]
    for s in summaries:
        for row in json.loads(s["coverage_by_radius"]):
            internal = " / ".join(map(str, row["internal_matches"]))
            covered = " / ".join(map(str, row["D0_covered_images"]))
            lines.append(f"| {s['edge_id']} | {s['stage']} | {row['radius_A']} "
                         f"| {row['internal_comparable']}/{row['internal_pairs']} | {internal} "
                         f"| {row['D0_comparable_images']}/{s['images']} | {covered} |")
    lines += ["", "## Change on relaxation", "",
              "| Path | Largest Li shift (Å) | Largest image framework RMS (Å) | Images with a different framework assignment |",
              "| --- | ---: | ---: | ---: |"]
    for case in PATHS:
        case_rows = [r for r in cross if r["edge_id"] == case]
        lines.append(f"| {case} | {max(r['Li_shift_A'] for r in case_rows):.3f} "
                     f"| {max(r['framework_RMS_A'] for r in case_rows):.3f} "
                     f"| {sum(r['alternative_mapping_rows'] > 0 for r in case_rows)} |")
    lines += ["", "Each case plot shows its Li trajectory, anion distances and shell counts,",
              "cross-stage shifts, D₀ coverage, and comparable within-path fingerprint distances."]
    with path.open("x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run(args):
    sources = {name: getattr(args, name) for name in EXPECTED}
    source_sha = {name: exact_sha(path.read_bytes()) for name, path in sources.items()}
    if source_sha != EXPECTED:
        raise ValueError(f"Input archives or index differ from the audited sources: {source_sha}")
    index = {r["edge_id"]: r for r in csv_rows(args.index.read_bytes())}
    references, n_ref, d0_xyz_sha, d0_rows_sha = load_reference(args.bundle)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest, image_rows, path_rows, reference_rows, internal_rows, cross_rows = [], [], [], [], [], []
    with ZipFile(args.neb) as archive:
        for case in PATHS:
            lock = index[case]
            mover = int(lock["moving_Li_row"])-1
            stages, structures = {}, {}
            for stage in STAGES:
                member, digest, images = parse_path(archive, case, stage, mover)
                if len(images) != int(lock["init_images"]) or len(images[0]) != int(lock["atoms_per_image"]):
                    raise ValueError(f"Unexpected stage shape: {member}")
                structures[stage] = images
                stages[stage] = analyze_stage(case, stage, images, mover, references)
                im, pairs, coverage, summary = stages[stage]
                manifest.append(dict(
                    edge_id=case, stage=stage, source_member=member, source_archive_sha256=source_sha["neb"],
                    member_sha256=digest, mobile_index_sha256=source_sha["index"], moving_Li_1based=mover+1,
                    images=len(images), atoms_per_image=len(images[0]),
                    energy_kind=im[0]["energy_kind"], force_kind="DFT" if stage == "relaxed" else "unused",
                    atom_mapping="explicit index/offset plus MIC" if stage == "init" else "archive row plus MIC assignment",
                    reference_bundle_sha256=source_sha["bundle"], reference_xyz_member_sha256=d0_xyz_sha,
                    reference_frames_member_sha256=d0_rows_sha, D0_frames=D0_FRAMES,
                    D0_Li_environments=n_ref, ase_version=ase.__version__, numpy_version=np.__version__,
                    scipy_version=scipy_version,
                ))
                image_rows.extend(im)
                internal_rows.extend(pairs)
                reference_rows.extend(coverage)
                path_rows.append(summary)
            cross = cross_stage(case, structures["init"], structures["relaxed"], mover,
                                stages["init"][0], stages["relaxed"][0])
            cross_rows.extend(cross)
            plot_case(args.output, case, stages, cross, lock["anion_family"].split("-"))
            print(f"{case}: {len(cross)} images per stage", flush=True)
    for name, rows in (
        ("geometry_manifest", manifest), ("image_geometry_features", image_rows),
        ("path_geometry_features", path_rows), ("reference_coverage", reference_rows),
        ("internal_environment_pairs", internal_rows), ("init_relaxed_geometry", cross_rows),
    ):
        export_csv(args.output / (name + ".csv"), rows)
    write_report(args.output / "near_duplicate_report.md", manifest, path_rows, cross_rows, n_ref)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("neb", "index", "bundle", "output"):
        cli.add_argument(name, type=Path)
    run(cli.parse_args())
