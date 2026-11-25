"""Audit Phase A0.1 coverage across radii, tolerances and reference material weights."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from phase_a0_geometry import (
    D0_FRAMES, EXPECTED, PATHS, STAGES, as_json, atoms_from_bytes, csv_rows, exact_sha,
    export_csv, minimum_vectors, parse_path, signature_from_vectors, validate_sequence,
)

RADII = (2.5, 2.8, 3.0, 3.2, 3.5)
TOLERANCES = (.03, .05, .10, .15, .20)
DEDUP_TOLERANCE = .03


def load_inputs(neb, index, bundle, a0):
    paths = dict(neb=neb, index=index, bundle=bundle)
    if {k: exact_sha(p.read_bytes()) for k, p in paths.items()} != EXPECTED:
        raise ValueError("The Phase A0 sources differ")
    with ZipFile(bundle) as archive:
        ref = atoms_from_bytes(archive.read("training.xyz"))[:D0_FRAMES]
        metadata = csv_rows(archive.read("training_frames.csv"))[:D0_FRAMES]
    if len(ref) != D0_FRAMES or len(metadata) != D0_FRAMES:
        raise ValueError("Incomplete D0")
    locked = {r["edge_id"]: r for r in csv_rows(index.read_bytes())}
    image_values = {(r["edge_id"], r["stage"], int(r["image"])): r
                    for r in csv_rows((a0/"image_geometry_features.csv").read_bytes())}
    manifest = {(r["edge_id"], r["stage"]): r for r in csv_rows((a0/"geometry_manifest.csv").read_bytes())}
    queries = []
    with ZipFile(neb) as archive:
        for case in PATHS:
            mover = int(locked[case]["moving_Li_row"])-1
            for stage in STAGES:
                name, digest, geometries = parse_path(archive, case, stage, mover)
                m = manifest[case, stage]
                if digest != m["member_sha256"] or name != m["source_member"] or len(geometries) != int(m["images"]):
                    raise ValueError("The source does not match the A0 geometry manifest")
                alternatives, _ = validate_sequence(geometries, mover, stage)
                if any(alternatives):
                    raise ValueError("Unexpected row-assignment change")
                for i, atoms in enumerate(geometries, 1):
                    row = image_values[case, stage, i]
                    if int(row["moving_Li_1based"]) != mover+1:
                        raise ValueError("The A0 Li index changed")
                    key = dict(edge_id=case, stage=stage, image=i,
                               reaction_coordinate=float(row["reaction_coordinate"]), Li_1based=mover+1)
                    queries.append((atoms, mover, key))
    return ref, metadata, queries


def matrix(signatures):
    return tuple(np.stack([sign[i] for sign in signatures]) for i in (1, 2))


def distances(query, radial, pairs):
    d = np.sqrt(np.mean((radial - query[1])**2, axis=1))
    if len(query[2]):
        d = np.maximum(d, np.sqrt(np.mean((pairs - query[2])**2, axis=1)))
    return d


def build_bank(frames, frame_rows, radii):
    groups = {r: defaultdict(list) for r in radii}
    for i, (atoms, row) in enumerate(zip(frames, frame_rows), 1):
        material = row["material_id"]
        centers = np.flatnonzero(atoms.numbers == 3)
        vectors = minimum_vectors(atoms.positions[None] - atoms.positions[centers][:, None],
                                  atoms.cell, atoms.pbc)
        symbols = np.asarray(atoms.get_chemical_symbols())
        for pos, li in zip(vectors, centers):
            for r in radii:
                signature = signature_from_vectors(pos, symbols, r)
                groups[r][material, signature[0]].append((i, int(li+1), signature))
    return groups


def collapse_groups(groups):
    """Assign each nonrepresentative to a retained representative within tolerance."""
    by_pattern = defaultdict(list)
    reference_rows = {}
    for (material, pattern), items in sorted(groups.items()):
        counts = reference_rows.setdefault(material, dict(raw=0, unique=0, representatives=[]))
        counts["raw"] += len(items)
        if not pattern:
            counts["unique"] += len(items)
            counts["representatives"].extend((i, li) for i, li, _ in items)
            continue
        radial, pairs = matrix([sign for _, _, sign in items])
        retain = np.zeros(len(items), dtype=bool)
        covered = np.zeros(len(items), dtype=bool)
        for j, (_, _, sign) in enumerate(items):
            if not covered[j]:
                retain[j] = True
                covered |= distances(sign, radial, pairs) <= DEDUP_TOLERANCE
        counts["unique"] += int(retain.sum())
        counts["representatives"].extend((i, li) for (i, li, _), keep in zip(items, retain) if keep)
        by_pattern[pattern].append((material, radial, pairs, retain))
    return by_pattern, reference_rows


def evaluate(signature, by_pattern, materials):
    matched = np.zeros((2, len(TOLERANCES)), dtype=int)
    balanced = np.zeros_like(matched, dtype=float)
    supporting_materials = np.zeros_like(matched)
    nearest = [None, None]
    thresholds = np.asarray(TOLERANCES)
    groups = by_pattern.get(signature[0], ())
    for material, radial, pairs, retain in groups:
        d = distances(signature, radial, pairs)
        for i, pool in enumerate((d, d[retain])):
            n = (pool[:, None] <= thresholds).sum(axis=0)
            matched[i] += n
            balanced[i] += n / materials[material][("raw","unique")[i]]
            supporting_materials[i] += n > 0
            closest = float(pool.min())
            nearest[i] = closest if nearest[i] is None else min(nearest[i], closest)
    totals = np.asarray([sum(m[k] for m in materials.values()) for k in ("raw","unique")])[:, None]
    return dict(matched=matched, pooled=matched/totals, balanced=balanced/len(materials),
                nearest=nearest, supporting=supporting_materials, comparable=len(groups))


def summarize_grid(rows):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ("edge_id","stage","radius_A","tolerance_A"))
        groups[key].append(row)
    return [dict(edge_id=e, stage=s, radius_A=r, tolerance_A=t, images=len(series),
                 images_with_comparable_material=sum(x["comparable_D0_materials"] > 0 for x in series),
                 raw_covered_images=sum(x["raw_matching_Li"] > 0 for x in series),
                 dedup_covered_images=sum(x["dedup_matching_Li"] > 0 for x in series),
                 avg_dedup_material_balanced_fraction=float(np.mean(
                     [x["dedup_material_balanced_fraction"] for x in series])))
            for (e,s,r,t), series in groups.items()]


def plot_grid(output, table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(9, 10), layout="constrained", sharex=True, sharey=True)
    for ax, (case, stage) in zip(axes.flat, ((e,s) for e in PATHS for s in STAGES)):
        lookup = {(x["radius_A"], x["tolerance_A"]):x for x in table if
                  (x["edge_id"], x["stage"]) == (case, stage)}
        n = next(iter(lookup.values()))["images"]
        counts = np.array([[lookup[r,t]["dedup_covered_images"] for t in TOLERANCES] for r in RADII])
        colors = ax.imshow(counts/n, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i,j in np.ndindex(counts.shape):
            ax.text(j, i, f"{counts[i,j]}/{n}", ha="center", va="center",
                    color="white" if counts[i,j]/n > .55 else "#12223b")
        ax.set(title=f"{case.split('_')[0]}  {stage}", xticks=range(len(TOLERANCES)),
               xticklabels=[f"{t:.2f}" for t in TOLERANCES], yticks=range(len(RADII)),
               yticklabels=RADII)
    fig.supxlabel("Fingerprint tolerance (Å)")
    fig.supylabel("Environment radius (Å)")
    fig.colorbar(colors, ax=axes, shrink=.65, label="Fraction of images with a deduplicated D₀ match")
    fig.savefig(output/"coverage_sensitivity.png", dpi=145)
    plt.close(fig)


def run(args):
    frames, metadata, queries = load_inputs(args.neb, args.index, args.bundle, args.a0)
    bank = build_bank(frames, metadata, RADII)
    rows, ref_rows = [], []
    for radius in RADII:
        by_pattern, materials = collapse_groups(bank[radius])
        for material, values in materials.items():
            ref_rows.append(dict(radius_A=radius, material_id=material, raw_Li=values["raw"],
                                 dedup_Li=values["unique"], frames=sum(
                                     r["material_id"]==material for r in metadata),
                                 dedup_representatives=as_json(values["representatives"])))
        for atoms, mover, key in queries:
            v = minimum_vectors(atoms.positions-atoms.positions[mover], atoms.cell, atoms.pbc)
            sig = signature_from_vectors(v, np.asarray(atoms.get_chemical_symbols()), radius)
            result = evaluate(sig, by_pattern, materials)
            for i,tol in enumerate(TOLERANCES):
                rows.append(dict(key, radius_A=radius, tolerance_A=tol,
                                 comparable_D0_materials=result["comparable"],
                                 raw_nearest_A=result["nearest"][0], dedup_nearest_A=result["nearest"][1],
                                 raw_matching_Li=int(result["matched"][0,i]), dedup_matching_Li=int(result["matched"][1,i]),
                                 raw_pooled_fraction=result["pooled"][0,i], dedup_pooled_fraction=result["pooled"][1,i],
                                 raw_material_balanced_fraction=result["balanced"][0,i],
                                 dedup_material_balanced_fraction=result["balanced"][1,i],
                                 raw_materials_with_match=int(result["supporting"][0,i]),
                                 dedup_materials_with_match=int(result["supporting"][1,i])))
    grid = summarize_grid(rows)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    for name, values in (("image_sensitivity", rows), ("path_sensitivity", grid),
                         ("reference_material_distribution", ref_rows)):
        export_csv(output/(name+".csv"),values)
    plot_grid(output, grid)
    report = [
        "# Phase A0.1 coverage sensitivity and reference composition", "",
        "The reference bank is the unchanged 278-frame D₀. Greedy deduplication operates within each",
        "material and element-count pattern, in original frame and Li order. Every removed site is",
        "within 0.03 Å fingerprint distance of a retained site. Cross-material memberships are preserved.",
        "The exact same Li-centered distance fingerprint from A0 is used throughout.", "",
        "Per-image nearest distance and any-match coverage describe the existence of a close reference.",
        "Material-balanced support averages the per-material fraction of matching Li environments,",
        "giving each material equal weight and assigning zero to materials with no matching environment.",
        "Pooled support weights every reference environment equally. Images do not constitute independent paths.", "",
        "| Radius (Å) | Raw Li | Deduplicated Li | Materials |",
        "| --- | ---: | ---: | ---: |",
    ]
    for radius in RADII:
        subset = [r for r in ref_rows if r["radius_A"] == radius]
        report.append(f"| {radius} | {sum(r['raw_Li'] for r in subset):,} "
                      f"| {sum(r['dedup_Li'] for r in subset):,} | {len(subset)} |")
    previous_shells = csv_rows((args.a0/"path_geometry_features.csv").read_bytes())
    report += ["", "The A0 first and second radial groups change under the 0.25–0.45 Å gap scan in "
               f"{sum(int(r['shell_gap_sensitive_images']) for r in previous_shells)}/34 images. "
               "A1 retains the continuous radial distributions alongside coordination counts.", "",
               "See the sensitivity figure and CSVs for each path and stage."]
    (output/"README.md").write_text("\n".join(report)+"\n",encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("neb","index","bundle","a0","output"):
        cli.add_argument(name,type=Path)
    run(cli.parse_args())
