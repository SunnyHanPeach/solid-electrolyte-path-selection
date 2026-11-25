"""Freeze the six B2 path/Replay schedules from Phase 1 and the archived NEB pool."""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from ase.geometry import get_distances

from audit_phase_b0 import FOUNDATION, SEEDS, frame_prefix
from audit_signed_barriers import REFERENCE_SHA256, file_sha, load_frozen_data, rows, sha
from phase_a0_geometry import PATHS as REFERENCES, atoms_from_bytes, parse_path, validate_sequence
from run_force_weighting import load_bundle
from freeze_net_path_panel import csv_bytes

D0_SHA256 = "dda468e74c0dbd46c3d5b07121b900184a917dd319415e826af6e25b56ee7081"
BASE_SHA256 = "6bfd1a816609771b8a5124b6e3000262eb5258969b6756243997b8d632126862"
QUALITY_POLICY = {"init_min_Li_contact_A": 1.2, "relaxed_min_any_contact_A": 1.0,
                  "policy": "Gross geometry and label integrity gates for new paths. Historical references are retained if relaxed geometry/labels pass."}
RULE = ("Three sequential maximin rounds. Prefer anion families not yet represented, require a distinct "
        "material and exact chemistry, then maximize the nearest IQR-scaled RMS distance across equally weighted "
        "frozen BVSE Init feature groups. Break ties by edge ID. Historical experiment inclusion uses path IDs only.")


def same_atoms_distance(atoms):
    _, distance = get_distances(atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
    np.fill_diagonal(distance, np.inf)
    return float(distance.min())


def check_path(archive, row, labeled_frame, metadata, raw_frames):
    edge, mover = row["edge_id"], int(row["moving_Li_row"]) - 1
    member, initial_sha, initial = parse_path(archive, edge, "init", mover)
    if any(validate_sequence(initial, mover, "init")[0]):
        raise ValueError("Init atom mapping changed")
    member, relaxed_sha, relaxed = parse_path(archive, edge, "relaxed", mover)
    if any(validate_sequence(relaxed, mover, "relaxed")[0]):
        raise ValueError("Relaxed atom mapping changed")
    expected_count, n = int(row["init_images"]), int(row["atoms_per_image"])
    if len(initial) != expected_count or len(relaxed) != expected_count or not all(
            len(a) == n and np.array_equal(a.numbers, initial[0].numbers) for a in initial + relaxed):
        raise ValueError("Frame count, species order or atom count mismatch")
    contact = min(float(np.delete(a.get_distances(mover, range(n), mic=True), mover).min()) for a in initial)
    relaxed_min = min(same_atoms_distance(a) for a in relaxed)
    if not np.isfinite(contact) or not np.isfinite(relaxed_min) or relaxed_min < QUALITY_POLICY["relaxed_min_any_contact_A"]:
        raise ValueError("Gross close contact in relaxed geometry")
    records, force_components = [], []
    for i, _, _, header, body in raw_frames(archive, member):
        energy, peak_component = labeled_frame(metadata(header), body)
        records.append(dict(edge_id=edge, material_id=row["material_id"], frame=i,
                            role="candidate", n_atoms=n, energy_eV=energy))
        force_components.append(peak_component)
    if len(records) != expected_count:
        raise ValueError("Incomplete labeled image series")
    checks = dict(init_sha256=initial_sha, relaxed_sha256=relaxed_sha,
                  init_Li_min_contact_A=contact, relaxed_min_any_contact_A=relaxed_min,
                  largest_DFT_force_component_eV_A=max(force_components))
    return (archive.read(member), records), checks


def make_orders(pure, new_ids, sizes, seed):
    n = sizes[new_ids[0]]
    controls = [i for i, count in enumerate(sizes[:278]) if count == n]
    rng = np.random.default_rng(seed)
    insert = [int(i) for _ in range(5) for i in rng.permutation(new_ids)]
    at = {((2*k + 1) * len(pure)) // (2 * len(insert)): j for k, j in enumerate(insert)}
    path, replay, used = [], [], Counter()
    for i, batch in enumerate(pure):
        if i in at:
            companion = batch[0]
            control = min((j for j in controls if j != companion), key=lambda j: (used[j], j))
            path.append([companion, at[i]])
            replay.append([companion, control])
            used[control] += 1
        path.append(batch)
        replay.append(batch)
    if (len(pure) != 710 or len(path) != len(replay) or len(path) != 710 + 5 * len(new_ids)
            or Counter(b[1] for b in path if b[1] in new_ids) != Counter(dict.fromkeys(new_ids, 5))
            or any([sizes[k] for k in p] != [sizes[k] for k in q] for p, q in zip(path, replay))):
        raise ValueError("Incomplete matched exposure")
    return path, replay


def write(path, data):
    with path.open("xb") as f:
        f.write(data)


def prepare(args):
    if file_sha(args.base) != BASE_SHA256:
        raise ValueError("Unrecognized Phase 1 base")
    plan, frozen = load_bundle(args.base)
    settings = plan["training"]
    if tuple(settings["seeds"]) != SEEDS or sha(frozen["foundation.model"]) != FOUNDATION:
        raise ValueError("Pure D0 protocol or initialization differs")
    d0 = rows(frozen["training_frames.csv"])[:278]
    if sha(frame_prefix(frozen["training.xyz"], 278)) != D0_SHA256:
        raise ValueError("Wrong canonical 278 frames")
    sizes = [int(r["n_atoms"]) for r in d0]
    population = Counter(sizes)
    d0_material = {r["material_id"] for r in d0}
    evaluation, old_orders, truth = load_frozen_data(args.base, args.project)
    held = {r["material_id"] for r in evaluation}
    source = args.project/"迁移路径数据"/"nebDFT2k.zip"
    if file_sha(source) != REFERENCE_SHA256 or file_sha(args.init) != plan["feature_file_sha256"]:
        raise ValueError("Candidate source changed")
    sys.path.insert(0, str(args.project/"experiment2"))
    from prepare_data import labeled_frame, metadata, raw_frames

    with ZipFile(args.base) as z:
        panel = rows(z.read("evaluation_panels.csv"))
    pool = {r["edge_id"] for r in rows(args.pool.read_bytes())}
    if len(pool) != 38 or not set(REFERENCES) <= pool:
        raise ValueError("Wrong historical pool")
    meta = {r["edge_id"]: r for r in rows(args.init.read_bytes()) if r["edge_id"] in pool}
    if meta.keys() != pool:
        raise ValueError("Missing initial features")
    previous = set(plan["selected"]) | set(REFERENCES)
    previous.update(r["path"] for r in json.loads(args.previous_queue.read_bytes())["paths"])
    for p in (args.previous_transfer,):
        with ZipFile(p) as z:
            previous.add(json.loads(z.read("plan.json"))["selected"]["edge_id"])
    columns = [name for group in plan["path_feature_groups"] for name in group]
    cases, quality, reason = {}, {}, {}
    with ZipFile(source) as z:
        for edge in sorted(pool):
            row = meta[edge]
            problems = []
            if population[int(row["atoms_per_image"])] < 2:
                problems.append("no exact-atom matched Replay")
            if row["material_id"] in held | d0_material:
                problems.append("material already in evaluation or D0")
            if not np.isfinite([float(row[c]) for c in columns]).all():
                problems.append("nonfinite DFT-free features")
            try:
                cases[edge], quality[edge] = check_path(z, row, labeled_frame, metadata, raw_frames)
                if quality[edge]["init_Li_min_contact_A"] < QUALITY_POLICY["init_min_Li_contact_A"]:
                    if edge not in REFERENCES:
                        problems.append("gross close contact in initial geometry")
            except (ValueError, IndexError, KeyError) as error:
                problems.append(f"invalid trajectory or labels: {error}")
            if edge in REFERENCES and problems:
                raise ValueError(f"Historical reference {edge}: {problems}")
            if edge in previous:
                problems.append("previous protected-schedule path or mandatory historical reference")
            reason[edge] = " | ".join(problems)

    eligible = [e for e in sorted(pool) if not reason[e]]
    matrix = np.array([[float(meta[e][c]) for c in columns] for e in eligible])
    q1, q3 = np.quantile(matrix, [.25, .75], axis=0)
    scale = np.where(q3 > q1, q3 - q1, 1.)
    group_scale = np.repeat(np.sqrt([len(g) for g in plan["path_feature_groups"]]),
                            [len(g) for g in plan["path_feature_groups"]])

    def vector(edge):
        return np.array([float(meta[edge][c]) for c in columns]) / (scale * group_scale)

    selected = list(REFERENCES)
    ranking = []
    for pick in range(1, 4):
        used_material = {meta[e]["material_id"] for e in selected}
        used_chem = {meta[e]["chemsys"] for e in selected}
        used_anion = {meta[e]["anion_family"] for e in selected}
        take = []
        excluded = []
        selected_vectors = np.array([vector(x) for x in selected])
        for edge in sorted(pool):
            row = meta[edge]
            excluded_reason = reason[edge]
            if not excluded_reason and (row["material_id"] in used_material or row["chemsys"] in used_chem):
                excluded_reason = "material or exact chemistry already selected"
            record = dict(round=pick, edge_id=edge, material_id=row["material_id"], chemsys=row["chemsys"],
                          anion_family=row["anion_family"], atoms=row["atoms_per_image"], images=row["init_images"],
                          **{c: row[c] for c in columns}, rank="", selected=0, excluded_reason=excluded_reason,
                          new_anion="", min_feature_distance="")
            if excluded_reason:
                excluded.append(record)
            else:
                record["new_anion"] = int(row["anion_family"] not in used_anion)
                record["min_feature_distance"] = float(np.linalg.norm(selected_vectors - vector(edge), axis=1).min())
                take.append(record)
        take.sort(key=lambda r: (-r["new_anion"], -r["min_feature_distance"], r["edge_id"]))
        if not take:
            raise ValueError("Insufficient valid and diverse candidates")
        take[0]["selected"] = 1
        for rank, record in enumerate(take, 1):
            record["rank"] = rank
        ranking.extend(take + excluded)
        selected.append(take[0]["edge_id"])

    added, train_rows, schedules, selection = io.BytesIO(), [], {}, []
    baselines = {}
    for seed in SEEDS:
        order = old_orders[seed]["pure"]
        schedules[str(seed)] = {"pure": order}
        base = args.baseline/f"seed{seed}"/"pure"
        marker = json.loads((base/"done_710.json").read_bytes())
        if (marker["bundle"], marker["schedule"], marker["optimizer_steps"], marker["arm"], marker["seed"]) != (
            BASE_SHA256, sha(json.dumps(order).encode()), 710, "pure", seed
        ):
            raise ValueError("Pure D0 identity differs")
        if sha((base/"predictions_710.csv").read_bytes()) != marker["files"]["predictions_710.csv"]:
            raise ValueError("Historical pure D0 prediction hash differs")
        baselines[str(seed)] = marker

    for index, edge in enumerate(selected, 1):
        raw, path_frames = cases[edge]
        ids = list(range(len(d0) + len(train_rows), len(d0) + len(train_rows) + len(path_frames)))
        added.write(raw)
        train_rows.extend(path_frames)
        sizes.extend([int(meta[edge]["atoms_per_image"])] * len(ids))
        for seed in SEEDS:
            job = schedules[str(seed)]
            job[f"path{index}"], job[f"replay{index}"] = make_orders(job["pure"], ids, sizes, seed)
        selection.append(dict(slot=index, cohort="historical_reference" if index <= 3 else "new_matched_label",
            edge_id=edge, material_id=meta[edge]["material_id"], chemsys=meta[edge]["chemsys"],
            anion_family=meta[edge]["anion_family"], images=len(ids), atoms=meta[edge]["atoms_per_image"],
            optimizer_steps=710 + len(ids) * 5, path_presentations_per_image=5, selected_ids=ids,
            prior_comparability=(
                "Phase1 protected single-path; same schedule after frame-ID remap" if index in (1, 3)
                else "Prior v2.0 P-versus-Q at 710 steps is a separate protocol" if index == 2
                else "No prior identical protected-schedule experiment selected"), **quality[edge]))

    # Compare treatment schedules by frame identity. Phase 1 stored different
    # frame indices, so compare keys rather than raw order integers.
    old_frames = rows(frozen["training_frames.csv"])
    with ZipFile(args.cr_delta) as z:
        cr = json.loads(z.read("plan.json"))
        cr_rows = rows(z.read("extra_frames.csv"))
    if cr["base_bundle"] != BASE_SHA256 or cr["selected"]["edge_id"] != REFERENCES[2]:
        raise ValueError("Cr reference delta changed")

    def series(records, batches):
        return [[(records[i]["edge_id"], int(records[i]["frame"])) for i in b] for b in batches]

    for seed in SEEDS:
        for arm in ("path", "replay"):
            current = schedules[str(seed)]
            if series(old_frames, old_orders[seed][arm+"3"]) != series(d0 + train_rows, current[arm+"1"]):
                raise ValueError("mp-753143 protected schedule is not preserved")
            if series(old_frames + cr_rows, cr["orders"][str(seed)][arm]) != series(d0 + train_rows, current[arm+"3"]):
                raise ValueError("Cr protected schedule is not preserved")
    table = [dict(r, dft_forward_barrier_meV=1000. * max(truth[r["panel"], r["edge_id"]])) for r in panel]
    if len(table) != len(held) or len(held) != 107:
        raise ValueError("Expected one path per material in the 107 panel")
    files = {"added.xyz": added.getvalue(), "added_frames.csv": csv_bytes(train_rows),
             "orders.json": json.dumps(schedules, separators=(",", ":")).encode(),
             "reference_barriers.csv": csv_bytes(table), "baselines.json": json.dumps(baselines).encode()}
    frozen_plan = dict(id="B2-Overnight-01", base_bundle_sha256=BASE_SHA256, d0_sha256=D0_SHA256,
                       foundation_sha256=FOUNDATION, training=settings, score_panel="project_holdout107",
                       planned_fits=36, reserved_end_seconds=1800, max_window_seconds=25200,
                       candidate_pool_sha256=file_sha(args.pool), features_sha256=file_sha(args.init),
                       reference_archive_sha256=REFERENCE_SHA256, selection_rule=RULE, quality_policy=QUALITY_POLICY,
                       feature_groups=plan["path_feature_groups"], feature_iqr_scale=dict(zip(columns, scale.tolist())),
                       excluded_prior_matched_ids=sorted(previous), candidates=selection,
                       sha256={k: sha(v) for k,v in files.items()})
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output/"plan.json", (json.dumps(frozen_plan, indent=2, ensure_ascii=False) + "\n").encode())
    write(args.output/"candidates.csv", csv_bytes(selection))
    write(args.output/"candidate_ranking.csv", csv_bytes(ranking))
    quality_rows = [dict(edge_id=e, eligibility_reason=reason[e],
                         **{k: quality.get(e, {}).get(k, "") for k in quality[REFERENCES[0]]})
                    for e in sorted(pool)]
    write(args.output/"candidate_integrity.csv", csv_bytes(quality_rows))
    with ZipFile(args.bundle, "x", compression=ZIP_DEFLATED) as z:
        for key, raw in files.items():
            z.writestr(key, raw)
        z.write(args.output/"plan.json", "plan.json")
    print(json.dumps(selection, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "base", "pool", "init", "baseline", "previous_queue", "previous_transfer",
                 "cr_delta", "output", "bundle"):
        p.add_argument(name, type=Path)
    prepare(p.parse_args())
