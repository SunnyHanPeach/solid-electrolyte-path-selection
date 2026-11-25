"""Freeze a new project-level panel from geometry and historical manifests."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

from audit_signed_barriers import file_sha, rows, sha


def csv_bytes(records):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    return stream.getvalue().encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "history", "archives", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--mp753143-confirmation", action="store_true")
    args = parser.parse_args()
    source = args.history / "work/label-diagnostics"
    history = args.history / "outputs"
    sys.path[:0] = [str(source), str(args.project / "experiment2")]
    import freeze_material_panel as prior
    from package_panel_reconstruction import blind_geometries
    from prepare_data import NEB

    meta_file = args.project / "experiment2/path_study/init_features.csv"
    meta = {r["edge_id"]: r for r in prior.read_rows(meta_file)}
    database = history / "MACE_19对路径价值与DFT前特征数据库.zip"
    trained, evaluated = prior.historical(args.project, history, database, meta)
    exclusions = (
        (history / "MACE_独立材料评估面板_锁定.zip", ("frozen_panel.csv",)),
        (args.archives / "MACE_protected_D0_night_GPU_bundle.zip", ("evaluation_panels.csv",)),
        (args.archives / "LiNiOTi_独立与补充面板_盲化锁定.zip",
         ("strict_original_test/manifest.csv", "project_unseen_train_pilot/manifest.csv")),
    )
    if args.mp753143_confirmation:
        exclusions += ((args.archives / "net-path-panel/panel_blind.zip", ("panel.csv",)),)
    for source_zip, members in exclusions:
        with ZipFile(source_zip) as archive:
            for member in members:
                evaluated.update(r["edge_id"] for r in rows(archive.read(member)))
            if "training_frames.csv" in archive.namelist():
                trained.update(r["edge_id"] for r in rows(archive.read("training_frames.csv")))
    known = [meta[edge] for edge in sorted(trained | evaluated)]
    known_materials = {r["material_id"] for r in known}
    training_systems = {meta[edge]["chemsys"] for edge in trained}
    split = "test" if args.mp753143_confirmation else "train"
    pool = [r for r in meta.values() if r["split"] == split and r["material_id"] not in known_materials]

    args.output.mkdir(parents=True, exist_ok=True)
    prior.HERE = args.output.resolve()
    with ZipFile(NEB) as archive:
        near, neighbors, matches, structures, passes = prior.scan(
            archive, pool, known, source / "match.exe"
        )
    counts = Counter()
    selected, blocked = [], set(near)
    salt = "mp753143-test-replication-v1:" if args.mp753143_confirmation else "net-path-panel-v1:"
    def order(row):
        return hashlib.sha256((salt + row["edge_id"]).encode()).digest()
    for row in sorted(pool, key=order):
        material = row["material_id"]
        if material in blocked or counts[row["chemsys"]] >= 4:
            continue
        blocked.update(neighbors[material])
        blocked.add(material)
        counts[row["chemsys"]] += 1
        selected.append(row)
    selected.sort(key=lambda row: row["edge_id"])
    if not selected or (not args.mp753143_confirmation and len(selected) < 24):
        raise ValueError(f"Only {len(selected)} clean materials")
    manifest = csv_bytes([dict(
        edge_id=r["edge_id"], material_id=r["material_id"], chemsys=r["chemsys"],
        anion_family=r["anion_family"], original_split=r["split"],
        novel_chemistry=int(r["chemsys"] not in training_systems),
    ) for r in selected])
    frames, xyz, n_images = blind_geometries(args.project, selected)
    protocol = dict(
        case="v2.0/pair_1", priority_edge="mp-1177833_2_3_0_0_1",
        matched_alternative_edge="mp-771389_2_3_0_0_0", seeds=[17, 29, 43],
        selection="Fixed-geometry, project-held-out audit. Candidate P had negative mean effects in all three historical panels.",
        primary="P minus archived pure D0 forward barrier MAE. Q is a matched alternative and diagnostic.",
        gate="Upper paired 95% interval below -0.25 meV, all three seed means below zero and all leave-one-material-out means below zero.",
        stop="Run exposure-matched retraining only when the archived-weight audit supports a repeatable improvement.",
        historical_edges=dict(training=len(trained), evaluated=len(evaluated)),
        unique_excluded_materials=len(known_materials),
        pool_materials=len({r["material_id"] for r in pool}),
        near_exclusions=len(near), near_pairs=len(matches), structures_scanned=structures,
        duplicate_scan_passes=passes, materially_distinct_paths=len(selected),
        material_families=dict(Counter(r["anion_family"] for r in selected)),
        chemistry_new_vs_project_training=sum(r["chemsys"] not in training_systems for r in selected),
        exact_chemistry_cap=4, images=n_images,
        caveat="Source Train split, project-held-out by the audited inventory. No foundation-pretraining isolation claim.",
        source_sha256={"init_features":file_sha(meta_file), "archive":file_sha(NEB),
                       "historical_pair_data":file_sha(database), "matcher":file_sha(source / "match.exe"),
                       **{z.name:file_sha(z) for z,_ in exclusions}},
        assets_sha256={name:sha(data) for name,data in
                       (("panel.csv",manifest),("frames.csv",frames),("blind.xyz",xyz))},
    )
    if args.mp753143_confirmation:
        protocol.update(
            case="Phase 1/mp-753143 material replication",
            priority_edge="mp-753143_5_13_0_0_0",
            matched_alternative_edge=None,
            seeds=[17, 29, 43, 53, 67],
            selection="All deduplicated, project-unused original test materials under the locked cap. The separately sealed four materials are excluded.",
            primary="Matched replay minus path forward barrier MAE. Pure D0 minus path is the absolute gain.",
            gate="Report +0.25 meV practical gain only if the paired 95% chemical-system bootstrap lower bound exceeds it, all five seed means are positive and leave-one-material-out means stay positive.",
            stop="Characterize transfer on this panel. No training or parameter selection on these results.",
            caveat="Original Test split, newly held out by audited project inventory. No foundation-pretraining isolation claim. The small panel is a replication probe.",
        )
    with ZipFile(args.output / "panel_blind.zip", "x", compression=ZIP_DEFLATED) as archive:
        for name, raw in (("panel.csv", manifest), ("frames.csv", frames), ("blind.xyz", xyz),
                          ("lock.json", (json.dumps(protocol, indent=2)+"\n").encode())):
            archive.writestr(name, raw)
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
