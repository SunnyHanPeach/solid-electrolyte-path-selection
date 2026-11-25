"""Freeze, run and score the Phase 1 single-path paired pilot."""
import argparse
from collections import Counter
import io
from itertools import combinations
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from audit_archived_net_path import blind_panel
from audit_signed_barriers import file_sha, load_frozen_data, rows, sha
from freeze_net_path_panel import csv_bytes
from run_force_weighting import atomic_json, load_bundle, train

SEEDS = (17, 29, 43)
FEATURES = (
    ("bvse_forward_eV", "hop_displacement_A"),
    ("min_Li_anion_distance_A", "coordination_range"),
    ("path_excess_length_A", "total_turn_deg"),
)
ALREADY_AUDITED = {"mp-1177833_2_3_0_0_1", "mp-771389_2_3_0_0_0"}
MARGIN = .25


def prepare(args):
    args.output.mkdir(parents=True)
    old, assets = load_bundle(args.protected)
    training = rows(assets["training_frames.csv"])[:278]
    sizes = [int(r["n_atoms"]) for r in training]
    counts = Counter(sizes)
    with ZipFile(args.history) as archive:
        edges = {r["edge"] for r in rows(archive.read("continuous_values.csv"))}
    columns = [name for group in FEATURES for name in group]
    meta_path = args.project / "experiment2/path_study/init_features.csv"
    meta = {r["edge_id"]: r for r in rows(meta_path.read_bytes())}
    if len(edges) != 38 or not edges <= meta.keys():
        raise ValueError("The frozen 38-path pool is incomplete")
    table = []
    for edge in sorted(edges):
        row = meta[edge]
        eligible = edge not in ALREADY_AUDITED and counts[int(row["atoms_per_image"])] >= 2
        table.append({k: row[k] for k in ("edge_id", "material_id", "chemsys", "anion_family",
                      "init_images", "atoms_per_image", *columns)} | {"eligible": int(eligible)})
    eligible = [r for r in table if r["eligible"]]
    features = np.array([[float(row[c]) for c in columns] for row in eligible])
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite candidate feature")
    qlo, qhi = np.quantile(features, [.25, .75], axis=0)
    distance = np.linalg.norm((features[:,None] - features[None,:]) /
                              np.where(qhi > qlo, qhi-qlo, 1), axis=2)

    def select(k):
        best = None
        for ids in combinations(range(len(eligible)), k):
            chosen = [eligible[i] for i in ids]
            if len({r["chemsys"] for r in chosen}) != k or len({r["anion_family"] for r in chosen}) < 2:
                continue
            trial = (float(distance[:, ids].min(axis=1).sum()), ids)
            if best is None or trial < best:
                best = trial
        if best is None:
            raise ValueError("No eligible separated cluster medoids")
        return best[1]
    groups = {str(k): [eligible[i]["edge_id"] for i in select(k)] for k in (3, 4, 5)}
    selected = groups["3"]
    evaluations, blind = blind_panel(args.panel)
    panel = rows(blind["panel.csv"])
    if ({meta[edge]["material_id"] for edge in selected} |
            {r["material_id"] for r in training}) & {r["material_id"] for r in panel}:
        raise ValueError("Training and evaluation materials overlap")

    sys.path.insert(0, str(args.project / "experiment2"))
    from prepare_data import NEB, labeled_frame, metadata, raw_frames
    if file_sha(NEB) != evaluations["source_sha256"]["archive"]:
        raise ValueError("Unexpected reference archive")
    original = io.BytesIO(assets["training.xyz"])
    xyz = io.BytesIO()
    for _ in range(278):
        n = original.readline()
        xyz.writelines([n, original.readline(), *(original.readline() for _ in range(int(n)))])
    frame_ids = {}
    with ZipFile(NEB) as archive:
        for number, edge in enumerate(selected, 1):
            arm = f"path{number}"
            frame_ids[arm] = []
            for i, _, n, head, body in raw_frames(archive, f"nebDFT2k/{edge}_relaxed.xyz"):
                energy, *_ = labeled_frame(metadata(head), body)
                if len(body) != int(meta[edge]["atoms_per_image"]):
                    raise ValueError("Candidate size changed")
                frame_ids[arm].append(len(training))
                training.append(dict(edge_id=edge, material_id=meta[edge]["material_id"], frame=i,
                                     role="candidate", n_atoms=len(body), energy_eV=energy))
                xyz.writelines([n, head, *body])
            if len(frame_ids[arm]) != int(meta[edge]["init_images"]):
                raise ValueError("Incomplete candidate path")
    historical = {r["seed"]: r["jobs"] for r in json.loads(assets["jobs.json"])}
    jobs = []
    manifest = []
    for seed in SEEDS:
        base = historical[seed]["pure_D0"]
        if len(base) != 710:
            raise ValueError("Unexpected D0 schedule")
        orders = {"pure": base}
        for arm, ids in frame_ids.items():
            n = len(ids)
            atom_count = int(training[ids[0]]["n_atoms"])
            pool = [i for i, count in enumerate(sizes) if count == atom_count]
            rng = np.random.default_rng(seed)
            planned = [int(i) for _ in range(5) for i in rng.permutation(ids)]
            positions = {((2*k+1)*len(base))//(2*len(planned)): cid
                         for k, cid in enumerate(planned)}
            if len(positions) != len(planned):
                raise ValueError("Insertion positions collided")
            used = Counter()
            order, replay = [], []
            for step, batch in enumerate(base):
                if step in positions:
                    companion = batch[0]
                    rid = min((i for i in pool if i != companion), key=lambda i:(used[i],i))
                    used[rid] += 1
                    order.append([companion, positions[step]])
                    replay.append([companion, rid])
                order.append(batch)
                replay.append(batch)
            all_sizes = [int(r["n_atoms"]) for r in training]
            if any([all_sizes[i] for i in p] != [all_sizes[i] for i in q]
                   for p,q in zip(order,replay)):
                raise ValueError("Per-slot atom exposure differs")
            orders[arm] = order
            orders[arm.replace("path","replay")] = replay
        jobs.append(dict(seed=seed, jobs=orders))
        for arm, order in orders.items():
            suffix = arm[-1]
            edge = selected[int(suffix)-1] if suffix.isdigit() else ""
            inserts = len(order)-710
            manifest.append(dict(seed=seed, arm=arm, edge_id=edge, optimizer_steps=len(order),
                                 insertion_updates=inserts,
                                 added_atom_images=inserts*int(meta[edge]["atoms_per_image"]) if edge else 0,
                                 order_sha256=sha(json.dumps(order).encode())))
    frames = rows(blind["frames.csv"])
    for row in frames:
        row["panel"] = "project_holdout107"
    panels = [dict(r,panel="project_holdout107",images=sum(f["edge_id"]==r["edge_id"] for f in frames))
              for r in panel]
    members = {"training_frames.csv":csv_bytes(training),"training.xyz":xyz.getvalue(),
               "evaluation_frames.csv":csv_bytes(frames),"evaluation_panels.csv":csv_bytes(panels),
               "evaluation.xyz":blind["blind.xyz"],"foundation.model":assets["foundation.model"],
               "jobs.json":json.dumps(jobs).encode()}
    lock = dict(training=old["training"], selected=selected, centers_sensitivity=groups,
                path_feature_groups=FEATURES, primary_effect_margin_meV=MARGIN,
                feature_file_sha256=file_sha(meta_path), evaluation_panel_sha256=file_sha(args.panel),
                source_bundle_sha256=file_sha(args.protected), selection_basis="BVSE Init only",
                sha256={n:sha(data) for n,data in members.items()})
    atomic_json(args.output/"plan.json",lock)
    (args.output/"candidate_path_features.csv").write_bytes(csv_bytes(table))
    (args.output/"experiment_manifest.csv").write_bytes(csv_bytes(manifest))
    with ZipFile(args.output/"bundle.zip","x",compression=ZIP_DEFLATED) as archive:
        for name, blob in members.items():
            archive.writestr(name,blob)
        archive.write(args.output/"plan.json","plan.json")
    print(json.dumps(dict(selected=selected, eligible=len(eligible), jobs=len(manifest))), flush=True)


def run(args):
    plan, files = load_bundle(args.bundle)
    jobs = {r["seed"]:r["jobs"] for r in json.loads(files["jobs.json"])}
    order = jobs[args.seed][args.arm]
    args.until = len(order)
    args.arm = [args.arm]
    train(args,files,rows(files["training_frames.csv"]),plan["training"],order,{args.arm[0]:{}},
          dict(bundle=file_sha(args.bundle), schedule=sha(json.dumps(order).encode()), seed=args.seed,
               implementation=file_sha(Path(__file__))), milestones=(400,len(order)))


def score(args):
    plan, files = load_bundle(args.bundle)
    _, _, refs = load_frozen_data(args.bundle,args.project)
    eval_rows = rows(files["evaluation_frames.csv"])
    grid = {(r["edge_id"],int(r["frame"])): r["material_id"] for r in eval_rows}
    jobs = {e["seed"]:e["jobs"] for e in json.loads(files["jobs.json"])}
    errors = {}
    env = set()
    for seed in SEEDS:
        for arm, order in jobs[seed].items():
            base = args.output/f"seed{seed}"/arm
            step = len(order)
            doc = json.loads((base/f"done_{step}.json").read_bytes())
            if (doc["arm"],doc["seed"],doc["bundle"],doc["schedule"],doc["optimizer_steps"]) != (
                    arm,seed,file_sha(args.bundle),sha(json.dumps(order).encode()),step):
                raise ValueError("Run identity error")
            name=f"predictions_{step}.csv"
            raw=(base/name).read_bytes()
            if sha(raw) != doc["files"][name]:
                raise ValueError("Prediction checksum changed")
            env.add(json.dumps(doc["environment"],sort_keys=True))
            records = rows(raw)
            curves = {(r["edge_id"],int(r["frame"])):float(r["energy_eV"]) for r in records}
            if len(records) != len(grid) or curves.keys() != grid.keys() or any(
                grid[r["edge_id"],int(r["frame"])] != r["material_id"] for r in records
            ) or not np.isfinite(list(curves.values())).all():
                raise ValueError("Wrong blind scoring panel")
            for (_,edge), ref in refs.items():
                c = np.array([curves[edge,i] for i in range(1,len(ref)+1)])
                errors[seed,arm,edge] = 1000 * abs(c.max()-c[0]-ref.max())
    if len(env)!=1:
        raise ValueError("Scoring requires a common run environment")
    edge_order = sorted(e for _,e in refs)
    metadata = {r["edge_id"]:r for r in rows(files["evaluation_panels.csv"])}
    families = sorted({metadata[e]["chemsys"] for e in edge_order})
    cluster = np.array([families.index(metadata[e]["chemsys"]) for e in edge_order])
    weights = np.random.default_rng(20261125).multinomial(
        len(families),np.full(len(families),1/len(families)),size=10000)[:,cluster]
    draws = weights / weights.sum(axis=1,keepdims=True)
    table, reports = [], {}
    for n, edge in enumerate(plan["selected"],1):
        candidate = f"path{n}"
        values = {}
        for metric, control in (("absolute","pure"),("net",f"replay{n}")):
            delta = np.array([[errors[s,control,e]-errors[s,candidate,e] for e in edge_order]
                             for s in SEEDS])
            means = delta.mean(axis=0)
            lo, hi = np.quantile(draws @ means,[.025,.975])
            per_seed = delta.mean(axis=1)
            status = ("positive" if lo > MARGIN and min(per_seed) > 0 else "negative"
                      if hi < -MARGIN and max(per_seed) < 0 else "equivalent"
                      if lo >= -MARGIN and hi <= MARGIN else "uncertain")
            values[metric] = dict(mean=float(delta.mean()), interval95=[float(lo),float(hi)],
                                  seed_means=per_seed.tolist(),status=status)
            for i,s in enumerate(SEEDS):
                table.extend(dict(path=edge,material=metadata[e]["material_id"],chemsys=metadata[e]["chemsys"],
                                  seed=s,metric=metric,gain_meV=float(v))
                             for e,v in zip(edge_order,delta[i]))
        reports[edge] = values
    (args.output/"paired_materials.csv").write_bytes(csv_bytes(table))
    atomic_json(args.output/"summary.json",dict(materials=len(edge_order),clusters=len(families),
                 axis="positive favors actual new path",margin_meV=MARGIN, results=reports))
    print(json.dumps(reports,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(required=True)
    prepare_cmd = modes.add_parser("prepare")
    for n in ("project","protected","history","panel","output"):
        prepare_cmd.add_argument(n,type=Path)
    prepare_cmd.set_defaults(function=prepare)
    run_cmd = modes.add_parser("run")
    for n in ("bundle","output"):
        run_cmd.add_argument(n,type=Path)
    run_cmd.add_argument("--arm",choices=("pure","path1","path2","path3","replay1","replay2","replay3"),required=True)
    run_cmd.add_argument("--seed",choices=SEEDS,type=int,required=True)
    run_cmd.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    run_cmd.add_argument("--threads",type=int,default=4)
    run_cmd.add_argument("--smoke",type=int)
    run_cmd.set_defaults(function=run)
    score_cmd = modes.add_parser("score")
    for n in ("project","bundle","output"):
        score_cmd.add_argument(n,type=Path)
    score_cmd.set_defaults(function=score)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
