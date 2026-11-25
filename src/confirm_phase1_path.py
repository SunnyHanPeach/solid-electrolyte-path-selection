"""Confirm the P0 net-positive path with two new paired seeds."""
import argparse
from collections import Counter
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from audit_signed_barriers import file_sha, load_frozen_data, rows, sha
from freeze_net_path_panel import csv_bytes
from run_force_weighting import atomic_json, load_bundle, train

SEEDS=(53,67)
ARMS=("path3","replay3","pure")
EDGE="mp-753143_5_13_0_0_0"


def prepare(args):
    _,files=load_bundle(args.bundle)
    _,prior=load_bundle(args.protected)
    frames=rows(files["training_frames.csv"])
    base_rows=rows(prior["training_frames.csv"])[:278]
    if frames[:278]!=base_rows:
        raise ValueError("Changed canonical D0 frames")
    base={r["seed"]:r["jobs"]["pure_D0"] for r in json.loads(prior["jobs.json"])}
    ids=[i for i,r in enumerate(frames) if i>=278 and r["edge_id"]==EDGE]
    if len(ids)!=5:
        raise ValueError("Expected a complete five-image path")
    sizes=[int(r["n_atoms"]) for r in frames]
    pool=[i for i in range(278) if sizes[i]==sizes[ids[0]]]
    jobs={}
    for seed in SEEDS:
        pure=base[seed]
        if len(pure)!=710 or any(len(b)!=2 or any(i<0 or i>=278 for i in b) for b in pure):
            raise ValueError("Unexpected new-seed D0 schedule")
        rng=np.random.default_rng(seed)
        planned=[int(i) for _ in range(5) for i in rng.permutation(ids)]
        positions={((2*k+1)*len(pure))//(2*len(planned)):cid for k,cid in enumerate(planned)}
        used=Counter()
        path,replay=[],[]
        for step,batch in enumerate(pure):
            if step in positions:
                companion=batch[0]
                rid=min((i for i in pool if i!=companion),key=lambda i:(used[i],i))
                used[rid]+=1
                path.append([companion,positions[step]])
                replay.append([companion,rid])
            path.append(batch)
            replay.append(batch)
        if len(path)!=735 or any([sizes[i] for i in a]!=[sizes[i] for i in b]
                                 for a,b in zip(path,replay)):
            raise ValueError("Exposure matching failed")
        jobs[str(seed)]={"path3":path,"replay3":replay,"pure":pure}
    atomic_json(args.output,dict(bundle=file_sha(args.bundle),canonical_source=file_sha(args.protected),
                path=EDGE,panel_role="Same 107 project-held-out materials; tests seed replication.",
                margin_meV=.25,orders=jobs))
    print("Verified all six new-seed schedules",flush=True)


def run(args):
    plan,files=load_bundle(args.bundle)
    lock=json.loads(args.design.read_bytes())
    if lock["bundle"]!=file_sha(args.bundle):
        raise ValueError("Wrong P0 bundle for seed confirmation")
    order=lock["orders"][str(args.seed)][args.arm]
    args.until=len(order)
    arm=args.arm
    args.arm=[arm]
    args.smoke=None
    args.device="cuda"
    args.threads=4
    train(args,files,rows(files["training_frames.csv"]),plan["training"],order,{arm:{}},
          dict(bundle=lock["bundle"],schedule=sha(json.dumps(order).encode()),seed=args.seed,
               confirmation=file_sha(args.design),implementation=file_sha(Path(__file__))),
          milestones=(len(order),))


def score(args):
    lock=json.loads(args.design.read_bytes())
    _, _, references=load_frozen_data(args.bundle,args.project)
    with ZipFile(args.bundle) as archive:
        materials={r["edge_id"]:r for r in rows(archive.read("evaluation_panels.csv"))}
    edges=sorted(materials)
    expected={(edge,i) for (_,edge),curve in references.items() for i in range(1,len(curve)+1)}
    errors={}
    environments=set()
    for seed in SEEDS:
        for arm in ARMS:
            order=lock["orders"][str(seed)][arm]
            step=len(order)
            base=args.output/f"seed{seed}"/arm
            doc=json.loads((base/f"done_{step}.json").read_bytes())
            if (doc["bundle"],doc["confirmation"],doc["schedule"],doc["arm"],doc["seed"]) != (
                    file_sha(args.bundle),file_sha(args.design),sha(json.dumps(order).encode()),arm,seed):
                raise ValueError("Invalid confirmation checkpoint")
            environments.add(json.dumps(doc["environment"],sort_keys=True))
            name=f"predictions_{step}.csv"
            raw=(base/name).read_bytes()
            if sha(raw)!=doc["files"][name]:
                raise ValueError("Changed blind prediction")
            curves={}
            for r in rows(raw):
                edge=r["edge_id"]
                key=edge,int(r["frame"])
                value=float(r["energy_eV"])
                if key in curves or not np.isfinite(value) or r["material_id"]!=materials[edge]["material_id"]:
                    raise ValueError("Invalid evaluation output")
                curves[key]=value
            if curves.keys()!=expected:
                raise ValueError("Incomplete evaluation output")
            for (_,edge),ref in references.items():
                c=np.array([curves[edge,i] for i in range(1,len(ref)+1)])
                errors[seed,arm,edge]=1000*abs(c.max()-c[0]-ref.max())
    if len(environments)!=1:
        raise ValueError("Different training environments across confirmation arms")
    families=sorted({r["chemsys"] for r in materials.values()})
    lookup=[families.index(materials[e]["chemsys"]) for e in edges]
    w=np.random.default_rng(20261125).multinomial(
        len(families),np.full(len(families),1/len(families)),size=10000)[:,lookup]
    w=w/w.sum(axis=1,keepdims=True)
    historical=rows(args.original.read_bytes())
    by_material=[]
    report={}
    for metric,control in (("absolute","pure"),("net","replay3")):
        old={(int(r["seed"]),r["material"]):float(r["gain_meV"]) for r in historical
             if r["path"]==EDGE and r["metric"]==metric}
        new=np.array([[errors[s,control,e]-errors[s,"path3",e] for e in edges] for s in SEEDS])
        previous=np.array([[old[seed,materials[e]["material_id"]] for e in edges]
                           for seed in (17,29,43)])
        out=report[metric]={}
        for kind, matrix in (("new_two_seeds",new),("combined_five_seeds",np.vstack([previous,new]))):
            avg=matrix.mean(axis=0)
            lo,hi=np.quantile(w@avg,[.025,.975])
            seed_means=matrix.mean(axis=1)
            out[kind]=dict(mean_meV=float(avg.mean()),interval95_meV=[float(lo),float(hi)],
                           seed_means_meV=seed_means.tolist(),
                           practical_positive=bool(lo>.25 and seed_means.min()>0))
        for i, seed in enumerate(SEEDS):
            by_material.extend(dict(material_id=materials[e]["material_id"],chemsys=materials[e]["chemsys"],
                                    seed=seed,metric=metric,gain_meV=float(v))
                               for e,v in zip(edges,new[i]))
    (args.output/"paired_materials.csv").write_bytes(csv_bytes(by_material))
    atomic_json(args.output/"summary.json",dict(
        edge=EDGE,materials=len(edges),seeds=SEEDS,results=report,
        interpretation="Same material panel, independent seeds. Positive gains favor the path.",
    ))
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(required=True)
    a=sub.add_parser("prepare")
    for n in ("bundle","protected","output"):
        a.add_argument(n,type=Path)
    a.set_defaults(function=prepare)
    b=sub.add_parser("run")
    for n in ("bundle","design","output"):
        b.add_argument(n,type=Path)
    b.add_argument("--arm",choices=ARMS,required=True)
    b.add_argument("--seed",type=int,choices=SEEDS,required=True)
    b.set_defaults(function=run)
    c=sub.add_parser("score")
    for n in ("project","bundle","design","output","original"):
        c.add_argument(n,type=Path)
    c.set_defaults(function=score)
    args=p.parse_args()
    args.function(args)
