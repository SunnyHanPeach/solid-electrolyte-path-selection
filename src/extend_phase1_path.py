"""Run a feature-selected transfer probe without re-uploading existing P0 assets."""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from audit_signed_barriers import file_sha, load_frozen_data, rows, sha
from freeze_net_path_panel import csv_bytes
from run_force_weighting import atomic_json, load_bundle, train

SEEDS=(17,29,43)


def prepare(args, selected=None, selection=None):
    plan,files=load_bundle(args.bundle)
    table=rows(args.features.read_bytes())
    done=set(plan["selected"])
    taken=[r for r in table if r["edge_id"] in done]
    covered={r["anion_family"] for r in taken}
    columns=[x for g in plan["path_feature_groups"] for x in g]
    eligible=[r for r in table if r["eligible"]=="1"]
    axis=np.array([[float(r[c]) for c in columns] for r in eligible])
    lo,hi=np.quantile(axis,[.25,.75],axis=0)
    scale=np.where(hi>lo,hi-lo,1)
    anchors=np.array([[float(r[c]) for c in columns] for r in taken])/scale
    remaining=[r for r in eligible if r["edge_id"] not in done and r["anion_family"] not in covered]
    if not remaining:
        raise ValueError("No eligible path from an uncovered anion category")
    def distance(row):
        return float(np.linalg.norm(anchors-np.array([float(row[c]) for c in columns])/scale,axis=1).min())
    choice=selected or min(remaining,key=lambda r:(-distance(r),r["edge_id"]))
    if choice not in eligible or choice["edge_id"] in done:
        raise ValueError("Next path already trained in P0 or ineligible")
    edge=choice["edge_id"]
    sys.path.insert(0,str(args.project/"experiment2"))
    from prepare_data import NEB, labeled_frame, metadata, raw_frames
    from audit_signed_barriers import REFERENCE_SHA256
    if file_sha(NEB)!=REFERENCE_SHA256:
        raise ValueError("DFT archive changed")
    path_rows,xyz=[],io.BytesIO()
    with ZipFile(NEB) as archive:
        for i,_,n,header,atoms in raw_frames(archive,f"nebDFT2k/{edge}_relaxed.xyz"):
            energy,*_=labeled_frame(metadata(header),atoms)
            path_rows.append(dict(edge_id=edge,material_id=choice["material_id"],frame=i,
                                 role="candidate",n_atoms=len(atoms),energy_eV=energy))
            xyz.writelines([n,header,*atoms])
    if len(path_rows)!=int(choice["init_images"]) or any(
            r["n_atoms"]!=int(choice["atoms_per_image"]) for r in path_rows):
        raise ValueError("Incomplete candidate path")
    if choice["material_id"] in {r["material_id"] for r in rows(files["evaluation_frames.csv"])}:
        raise ValueError("Candidate material overlaps the P0 evaluation panel")
    frames=rows(files["training_frames.csv"])
    sizes=[int(r["n_atoms"]) for r in frames]
    n=int(choice["atoms_per_image"])
    control=[i for i in range(278) if sizes[i]==n]
    ids=list(range(len(frames),len(frames)+len(path_rows)))
    jobs={}
    for item in json.loads(files["jobs.json"]):
        seed=item["seed"]
        base=item["jobs"]["pure"]
        rng=np.random.default_rng(seed)
        extra=[int(i) for _ in range(5) for i in rng.permutation(ids)]
        positions={((2*k+1)*len(base))//(2*len(extra)):frame for k,frame in enumerate(extra)}
        order,replay=[],[]
        used=Counter()
        for k,batch in enumerate(base):
            if k in positions:
                companion=batch[0]
                r=min((i for i in control if i!=companion),key=lambda i:(used[i],i))
                used[r]+=1
                order.append([companion,positions[k]])
                replay.append([companion,r])
            order.append(batch)
            replay.append(batch)
        extended=sizes+[n]*len(ids)
        if len(order)!=710+5*len(ids) or any([extended[i] for i in p]!=[extended[i] for i in q]
                                             for p,q in zip(order,replay)):
            raise ValueError("Exposure matching failed")
        jobs[str(seed)]=dict(path=order,replay=replay,pure=base)
    content={"extra.xyz":xyz.getvalue(),"extra_frames.csv":csv_bytes(path_rows)}
    lock=dict(base_bundle=file_sha(args.bundle),feature_table=file_sha(args.features),selected=choice,
              selection=selection or "Farthest minimum IQR-scaled BVSE Init feature distance to P0 medoids within previously uncovered anion classes.",
              margin_meV=.25,orders=jobs,sha256={k:sha(v) for k,v in content.items()})
    with ZipFile(args.delta,"x",compression=ZIP_DEFLATED) as archive:
        archive.writestr("plan.json",json.dumps(lock,indent=2))
        for name,data in content.items():
            archive.writestr(name,data)
    print(json.dumps(dict(path=edge,chemistry=choice["chemsys"],images=len(path_rows),
                          atoms=n,updates=len(order)),indent=2),flush=True)


def next_block(args):
    from types import SimpleNamespace

    plan,_=load_bundle(args.bundle)
    current,_=load_delta(args.bundle,args.current)
    table=rows(args.features.read_bytes())
    known=set(plan["selected"])|{current["selected"]["edge_id"]}
    previous=[r for r in table if r["edge_id"] in known]
    target=next(r for r in previous if r["edge_id"]=="mp-753143_5_13_0_0_0")
    features=[name for group in plan["path_feature_groups"] for name in group]
    eligible=[r for r in table if r["eligible"]=="1"]
    matrix=np.array([[float(r[k]) for k in features] for r in eligible])
    low,high=np.quantile(matrix,[.25,.75],axis=0)
    scale=np.where(high>low,high-low,1)
    def vector(row):
        return np.array([float(row[k]) for k in features])/scale
    def novel(row):
        return min(float(np.linalg.norm(vector(row)-vector(r))) for r in previous)
    used_materials={r["material_id"] for r in previous}
    used_systems={r["chemsys"] for r in previous}
    def pool():
        return [r for r in eligible if r["material_id"] not in used_materials and r["chemsys"] not in used_systems]

    args.output.mkdir(parents=True)
    records=[]
    for label in ("new_anion","near_geometry","feature_gap"):
        options=pool()
        if label=="new_anion":
            categories={r["anion_family"] for r in previous}
            options=[r for r in options if r["anion_family"] not in categories]
            rule="Uncovered anion category. Maximize minimum IQR-scaled feature distance."
        elif label=="near_geometry":
            options=[r for r in options if r["anion_family"]==target["anion_family"]
                     and r["coordination_range"]!=target["coordination_range"]]
            if not options:
                options=[r for r in pool() if r["coordination_range"]!=target["coordination_range"]]
            rule="Nearest IQR-scaled BVSE Init features to the F–Li–Ni–O path with different coordination range and chemistry. Prefer the same anion category."
        else:
            rule="Maximize minimum IQR-scaled distance to prior choices, with distinct material and exact chemistry."
        if not options:
            raise ValueError(f"No candidate for {label}")
        if label=="near_geometry":
            row=min(options,key=lambda r:(float(np.linalg.norm(vector(r)-vector(target))),r["edge_id"]))
        else:
            row=min(options,key=lambda r:(-novel(r),r["edge_id"]))
        path=args.output/f"{label}.zip"
        prepare(SimpleNamespace(project=args.project,bundle=args.bundle,features=args.features,delta=path),
                selected=row,selection=rule)
        used_materials.add(row["material_id"])
        used_systems.add(row["chemsys"])
        previous.append(row)
        records.append(dict(arm=label,path=row["edge_id"],chemistry=row["chemsys"],
                            delta_sha256=file_sha(path),selection_rule=rule))
    atomic_json(args.output/"queue.json",dict(
        planned_fits=18,seeds=SEEDS,previous_transfer_sha256=file_sha(args.current),paths=records,
    ))


def load_delta(bundle, path):
    with ZipFile(path) as archive:
        lock=json.loads(archive.read("plan.json"))
        files={k:archive.read(k) for k in lock["sha256"]}
    if lock["base_bundle"]!=file_sha(bundle) or any(sha(v)!=lock["sha256"][k] for k,v in files.items()):
        raise ValueError("Delta identity mismatch")
    return lock,files


def run(args):
    plan,files=load_bundle(args.bundle)
    lock,addition=load_delta(args.bundle,args.delta)
    frames=rows(files["training_frames.csv"])+rows(addition["extra_frames.csv"])
    files["training.xyz"]+=addition["extra.xyz"]
    arm=args.arm
    order=lock["orders"][str(args.seed)][arm]
    args.arm=[arm]
    args.until=len(order)
    args.device="cuda"
    args.threads=4
    args.smoke=None
    train(args,files,frames,plan["training"],order,{arm:{}},
          dict(bundle=lock["base_bundle"],delta=file_sha(args.delta),seed=args.seed,
               schedule=sha(json.dumps(order).encode()),implementation=file_sha(Path(__file__))),
          milestones=(400,len(order)))


def score(args):
    lock,_=load_delta(args.bundle,args.delta)
    _,_,truth=load_frozen_data(args.bundle,args.project)
    with ZipFile(args.bundle) as archive:
        manifest={r["edge_id"]:r for r in rows(archive.read("evaluation_panels.csv"))}
    edges=sorted(manifest)
    expected={(e,i) for (_,e),v in truth.items() for i in range(1,len(v)+1)}
    errors={}
    environments=set()
    for seed in SEEDS:
        for arm in ("pure","path","replay"):
            order=lock["orders"][str(seed)][arm]
            step=len(order)
            root=args.original if arm=="pure" else args.output
            folder=root/f"seed{seed}"/arm
            doc=json.loads((folder/f"done_{step}.json").read_bytes())
            if (doc["bundle"],doc["schedule"],doc["arm"],doc["seed"])!=(
                    lock["base_bundle"],sha(json.dumps(order).encode()),arm,seed) or (
                    arm!="pure" and doc["delta"]!=file_sha(args.delta)):
                raise ValueError("Invalid train or pure-baseline identity")
            environments.add(json.dumps(doc["environment"],sort_keys=True))
            name=f"predictions_{step}.csv"
            raw=(folder/name).read_bytes()
            if sha(raw)!=doc["files"][name]:
                raise ValueError("Prediction checksum mismatch")
            records=rows(raw)
            pred={(r["edge_id"],int(r["frame"])):float(r["energy_eV"]) for r in records}
            if len(records)!=len(expected) or pred.keys()!=expected or not np.isfinite(list(pred.values())).all():
                raise ValueError("Invalid evaluation predictions")
            for (_,edge),ref in truth.items():
                c=np.array([pred[edge,i] for i in range(1,len(ref)+1)])
                errors[seed,arm,edge]=1000*abs(c.max()-c[0]-ref.max())
    if len(environments)!=1:
        raise ValueError("Distinct environments for baseline and transfer arm")
    systems=sorted({r["chemsys"] for r in manifest.values()})
    cols=[systems.index(manifest[e]["chemsys"]) for e in edges]
    w=np.random.default_rng(20261125).multinomial(
        len(systems),np.full(len(systems),1/len(systems)),size=10000)[:,cols]
    draws=w/w.sum(axis=1,keepdims=True)
    gains={}
    table=[]
    for name,control in (("absolute","pure"),("net","replay")):
        d=np.array([[errors[s,control,e]-errors[s,"path",e] for e in edges] for s in SEEDS])
        average=d.mean(axis=0)
        lo,hi=np.quantile(draws@average,[.025,.975])
        sm=d.mean(axis=1)
        gains[name]=dict(mean=float(average.mean()),interval95=[float(lo),float(hi)],
                         seed_means=sm.tolist(),
                         positive=bool(lo>.25 and sm.min()>0),negative=bool(hi<-.25 and sm.max()<0))
        for i,s in enumerate(SEEDS):
            table.extend(dict(material_id=manifest[e]["material_id"],chemsys=manifest[e]["chemsys"],
                              seed=s,metric=name,gain_meV=float(v)) for e,v in zip(edges,d[i]))
    (args.output/"paired_materials.csv").write_bytes(csv_bytes(table))
    atomic_json(args.output/"summary.json",dict(path=lock["selected"],materials=len(edges),results=gains))
    print(json.dumps(gains,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(required=True)
    for mode,action,paths in (
        ("prepare",prepare,("project","bundle","features","delta")),
        ("next-block",next_block,("project","bundle","features","current","output")),
        ("run",run,("bundle","delta","output")),
        ("score",score,("project","bundle","delta","output","original")),
    ):
        cmd=sub.add_parser(mode)
        for n in paths:
            cmd.add_argument(n,type=Path)
        if mode=="run":
            cmd.add_argument("--seed",type=int,choices=SEEDS,required=True)
            cmd.add_argument("--arm",choices=("path","replay"),required=True)
        cmd.set_defaults(function=action)
    args=p.parse_args()
    args.function(args)
