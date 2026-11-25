"""Score blind P1 force predictions on validated DFT geometries."""
import argparse
from collections import defaultdict
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np

from audit_signed_barriers import file_sha, rows
from freeze_net_path_panel import csv_bytes
from run_force_weighting import atomic_json, load_bundle


def references(args, files):
    from ase.geometry import find_mic
    from ase.io import read

    sys.path.insert(0, str(args.project/"experiment2"))
    from prepare_data import NEB, raw_frames
    from audit_signed_barriers import REFERENCE_SHA256
    if file_sha(NEB)!=REFERENCE_SHA256:
        raise ValueError("DFT reference archive changed")
    frames=rows(files["evaluation_frames.csv"])
    blind=read(io.StringIO(files["evaluation.xyz"].decode()),index=":",format="extxyz")
    paths=defaultdict(list)
    for i,row in enumerate(frames):
        paths[row["edge_id"]].append(i)
    offsets=np.cumsum([0]+[int(r["n_atoms"]) for r in frames])
    info={r["edge_id"]:r for r in rows((args.project/"experiment2/path_study/init_features.csv").read_bytes())}
    truth=np.empty((offsets[-1],3))
    details={}
    excluded=[]
    with ZipFile(NEB) as archive:
        for edge, ix in paths.items():
            stream=io.BytesIO()
            for _,_,n,head,body in raw_frames(archive,f"nebDFT2k/{edge}_relaxed.xyz"):
                stream.writelines([n,head,*body])
            labeled=read(io.StringIO(stream.getvalue().decode()),index=":",format="extxyz")
            if len(labeled)!=len(ix):
                raise ValueError(f"Wrong DFT path count: {edge}")
            energy=[]
            for i,a in zip(ix,labeled):
                b=blind[i]
                if not (np.array_equal(a.numbers,b.numbers) and np.allclose(a.cell,b.cell,atol=1e-8,rtol=0)
                        and np.allclose(a.positions,b.positions,atol=1e-8,rtol=0)):
                    raise ValueError(f"Geometry identity differs: {edge}")
                truth[offsets[i]:offsets[i+1]]=a.get_forces(apply_constraint=False)
                energy.append(a.get_potential_energy())
            li=int(info[edge]["moving_Li_row"])-1
            starts,ends=labeled[0],labeled[-1]
            _, moved=find_mic(ends.positions-starts.positions,starts.cell,True)
            lithium=np.flatnonzero(starts.numbers==3)
            if li not in lithium or li!=lithium[np.argmax(moved[lithium])]:
                excluded.append(edge)
                li=None
            interior=sorted(range(1,len(ix)-1),key=lambda i:(-energy[i],i))
            cut=(len(interior)+3)//4
            details[edge] = (li, {"high_interior":[ix[i] for i in interior[:cut]],
                                  "other_interior":[ix[i] for i in interior[cut:]],
                                  "endpoints":[ix[0],ix[-1]]})
    if not np.isfinite(truth).all():
        raise ValueError("DFT force labels are incomplete")
    return truth,offsets,details,info,excluded


def run(args):
    _, files=load_bundle(args.bundle)
    truth,offsets,paths,metadata,excluded=references(args,files)
    jobs={r["seed"]:r["jobs"] for r in json.loads(files["jobs.json"])}
    data={}
    with ZipFile(args.predictions) as archive:
        manifest=json.loads(archive.read("manifest.json"))
        if manifest["bundle"]!=file_sha(args.bundle):
            raise ValueError("Blind force bundle differs")
        for name,digest in manifest["models"].items():
            phase, arm, stem=name.split("/")
            seed=int(stem[4:-4])
            final=len(jobs[seed][arm])
            model=400 if phase=="early" else final
            doc=json.loads((args.runs/f"seed{seed}"/arm/f"done_{final}.json").read_bytes())
            if digest!=doc["files"][f"model_{model}.pt"]:
                raise ValueError("Checkpoint mismatch")
            pred=np.load(io.BytesIO(archive.read(name)),allow_pickle=False)
            if pred.shape!=truth.shape or not np.isfinite(pred).all():
                raise ValueError("Force prediction geometry differs")
            se=np.square(pred-truth)
            for edge,(li,regions) in paths.items():
                for region,idx in regions.items():
                    chunks=[se[offsets[i]:offsets[i+1]] for i in idx]
                    data[phase,arm,seed,edge,region,"all"]=np.sqrt(np.mean(chunks))
                    if li is not None:
                        data[phase,arm,seed,edge,region,"Li"]=np.sqrt(np.mean([c[li] for c in chunks]))
    table=[]
    summary={}
    for phase in ("early","final"):
        subsection=summary[phase]={}
        for a,b in (("path3","replay3"),("path1","replay1"),("path3","pure"),("path1","pure")):
            block=subsection[f"{a}_minus_{b}"]={}
            for region in ("high_interior","other_interior","endpoints"):
                for target in ("all","Li"):
                    edges=[edge for edge in sorted(paths) if target=="all" or edge not in excluded]
                    families=sorted({metadata[e]["chemsys"] for e in edges})
                    indices=[families.index(metadata[e]["chemsys"]) for e in edges]
                    w=np.random.default_rng(20261125).multinomial(
                        len(families),np.full(len(families),1/len(families)),size=10000)[:,indices]
                    delta=np.array([[data[phase,a,seed,e,region,target]-data[phase,b,seed,e,region,target]
                                     for e in edges] for seed in (17,29,43)])
                    val=delta.mean(axis=0)
                    ci=np.quantile((w@val)/w.sum(axis=1),[.025,.975])
                    block[f"{region}_{target}"]=dict(materials=len(edges),mean_eV_A=float(val.mean()),
                           interval95_eV_A=ci.tolist(),seed_means_eV_A=delta.mean(axis=1).tolist())
                    table.extend(dict(phase=phase,path=a,baseline=b,region=region,target=target,
                                      material=metadata[e]["material_id"],chemsys=metadata[e]["chemsys"],
                                      paired_delta_eV_A=float(d)) for e,d in zip(edges,val))
    args.output.mkdir(exist_ok=True)
    (args.output/"paired_forces.csv").write_bytes(csv_bytes(table))
    atomic_json(args.output/"summary.json",dict(
        scope="Observed P0 positive and negative cases, exploratory. Lower force RMSE is better.",
        high_region="Highest DFT relative-energy quartile of interior images, rounded up.",
        mobile_Li="Both pre-DFT Init row and largest end-to-end MIC displacement among relaxed Li agree.",
        omitted_Li_edges=excluded, predictions_sha256=file_sha(args.predictions),results=summary,
    ))
    for phase in summary:
        for key in ("path1_minus_replay1","path3_minus_replay3"):
            print(phase,key,json.dumps({k:v for k,v in summary[phase][key].items()
                                        if k.startswith("high_")}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("project","bundle","runs","predictions","output"):
        p.add_argument(name,type=Path)
    run(p.parse_args())
