"""Archive B5-R reproducibility diagnostics and its two summary figures."""
import argparse
import csv
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_phase_b5_r import load_json, rows, write_csv


def plot(out):
    figures=out/"figures"
    figures.mkdir(exist_ok=True)
    stats=rows(out/"feature_comparison.csv")
    methods=["train_mean","SOAP","SOAP+Activation","SOAP_KRR","Activation_KRR","SOAP+Activation_KRR"]
    display=["Training mean","SOAP novelty","+ activation novelty","SOAP kernel","Activation kernel","Combined kernel"]
    fig, axes=plt.subplots(1,2,figsize=(10.8,4.4),sharey=True,layout="constrained")
    y=np.arange(len(methods))
    for ax, target in zip(axes,("Delta_abs","Delta_net")):
        for i,(fold,label,color) in enumerate((("LOPO","Leave one path out","#17638a"),
                                                ("leave_anion_family_out","Leave anion family out","#ce7334"))):
            measures={r["method"]:float(r["mae_meV"]) for r in stats if r["aggregation"]=="mean"
                      and r["fold_scheme"]==fold and r["target"]==target}
            ax.barh(y+(i-.5)*.33,[measures[k] for k in methods],height=.3,label=label,color=color)
        ax.set_yticks(y,display)
        ax.invert_yaxis() if ax is axes[0] else None
        ax.set_xlabel("Held-out prediction MAE (meV)")
        ax.set_title("Absolute improvement" if target=="Delta_abs" else "Net improvement")
        ax.grid(axis="x",alpha=.24)
    axes[0].legend(loc="lower right",fontsize=8)
    fig.savefig(figures/"historical_utility_mae.png",dpi=180,bbox_inches="tight",pad_inches=.18)
    plt.close(fig)

    rank=rows(out/"independent_candidate_ranking.csv")
    ordered=list(dict.fromkeys(x["method"] for x in rank if x["aggregation"]=="mean"))
    labels=[k.replace("Activation_","") for k in ordered]
    fig,axes=plt.subplots(1,3,figsize=(13.3,5.4),layout="constrained",sharey=True)
    for ax,k in zip(axes,(3,5,8)):
        top={m:{r["edge_id"] for r in rank if r["aggregation"]=="mean" and r["method"]==m and int(r["rank"])<=k}
             for m in ordered}
        counts=np.array([[len(top[a]&top[b]) for b in ordered] for a in ordered])
        im=ax.imshow(counts/k,vmin=0,vmax=1,cmap="Blues")
        for i,j in product(range(len(ordered)),repeat=2):
            ax.text(j,i,f"{counts[i,j]}/{k}",ha="center",va="center",
                    color="white" if counts[i,j]/k>.65 else "black",fontsize=8)
        ax.set_xticks(range(len(ordered)),labels,rotation=60,ha="right",fontsize=8)
        ax.set_yticks(range(len(ordered)),labels,fontsize=8)
        ax.set_title(f"Top {k} paths")
    fig.colorbar(im,ax=list(axes),label="Fraction in common",shrink=.75)
    fig.savefig(figures/"candidate_overlap_mean.png",dpi=180,bbox_inches="tight",pad_inches=.18)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("assets",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    work,out=args.assets,args.out
    pools=load_json(work/"geometry"/"candidate_manifest.json")
    pool_rows=[]
    for i,row in enumerate(pools["rows"],1):
        keep=("edge_id","material_id","chemsys","em","_split","材料家族","family","status",
              "atoms","images","steps","converged","parent_sha256","centroid_sha256","trajectory_sha256",
              "periodic_mapping_error_A","exception","error")
        result={k:row[k] for k in keep if k in row}
        if row.get("final"):
            result.update(min_Li_contact_A=min(r["min_Li_contact_A"] for r in row["final"]),
                          path_length_A=row["final"][-1]["arc_length_A"])
        pool_rows.append(dict(scan_rank=i,**result))
    write_csv(out/"candidate_selection_audit.csv",pool_rows)
    cpu=load_json(work/"bench_cpu"/"features.json")
    gpu=load_json(work/"bench_gpu_safe"/"features.json")
    if cpu["image_index"]!=gpu["image_index"] or cpu["geometry_sha256"]!=gpu["geometry_sha256"]:
        raise ValueError("Device benchmark inputs differ")
    parity={}
    with np.load(work/"bench_cpu"/"features.npz") as x, np.load(work/"bench_gpu_safe"/"features.npz") as y:
        for k in x.files:
            parity[k]=float(np.max(np.linalg.norm(x[k]-y[k],axis=1)/np.linalg.norm(x[k],axis=1)))
    if max(parity.values())>1e-6:
        raise ValueError("Device benchmark parity failed")
    benchmarks=[dict({"status":"ok"},**r) if "status" not in r else dict(r)
                for bench in (cpu,gpu) for r in bench["benchmark"]]
    benchmarks.append(dict(device="cpu",batch=1,images=248,status="full_extraction_ok",
                           seconds=load_json(work/"activation_cpu"/"features.json")["elapsed"],cpu_threads=4))
    for row in benchmarks:
        row.update(cloud=True,concurrent_QE=True)
    write_csv(out/"compute_benchmark.csv",benchmarks)
    manifest=load_json(out/"manifest.json")
    manifest.update(device_parity_relative_l2=parity,cloud_gpu_name=gpu.get("device_name"),
                    benchmark_concurrent_QE=True,activation_selected_device="cloud CPU, batch 1, 4 threads",
                    benchmark_gpu_allocation_fraction_cap=.075,
                    candidate_history_parents_checked=pools["historical_parents_with_metadata"],
                    candidate_history_parent_count=pools["historical_parent_total"],
                    public_panel_decision="unreleased",
                    additional_controls="Equal-lambda SOAP, Activation and equal-weight combined KRR comparator; no tuning.")
    manifest["protocol_sha256"]=sha256((out/"protocol.json").read_bytes()).hexdigest()
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    plot(out)
    assert len(rows(out/"independent_candidate_ranking.csv"))==24*7*3
    assert len(rows(out/"historical_utility_analysis.csv"))==14*13*2*2*3
    print("Verified device parity and the two complete prediction matrices")


if __name__=="__main__":
    main()
