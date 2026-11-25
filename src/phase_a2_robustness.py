"""Run the Phase A2 CPU parameter, reference and trajectory robustness checks."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import io
import json
import os
from pathlib import Path
import time
from zipfile import ZipFile

from ase import Atoms
import numpy as np
from scipy.spatial.distance import cdist

from phase_a0_geometry import PATHS, STAGES, csv_rows, exact_sha, export_csv, minimum_vectors
from phase_a01_coverage import load_inputs
from phase_a1_descriptors import D0_WEIGHT, D0_WEIGHT_SHA256, METHODS, SOAP_SETTINGS, load_representatives, taper

SOAP_VARIANTS = {
    "baseline": {}, "cutoff_3.5": {"r_cut":3.5}, "cutoff_4.5":{"r_cut":4.5},
    "sigma_0.30": {"sigma":.30}, "sigma_0.50":{"sigma":.50},
    "n_4":{"n_max":4}, "n_6":{"n_max":6}, "l_2":{"l_max":2}, "l_4":{"l_max":4},
    "cutoff_3.5_sigma_0.30":{"r_cut":3.5,"sigma":.30},
}
RDF_VARIANTS = {
    "baseline":(.15,.12), "grid_0.10":(.10,.12), "grid_0.20":(.20,.12),
    "sigma_0.08":(.15,.08), "sigma_0.18":(.15,.18),
}
CN_VARIANTS = {"baseline":3., "cutoff_2.8":2.8, "cutoff_3.2":3.2}
LAYER_VARIANTS = ("baseline","layer_1","layer_2")
N_DRAWS = 16
REFERENCE_SEED = 20261126
GRID = np.linspace(0.,1.,201)
CR = PATHS[2]


def li_indices(atoms):
    return np.flatnonzero(atoms.numbers==3)


def geometry(atoms):
    return Atoms(numbers=atoms.numbers, positions=atoms.positions, cell=atoms.cell, pbc=atoms.pbc)


def batch_distance(query, bank, method):
    metric = METHODS[method][1]
    if metric=="euclidean":
        return cdist(query,bank)
    q, b = np.linalg.norm(query,axis=1), np.linalg.norm(bank,axis=1)
    values = cdist(query,bank,metric="cosine")
    zero = q[:,None]*b[None,:] <= 1e-20
    both_zero = (q[:,None]<=1e-10) & (b[None,:]<=1e-10)
    return np.where(zero,np.where(both_zero,0.,1.),np.clip(values,0.,2.))


def soap_distances(job):
    tag, args, species, queries, reference = job
    from dscribe.descriptors import SOAP
    matrices = {}
    start = time.perf_counter()
    for method, chem in (("soap_geometry",False),("soap_elements",True)):
        if chem:
            soap = SOAP(species=species, compression={"mode":"mu1nu1","species_weighting":None},**args)
        else:
            soap = SOAP(species=["H"],**args)

        def descriptor(atoms, selected):
            if not chem:
                atoms = atoms.copy()
                atoms.numbers[:] = 1
            return np.atleast_2d(soap.create(atoms,centers=selected.tolist()))
        q = np.concatenate([descriptor(a,i) for a,i in queries])
        parts = [batch_distance(q,descriptor(a,i),method) for a,i in reference]
        matrices[method,tag] = np.concatenate(parts,axis=1)
    return matrices, time.perf_counter()-start


def radial_inputs(structures, species):
    result = []
    for atoms, selected in structures:
        vectors = minimum_vectors(atoms.positions[None]-atoms.positions[selected][:,None],atoms.cell,atoms.pbc)
        r = np.linalg.norm(vectors,axis=-1)
        mask = r > 1e-10
        symbols = np.asarray(atoms.get_chemical_symbols())
        bins = (symbols[:,None] == np.asarray(species)[None,:]).astype(float)
        result.append((r,mask,bins))
    return result


def radial_distances(queries, reference, species):
    queries = radial_inputs(queries,species)
    reference = radial_inputs(reference,species)
    matrices = {}
    for tag,center in CN_VARIANTS.items():
        def coordinates(data):
            r,mask,bins=data
            return ((r<=center)&mask)@bins, (taper(r,center-.4,center+.4)*mask)@bins
        q = list(zip(*(coordinates(data) for data in queries)))
        bank = list(zip(*(coordinates(data) for data in reference)))
        for k,m in enumerate(("coordination_hard","coordination_smooth")):
            matrices[m,tag] = batch_distance(np.concatenate(q[k]),np.concatenate(bank[k]),m)
    for tag,(width,sigma) in RDF_VARIANTS.items():
        grid = np.arange(1.,5.,width)
        def density(data):
            r,mask,bins=data
            g = np.exp(-.5*((r[:,:,None]-grid)/sigma)**2)
            g *= (taper(r,4.6,5.)*mask)[:,:,None]
            g /= (np.sqrt(2.*np.pi)*sigma)*(4.*np.pi*grid**2)
            element = np.einsum("cag,ae->ceg",g,bins,optimize=True)
            return g.sum(axis=1),element.reshape(r.shape[0],-1)
        q = [np.concatenate(v) for v in zip(*(density(x) for x in queries))]
        parts = [[],[]]
        for data in reference:
            for k,(m,feat) in enumerate(zip(("rdf_geometry","rdf_elements"),density(data))):
                parts[k].append(batch_distance(q[k],feat,m))
        for k,m in enumerate(("rdf_geometry","rdf_elements")):
            matrices[m,tag] = np.concatenate(parts[k],axis=1)
    return matrices


def mace_distances(queries, reference, checkpoint, threads):
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD","1")
    import torch
    from mace.calculators import MACECalculator

    torch.set_num_threads(threads)
    torch.set_default_dtype(torch.float64)
    with ZipFile(checkpoint) as archive:
        data = archive.read(D0_WEIGHT)
    if exact_sha(data)!=D0_WEIGHT_SHA256:
        raise ValueError("D0 MACE checkpoint differs from the A1 artifact")
    model = torch.load(io.BytesIO(data),map_location="cpu",weights_only=False)
    calc = MACECalculator(models=model,device="cpu",default_dtype="float64")
    model.eval()
    if int(model.num_interactions)!=2:
        raise ValueError("Expected the audited two-layer MACE checkpoint")
    def extract(structures):
        matrices = []
        for atoms,selected in structures:
            x = calc.get_descriptors(atoms=atoms,invariants_only=True,num_layers=2)[selected]
            if x.shape != (len(selected),256) or not np.isfinite(x).all():
                raise ValueError("Unexpected MACE layout")
            matrices.append(x)
        return np.concatenate(matrices)
    q,bank = extract(queries),extract(reference)
    return {("mace_invariants",tag):batch_distance(q[:,s],bank[:,s],"mace_invariants")
            for tag,s in zip(LAYER_VARIANTS,(slice(None),slice(0,128),slice(128,None)))}


def reference_banks(ids, reduced, metadata, key_manifest):
    material = np.asarray([row[0] for row in ids])
    chemistry = np.asarray([row[1] for row in ids])
    anion = np.asarray([row[2] for row in ids])
    choices = np.flatnonzero(reduced)
    groups = [choices[material[choices]==m] for m in sorted(set(material))]
    quota = min(8,min(map(len,groups)))
    if quota<=0:
        raise ValueError("A reference material has no representative")
    rng = np.random.default_rng(REFERENCE_SEED)
    balanced = np.array([np.concatenate([rng.choice(g,quota,replace=False) for g in groups])
                         for _ in range(N_DRAWS)])
    matched = np.array([rng.choice(choices,len(groups)*quota,replace=False) for _ in range(N_DRAWS)])
    raw = np.arange(len(ids))
    by_path = {}
    for edge in PATHS:
        rec = metadata[edge]
        masks = {"exclude_target_material":material!=rec["material_id"],
                 "exclude_exact_chemistry":chemistry!=rec["chemsys"],
                 "exclude_anion_family":anion!=rec["anion_family"]}
        banks = {"raw":raw[None,:],"deduplicated":choices[None,:],
                 "material_balanced":balanced,"size_matched_unstratified":matched}
        banks.update({k:raw[mask][None,:] for k,mask in masks.items()})
        if any(not x.shape[1] for x in banks.values()):
            raise ValueError(f"Empty reference policy for {edge}")
        by_path[edge] = banks
    evidence = dict(quota_per_material=quota,draws=N_DRAWS,seed=REFERENCE_SEED,
                    raw_environments=len(raw),deduplicated_environments=len(choices),materials=len(groups),
                    index_sha256=key_manifest)
    return by_path, material, evidence


def peak_profile(s, v):
    s,v = np.asarray(s,dtype=float),np.asarray(v,dtype=float)
    if len(s)!=len(v) or len(s)<3 or s[0]!=0. or s[-1]!=1. or np.any(np.diff(s)<=0):
        raise ValueError("Expected ordered, strictly increasing normalized coordinates")
    x = np.unique(np.r_[GRID,s])
    values = np.interp(x,s,v)
    excess = values-(values[0]+x*(values[-1]-values[0]))
    i = int(np.argmax(excess))
    h = max(0.,float(excess[i]))
    lo,hi = i,i
    if h>1e-12:
        while lo>0 and excess[lo-1]>=h/2.:
            lo -= 1
        while hi+1<len(x) and excess[hi+1]>=h/2.:
            hi += 1
        left = np.interp(h/2.,excess[[lo-1,lo]],x[[lo-1,lo]])
        right = np.interp(h/2.,excess[[hi+1,hi]],x[[hi+1,hi]])
        width = float(right-left)
    else:
        width = 0.
    j = int(np.argmax(v))
    fraction = h / max(float(v[j]),1e-12)
    # Report ties at original samples, without counting interpolated plateau points as independent images.
    original_excess = v-(v[0]+s*(v[-1]-v[0]))
    pos = float(x[i]) if h>1e-12 else None
    return dict(raw_peak_s=float(s[j]),raw_peak=float(v[j]),
                raw_peak_tied_images=int(np.count_nonzero(np.isclose(v,v[j],atol=1e-12,rtol=0))),
                excess_peak_s=pos,excess_peak=h,excess_fraction=fraction,
                excess_peak_tied_images=int(np.count_nonzero(np.isclose(original_excess,h,atol=1e-12,rtol=0))),
                excess_half_width=width,excess_positive_area=float(np.trapezoid(np.maximum(excess,0.),x)),
                central_feature=int(pos is not None and .20<=pos<=.80 and fraction>=.10))


def profiles(distance, keys, by_path, materials):
    parameter_rows, ref_rows, peaks = [], [], []
    curves = {}
    for (method, config), distances in sorted(distance.items()):
        if distances.shape != (len(keys),len(materials)) or not np.isfinite(distances).all():
            raise ValueError(f"Invalid distance matrix for {method}/{config}")
        for edge in PATHS:
            group = [i for i,k in enumerate(keys) if k["edge_id"]==edge]
            banks = by_path[edge] if config=="baseline" else {"raw":by_path[edge]["raw"]}
            for bank,choices in banks.items():
                sampled = np.stack([distances[np.ix_(group,x)].min(axis=1) for x in choices])
                quartiles = np.quantile(sampled,[.25,.50,.75],axis=0)
                for j,k in enumerate(group):
                    record = dict(keys[k], representation=method, variant=config, reference=bank,
                                  novelty=float(quartiles[1,j]), q25=float(quartiles[0,j]), q75=float(quartiles[2,j]),
                                  reference_sites=choices.shape[1],
                                  reference_materials=float(np.median([len(set(materials[x])) for x in choices])),
                                  draws=len(choices))
                    if bank=="raw":
                        parameter_rows.append(record)
                    if config=="baseline":
                        ref_rows.append(record)
                for stage in STAGES:
                    jj = [j for j,k in enumerate(group) if keys[k]["stage"]==stage]
                    s = [keys[group[j]]["reaction_coordinate"] for j in jj]
                    for label, values in [("summary",quartiles[1]), *(
                            (f"draw_{i+1:02d}",v) for i,v in enumerate(sampled) if len(sampled)>1)]:
                        v = values[jj]
                        key = (edge,method,config,bank,label,stage)
                        m = peak_profile(s,v)
                        curves[key] = (np.asarray(s),np.asarray(v),m)
                        peaks.append(dict(edge_id=edge,representation=method,variant=config,reference=bank,
                                          draw=label,stage=stage,**m))
    return parameter_rows,ref_rows,peaks,curves

def compare_stages(curves):
    rows = []
    for key,(s0,v0,m0) in curves.items():
        if key[-1]!="init":
            continue
        s1,v1,m1 = curves[(*key[:-1],"relaxed")]
        grid = np.unique(np.r_[GRID,s0,s1])
        diff = np.interp(grid,s1,v1)-np.interp(grid,s0,v0)
        positions = (m0["excess_peak_s"],m1["excess_peak_s"])
        e,m,c,r,draw,_ = key
        rows.append(dict(edge_id=e,representation=m,variant=c,reference=r,draw=draw,
                         init_excess=m0["excess_peak"],relaxed_excess=m1["excess_peak"],
                         init_feature=m0["central_feature"],relaxed_feature=m1["central_feature"],
                         relaxed_minus_init_excess=m1["excess_peak"]-m0["excess_peak"],
                         relative_excess_change=((m1["excess_peak"]-m0["excess_peak"])/m0["excess_peak"]
                                                 if m0["excess_peak"]>1e-12 else None),
                         peak_shift_s=(positions[1]-positions[0] if None not in positions else None),
                         mean_abs_profile_shift=float(np.trapezoid(np.abs(diff),grid)),
                         signed_profile_shift=float(np.trapezoid(diff,grid))))
    return rows


def compare_alignment(curves):
    rows = []
    for key,(s,v,stats) in curves.items():
        edge,method,tag,bank,draw,stage = key
        if method not in ("soap_geometry","soap_elements"):
            continue
        for layer in (LAYER_VARIANTS if tag=="baseline" and bank=="raw" else ("baseline",)):
            other = curves[edge,"mace_invariants",layer,bank,draw,stage][2]
            a,b = stats["excess_peak_s"],other["excess_peak_s"]
            gap = abs(a-b) if a is not None and b is not None else None
            rows.append(dict(edge_id=edge,stage=stage,SOAP=method,soap_variant=tag,mace_variant=layer,
                             reference=bank,draw=draw,soap_peak_s=a,mace_peak_s=b,gap_s=gap,
                             central_alignment=int(bool(stats["central_feature"] and other["central_feature"]
                                                        and gap is not None and gap<=.10))))
    return rows


def summarize_robustness(stage_rows, banks):
    rows = []
    for edge in PATHS:
        for method in METHODS:
            matching = [r for r in stage_rows if (r["edge_id"],r["representation"])==(edge,method)]
            for axis, subset in (
                ("parameters",[r for r in matching if r["reference"]=="raw"]),
                ("reference_policies",[r for r in matching if r["variant"]=="baseline" and r["draw"]=="summary"
                  and (not r["reference"].startswith("exclude_") or
                       banks[edge][r["reference"]].shape[1]<banks[edge]["raw"].shape[1])]),
                ("balanced_draws",[r for r in matching if r["reference"]=="material_balanced"
                                   and r["draw"]!="summary"]),
            ):
                changes = [r["relative_excess_change"] for r in subset if r["relative_excess_change"] is not None]
                rows.append(dict(edge_id=edge,representation=method,axis=axis,n=len(subset),
                                 init_central=sum(r["init_feature"] for r in subset),
                                 relaxed_central=sum(r["relaxed_feature"] for r in subset),
                                 attenuation_count=sum(r["relaxed_minus_init_excess"] < -1e-12 for r in subset),
                                 median_relative_excess_change=float(np.median(changes)) if changes else None))
    return rows


def baseline_agreement(a1, rows):
    prior = {(r["edge_id"],r["stage"],int(r["image"]),r["representation"]):r
             for r in csv_rows((a1/"novelty_profiles.csv").read_bytes())}
    differences = defaultdict(float)
    kept = set()
    for row in rows:
        field = {"raw":"raw_nearest","deduplicated":"dedup_nearest"}.get(row["reference"])
        if field:
            key = (row["edge_id"],row["stage"],row["image"],row["representation"])
            value = float(prior[key][field])
            differences[row["representation"]] = max(differences[row["representation"]],abs(value-row["novelty"]))
            kept.add((*key,field))
    if len(kept)!=2*len(prior) or max(differences.values())>1e-6:
        raise ValueError(f"A2 baseline differs from A1: {dict(differences)}")
    return dict(differences)


def plot_cr(output, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig,axes = plt.subplots(2,2,figsize=(11,7.5),sharex=True,sharey="row",layout="constrained")
    for li,method in enumerate(("soap_geometry","mace_invariants")):
        for col,stage in enumerate(STAGES):
            ax = axes[li,col]
            variants = SOAP_VARIANTS if li==0 else LAYER_VARIANTS
            for tag in variants:
                s,v,_ = curves[CR,method,tag,"raw","summary",stage]
                ax.plot(s,v,alpha=.8 if tag=="baseline" else .26, color="#0072B2",
                        linewidth=2.4 if tag=="baseline" else 1.0,
                        marker="o" if tag=="baseline" else None,
                        label="Raw baseline" if tag=="baseline" else None)
            for policy,color,label in (
                ("deduplicated","#009E73","Deduplicated"),("material_balanced","#D55E00","Equal-material (median)"),
                ("exclude_anion_family","#CC79A7","Exclude anion family")
            ):
                s,v,_ = curves[CR,method,"baseline",policy,"summary",stage]
                ax.plot(s,v,marker=".",ls="--",color=color,label=label)
            ax.set(title=f"{method.replace('_',' ')} · {stage}",ylabel="Cosine distance")
            ax.grid(alpha=.2)
    axes[0,1].legend(fontsize=8)
    fig.supxlabel("Normalized reaction coordinate")
    fig.suptitle("Cr–Ga–Li–O novelty robustness")
    fig.savefig(output/"cr_novelty_robustness.png",dpi=145)
    plt.close(fig)


def write_report(output, evidence, stages, alignment, robustness):
    cr_baseline = [r for r in stages if (r["edge_id"],r["variant"],r["reference"],r["draw"])==
                   (CR,"baseline","raw","summary") and
                   r["representation"] in ("soap_geometry","mace_invariants")]
    lines = ["# Phase A2 trajectory novelty robustness", "",
             "The scope is three case-study paths. All profiles reuse the A0/A1 geometries and one frozen",
             "seed-17 MACE checkpoint. Only Init is available for DFT-free use. No gain labels enter this analysis.", "",
             f"Recomputed A1 raw and deduplicated profiles with maximum absolute disagreement of "
             f"{max(evidence['a1_baseline_max_abs_difference'].values()):.3g}. Total CPU run time "
             f"was {evidence['elapsed_seconds']:.0f} seconds wall clock. MACE took "
             f"{evidence['component_wall_seconds']['mace']:.0f} seconds and the largest single SOAP "
             f"job took {max(v for k,v in evidence['component_wall_seconds'].items() if k.startswith('soap_')):.1f} seconds.", "",
             "## Cr–Ga–Li–O central peaks", "",
             "| Representation | Comparison | Init central | Relaxed central | Excess smaller after relaxation |",
             "| --- | --- | ---: | ---: | ---: |"]
    for r in robustness:
        if r["edge_id"]==CR and r["representation"] in ("soap_geometry","soap_elements","rdf_geometry","mace_invariants"):
            n = r["n"]
            lines.append(f"| {r['representation']} | {r['axis']} | {r['init_central']}/{n} "
                         f"| {r['relaxed_central']}/{n} | {r['attenuation_count']}/{n} |")
    lines += ["", "Excess is measured above the line between endpoint distances. On raw D₀ with the",
              "baseline parameters, peak excess declined by " + " and ".join(
                  f"{-100.*r['relative_excess_change']:.1f}% ({r['representation']})" for r in cr_baseline) + ".",
              "Parameter settings",
              "and reference draws are sensitivity cases. Independent material count remains three.", "",
              "## Reference exclusions", "",
              "| Path | Target sites removed | Exact-system sites removed | Anion-family sites removed |",
              "| --- | ---: | ---: | ---: |"]
    for edge, removed in evidence["reference_exclusions"].items():
        lines.append(f"| {edge} | {removed['exclude_target_material']} | {removed['exclude_exact_chemistry']} "
                     f"| {removed['exclude_anion_family']} |")
    lines += ["", "Zero means the exclusion reproduces the raw bank. The anion-family knockout changes",
              "the reference population. Equal-material thinning uses the same quota for every material,",
              "alongside a matched-size unstratified control. Thinning draw summaries are descriptive.", "",
              "## SOAP–MACE alignment", ""]
    for stage in STAGES:
        chosen = [r for r in alignment if (r["edge_id"],r["stage"],r["SOAP"],r["reference"],r["mace_variant"])==
                  (CR,stage,"soap_geometry","raw","baseline")]
        n = sum(r["central_alignment"] for r in chosen)
        lines.append(f"- The geometric SOAP grid on Cr–Ga–Li–O aligns with the combined MACE central peak "
                     f"at {n}/{len(chosen)} settings on {stage}. The tolerance is 0.10 in normalized coordinate.")
    lines += ["", "A center-position match does not identify the same atomistic mechanism. Five to seven",
              "native points limit peak-width resolution. Frequency across these descriptors cannot establish",
              "prediction of training value or generalization to independent paths."]
    (output/"a2_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def run(args):
    start=time.perf_counter()
    frames, metadata, queries = load_inputs(args.neb,args.index,args.bundle,args.a0)
    keys = [k for _,_,k in queries]
    query = [(geometry(a),np.array([i])) for a,i,_ in queries]
    reference = [(geometry(a),li_indices(a)) for a in frames]
    phase1 = json.loads((args.a1/"manifest.json").read_bytes())
    if phase1["checkpoint_sha256"]!=D0_WEIGHT_SHA256:
        raise ValueError("A1 checkpoint changed")
    species = phase1["species"]
    idx = {r["edge_id"]:r for r in csv_rows(args.index.read_bytes())}
    reduced = load_representatives(args.a01/"reference_material_distribution.csv",frames)
    ids, retain = [], []
    for frame,((_,centers),info) in enumerate(zip(reference,metadata),1):
        rec = idx[info["edge_id"]]
        if rec["material_id"] != info["material_id"]:
            raise ValueError("Reference ancestry mismatch")
        for li in centers:
            ids.append((rec["material_id"],rec["chemsys"],rec["anion_family"]))
            retain.append((frame,int(li+1)) in reduced)
    if len(ids)!=9517 or sum(retain)!=2469:
        raise ValueError("Reference counts changed")
    banks, materials, evidence = reference_banks(ids,np.asarray(retain),idx,exact_sha(args.index.read_bytes()))
    jobs = [(k,dict(SOAP_SETTINGS,**v),species,query,reference) for k,v in SOAP_VARIANTS.items()]
    matrices = {}
    timings = {}
    if args.soap_workers > 1:
        with ProcessPoolExecutor(max_workers=args.soap_workers) as pool:
            pending = {tag:pool.submit(soap_distances,job) for tag,job in zip(SOAP_VARIANTS,jobs)}
            t=time.perf_counter()
            matrices.update(radial_distances(query,reference,species))
            timings["radial_and_coordination"] = time.perf_counter()-t
            t=time.perf_counter()
            matrices.update(mace_distances(query,reference,args.checkpoints,args.threads))
            timings["mace"] = time.perf_counter()-t
            print("Radial, coordination and MACE complete",flush=True)
            for tag,work in pending.items():
                values,elapsed=work.result()
                matrices.update(values)
                timings["soap_"+tag]=elapsed
                print(f"SOAP {tag}: {elapsed:.1f}s",flush=True)
    else:
        for tag,job in zip(SOAP_VARIANTS,jobs):
            values,elapsed=soap_distances(job)
            matrices.update(values)
            timings["soap_"+tag]=elapsed
            print(f"SOAP {tag}: {elapsed:.1f}s",flush=True)
        t = time.perf_counter()
        matrices.update(radial_distances(query,reference,species))
        timings["radial_and_coordination"] = time.perf_counter()-t
        t = time.perf_counter()
        matrices.update(mace_distances(query,reference,args.checkpoints,args.threads))
        timings["mace"] = time.perf_counter()-t
    parameters, ref, peaks, curves = profiles(matrices,keys,banks,materials)
    agreement = baseline_agreement(args.a1,ref)
    stages = compare_stages(curves)
    alignment = compare_alignment(curves)
    robustness = summarize_robustness(stages,banks)
    args.output.mkdir(parents=True,exist_ok=False)
    for name,arr in (
        ("parameter_sensitivity",parameters),("reference_sensitivity",ref),("peak_metrics",peaks),
        ("stage_comparison",stages),("robustness_summary",robustness),("soap_mace_alignment",alignment)
    ):
        export_csv(args.output/(name+".csv"),arr)
    plot_cr(args.output,curves)
    evidence.update(dict(
        soap_parameters={k:dict(SOAP_SETTINGS,**v) for k,v in SOAP_VARIANTS.items()},
        rdf_parameters=RDF_VARIANTS,coordination_parameters=CN_VARIANTS,mace_layers=LAYER_VARIANTS,
        a1_baseline_max_abs_difference=agreement, elapsed_seconds=time.perf_counter()-start,
        component_wall_seconds=timings, soap_workers=args.soap_workers, torch_threads=args.threads,
        reference_exclusions={e:{k:len(ids)-v.shape[1] for k,v in bank.items() if k.startswith("exclude_")}
                              for e,bank in banks.items()},
        inputs_sha256={name:exact_sha(p.read_bytes()) for name,p in {
            "a0_manifest":args.a0/"geometry_manifest.csv","a01_representatives":args.a01/"reference_material_distribution.csv",
            "a1_manifest":args.a1/"manifest.json","a1_profiles":args.a1/"novelty_profiles.csv"
        }.items()},
        mace_checkpoint_sha256=D0_WEIGHT_SHA256,
    ))
    (args.output/"manifest.json").write_text(json.dumps(evidence,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    write_report(args.output,evidence,stages,alignment,robustness)
    print(f"A2 complete. A1 reproduced across {len(agreement)} methods.",flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ("neb","index","bundle","a0","a01","a1","checkpoints","output"):
        cli.add_argument(name,type=Path)
    cli.add_argument("--soap-workers",type=int,default=4)
    cli.add_argument("--threads",type=int,default=4)
    run(cli.parse_args())
