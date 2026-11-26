"""Reconcile frozen B2 utility and audit complete-path associations without fitting."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import rankdata

from phase_a0_geometry import export_csv
from phase_b1_inference import file_sha, read_json, rows, write_json
from verify_phase_b1 import verify as verify_b1

SEEDS = (17,29,43)
RECIPE = "B2 protected D0, MACE-MP-0 Small, Adam lr=1e-4, E=1 F=100, FP32, clip=10"
PANEL = "project_holdout107 (107 materials, 56 exact evaluation chemical systems)"
SUMMARY = ("mean","maximum","p90","reaction_integral","peak_position")
NUISANCE = ("arc_length_A","init_images","initial_hard_cn_3A","init_atoms","added_atom_presentations",
            "optimizer_steps","init_Li_min_contact_A")
UTILITY = ("Delta_abs","Delta_net")
SHORT = {"soap_geometry":"SOAP geometry", "soap_elements":"SOAP elements",
         "coordination_hard":"Hard coordination", "coordination_smooth":"Smooth coordination",
         "rdf_geometry":"RDF geometry", "rdf_elements":"RDF elements", "mace_invariants":"MACE Activation",
         "Li_force_sd":"Li force SD (eV/Å)", "Li_tangent_sd":"Li tangent SD (eV/Å)",
         "Li_normal_sd":"Li normal SD (eV/Å)", "energy_centered_sd":"Centered energy SD (meV)",
         "energy_raw_sd":"Raw energy SD (meV)", "energy_from_start_sd":"Start-anchored energy SD (meV)",
         "forward_barrier_sd_meV":"Barrier SD (meV)"}


def check_close(observed, expected):
    np.testing.assert_allclose(observed,expected,rtol=0,atol=1e-9)


def lock_utility(repo, out, candidates, protocol):
    base = repo/"results/b2_overnight_01/six_hour_results"
    proof, execution = read_json(base/"verification.json"), read_json(base/"status.json")
    if proof["status"]!="pass" or execution["state"]!="complete" or proof["complete_fits"]!=88:
        raise ValueError("Incomplete or unverified B2")
    raw = rows(base/"material_gains.csv")
    source = rows(base/"intervals.csv")
    paired = [r for r in raw if r["analysis"]=="regular"]
    metrics = {(r["candidate_edge"],r["metric"]):r for r in source if r["analysis"]=="regular"}
    group = defaultdict(dict)
    chemistry, materials = {}, {}
    for r in paired:
        edge, seed, material = r["candidate_edge"], int(r["seed"]), r["material_id"]
        e = [abs(float(r[f"{arm}_prediction_meV"])-float(r["dft_forward_barrier_meV"]))
             for arm in ("pure","path","replay")]
        check_close([float(r[f"{arm}_error_meV"]) for arm in ("pure","path","replay")],e)
        gain = np.array([e[0]-e[1],e[2]-e[1]])
        check_close(gain,[float(r["absolute_gain_meV"]),float(r["net_gain_meV"])])
        if (seed,material) in group[edge]:
            raise ValueError("Duplicate B2 path/seed/material")
        group[edge][seed,material] = gain
        if material in chemistry and (chemistry[material],materials[material])!=(r["chemsys"],r["evaluation_edge"]):
            raise ValueError("Inconsistent evaluation material mapping")
        chemistry[material] = r["chemsys"]
        materials[material] = r["evaluation_edge"]
    order = sorted(materials)
    groups = sorted(set(chemistry.values()))
    n = len(groups)
    if not (len(paired)==14*3*107 and len(order)==107 and n==56 and len(metrics)==28):
        raise ValueError("B2 cohort size changed")
    counts = np.random.default_rng(protocol["bootstrap"]["rng_seed"]).multinomial(
        n,np.full(n,1/n),size=protocol["bootstrap"]["replicates"])
    w = counts[:,[groups.index(chemistry[m]) for m in order]]
    w = w/w.sum(axis=1,keepdims=True)
    margin = protocol["effect_margin_meV"]
    labels, subgroup = [], []
    for c in candidates:
        edge = c["edge_id"]
        g = group[edge]
        if set(g)!={(s,m) for s in SEEDS for m in order}:
            raise ValueError(f"Incomplete common panel for {edge}")
        value = np.array([[g[s,m] for m in order] for s in SEEDS])
        label = dict(edge_id=edge,slot=c["slot"],chemsys=c["chemsys"],cohort="B2 regular",
                     seeds="17;29;43",materials=107,evaluation_chemistry_groups=n,margin_meV=margin,
                     evaluation_panel=PANEL)
        for i,(name,source_metric) in enumerate(zip(UTILITY,("absolute","net"))):
            r = metrics[edge,source_metric]
            seed_value = value[:,:,i].mean(axis=1)
            v = value[:,:,i].mean(axis=0)
            low, high = np.quantile(w@v,[.025,.975])
            check_close([v.mean(),low,high,*seed_value],
                        [float(r[k]) for k in ("mean_meV","low95_meV","high95_meV")]
                        +[float(s) for s in r["seed_gains_meV"].split(";")])
            state = ("positive" if low>margin and seed_value.min()>0 else
                     "negative" if high < -margin and seed_value.max()<0 else
                     "near_zero" if low>=-margin and high<=margin else "unresolved")
            if state!=r["classification"] or r["seeds"]!="17;29;43":
                raise ValueError("B2 classification changed")
            label.update({name:float(r["mean_meV"]),f"{name}_low95":float(r["low95_meV"]),
                          f"{name}_high95":float(r["high95_meV"]),f"{name}_class":state,
                          **{f"{name}_seed{s}_meV":float(u) for s,u in zip(SEEDS,seed_value)}})
            for family in sorted({panel_family(c) for c in chemistry.values()}):
                mask = [panel_family(chemistry[m])==family for m in order]
                subgroup.append(dict(edge_id=edge,utility=name,panel_group=family,materials=sum(mask),
                                     mean_meV=float(v[mask].mean())))
            largest = int(np.argmax(abs(v)))
            label[f"{name}_largest_material"] = order[largest]
            label[f"{name}_largest_material_gain"] = float(v[largest])
            label[f"{name}_without_largest"] = float(np.delete(v,largest).mean())
        labels.append(label)
    export_csv(out/"path_utility_labels.csv",labels)
    export_csv(out/"paired_material_gains.csv",paired)
    export_csv(out/"supplemental_utility_labels.csv",[r for r in source if r["analysis"]!="regular"])
    export_csv(out/"panel_group_effects.csv",subgroup)
    export_csv(out/"evaluation_panel_composition.csv",[
        dict(chemsys=g,panel_group=panel_family(g),materials=sum(chemistry[m]==g for m in order)) for g in groups])
    return labels, base, proof


def panel_family(chemsys):
    elements = set(chemsys.split("-"))
    if "O" not in elements:
        return "non-oxide"
    if "F" in elements:
        return "F-containing oxide"
    if "Si" in elements:
        return "Si-containing oxide"
    if "P" in elements:
        return "P-containing oxide without Si"
    return "other oxide"


def aggregate(values, s):
    values, s = np.asarray(values,dtype=float), np.asarray(s,dtype=float)
    if not np.isfinite(values).all() or np.any(np.diff(s)<=0) or s[0]!=0 or s[-1]!=1:
        raise ValueError("Incomplete or invalid image profile")
    flat = np.ptp(values)<=1e-12
    return dict(zip(SUMMARY, (float(values.mean()),float(values.max()),float(np.quantile(values,.90,method="linear")),
                            float(np.trapezoid(values,s)),
                            None if flat else float(s[np.argmax(np.isclose(values,values.max(),rtol=0,atol=1e-12))]))))


def build_matrix(repo,out,candidates,protocol):
    sources = {r["edge_id"]:r for r in rows(out/"path_id_manifest.csv")}
    coord = {r["edge_id"]:float(r["initial_hard_cn_3A"]) for r in rows(out/"image_coordination_counts.csv")
             if r["image"]=="1"}
    static, dynamic = defaultdict(list), defaultdict(list)
    for r in rows(out/"image_static_profiles.csv"):
        if r["stage"]!="init":
            raise ValueError("Unexpected post-DFT descriptor geometry")
        static[r["edge_id"],r["representation"]].append(r)
    b1 = repo/"results/phase_b1"
    b1_summary = {r["edge_id"]:r for r in rows(b1/"path_summary.csv")}
    for r in rows(b1/"image_disagreement.csv"):
        if r["stage"]!="init":
            raise ValueError("Unexpected B1 geometry")
        dynamic[r["edge_id"]].append(r)
    verified = verify_b1(b1)
    matrix, coverage, identity, protocols = [], [], [], []
    ensembles = protocol["ensemble_image_fields"]
    methods = protocol["static_methods"]
    for c in candidates:
        edge = c["edge_id"]
        source = sources[edge]
        count, atoms = int(c["images"]),int(c["atoms"])
        anchor = edge in b1_summary
        data = dict(edge_id=edge,material_id=c["material_id"],chemsys=c["chemsys"],anion_family=c["anion_family"],
                    cohort=c["cohort"],arc_length_A=float(source["arc_length_A"]),init_images=count,
                    initial_hard_cn_3A=coord[edge],init_atoms=atoms,optimizer_steps=int(c["optimizer_steps"]),
                    added_atom_presentations=atoms*count*int(c["path_presentations_per_image"]),
                    init_Li_min_contact_A=float(c["init_Li_min_contact_A"]))
        keys = list(range(1,count+1))
        for method in methods:
            stack = sorted(static[edge,method],key=lambda r:int(r["image"]))
            if [int(r["image"]) for r in stack]!=keys:
                raise ValueError(f"Missing static geometry {edge}/{method}")
            s = [float(r["reaction_coordinate"]) for r in stack]
            for k,v in aggregate([r["dedup_nearest"] for r in stack],s).items():
                data[f"{method}__{k}"] = v
            for ref in protocol["static_reference_sensitivity"]:
                data[f"{method}__{ref}_mean"] = float(np.mean([float(r[ref]) for r in stack]))
            coverage.append(dict(edge_id=edge,feature=method,initial_status=("A" if anchor else
                "C" if method=="mace_invariants" else "B"),final_status="A",images_available=count,
                images_required=count,profile_available=True,
                source=("A1, Init" if anchor else "B3-A CPU, frozen A1 D0 Activation" if method=="mace_invariants"
                        else "B3-A from archived Init geometry, frozen A1 settings")))
        block = sorted(dynamic.get(edge,[]),key=lambda r:int(r["image"]))
        if anchor:
            rkeys = [int(r["image"]) for r in block]
            if rkeys!=keys or int(b1_summary[edge]["n_images"])!=count:
                raise ValueError("B1 image count changed")
            check_close([float(r["reaction_coordinate"]) for r in block],s)
        for feature in ensembles:
            values = aggregate([r[feature] for r in block],s) if anchor else dict.fromkeys(SUMMARY)
            data.update({f"{feature}__{k}":v for k,v in values.items()})
            coverage.append(dict(edge_id=edge,feature=feature,initial_status="A" if anchor else "C",
              final_status="A" if anchor else "C",images_available=count if anchor else 0,images_required=count,
              profile_available=anchor,source="B1 cached image profiles" if anchor else "requires 3-member CPU inference"))
        data[protocol["ensemble_path_field"]] = float(b1_summary[edge]["forward_barrier_sd_meV"]) if anchor else None
        coverage.append(dict(edge_id=edge,feature=protocol["ensemble_path_field"],initial_status="B" if anchor else "C",
              final_status="A" if anchor else "C",images_available=count if anchor else 0,images_required=count,
              profile_available=False,source="B1 path statistic from cached three seed barriers" if anchor
              else "requires 3-member CPU inference on complete path"))
        matrix.append(data)
        identity.append(dict(source,static_features_complete=True,ensemble_features_complete=anchor,utility_label=True,
                     B2_cohort=c["cohort"],training_recipe=RECIPE,seeds="17;29;43",evaluation_panel=PANEL))
        protocols.append(dict(edge_id=edge,source="B2 regular",included=True,steps=c["optimizer_steps"],
                         seed_set="17;29;43",dose_images=c["images"],presentations_per_image=5,
                         path_replay_hardware="RTX 2080 Ti",pure_training_hardware="historical RTX 4090",
                         evaluation_panel=PANEL,reason="Same B2 protocol. D0 steps 710; absolute reference crosses hardware."))
        if anchor:
            protocols.append(dict(edge_id=edge,source="pre-B2 historical",included=False,steps=(
                710 if c["material_id"]=="mp-1177833" else c["optimizer_steps"]),
                seed_set="historical; see source manifest",dose_images=c["images"],presentations_per_image="",
                path_replay_hardware="historical, use provenance",pure_training_hardware="historical",
                evaluation_panel="historical panel identity varies",reason=c["prior_comparability"]))
    protocols.append(dict(edge_id=candidates[0]["edge_id"],source="B2 seed 53/67 and five-seed replication",
                          included=False,steps=735,seed_set="53;67 or 17;29;43;53;67",dose_images=5,
                          presentations_per_image=5,path_replay_hardware="RTX 2080 Ti",
                          pure_training_hardware="historical RTX 4090",evaluation_panel=PANEL,
                          reason="Same path; keep supplementary seed replication separate"))
    for name, content in (("path_feature_matrix.csv",matrix),("asset_coverage.csv",coverage),
                           ("path_identity.csv",identity),("protocol_compatibility.csv",protocols)):
        export_csv(out/name,content)
    return matrix,coverage,verified


def audit_a2(repo,out,matrix):
    directory = repo/"results/phase_a2"
    p = directory/"parameter_sensitivity.csv"
    initial = [r for r in rows(p) if r["stage"]=="init" and r["reference"]=="raw"]
    blocks = defaultdict(dict)
    for r in initial:
        series = blocks[r["edge_id"],r["representation"],r["variant"]]
        i = int(r["image"])
        if i in series:
            raise ValueError("Repeated A2 parameter result")
        series[i] = float(r["novelty"])
    size = {r["edge_id"]:r["init_images"] for r in matrix if r["cohort"]=="historical_reference"}
    assets = []
    for (edge,method,variant),series in sorted(blocks.items()):
        base = blocks[edge,method,"baseline"]
        images = list(range(1,size[edge]+1))
        if set(base)!=set(series) or set(series)!=set(images):
            raise ValueError("A2 variant is missing a matched Init image")
        v,b = np.array([series[i] for i in images]),np.array([base[i] for i in images])
        if not np.isfinite(v).all():
            raise ValueError("A2 variant has nonfinite distances")
        assets.append(dict(edge_id=edge,representation=method,variant=variant,reference="raw",images=len(v),
                           mean_shift_from_baseline=float((v-b).mean()),max_abs_shift=float(np.max(np.abs(v-b))),
                           mean_A2_distance=float(v.mean())))
    export_csv(out/"a2_parameter_audit.csv",assets)


def correlation(records,x,y):
    eligible = [r for r in records if r.get(x) is not None and r.get(y) is not None]
    p,q = np.asarray([r[x] for r in eligible]), np.asarray([r[y] for r in eligible])
    n = len(p)
    rho = None if n<3 or np.ptp(p)<=1e-12 or np.ptp(q)<=1e-12 else float(
        np.corrcoef(rankdata(p),rankdata(q))[0,1])
    return n,rho


def analyze(matrix, labels, protocol, out):
    utilities = {r["edge_id"]:{k:r[k] for k in UTILITY} for r in labels}
    records = [dict(r,**utilities[r["edge_id"]]) for r in matrix]
    static = protocol["static_methods"]
    primary = [f"{k}__mean" for k in static+protocol["ensemble_image_fields"][:4]] + [protocol["ensemble_path_field"]]
    image_features = static + protocol["ensemble_image_fields"]
    targets = [f"{k}__{s}" for k in image_features for s in SUMMARY] + [protocol["ensemble_path_field"]]
    variant = [f"{k}__{r}_mean" for k in static for r in protocol["static_reference_sensitivity"]]
    scopes = {"all":records, "O_only":[r for r in records if r["anion_family"]=="O"],
              "five_images":[r for r in records if r["init_images"]==5],
              "seven_images":[r for r in records if r["init_images"]==7],
              "without_anchors":[r for r in records if r["cohort"]!="historical_reference"]}
    associations, omit, matrix_rows = [], [], []

    def add(category,scope,data,x,y):
        n,rho = correlation(data,x,y)
        entry = dict(category=category,scope=scope,x=x,y=y,n_paths=n,spearman_rho=rho,
                     descriptive_only=n<=3 or scope!="all")
        associations.append(entry)
        return entry

    for x in targets+variant:
        for y in UTILITY:
            full = add("feature_utility","all",records,x,y)
            for r in records:
                n,rho = correlation([q for q in records if q["edge_id"]!=r["edge_id"]],x,y)
                omit.append(dict(feature=x,utility=y,omitted_path=r["edge_id"],n_paths=n,
                                 spearman_rho=rho,full_n_paths=full["n_paths"],full_rho=full["spearman_rho"]))
            if x in primary:
                for name,subset in scopes.items():
                    if name!="all":
                        add("feature_utility",name,subset,x,y)
    for x,y in combinations(primary,2):
        add("feature_pair","all",records,x,y)
    for k in NUISANCE:
        for y in UTILITY:
            add("confounder_utility","all",records,k,y)
        for x in primary:
            add("feature_confounder","all",records,x,k)
    for x in primary:
        for y in primary:
            n,rho = correlation(records,x,y)
            matrix_rows.append(dict(x=x,y=y,n_paths=n,spearman_rho=rho))
    export_csv(out/"feature_correlation.csv",associations)
    export_csv(out/"leave_one_path_out.csv",omit)
    export_csv(out/"feature_pair_matrix.csv",matrix_rows)
    ranked = []
    for y in UTILITY:
        for view,selection in (("primary",primary),("secondary",[k for k in targets+variant if k not in primary])):
            matches = [r for r in associations if r["category"]=="feature_utility" and r["scope"]=="all"
                       and r["y"]==y and r["x"] in selection]
            matches.sort(key=lambda r:(-r["n_paths"], -abs(r["spearman_rho"]) if r["spearman_rho"] is not None
                                       else float("inf"),r["x"]))
            for rank,r in enumerate(matches,1):
                drop = [q["spearman_rho"] for q in omit if q["feature"]==r["x"] and q["utility"]==y
                        and q["n_paths"]<r["n_paths"] and q["spearman_rho"] is not None]
                same = bool(drop) and all(r["spearman_rho"]*v>0 for v in drop)
                stable = view=="primary" and r["n_paths"]>=10 and len(drop)==r["n_paths"] and same
                ranked.append(dict(view=view,utility=y,rank=rank,feature=r["x"],n_paths=r["n_paths"],
                    spearman_rho=r["spearman_rho"],leave_one_min=min(drop) if drop else None,
                    leave_one_max=max(drop) if drop else None,direction_retained=same,
                    exploratory_lead=bool(stable and abs(r["spearman_rho"])>=.5)))
    export_csv(out/"spearman_ranking.csv",ranked)
    export_csv(out/"utility_class_contrast.csv",[dict(edge_id=r["edge_id"],chemsys=r["chemsys"],
         **{k:r[k] for k in (*UTILITY,"Delta_abs_class","Delta_net_class")},
         **{k:next(p[k] for p in matrix if p["edge_id"]==r["edge_id"]) for k in primary}) for r in labels])
    return records,primary,matrix_rows


def figures(out,records,labels,primary,matrix):
    figures = out/"figures"
    figures.mkdir()
    conf = {r["edge_id"]:r for r in labels}
    statics, ensembles = primary[:7],primary[7:]
    for name, chosen in (("static_utility",statics),("ensemble_utility",ensembles)):
        fig,axes = plt.subplots(len(chosen),2,figsize=(10,2.4*len(chosen)),squeeze=False,layout="constrained")
        for ax_row,x in zip(axes,chosen):
            for ax,y in zip(ax_row,UTILITY):
                data = [r for r in records if r[x] is not None]
                for r in data:
                    v = r[y]
                    low,high = (conf[r["edge_id"]][f"{y}_{k}95"] for k in ("low","high"))
                    c = "#D55E00" if r["cohort"]=="historical_reference" else "#0072B2"
                    ax.vlines(r[x],low,high,color=c,alpha=.35,linewidth=.7)
                    ax.scatter(r[x],v,c=c,s=21)
                    if len(data)<=3:
                        right = r[x]==max(d[x] for d in data)
                        ax.annotate(r["material_id"].replace("mp-",""),(r[x],v),xytext=(-3 if right else 3,4),
                                    textcoords="offset points",fontsize=7,ha="right" if right else "left")
                n,rho = correlation(records,x,y)
                ax.axhline(0,color="#777",linewidth=.6)
                ax.set(xlabel=SHORT[x.split("__")[0]],ylabel=f"{y} (meV)",
                       title=f"n={n}, Spearman ρ={rho:.2f}" if rho is not None else f"n={n}, ρ undefined")
                ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.grid(alpha=.14)
        fig.savefig(figures/f"{name}.png",dpi=145)
        plt.close(fig)
    n = len(primary)
    fig,ax = plt.subplots(figsize=(11,9),layout="constrained")
    vals = np.array([r["spearman_rho"] if r["spearman_rho"] is not None and r["n_paths"]>=4 else np.nan
                     for r in matrix]).reshape(n,n)
    image = ax.imshow(vals,vmin=-1,vmax=1,cmap=plt.get_cmap("RdBu_r").with_extremes(bad="#e6e6e6"))
    captions = [SHORT[k.split("__")[0]] for k in primary]
    ax.set_xticks(range(n),captions,rotation=45,ha="right")
    ax.set_yticks(range(n),captions)
    for i,r in enumerate(matrix):
        ax.text(i%n,i//n,str(r["n_paths"]),ha="center",va="center",fontsize=7,
                color="white" if abs(vals.flat[i])>.68 else "black")
    fig.colorbar(image,ax=ax,label="Pairwise Spearman ρ")
    ax.set_title("Feature ranks (cell text = path count, gray = fewer than four paths)")
    fig.savefig(figures/"feature_correlation_matrix.png",dpi=155)
    plt.close(fig)


def main(repo,out):
    protocol = read_json(out/"analysis_protocol.json")
    extraction = read_json(out/"feature_manifest.json")
    if extraction["source_git_commit"]!=protocol["source_commit"]:
        raise ValueError("Unexpected repo version")
    for name,digest in extraction["output_sha256"].items():
        if file_sha(out/name)!=digest:
            raise ValueError(f"Modified B3-A feature asset: {name}")
    b2 = repo/"results/b2_overnight_01"
    candidates = rows(b2/"six_hour_candidates.csv")
    labels,base,proof = lock_utility(repo,out,candidates,protocol)
    matrix,coverage,b1 = build_matrix(repo,out,candidates,protocol)
    audit_a2(repo,out,matrix)
    records,primary,pair = analyze(matrix,labels,protocol,out)
    figures(out,records,labels,primary,pair)
    statuses = Counter((r["feature"],r["final_status"]) for r in coverage)
    export_csv(out/"asset_counts.csv",[
        dict(feature=k,complete=statuses[k,"A"],available_from_cache_or_source=statuses[k,"B"],
             needs_inference=statuses[k,"C"],unrecoverable=statuses[k,"D"])
        for k in dict.fromkeys(r["feature"] for r in coverage)])
    tracked = ("intervals.csv","material_gains.csv","verification.json","status.json","training_diagnostics.csv")
    write_json(out/"analysis_manifest.json",dict(
        source_commit=protocol["source_commit"],scoring_script_sha256=file_sha(Path(__file__)),
        protocol_sha256=file_sha(out/"analysis_protocol.json"),
        feature_manifest_sha256=file_sha(out/"feature_manifest.json"),
        B2_input_sha256={n:file_sha(base/n) for n in tracked},B2_independent_verification=proof,
        B1_reverified=b1,regular_path_labels=len(labels),regular_material_rows=14*107*3,
        analysis_outputs={p.relative_to(out).as_posix():file_sha(p) for p in sorted(out.rglob("*"))
                          if p.is_file() and p.suffix in (".csv",".png")}))
    print(f"Verified {len(labels)} path labels, {len(coverage)} availability entries, {len(primary)} primary features")


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("repo",type=Path)
    p.add_argument("out",type=Path)
    main(**vars(p.parse_args()))
