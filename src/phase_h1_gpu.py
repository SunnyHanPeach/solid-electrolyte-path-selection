"""Read-only B5-R cache verification and H1 GPU kernel experiments."""
import argparse
from collections import defaultdict
import csv
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import spearmanr
import torch


DEVICE = "cuda"
METHODS = ("SOAP", "SOAP+Activation", "Activation")
TARGETS = ("Delta_abs", "Delta_net")


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def js(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def t(value):
    return torch.as_tensor(value, dtype=torch.float64, device=DEVICE)


def ix(matrix, a, b):
    return matrix[..., a, :][..., b]


def verify_labels(data, ids, frozen, labels):
    paired = read(data / "paired_material_gains.csv")
    checkpoints = {(r["arm"], int(r["seed"])):r for r in read(data / "training_diagnostics.csv")}
    expected_hashes = {(r["arm"], int(r["seed"])):r for r in read(data / "checkpoint_hashes.csv")}
    panels, gains = defaultdict(set), defaultdict(list)
    for row in paired:
        if row["analysis"] != "regular":
            continue
        edge, seed = row["candidate_edge"], int(row["seed"])
        mat = row["material_id"]
        target = float(row["dft_forward_barrier_meV"])
        pure, path, replay = (abs(float(row[k+"_prediction_meV"])-target) for k in ("pure","path","replay"))
        value = np.array([pure-path,replay-path])
        recorded = [float(row["absolute_gain_meV"]),float(row["net_gain_meV"])]
        if not np.allclose(value,recorded,atol=1e-9,rtol=0) or mat in panels[edge,seed]:
            raise ValueError("Material-level gain mismatch or duplicate")
        gains[edge,seed].append(value)
        panels[edge,seed].add(mat)
    if len(panels)!=42 or len({frozenset(p) for p in panels.values()})!=1 or len(next(iter(panels.values())))!=107:
        raise ValueError("Material/seed panel does not reproduce 14×3×107")
    audited, seed_labels = [], {}
    for edge in ids:
        row, stored = frozen[edge], labels[edge]
        n, atoms, steps, exposure = map(int, [row[k] for k in
            ("images","atoms","optimizer_steps","path_presentations_per_image")])
        if steps!=710+n*exposure or len(json.loads(row["selected_ids"]))!=n:
            raise ValueError(f"Unexpected schedule {edge}")
        by_seed = []
        for seed in (17,29,43):
            arm_rows=[]
            for arm in ("path"+row["slot"],"replay"+row["slot"]):
                r, expected = checkpoints[arm,seed], expected_hashes[arm,seed]
                if int(r["steps"])!=steps or r["checkpoint_sha256"]!=expected["checkpoint_sha256"]:
                    raise ValueError(f"Schedule or checkpoint mismatch {arm}/{seed}")
                arm_rows.append(expected)
            if arm_rows[0]["training_hardware"]!=arm_rows[1]["training_hardware"]:
                raise ValueError("Path vs replay hardware mismatch")
            mu = np.mean(gains[edge,seed],axis=0)
            by_seed.append(mu)
            vals = [float(stored[f"{k}_seed{seed}_meV"]) for k in TARGETS]
            if not np.allclose(mu,vals,atol=1e-8,rtol=0):
                raise ValueError(f"Frozen per-seed labels differ {edge}")
            audited.append(dict(edge_id=edge,seed=seed,materials=107,path_steps=steps,replay_steps=steps,
                                pure_steps=710,added_frames=n,presentations_per_image=exposure,
                                path_atom_presentations=n*atoms*exposure,
                                path_delta_abs_meV=mu[0],path_delta_net_meV=mu[1],
                                path_checkpoint_sha256=arm_rows[0]["checkpoint_sha256"],
                                replay_checkpoint_sha256=arm_rows[1]["checkpoint_sha256"],
                                path_replay_training_hardware=arm_rows[0]["training_hardware"],
                                pure_training_hardware=expected_hashes["pure",seed]["training_hardware"]))
        seed_labels[edge]=np.array(by_seed)
        if stored["seeds"]!="17;29;43" or not np.allclose(
            np.mean(by_seed,axis=0),[float(stored[k]) for k in TARGETS],atol=1e-8,rtol=0):
            raise ValueError(f"Frozen pooled labels differ {edge}")
    return audited, np.array([seed_labels[edge] for edge in ids])


def control_kernel(x, train, test):
    if x.shape[1]==0:
        return 0.0, 0.0
    mu = x[train].mean(dim=0)
    sigma = x[train].std(dim=0,correction=0)
    sigma = torch.where(sigma>1e-10,sigma,1)
    train_x,test_x=(x[train]-mu)/sigma,(x[test]-mu)/sigma
    return train_x @ train_x.T / x.shape[1], test_x @ train_x.T / x.shape[1]


def predict(kernel, controls, y, folds, ridge):
    if kernel.ndim==2:
        kernel=kernel.unsqueeze(0)
    result=torch.empty(kernel.shape[0],len(y),y.shape[1],device=DEVICE,dtype=torch.float64)
    for train,test in folds:
        controll, controlt=control_kernel(controls,train,test)
        k,cross=ix(kernel,train,train)+controll,ix(kernel,test,train)+controlt
        kmean=k.mean(dim=-2,keepdim=True)
        mean=kmean.mean(dim=-1,keepdim=True)
        kc=k-kmean-k.mean(dim=-1,keepdim=True)+mean
        cc=cross-kmean-cross.mean(dim=-1,keepdim=True)+mean
        ymean=y[train].mean(dim=0)
        m=kc+ridge*torch.eye(len(train),device=DEVICE,dtype=kc.dtype)
        result[:,test]=ymean+cc@torch.linalg.solve(m,(y[train]-ymean).unsqueeze(0).expand(len(k),-1,-1))
    return result


def matrices_and_folds(data, protocol, out):
    keys=read(data/"matrix_index.csv")
    positions=[i for i,r in enumerate(keys) if r["cohort"]=="historical"]
    ids=[keys[i]["edge_id"] for i in positions]
    labels={r["edge_id"]:r for r in read(data/"path_utility_labels.csv")}
    frozen={r["edge_id"]:r for r in read(data/"six_hour_candidates.csv")}
    if len(ids)!=14 or set(ids)!=set(labels) or set(ids)!=set(frozen):
        raise ValueError("Unexpected training path identity")
    if digest(data/"path_utility_labels.csv")!=js(data/"b5_manifest.json")["untouched_utility_label_sha256"]:
        raise ValueError("Historical labels changed")
    label_rows, seed_y=verify_labels(data,ids,frozen,labels)
    write(out/"label_exposure_verification.csv",label_rows)

    with np.load(data/"mean.npz") as archive:
        km={key:t(archive[key]) for key in archive.files}
    if any(float(torch.linalg.eigvalsh(k).min().item())< -1e-8 for k in km.values()):
        raise ValueError("Frozen B5-R Gram matrix is invalid")
    images=read(data/"image_index.csv")
    with np.load(data/"image_features.npz") as arrays:
        for method in ("activation_full","activation_li"):
            feat=t(arrays[method])
            feature_paths=[i for i,k in enumerate(keys) if k["cohort"]!="reference"]
            z=torch.stack([feat[[i for i,a in enumerate(images) if a["edge_id"]==keys[j]["edge_id"]]].mean(dim=0)
                           for j in feature_paths])
            z=torch.nn.functional.normalize(z,dim=1)
            torch.testing.assert_close(z@z.T,ix(km[method],feature_paths,feature_paths),atol=1e-10,rtol=0)
    family=np.array([frozen[i]["anion_family"] for i in ids])
    steps=np.array([int(frozen[i]["optimizer_steps"]) for i in ids])
    count=np.array([int(frozen[i]["images"]) for i in ids])
    atom_exposure=np.array([int(frozen[i]["atoms"])*int(frozen[i]["path_presentations_per_image"]) for i in ids])*count
    contact=np.array([float(frozen[i]["init_Li_min_contact_A"]) for i in ids])
    other=np.column_stack([contact,*[family==s for s in sorted(set(family))]])
    exposure=t(np.column_stack((other,steps,np.log(atom_exposure))))
    masks=[np.arange(len(ids))==i for i in range(len(ids))]
    group_masks=[family==s for s in sorted(set(family))]
    folds={k:[(np.flatnonzero(~m).tolist(),np.flatnonzero(m).tolist()) for m in v]
           for k,v in (("LOPO",masks),("anion-family-held-out",group_masks))}
    variants={
        "primary":("soap_full","activation_full",exposure),
        "local_features":("soap_li","activation_li",exposure),
        "no_exposure_controls":("soap_full","activation_full",t(other)),
        "no_controls":("soap_full","activation_full",t(np.empty((len(ids),0)))),
    }
    summary, predictions, permutations, intervals, seed_sensitivity=[],[],[],[],[]
    y=t(seed_y.mean(axis=1))
    config=protocol["sensitivity"]
    rng=np.random.default_rng(config["seed"])
    blocks=defaultdict(list)
    for i, pair in enumerate(zip(family,steps)):
        blocks[pair].append(i)
    shuffled=np.tile(np.arange(len(ids)),(config["permutations"],1))
    for row in shuffled:
        for group in blocks.values():
            row[group]=rng.permutation(group)
    permutation=torch.as_tensor(shuffled,device=DEVICE,dtype=torch.long)
    gates={}
    for design, fold in folds.items():
        for variant,(sm,am,control) in variants.items():
            s,a=ix(km[sm],positions,positions),ix(km[am],positions,positions)
            combined=.5*(s+a)
            est=predict(torch.stack((s,combined,a)),control,y,fold,protocol["primary"]["ridge_lambda"])
            error=est-y.unsqueeze(0)
            scores=est.cpu().numpy()
            true=y.cpu().numpy()
            for m,method in enumerate(METHODS):
                for j,target in enumerate(TARGETS):
                    rho=float(spearmanr(scores[m,:,j],true[:,j]).statistic)
                    summary.append(dict(folds=design,variant=variant,method=method,target=target,n=len(ids),
                                        MAE_meV=float(error[m,:,j].abs().mean()),
                                        RMSE_meV=float(error[m,:,j].square().mean().sqrt()),
                                        Spearman=rho if np.isfinite(rho) else None))
                    for i,edge in enumerate(ids):
                        predictions.append(dict(folds=design,variant=variant,method=method,target=target,
                                                fold_id=edge if design=="LOPO" else family[i],edge_id=edge,
                                                observed_meV=true[i,j],prediction_meV=scores[m,i,j],
                                                absolute_error_meV=abs(scores[m,i,j]-true[i,j])))
            if variant!="primary":
                continue
            null_kernel=.5*(s+a[permutation.unsqueeze(2),permutation.unsqueeze(1)])
            null_est=predict(null_kernel,control,y,fold,protocol["primary"]["ridge_lambda"])
            null_mae=(null_est-y).abs().mean(dim=1)
            gain_by_path=error[0].abs()-error[1].abs()
            observed=gain_by_path.mean(dim=0)
            null=error[0].abs().mean(dim=0)-null_mae
            blocks_boot=np.eye(len(ids)) if design=="LOPO" else np.array(group_masks).astype(float)
            samples=rng.integers(0,len(blocks_boot),size=(config["heldout_loss_bootstrap"],len(blocks_boot)))
            weight=t(blocks_boot[samples].sum(axis=1))
            boot=(weight@gain_by_path)/weight.sum(dim=1,keepdim=True)
            low,high=torch.quantile(boot,t([.025,.975]),dim=0).tolist()
            for j,target in enumerate(TARGETS):
                p=(1+int(torch.count_nonzero(null[:,j]>=observed[j]-1e-12)))/(len(null)+1)
                intervals.append(dict(folds=design,target=target,observed_SOAP_minus_combined_MAE_meV=float(observed[j]),
                                      low95_meV=low[j],high95_meV=high[j],blocked_p_one_sided=p,
                                      permutation_draws=len(null),unique_permutations=int(len(np.unique(shuffled,axis=0))),
                                      resampling_clusters=len(blocks_boot)))
                rho=[spearmanr(scores[m,:,j],true[:,j]).statistic for m in (0,1)]
                if target=="Delta_net":
                    gates[design]=bool(observed[j]>0 and low[j]>0 and rho[1]>=rho[0] and p<=.05)
                for i,draw in enumerate(null[:,j].tolist(),1):
                    permutations.append(dict(folds=design,target=target,draw=i,SOAP_minus_permuted_combined_MAE_meV=draw))
            for i,seed in enumerate(config["seed_specific"]):
                ys=t(seed_y[:,i,:])
                fits=predict(torch.stack((s,combined,a)),control,ys,fold,protocol["primary"]["ridge_lambda"])
                for m,method in enumerate(METHODS):
                    for j,target in enumerate(TARGETS):
                        seed_sensitivity.append(dict(folds=design,seed=seed,target=target,method=method,
                            MAE_meV=float((fits[m,:,j]-ys[:,j]).abs().mean())))
    for name, table in (("fold_predictions",predictions),("utility_metrics",summary),("blocked_permutations",permutations),
                        ("paired_uncertainty",intervals),("seed_sensitivity",seed_sensitivity)):
        write(out/f"{name}.csv",table)
    return keys,km,gates


def gpu_score_rankings(data, out, keys, km, plan):
    frozen=read(data/"independent_candidate_ranking.csv")
    refs=[i for i,k in enumerate(keys) if k["cohort"]=="reference"]
    candidates=[i for i,k in enumerate(keys) if k["cohort"]=="candidate"]
    activation,soap=km["activation_full"],km["soap_full"]
    modes=(("SOAP",soap,"kcenter"),("SOAP+Activation",.5*(soap+activation),"kcenter"),
           ("Activation_LCMD",activation,"lcmd"),("Activation_PV",activation,"pv"))
    ordering, validation=[],[]
    for name,kernel,rule in modes:
        pool=list(candidates)
        selected=[]
        sqdist=torch.clamp(2-2*kernel,min=0.0)
        while pool:
            support=refs+selected
            if rule=="pv":
                z=ix(kernel,support,pool)
                b=ix(kernel,support,support)+.001*torch.eye(len(support),device=DEVICE,dtype=kernel.dtype)
                value=kernel[pool,pool]-(z*torch.linalg.solve(b,z)).sum(dim=0)
            else:
                scores,nearest=ix(sqdist,pool,support).min(dim=1)
                if rule=="lcmd":
                    mass=torch.zeros(len(support),device=DEVICE,dtype=kernel.dtype)
                    mass.scatter_add_(0,nearest,scores)
                    value=torch.where(nearest==mass.argmax() if mass.max()>0 else torch.ones_like(nearest,dtype=torch.bool),
                                      scores,-torch.inf)
                else:
                    value=scores
            chosen=int(value.argmax())
            selected.append(pool.pop(chosen))
            ordering.append(dict(method=name,rank=len(selected),edge_id=keys[selected[-1]]["edge_id"],
                                 gpu_priority=float(value[chosen])))
        old=sorted((r for r in frozen if r["aggregation"]=="mean" and r["method"]==name),key=lambda r:int(r["rank"]))
        new=[keys[i]["edge_id"] for i in selected]
        if [r["edge_id"] for r in old]!=new:
            raise ValueError(f"GPU verification does not reproduce B5-R frozen ranking {name}")
        validation.append(dict(method=name,paths=len(new),rank_matches=len(new)))
    comparer=[dict(edge_a=keys[a]["edge_id"],edge_b=keys[b]["edge_id"],mean_soap_cosine_distance=float(1-soap[a,b]),
                   mean_activation_cosine_distance=float(1-activation[a,b])) for a,b in combinations(candidates,2)]
    write(out/"gpu_verified_rankings.csv",ordering)
    write(out/"candidate_pair_distances.csv",comparer)
    return validation


def local_soap_curve_distances(data, out):
    manifest=js(data/"b5_manifest.json")
    if digest(data/"soap_features.npz")!=manifest["features_sha256"]["soap"]:
        raise ValueError("B5-R SOAP feature archive changed")
    meta=js(data/"soap_features.json")["image_index"]
    with np.load(data/"soap_features.npz") as soap:
        features=t(soap["soap_li"])
    groups=defaultdict(list)
    for i,row in enumerate(meta):
        if row["cohort"]!="reference":
            groups[row["edge_id"]].append(i)
    keys=list(groups)
    curves=[]
    common=t(np.linspace(0,1,5))
    for edge in keys:
        ids=groups[edge]
        s=t([meta[i]["s"] for i in ids])
        vectors=features[ids]
        left=torch.searchsorted(s,common).clamp(1,len(s)-1)-1
        fraction=(common-s[left])/(s[left+1]-s[left])
        values=vectors[left]+fraction[:,None]*(vectors[left+1]-vectors[left])
        curves.append(torch.nn.functional.normalize(values,dim=1))
    z=torch.stack(curves)
    aligned=(z[:,None]*z[None,:]).sum(dim=-1).mean(dim=-1)
    reverse=(z[:,None]*z.flip(1)[None,:]).sum(dim=-1).mean(dim=-1)
    both=torch.clamp(1-torch.maximum(aligned,reverse),min=0).cpu().numpy()
    flipped=(reverse>aligned).cpu().numpy()
    write(out/"li_soap_curve_distances.csv",[
        dict(edge_a=keys[i],edge_b=keys[j],cosine_distance=both[i,j],reversed_orientation=bool(flipped[i,j]))
        for i,j in combinations(range(len(keys)),2)])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ("data","protocol","out"):
        ap.add_argument(name,type=Path)
    args=ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("H1 GPU step requires CUDA")
    torch.cuda.set_per_process_memory_fraction(.05)
    torch.cuda.reset_peak_memory_stats()
    args.out.mkdir()
    plan=js(args.protocol)
    start=perf_counter()
    torch.cuda.synchronize()
    keys,matrices,gates=matrices_and_folds(args.data,plan,args.out)
    rankings=gpu_score_rankings(args.data,args.out,keys,matrices,plan)
    local_soap_curve_distances(args.data,args.out)
    torch.cuda.synchronize()
    result=dict(GPU=torch.cuda.get_device_name(),torch=torch.__version__,CUDA=torch.version.cuda,dtype="float64",
                elapsed_seconds=perf_counter()-start,peak_torch_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                peak_torch_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
                protocol_sha256=digest(args.protocol),source_commit=plan["source_commit"],
                model_weight_sha256=js(args.data/"b5_manifest.json")["checkpoint_sha256"],
                input_file_hashes={p.name:digest(p) for p in args.data.iterdir() if p.is_file()},
                rankings_unchanged=rankings,net_primary_gate=gates)
    (args.out/"gpu_manifest.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result))


if __name__=="__main__":
    main()
