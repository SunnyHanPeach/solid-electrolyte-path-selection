"""Compare frozen B5-R path acquisition rules and small-sample utility baselines."""
import argparse
from collections import defaultdict
import csv
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


AGGS = ("mean", "p90", "concat")


def load_json(path):
    return json.loads(path.read_bytes())


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, records):
    columns = list(dict.fromkeys(k for row in records for k in row))
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows(records)


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def normalize(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def aggregates(matrix, index):
    groups = defaultdict(list)
    for i, row in enumerate(index):
        groups[row["cohort"], row["edge_id"]].append(i)
    output = {agg:[] for agg in AGGS}
    for ids in groups.values():
        curve = matrix[ids]
        s = [index[i]["s"] for i in ids]
        if len(ids)>1 and (s[0] != 0 or not np.isclose(s[-1],1) or np.any(np.diff(s) <= 0)):
            raise ValueError("Path reaction coordinate changed or is unordered")
        output["mean"].append(curve.mean(axis=0))
        output["p90"].append(np.quantile(curve,0.9,axis=0))
        output["concat"].append(np.array([np.interp(np.linspace(0,1,5),s,v) for v in curve.T]).T.ravel())
    return {agg:normalize(np.array(vec)) for agg,vec in output.items()}, list(groups)


def gp_var(kernel, train, test, reg=0.001):
    if len(train)==0:
        return kernel[test,test]
    x = kernel[np.ix_(train,test)]
    coef = np.linalg.solve(kernel[np.ix_(train,train)] + reg*np.eye(len(train)),x)
    var = kernel[test,test] - np.sum(x*coef,axis=0)
    if np.any(var < -1e-8):
        raise ValueError("Negative covariance posterior variance")
    return np.maximum(var,0)


def ranking(kernel, anchors, pool, rule):
    selected, priority = [], []
    remain = list(pool)
    distance = np.maximum(2.0 - 2.0*kernel, 0.0)
    while remain:
        support = anchors + selected
        if rule == "pv":
            scores = gp_var(kernel, support, remain)
            choice = int(np.argmax(scores))
        else:
            dist = distance[np.ix_(remain,support)]
            nearest = dist.argmin(axis=1)
            scores = dist[np.arange(len(remain)),nearest]
            if rule == "lcmd":
                mass = np.bincount(nearest,weights=scores,minlength=len(support))
                eligible = nearest == int(np.argmax(mass)) if mass.max()>0 else np.ones(len(nearest),bool)
                choice = int(np.argmax(np.where(eligible, scores,-np.inf)))
            else:
                choice = int(np.argmax(scores))
        selected.append(remain.pop(choice))
        priority.append(float(scores[choice]))
    return zip(selected,priority)


def scaled(x, tr, te):
    mean = x[tr].mean(axis=0)
    std = x[tr].std(axis=0)
    return (x[tr]-mean)/np.where(std<1e-10,1.0,std), (x[te]-mean)/np.where(std<1e-10,1.0,std)


def ridge_predict(x, y, tr, te, reg):
    train, test = scaled(x,tr,te)
    centered = y[tr] - y[tr].mean()
    a = train @ train.T + reg*np.eye(len(tr))
    return y[tr].mean() + test@train.T@np.linalg.solve(a,centered)


def krr_predict(kernel, control, y, tr, te, reg):
    train, test = scaled(control,tr,te)
    k = kernel[np.ix_(tr,tr)] + (train@train.T)/train.shape[1]
    cross = kernel[np.ix_(te,tr)] + (test@train.T)/train.shape[1]
    kcent = k-k.mean(axis=0)-k.mean(axis=1,keepdims=True)+k.mean()
    ccent = cross-cross.mean(axis=1,keepdims=True)-k.mean(axis=0)+k.mean()
    return y[tr].mean() + ccent@np.linalg.solve(kcent+reg*np.eye(len(tr)), y[tr]-y[tr].mean())


def historical_analysis(keys, refs, agg, kernels, candidates, labels, geom_history):
    block = {k:v for k,v in kernels.items() if k in ("soap_full","soap_li","activation_full","activation_li")}
    hist = [i for i,k in enumerate(keys) if k[0]=="historical"]
    ids = [keys[i][1] for i in hist]
    if len(hist)!=14 or set(ids)!=set(labels):
        raise ValueError("Historical target mapping differs")
    c = [candidates[i] for i in ids]
    families = np.array([r["anion_family"] for r in c])
    numeric = np.array([[float(r["init_Li_min_contact_A"]), int(r["images"]), int(r["optimizer_steps"]),
                         int(r["images"])*int(r["atoms"])*int(r["path_presentations_per_image"])] for r in c])
    flags = np.column_stack([families == group for group in sorted(set(families))]).astype(float)
    controls = np.column_stack((numeric,flags))
    novelty = {method:1-kernel[np.ix_(hist,refs)].max(axis=1) for method,kernel in block.items()}
    bvse = np.array([float(geom_history[i]["bvse_forward_eV"]) for i in ids])
    variants = {
        "Controls":controls, "BVSE":np.column_stack((controls,bvse)),
        "SOAP":np.column_stack((controls,novelty["soap_full"])),
        "Activation":np.column_stack((controls,novelty["activation_full"])),
        "SOAP+Activation":np.column_stack((controls,novelty["soap_full"], novelty["activation_full"])),
        "SOAP_Li":np.column_stack((controls,novelty["soap_li"])),
        "Activation_Li":np.column_stack((controls,novelty["activation_li"])),
        "SOAP+Activation_Li":np.column_stack((controls,novelty["soap_li"],novelty["activation_li"])),
        "Posterior_variance":np.column_stack((controls,gp_var(block["activation_full"],refs,hist))),
    }
    preds, summaries = [], []
    for design in ("LOPO","leave_anion_family_out"):
        masks = [np.arange(len(hist))==i for i in range(len(hist))] if design=="LOPO" else [
            families==f for f in sorted(set(families))]
        for target in ("Delta_abs","Delta_net"):
            y = np.array([float(labels[i][target]) for i in ids])
            krr = {"SOAP_KRR":block["soap_full"], "Activation_KRR":block["activation_full"],
                   "SOAP+Activation_KRR":0.5*(block["soap_full"]+block["activation_full"])}
            methods = ("train_mean", *variants, *krr)
            for method in methods:
                prediction = np.zeros(len(y))
                for mask in masks:
                    tr,te = np.flatnonzero(~mask), np.flatnonzero(mask)
                    if method in krr:
                        k = krr[method][np.ix_(hist,hist)]
                        prediction[te] = krr_predict(k,controls,y,tr,te,0.1)
                    elif method == "train_mean":
                        prediction[te] = y[tr].mean()
                    else:
                        prediction[te] = ridge_predict(variants[method],y,tr,te,1.0)
                for i,(truth,hat) in enumerate(zip(y,prediction)):
                    preds.append(dict(aggregation=agg,fold_scheme=design,target=target,method=method,
                                      edge_id=ids[i],anion_family=families[i],observed_meV=truth,predicted_meV=hat,
                                      absolute_error_meV=abs(truth-hat),label_class=labels[ids[i]][target+"_class"]))
                rho = spearmanr(y,prediction).statistic
                summaries.append(dict(aggregation=agg,fold_scheme=design,target=target,method=method,n=len(y),
                                      mae_meV=float(np.abs(y-prediction).mean()),spearman=float(rho) if np.isfinite(rho) else None))
    return preds,summaries


def candidate_rankings(keys,refs,kernels,candidate_rows,agg):
    pool = [i for i,k in enumerate(keys) if k[0]=="candidate"]
    rows = {i:candidate_rows[keys[i][1]] for i in pool}
    rnd = lambda i: sha256(f"b5r-20261126|random|{keys[i][1]}".encode()).hexdigest()
    strategies = {
        "Random": [(i,"") for i in sorted(pool,key=rnd)],
        "BVSE": [(i,float(rows[i]["em"])) for i in sorted(pool,key=lambda j:(float(rows[j]["em"]),keys[j][1]))],
        "SOAP": ranking(kernels["soap_full"],refs,pool,"kcenter"),
        "Activation": ranking(kernels["activation_full"],refs,pool,"kcenter"),
        "SOAP+Activation":ranking(0.5*(kernels["soap_full"]+kernels["activation_full"]),refs,pool,"kcenter"),
        "Activation_LCMD":ranking(kernels["activation_full"],refs,pool,"lcmd"),
        "Activation_PV":ranking(kernels["activation_full"],refs,pool,"pv"),
    }
    records, overlaps, diversity = [], [], []
    top = {}
    for name,result in strategies.items():
        order = list(result)
        top[name] = [i for i,_ in order]
        for rank,(i,val) in enumerate(order,1):
            row = rows[i]
            records.append(dict(aggregation=agg,method=name,rank=rank,edge_id=keys[i][1],
                                chemsys=row["chemsys"],family_proxy=row["family"],pool_category=row["材料家族"],
                                source_BVSE_eV=row["em"],rule_priority=val,
                                atoms=row["atoms"],images=row["images"],atom_images_proxy=row["atoms"]*row["images"]))
        for k in (3,5,8):
            ids = top[name][:k]
            pair = kernels["soap_full"][np.ix_(ids,ids)]
            diversity.append(dict(aggregation=agg,method=name,k=k,
                                  broad_families=len({rows[i]["材料家族"] for i in ids}),
                                  crystal_family_proxies=len({rows[i]["family"] for i in ids}),
                                  mean_pairwise_SOAP_cosine_distance=float(np.mean(1-pair[np.triu_indices(len(ids),1)])),
                                  total_atom_images_proxy=sum(rows[i]["atoms"]*rows[i]["images"] for i in ids)))
    for a,b in combinations(strategies,2):
        for k in (3,5,8):
            t1,t2 = set(top[a][:k]), set(top[b][:k])
            overlaps.append(dict(aggregation=agg,method_a=a,method_b=b,k=k,intersection=len(t1&t2),
                                 top_k_overlap=len(t1&t2)/k,jaccard=len(t1&t2)/len(t1|t2)))
    return records,overlaps,diversity


def run(args):
    repo,out = args.repo,args.output
    out.mkdir(exist_ok=True)
    (out/"activation_embeddings").mkdir()
    (out/"kernel_matrices").mkdir()
    soup, act = (load_json(p/"features.json") for p in (args.soap,args.activation))
    index = soup["image_index"]
    if index!=act["image_index"] or soup["geometry_sha256"]!=act["geometry_sha256"]:
        raise ValueError("SOAP and activation are for different images")
    with np.load(args.soap/"features.npz") as a, np.load(args.activation/"features.npz") as b:
        features = dict(a) | dict(b)
    # Verify that new pooling retains the historical migrating-site definitions exactly.
    hix = rows(repo/"results/phase_b3_a/image_feature_index.csv")
    old = {(r["edge_id"],int(r["image"])-1):int(r["feature_row_0based"]) for r in hix}
    keyrows = [(i,old[r["edge_id"],r["image"]]) for i,r in enumerate(index) if r["cohort"]=="historical"]
    checks = {}
    with np.load(repo/"results/phase_b3_a/initial_image_features.npz") as cached:
        for method,key in (("activation_li","mace_invariants"),("soap_li","soap_elements")):
            baseline = np.array([cached[key][j] for _,j in keyrows])
            now = np.array([features[method][i] for i,_ in keyrows])
            rel = np.linalg.norm(now-baseline,axis=1)/np.maximum(np.linalg.norm(baseline,axis=1),1e-12)
            checks[method]=float(rel.max())
            if checks[method]>1e-6:
                raise ValueError(f"{method} cache parity failed {checks[method]}")
            # Prefer the frozen values after verifying both implementations.
            for row,cached_value in zip(keyrows,baseline):
                features[method][row[0]]=cached_value

    cand = {r["edge_id"]:r for r in load_json(args.geometry/"candidate_manifest.json")["rows"] if r["status"]=="included"}
    old_candidates = {r["edge_id"]:r for r in rows(repo/"results/b2_overnight_01/six_hour_candidates.csv")}
    utility = repo/"results/phase_b3_a/path_utility_labels.csv"
    labels = {r["edge_id"]:r for r in rows(utility)}
    history = {r["edge_id"]:r for r in rows(args.history) if r["edge_id"] in labels}
    if not all(labels[edge]["seeds"]=="17;29;43" and labels[edge]["materials"]=="107" for edge in labels):
        raise ValueError("Protocol mixing detected")
    matrices, keys = {}, None
    for method,x in features.items():
        matrices[method],current_keys = aggregates(x,index)
        if keys is not None and keys!=current_keys:
            raise ValueError("Path index changed")
        keys = current_keys
    refs = [i for i,k in enumerate(keys) if k[0]=="reference"]
    keep = [i for i,r in enumerate(index) if r["cohort"]!="reference"]
    np.savez_compressed(out/"activation_embeddings"/"image_features.npz", **{m:x[keep] for m,x in features.items() if m.startswith("activation")})
    write_csv(out/"activation_embeddings"/"image_index.csv", [index[i] for i in keep])
    write_csv(out/"kernel_matrices"/"matrix_index.csv", [dict(cohort=c,edge_id=e) for c,e in keys])
    predictions,summaries,ranks,overlaps,diversity = [],[],[],[],[]
    for agg in AGGS:
        kernels = {m:v[agg] @ v[agg].T for m,v in matrices.items()}
        for matrix in kernels.values():
            if np.linalg.eigvalsh(matrix).min() < -1e-8:
                raise ValueError("Non-PSD linear kernel")
        np.savez_compressed(out/"kernel_matrices"/f"{agg}.npz",**kernels)
        p,s = historical_analysis(keys,refs,agg,kernels,old_candidates,labels,history)
        r,o,d = candidate_rankings(keys,refs,kernels,cand,agg)
        predictions.extend(p); summaries.extend(s)
        ranks.extend(r); overlaps.extend(o); diversity.extend(d)
    for name,x in (("historical_utility_analysis",predictions),("feature_comparison",summaries),
                   ("independent_candidate_ranking",ranks),("top_k_overlap",overlaps),("candidate_diversity",diversity)):
        write_csv(out/f"{name}.csv",x)

    result = dict(checkpoint_sha256=act["checkpoint_sha256"], untouched_utility_label_sha256=sha(utility),
                  frozen_candidate_manifest_sha256=sha(args.geometry/"candidate_manifest.json"),
                  geometry_sha256=act["geometry_sha256"], features_sha256={
                      "act":sha(args.activation/"features.npz"),"soap":sha(args.soap/"features.npz")},
                  cache_parity_max_relative_l2=checks, history_paths=len(labels),candidate_paths=len(cand),
                  references=len(refs), activation_runtime=act["versions"],soap_runtime=soup["versions"],
                  new_DFT=0, reserved_test_labels_read=0, new_MACE_training=0)
    (out/"manifest.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("repo","geometry","soap","activation","history","output"):
        p.add_argument(name,type=Path)
    run(p.parse_args())


if __name__=="__main__":
    main()
