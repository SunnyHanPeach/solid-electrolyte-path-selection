"""Join B1 and B3-A supplement predictions and repeat the frozen path-level analysis."""
import argparse
from collections import defaultdict
from math import fsum
from pathlib import Path
from statistics import stdev

import numpy as np

from infer_phase_b3_a_ensemble import B1, B3A
from phase_a0_geometry import export_csv
from phase_b1_analysis import force_components
from phase_b1_inference import SEEDS, file_sha, read_json, rows, strict_check, write_json
from score_phase_b3_a import NUISANCE, aggregate, analyze, figures
from verify_phase_b1 import pairwise_spread, verify as verify_b1


def close(actual,expected,tol=1e-10):
    np.testing.assert_allclose(actual,expected,rtol=0,atol=tol)


def verified_profiles(repo,out,manifest,protocol):
    for p,h in manifest["source_input_sha256"].items():
        strict_check(file_sha(repo/p)==h,f"Upstream manifest changed: {p}")
    for p,h in manifest["output_sha256"].items():
        strict_check(file_sha(out/p)==h,f"Supplement file changed: {p}")
    indexed = {(r["edge_id"],int(r["image"])):r for r in rows(repo/B3A/"image_feature_index.csv")}
    geometry = defaultdict(list)
    for r in rows(out/"image_geometry.csv"):
        k = r["edge_id"],int(r["image"])
        strict_check(indexed[k]["stage"]==r["stage"]=="init" and indexed[k]["Li_1based"]==r["Li_1based"],
                     f"B3-A geometry mismatch at {k}")
        close(float(indexed[k]["reaction_coordinate"]),float(r["reaction_coordinate"]))
        geometry[k[0]].append(r)
    weights = {r["checkpoint_id"]:r["sha256"] for r in manifest["checkpoints"]}
    atomic = defaultdict(list)
    for r in rows(out/"raw_atomic_predictions.csv"):
        atomic[r["edge_id"],int(r["image"]),int(r["seed"])].append(r)
    energy,force = {},{}
    for r in rows(out/"raw_image_predictions.csv"):
        key = r["edge_id"],int(r["image"]),int(r["seed"])
        expected_id = f"e0-A-{key[2]}"
        atoms = atomic[key]
        n,li = int(r["n_atoms"]),int(r["Li_1based"])
        strict_check(key not in energy and r["stage"]=="init" and r["checkpoint_id"]==expected_id and
                     r["checkpoint_sha256"]==weights[expected_id] and len(atoms)==n and
                     [int(a["atom_1based"]) for a in atoms]==list(range(1,n+1)),"Prediction keys differ")
        strict_check(all(a["stage"]=="init" for a in atoms) and atoms[li-1]["element"]=="Li" and
                     [int(a["atom_1based"]) for a in atoms if a["is_moving_Li"]=="1"]==[li],
                     f"Invalid moving Li key: {key}")
        energies = [float(a["atomic_model_energy_eV"]) for a in atoms]
        e = float(r["total_energy_eV"])
        strict_check(np.isfinite([*energies,e]).all(),f"Nonfinite energy: {key}")
        close([fsum(energies),float(r["mean_energy_per_atom_eV"])*n],e,1e-9)
        all_force = np.asarray([[float(a[f"F{v}_eV_per_A"]) for v in "xyz"] for a in atoms])
        strict_check(np.isfinite(all_force).all(),f"Nonfinite force: {key}")
        l = all_force[li-1]
        close(l,[float(r[f"Li_F{v}_eV_per_A"]) for v in "xyz"])
        energy[key],force[key] = e,l
    strict_check(set(atomic)==set(energy) and len(energy)==183 and len(geometry)==11,
                 "Incomplete supplemental model grid")
    required = {(edge,int(r["image"]),seed) for edge,path in geometry.items() for r in path for seed in SEEDS}
    strict_check(required==set(energy),"Missing seed/image keys")
    profiles, seed_barriers = [], []
    fields = protocol["ensemble_image_fields"]
    for edge,g in geometry.items():
        g.sort(key=lambda r:int(r["image"]))
        keys = [int(r["image"]) for r in g]
        e = np.array([[energy[edge,image,s] for s in SEEDS] for image in keys])
        raw = e.std(axis=1,ddof=1)*1000
        mean = (e-e.mean(axis=0)).std(axis=1,ddof=1)*1000
        initial = (e-e[0]).std(axis=1,ddof=1)*1000
        center = [fsum(e[:,j])/len(e) for j in range(len(SEEDS))]
        for j,seed in enumerate(SEEDS):
            barrier = 1000*(max(e[:,j])-e[0,j])
            seed_barriers.append(dict(edge_id=edge,seed=seed,forward_barrier_meV=float(barrier),
                                 peak_image=keys[int(e[:,j].argmax())],source="B3-A CPU supplement"))
        for i,r in enumerate(g):
            f = np.stack([force[edge,keys[i],s] for s in SEEDS])
            t = np.asarray([float(r[f"tangent_{v}"]) for v in "xyz"])
            close(np.linalg.norm(t),1.)
            components = force_components(f,t)
            parallel = f@t
            normal = f-parallel[:,None]*t
            close([components[k] for k in ("Li_force_sd","Li_tangent_sd","Li_normal_sd")],
                  [pairwise_spread(f),stdev(parallel.tolist()),pairwise_spread(normal)])
            close([raw[i],mean[i],initial[i]],
                  [1000*stdev(v) for v in (e[i].tolist(),
                   [x-y for x,y in zip(e[i],center)],(e[i]-e[0]).tolist())],1e-9)
            values = dict(components,energy_raw_sd=float(raw[i]),energy_centered_sd=float(mean[i]),
                          energy_from_start_sd=float(initial[i]))
            profiles.append(dict(edge_id=edge,stage="init",image=keys[i],reaction_coordinate=float(r["reaction_coordinate"]),
                            Li_1based=int(r["Li_1based"]),source="B3-A CPU supplement",**{k:values[k] for k in fields}))
    historical = verify_b1(repo/B1)
    for r in rows(repo/B1/"image_disagreement.csv"):
        strict_check(r["stage"]=="init","Unexpected B1 stage")
        profiles.append(dict(edge_id=r["edge_id"],stage=r["stage"],image=int(r["image"]),
            reaction_coordinate=float(r["reaction_coordinate"]),Li_1based=int(r["Li_1based"]),
            source="B1 cached",**{k:float(r[k]) for k in fields}))
    for r in rows(repo/B1/"per_seed_barriers.csv"):
        seed_barriers.append(dict(r,source="B1 cached"))
    all_keys = {(r["edge_id"],r["image"]) for r in profiles}
    strict_check(len(profiles)==len(all_keys)==78 and all_keys==set(indexed),"Joined profiles are not a full unique cohort")
    return sorted(profiles,key=lambda r:(r["edge_id"],r["image"])),seed_barriers,historical


def full_matrix(repo,protocol,profiles,barriers):
    original = rows(repo/B3A/"path_feature_matrix.csv")
    old = defaultdict(list)
    values = defaultdict(list)
    for r in profiles:
        values[r["edge_id"]].append(r)
    for r in barriers:
        old[r["edge_id"]].append(r)
    full = []
    coverage = []
    for row in original:
        edge = row["edge_id"]
        profile = values[edge]
        barrier = old[edge]
        n = int(row["init_images"])
        strict_check([r["image"] for r in profile]==list(range(1,n+1)) and len(barrier)==3 and
                     {int(r["seed"]) for r in barrier}==set(SEEDS),f"Incomplete path profile: {edge}")
        matrix = {k:(None if v=="" else float(v)) if "__" in k or k in NUISANCE or k=="forward_barrier_sd_meV" else v
                  for k,v in row.items()}
        for feature in protocol["ensemble_image_fields"]:
            aggs = aggregate([r[feature] for r in profile],[r["reaction_coordinate"] for r in profile])
            for suffix,value in aggs.items():
                key = f"{feature}__{suffix}"
                if matrix[key] is not None:
                    close(value,matrix[key])
                matrix[key] = value
        sd = stdev([float(r["forward_barrier_meV"]) for r in barrier])
        if matrix["forward_barrier_sd_meV"] is not None:
            close(sd,matrix["forward_barrier_sd_meV"])
        matrix["forward_barrier_sd_meV"] = sd
        full.append(matrix)
        coverage.append(dict(edge_id=edge,images=n,nominal_members=3,status="complete",source=profile[0]["source"]))
    return full,coverage


def main(repo,out):
    protocol = read_json(repo/B3A/"analysis_protocol.json")
    old_analysis = read_json(repo/B3A/"analysis_manifest.json")
    for name in ("path_feature_matrix.csv","path_utility_labels.csv","spearman_ranking.csv"):
        strict_check(file_sha(repo/B3A/name)==old_analysis["analysis_outputs"][name],f"Frozen B3-A data changed: {name}")
    manifest = read_json(out/"run_manifest.json")
    profiles,barriers,prior = verified_profiles(repo,out,manifest,protocol)
    matrix,coverage = full_matrix(repo,protocol,profiles,barriers)
    labels = [{k:float(v) if k in ("Delta_abs","Delta_net") else v for k,v in r.items()}
              for r in rows(repo/B3A/"path_utility_labels.csv")]
    for name,data in (("image_ensemble_profiles.csv",profiles),("per_seed_barriers.csv",barriers),
                      ("path_feature_matrix.csv",matrix),("ensemble_coverage.csv",coverage)):
        export_csv(out/name,data)
    records,primary,heatmap = analyze(matrix,labels,protocol,out)
    figures(out,records,labels,primary,heatmap)
    manifest = dict(run_manifest_sha256=file_sha(out/"run_manifest.json"),source_B3A_labels_sha256=file_sha(
                    repo/B3A/"path_utility_labels.csv"),script_sha256=file_sha(Path(__file__)),
                    frozen_protocol_sha256=file_sha(repo/B3A/"analysis_protocol.json"),
                    B1_cached_verification=prior,verified_new_model_runs=183,complete_paths=len(matrix),
                    analysis_sha256={p.relative_to(out).as_posix():file_sha(p) for p in sorted(out.rglob("*"))
                                     if p.suffix in (".csv",".png")})
    write_json(out/"analysis_manifest.json",manifest)
    print(f"Verified and joined 78 image profiles for {len(matrix)} paths; reran the frozen association audit")


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("repo",type=Path)
    p.add_argument("out",type=Path)
    main(**vars(p.parse_args()))
