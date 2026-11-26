"""Extend frozen A1 baseline profiles to the B2 paths using only archived Init geometry."""
import argparse
from collections import defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from ase import Atoms

from audit_phase_b0 import frame_prefix
from phase_a0_geometry import (
    EXPECTED, atoms_from_bytes, csv_rows, export_csv, parse_path, reaction_coordinate, validate_sequence,
)
from phase_a1_descriptors import (
    D0_WEIGHT, GRID, METHODS, SOAP_SETTINGS, load_representatives, make_extractors, reference_profiles,
)
from phase_b1_inference import file_sha, read_json, rows, write_json

B2 = Path("results/b2_overnight_01")
D0_SHA = "dda468e74c0dbd46c3d5b07121b900184a917dd319415e826af6e25b56ee7081"

def require(ok, message):
    if not ok:
        raise ValueError(message)


def coordinates(args, candidates, a1_index):
    movers = {r["edge_id"]:r for r in rows(args.index)}
    queries, source = {}, []
    with ZipFile(args.neb) as z:
        for row in candidates:
            edge = row["edge_id"]
            mover = int(movers[edge]["moving_Li_row"]) - 1
            member, digest, images = parse_path(z,edge,"init",mover)
            require(digest==row["init_sha256"] and len(images)==int(row["images"]) and
                    all(len(a)==int(row["atoms"]) for a in images), f"Init evidence differs for {edge}")
            changed, _ = validate_sequence(images,mover,"init")
            require(not any(changed), f"Unstable Init atom mapping for {edge}")
            _, arc, s = reaction_coordinate([a.positions[mover] for a in images],images[0].cell,images[0].pbc)
            require(np.all(np.diff(s)>0), f"Duplicate Li positions for {edge}")
            for i,(a,v) in enumerate(zip(images,s),1):
                geo = Atoms(numbers=a.numbers,positions=a.positions,cell=a.cell,pbc=a.pbc)
                key = dict(edge_id=edge,stage="init",image=i,reaction_coordinate=float(v),Li_1based=mover+1)
                queries[edge,i] = (geo,mover,key)
                if (edge,i) in a1_index:
                    row_a1 = a1_index[edge,i]
                    require(abs(float(row_a1["reaction_coordinate"])-v)<1e-12 and int(row_a1["Li_1based"])==mover+1,
                            f"A1 image identity differs for {edge}/{i}")
            source.append(dict(edge_id=edge,material_id=row["material_id"],chemsys=row["chemsys"],
                               anion_family=row["anion_family"],init_images=len(images),init_atoms=len(images[0]),
                               original_neb=args.neb.name,original_member=member,init_sha256=digest,Li_1based=mover+1,
                               arc_length_A=float(arc[-1]),frozen_selection_arc_length_A=float(row["path_length_A"])))
    return queries, source


def verified_inputs(args):
    repo = args.repo
    phase = repo/"results"
    paths = {n:phase/n for n in ("phase_a0","phase_a01","phase_a1","phase_a2","phase_b0","phase_b1")}
    b1 = read_json(paths["phase_b1"]/"run_manifest.json")
    for name,sha in b1["inputs_sha256"].items():
        if name.startswith("results/"):
            require(file_sha(repo/name)==sha, f"B1's input changed: {name}")
    require(file_sha(args.weights)==b1["inputs_sha256"]["weights_archive"],"Checkpoint archive differs")
    source = dict(neb=args.neb,index=args.index,bundle=args.bundle)
    require(all(file_sha(p)==EXPECTED[k] for k,p in source.items()), "Frozen geometry sources differ")
    a1 = read_json(paths["phase_a1"]/"manifest.json")
    a2 = read_json(paths["phase_a2"]/"manifest.json")
    require(a1["checkpoint_sha256"]==a2["mace_checkpoint_sha256"]==b1["checkpoints"][0]["sha256"],
            "A1, A2 and B1 reference models differ")
    require(a1["soap_settings"]==a2["soap_parameters"]["baseline"]==SOAP_SETTINGS
            and a1["rdf_grid_A"]==GRID.tolist() and a1["coordination_A"]=={"hard":3.,"smooth":[2.6,3.4]}
            and a1["rdf_sigma_A"]==.12 and a1["rdf_upper_taper_A"]==[4.6,5.], "A1/A2 extractor settings differ")
    for package,wanted in (("ase",a1["ase"]),("dscribe",a1["dscribe"])):
        require(version(package)==wanted, f"Runtime {package} differs from A1")
    require(version("mace-torch")==b1["runtime"]["packages"]["mace-torch"], "MACE runtime differs from B1")
    b0 = {r["checkpoint_id"]:r for r in rows(paths["phase_b0"]/"checkpoint_inventory.csv")}
    with ZipFile(args.weights) as z:
        for r in b1["checkpoints"]:
            expected = b0[r["checkpoint_id"]]["sha256_recorded"]
            require(expected==r["sha256"] and expected==sha256(z.read(r["member"])).hexdigest(),
                    f"Checkpoint bytes differ for {r['checkpoint_id']}")
    for path,expect in ((paths["phase_a0"]/"geometry_manifest.csv",a1["A0_manifest_sha256"]),
                        (paths["phase_a01"]/"reference_material_distribution.csv",a1["A01_materials_sha256"])):
        require(file_sha(path)==expect,"A1 reference provenance changed")
    a1index = {(r["edge_id"],int(r["image"])):r for r in rows(paths["phase_a1"]/"feature_index.csv")
               if r["stage"]=="init" and r["DFT_preselectable"]=="1"}
    require(len(a1index)==17, "Incomplete A1 Init cache")
    return paths,b1,a1,a2,a1index


def extract(args):
    started = perf_counter()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    require(torch.version.cuda is None, "CPU PyTorch build required")
    paths,b1,a1,a2,cached = verified_inputs(args)
    require(str(torch.__version__)==a1["torch"],"PyTorch runtime differs from A1")
    repo = args.repo
    out = args.output
    require((out/"analysis_protocol.json").is_file(),"Freeze the aggregation protocol before extracting")
    require(not (out/"feature_manifest.json").exists(), "B3-A features already exist")
    candidates = rows(repo/B2/"six_hour_candidates.csv")
    require(len(candidates)==14 and len({r["edge_id"] for r in candidates})==14, "Incomplete B2 candidate matrix")
    queries, sources = coordinates(args,candidates,cached)
    with ZipFile(args.bundle) as z:
        raw = frame_prefix(z.read("training.xyz"),278)
        require(sha256(raw).hexdigest()==D0_SHA,"Canonical D0 frame bytes differ")
        frames = atoms_from_bytes(raw)
        metadata = csv_rows(z.read("training_frames.csv"))[:278]
    require(len(frames)==len(metadata)==278, "Incomplete D0 data")
    species = a1["species"]
    require({s for a,_,_ in queries.values() for s in a.get_chemical_symbols()} <=set(species),
            "New species would change the frozen feature basis")
    reps = load_representatives(paths["phase_a01"]/"reference_material_distribution.csv",frames)
    calc, trace = make_extractors(SimpleNamespace(checkpoints=args.weights,threads=args.threads),species)
    require(trace["checkpoint"]==D0_WEIGHT and trace["layers"]==a1["layers"], "A1 feature extraction differs")
    bank, ids = defaultdict(list), []
    keys = list(queries)
    vectors = defaultdict(list)
    with np.load(paths["phase_a1"]/"path_features.npz",allow_pickle=False) as stored:
        anchor_key = next(iter(cached))
        atoms, mover, _ = queries[anchor_key]
        check = calc(atoms,np.array([mover]))
        anchor = int(cached[anchor_key]["feature_row_0based"])
        parity = {m:float(np.max(np.abs(check[m][0]-stored[m][anchor]))) for m in METHODS}
        require(max(parity.values())<1e-8,"Smoke descriptor does not reproduce the A1 anchor")
        for key in keys:
            a,m,_ = queries[key]
            # Cache rows include relaxed geometries too. Only Init keys enter B3-A.
            current = ({method:stored[method][int(cached[key]["feature_row_0based"]):][:1]
                        for method in METHODS} if key in cached else calc(a,np.array([m])))
            for k,v in current.items():
                vectors[k].append(v)
    vectors = {k:np.concatenate(v) for k,v in vectors.items()}
    print(f"{len(keys)} Init images ready, reused {len(cached)} cached vectors",flush=True)
    for i,(atoms,meta) in enumerate(zip(frames,metadata),1):
        centers = np.flatnonzero(atoms.numbers==3)
        for name,v in calc(atoms,centers).items():
            bank[name].append(v)
        ids.extend((i,int(li+1),meta["material_id"]) for li in centers)
        if i%70==0:
            print(f"D0 reference {i}/278",flush=True)
    bank = {k:np.concatenate(v) for k,v in bank.items()}
    use = np.asarray([(i,li) in reps for i,li,_ in ids])
    require(len(ids)==9517 and use.sum()==2469,"D0 bank counts differ")
    all_profiles = reference_profiles(vectors,bank,[queries[k][2] for k in keys],ids,use)
    old = {(r["edge_id"],int(r["image"]),r["representation"]):r
           for r in rows(paths["phase_a1"]/"novelty_profiles.csv") if r["stage"]=="init"}
    a2base = {(r["edge_id"],int(r["image"]),r["representation"],r["reference"]):float(r["novelty"])
              for r in rows(paths["phase_a2"]/"reference_sensitivity.csv")
              if r["stage"]=="init" and r["variant"]=="baseline" and r["reference"] in ("raw","deduplicated")}
    old_gaps = defaultdict(float)
    for p in all_profiles:
        key = p["edge_id"],p["image"],p["representation"]
        p["asset_origin"] = "A1 cached profile" if key in old else "B3-A Init extraction"
        if key in old:
            orig = old[key]
            for k,ref in (("raw_nearest","raw"),("dedup_nearest","deduplicated")):
                pair = (float(orig[k]),a2base[key+(ref,)])
                delta = max(abs(p[k]-c) for c in pair)
                old_gaps[k] = max(old_gaps[k],delta)
                require(delta<1e-8, f"Rebuilt reference does not reproduce A1/A2 {key} {k}")
            for field in ("raw_nearest","dedup_nearest","median_material_nearest","dedup_nearest_D0_frame",
                          "dedup_nearest_D0_Li","dedup_nearest_material"):
                p[field] = orig[field]
    export_csv(out/"path_id_manifest.csv",sources)
    export_csv(out/"image_static_profiles.csv",all_profiles)
    export_csv(out/"image_feature_index.csv",[
        dict(queries[k][2],feature_row_0based=i,asset_origin="A1" if k in cached else "B3-A")
        for i,k in enumerate(keys)])
    coord = {k:float(vectors["coordination_hard"][i].sum()) for i,k in enumerate(keys)}
    export_csv(out/"image_coordination_counts.csv",[dict(queries[k][2],initial_hard_cn_3A=coord[k]) for k in keys])
    np.savez_compressed(out/"initial_image_features.npz",**vectors)
    write_json(out/"feature_manifest.json",dict(
        source_git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip(),
        source_script_sha256=file_sha(Path(__file__)),
        a1_extractor_sha256=file_sha(repo/"src/phase_a1_descriptors.py"),
        geometry_helpers_sha256=file_sha(repo/"src/phase_a0_geometry.py"),
        runtime=dict(trace,device="cpu",threads=args.threads,python=platform.python_version(),platform=platform.platform(),
                     packages={n:version(n) for n in ("ase","dscribe","mace-torch","torch","numpy","scipy","e3nn")}),
        checkpoint_sha256={r["checkpoint_id"]:r["sha256"] for r in b1["checkpoints"]},
        frozen_settings={k:a1[k] for k in ("species","rdf_grid_A","soap_settings","coordination_A","rdf_sigma_A",
                         "rdf_upper_taper_A","chemical_soap_compression","reference_dedup_radius_A","methods")},
        input_sha256=dict(b1["inputs_sha256"],**{
             "B2_candidates":file_sha(repo/B2/"six_hour_candidates.csv"),
             "A2_parameter_sensitivity":file_sha(paths["phase_a2"]/"parameter_sensitivity.csv"),"D0_XYZ":D0_SHA,
             "reference_representatives":file_sha(paths["phase_a01"]/"reference_material_distribution.csv"),
             "aggregation":file_sha(out/"analysis_protocol.json")}),
        reference_counts=dict(raw=len(ids),deduplicated=int(use.sum()),materials=50),
        path_count=len(sources),image_count=len(keys),cached_initial_images=len(cached),
        smoke_cached_vector_abs_error=parity,anchor_reference_profile_abs_error=dict(old_gaps),
        output_sha256={name:file_sha(out/name) for name in ("path_id_manifest.csv","image_static_profiles.csv",
                       "image_feature_index.csv","image_coordination_counts.csv","initial_image_features.npz")},
        elapsed_seconds=perf_counter()-started))
    print("B3-A static features and reference parity verified",flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for f in ("repo","neb","index","bundle","weights","output"):
        p.add_argument(f,type=Path)
    p.add_argument("--threads",type=int,default=8)
    extract(p.parse_args())
