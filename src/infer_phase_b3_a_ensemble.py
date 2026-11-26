"""Complete B1-style ensemble predictions for the 11 B3-A paths without cached runs."""
import argparse
from importlib.metadata import version
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np

from extract_phase_b3_a import B2, coordinates
from phase_a0_geometry import EXPECTED, export_csv, reaction_coordinate
from phase_b1_inference import (
    PERIODIC_ATOL, REPEAT_ATOL, check_mic, compare, file_sha, frozen_models, load_calculators,
    predict, predictions, read_json, rows, strict_check, tangent_vectors, write_json,
)

B3A = Path("results/phase_b3_a")
B1 = Path("results/phase_b1")
THREADS = 16


def prepare(args):
    b1 = read_json(args.repo/B1/"run_manifest.json")
    b3 = read_json(args.repo/B3A/"feature_manifest.json")
    for path,digest in b3["output_sha256"].items():
        strict_check(file_sha(args.repo/B3A/path)==digest,f"B3-A asset changed: {path}")
    for path,digest in b1["output_sha256"].items():
        strict_check(file_sha(args.repo/B1/path)==digest,f"B1 raw predictions changed: {path}")
    for k in ("neb","index","bundle"):
        strict_check(file_sha(getattr(args,k))==EXPECTED[k],f"Frozen geometry source changed: {k}")
    for label,k in (("weights_archive","weights"),("training_archive","training_archive"),
                    ("training_protocol_archive","protocol")):
        strict_check(file_sha(getattr(args,k))==b1["inputs_sha256"][label],f"B1 model provenance changed: {k}")
    models,protocol = frozen_models(args)
    strict_check([m for _,m in models.values()]==b1["checkpoints"] and protocol==b1["training_protocol"],
                 "B1 protocol or model identity changed")
    known = rows(args.repo/B1/"image_geometry.csv")
    cached_keys = {(r["edge_id"],int(r["image"])) for r in known}
    cached_edges = {e for e,_ in cached_keys}
    index = {(r["edge_id"],int(r["image"])):r for r in rows(args.repo/B3A/"image_feature_index.csv")}
    candidates = rows(args.repo/B2/"six_hour_candidates.csv")
    source, geometry = coordinates(args,candidates,index)
    strict_check(set(source)==set(index) and len(source)==78 and len(cached_keys)==17,"Unexpected geometry coverage")
    stored = {r["edge_id"]:r for r in rows(args.repo/B3A/"path_id_manifest.csv")}
    strict_check(len(stored)==len(geometry)==14 and all(g["init_sha256"]==stored[g["edge_id"]]["init_sha256"]
                 and g["original_member"]==stored[g["edge_id"]]["original_member"] for g in geometry),
                 "B3-A path identity differs")
    output = []
    for c in candidates:
        edge = c["edge_id"]
        if edge in cached_edges:
            strict_check({(edge,i) for i in range(1,int(c["images"])+1)}<=cached_keys,"Incomplete cached path")
            continue
        path = [source[edge,i] for i in range(1,int(c["images"])+1)]
        cell = path[0][0].cell.array
        li = np.array([atoms.positions[mover] for atoms,mover,_ in path])
        unwrapped,arc,s = reaction_coordinate(li,cell,path[0][0].pbc)
        check_mic(li,unwrapped,cell)
        tangent = tangent_vectors(unwrapped)
        for j,(atoms,mover,key) in enumerate(path):
            fields = dict(edge_id=edge,stage="init",image=j+1,
                          feature_row_0based=int(index[edge,j+1]["feature_row_0based"]),n_atoms=len(atoms),
                          Li_1based=mover+1,arc_A=float(arc[j]),reaction_coordinate=float(s[j]),
                          **{f"Li_unwrapped_{v}_A":float(x) for v,x in zip("xyz",unwrapped[j])},
                          **{f"tangent_{v}":float(x) for v,x in zip("xyz",tangent[j])})
            output.append((atoms,fields))
    strict_check(len(output)==61 and len({k["edge_id"] for _,k in output})==11,"Unexpected new query count")
    return output,models,protocol,b1


def smoke(calculators,new):
    atoms,key = max((entry for entry in new if .25 < entry[1]["reaction_coordinate"] < .75),
                   key=lambda entry:len(entry[0]))
    shifted = atoms.copy()
    shifted.positions += [.70,1.20,-.40]
    shifted.wrap()
    reports = []
    for s,calc in calculators.items():
        original = predict(calc,atoms)
        repeat = compare(original,predict(calc,atoms))
        wrapped = compare(original,predict(calc,shifted))
        strict_check(max(repeat.values())<=REPEAT_ATOL and max(wrapped.values())<=PERIODIC_ATOL,
                     f"Repeat or PBC acceptance failed for seed {s}")
        reports.append(dict(seed=s,repeat_abs_errors=repeat,translate_wrap_abs_errors=wrapped,
                            elements=len(original[1]),energy_eV=original[0]))
    return dict(status="pass",image=key,checks=reports,repeat_tolerance=REPEAT_ATOL,wrap_tolerance=PERIODIC_ATOL)


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    new,models,protocol,b1 = prepare(args)
    runtime = dict(device="cpu",dtype="float64",threads=THREADS,torch=str(torch.__version__),
                   python=platform.python_version(),platform=platform.platform(),
                   packages={n:version(n) for n in b1["runtime"]["packages"]})
    strict_check(torch.version.cuda is None and all(runtime[k]==b1["runtime"][k] for k in ("torch","packages","threads")),
                 "Expected the audited B1 CPU runtime")
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    strict_check(not (args.output/"run_manifest.json").exists(),"Output exists")
    calculators = load_calculators(models,torch)
    lock = smoke(calculators,new)
    write_json(args.output/"smoke.json",lock)
    started = perf_counter()
    result,atoms = predictions(calculators,new,models)
    elapsed = perf_counter()-started
    names = ("image_geometry.csv","raw_image_predictions.csv","raw_atomic_predictions.csv")
    for name,table in zip(names,([k for _,k in new],result,atoms)):
        export_csv(args.output/name,table)
    inputs = {p.as_posix():file_sha(args.repo/p) for p in (
        B3A/"feature_manifest.json",B3A/"analysis_manifest.json",B3A/"analysis_protocol.json",
        B1/"run_manifest.json",B1/"analysis_manifest.json")}
    write_json(args.output/"run_manifest.json",dict(source_commit=subprocess.check_output(
        ["git","rev-parse","HEAD"],cwd=args.repo,text=True).strip(),runtime=runtime,
        script_sha256=file_sha(Path(__file__)),new_paths=11,new_images=61,model_runs=len(result),
        raw_atomic_rows=len(atoms),batch_seconds=elapsed,source_input_sha256=inputs,
        base_archive_sha256=file_sha(args.neb),checkpoint_archive_sha256=file_sha(args.weights),
        source_B1_protocol=protocol,checkpoints=[m for _,m in models.values()],
        output_sha256={n:file_sha(args.output/n) for n in (*names,"smoke.json")}))
    print(f"CPU ensemble supplement: {len(result)} predictions in {elapsed:.1f} s, {len(atoms)} atom rows",flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ("repo","neb","index","bundle","weights","training_archive","protocol","output"):
        parser.add_argument(n,type=Path)
    main(parser.parse_args())
