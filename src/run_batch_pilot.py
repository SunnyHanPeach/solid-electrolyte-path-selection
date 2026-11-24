"""Train the frozen four-path pilot and write label-blind evaluation predictions."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from zipfile import ZipFile

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def csv_rows(raw):
    return list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))

def save_json(path, value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)

def checked(bundle):
    with ZipFile(bundle) as z:
        m=json.loads(z.read('execution.json'))
        payload={name:z.read(name) for name in m['files']}
    if any(sha(payload[name])!=checksum for name,checksum in m['files'].items()):
        raise ValueError('A deployment member changed')
    f=json.loads(payload['freeze_manifest.json'])
    if any(sha(payload[name])!=checksum for name,checksum in f['frozen_members_sha256'].items()):
        raise ValueError('The frozen candidate or evaluation manifest changed')
    if sha(payload['foundation.model'])!=f['configuration_sha256']['foundation_checkpoint']:
        raise ValueError('The initial foundation checkpoint changed')
    evaluation=csv_rows(payload['evaluation_materials.csv'])
    if len({r['material_id'] for r in evaluation})!=91 or len(m['arms'])!=5:
        raise ValueError('Unexpected panel or training arms')
    return m,payload

def geometries(m,payload):
    from ase.io import read
    train=read(io.StringIO(payload['training.xyz'].decode()),index=':',format='extxyz')
    evaluation=read(io.StringIO(payload['evaluation_geometry.xyz'].decode()),index=':',format='extxyz')
    rows=csv_rows(payload['training_frames.csv'])
    frames=csv_rows(payload['evaluation_frames.csv'])
    if len(train)!=len(rows) or len(evaluation)!=len(frames) or len(frames)!=501:
        raise ValueError('Unexpected geometry count')
    for atoms,r in zip(evaluation,frames):
        if atoms.calc is not None or atoms.info or set(atoms.arrays)!={'numbers','positions'} or not atoms.pbc.all():
            raise ValueError('Evaluation geometry contains labels or invalid properties')
        if len(atoms)!=int(r['n_atoms']):
            raise ValueError('Evaluation atom count changed')
    for atoms,r in zip(train,rows):
        if len(atoms)!=int(r['n_atoms']) or abs(atoms.get_potential_energy()-float(r['energy_eV']))>1e-8:
            raise ValueError('Training energy label disagrees with the frame index')
    return train,evaluation,rows,frames

def ranks(n,cfg,seed,steps,schedule):
    config=dict(cfg,epochs=math.ceil(steps/math.ceil(n/cfg['batch_size'])))
    from itertools import islice
    order=list(islice(schedule(n,config,seed),steps))
    if len(order)!=steps:
        raise ValueError('Incomplete training order')
    return order

def predict(model,frames,graphs,device,batch):
    import torch
    out=[]
    with torch.no_grad():
        for first in range(0,len(frames),2):
            ids=list(range(first,min(first+2,len(frames))))
            energies=model(batch(graphs,ids,device).to_dict(),training=False,compute_force=False)['energy']
            if not torch.isfinite(energies).all():
                raise ValueError('Nonfinite model prediction')
            for i,energy in zip(ids,energies.cpu().tolist()):
                out.append({k:frames[i][k] for k in ('edge_id','material_id','frame')}|{'energy_eV':energy})
    return out

def run(args):
    m, data=checked(args.bundle)
    train,evaluation,rows,frames=geometries(m,data)
    print('Preflight passed:',len(rows),'labeled training images,',len(frames),'blind evaluation images',flush=True)
    if args.preflight:
        return
    sys.path.insert(0,str(args.bundle.resolve()))
    from run_ef import build_batch, fit, make_graphs, schedule, write_csv
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD','1')
    import ase,mace,numpy as np,torch
    with torch.serialization.safe_globals([slice]):
        from e3nn import o3
    from mace.calculators import MACECalculator
    from mace import data as mace_data
    device='cpu' if args.smoke_cpu else 'cuda'
    if (mace.__version__,ase.__version__,np.__version__)!=('0.3.14','3.29.0','2.2.6'):
        raise RuntimeError('Unexpected MACE/ASE/NumPy environment')
    env=dict(device=device,torch=str(torch.__version__),mace=mace.__version__,ase=ase.__version__,numpy=np.__version__)
    if device=='cuda':
        if not torch.cuda.is_available() or str(torch.__version__)!='2.8.0+cu128':
            raise RuntimeError('Expected the verified PyTorch CUDA environment')
        env.update(gpu=torch.cuda.get_device_name(),driver=subprocess.check_output(
            ['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).splitlines()[0].strip())
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    with tempfile.NamedTemporaryFile(dir=args.bundle.parent,suffix='.model',delete=False) as temp:
        temp.write(data['foundation.model'])
        checkpoint=Path(temp.name)
    try:
        calc=MACECalculator(model_paths=str(checkpoint),device=device,default_dtype='float32')
    finally:
        checkpoint.unlink()
    foundation=calc.models[0].eval()
    cfg=m['training']
    if args.smoke_cpu or args.smoke_gpu:
        # The CPU smoke samples paths; the GPU smoke checks the largest training graphs.
        one_per_path={}
        for i,r in enumerate(rows):
            one_per_path.setdefault(r['edge_id'],i)
        ids=list(one_per_path.values()) if args.smoke_cpu else list(range(len(train)))
        train_graphs=make_graphs(train,rows,ids,calc)
        if args.smoke_gpu:
            largest=sorted(ids,key=lambda i:train_graphs[i].edge_index.shape[1],reverse=True)[:2]
            torch.cuda.reset_peak_memory_stats()
        else:
            largest=[max((i for i in ids if rows[i]['role']==role),key=lambda i:int(rows[i]['n_atoms']))
                     for role in ('initial','candidate')]
        model,trace,_,grad,_,elapsed=fit(foundation,train_graphs,largest,[[0,1],[0,1]],cfg,17,device,True)
        output=predict(model,frames[:2],
            {i:mace_data.AtomicData.from_config(mace_data.config_from_atoms(a,key_specification=mace_data.KeySpecification(),head_name=calc.head),z_table=calc.z_table,
                    cutoff=calc.r_max,heads=calc.available_heads) for i,a in enumerate(evaluation[:2])},device,build_batch)
        assert len(output)==2
        memory=dict(peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved()) if args.smoke_gpu else {}
        print(json.dumps(dict(smoke_device=device,order_steps=len(trace),force_gradient=grad,
                              training_seconds=elapsed,**memory)))
        return
    train_graphs=make_graphs(train,rows,range(len(train)),calc)
    label_free=mace_data.KeySpecification()
    blind={}
    for i,atoms in enumerate(evaluation):
        config=mace_data.config_from_atoms(atoms,key_specification=label_free,head_name=calc.head)
        blind[i]=mace_data.AtomicData.from_config(config,z_table=calc.z_table,cutoff=calc.r_max,heads=calc.available_heads)
    args.out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    finished=[]
    for arm,indices in m['arms'].items():
        for seed in cfg['seeds']:
            job=args.out/arm/f'seed{seed}'
            job.mkdir(parents=True,exist_ok=True)
            path=job/'result.json'
            order=ranks(len(indices),cfg,seed,710,schedule)
            expected=dict(arm=arm,seed=seed,environment=env,optimizer_steps=len(order),
                          order_sha256=sha(json.dumps(order).encode()),package_sha256=m['freeze_zip_sha256'],
                          execution_sha256=sha(json.dumps(m,sort_keys=True).encode()))
            if path.exists():
                prior=json.loads(path.read_bytes())
                if any(prior.get(k)!=v for k,v in expected.items()) or any(
                      sha((job/name).read_bytes())!=h for name,h in prior['files'].items()):
                    raise ValueError(f'Checkpoint provenance failed at {job}')
            else:
                now=time.monotonic()
                if now-start+190>args.max_runtime_seconds:
                    print('Stopped before starting the next fit due to the time limit',flush=True)
                    return
                print(f'{arm} seed {seed}',flush=True)
                torch.cuda.reset_peak_memory_stats()
                model,trace,_,_,_,training=fit(foundation,train_graphs,indices,order,cfg,seed,device,False)
                output=predict(model,frames,blind,device,build_batch)
                write_csv(job/'training.csv',trace)
                write_csv(job/'blind_predictions.csv',output)
                torch.save({name:value.cpu() for name,value in model.state_dict().items()},job/'weights.pt')
                files={name:sha((job/name).read_bytes()) for name in ('training.csv','blind_predictions.csv','weights.pt')}
                save_json(path,dict(expected,files=files,wall_seconds=time.monotonic()-now,
                                    training_seconds=training,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
                del model
            finished.append(path.relative_to(args.out).as_posix())
            save_json(args.out/'progress.json',dict(completed=finished,planned=15))
    save_json(args.out/'done.json',dict(completed=finished,all_finished=len(finished)==15))
    print(len(finished),'fits complete',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle',type=Path)
    parser.add_argument('--out',type=Path,default=Path('batch_pilot_results'))
    parser.add_argument('--max-runtime-seconds',type=float,default=float('inf'))
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight',action='store_true')
    mode.add_argument('--smoke-cpu',action='store_true')
    mode.add_argument('--smoke-gpu',action='store_true')
    run(parser.parse_args())
