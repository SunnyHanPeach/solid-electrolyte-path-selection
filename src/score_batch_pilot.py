"""Score the fixed batch pilot locally after all training predictions are written."""
import argparse
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile
import json
import sys

import numpy as np
from run_batch_pilot import checked, csv_rows, sha

def barrier(by_frame):
    if sorted(by_frame)!=list(range(1,len(by_frame)+1)):
        raise ValueError('Incomplete frame sequence')
    return max(by_frame.values())-by_frame[1]

def collect_predictions(folder, plan, data):
    all_jobs=[f'{arm}/seed{s}/result.json' for arm in plan['arms'] for s in plan['training']['seeds']]
    complete=json.loads((folder/'done.json').read_bytes())
    if not complete['all_finished'] or complete['completed']!=all_jobs:
        raise ValueError('Wait for all 15 blind predictions before scoring')
    expected_frames={(r['edge_id'],int(r['frame'])):r['material_id']
                     for r in csv_rows(data['evaluation_frames.csv'])}
    digest=sha(json.dumps(plan,sort_keys=True).encode())
    jobs={}
    environments=set()
    for name in all_jobs:
        run=folder/name
        job=json.loads(run.read_bytes())
        arm,seed,_=name.split('/')
        assert job['arm']==arm and job['seed']==int(seed.removeprefix('seed'))
        assert job['optimizer_steps']==710 and job['execution_sha256']==digest
        assert job['package_sha256']==plan['freeze_zip_sha256']
        assert job['environment']['device']=='cuda'
        environments.add(json.dumps(job['environment'],sort_keys=True))
        path=run.parent/'blind_predictions.csv'
        raw=path.read_bytes()
        if sha(raw)!=job['files'][path.name]:
            raise ValueError(f'Prediction hash differs for {run}')
        observed={}
        numbers=defaultdict(dict)
        for r in csv_rows(raw):
            key=r['edge_id'],int(r['frame'])
            if key in observed or not np.isfinite(float(r['energy_eV'])):
                raise ValueError(f'Duplicate or nonfinite prediction in {run}')
            observed[key]=r['material_id']
            numbers[key[0]][key[1]]=float(r['energy_eV'])
        if observed!=expected_frames:
            raise ValueError(f'Changed evaluation frames in {run}')
        jobs[arm,job['seed']]={e:barrier(x) for e,x in numbers.items()}
    if len(environments)!=1:
        raise ValueError('Models were not assessed in the same runtime')
    return jobs,json.loads(environments.pop())

def labels(project,data):
    sys.path.insert(0,str(project/'experiment2'))
    from prepare_data import metadata, raw_frames
    p=project/'迁移路径数据/nebDFT2k.zip'
    lock=json.loads(data['freeze_manifest.json'])
    if sha(p.read_bytes())!=lock['configuration_sha256']['init_archive']:
        raise ValueError('DFT input archive changed')
    reference={}
    with ZipFile(p) as z:
        for r in csv_rows(data['evaluation_materials.csv']):
            e=r['edge_id']
            energies={i:float(metadata(head)[b'energy'])
                      for i,_,_,head,_ in raw_frames(z,f'nebDFT2k/{e}_relaxed.xyz')}
            if len(energies)!=int(r['images']):
                raise ValueError(f'Incomplete DFT path at {e}')
            reference[e]=barrier(energies)
    return reference

def summarize(delta, resample):
    point=delta.mean(axis=0)
    interval=np.quantile(point[resample].mean(axis=1),[.025,.975])
    seed_means=delta.mean(axis=1)
    return dict(mean_meV=float(point.mean()),interval_95_meV=interval.tolist(),
                by_seed_meV=seed_means.tolist(),
                passes_lower_error_gate=bool(interval[1]<-.25 and max(seed_means)<0))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('bundle','gpu_results','project','result'):
        p.add_argument(n,type=Path)
    args=p.parse_args()
    plan,data=checked(args.bundle)
    predictions,environment=collect_predictions(args.gpu_results,plan,data)
    truth=labels(args.project,data)
    seed_ids=plan['training']['seeds']
    errors={a:np.array([[1000*abs(predictions[a,s][key]-v) for key,v in truth.items()]
                        for s in seed_ids]) for a in plan['arms']}
    draw=np.random.default_rng(20261124).integers(len(truth),size=(10000,len(truth)))
    results={}
    for name in ('batch1','batch2'):
        r,b=(errors[f'{name}_{mode}'] for mode in ('random','BVSE'))
        results[name]=dict(random_vs_D0=summarize(r-errors['D0'],draw),
                           BVSE_vs_D0=summarize(b-errors['D0'],draw),
                           BVSE_vs_random=summarize(b-r,draw))
    passes={k:all(res[k]['passes_lower_error_gate'] for res in results.values()) for k in results['batch1']}
    record=dict(metrics=results,reproduced_two_blocks=passes,
                BVSE_selection_advantage_supported=passes['BVSE_vs_D0'] and passes['BVSE_vs_random'],
                seeds=seed_ids,materials=len(truth),environment=environment,additional_Test_labels_read=False)
    with args.result.open('x',encoding='utf-8') as f:
        json.dump(record,f,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':
    main()
