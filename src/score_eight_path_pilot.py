"""Compare nested eight-path and four-path arms after all cloud fits finish."""
import argparse
import csv
from collections import defaultdict
import io
import json
import math
from pathlib import Path
import tarfile
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from run_eight_path_pilot import checked, csv_rows, sha
from score_batch_pilot import barrier, labels, summarize
from verify_and_score_batch_remote import ArchivedPath, verify


def check_extended(bundle, results):
    plan, data = checked(bundle)
    jobs = [(a, s) for a in plan['arms'] for s in plan['training']['seeds']]
    completed = [f'{a}/seed{s}/result.json' for a,s in jobs]
    expected = {(r['edge_id'],int(r['frame'])):r['material_id']
                for r in csv_rows(data['evaluation_frames.csv'])}
    execution_sha = sha(json.dumps(plan,sort_keys=True).encode())
    records = {}
    predictions = {}
    environments = set()
    with tarfile.open(results, 'r:*') as archive:
        if not ArchivedPath(archive,'run_status').read_bytes().decode().startswith('completed '):
            raise ValueError('Cloud run is incomplete')
        root=ArchivedPath(archive,'results')
        done=json.loads((root/'done.json').read_bytes())
        if len(jobs)!=12 or not done['all_finished'] or done['completed']!=completed:
            raise ValueError('Expected all 12 blind jobs')
        for arm,seed in jobs:
            location=root/arm/f'seed{seed}'
            run=json.loads((location/'result.json').read_bytes())
            if (run['arm'],run['seed'],run['optimizer_steps'],run['execution_sha256']) != (arm,seed,710,execution_sha):
                raise ValueError(f'Changed job identity {arm}/{seed}')
            if run['package_sha256']!=plan['freeze_zip_sha256']:
                raise ValueError(f'Changed freeze at {arm}/{seed}')
            for name,checksum in run['files'].items():
                if sha((location/name).read_bytes())!=checksum:
                    raise ValueError(f'Changed artifact {name} at {arm}/{seed}')
            values=defaultdict(dict)
            identifiers={}
            for row in csv_rows((location/'blind_predictions.csv').read_bytes()):
                edge, frame = row['edge_id'], int(row['frame'])
                key=(edge,frame)
                energy=float(row['energy_eV'])
                if key in identifiers or not math.isfinite(energy):
                    raise ValueError(f'Invalid frame at {arm}/{seed}/{edge}/{frame}')
                identifiers[key]=row['material_id']
                values[edge][frame]=energy
            if identifiers!=expected:
                raise ValueError('Prediction frame identity changed')
            predictions[arm,seed]={e:barrier(p) for e,p in values.items()}
            records[arm,seed]=run
            environments.add(json.dumps(run['environment'],sort_keys=True))
    if len(environments)!=1:
        raise ValueError('Eight-path environments differ')
    return plan,data,predictions,json.loads(environments.pop()),records


def score(old_bundle,old_results,new_bundle,new_results,project,output):
    original=verify(old_bundle,old_results)
    extended=check_extended(new_bundle,new_results)
    old_plan,old_data,old_pred,old_env,_,_=original
    new_plan,new_data,new_pred,new_env,new_runs=extended
    if old_env!=new_env or old_plan['training']!=new_plan['training']:
        raise ValueError('The four-path and eight-path runtime or training recipe differs')
    if new_plan['previous_run_bundle_sha256']!=sha(old_bundle.read_bytes()):
        raise ValueError('Four-path source changed')
    for member in ('evaluation_materials.csv','evaluation_frames.csv','evaluation_geometry.xyz'):
        if old_data[member]!=new_data[member]:
            raise ValueError(f'Different held-out material files: {member}')
    truth=labels(project,old_data)
    seeds=old_plan['training']['seeds']
    arms=list(new_plan['arms'])
    baseline=np.asarray([[1000*abs(old_pred['D0',s][e]-x) for e,x in truth.items()] for s in seeds])
    def errors(pred,arm):
        return np.asarray([[1000*abs(pred[arm,s][e]-x) for e,x in truth.items()] for s in seeds])
    before={a:errors(old_pred,a) for a in arms}
    after={a:errors(new_pred,a) for a in arms}
    draws=np.random.default_rng(20261124).integers(len(truth),size=(10000,len(truth)))
    stats={}
    for arm in arms:
        inc=after[arm]-before[arm]
        stats[arm]=dict(eight_vs_four=summarize(inc,draws),
                        eight_vs_D0=summarize(after[arm]-baseline,draws),
                        four_vs_D0=summarize(before[arm]-baseline,draws),
                        eight_worse_than_four=summarize(-inc,draws)['passes_lower_error_gate'])
    by_rule={}
    for block in ('batch1','batch2'):
        a=block+'_BVSE'
        b=block+'_random'
        by_rule[block]=summarize(after[a]-after[b],draws)
    output_stream=io.StringIO(newline='')
    writer=csv.writer(output_stream)
    writer.writerow(['edge_id','seed','D0_error_meV',
                     *[f'{a}_{n}_error_meV' for a in arms for n in (4,8)]])
    for i,e in enumerate(truth):
        for j,s in enumerate(seeds):
            writer.writerow([e,s,baseline[j,i],*[x[a][j,i] for a in arms for x in (before,after)]])
    wall=[r['wall_seconds'] for r in new_runs.values()]
    report=dict(samples=len(truth),trainings=len(new_runs),gpu=new_env,
                training_and_blind_evaluation_wall_seconds=sum(wall),
                median_fit_wall_seconds=float(np.median(wall)),
                max_formal_gpu_allocation_GiB=max(r['peak_cuda_bytes'] for r in new_runs.values())/2**30,
                absolute_D0_MAE_meV=float(baseline.mean()),statistics=stats,
                BVSE_minus_random_at_eight=by_rule,
                old_four_path_archive_hash=sha(old_results.read_bytes()),
                eight_path_archive_hash=sha(new_results.read_bytes()),
                interpretation='Reuses the original 91-material panel and the paired seeds. Increasing path count at fixed steps also changes D0 exposure.',
                previously_sealed_test_labels_read=False)
    with ZipFile(output,'x',compression=ZIP_DEFLATED) as z:
        z.writestr('summary.json',json.dumps(report,indent=2,allow_nan=False)+'\n')
        z.writestr('per_material_and_seed.csv','\ufeff'+output_stream.getvalue())
    with ZipFile(output) as z:
        assert z.testzip() is None
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('four_bundle','four_results','eight_bundle','eight_results','project','output'):
        p.add_argument(n,type=Path)
    a=p.parse_args()
    score(a.four_bundle,a.four_results,a.eight_bundle,a.eight_results,a.project,a.output)
