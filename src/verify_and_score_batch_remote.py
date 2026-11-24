"""Verify remote training records and score the frozen pilot on the local DFT archive."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from tarfile import open as open_tar
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from score_batch_pilot import checked, collect_predictions, labels, summarize


class ArchivedPath:
    def __init__(self, archive, path):
        self.archive = archive
        self.path = PurePosixPath(path)

    def __truediv__(self, member):
        return ArchivedPath(self.archive, self.path / member)

    @property
    def parent(self):
        return ArchivedPath(self.archive, self.path.parent)

    @property
    def name(self):
        return self.path.name

    def read_bytes(self):
        member = self.archive.getmember(str(self.path))
        if not member.isfile():
            raise ValueError(f"Expected a file at {self.path}")
        with self.archive.extractfile(member) as stream:
            return stream.read()


def verify(bundle, archive_path):
    plan, data = checked(bundle)
    arms = plan['arms']
    seeds = plan['training']['seeds']
    with open_tar(archive_path, 'r:*') as archive:
        root = ArchivedPath(archive, 'batch_pilot_results')
        predictions, environment = collect_predictions(root, plan, data)
        result = {}
        for arm in arms:
            for seed in seeds:
                job = root / arm / f'seed{seed}'
                record = json.loads((job/'result.json').read_bytes())
                for name, checksum in record['files'].items():
                    if hashlib.sha256((job/name).read_bytes()).hexdigest()!=checksum:
                        raise ValueError(f'Hash mismatch for {arm}/{seed}/{name}')
                result[arm,seed] = record
        status = ArchivedPath(archive,'run_status').read_bytes().decode().strip()
        if not status.startswith('completed '):
            raise ValueError(f'Cloud did not report successful completion: {status}')
    return plan,data,predictions,environment,result,status


def score(result, project, output):
    plan,data,predictions,environment,timing,status=result
    truth=labels(project,data)
    seeds=plan['training']['seeds']
    arms=plan['arms']
    errors={a:np.array([[1000*abs(predictions[a,s][edge]-ref) for edge,ref in truth.items()]
                        for s in seeds]) for a in arms}
    draw=np.random.default_rng(20261124).integers(len(truth),size=(10000,len(truth)))
    metrics={}
    for block in ('batch1','batch2'):
        random,bvse=(errors[f'{block}_{mode}'] for mode in ('random','BVSE'))
        metrics[block]=dict(random_vs_D0=summarize(random-errors['D0'],draw),
                            BVSE_vs_D0=summarize(bvse-errors['D0'],draw),
                            BVSE_vs_random=summarize(bvse-random,draw))
    repeated={k:all(group[k]['passes_lower_error_gate'] for group in metrics.values())
              for k in metrics['batch1']}
    table=io.StringIO(newline='')
    writer=csv.writer(table)
    writer.writerow(['edge_id','seed','reference_barrier_eV',
                     *[f'{a}_absolute_error_meV' for a in arms],
                     *[f'{a}_predicted_barrier_eV' for a in arms]])
    for i,(e,value) in enumerate(truth.items()):
        for j,seed in enumerate(seeds):
            writer.writerow([e,seed,value,*[errors[a][j,i] for a in arms],
                             *[predictions[a,seed][e] for a in arms]])
    timings=[row['wall_seconds'] for row in timing.values()]
    summary=dict(metrics=metrics,repeated=repeated,rule_supported=bool(repeated['BVSE_vs_D0'] and
                 repeated['BVSE_vs_random']),cloud_completion_utc=status.split()[1],
                 inference_and_training_wall_seconds=sum(timings),
                 fit_wall_median_seconds=float(np.median(timings)),
                 max_training_peak_gib=max(r['peak_cuda_bytes'] for r in timing.values())/2**30,
                 models=len(timing),materials=len(truth),environment=environment,
                 test_previously_sealed_labels_read=False,
                 arm_MAE_meV={a:float(x.mean()) for a,x in errors.items()})
    with ZipFile(output,'x',compression=ZIP_DEFLATED) as z:
        z.writestr('summary.json',json.dumps(summary,indent=2,allow_nan=False)+'\n')
        z.writestr('per_material_and_seed.csv','\ufeff'+table.getvalue())
    with ZipFile(output) as z:
        assert z.testzip() is None
    print(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for field in ('bundle','remote_archive','project','output'):
        cli.add_argument(field,type=Path)
    cli.add_argument('--verify-only',action='store_true')
    args=cli.parse_args()
    result=verify(args.bundle,args.remote_archive)
    if not args.verify_only:
        score(result,args.project,args.output)
    else:
        print(f'Verified and copied all {len(result[4])} fits, including predictions and weights.')
