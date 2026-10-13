"""One bounded opt-in augmentation experiment; existing trainers keep running."""
import json
import hashlib
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops.supervisor import RUNS, CODE, END, best_checkpoint, run


def main():
    import fcntl
    name = 'lr_candidate_augmented460'
    directory = RUNS/name
    directory.mkdir(exist_ok=True)
    with (directory/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        completion = directory/'controller-complete.json'
        if completion.exists() and json.loads(completion.read_text()).get('process_succeeded') is True:
            print('Experiment already completed; preserving its artifacts.',flush=True)
            return
        parent = best_checkpoint()
        options = ['data_root=/work/data/datasets','train_parts=[train-clean-100,train-clean-360]',
                   f'init_checkpoint={parent}','batch_size=128','epochs=40','lr=0.00005',
                   'warmup_steps=100','augmentation_profile=effective_v2','data_epoch_offset=80',
                   f'end_utc={END}']
        record = {'controller_pid':os.getpid(),'gpu':0,'options':options,
                  'source_commit':os.environ.get('DLA_SOURCE_COMMIT'),
                  'source_files_sha256':{str(path):hashlib.sha256((CODE/path).read_bytes()).hexdigest()
                                        for path in map(Path,['train.py','src/asr/data.py','src/asr/runtime.py',
                                                              'src/asr/model.py','src/configs/asr_train.yaml'])},
                  'purpose':'Audit correction candidate; concurrent with preserved baseline; select only on dev'}
        (directory/'operator.json').write_text(json.dumps(record,indent=2))
        succeeded = run(name,0,options)
        (directory/'controller-complete.json').write_text(json.dumps({'process_succeeded':succeeded}))
        if not succeeded:
            raise RuntimeError('Augmentation experiment exhausted bounded retries; baseline preserved')


if __name__ == '__main__':
    main()
