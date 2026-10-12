"""Detached bounded-retry supervisor; touches only this project's child jobs."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/work')
CODE = ROOT / 'code'
PYTHON = '/work/env/bin/python'
RUNS = ROOT / 'runs'
RUNS.mkdir(exist_ok=True)


def run(name, gpu, overrides):
    batch = 4 if name == 'overfit' else 32
    accumulation = 1
    for attempt in range(5):
        directory = RUNS / name
        directory.mkdir(exist_ok=True)
        options = [f'run_dir={directory}', f'batch_size={batch}', f'accumulation={accumulation}', *overrides]
        if attempt and (directory / 'last.pt').exists():
            options.append(f'resume={directory}/last.pt')
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'WANDB_MODE': 'offline', 'PYTHONUNBUFFERED': '1'}
        with (directory / f'console-{attempt}.log').open('a') as log:
            process = subprocess.Popen([PYTHON, '-u', 'train.py', *options], cwd=CODE, env=environment, stdout=log, stderr=subprocess.STDOUT)
            (directory / 'process.json').write_text(json.dumps({'pid':process.pid,'gpu':gpu,'attempt':attempt,'command':process.args}))
            result = process.wait()
        print(f'{name}: attempt={attempt} exit={result}', flush=True)
        if result == 0:
            return True
        text = (directory / f'console-{attempt}.log').read_text(errors='replace')
        if 'out of memory' in text.lower():
            batch = max(1, batch // 2)
            accumulation *= 2
        elif 'CUDNN_STATUS_NOT_SUPPORTED' in text or 'CUDNN_STATUS_BAD_PARAM' in text:
            overrides = [x for x in overrides if not x.startswith('bf16=')] + ['bf16=false']
        else:
            time.sleep(20)
    return False


def wait_for(path):
    while not path.exists():
        if datetime.now(timezone.utc).isoformat() >= '2026-10-13T09:30:00':
            raise TimeoutError(f'Data not available before training cutoff: {path}')
        time.sleep(10)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'main':
        wait_for(ROOT / 'data/bootstrap.ready')
        ok = run('overfit', 0, ['custom_train=/work/data/bootstrap','overfit=true','augment=false','valid_parts=[]',
                                   'epochs=2000','max_steps=1500','warmup_steps=10','workers=0','log_every=20', 'checkpoint_seconds=300'])
        (RUNS / 'overfit_check.json').write_text(json.dumps({'process_succeeded':ok}))
        wait_for(ROOT / 'data/train-clean-100.ready')
        wait_for(ROOT / 'data/dev-clean.ready')
        wait_for(ROOT / 'data/dev-other.ready')
        run('main', 0, ['data_root=/work/data/datasets'])
    elif mode == 'ablation':
        for part in ['train-clean-100','dev-clean','dev-other']:
            wait_for(ROOT / f'data/{part}.ready')
        run('no_aug', 1, ['data_root=/work/data/datasets','augment=false','epochs=10','end_utc=2026-10-13T04:00:00+00:00'])
    else:
        raise ValueError(mode)
