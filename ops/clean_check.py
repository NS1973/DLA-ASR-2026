"""Run inside the project container: no inherited site packages, no network."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

root = Path('/work')
logdir = root/'runs/clean-env'
logdir.mkdir(parents=True, exist_ok=True)
archive = root/'incoming/clean-wheels.tar.gz'
deadline = time.monotonic()+900
while not archive.exists():
    if time.monotonic() > deadline:
        raise TimeoutError('Clean environment wheel transfer did not finish')
    time.sleep(5)
with tarfile.open(archive) as source:
    source.extractall(root, filter='data')
subprocess.run([sys.executable,'-m','venv',str(root/'clean-env')],check=True)
python = str(root/'clean-env/bin/python')
commands = [
    [python,'-m','pip','install','--no-index','--find-links=/work/clean-wheels','-r','requirements.txt'],
    [python,'-m','pip','check'],
    [python,'-m','pytest','tests','-q'],
    [python,'inference.py','checkpoint=/work/runs/overfit/last_model.pt',
     'custom_dir=/work/data/bootstrap','limit=4','workers=0','device=cpu','cpu_threads=4',
     'batch_size=2',f'output_dir={logdir}/predictions'],
]
environment = {**os.environ,'WANDB_MODE':'offline','PYTHONUNBUFFERED':'1'}
status = {'commands':[], 'complete':False}
try:
    with (logdir/'console.log').open('a') as output:
        for command in commands:
            output.write(json.dumps(command)+'\n'); output.flush()
            result = subprocess.run(command,cwd=root/'code',env=environment,stdout=output,stderr=subprocess.STDOUT)
            status['commands'].append({'command':command,'returncode':result.returncode})
            (logdir/'status.json').write_text(json.dumps(status,indent=2))
            if result.returncode:
                raise RuntimeError(f'Clean installation check failed: {command[1:3]}')
    freeze = subprocess.check_output([python,'-m','pip','freeze'],text=True)
    (logdir/'requirements-lock-cpu.txt').write_text(freeze)
    status['complete'] = True
finally:
    (logdir/'status.json').write_text(json.dumps(status,indent=2))
