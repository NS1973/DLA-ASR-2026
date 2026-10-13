"""Compact local and remote evidence for the ten-minute heartbeat."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
for path in sorted((root/'data/downloads').glob('*.partial')):
    print('DOWNLOADING',path.name,round(path.stat().st_size/2**30,2),'GiB',flush=True)
for log in ['data-download','data-transfer','dev-transfer','staging-transfer','train360-download-parallel','train360-transfer-stream']:
    path=root/'logs'/f'{log}.log'
    if path.exists():
        print(log, '\n'.join(path.read_text(errors='replace').splitlines()[-2:]),flush=True)
command="""python3 - <<'PY'
import pathlib,json
root=pathlib.Path('/home/protei_ii/dla-asr-2026')
print('READY_DATA',[p.name for p in (root/'data').glob('*.ready')])
for run in (root/'runs').iterdir():
 p=run/'metrics.jsonl'
 if not p.exists():continue
 lines=p.read_text().splitlines()
 events=[]
 for s in lines:
  try:events.append(json.loads(s))
  except json.JSONDecodeError:continue
 latest=next((e for e in reversed(events) if e.get('event') in ['train','validation','overfit_eval','train_exit']),{})
 print('RUN',run.name,json.dumps({k:v for k,v in latest.items() if k not in ['examples','samples']}))
 if run.name in ['main','no_aug','finetune460']:
  for split in ['dev-clean','dev-other']:
   valid=next((e for e in reversed(events) if e.get('event')=='validation' and e.get('split')==split),None)
   if valid:print('VALID',run.name,json.dumps({k:v for k,v in valid.items() if k!='examples'}))
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
df -h /home"""
subprocess.run([sys.executable,str(root/'ops/gmc.py'),'exec',command,'--timeout','30'],check=True)
