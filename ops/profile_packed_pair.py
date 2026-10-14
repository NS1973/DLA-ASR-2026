"""Alternating same-batch packed-stack probe while the independent trainers continue."""
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.asr.runtime import load_model,make_loader,seed_all
from ops.build_mixed_manifest import load_manifest

p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
args=p.parse_args();cfg=json.loads(Path(args.config).read_text());seed_all(cfg['seed']);torch.set_num_threads(2)
entries=load_manifest(cfg['training_manifest_path'],cfg['data_root'],cfg['training_manifest_sha256'])
ordered=sorted(entries,key=lambda row:row['duration']);count=min(12000,len(ordered))
entries=[ordered[i*len(ordered)//count] for i in range(count)]
cfg.update(batch_size=256,workers=4)
loader=make_loader(entries,cfg,augment=True,shuffle=True);iterator=iter(loader)
models={mode:load_model(cfg['init_checkpoint'],torch.device('cuda')).train() for mode in ['legacy','packed']}
models['packed'].packed_stack=True
optimizers={mode:torch.optim.AdamW(model.parameters(),lr=cfg['lr']) for mode,model in models.items()}
loss_fn=torch.nn.CTCLoss(blank=0,zero_infinity=False);records=[]
for i in range(120):
 start=time.perf_counter()
 try:batch=next(iterator)
 except StopIteration:iterator=iter(loader);batch=next(iterator)
 spec=batch['spectrogram'].to('cuda',non_blocking=True);targets=batch['targets'].to('cuda',non_blocking=True)
 torch.cuda.synchronize();data_seconds=time.perf_counter()-start
 order=['legacy','packed'] if i%2==0 else ['packed','legacy']
 for mode in order:
  start=time.perf_counter();optimizer=optimizers[mode];optimizer.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):out=models[mode](spec,batch['spectrogram_length'])
  loss=loss_fn(out['log_probs'].transpose(0,1),targets,out['log_probs_length'],batch['target_lengths'])
  loss.backward();norm=torch.nn.utils.clip_grad_norm_(models[mode].parameters(),5)
  assert torch.isfinite(loss) and torch.isfinite(norm)
  optimizer.step();torch.cuda.synchronize();seconds=time.perf_counter()-start
  if i>=20:records.append({'mode':mode,'compute_seconds':seconds,'data_seconds':data_seconds,
     'audio_seconds':float(batch['spectrogram_length'].sum())*.01,'loss':float(loss.detach()),'gradient_norm':float(norm)})
summary={}
for mode in models:
 rows=[r for r in records if r['mode']==mode]
 summary[mode]={'steps':len(rows),'audio_seconds_per_second':sum(r['audio_seconds'] for r in rows)/sum(r['compute_seconds']+r['data_seconds'] for r in rows)}
summary['measured_gain_fraction']=summary['packed']['audio_seconds_per_second']/summary['legacy']['audio_seconds_per_second']-1
result={'summary':summary,'records':records,'co_loaded_with_training':True,
        'comparison':'alternating order, identical real augmented batch, separate parent-identical models; includes observed data wait',
        'promotion':'requires review of co-load uncertainty and full compatibility tests; no automatic trainer changes'}
Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(summary),flush=True)
