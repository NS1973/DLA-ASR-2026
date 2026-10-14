import copy,hashlib,json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import torch
from ops.build_mixed_manifest import load_manifest


def fixture_manifest(tmp_path):
 root=tmp_path/'data';root.mkdir();wave=(.08*np.sin(np.arange(32000)*.013)).astype('float32')
 for name in ['a','b']:sf.write(root/(name+'.wav'),wave,16000)
 a={'uid':'a','path':'a.wav','text':'hello','duration':2.};b={'uid':'b','path':'b.wav','text':'test','duration':2.}
 path=tmp_path/'manifest.json';path.write_text(json.dumps({'schema_version':1,'entries':[a,b,a,b,a,b,a,b,a,b]}));return root,path,hashlib.sha256(path.read_bytes()).hexdigest()


def test_interning_preserves_every_exposure_value_order_and_source_sha(tmp_path):
 root,path,sha=fixture_manifest(tmp_path);old=load_manifest(path,root,sha);new=load_manifest(path,root,sha,intern_entries=True)
 assert new==old and len(new)==10 and len({id(x) for x in new})==2
 assert new[0] is new[2] and old[0] is not old[2]
 assert json.dumps(new)==json.dumps(old) and hashlib.sha256(path.read_bytes()).hexdigest()==sha


def test_same_uid_with_different_metadata_is_not_silently_merged(tmp_path):
 root,path,_=fixture_manifest(tmp_path);doc=json.loads(path.read_text());doc['entries'][2]['text']='changed';path.write_text(json.dumps(doc));sha=hashlib.sha256(path.read_bytes()).hexdigest()
 old=load_manifest(path,root,sha);new=load_manifest(path,root,sha,intern_entries=True)
 assert new==old and new[0] is not new[2]


@pytest.mark.parametrize('damage',['sha','traversal'])
def test_interning_keeps_checksum_and_relative_path_guards(tmp_path,damage):
 root,path,sha=fixture_manifest(tmp_path)
 if damage=='sha':sha='0'*64
 else:
  doc=json.loads(path.read_text());doc['entries'][2]['path']='../escape.wav';path.write_text(json.dumps(doc));sha=hashlib.sha256(path.read_bytes()).hexdigest()
 with pytest.raises(ValueError):load_manifest(path,root,sha,intern_entries=True)


def test_sampler_and_index_seeded_augmentations_are_exact_after_interning(tmp_path):
 from src.asr.data import SpeechDataset,LengthBatchSampler
 root,path,sha=fixture_manifest(tmp_path);old=load_manifest(path,root,sha);new=load_manifest(path,root,sha,intern_entries=True)
 a=LengthBatchSampler(old,2,19);b=LengthBatchSampler(new,2,19);a.epoch=b.epoch=3100;assert list(a)==list(b)
 x=SpeechDataset(old,True,19,'effective_v2');y=SpeechDataset(new,True,19,'effective_v2');x.epoch[0]=y.epoch[0]=3100
 for i in range(len(old)):
  left,right=x[i],y[i]
  assert left['uid']==right['uid']
  for key in ['spectrogram','waveform','encoded']:assert torch.equal(left[key],right[key])


def test_real_packed_spec_training_and_full_resume_match_legacy_entries(tmp_path,monkeypatch):
 from src.asr import runtime
 from omegaconf import OmegaConf
 root,path,sha=fixture_manifest(tmp_path)
 class Quiet:
  wandb=None
  def __init__(self,*args):pass
  def emit(self,*args):pass
 monkeypatch.setattr(runtime,'RunLog',Quiet);monkeypatch.setattr(runtime,'log_media',lambda *a,**kw:None)
 original_loader=runtime.make_loader;unique_counts=[];fetch_counts=[]
 original_getitem=runtime.SpeechDataset.__getitem__
 def count_fetch(self,index):
  fetch_counts[-1]+=1
  return original_getitem(self,index)
 monkeypatch.setattr(runtime.SpeechDataset,'__getitem__',count_fetch)
 def capture_loader(entries,*args,**kwargs):
  unique_counts.append(len({id(row) for row in entries}))
  fetch_counts.append(0)
  return original_loader(entries,*args,**kwargs)
 monkeypatch.setattr(runtime,'make_loader',capture_loader)
 cfg=OmegaConf.to_container(OmegaConf.load('src/configs/asr_train.yaml'),resolve=True)
 cfg.update(device='cpu',run_dir=str(tmp_path/'baseline'),data_root=str(root),training_manifest_path=str(path),training_manifest_sha256=sha,workers=0,cpu_threads=1,valid_parts=[],batch_size=2,epochs=1,max_steps=2,end_utc=None,bf16=False,packed_stack=True,specaugment=True,augmentation_profile='effective_v2',data_epoch_offset=3100,seed=19,skip_resumed_audio=True)
 cfg['model'].update(hidden=8,layers=1,conv_channels=2)
 runtime.train(copy.deepcopy(cfg));expected=torch.load(tmp_path/'baseline/last.pt',weights_only=False)
 first={**copy.deepcopy(cfg),'run_dir':str(tmp_path/'interrupted'),'intern_manifest_entries':True,'max_steps':1};runtime.train(first)
 resumed={**copy.deepcopy(cfg),'run_dir':str(tmp_path/'resumed'),'intern_manifest_entries':True,'resume':str(tmp_path/'interrupted/last.pt')};runtime.train(resumed)
 actual=torch.load(tmp_path/'resumed/last.pt',weights_only=False)
 for name,value in expected['model'].items():assert torch.equal(value,actual['model'][name])
 assert expected['state']==actual['state'] and expected['scheduler']==actual['scheduler']
 for key,values in expected['optimizer']['state'].items():
  for name,value in values.items():assert torch.equal(value,actual['optimizer']['state'][key][name])
 assert actual['config']['intern_manifest_entries']
 assert unique_counts==[10,2,2]
 assert fetch_counts==[4,2,2]


def test_equal_values_with_different_field_order_keep_serialized_recipe_identical(tmp_path):
 root,path,_=fixture_manifest(tmp_path);doc=json.loads(path.read_text());doc['entries'][2]=dict(reversed(list(doc['entries'][2].items())));path.write_text(json.dumps(doc));sha=hashlib.sha256(path.read_bytes()).hexdigest()
 old=load_manifest(path,root,sha);new=load_manifest(path,root,sha,intern_entries=True)
 assert json.dumps(new)==json.dumps(old)


def test_default_training_recipe_has_no_expired_campaign_cutoff(tmp_path,monkeypatch):
 from src.asr import runtime
 from omegaconf import OmegaConf
 root,path,sha=fixture_manifest(tmp_path)
 class Quiet:
  wandb=None
  def __init__(self,*args):pass
  def emit(self,*args):pass
 monkeypatch.setattr(runtime,'RunLog',Quiet);monkeypatch.setattr(runtime,'log_media',lambda *a,**kw:None)
 cfg=OmegaConf.to_container(OmegaConf.load('src/configs/asr_train.yaml'),resolve=True)
 cfg.update(device='cpu',run_dir=str(tmp_path/'run'),data_root=str(root),training_manifest_path=str(path),training_manifest_sha256=sha,workers=0,cpu_threads=1,valid_parts=[],batch_size=2,epochs=1,max_steps=1,bf16=False)
 cfg['model'].update(hidden=8,layers=1,conv_channels=2)
 runtime.train(cfg)
 assert torch.load(tmp_path/'run/last.pt',weights_only=False)['state']['step']==1


def test_training_manifest_export_streams_without_large_intermediate_string(tmp_path,monkeypatch):
 from src.asr import runtime
 from omegaconf import OmegaConf
 root,path,sha=fixture_manifest(tmp_path)
 original=runtime.json.dumps
 def bounded_dumps(value,*args,**kwargs):
  if isinstance(value,list) and value and isinstance(value[0],dict) and 'uid' in value[0]:
   raise AssertionError('training exposure manifest must be streamed')
  return original(value,*args,**kwargs)
 class Quiet:
  wandb=None
  def __init__(self,*args):pass
  def emit(self,*args):pass
 monkeypatch.setattr(runtime.json,'dumps',bounded_dumps)
 monkeypatch.setattr(runtime,'RunLog',Quiet);monkeypatch.setattr(runtime,'log_media',lambda *a,**kw:None)
 cfg=OmegaConf.to_container(OmegaConf.load('src/configs/asr_train.yaml'),resolve=True)
 cfg.update(device='cpu',run_dir=str(tmp_path/'run'),data_root=str(root),training_manifest_path=str(path),training_manifest_sha256=sha,workers=0,cpu_threads=1,valid_parts=[],batch_size=2,epochs=1,max_steps=1,bf16=False)
 cfg['model'].update(hidden=8,layers=1,conv_channels=2)
 runtime.train(cfg)
 assert (tmp_path/'run/train_manifest.json').read_text()==original(load_manifest(path,root,sha),indent=2)


def test_prefetch_factor_is_configurable_and_omitted_for_zero_workers(tmp_path):
 from src.asr.runtime import make_loader
 root,path,sha=fixture_manifest(tmp_path);entries=load_manifest(path,root,sha)
 cfg=dict(batch_size=2,seed=1,workers=2,prefetch_factor=1)
 loader=make_loader(entries,cfg)
 assert loader.prefetch_factor==1
 cfg['workers']=0
 assert make_loader(entries,cfg).prefetch_factor is None


def test_validation_loader_can_release_workers_without_changing_training_loader(tmp_path):
 from src.asr.runtime import make_loader,make_validation_loader
 root,path,sha=fixture_manifest(tmp_path);entries=load_manifest(path,root,sha)
 cfg=dict(batch_size=2,seed=1,workers=8,prefetch_factor=1,validation_workers=0)
 train=make_loader(entries,cfg,augment=True,shuffle=True)
 valid=make_validation_loader(entries,cfg)
 assert train.num_workers==8 and train.persistent_workers
 assert valid.num_workers==0 and not valid.persistent_workers
 assert cfg['workers']==8


def test_deadline_reached_during_batch_fetch_prevents_optimizer_step(tmp_path,monkeypatch):
 from src.asr import runtime
 from omegaconf import OmegaConf
 from datetime import datetime,timezone
 root,path,sha=fixture_manifest(tmp_path);clock=[0.]
 deadline=datetime(2099,1,1,tzinfo=timezone.utc).timestamp()
 original=runtime.SpeechDataset.__getitem__
 def expire_during_fetch(self,index):
  value=original(self,index);clock[0]=deadline+1;return value
 monkeypatch.setattr(runtime.SpeechDataset,'__getitem__',expire_during_fetch)
 monkeypatch.setattr(runtime.time,'time',lambda:clock[0])
 class Quiet:
  wandb=None
  def __init__(self,*args):pass
  def emit(self,*args):pass
 monkeypatch.setattr(runtime,'RunLog',Quiet);monkeypatch.setattr(runtime,'log_media',lambda *a,**kw:None)
 cfg=OmegaConf.to_container(OmegaConf.load('src/configs/asr_train.yaml'),resolve=True)
 cfg.update(device='cpu',run_dir=str(tmp_path/'run'),data_root=str(root),training_manifest_path=str(path),training_manifest_sha256=sha,workers=0,cpu_threads=1,valid_parts=[],batch_size=2,epochs=1,max_steps=1,bf16=False,end_utc='2099-01-01T00:00:00+00:00')
 cfg['model'].update(hidden=8,layers=1,conv_channels=2)
 runtime.train(cfg)
 assert torch.load(tmp_path/'run/last.pt',weights_only=False)['state']['step']==0
