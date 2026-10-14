from pathlib import Path
import json
import pytest
from ops.queue_guards import process_matches_launch, projected_scratch_quota, validate_post_exit_capacity, stable_registry_paths

GIB=2**30

def fake_process(root,pid,config,state='S',ticks=123,foreign=False):
 p=root/str(pid);p.mkdir(parents=True);fields=[state]+['0']*18+[str(ticks)];(p/'stat').write_text(f'{pid} (python) '+ ' '.join(fields));(p/'cmdline').write_bytes((b'/bin/sleep\x0060\x00') if foreign else (b'python\x00-c\x00train\x00'+str(config).encode()+b'\x00'));(p/'ns').mkdir();(p/'ns/pid').write_text('namespace');return p

def test_reused_foreign_pid_does_not_keep_closed_stage_alive(tmp_path):
 cfg=tmp_path/'run/launch_config.json';proc=tmp_path/'proc';fake_process(proc,7,cfg,foreign=True)
 assert not process_matches_launch(7,cfg,proc)

def test_owned_live_process_requires_exact_launch_arg_and_identity(tmp_path):
 cfg=tmp_path/'run/launch_config.json';proc=tmp_path/'proc';p=fake_process(proc,7,cfg);namespace=(p/'ns/pid').stat().st_ino
 assert process_matches_launch(7,cfg,proc,expected_start_ticks=123,expected_namespace=namespace)
 assert not process_matches_launch(7,cfg,proc,expected_start_ticks=124,expected_namespace=namespace)
 assert not process_matches_launch(7,cfg,proc,expected_start_ticks=123,expected_namespace=namespace+1)
 assert not process_matches_launch(7,Path(str(cfg)+'-other'),proc)

def test_zombie_or_missing_process_is_terminal(tmp_path):
 cfg=tmp_path/'run/launch_config.json';proc=tmp_path/'proc';fake_process(proc,7,cfg,state='Z')
 assert not process_matches_launch(7,cfg,proc)
 assert not process_matches_launch(8,cfg,proc)

def test_final_storage_projection_expands_quota_instead_of_waiting_for_full(tmp_path):
 required=int((3.008778+2.553+0.365+2+5)*GIB)
 assert projected_scratch_quota(56*GIB,int(6.60*GIB),required)==64*GIB
 assert projected_scratch_quota(56*GIB,16*GIB,required)==56*GIB
 with pytest.raises(ValueError):projected_scratch_quota(56*GIB,GIB,50*GIB)

def test_resize_requires_real_exit_and_preserves_both_memory_reserves():
 assert validate_post_exit_capacity(True,43*GIB,30*GIB,8*GIB) is None
 for terminal,available,headroom in [(False,43*GIB,30*GIB),(True,30*GIB,30*GIB),(True,43*GIB,14*GIB)]:
  with pytest.raises(ValueError):validate_post_exit_capacity(terminal,available,headroom,8*GIB)

def test_registry_waits_for_late_writer_and_stable_inventory(tmp_path):
 old=tmp_path/'old';old.mkdir();(old/'verified.json').write_text('{}');state={'alive':True,'slept':0}
 def sleep(_):
  state['slept']+=1
  if state['alive']:
   new=tmp_path/'late';new.mkdir();(new/'verified.json').write_text('{}');state['alive']=False
 paths=stable_registry_paths(tmp_path,lambda:state['alive'],sleep=sleep)
 assert {p.parent.name for p in paths}=={'old','late'} and state['slept']>=2

def test_registry_rescans_if_inventory_changes_after_writer_exit(tmp_path):
 a=tmp_path/'a';a.mkdir();(a/'verified.json').write_text('{}');calls=[]
 def sleep(_):
  calls.append(1)
  if len(calls)==1:
   b=tmp_path/'b';b.mkdir();(b/'verified.json').write_text('{}')
 assert {p.parent.name for p in stable_registry_paths(tmp_path,lambda:False,sleep=sleep)}=={'a','b'}
 assert len(calls)==2
