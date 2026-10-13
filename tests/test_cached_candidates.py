import hashlib
from pathlib import Path
import pytest
import torch
from ops.cached_candidates import add_cached_candidates
from ops.final_evaluation import freeze_model


def record(tmp_path):
 p=tmp_path/'cached.pt';torch.save({'model':{'w':torch.tensor([1.])},'state':{'step':7}},p)
 return {'model_path':str(p),'model_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
  'resume_path':str(tmp_path/'deferred-full.pt'),'resume_sha256':'a'*64,'state':{'step':7},
  'source_run':str(tmp_path/'original-run'),'run':'original','snapshot_proof_sha256':'b'*64,
  'matching_model_full_tensors_verified':True}


def test_cached_model_can_win_after_original_best_last_are_overwritten(tmp_path):
 row=record(tmp_path);output=tmp_path/'out'
 sources=add_cached_candidates({},[row],output,freeze_model)
 key=next(iter(sources));candidate=sources[key]
 assert (output/'candidates'/key/'model.pt').read_bytes()==Path(row['model_path']).read_bytes()
 assert candidate['sha256']==row['model_sha256'] and candidate['resume_source']==row['resume_path']
 assert candidate['resume_sha256']==row['resume_sha256'] and not Path(row['resume_path']).exists()
 assert candidate['source']==row['source_run']


def test_cached_alias_is_deduplicated_against_current_best(tmp_path):
 row=record(tmp_path);existing={'current':{'sha256':row['model_sha256'],'aliases':[]}}
 result=add_cached_candidates(existing,[row],tmp_path/'out',freeze_model)
 assert list(result)==['current'] and len(result['current']['aliases'])==1


@pytest.mark.parametrize('damage',['model_sha','state','full_sha','verified'])
def test_cached_candidate_rejects_inconsistent_reserve_before_selection(tmp_path,damage):
 row=record(tmp_path)
 if damage=='model_sha':row['model_sha256']='0'*64
 if damage=='state':row['state']={'step':8}
 if damage=='full_sha':row['resume_sha256']='not-a-sha'
 if damage=='verified':row['matching_model_full_tensors_verified']=False
 with pytest.raises(ValueError):add_cached_candidates({},[row],tmp_path/'out',freeze_model)
