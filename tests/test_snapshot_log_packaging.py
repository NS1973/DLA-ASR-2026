import hashlib,json,zipfile
from pathlib import Path
import pytest
from ops.snapshot_log_packaging import write_snapshot_logs


def snapshot(tmp_path,event='train_exit'):
 store=tmp_path/'cache';store.mkdir();files={};resolved={}
 for name,data in {'run/status.json':json.dumps({'event':event}).encode(),'run/metrics.jsonl':b'{"event":"train_exit"}\n','run/best_model.pt':b'model-bytes','run/best.pt':b'full-bytes','run/media/image.png':b'image'}.items():
  sha=hashlib.sha256(data).hexdigest();p=store/sha;p.write_bytes(data);files[name]={'sha256':sha,'bytes':len(data)};resolved[name]=str(p)
 proof={'all_model_full_pairs_tensor_state_verified':True,'states':[{'model':'run/best_model.pt','resume':'run/best.pt','state':{'step':7}}],'files':files,'resolved_files':resolved}
 path=tmp_path/'proof.json';path.write_text(json.dumps(proof));return path,proof


def test_closed_cached_snapshot_logs_have_logical_names_and_no_weight_blobs(tmp_path):
 path,proof=snapshot(tmp_path);out=tmp_path/'logs.zip';write_snapshot_logs(out,[path])
 with zipfile.ZipFile(out) as z:
  assert set(z.namelist())=={'run/status.json','run/metrics.jsonl','run/media/image.png','snapshot-log-manifest.json'}
  assert z.read('run/metrics.jsonl')==b'{"event":"train_exit"}\n'
  manifest=json.loads(z.read('snapshot-log-manifest.json'))
  assert manifest['all_sources_closed'] and manifest['files']['run/media/image.png']['sha256']==proof['files']['run/media/image.png']['sha256']
  assert z.testzip() is None


@pytest.mark.parametrize('damage',['active','corrupt','traversal','unverified'])
def test_log_packaging_rejects_partial_or_inconsistent_snapshot(tmp_path,damage):
 path,proof=snapshot(tmp_path,'train' if damage=='active' else 'train_exit')
 if damage=='corrupt':Path(proof['resolved_files']['run/metrics.jsonl']).write_bytes(b'corrupt')
 if damage=='traversal':proof['files']['../escape.txt']=proof['files']['run/metrics.jsonl'];proof['resolved_files']['../escape.txt']=proof['resolved_files']['run/metrics.jsonl']
 if damage=='unverified':proof['all_model_full_pairs_tensor_state_verified']=False
 path.write_text(json.dumps(proof));out=tmp_path/'logs.zip'
 with pytest.raises(ValueError):write_snapshot_logs(out,[path])
 assert not out.exists()


def test_existing_log_archive_is_preserved(tmp_path):
 path,_=snapshot(tmp_path);out=tmp_path/'logs.zip';out.write_bytes(b'original')
 with pytest.raises(FileExistsError):write_snapshot_logs(out,[path])
 assert out.read_bytes()==b'original'
