import io,json,tarfile
from pathlib import Path
import pytest,torch
from scripts import preserve_runs


def test_capture_keeps_all_files_deduplicates_and_verifies_checkpoint_pairs(tmp_path):
    runs=tmp_path/'runs';run=runs/'main';run.mkdir(parents=True)
    model={'model':{'weight':torch.ones(3)},'state':{'step':7},'model_spec':{}}
    torch.save(model,run/'best_model.pt');torch.save({**model,'optimizer':{},'rng_torch':torch.get_rng_state()},run/'best.pt')
    (run/'metrics.jsonl').write_bytes(b'{"event":"train","step":7}\n{"partial":')
    (run/'media').mkdir();(run/'media/a.wav').write_bytes(b'audio bytes')
    (run/'duplicate.wav').write_bytes(b'audio bytes')
    result=preserve_runs.capture(runs,tmp_path/'snapshot',parts=2,reserve_bytes=0)
    assert result['checkpoint_pairs'][0]['verified'] is True
    assert len(result['files'])==5 and result['unique_blobs']==4
    record=result['files']['main/metrics.jsonl']
    with tarfile.open(tmp_path/'snapshot'/record['archive']) as archive:
        assert archive.extractfile('blobs/'+record['sha256']).read()==b'{"event":"train","step":7}\n'
    assert (tmp_path/'snapshot/complete.json').exists()


def test_mismatched_checkpoint_is_not_declared_preserved(tmp_path):
    runs=tmp_path/'runs';run=runs/'main';run.mkdir(parents=True)
    torch.save({'model':{'weight':torch.ones(2)},'state':{'step':1}},run/'best_model.pt')
    torch.save({'model':{'weight':torch.zeros(2)},'state':{'step':1}},run/'best.pt')
    with pytest.raises(ValueError,match='matching'):
        preserve_runs.capture(runs,tmp_path/'snapshot',parts=2,reserve_bytes=0)
    assert not (tmp_path/'snapshot/complete.json').exists()


def test_existing_snapshot_cannot_be_overwritten(tmp_path):
    (tmp_path/'runs').mkdir()
    existing=tmp_path/'snapshot';existing.mkdir();(existing/'precious').write_bytes(b'keep')
    with pytest.raises(FileExistsError):
        preserve_runs.capture(tmp_path/'runs',existing,parts=2,reserve_bytes=0)
    assert (existing/'precious').read_bytes()==b'keep'
