import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from ops.incremental_snapshot import ingest_snapshot


def packet(path, files, include_blobs, pairs=()):
    records={name:{'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)} for name,data in files.items()}
    manifest={'files':records,'checkpoint_pairs':list(pairs)}
    with tarfile.open(path,'w:gz') as archive:
        for name,data in files.items():
            if include_blobs:
                member=tarfile.TarInfo('blobs/'+records[name]['sha256']);member.size=len(data);archive.addfile(member,io.BytesIO(data))
        data=json.dumps(manifest).encode();member=tarfile.TarInfo('manifest.json');member.size=len(data);archive.addfile(member,io.BytesIO(data))
    return records


@pytest.mark.parametrize('materialize',[True,False])
def test_cached_blobs_restore_exact_files_without_retransmission(tmp_path,materialize):
    files={'run/config.json':b'{"seed":13}','run/metrics.jsonl':b'{"step":20}\n'}
    first=tmp_path/'first.tar.gz';records=packet(first,files,True)
    store=tmp_path/'store'
    a=ingest_snapshot(first,store,tmp_path/'one')
    assert a['new_blobs']==2 and a['all_blobs_and_logical_files_sha_verified']
    second=tmp_path/'second.tar.gz';packet(second,files,False)
    b=ingest_snapshot(second,store,tmp_path/'two',materialize=materialize)
    assert b['new_blobs']==0
    assert Path(b['resolved_files']['run/metrics.jsonl']).read_bytes()==files['run/metrics.jsonl']
    digest=records['run/config.json']['sha256']
    (store/digest).write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='SHA|size'):
        ingest_snapshot(second,store,tmp_path/'corrupt')


def test_missing_reference_and_unsafe_logical_path_are_rejected(tmp_path):
    path=tmp_path/'missing.tar.gz';packet(path,{'run/status.json':b'{}'},False)
    with pytest.raises(ValueError,match='Missing'):
        ingest_snapshot(path,tmp_path/'store',tmp_path/'missing')
    path=tmp_path/'unsafe.tar.gz';packet(path,{'../outside':b'bad'},True)
    with pytest.raises(ValueError,match='Unsafe'):
        ingest_snapshot(path,tmp_path/'store',tmp_path/'unsafe')
    assert not (tmp_path/'unsafe/outside').exists()


def test_full_checkpoint_pair_is_verified_after_cached_restore(tmp_path):
    import torch
    from src.asr.model import DeepSpeech2CTC
    from src.asr.runtime import save_checkpoint,export_model
    model=DeepSpeech2CTC(n_feats=8,n_tokens=6,hidden=4,layers=1,conv_channels=2)
    optimizer=torch.optim.AdamW(model.parameters());scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1)
    state={'step':1}
    export_model(model,tmp_path/'model.pt',state)
    save_checkpoint(tmp_path/'full.pt',model,optimizer,scheduler,state,{'model':model.spec})
    files={'run/last_model.pt':(tmp_path/'model.pt').read_bytes(),'run/last.pt':(tmp_path/'full.pt').read_bytes()}
    pair={'model':'run/last_model.pt','resume':'run/last.pt','state':state}
    path=tmp_path/'pairs.tar.gz';packet(path,files,True,pairs=[pair])
    proof=ingest_snapshot(path,tmp_path/'store',tmp_path/'restored')
    assert proof['all_model_full_pairs_tensor_state_verified'] and proof['states']==[pair]
