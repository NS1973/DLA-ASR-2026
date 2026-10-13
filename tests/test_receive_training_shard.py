import hashlib
import io
import tarfile

import numpy as np
import pytest
import soundfile as sf

from ops.receive_training_shard import receive_shard, verified_received
import ops.receive_training_shard as receiver_module


def payload():
    audio = io.BytesIO()
    sf.write(audio, np.zeros(1600, dtype=np.float32), 16000, format='FLAC')
    prefix = 'LibriSpeech/train-other-500/1/2/'
    files = {prefix+'1-2-0000.flac': audio.getvalue(), prefix+'1-2.trans.txt': b'1-2-0000 A\n'}
    manifest = {'archive':'shard-001.tar.gz', 'files':{name:{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for name,data in files.items()},
                'entries':[{'uid':'1-2-0000','path':prefix+'1-2-0000.flac','text':'a','duration':.1}]}
    return files, manifest


def stream(files):
    result = io.BytesIO()
    with tarfile.open(fileobj=result,mode='w') as archive:
        for name, data in files.items():
            member=tarfile.TarInfo(name);member.size=len(data);archive.addfile(member,io.BytesIO(data))
    result.seek(0)
    return result


def test_partial_transfer_preserves_verified_files_and_resumes_missing_members(tmp_path):
    files, manifest = payload()
    first = dict(list(files.items())[:1])
    with pytest.raises(ValueError, match='Incomplete'):
        receive_shard(stream(first), tmp_path, manifest)
    present = verified_received(tmp_path,manifest)
    assert set(present) == set(first)
    result = receive_shard(stream({k:v for k,v in files.items() if k not in present}),tmp_path,manifest)
    assert result['files_total'] == 2 and result['new_files_verified'] == 1
    assert result['all_members_sha_verified']


def test_receiver_rejects_leakage_corruption_and_unexpected_paths(tmp_path):
    files, manifest = payload()
    with pytest.raises(ValueError, match='excluded'):
        receive_shard(stream(files),tmp_path/'leak',manifest,excluded_ids={'1-2-0000'})
    wrong = dict(files); wrong[next(iter(wrong))] = b'bad'
    with pytest.raises(ValueError, match='size|SHA'):
        receive_shard(stream(wrong),tmp_path/'bad',manifest)
    with pytest.raises(ValueError, match='Unexpected|Unsafe'):
        receive_shard(stream({'../escape':b'bad'}),tmp_path/'safe',manifest)
    assert not (tmp_path/'escape').exists()


def test_container_reserve_is_independent_of_host_free_ram(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(receiver_module.shutil,'disk_usage',lambda path:SimpleNamespace(free=100*2**30))
    monkeypatch.setattr(receiver_module,'host_available_memory',lambda:128*2**30)
    monkeypatch.setattr(receiver_module,'container_memory_headroom',lambda:7*2**30)
    with pytest.raises(OSError,match='Container'):
        receiver_module.check_reserves(tmp_path,1024,5*2**30,32*2**30)
