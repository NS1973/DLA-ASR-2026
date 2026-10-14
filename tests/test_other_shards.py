import hashlib
import io
import json
import tarfile

import numpy as np
import pytest
import soundfile as sf

from ops.shard_other_archive import shard_archive


def fixture_archive(tmp_path, bad_text=False):
    source = tmp_path / 'source.tar.gz'
    audio = io.BytesIO()
    sf.write(audio, np.zeros(1600, dtype=np.float32), 16000, format='FLAC')
    files = {}
    with tarfile.open(source, 'w:gz') as archive:
        for chapter in [2, 3]:
            texts = []
            for i in range(2):
                uid = f'1-{chapter}-{i:04}'
                name = f'LibriSpeech/train-other-500/1/{chapter}/{uid}.flac'
                data = audio.getvalue()
                member = tarfile.TarInfo(name); member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
                files[uid] = {'sha256': hashlib.sha256(data).hexdigest(), 'duration': .1, 'source_member': name}
                texts.append(uid + (' THIS TARGET IS TOO LONG' if bad_text else ' A'))
            data = ('\n'.join(texts) + '\n').encode()
            member = tarfile.TarInfo(f'LibriSpeech/train-other-500/1/{chapter}/1-{chapter}.trans.txt');member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return source, files


def test_shards_keep_whole_chapters_decode_audio_and_omit_identical_existing_data(tmp_path):
    source, files = fixture_archive(tmp_path)
    known = {k:v for k,v in files.items() if k.startswith('1-2-')}
    result = shard_archive(source, tmp_path / 'shards', target_hours=.00004, known_files=known)
    assert result['unique_utterances'] == 4
    assert result['existing_utterances_reused'] == 2
    assert len(result['shards']) == 1
    proof = result['shards'][0]
    assert proof['unique_utterances'] == 2
    path = tmp_path / 'shards' / proof['archive']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == proof['sha256']
    with tarfile.open(path) as archive:
        names = archive.getnames()
    assert len(names) == 3 and all('/1/3/' in name for name in names)
    manifest = json.loads(path.with_suffix('.manifest.json').read_text())
    assert {row['uid'] for row in manifest['entries']} == {'1-3-0000', '1-3-0001'}
    assert result['unique_hours'] == .4 / 3600


def test_shards_reject_excluded_ids_and_infeasible_ctc(tmp_path):
    source, _ = fixture_archive(tmp_path)
    with pytest.raises(ValueError, match='excluded'):
        shard_archive(source, tmp_path / 'leak', excluded_ids={'1-2-0000'})
    other = tmp_path / 'bad';other.mkdir()
    source, _ = fixture_archive(other, bad_text=True)
    with pytest.raises(ValueError, match='CTC'):
        shard_archive(source, tmp_path / 'ctc')


def test_explicit_ctc_exclusions_preserve_original_audio_and_transcripts(tmp_path):
    source, files = fixture_archive(tmp_path,bad_text=True)
    original=hashlib.sha256(source.read_bytes()).hexdigest()
    result=shard_archive(source,tmp_path/'filtered',allow_ctc_exclusions=True,shard_index_offset=1)
    assert result['unique_utterances']==4 and result['training_unique_utterances']==0
    assert result['training_unique_hours']==0 and len(result['excluded_ctc'])==4
    assert all(row['required_ctc_steps']>row['available_ctc_steps'] for row in result['excluded_ctc'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==original
    proof=result['shards'][0]
    assert proof['archive']=='shard-002.tar.gz'
    manifest=json.loads((tmp_path/'filtered'/proof['manifest']).read_text())
    assert manifest['entries']==[] and len(manifest['excluded_ctc'])==4
    with tarfile.open(tmp_path/'filtered'/proof['archive']) as archive:
        assert len(archive.getnames())==6
        for uid,item in files.items():
            assert hashlib.sha256(archive.extractfile(item['source_member']).read()).hexdigest()==item['sha256']
