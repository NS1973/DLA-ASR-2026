import hashlib
import json
from pathlib import Path

import pytest

from ops.build_mixed_manifest import build_manifest, load_manifest


def entries(prefix, durations):
    return [dict(uid=f'{prefix}{i}', path=f'LibriSpeech/{prefix}/{i}.flac',
                 text='A TEST', duration=d) for i, d in enumerate(durations)]


@pytest.mark.parametrize('ratio', [.75, .85])
def test_duration_mix_determinism_coverage_and_relocation(tmp_path, ratio):
    clean, other = entries('clean', [10, 11, 12]), entries('other', [4, 5])
    result = build_manifest(clean, other, ratio, seed=10)
    assert result == build_manifest(clean, other, ratio, seed=10)
    rows = result['entries']
    assert {x['uid'] for x in clean + other} <= {x['uid'] for x in rows}
    other_time = sum(x['duration'] for x in rows if x['uid'].startswith('other'))
    total = sum(x['duration'] for x in rows)
    assert abs(other_time / total - ratio) <= max(x['duration'] for x in clean) / total
    assert result['statistics']['unique_other_hours'] == 9 / 3600
    assert result['statistics']['other_repeats'] > 1
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(result))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_manifest(path, tmp_path / 'relocated', sha)
    assert len(loaded) == len(rows)
    assert all(Path(x['path']).is_relative_to(tmp_path / 'relocated') for x in loaded)
    with pytest.raises(ValueError, match='SHA'):
        load_manifest(path, tmp_path, '0' * 64)


def test_rejects_leakage_duplicates_and_unsafe_paths():
    clean, other = entries('clean', [10]), entries('other', [5])
    with pytest.raises(ValueError, match='excluded'):
        build_manifest(clean, other, .75, excluded_ids={'other0'})
    with pytest.raises(ValueError, match='unique'):
        build_manifest(clean, clean, .75)
    with pytest.raises(ValueError, match='relative'):
        build_manifest(clean, [{**other[0], 'path': '../escape.flac'}], .75)


def test_dataset_view_allows_existing_operator_managed_symlinks(tmp_path):
    root = tmp_path / 'dataset'
    root.mkdir()
    external = tmp_path / 'existing-clean'
    external.mkdir()
    try:
        (root / 'clean').symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip('OS does not permit creating symlinks')
    document = {'schema_version': 1, 'entries': [dict(uid='a', path='clean/a.flac', text='A', duration=1)]}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(document))
    rows = load_manifest(path, root, hashlib.sha256(path.read_bytes()).hexdigest())
    assert rows[0]['path'] == str(root / 'clean/a.flac')
