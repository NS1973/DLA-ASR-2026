import hashlib,json,os
from pathlib import Path
import pytest
from ops.closed_storage_dedup import deduplicate_closed_runs


def fixture_runs(tmp_path):
    root=tmp_path/'runs';root.mkdir()
    for name,event in [('a','train_exit'),('b','train_exit'),('active','train')]:
        run=root/name;run.mkdir()
        (run/'status.json').write_text(json.dumps({'event':event}))
        (run/'process.json').write_text(json.dumps({'pid':99999999}))
        (run/'best.pt').write_bytes(b'identical checkpoint')
        (run/'train_manifest.json').write_bytes(b'[{"uid":"x"}]')
    return root


def test_only_closed_identical_files_share_storage_and_keep_all_bytes_and_names(tmp_path):
    root=fixture_runs(tmp_path)
    before={p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}
    active_inode=(root/'active/best.pt').stat().st_ino
    dry=deduplicate_closed_runs(root)
    assert dry['reclaimed_bytes']==0 and dry['reclaimable_bytes']>0
    proof=deduplicate_closed_runs(root,apply=True)
    assert proof['reclaimed_bytes']==dry['reclaimable_bytes']
    assert proof['all_original_paths_and_sha_preserved']
    assert before=={p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert (root/'a/best.pt').stat().st_ino==(root/'b/best.pt').stat().st_ino
    assert (root/'active/best.pt').stat().st_ino==active_inode
    assert deduplicate_closed_runs(root,apply=True)['reclaimed_bytes']==0


def test_status_saying_exit_does_not_allow_touching_a_live_process(tmp_path):
    root=fixture_runs(tmp_path);proc=tmp_path/'proc';(proc/'99999999').mkdir(parents=True)
    (proc/'99999999/stat').write_text('99999999 (trainer) S 1 2 3')
    assert deduplicate_closed_runs(root,apply=True,proc_root=proc)['closed_runs']==[]


def test_symlink_artifact_is_rejected_before_any_replacement(tmp_path):
    root=fixture_runs(tmp_path);outside=tmp_path/'outside.pt';outside.write_bytes(b'external')
    link=root/'b/link.pt'
    try:link.symlink_to(outside)
    except OSError:pytest.skip('Symlink privilege unavailable')
    with pytest.raises(ValueError,match='symlink'):deduplicate_closed_runs(root,apply=True)
    assert outside.read_bytes()==b'external'


def test_every_duplicate_alias_is_redirected_and_storage_counted_once(tmp_path):
    root=fixture_runs(tmp_path);third=root/'c';third.mkdir()
    (third/'status.json').write_text('{"event":"train_exit"}')
    (third/'process.json').write_text('{"pid":99999999}')
    os.link(root/'b/best.pt',third/'best.pt')
    proof=deduplicate_closed_runs(root,apply=True)
    assert len({(root/name/'best.pt').stat().st_ino for name in ['a','b','c']})==1
    assert deduplicate_closed_runs(root,apply=True)['reclaimed_bytes']==0


def test_offserver_inventory_mismatch_prevents_any_replacement(tmp_path):
    root=fixture_runs(tmp_path);before=(root/'b/best.pt').stat().st_ino
    with pytest.raises(ValueError,match='reserve'):
        deduplicate_closed_runs(root,apply=True,expected_sha_by_path={})
    assert (root/'b/best.pt').stat().st_ino==before
