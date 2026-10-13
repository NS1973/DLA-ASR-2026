from pathlib import Path


def test_completed_archive_rename_retries_without_downloading(tmp_path, monkeypatch):
    from ops.download_data import finalize_archive
    temp, final = tmp_path/'archive.partial', tmp_path/'archive.tar.gz'
    temp.write_bytes(b'complete-verified-archive')
    original = Path.replace
    attempts = []
    def occasionally_locked(path, destination):
        attempts.append(path)
        if len(attempts) < 3:
            raise PermissionError('Windows reader has not closed yet')
        return original(path, destination)
    monkeypatch.setattr(Path,'replace',occasionally_locked)
    monkeypatch.setattr('ops.download_data.time.sleep',lambda seconds:None)
    finalize_archive(temp, final)
    assert final.read_bytes() == b'complete-verified-archive'
    assert not temp.exists()
    assert len(attempts) == 3
