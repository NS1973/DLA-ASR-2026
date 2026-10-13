import io
import os
import tarfile
import zipfile

import pytest


def test_log_archive_stream_preserves_old_timestamp_records_and_excludes_weights(tmp_path):
    from scripts.package_delivery import write_logs_zip
    source = tmp_path / 'logs.tar.gz'
    with tarfile.open(source, 'w:gz') as archive:
        for name, content in [('run/metrics.jsonl', b'{"step":1}\n'),
                              ('run/media/audio.wav', b'wave bytes'),
                              ('DRAFT_COLLECTION.json', b'{}'),
                              ('run/last.pt', b'weights'), ('run/cache.npz', b'cache')]:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            member.mtime = 0
            archive.addfile(member, io.BytesIO(content))
    destination = tmp_path / 'logs.zip'
    write_logs_zip(destination, logs_archive=source)
    with zipfile.ZipFile(destination) as archive:
        assert set(archive.namelist()) == {'run/metrics.jsonl', 'run/media/audio.wav', 'DRAFT_COLLECTION.json'}
        assert archive.read('run/metrics.jsonl') == b'{"step":1}\n'
        assert archive.read('run/media/audio.wav') == b'wave bytes'
        assert archive.testzip() is None


def test_unsafe_log_archive_keeps_previous_complete_zip(tmp_path):
    from scripts.package_delivery import write_logs_zip
    source = tmp_path / 'unsafe.tar'
    with tarfile.open(source, 'w') as archive:
        member = tarfile.TarInfo('../outside.txt')
        member.size = 1
        archive.addfile(member, io.BytesIO(b'x'))
    destination = tmp_path / 'logs.zip'
    destination.write_bytes(b'previous complete artifact')
    with pytest.raises(ValueError, match='Unsafe'):
        write_logs_zip(destination, logs_archive=source)
    assert destination.read_bytes() == b'previous complete artifact'
    assert not (tmp_path / 'logs.zip.tmp').exists()


def test_directory_logs_with_epoch_timestamp_remain_supported(tmp_path):
    from scripts.package_delivery import write_logs_zip
    root = tmp_path / 'run'
    root.mkdir()
    record = root / 'collection.json'
    record.write_bytes(b'{"captured":"2026-10-13"}')
    os.utime(record, (0, 0))
    destination = tmp_path / 'logs.zip'
    write_logs_zip(destination, directory=root)
    with zipfile.ZipFile(destination) as archive:
        assert archive.read('collection.json') == record.read_bytes()
        assert archive.getinfo('collection.json').date_time[0] >= 1980
