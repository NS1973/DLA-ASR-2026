import hashlib
import sys
from types import SimpleNamespace
from pathlib import Path
import pytest
from scripts.download_resources import fetch


def test_drive_resource_download_checks_hash_without_browser_cookies(tmp_path, monkeypatch):
    payload=b'verified model bytes'
    calls=[]
    def download(**kwargs):
        calls.append(kwargs)
        Path(kwargs['output']).write_bytes(payload)
        return kwargs['output']
    monkeypatch.setitem(sys.modules,'gdown',SimpleNamespace(download=download))
    target=tmp_path/'model.pt'
    digest=hashlib.sha256(payload).hexdigest()
    fetch('https://drive.google.com/file/d/example/view',target,digest)
    assert target.read_bytes()==payload
    assert calls[0]['use_cookies'] is False
    assert calls[0]['resume'] is True
    fetch('https://drive.google.com/file/d/example/view',target,digest)
    assert len(calls)==1


def test_corrupt_drive_resource_never_becomes_final(tmp_path,monkeypatch):
    def download(**kwargs):
        Path(kwargs['output']).write_bytes(b'bad bytes')
        return kwargs['output']
    monkeypatch.setitem(sys.modules,'gdown',SimpleNamespace(download=download))
    target=tmp_path/'model.pt'
    with pytest.raises(ValueError,match='SHA-256 mismatch'):
        fetch('https://drive.google.com/file/d/example/view',target,'0'*64)
    assert not target.exists()
