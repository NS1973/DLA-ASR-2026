from pathlib import Path
import subprocess
import zipfile

import pytest

from scripts.package_delivery import export_source


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()


def test_nondefault_branch_zip_bundle_and_restored_head_are_identical(tmp_path):
    root=tmp_path/'repo';root.mkdir()
    git(root,'init','-b','codex/old');git(root,'config','user.name','Fixture');git(root,'config','user.email','fixture@example.invalid')
    (root/'old.txt').write_text('parent');git(root,'add','.');git(root,'commit','-m','parent')
    parent=git(root,'rev-parse','HEAD')
    git(root,'checkout','-b','codex/overnight')
    (root/'new.txt').write_text('overnight');git(root,'add','.');git(root,'commit','-m','overnight')
    current=git(root,'rev-parse','HEAD')
    out=tmp_path/'delivery'
    result=export_source(root,out,source_ref='codex/overnight')
    assert result['commit']==current and result['source_branch']=='codex/overnight'
    with zipfile.ZipFile(out/'source.zip') as archive:
        assert archive.read('new.txt')==b'overnight'
        assert archive.comment.decode()==current
    restored=tmp_path/'restore'
    subprocess.run(['git','clone','--branch',result['source_branch'],str(out/'source.bundle'),str(restored)],check=True,capture_output=True)
    assert git(restored,'rev-parse','HEAD')==current
    assert git(restored,'merge-base',parent,current)==parent
    assert export_source(root,tmp_path/'default')['source_branch']=='codex/overnight'
    with pytest.raises(RuntimeError,match='HEAD'):
        export_source(root,tmp_path/'wrong',source_ref='codex/old')
    with pytest.raises(FileExistsError):
        export_source(root,out,source_ref='codex/overnight')
