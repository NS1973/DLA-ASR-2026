import hashlib
import ast
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import zipfile

import pytest

from scripts.bootstrap_colab import extract_verified_source_archive


@pytest.fixture
def real_source(tmp_path):
    archive = tmp_path / 'source.zip'
    if os.environ.get('ASR_SOURCE_TEST_ZIP'):
        # Clean source exports have no .git: the verifier supplies the actual
        # SHA-checked Git export instead of synthesizing project contents.
        shutil.copyfile(os.environ['ASR_SOURCE_TEST_ZIP'], archive)
    else:
        subprocess.run(['git', 'archive', '--format=zip', '-o', str(archive), 'HEAD'],
                       cwd=Path(__file__).resolve().parents[1], check=True)
    return archive, hashlib.sha256(archive.read_bytes()).hexdigest()


def test_installs_real_git_source_without_partial_target(real_source, tmp_path):
    archive, digest = real_source
    target = tmp_path / 'project'
    assert extract_verified_source_archive(archive, digest, target) == target
    assert (target / 'inference.py').is_file()
    assert (target / 'src/asr/runtime.py').is_file()
    assert not list(tmp_path.glob('.asr-source-*'))


def test_wrong_hash_never_creates_project(real_source, tmp_path):
    archive, _ = real_source
    target = tmp_path / 'project'
    with pytest.raises(ValueError, match='SHA-256'):
        extract_verified_source_archive(archive, '0' * 64, target)
    assert not target.exists()


@pytest.mark.parametrize('name', ['../escape.py', '/absolute.py', '..\\escape.py', 'C:/escape.py'])
def test_archive_paths_cannot_escape_target(tmp_path, name):
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr(name, 'unsafe')
    target = tmp_path / 'project'
    with pytest.raises(ValueError, match='Unsafe'):
        extract_verified_source_archive(archive, hashlib.sha256(archive.read_bytes()).hexdigest(), target)
    assert not target.exists()
    assert not (tmp_path / 'escape.py').exists()


def test_symlink_archive_does_not_create_project(tmp_path):
    archive = tmp_path / 'link.zip'
    info = zipfile.ZipInfo('inference.py')
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr(info, '../outside.py')
    with pytest.raises(ValueError, match='Unsafe'):
        extract_verified_source_archive(archive, hashlib.sha256(archive.read_bytes()).hexdigest(), tmp_path / 'project')
    assert not (tmp_path / 'project').exists()


def test_existing_project_is_preserved(real_source, tmp_path):
    archive, digest = real_source
    target = tmp_path / 'project'
    target.mkdir()
    (target / 'keep.txt').write_text('existing')
    with pytest.raises(FileExistsError):
        extract_verified_source_archive(archive, digest, target)
    assert (target / 'keep.txt').read_text() == 'existing'


def test_notebook_bootstraps_archive_and_frozen_manifest_without_git(real_source, tmp_path, monkeypatch):
    import gdown
    archive, digest = real_source
    manifest = tmp_path / 'release.json'
    content = {'resources': [{'filename': 'model.pt', 'sha256': 'a' * 64}], 'model_sha256': 'a' * 64}
    manifest.write_text(json.dumps(content))
    downloads, commands = [], []
    def download(url, output, **kwargs):
        downloads.append(url)
        shutil.copyfile(archive if url == 'source-fixture' else manifest, output)
        return output
    monkeypatch.setattr(gdown, 'download', download)
    monkeypatch.setattr(subprocess, 'run', lambda command, **kwargs: commands.append(command))
    monkeypatch.chdir(tmp_path)
    notebook = json.loads((Path(__file__).resolve().parents[1] / 'DLA_ASR_Demo.ipynb').read_text(encoding='utf-8'))
    code = ''.join(notebook['cells'][3]['source'])
    module = ast.parse(code)
    target = tmp_path / 'project'
    for node in module.body:
        if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'PROJECT' for x in node.targets):
            node.value = ast.Call(func=ast.Name(id='Path', ctx=ast.Load()), args=[ast.Constant(str(target))], keywords=[])
    ast.fix_missing_locations(module)
    namespace = {'SOURCE_ARCHIVE_URL': 'source-fixture', 'SOURCE_ARCHIVE_SHA256': digest,
                 'RELEASE_MANIFEST_URL': 'manifest-fixture',
                 'RELEASE_MANIFEST_SHA256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                 'REPOSITORY_URL': 'must-not-clone'}
    exec(compile(module, 'actual-notebook-bootstrap', 'exec'), namespace)
    assert downloads == ['source-fixture', 'manifest-fixture']
    assert not any(command[0] == 'git' for command in commands)
    assert (target / 'inference.py').is_file()
    assert json.loads((target / 'resources.json').read_text()) == content


@pytest.mark.parametrize('damage', ['hash', 'model_identity'])
def test_bad_release_manifest_preserves_installed_manifest(real_source, tmp_path, monkeypatch, damage):
    import gdown
    archive, digest = real_source
    target = tmp_path / 'project'
    extract_verified_source_archive(archive, digest, target)
    original = (target / 'resources.json').read_bytes()
    manifest = tmp_path / 'release.json'
    manifest.write_text(json.dumps({'resources': [{'sha256': 'a' * 64}], 'model_sha256': 'b' * 64}))
    monkeypatch.setattr(gdown, 'download', lambda url, output, **kwargs: shutil.copyfile(manifest, output))
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: None)
    monkeypatch.chdir(tmp_path)
    notebook = json.loads((Path(__file__).resolve().parents[1] / 'DLA_ASR_Demo.ipynb').read_text(encoding='utf-8'))
    module = ast.parse(''.join(notebook['cells'][3]['source']))
    for node in module.body:
        if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'PROJECT' for x in node.targets):
            node.value = ast.Call(func=ast.Name(id='Path', ctx=ast.Load()), args=[ast.Constant(str(target))], keywords=[])
    ast.fix_missing_locations(module)
    namespace = {'SOURCE_ARCHIVE_URL': '', 'SOURCE_ARCHIVE_SHA256': '',
                 'RELEASE_MANIFEST_URL': 'manifest-fixture',
                 'RELEASE_MANIFEST_SHA256': '0' * 64 if damage == 'hash' else hashlib.sha256(manifest.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match='manifest'):
        exec(compile(module, 'actual-notebook-bootstrap', 'exec'), namespace)
    assert (target / 'resources.json').read_bytes() == original
