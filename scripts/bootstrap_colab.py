"""Verify and atomically install an exported Git ZIP before project imports."""
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile


def extract_verified_source_archive(archive, expected_sha256, destination):
    """Install flat Git-archive contents; preserve any existing project."""
    archive, destination = Path(archive), Path(destination)
    if destination.exists():
        raise FileExistsError(f'Project already exists: {destination}')
    with archive.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected_sha256:
        raise ValueError(f'Source SHA-256 mismatch: {actual}')
    with zipfile.ZipFile(archive) as source:
        members, names = source.infolist(), set()
        if sum(member.file_size for member in members) > 128 * 2**20:
            raise ValueError('Source archive exceeds 128 MiB uncompressed')
        for member in members:
            name = PurePosixPath(member.filename)
            if (not name.parts or name.is_absolute() or '..' in name.parts
                    or '\\' in member.filename or any(':' in part for part in name.parts)
                    or stat.S_ISLNK(member.external_attr >> 16) or str(name) in names):
                raise ValueError(f'Unsafe source archive member: {member.filename}')
            names.add(str(name))
        required = {'inference.py', 'train.py', 'requirements.txt', 'src/asr/runtime.py'}
        if not required <= names:
            raise ValueError('Source archive lacks required ASR project files')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.asr-source-', dir=destination.parent) as temporary:
            staging = Path(temporary) / 'project'
            staging.mkdir()
            for member in members:
                target = staging / member.filename
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(member) as incoming, target.open('xb') as outgoing:
                        shutil.copyfileobj(incoming, outgoing)
            if not all((staging / name).is_file() for name in required):
                raise ValueError('Required ASR source entry is not a regular file')
            staging.replace(destination)
    return destination
