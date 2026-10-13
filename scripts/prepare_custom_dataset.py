"""Download and safely extract a user's ZIP/TAR or Google Drive dataset link."""
import argparse
from pathlib import Path
import tarfile
import zipfile
import shutil
import requests


def extract(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    def safe(name):
        target = (destination / name).resolve()
        if target != destination and destination not in target.parents:
            raise ValueError(f'Unsafe archive path: {name}')
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                safe(member.filename)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Archive symlinks are not supported')
            source.extractall(destination)
    else:
        with tarfile.open(archive) as source:
            for member in source:
                safe(member.name)
                if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                    raise ValueError('Unsupported archive member')
            source.extractall(destination, filter='data')
    matches = [destination] if (destination / 'audio').is_dir() else [p.parent for p in destination.rglob('audio') if p.is_dir()]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one dataset containing audio/, found {len(matches)}')
    return matches[0]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--url', required=True)
    p.add_argument('--output-dir', default='data/custom')
    args = p.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / 'input.archive'
    if 'drive.google.com' in args.url:
        import gdown
        if not gdown.download(url=args.url, output=str(archive), fuzzy=True, quiet=False):
            raise RuntimeError('Google Drive download failed')
    else:
        with requests.get(args.url, stream=True, timeout=(20,120)) as r:
            r.raise_for_status()
            with archive.open('wb') as stream:
                for chunk in r.iter_content(2**20):
                    if chunk: stream.write(chunk)
    dataset = extract(archive, root / 'extracted')
    (root / 'dataset_path.txt').write_text(str(dataset.resolve()), encoding='utf-8')
    print(dataset)
