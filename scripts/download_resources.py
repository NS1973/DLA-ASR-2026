"""Download release artifacts with SHA-256 checking and resumable requests."""
import argparse
import hashlib
import json
from pathlib import Path
import requests
from urllib.parse import urlparse


def fetch(url, destination, sha256):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(destination) + '.partial')
    if destination.exists():
        with destination.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() == sha256:
                return destination
    offset = partial.stat().st_size if partial.exists() else 0
    if urlparse(url).hostname == 'drive.google.com':
        import gdown
        result = gdown.download(url=url,output=str(partial),fuzzy=True,
                                quiet=False,resume=True,use_cookies=False)
        if not result:
            raise RuntimeError('Public Drive resource download failed')
    else:
        with requests.get(url, headers={'Range':f'bytes={offset}-'} if offset else {}, stream=True, timeout=(20,120)) as response:
            response.raise_for_status()
            mode = 'ab' if offset and response.status_code == 206 else 'wb'
            if mode == 'ab' and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                raise ValueError('Unexpected resume Content-Range')
            with partial.open(mode) as stream:
                for chunk in response.iter_content(2**20):
                    if chunk: stream.write(chunk)
    with partial.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != sha256:
        raise ValueError(f'SHA-256 mismatch for {destination.name}')
    partial.replace(destination)
    return destination


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default='resources.json')
    p.add_argument('--output-dir', default='artifacts')
    args = p.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    if not manifest.get('resources'):
        raise ValueError('Release resources not populated yet; final manifest is generated from verified artifacts')
    for resource in manifest['resources']:
        downloaded = fetch(resource['url'], Path(args.output_dir) / resource['filename'], resource['sha256'])
        if resource.get('extract_to'):
            from prepare_custom_dataset import extract
            extract(downloaded, Path(args.output_dir) / resource['extract_to'])
