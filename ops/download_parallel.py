"""Bounded parallel ranges, appended in order for the growing SFTP mirror."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import tarfile
import time
import requests
from download_data import BASE, finalize_archive


def fetch_range(url, start, end):
    with requests.get(url, headers={'Range':f'bytes={start}-{end}'}, timeout=(15,60)) as response:
        response.raise_for_status()
        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range',''))
        if response.status_code != 206 or not match or int(match[1]) != start or int(match[2]) != end:
            raise ValueError('Server did not honor the exact bounded range')
        if len(response.content) != end-start+1:
            raise ValueError('Truncated range response')
        return response.content, int(match[3])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part',choices=['train-clean-360'],default='train-clean-360')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--root',default='data/downloads')
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True,exist_ok=True)
    destination = root/f'{args.part}.tar.gz'
    partial = Path(str(destination)+'.partial')
    if destination.exists():
        print('Already downloaded',flush=True)
        return
    url = BASE+destination.name
    _, total = fetch_range(url,0,0)
    chunk_size = 16*2**20
    for attempt in range(12):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            with ThreadPoolExecutor(max_workers=args.workers) as pool, partial.open('ab') as stream:
                while offset < total:
                    spans = [(start,min(total-1,start+chunk_size-1))
                             for start in range(offset,min(total,offset+args.workers*chunk_size),chunk_size)]
                    futures = [pool.submit(fetch_range,url,start,end) for start,end in spans]
                    for (start,end),future in zip(spans,futures):
                        content, remote_total = future.result()
                        if remote_total != total or start != offset:
                            raise ValueError('Inconsistent remote archive or append offset')
                        stream.write(content)
                        stream.flush()
                        offset = end+1
                    print(json.dumps({'downloaded_bytes':offset,'total_bytes':total}),flush=True)
            with tarfile.open(partial,'r:gz') as archive:
                for member in archive:
                    if member.name.endswith('.trans.txt'):
                        archive.extractfile(member).read()
            finalize_archive(partial,destination)
            with destination.open('rb') as stream:
                digest = hashlib.file_digest(stream,'sha256').hexdigest()
            Path(str(destination)+'.sha256').write_text(digest+'\n',encoding='ascii')
            print(json.dumps({'downloaded':args.part,'bytes':total,'sha256':digest}),flush=True)
            return
        except Exception as exc:
            print(f'RETRY {attempt}: {type(exc).__name__}: {exc}',flush=True)
            time.sleep(min(30,2**attempt))
    raise RuntimeError('Parallel download exhausted; contiguous partial is preserved')


if __name__=='__main__':
    main()
