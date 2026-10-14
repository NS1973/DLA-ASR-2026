"""Receive only missing immutable tar members, verifying identity before atomic placement."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile

import numpy as np
import soundfile as sf
from src.asr.codec import Codec


def member_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe shard member path')
    if relative.parts[:2] != ('LibriSpeech', 'train-other-500') or len(relative.parts) != 5:
        raise ValueError('Unexpected training partition path')
    path = Path(root).resolve().joinpath(*relative.parts)
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Unsafe shard write through dataset symlink')
    return path


def verified_received(root, manifest):
    result = {}
    for name, expected in manifest['files'].items():
        path = member_path(root,name)
        if path.is_file() and path.stat().st_size == expected['bytes']:
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream,'sha256').hexdigest()
            if digest == expected['sha256']:
                result[name] = digest
    return result


def host_available_memory():
    return next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))


def container_memory_headroom():
    pairs=[('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),
           ('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes')]
    for limit_path,usage_path in pairs:
        limit,usage=Path(limit_path),Path(usage_path)
        if limit.exists() and usage.exists():
            value=limit.read_text().strip()
            if value!='max':return int(value)-int(usage.read_text())
    return float('inf')


def check_reserves(root, needed, reserve_bytes, min_mem_available):
    if shutil.disk_usage(root).free < reserve_bytes+needed:
        raise OSError('Own dataset scratch reserve would be violated')
    if min_mem_available:
        available = host_available_memory()
        if available-needed < min_mem_available:
            raise OSError('Host available RAM reserve would be violated')
        if container_memory_headroom()-needed<8*2**30:
            raise OSError('Container memory reserve would be violated')
        if shutil.disk_usage('/home').free < 20*2**30 or shutil.disk_usage('/').free < 2**30:
            raise OSError('Host disk reserve would be violated')


def receive_shard(stream, root, manifest, excluded_ids=(), reserve_bytes=0, min_mem_available=0):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    expected=manifest['files'];rows={row['uid']:row for row in manifest['entries']}
    if len(rows)!=len(manifest['entries']):raise ValueError('Duplicate shard UID')
    excluded_ctc={row['uid']:row for row in manifest.get('excluded_ctc',[])}
    if len(excluded_ctc)!=len(manifest.get('excluded_ctc',[])) or set(rows)&set(excluded_ctc):
        raise ValueError('Conflicting CTC exclusion UID')
    rows.update(excluded_ctc)
    if set(rows)&set(excluded_ids):raise ValueError('Shard contains excluded clean/dev/test UID')
    for name in expected:member_path(root,name)
    received=verified_received(root,manifest)
    initial=set(received);input_seen=set();codec=Codec()
    identity=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    status_path=root/('receive-'+manifest['archive']+'.json')
    def save_status(complete=False):
        payload={'manifest_identity':identity,'received_files':received,'complete':complete}
        temp=Path(str(status_path)+'.tmp');temp.write_text(json.dumps(payload));temp.replace(status_path)
    with tarfile.open(fileobj=stream,mode='r|') as archive:
        for member in archive:
            name=member.name
            member_path(root,name)
            if not member.isfile() or name not in expected:raise ValueError('Unexpected tar member')
            if name in input_seen:raise ValueError('Duplicate incoming member')
            input_seen.add(name);info=expected[name]
            if member.size!=info['bytes'] or member.size>64*2**20:raise ValueError('Tar member size mismatch')
            check_reserves(root,member.size,reserve_bytes,min_mem_available)
            data=archive.extractfile(member).read()
            if hashlib.sha256(data).hexdigest()!=info['sha256']:raise ValueError('Tar member SHA mismatch')
            if name.endswith('.flac'):
                uid=PurePosixPath(name).stem
                if uid not in rows or rows[uid]['path']!=name:raise ValueError('Unexpected audio UID')
                wave,rate=sf.read(io.BytesIO(data),dtype='float32',always_2d=True)
                if rate!=16000 or wave.shape[1]!=1 or not len(wave) or not np.isfinite(wave).all():raise ValueError('Invalid audio')
                if abs(len(wave)/rate-rows[uid]['duration'])>1e-8:raise ValueError('Audio duration mismatch')
                tokens=codec.encode(rows[uid]['text']);required=len(tokens)+sum(a==b for a,b in zip(tokens,tokens[1:]))
                available=(max(400,len(wave))//160+4)//4
                feasible=bool(tokens) and available>=required
                if uid in excluded_ctc:
                    item=excluded_ctc[uid]
                    if feasible or item['required_ctc_steps']!=required or item['available_ctc_steps']!=available:
                        raise ValueError('Invalid CTC exclusion declaration')
                elif not feasible:raise ValueError('Infeasible CTC length')
            elif name.endswith('.trans.txt'):
                for line in data.decode('utf-8').splitlines():
                    uid,text=line.split(' ',1)
                    if uid not in rows or codec.normalize(text)!=rows[uid]['text']:raise ValueError('Transcript identity mismatch')
            else:raise ValueError('Unexpected member suffix')
            path=member_path(root,name)
            if name not in received:
                path.parent.mkdir(parents=True,exist_ok=True)
                temporary=Path(str(path)+'.partial')
                with temporary.open('wb') as out:out.write(data);out.flush()
                temporary.replace(path)
            received[name]=info['sha256']
            if len(input_seen)%100==0:save_status()
    save_status()
    if set(received)!=set(expected):raise ValueError('Incomplete shard; verified members retained for resume')
    save_status(complete=True)
    return {'files_total':len(expected),'new_files_verified':len(set(received)-initial),
            'all_members_sha_verified':True,'audio_ctc_and_transcripts_verified':True,'manifest_identity':identity,
            'all_training_ctc_feasible':True,'excluded_ctc_preserved_count':len(excluded_ctc)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--excluded-ids',required=True)
    args=p.parse_args();content=Path(args.manifest).read_bytes()
    if hashlib.sha256(content).hexdigest()!=args.manifest_sha256:raise ValueError('Frozen shard manifest SHA mismatch')
    result=receive_shard(sys.stdin.buffer,args.root,json.loads(content),json.loads(Path(args.excluded_ids).read_text()),
                         reserve_bytes=5*2**30,min_mem_available=32*2**30)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
