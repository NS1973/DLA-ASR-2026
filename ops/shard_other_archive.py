"""Validate whole chapters and emit fixed incremental training shards.

The caller must verify the official compressed archive checksum before use.
Existing audio is omitted only after matching its UID, bytes and duration.
"""
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np
import soundfile as sf
from src.asr.codec import Codec


class HashingWriter:
    def __init__(self, path):
        self.stream = path.open('xb')
        self.digest = hashlib.sha256()
        self.bytes = 0

    def write(self, data):
        written = self.stream.write(data)
        self.digest.update(data[:written]);self.bytes += written
        return written

    def tell(self):
        return self.bytes

    def flush(self):
        self.stream.flush()


def shard_archive(source, output, target_hours=100, known_files=None, excluded_ids=()):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Shard destination must be new and empty')
    known_files = known_files or {}
    excluded_ids = set(excluded_ids)
    codec = Codec()
    seen = set();reused = set();total_seconds = 0
    shards = [];archive = writer = temporary = None
    shard_rows = [];shard_files = {};shard_seconds = 0

    def finish_shard():
        nonlocal archive,writer,temporary,shard_rows,shard_files,shard_seconds
        if archive is None:
            return
        archive.close();writer.flush();writer.stream.close()
        final = output / temporary.name.removesuffix('.partial')
        temporary.rename(final)
        proof = {'archive': final.name, 'bytes': writer.bytes, 'sha256': writer.digest.hexdigest(),
                 'unique_hours': shard_seconds/3600, 'unique_utterances': len(shard_rows)}
        manifest = {**proof, 'entries': shard_rows, 'files': shard_files, 'whole_chapters': True,
                    'audio_full_decode_rate_channels_finite_ctc_verified': True}
        manifest_path = final.with_suffix('.manifest.json')
        manifest_path.write_text(json.dumps(manifest, separators=(',', ':')))
        proof['manifest'] = manifest_path.name
        proof['manifest_sha256'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        shards.append(proof)
        (output/'progress.json').write_text(json.dumps({'shards':shards,'unique_utterances_seen':len(seen),
            'unique_hours_seen':total_seconds/3600,'existing_utterances_reused':len(reused)},indent=2))
        archive=writer=temporary=None;shard_rows=[];shard_files={};shard_seconds=0

    def process_chapter(files):
        nonlocal archive,writer,temporary,total_seconds,shard_seconds
        if not files:
            return
        transcripts = {}
        for name,data in files:
            if name.endswith('.trans.txt'):
                for line in data.decode('utf-8').splitlines():
                    uid,text=line.split(' ',1)
                    if uid in transcripts:raise ValueError('Duplicate transcript UID')
                    transcripts[uid]=text
        rows=[]
        for name,data in files:
            if not name.endswith('.flac'):
                continue
            uid=PurePosixPath(name).stem
            if uid in seen:raise ValueError('Duplicate audio UID')
            if uid in excluded_ids:raise ValueError('Training contains excluded dev/test UID')
            if uid not in transcripts:raise ValueError('Missing transcript')
            wave,rate=sf.read(io.BytesIO(data),dtype='float32',always_2d=True)
            if rate!=16000 or wave.shape[1]!=1 or not len(wave) or not np.isfinite(wave).all():
                raise ValueError('Audio must be finite mono16k')
            tokens=codec.encode(transcripts[uid])
            required=len(tokens)+sum(a==b for a,b in zip(tokens,tokens[1:]))
            available=(max(400,len(wave))//160+4)//4
            if not tokens or available<required:raise ValueError('Infeasible CTC target')
            digest=hashlib.sha256(data).hexdigest();duration=len(wave)/rate
            row={'uid':uid,'path':name,'text':codec.normalize(transcripts[uid]),'duration':duration,'sha256':digest}
            if uid in known_files:
                prior=known_files[uid]
                if prior['sha256']!=digest or abs(prior['duration']-duration)>1e-8:
                    raise ValueError('Existing audio identity mismatch')
                if prior.get('transcription_sha256') and hashlib.sha256((transcripts[uid]+'\n').encode()).hexdigest()!=prior['transcription_sha256']:
                    raise ValueError('Existing transcript identity mismatch')
                reused.add(uid)
            rows.append(row);seen.add(uid);total_seconds+=duration
        if set(transcripts)!={row['uid'] for row in rows}:raise ValueError('Transcript/audio UID mismatch')
        flags=[row['uid'] in known_files for row in rows]
        if any(flags):
            if not all(flags):raise ValueError('Existing subset ends inside a chapter')
            return
        if archive is None:
            temporary=output/f'shard-{len(shards)+1:03d}.tar.gz.partial'
            writer=HashingWriter(temporary)
            archive=tarfile.open(fileobj=writer,mode='w:gz',compresslevel=1)
        for name,data in files:
            member=tarfile.TarInfo(name);member.size=len(data)
            archive.addfile(member,io.BytesIO(data))
            shard_files[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        shard_rows.extend(rows);shard_seconds+=sum(row['duration'] for row in rows)
        if shard_seconds>=target_hours*3600:finish_shard()

    chapter=None;buffer=[];buffer_bytes=0
    try:
        with gzip.open(source,'rb') as raw:
            with tarfile.open(fileobj=raw,mode='r|') as source_tar:
                for member in source_tar:
                    path=PurePosixPath(member.name)
                    if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe archive path')
                    if not member.isfile() or path.parts[:2]!=('LibriSpeech','train-other-500'):continue
                    if len(path.parts)!=5 or not (member.name.endswith('.flac') or member.name.endswith('.trans.txt')):
                        raise ValueError('Unexpected training member')
                    current=path.parts[:4]
                    if chapter is not None and current!=chapter:
                        process_chapter(buffer);buffer=[];buffer_bytes=0
                    chapter=current;data=source_tar.extractfile(member).read()
                    buffer.append((member.name,data));buffer_bytes+=len(data)
                    if buffer_bytes>256*2**20:raise ValueError('Chapter exceeds bounded buffer')
            while raw.read(2**20):pass  # Consume trailer and validate full gzip CRC.
        process_chapter(buffer);finish_shard()
    except Exception:
        if archive is not None:
            archive.close();writer.stream.close()
        raise
    if set(known_files)-reused:raise ValueError('Existing IDs absent from official full archive')
    result={'shards':shards,'unique_utterances':len(seen),'unique_hours':total_seconds/3600,
            'existing_utterances_reused':len(reused),'full_audio_and_ctc_validated':True,
            'source':str(source),'selection':'All full-archive chapters; existing identical chapters omitted from transfer'}
    (output/'ready.json').write_text(json.dumps(result,indent=2))
    return result
