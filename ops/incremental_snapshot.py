"""Restore and deeply verify snapshots whose unchanged blobs remain in an immutable cache."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def ingest_snapshot(archive_path, store, output, materialize=True):
    store,output=Path(store),Path(output)
    store.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    new_blobs=0;manifest=None
    with tarfile.open(archive_path,'r|*') as archive:
        for member in archive:
            if not member.isfile():raise ValueError('Unsafe snapshot member type')
            if member.name=='manifest.json':
                manifest=json.load(archive.extractfile(member));continue
            if not member.name.startswith('blobs/') or not re.fullmatch('[a-f0-9]{64}',member.name[6:]):
                raise ValueError('Unsafe blob member name')
            digest=member.name[6:];target=store/digest
            temporary=Path(str(target)+'.partial')
            h=hashlib.sha256();count=0
            with archive.extractfile(member) as source,temporary.open('wb') as destination:
                while block:=source.read(2**20):destination.write(block);h.update(block);count+=len(block)
            if count!=member.size or h.hexdigest()!=digest:raise ValueError('Blob SHA or size mismatch')
            temporary.replace(target);new_blobs+=1
    if manifest is None:raise ValueError('Missing snapshot manifest')
    verified={};resolved={}
    for name,item in manifest['files'].items():
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
            raise ValueError('Unsafe logical snapshot path')
        digest=item['sha256']
        if not re.fullmatch('[a-f0-9]{64}',digest):raise ValueError('Unsafe blob SHA')
        cached=store/digest
        if not cached.is_file():raise ValueError('Missing cached blob: '+digest)
        if digest not in verified:
            if cached.stat().st_size!=item['bytes'] or sha(cached)!=digest:raise ValueError('Cached blob SHA or size mismatch: '+digest)
            verified[digest]=item['bytes']
        elif verified[digest]!=item['bytes']:raise ValueError('Conflicting blob size')
        target=cached
        if materialize:
            target=output/'files'/Path(*path.parts);target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists() or target.stat().st_size!=item['bytes'] or sha(target)!=digest:
                temporary=Path(str(target)+'.partial');shutil.copyfile(cached,temporary);temporary.replace(target)
        resolved[name]=str(target)
    pairs=manifest.get('checkpoint_pairs',[])
    if pairs:
        import torch
        for pair in pairs:
            model=torch.load(resolved[pair['model']],map_location='cpu',weights_only=True)
            full=torch.load(resolved[pair['resume']],map_location='cpu',weights_only=False)
            if model['state']!=full['state'] or model['state']!=pair['state']:raise ValueError('Checkpoint state mismatch')
            if model['model'].keys()!=full['model'].keys() or not all(torch.equal(v,full['model'][k]) for k,v in model['model'].items()):
                raise ValueError('Checkpoint tensor mismatch')
            if not {'optimizer','scheduler','rng_python','rng_numpy','rng_torch','rng_cuda','gradients'}<=set(full):
                raise ValueError('Incomplete full resume state')
            del model,full
    proof={'bytes':Path(archive_path).stat().st_size,'sha256':sha(archive_path),'new_blobs':new_blobs,
           'all_blobs_and_logical_files_sha_verified':True,'all_model_full_pairs_tensor_state_verified':True,
           'states':pairs,'files':manifest['files'],'resolved_files':resolved,'blob_store':str(store),
           'requires_cached_blobs':True,'self_contained_archive':False}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'verification.json').write_text(json.dumps(proof,indent=2))
    return proof
