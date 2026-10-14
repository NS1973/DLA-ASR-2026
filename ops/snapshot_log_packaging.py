"""Materialize closed logical logs from verified content-addressed snapshot proofs.

Raw snapshot TAR members are named blobs/<sha> and may include full checkpoints.
They must not be passed directly to the generic log TAR packager.
"""
import hashlib
import json
from pathlib import Path,PurePosixPath
import uuid
import zipfile


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write_snapshot_logs(destination, proof_paths):
    destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    files={};origins=[]
    for proof_path in proof_paths:
        proof_path=Path(proof_path);proof=json.loads(proof_path.read_text())
        if not proof.get('all_model_full_pairs_tensor_state_verified') or not proof.get('states'):
            raise ValueError('Unverified checkpoint reserve')
        runs={pair['model'].split('/')[0] for pair in proof['states']}
        for run in runs:
            name=run+'/status.json';meta=proof['files'][name];path=Path(proof['resolved_files'][name])
            if path.stat().st_size!=meta['bytes'] or digest(path)!=meta['sha256']:
                raise ValueError('Closed status SHA mismatch')
            if json.loads(path.read_text()).get('event')!='train_exit':
                raise ValueError('Active or failed stage is not a closed log source')
        origins.append({'proof_sha256':digest(proof_path),'runs':sorted(runs),'checkpoint_pairs':proof['states']})
        for name,meta in proof['files'].items():
            path=PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name or not path.parts or path.parts[0] not in runs:
                raise ValueError('Unsafe logical snapshot path')
            if path.suffix in {'.pt','.pth','.npz','.tmp','.partial'}:continue
            source=Path(proof['resolved_files'][name])
            if source.stat().st_size!=meta['bytes'] or digest(source)!=meta['sha256']:
                raise ValueError('Log blob SHA or size mismatch')
            if name in files and files[name]['sha256']!=meta['sha256']:
                raise ValueError('Conflicting closed snapshots for '+name)
            files[name]={**meta,'source':str(source)}
    if not files:raise ValueError('No closed logs supplied')
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(str(destination)+'.partial-'+uuid.uuid4().hex)
    try:
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
            for name,meta in sorted(files.items()):archive.write(meta['source'],name)
            manifest={'all_sources_closed':True,'origins':origins,'files':{n:{k:v for k,v in m.items() if k!='source'} for n,m in files.items()}}
            archive.writestr('snapshot-log-manifest.json',json.dumps(manifest,indent=2))
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:raise ValueError('Written ZIP CRC mismatch')
            for name,meta in manifest['files'].items():
                with archive.open(name) as stream:
                    if hashlib.file_digest(stream,'sha256').hexdigest()!=meta['sha256']:
                        raise ValueError('Written ZIP member SHA mismatch')
        if destination.exists():raise FileExistsError(destination)
        temporary.replace(destination)
    finally:temporary.unlink(missing_ok=True)
    return {'sha256':digest(destination),'bytes':destination.stat().st_size,'logical_files':len(files),'all_member_sha_and_crc_verified':True,'all_sources_closed':True}
