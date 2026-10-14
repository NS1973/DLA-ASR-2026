"""Share identical immutable closed artifacts while retaining every path and SHA.

This never visits active stages, never removes an original logical file, and
never changes artifact contents. Hardlink aliases share inode timestamps; their
previous timestamps are retained in the operation proof. Future artifact writes
must use atomic replacement, as the project's checkpoint writers already do.
"""
import hashlib
import json
import os
from pathlib import Path
import uuid


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def closed(run,proc_root):
    if json.loads((run/'status.json').read_text()).get('event')!='train_exit':
        return False
    pid=json.loads((run/'process.json').read_text())['pid']
    process=proc_root/str(pid)
    return not process.exists() or (process/'stat').read_text().split(') ',1)[1].split()[0]=='Z'


def deduplicate_closed_runs(root,apply=False,proc_root=Path('/proc'),expected_sha_by_path=None):
    root=Path(root).resolve();proc_root=Path(proc_root)
    files=[];runs=[];groups={}
    for run in sorted(root.iterdir()):
        if run.is_symlink():
            raise ValueError('Run symlink is not an owned closed directory')
        if not run.is_dir() or not (run/'status.json').exists() or not (run/'process.json').exists():
            continue
        if not closed(run,proc_root):
            continue
        runs.append(run.name)
        paths=list(run.glob('*.pt'))+[run/'train_manifest.json']
        for path in sorted(set(paths)):
            if path.is_symlink():
                raise ValueError('Artifact symlink is not eligible')
            if not path.exists():
                continue
            if not path.is_file() or not path.resolve().is_relative_to(root):
                raise ValueError('Artifact outside owned closed directory')
            stat=path.stat();sha=digest(path)
            item={'path':str(path),'sha256':sha,'bytes':stat.st_size,
                  'inode':stat.st_ino,'device':stat.st_dev,'mtime_ns':stat.st_mtime_ns,
                  'nlink':stat.st_nlink}
            files.append(item);groups.setdefault((sha,stat.st_dev),[]).append(item)
    if expected_sha_by_path is not None:
        for item in files:
            name=Path(item['path']).relative_to(root).as_posix()
            if expected_sha_by_path.get(name)!=item['sha256']:
                raise ValueError('Verified offserver reserve is missing or different: '+name)
    candidates=[];potential=0
    for rows in groups.values():
        canonical=rows[0];other_inodes={}
        for item in rows[1:]:
            if item['inode']!=canonical['inode']:
                candidates.append((canonical,item))
                other_inodes.setdefault(item['inode'],[]).append(item)
        for aliases in other_inodes.values():
            if len(aliases)==aliases[0]['nlink']:
                potential+=aliases[0]['bytes']
    replacements=[]
    if apply:
        for canonical,item in candidates:
            source=Path(canonical['path']);target=Path(item['path'])
            if not closed(source.parent,proc_root) or not closed(target.parent,proc_root):
                raise ValueError('Stage became active; deduplication stopped')
            if digest(source)!=item['sha256'] or digest(target)!=item['sha256']:
                raise ValueError('Artifact changed after inventory')
            temporary=target.with_name(target.name+'.hardlink-'+uuid.uuid4().hex)
            try:
                os.link(source,temporary)
                os.replace(temporary,target)
            finally:
                if temporary.exists():
                    temporary.unlink()
            replacements.append({'path':str(target),'canonical':str(source),
                                 'sha256':item['sha256'],'bytes':item['bytes'],
                                 'previous_mtime_ns':item['mtime_ns']})
    for item in files:
        path=Path(item['path'])
        if not path.is_file() or path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:
            raise ValueError('Original path or SHA was not preserved')
    return {'closed_runs':runs,'all_original_paths_and_sha_preserved':True,
            'reclaimable_bytes':potential,'reclaimed_bytes':potential if apply else 0,
            'applied':bool(apply),'inventory':files,'replacements':replacements}
