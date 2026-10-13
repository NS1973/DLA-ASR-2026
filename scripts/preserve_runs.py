"""Seal a content-addressed snapshot of this project's runs without stopping training."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import time

import torch


def matching_pair(model_path, resume_path):
    for attempt in range(5):
        model_bytes, resume_bytes = model_path.read_bytes(), resume_path.read_bytes()
        model = torch.load(io.BytesIO(model_bytes), map_location='cpu', weights_only=True)
        resume = torch.load(io.BytesIO(resume_bytes), map_location='cpu', weights_only=False)
        if (model['state'] == resume['state'] and model['model'].keys() == resume['model'].keys()
                and all(torch.equal(value,resume['model'][key]) for key,value in model['model'].items())):
            return model_bytes, resume_bytes, model['state']
        time.sleep(.2)
    raise ValueError(f'Could not capture matching model/full checkpoint: {model_path}')


def capture(runs, output, parts=4, reserve_bytes=20*2**30):
    runs, output = Path(runs).resolve(), Path(output).resolve()
    if not runs.is_dir():
        raise FileNotFoundError(runs)
    output.mkdir(parents=True,exist_ok=False)
    paths = sorted(p for p in runs.rglob('*') if p.is_file() and not p.is_symlink()
                   and p.suffix not in {'.tmp','.partial'})
    estimated=sum(p.stat().st_size for p in paths)
    if shutil.disk_usage(output).free < estimated+reserve_bytes:
        raise OSError('Snapshot would violate the free-space reserve')
    archives=[tarfile.open(output/f'part-{i:02d}.tar.partial','w') for i in range(parts)]
    sizes=[0]*parts
    files, blobs, pairs, handled = {}, {}, [], set()
    started=time.time()
    def add(path, content):
        digest=hashlib.sha256(content).hexdigest()
        if digest not in blobs:
            index=min(range(parts),key=lambda i:sizes[i])
            member=tarfile.TarInfo('blobs/'+digest)
            member.size=len(content)
            archives[index].addfile(member,io.BytesIO(content))
            sizes[index]+=len(content)
            blobs[digest]={'archive':f'part-{index:02d}.tar','bytes':len(content)}
        files[str(path.relative_to(runs)).replace('\\','/')]={**blobs[digest],'sha256':digest}
        handled.add(path)
    try:
        for model_path in paths:
            if model_path.name not in {'best_model.pt','last_model.pt'}:
                continue
            resume_path=model_path.with_name(model_path.name.replace('_model',''))
            if resume_path not in paths:
                continue
            model_bytes,resume_bytes,state=matching_pair(model_path,resume_path)
            add(model_path,model_bytes);add(resume_path,resume_bytes)
            pairs.append({'model':str(model_path.relative_to(runs)).replace('\\','/'),
                          'resume':str(resume_path.relative_to(runs)).replace('\\','/'),
                          'state':state,'verified':True})
            print(json.dumps({'event':'pair_preserved','model':pairs[-1]['model'],'step':state.get('step')}),flush=True)
        for path in paths:
            if path in handled:
                continue
            content=path.read_bytes()
            if path.suffix=='.jsonl':
                content=content[:content.rfind(b'\n')+1]
                for line in content.splitlines():
                    json.loads(line)
            add(path,content)
    finally:
        for archive in archives:
            archive.close()
    archive_records={}
    for i in range(parts):
        temporary=output/f'part-{i:02d}.tar.partial'
        final=output/f'part-{i:02d}.tar'
        temporary.replace(final)
        with final.open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        archive_records[final.name]={'bytes':final.stat().st_size,'sha256':digest}
    result={'runs':str(runs),'started_unix':started,'completed_unix':time.time(),
            'files':files,'archives':archive_records,'checkpoint_pairs':pairs,'unique_blobs':len(blobs),
            'excluded':'symlinks and .tmp/.partial; JSONL unfinished tail trimmed; active W&B files are captured snapshots'}
    (output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (output/'complete.json').write_text(json.dumps({'files':len(files),'unique_blobs':len(blobs),
        'archives':archive_records,'checkpoint_pairs':len(pairs)}),encoding='utf-8')
    print(json.dumps({'event':'snapshot_complete','files':len(files),'unique_blobs':len(blobs),
                      'bytes':sum(x['bytes'] for x in archive_records.values())}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parts',type=int,choices=[2,4],default=4)
    args=parser.parse_args()
    torch.set_num_threads(2)
    capture(args.runs,args.output,args.parts)
