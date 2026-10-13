"""Enroll immutable off-server reserves without duplicating every full resume in RAM.

The operator must verify each complete model/full pair before publishing its
record. The selected full checkpoint is fetched and matched again before test.
"""
import hashlib
import io
from pathlib import Path
import re
import torch


def add_cached_candidates(sources, records, output, freeze):
    by_sha={item['sha256']:name for name,item in sources.items()}
    for item in records:
        for key in ['model_sha256','resume_sha256','snapshot_proof_sha256']:
            if not re.fullmatch('[a-f0-9]{64}',item.get(key,'')):
                raise ValueError('Invalid cached candidate '+key)
        if item.get('matching_model_full_tensors_verified') is not True:
            raise ValueError('Full checkpoint reserve not verified')
        source=Path(item['model_path']);content=source.read_bytes()
        digest=hashlib.sha256(content).hexdigest()
        if digest!=item['model_sha256']:raise ValueError('Cached candidate model SHA mismatch')
        payload=torch.load(io.BytesIO(content),map_location='cpu',weights_only=True)
        if payload['state']!=item['state']:raise ValueError('Cached candidate state mismatch')
        key='lr_candidate_cached_'+digest
        alias={'candidate':key,'source':item['source_run'],'run':item['run'],
               'checkpoint_kind':'off_server_reserve','model_source':str(source),
               'resume_source':item['resume_path'],'resume_sha256':item['resume_sha256'],
               'snapshot_proof_sha256':item['snapshot_proof_sha256']}
        if digest in by_sha:
            sources[by_sha[digest]]['aliases'].append(alias)
        else:
            destination=Path(output)/'candidates'/key/'model.pt'
            destination.parent.mkdir(exist_ok=True,parents=True)
            freeze(source,destination)
            sources[key]={**alias,'sha256':digest,'aliases':[alias]}
            by_sha[digest]=key
    return sources
