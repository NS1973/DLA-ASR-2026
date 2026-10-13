import hashlib, io, json, zipfile
import pytest
import torch
from scripts import collect_initializations as collector


def fixture(tmp_path, legacy=False):
    runs=tmp_path/'runs'; child=runs/'main'; child.mkdir(parents=True)
    content=io.BytesIO(); torch.save({'model':{'weight':torch.ones(2)},'state':{'step':7}},content)
    raw=content.getvalue(); digest=hashlib.sha256(raw).hexdigest()
    parent=runs/'parents'/f'{digest}.pt'; parent.parent.mkdir(); parent.write_bytes(raw)
    event={'event':'initialize_from_model','utc':'2026-10-13T01:00:00+00:00','path':str(runs/'mutable.pt'),'source_state':{'step':7}}
    proof=None
    if legacy:
        proof={'run':'main','source':event['path'],'source_sha256':digest,'source_state':event['source_state'],'snapshot':str(parent),'initialize_utc':event['utc'],'proof':'retrospective evidence'}
    else: event.update(snapshot=str(parent),source_sha256=digest)
    (child/'metrics.jsonl').write_text(json.dumps(event)+'\n')
    return runs,parent,digest,event,proof


def test_collects_exact_immutable_parent_and_legacy_proof_without_mutable_source(tmp_path):
    runs,parent,digest,event,proof=fixture(tmp_path,legacy=True)
    second=dict(event,utc='2026-10-13T02:00:00+00:00')
    with (runs/'main/metrics.jsonl').open('a') as stream:stream.write(json.dumps(second)+'\n')
    second_proof=dict(proof,initialize_utc=second['utc'])
    evidence=runs/'legacy.json'; evidence.write_text(json.dumps([proof,second_proof]))
    output=tmp_path/'initializations.zip'
    summary=collector.collect(runs,output,evidence)
    assert summary['unique_models']==1 and summary['initialization_events']==2
    with zipfile.ZipFile(output) as z:
        assert hashlib.sha256(z.read(f'{digest}.pt')).hexdigest()==digest
        metadata=json.loads(z.read('initializations.json'))
        assert metadata['events'][0]['evidence_kind']=='retrospective'
        assert metadata['events'][0]['event']==event


def test_corrupt_parent_keeps_previous_archive(tmp_path):
    runs,parent,_,_,_=fixture(tmp_path)
    parent.write_bytes(b'corrupt');output=tmp_path/'initializations.zip';output.write_bytes(b'previous verified output')
    with pytest.raises(ValueError,match='SHA-256'):
        collector.collect(runs,output)
    assert output.read_bytes()==b'previous verified output'
    assert not (tmp_path/'initializations.zip.tmp').exists()


def test_missing_legacy_proof_is_reported_instead_of_omitting_parent(tmp_path):
    runs,_,_,_,_=fixture(tmp_path,legacy=True)
    with pytest.raises(ValueError,match='Missing initialization evidence'):
        collector.collect(runs,tmp_path/'initializations.zip')

def test_parent_state_must_match_logged_source_state(tmp_path):
    runs,parent,digest,event,_=fixture(tmp_path)
    event['source_state']={'step':99}
    (runs/'main/metrics.jsonl').write_text(json.dumps(event)+'\n')
    with pytest.raises(ValueError,match='state mismatch'):
        collector.collect(runs,tmp_path/'initializations.zip')
