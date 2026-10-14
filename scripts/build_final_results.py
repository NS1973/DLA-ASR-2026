"""Derive report inputs from a SHA-verified closed evaluation capture."""
from pathlib import Path
from datetime import datetime
import hashlib,json,math,sys,tarfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops.final_evaluation import choose_checkpoint,choose_decoder
from ops.delivery_evidence import validate_terminal_proof

COUNTS={'dev-clean':(2703,54402,288456),'dev-other':(2864,50948,265951),
        'test-clean':(2620,52576,281530),'test-other':(2939,52343,272758)}

def check_metrics(metrics,part):
    n,words,chars=COUNTS[part]
    assert (metrics['utterances'],metrics['words'],metrics['chars'])==(n,words,chars)
    for field,errors,denominator in [('wer','word_errors',words),('cer','char_errors',chars)]:
        assert type(metrics[errors]) is int and metrics[errors]>=0
        assert math.isfinite(metrics[field]) and math.isclose(metrics[field],metrics[errors]/denominator,abs_tol=1e-12,rel_tol=1e-12)

def build_results(archive,proof_path,require_terminal_proof=True,expected_stage_count=22):
    archive=Path(archive);proof=json.loads(Path(proof_path).read_text())
    with archive.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    if sha!=proof['sha256'] or archive.stat().st_size!=proof['bytes']:
        raise ValueError('Evaluation archive SHA/size mismatch')
    assert proof['all_logical_file_sha_verified']
    prefix='runs/final-evaluation/'
    wanted={'complete.json','selection.json','frozen_decoder.json','events.jsonl',
            'acceptance/resume.json','acceptance/trainers-closed.json'}
    for part in COUNTS:
        wanted.add(('dev-decoders/' if part.startswith('dev') else 'test/')+part+'/comparison.json')
        wanted.add('analysis/'+part+'.json')
    files={}
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            if member.name.startswith(prefix) and member.name[len(prefix):] in wanted:
                files[member.name[len(prefix):]]=tar.extractfile(member).read()
    if require_terminal_proof and 'acceptance/trainers-closed.json' not in files:
        raise ValueError('Actual terminal-PID proof absent from new final evaluation')
    complete=json.loads(files['complete.json']);selection=json.loads(files['selection.json'])
    decoder=json.loads(files['frozen_decoder.json']);resume=json.loads(files['acceptance/resume.json'])
    assert complete==proof['complete'] and complete['decoder']==decoder
    assert selection['selected']==choose_checkpoint(selection['scores'],objective='other')
    assert complete['selected']==selection['selected']
    assert complete['model_sha256']==selection['candidates'][selection['selected']]['sha256']==resume['model_sha256']==decoder['checkpoint_sha256']
    assert resume['exact_model_optimizer_scheduler_rng_and_gradients']
    for candidate,parts in selection['scores'].items():
        assert candidate in selection['candidates']
        for part in ['dev-clean','dev-other']:check_metrics(parts[part],part)
    sweeps={};comparisons={};analyses={}
    for part in COUNTS:
        key=('dev-decoders/' if part.startswith('dev') else 'test/')+part+'/comparison.json'
        comparison=json.loads(files[key]);assert comparison['checkpoint_sha256']==complete['model_sha256']
        assert comparison['utterances']==COUNTS[part][0]
        for metrics in comparison['decoders'].values():check_metrics(metrics,part)
        assert decoder['name'] in comparison['decoders'];comparisons[part]=comparison
        if part.startswith('dev'):
            assert len(comparison['decoders'])==13;sweeps[part]=comparison
        analysis=json.loads(files['analysis/'+part+'.json']);analyses[part]=analysis
        selected=comparison['decoders'][decoder['name']]
        assert analysis['metrics']=={k:v for k,v in selected.items() if k!='seconds'}
        assert sum(analysis['error_counts'].values())==selected['word_errors']
    assert choose_decoder(sweeps,objective='other')['name']==decoder['name']
    events=[json.loads(line) for line in files['events.jsonl'].splitlines()]
    model_events=[x for x in events if x['event']=='model_frozen']
    decoder_events=[x for x in events if x['event']=='decoder_frozen']
    tests=[x for x in events if x['event']=='test_evaluated']
    assert len(model_events)==len(decoder_events)==1 and {x['split'] for x in tests}=={'test-clean','test-other'}
    stamp=lambda x:datetime.fromisoformat(x['utc'])
    assert stamp(model_events[0])<=stamp(decoder_events[0])<=min(map(stamp,tests))
    terminal=None
    if 'acceptance/trainers-closed.json' in files:
        terminal=json.loads(files['acceptance/trainers-closed.json'])
        verified=validate_terminal_proof(terminal,expected_stage_count)
        assert verified<=stamp(model_events[0])
    result={**complete,'status':'evaluated_frozen_candidate; resource/Colab/final-packaging_acceptance_pending',
            'selection_criterion':selection['selection'],'resume_sha256':resume['resume_sha256'],
            'dev':{p:comparisons[p]['decoders'][decoder['name']] for p in ['dev-clean','dev-other']},
            'test':{p:comparisons[p]['decoders'][decoder['name']] for p in ['test-clean','test-other']},
            'quality_targets_met':{p:comparisons[p]['decoders'][decoder['name']]['wer']<=limit for p,limit in [('test-clean',.2),('test-other',.3)]},
            'unique_candidates':len(selection['candidates']),
            'candidate_aliases':sum(len(x.get('aliases',[x])) for x in selection['candidates'].values()),
            'selection':selection,'all_decoder_comparisons':comparisons,'error_analysis':analyses,
            'freeze_before_test_verified':True,'actual_terminal_PID_proof':terminal,
            'evaluation_archive_sha256':sha,'evaluation_resume_acceptance':resume}
    return result

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--archive',required=True);parser.add_argument('--proof',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--expected-stage-count',type=int,default=22)
    args=parser.parse_args();output=Path(args.output);assert not output.exists()
    result=build_results(args.archive,args.proof,expected_stage_count=args.expected_stage_count);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':str(output),'model_sha256':result['model_sha256'],'test':result['test']}),flush=True)
