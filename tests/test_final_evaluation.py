import pytest
import torch

from ops.final_evaluation import choose_checkpoint, choose_decoder, copy_matching_resume, freeze_model


def result(errors, words):
    return {'word_errors': errors, 'words': words}


def test_model_selection_pools_both_full_dev_splits():
    scores = {
        'main': {'dev-clean': result(21, 100), 'dev-other': result(20, 100)},
        'candidate': {'dev-clean': result(20, 100), 'dev-other': result(80, 100)},
    }
    assert choose_checkpoint(scores) == 'main'


def test_decoder_selection_pools_errors_and_rejects_mixed_checkpoints():
    scores = {
        'dev-clean': {'checkpoint_sha256': 'same', 'decoders': {
            'greedy': result(0, 10), 'beam20_lm0.1_bonus1.0': result(3, 10)}},
        'dev-other': {'checkpoint_sha256': 'same', 'decoders': {
            'greedy': result(40, 100), 'beam20_lm0.1_bonus1.0': result(10, 100)}},
    }
    chosen = choose_decoder(scores)
    assert chosen['name'] == 'beam20_lm0.1_bonus1.0'
    assert chosen['wer'] == 13 / 110
    assert chosen['width'] == 20 and chosen['lm_weight'] == .1
    scores['dev-other']['checkpoint_sha256'] = 'different'
    with pytest.raises(ValueError, match='checkpoint'):
        choose_decoder(scores)
    scores['test-clean'] = scores['dev-clean']
    with pytest.raises(ValueError, match='dev'):
        choose_decoder(scores)


def test_resume_export_must_match_frozen_model(tmp_path):
    model = tmp_path / 'model.pt'
    resume = tmp_path / 'best.pt'
    output = tmp_path / 'selected_resume.pt'
    payload = {'model': {'weight': torch.tensor([1.])}, 'state': {'step': 7}}
    torch.save(payload, model)
    torch.save({**payload, 'optimizer': {}}, resume)
    snapshot = tmp_path / 'snapshot.pt'
    snapshot.write_bytes(b'interrupted model write')
    assert freeze_model(model, snapshot) == model.read_bytes()
    assert snapshot.read_bytes() == model.read_bytes()
    copy_matching_resume(model, resume, output)
    assert output.read_bytes() == resume.read_bytes()
    torch.save({**payload, 'model': {'weight': torch.tensor([2.])}}, resume)
    with pytest.raises(ValueError, match='match'):
        copy_matching_resume(model, resume, tmp_path / 'invalid.pt')
    assert not (tmp_path / 'invalid.pt').exists()


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    import hashlib
    import sys
    from types import SimpleNamespace
    from ops import final_evaluation
    from scripts import evaluate_decoders
    class FinishedClock:
        @staticmethod
        def now(tz):
            return datetime(2026, 10, 13, 9, 31, tzinfo=timezone.utc)
    monkeypatch.setattr(final_evaluation, 'datetime', FinishedClock)
    monkeypatch.setattr(final_evaluation, 'own_trainers', lambda: [])
    monkeypatch.setitem(sys.modules, 'fcntl', SimpleNamespace(
        flock=lambda *args: None, LOCK_EX=1, LOCK_NB=2))
    for name, weight in [('main', 1.), ('lr_candidate_finetune460', 2.)]:
        directory = tmp_path/'runs'/name
        directory.mkdir(parents=True)
        payload = {'model': {'weight': torch.tensor([weight])}, 'state': {'step': 7}}
        for kind in ('best', 'last'):
            torch.save(payload, directory/f'{kind}_model.pt')
            torch.save({**payload, 'optimizer': {}}, directory/f'{kind}.pt')
    output = tmp_path/'runs/final-evaluation'
    caches, test_calls, candidate_calls = {}, [], []
    errors_by_weight = {1.: 20, 2.: 10, 3.: 1}
    def cache(content, root, part, limit, destination, device, batch_size, workers):
        assert limit == 0
        if part.startswith('test'):
            assert (output/'selected_model.pt').read_bytes() == content
            assert (output/'frozen_decoder.json').exists()
            test_calls.append(part)
        path = destination/f'{part}.npz'
        import io
        weight = torch.load(io.BytesIO(content), weights_only=True)['model']['weight'].item()
        caches[path] = (hashlib.sha256(content).hexdigest(), weight)
        if 'candidates' in destination.parts:
            candidate_calls.append((destination.name, part))
        return path
    def compare(path, destination, widths, topk, workers, *args):
        if not widths:
            errors = errors_by_weight[caches[path][1]]
            decoders = {'greedy': result(errors, 100)}
        else:
            decoders = {'greedy': result(10, 100), 'beam20': result(9, 100),
                        'beam20_lm0.1_bonus1.0': result(8, 100)}
        return {'checkpoint_sha256': caches[path][0], 'decoders': decoders}
    monkeypatch.setattr(evaluate_decoders, 'cache_emissions', cache)
    monkeypatch.setattr(evaluate_decoders, 'compare', compare)
    monkeypatch.setattr(sys, 'argv', ['final_evaluation', '--root', str(tmp_path)])
    return SimpleNamespace(run=final_evaluation.main, root=tmp_path, output=output,
                           test_calls=test_calls, candidate_calls=candidate_calls)


def test_final_pipeline_freezes_model_and_decoder_before_test(pipeline):
    import json
    pipeline.run()
    output = pipeline.output
    assert pipeline.test_calls == ['test-clean', 'test-other']
    complete = json.loads((output/'complete.json').read_text())
    assert complete['selected'] == 'lr_candidate_finetune460__best'
    assert complete['decoder']['name'] == 'beam20_lm0.1_bonus1.0'


def test_checkpoint_selection_uses_total_words_instead_of_mean_split_rates():
    scores={
        "main":{"dev-clean":result(1,10),"dev-other":result(100,1000)},
        "candidate":{"dev-clean":result(0,10),"dev-other":result(110,1000)},
    }
    assert choose_checkpoint(scores)=="main"


def write_pair(directory, kind, weight, step=8):
    payload = {'model': {'weight': torch.tensor([weight])}, 'state': {'step': step}}
    torch.save(payload, directory/f'{kind}_model.pt')
    torch.save({**payload, 'optimizer': {'matching': kind}}, directory/f'{kind}.pt')


def test_last_candidate_can_win_and_exports_matching_resume(pipeline):
    import json
    directory = pipeline.root/'runs/main'
    write_pair(directory, 'last', 3.)
    pipeline.run()
    selection = json.loads((pipeline.output/'selection.json').read_text())
    assert selection['selected'] == 'main__last'
    chosen = selection['candidates']['main__last']
    assert chosen['run'] == 'main'
    assert chosen['checkpoint_kind'] == 'last'
    assert chosen['model_source'] == str(directory/'last_model.pt')
    assert chosen['resume_source'] == str(directory/'last.pt')
    assert (pipeline.output/'selected_resume.pt').read_bytes() == (directory/'last.pt').read_bytes()
    assert (pipeline.output/'selected_model.pt').read_bytes() == (directory/'last_model.pt').read_bytes()
    assert pipeline.test_calls == ['test-clean', 'test-other']


def test_identical_bytes_are_scored_once_with_deterministic_best_alias(pipeline):
    import json
    main = pipeline.root/'runs/main'
    other = pipeline.root/'runs/lr_candidate_finetune460'
    # Force byte identity rather than assuming torch.save uses the same archive name.
    for directory in (main, other):
        for kind in ('best', 'last'):
            (directory/f'{kind}_model.pt').write_bytes((main/'best_model.pt').read_bytes())
            (directory/f'{kind}.pt').write_bytes((main/'best.pt').read_bytes())
    pipeline.run()
    selection = json.loads((pipeline.output/'selection.json').read_text())
    assert selection['selected'] == 'lr_candidate_finetune460__best'
    assert len(selection['scores']) == 1
    assert len(pipeline.candidate_calls) == 2
    metadata = selection['candidates'][selection['selected']]
    assert {alias['candidate'] for alias in metadata['aliases']} == {
        'main__best', 'main__last', 'lr_candidate_finetune460__best',
        'lr_candidate_finetune460__last'}
    assert metadata['checkpoint_kind'] == 'best'


@pytest.mark.parametrize('filename,damage', [
    ('last_model.pt', 'missing'), ('best_model.pt', 'missing'),
    ('last.pt', 'missing'), ('best.pt', 'missing'),
    ('last_model.pt', 'corrupt'), ('best_model.pt', 'corrupt'),
    ('last.pt', 'corrupt'), ('best.pt', 'corrupt'),
])
def test_missing_or_corrupt_source_cannot_start_test(pipeline, filename, damage):
    # Both main snapshots compete; corruption must block even a losing alias.
    directory = pipeline.root/'runs/main'
    write_pair(directory, 'best', 3.)
    write_pair(directory, 'last', 3.)
    source = directory/filename
    if damage == 'missing':
        source.unlink()
    else:
        source.write_bytes(b'not a torch checkpoint')
    with pytest.raises((ValueError, FileNotFoundError), match=filename):
        pipeline.run()
    assert pipeline.test_calls == []
    assert not (pipeline.output/'complete.json').exists()


def test_mismatching_last_resume_cannot_start_test(pipeline):
    directory = pipeline.root/'runs/main'
    write_pair(directory, 'last', 3.)
    (directory/'last.pt').write_bytes((directory/'best.pt').read_bytes())
    with pytest.raises(ValueError, match='match'):
        pipeline.run()
    assert pipeline.test_calls == []


def test_existing_snapshot_cannot_hide_corrupt_source(pipeline):
    directory = pipeline.root/'runs/main'
    destination = pipeline.output/'candidates/main__last'
    destination.mkdir(parents=True)
    freeze_model(directory/'last_model.pt', destination/'model.pt')
    (directory/'last_model.pt').write_bytes(b'interrupted source write')
    with pytest.raises(ValueError, match='last_model.pt'):
        pipeline.run()
    assert pipeline.test_calls == []


def test_completed_run_preserves_artifacts_without_reopening_test(pipeline):
    pipeline.run()
    complete = (pipeline.output/'complete.json').read_bytes()
    model = (pipeline.output/'selected_model.pt').read_bytes()
    calls = list(pipeline.candidate_calls)
    (pipeline.root/'runs/main/last_model.pt').unlink()
    pipeline.run()
    assert (pipeline.output/'complete.json').read_bytes() == complete
    assert (pipeline.output/'selected_model.pt').read_bytes() == model
    assert pipeline.candidate_calls == calls
    assert pipeline.test_calls == ['test-clean', 'test-other']
