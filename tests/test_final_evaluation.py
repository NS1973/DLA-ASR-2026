import pytest
import torch

from ops.final_evaluation import choose_checkpoint, choose_decoder, copy_matching_resume, freeze_model


def result(errors, words):
    return {'word_errors': errors, 'words': words}


def test_model_selection_uses_full_dev_clean_only():
    scores = {
        'main': {'dev-clean': result(21, 100), 'dev-other': result(20, 100)},
        'candidate': {'dev-clean': result(20, 100), 'dev-other': result(80, 100)},
    }
    assert choose_checkpoint(scores) == 'candidate'


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


def test_final_pipeline_freezes_model_and_decoder_before_test(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    import hashlib
    import json
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
        torch.save(payload, directory/'best_model.pt')
        torch.save({**payload, 'optimizer': {}}, directory/'best.pt')
    output = tmp_path/'runs/final-evaluation'
    caches, test_calls = {}, []
    def cache(content, root, part, limit, destination, device, batch_size, workers):
        assert limit == 0
        if part.startswith('test'):
            assert (output/'selected_model.pt').read_bytes() == content
            assert (output/'frozen_decoder.json').exists()
            test_calls.append(part)
        path = destination/f'{part}.npz'
        caches[path] = hashlib.sha256(content).hexdigest()
        return path
    def compare(path, destination, widths, topk, workers, *args):
        if not widths:
            errors = 10 if 'lr_candidate_' in str(path) else 20
            decoders = {'greedy': result(errors, 100)}
        else:
            decoders = {'greedy': result(10, 100), 'beam20': result(9, 100),
                        'beam20_lm0.1_bonus1.0': result(8, 100)}
        return {'checkpoint_sha256': caches[path], 'decoders': decoders}
    monkeypatch.setattr(evaluate_decoders, 'cache_emissions', cache)
    monkeypatch.setattr(evaluate_decoders, 'compare', compare)
    monkeypatch.setattr(sys, 'argv', ['final_evaluation', '--root', str(tmp_path)])
    final_evaluation.main()
    assert test_calls == ['test-clean', 'test-other']
    complete = json.loads((output/'complete.json').read_text())
    assert complete['selected'] == 'lr_candidate_finetune460'
    assert complete['decoder']['name'] == 'beam20_lm0.1_bonus1.0'
