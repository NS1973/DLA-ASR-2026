import pytest
from ops.final_evaluation import choose_checkpoint, choose_decoder


def metrics(errors):
    return {'word_errors': errors, 'words': 100}


def test_other_model_selection_preserves_clean_guard():
    scores = {
        'clean_winner': {'dev-clean': metrics(5), 'dev-other': metrics(40)},
        'other_winner': {'dev-clean': metrics(19), 'dev-other': metrics(30)},
        'unsafe_clean': {'dev-clean': metrics(21), 'dev-other': metrics(10)},
    }
    assert choose_checkpoint(scores, objective='other') == 'other_winner'
    assert choose_checkpoint(scores) == 'unsafe_clean'


def test_other_model_selection_requires_an_eligible_candidate():
    scores = {'unsafe': {'dev-clean': metrics(21), 'dev-other': metrics(10)}}
    with pytest.raises(ValueError, match='clean'):
        choose_checkpoint(scores, objective='other')


def decoder_scores():
    return {
        'dev-clean': {'checkpoint_sha256': 'same', 'decoders': {
            'greedy': metrics(5), 'beam10_lm0.1_bonus1.0': metrics(19), 'beam50': metrics(21)}},
        'dev-other': {'checkpoint_sha256': 'same', 'decoders': {
            'greedy': metrics(40), 'beam10_lm0.1_bonus1.0': metrics(30), 'beam50': metrics(10)}},
    }


def test_other_decoder_selection_uses_other_and_reports_actual_objective():
    chosen = choose_decoder(decoder_scores(), objective='other')
    assert chosen['name'] == 'beam10_lm0.1_bonus1.0'
    assert chosen['wer'] == .30
    assert chosen['pooled_wer'] == .245
    assert chosen['dev_clean_wer'] == .19
    assert chosen['width'] == 10 and chosen['lm_weight'] == .1
    assert 'dev-other' in chosen['selection'] and '20%' in chosen['selection']


def test_other_decoder_selection_requires_an_eligible_setting():
    scores = decoder_scores()
    for setting in scores['dev-clean']['decoders'].values():
        setting['word_errors'] = 21
    with pytest.raises(ValueError, match='clean'):
        choose_decoder(scores, objective='other')
