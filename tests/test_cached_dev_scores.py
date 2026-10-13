import json
import pytest
from ops.final_evaluation import read_cached_dev_comparison


def report():
    return {'checkpoint_sha256': 'model', 'utterances': 2703, 'decoders': {'greedy': {
        'utterances': 2703, 'words': 54402, 'word_errors': 1, 'wer': 1/54402,
        'chars': 288456, 'char_errors': 1, 'cer': 1/288456}}}


def test_reuse_requires_exact_model_and_full_corpus_counts(tmp_path):
    path = tmp_path/'comparison.json'
    path.write_text(json.dumps(report()))
    assert read_cached_dev_comparison(path, 'model', 'dev-clean', {'greedy'}) == report()


@pytest.mark.parametrize('damage', ['model', 'subset', 'word_count', 'rate', 'settings'])
def test_invalid_cached_dev_results_are_rejected(tmp_path, damage):
    data = report()
    if damage == 'model': data['checkpoint_sha256'] = 'different'
    if damage == 'subset': data['utterances'] = 256
    if damage == 'word_count': data['decoders']['greedy']['words'] = 100
    if damage == 'rate': data['decoders']['greedy']['wer'] = .5
    if damage == 'settings': data['decoders']['beam10'] = data['decoders']['greedy']
    path = tmp_path/'comparison.json'; path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        read_cached_dev_comparison(path, 'model', 'dev-clean', {'greedy'})


def test_reused_decoder_sweep_requires_exact_lm(tmp_path):
    data = report();data['lm_sha256'] = 'wrong'
    path = tmp_path/'comparison.json';path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='LM'):
        read_cached_dev_comparison(path, 'model', 'dev-clean', {'greedy'}, lm_sha='official')
