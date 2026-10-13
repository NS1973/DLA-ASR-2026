from scripts import evaluate_decoders
import hashlib
import numpy as np
import pytest
import torch


def test_all_splits_use_one_checkpoint_snapshot(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'best_model.pt'
    checkpoint.write_bytes(b'first checkpoint')
    snapshots = []

    def cache(snapshot, root, part, limit, destination, device, batch_size, workers):
        snapshots.append((part, snapshot))
        # Simulate the trainer atomically updating best while decoding split one.
        replacement = tmp_path / 'replacement.pt'
        replacement.write_bytes(b'new checkpoint')
        replacement.replace(checkpoint)
        return destination / (part + '.npz')

    monkeypatch.setattr(evaluate_decoders, 'cache_emissions', cache)
    monkeypatch.setattr(evaluate_decoders, 'compare', lambda *args: None)
    monkeypatch.setattr('sys.argv', ['evaluate_decoders', '--checkpoint', str(checkpoint),
                                  '--output-dir', str(tmp_path / 'output'), '--device', 'cpu'])
    evaluate_decoders.main()
    assert snapshots == [('dev-clean', b'first checkpoint'),
                         ('dev-other', b'first checkpoint')]


def test_emissions_cache_recovers_truncation_and_interrupted_write(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_decoders, 'librispeech_entries', lambda *args: [{}])
    monkeypatch.setattr(evaluate_decoders, 'SpeechDataset', lambda entries: entries)
    batch = {'spectrogram': torch.zeros(1, 80, 10), 'spectrogram_length': torch.tensor([10]),
             'uids': ['example'], 'texts': ['hello']}
    monkeypatch.setattr(evaluate_decoders, 'DataLoader', lambda *args, **kwargs: [batch])
    monkeypatch.setattr(evaluate_decoders, 'load_model', lambda *args: lambda *args: {
        'log_probs': torch.zeros(1, 2, 29), 'log_probs_length': torch.tensor([2])})
    content = b'frozen model bytes'
    def generate():
        return evaluate_decoders.cache_emissions(content, tmp_path, 'dev-clean', 0,
                                                 tmp_path, torch.device('cpu'), 1, 0)
    path = generate()
    path.write_bytes(b'interrupted ZIP')
    assert generate() == path
    with np.load(path, allow_pickle=False) as data:
        assert str(data['checkpoint_sha256']) == hashlib.sha256(content).hexdigest()
        assert data['emissions'].shape == (2, 29)
    path.unlink()
    original = np.savez_compressed
    def interrupted(stream, **kwargs):
        stream.write(b'partial')
        raise OSError('simulated write interruption')
    monkeypatch.setattr(evaluate_decoders.np, 'savez_compressed', interrupted)
    with pytest.raises(OSError, match='interruption'):
        generate()
    assert not path.exists()
    monkeypatch.setattr(evaluate_decoders.np, 'savez_compressed', original)
    assert generate().exists()
