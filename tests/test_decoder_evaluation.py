from scripts import evaluate_decoders


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
