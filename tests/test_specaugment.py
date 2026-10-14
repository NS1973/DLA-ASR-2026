import torch

from src.asr.specaugment import MildSpecAugment


def test_mask_only_valid_frames_and_restore_rng():
    transform = MildSpecAugment().train()
    spec = torch.ones(3, 80, 100)
    lengths = torch.tensor([100, 47, 3])
    torch.manual_seed(42)
    state = torch.get_rng_state()
    result = transform(spec, lengths)
    torch.set_rng_state(state)
    assert torch.equal(result, transform(spec, lengths))
    assert torch.equal(spec, torch.ones_like(spec))
    assert result.shape == spec.shape and (result == 0).any()
    for i, length in enumerate(lengths):
        assert torch.equal(result[i, :, length:], spec[i, :, length:])
        assert result[i, :, :length].sum() > 0
        # Two masks with at most five percent valid-time each.
        entirely_masked = (result[i, :, :length] == 0).all(0).sum()
        assert entirely_masked <= 2 * min(40, int(length * .05))


def test_eval_is_identity_and_one_band_short_input_stays_nonempty():
    transform = MildSpecAugment().eval()
    spec = torch.ones(2, 1, 1)
    lengths = torch.ones(2, dtype=torch.long)
    assert transform(spec, lengths) is spec
    transform.train()
    assert torch.equal(transform(spec, lengths), spec)
