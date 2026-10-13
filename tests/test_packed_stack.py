import copy

import torch

import src.asr.model as model_module
from src.asr.model import DeepSpeech2CTC


def test_training_stack_packs_once_and_preserves_valid_outputs_and_gradients(monkeypatch):
    torch.manual_seed(7)
    slow = DeepSpeech2CTC(n_feats=8, n_tokens=6, hidden=8, layers=2, conv_channels=2, dropout=0).train()
    fast = copy.deepcopy(slow)
    fast.packed_stack = True
    spec = torch.randn(3, 8, 37)
    lengths = torch.tensor([37, 21, 8])
    expected = slow(spec, lengths)
    calls = []
    original = model_module.pack_padded_sequence
    def tracked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(model_module, 'pack_padded_sequence', tracked)
    actual = fast(spec, lengths)
    assert len(calls) == 1
    valid = torch.arange(actual['log_probs'].shape[1])[None] < actual['log_probs_length'][:, None]
    torch.testing.assert_close(actual['log_probs'][valid], expected['log_probs'][valid], atol=2e-6, rtol=2e-5)
    actual['log_probs'][valid].square().sum().backward()
    expected['log_probs'][valid].square().sum().backward()
    for (name, parameter), (other_name, other_parameter) in zip(fast.named_parameters(), slow.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(parameter.grad, other_parameter.grad, atol=2e-5, rtol=2e-4)
    assert fast.spec == slow.spec and fast.state_dict().keys() == slow.state_dict().keys()


def test_fast_training_padding_rng_restore_and_legacy_eval():
    torch.manual_seed(12)
    model = DeepSpeech2CTC(n_feats=8, n_tokens=6, hidden=8, layers=2, conv_channels=2, dropout=.1)
    model.packed_stack = True
    spec = torch.randn(2, 8, 29)
    lengths = torch.tensor([29, 8])
    rng = torch.get_rng_state()
    first = model(spec, lengths)['log_probs']
    torch.set_rng_state(rng)
    repeated = model(spec, lengths)['log_probs']
    torch.testing.assert_close(first, repeated, atol=0, rtol=0)
    model.eval()
    reference = model(spec, lengths)['log_probs']
    model.packed_stack = False
    torch.testing.assert_close(reference, model(spec, lengths)['log_probs'], atol=0, rtol=0)
