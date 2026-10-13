import torch
from src.asr.data import augment_waveform
from src.asr.model import DeepSpeech2CTC


def test_fast_utterance_speed_up_keeps_required_ctc_timesteps():
    class MaximumSpeed:
        def uniform(self, low, high):
            return high

    # Regression: train100 460-172359-0021 has 7.095s and needs164 CTC steps.
    wave = torch.zeros(round(7.095*16000))
    required = 164
    unconstrained = augment_waveform(wave, 'speed', rng=MaximumSpeed())
    constrained = augment_waveform(wave, 'speed', rng=MaximumSpeed(),
                                   min_samples=160*(4*required-4))
    lengths = torch.tensor([len(unconstrained)//160+1, len(constrained)//160+1])
    available = DeepSpeech2CTC.transform_input_lengths(lengths)
    assert available[0] < required
    assert available[1] >= required
    assert len(constrained) < len(wave)  # Still a genuine speed-up.
