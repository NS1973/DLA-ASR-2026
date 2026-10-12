import torch
import numpy as np
import soundfile as sf


def test_downsampling_lengths_ctc_and_backward():
    from src.asr.model import DeepSpeech2CTC
    model = DeepSpeech2CTC(hidden=16, layers=2, conv_channels=4)
    lengths = torch.tensor([101, 80])
    output = model(torch.randn(2, 80, 101), lengths)
    assert output["log_probs"].shape == (2, 26, 29)
    assert output["log_probs_length"].tolist() == [26, 20]
    loss = torch.nn.CTCLoss(blank=0)(output["log_probs"].transpose(0, 1), torch.tensor([1, 2, 3, 4]), output["log_probs_length"], torch.tensor([2, 2]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_padding_does_not_change_shorter_prediction():
    from src.asr.model import DeepSpeech2CTC
    torch.manual_seed(1)
    model = DeepSpeech2CTC(hidden=16, layers=2, conv_channels=4).eval()
    x = torch.randn(1, 80, 79)
    with torch.no_grad():
        first = model(x, torch.tensor([79]))["log_probs"]
        padded = torch.nn.functional.pad(x, (0, 42))
        second = model(padded, torch.tensor([79]))["log_probs"]
    assert torch.allclose(first[:, :20], second[:, :20], atol=1e-5)


def test_custom_dataset_resamples_stereo_and_accepts_missing_text(tmp_path):
    from src.asr.data import CustomDirDataset, collate
    (tmp_path / "audio").mkdir()
    audio = np.zeros((8000, 2), dtype=np.float32)
    audio[:, 0] = np.sin(np.arange(8000) * .03) * .1
    sf.write(tmp_path / "audio" / "sample.wav", audio, 8000)
    ds = CustomDirDataset(tmp_path)
    item = ds[0]
    assert item["uid"] == "sample"
    assert item["text"] is None
    assert item["spectrogram"].shape[0] == 80
    assert item["waveform"].numel() == 16000
    batch = collate([item])
    assert batch["spectrogram"].shape[0] == 1


def test_augmentations_preserve_valid_audio():
    from src.asr.data import augment_waveform
    wave = torch.sin(torch.arange(16000) * .07) * .1
    for kind in ["gain", "noise", "speed", "reverb"]:
        result = augment_waveform(wave, kind)
        assert torch.isfinite(result).all()
        assert result.numel() > 10000
        assert result.abs().max() <= 1
        assert result.shape != wave.shape or not torch.equal(result, wave)
