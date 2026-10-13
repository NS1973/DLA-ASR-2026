import numpy as np
import pytest
import soundfile as sf
import torch


def test_t4_uses_fp32_and_ampere_supports_native_bf16(monkeypatch):
    from src.asr.runtime import supports_native_bf16
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda device:(7,5))
    assert not supports_native_bf16(torch.device('cuda'))
    assert not supports_native_bf16(torch.device('cpu'))
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda device:(8,0))
    assert supports_native_bf16(torch.device('cuda'))


def test_metrics_reject_missing_prediction(tmp_path):
    from calc_metrics import directory_metrics
    refs, preds = tmp_path / 'refs', tmp_path / 'preds'
    refs.mkdir(); preds.mkdir()
    (refs / 'one.txt').write_text('hello')
    with pytest.raises(ValueError, match='ID mismatch'):
        directory_metrics(refs, preds)


def test_inference_saves_ids_without_ground_truth(tmp_path):
    from src.asr.model import DeepSpeech2CTC
    from src.asr.runtime import export_model
    from src.asr.infer import infer
    (tmp_path / 'input' / 'audio').mkdir(parents=True)
    sf.write(tmp_path / 'input/audio/known-id.wav', np.sin(np.arange(16000) * .02) * .1, 16000)
    checkpoint = tmp_path / 'model.pt'
    export_model(DeepSpeech2CTC(hidden=8, layers=1, conv_channels=2), checkpoint, {'step':0})
    result = infer(dict(checkpoint=str(checkpoint), device='cpu', cpu_threads=1, custom_dir=str(tmp_path / 'input'),
                        limit=None, batch_size=1, workers=0, output_dir=str(tmp_path / 'preds'), decoder='greedy',
                        beam_workers=1, bf16=False))
    assert (tmp_path / 'preds/known-id.txt').exists()
    assert result['decoded'] == 1
    assert result['metrics'] is None
