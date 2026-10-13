import random
import torch
from src.asr.data import SpeechDataset,augment_waveform


def speech_like():
    t=torch.arange(32000)/16000
    return (0.07*torch.sin(2*torch.pi*190*t)+0.04*torch.sin(2*torch.pi*530*t))*(0.6+0.4*torch.sin(2*torch.pi*3*t).square())


def normalized_features(wave):
    spec=SpeechDataset([]).mel(wave).clamp_min(1e-10).log()
    return (spec-spec.mean())/spec.std().clamp_min(1e-5)


def test_effective_gain_survives_frontend_normalization():
    wave=speech_like()
    changed=augment_waveform(wave,'gain',random.Random(3),profile='effective_v2')
    assert (normalized_features(changed)-normalized_features(wave)).square().mean().sqrt()>0.005
    assert changed.shape==wave.shape and changed.abs().max()<=1


def test_effective_reverb_has_controlled_nontrivial_energy():
    wave=speech_like()
    changed=augment_waveform(wave,'reverb',random.Random(7),torch.Generator().manual_seed(7),profile='effective_v2')
    relative=(changed-wave).square().mean().sqrt()/wave.square().mean().sqrt()
    assert 0.03<relative<0.4
    assert torch.isfinite(changed).all() and changed.shape==wave.shape


def test_legacy_gain_remains_exactly_reproducible():
    wave=speech_like()
    expected=(wave*random.Random(3).uniform(.75,1.25)).clamp(-1,1)
    assert torch.equal(augment_waveform(wave,'gain',random.Random(3),profile='legacy'),expected)

def test_fresh_phase_offsets_data_order_and_augmentations_without_changing_dev_seed(tmp_path,monkeypatch):
    import numpy as np
    import soundfile as sf
    from omegaconf import OmegaConf
    from src.asr import runtime
    (tmp_path/'data/audio').mkdir(parents=True)
    (tmp_path/'data/transcriptions').mkdir()
    sf.write(tmp_path/'data/audio/one.wav',speech_like().numpy(),16000)
    (tmp_path/'data/transcriptions/one.txt').write_text('a')
    class QuietLog:
        wandb=None
        def __init__(self,*args):pass
        def emit(self,*args):pass
    monkeypatch.setattr(runtime,'RunLog',QuietLog)
    monkeypatch.setattr(runtime,'log_media',lambda *args,**kwargs:None)
    original=runtime.make_loader;loaders=[]
    def capture(*args,**kwargs):
        loader=original(*args,**kwargs);loaders.append(loader);return loader
    monkeypatch.setattr(runtime,'make_loader',capture)
    config=OmegaConf.to_container(OmegaConf.load('src/configs/asr_train.yaml'),resolve=True)
    config.update(device='cpu',custom_train=str(tmp_path/'data'),run_dir=str(tmp_path/'run'),workers=0,cpu_threads=1,valid_parts=[],batch_size=1,epochs=1,max_steps=1,end_utc=None,bf16=False,data_epoch_offset=80,augmentation_profile='effective_v2')
    config['model'].update(hidden=8,layers=1,conv_channels=2)
    runtime.train(config)
    assert loaders[0].batch_sampler.epoch==80
    assert int(loaders[0].dataset.epoch[0])==80
    assert loaders[0].dataset.seed==config['seed']==1
