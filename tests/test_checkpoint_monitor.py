import pytest,torch,soundfile as sf
from omegaconf import OmegaConf
from src.asr import runtime

@pytest.mark.parametrize('monitor,expected_epoch,expected_wer',[(None,1,.05),('dev-other',0,.5)])
def test_best_checkpoint_tracks_requested_validation_split(tmp_path,monkeypatch,monitor,expected_epoch,expected_wer):
    root=tmp_path/'data';(root/'audio').mkdir(parents=True);(root/'transcriptions').mkdir()
    sf.write(root/'audio/one.wav',torch.sin(torch.arange(8000)*.1).numpy()*.05,16000)
    (root/'transcriptions/one.txt').write_text('a')
    class QuietLog:
        wandb=None
        def __init__(self,*a):pass
        def emit(self,*a):pass
    monkeypatch.setattr(runtime,'RunLog',QuietLog);monkeypatch.setattr(runtime,'log_media',lambda *a,**kw:None)
    entries=runtime.CustomDirDataset(root).entries
    monkeypatch.setattr(runtime,'librispeech_entries',lambda *a:entries)
    values=iter([.1,.5,.05,.6])
    monkeypatch.setattr(runtime,'evaluate',lambda *a,**kw:({'wer':next(values)},[]))
    cfg=OmegaConf.to_container(OmegaConf.load('src/configs/asr_train.yaml'),resolve=True)
    cfg.update(device='cpu',custom_train=str(root),run_dir=str(tmp_path/'run'),workers=0,cpu_threads=1,batch_size=1,epochs=2,end_utc=None,bf16=False,augment=False)
    cfg['model'].update(hidden=8,layers=1,conv_channels=2)
    if monitor is not None:cfg['checkpoint_monitor']=monitor
    runtime.train(cfg)
    best=torch.load(tmp_path/'run/best.pt',map_location='cpu',weights_only=False)
    assert best['state']['epoch']==expected_epoch
    assert best['state']['best_wer']==expected_wer
