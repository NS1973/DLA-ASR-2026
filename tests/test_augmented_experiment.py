import sys,types
from pathlib import Path
import pytest
from ops import augmented_experiment as experiment


def test_failed_experiment_can_resume_and_successful_one_is_not_restarted(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'fcntl',types.SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *args:None))
    monkeypatch.setattr(experiment,'RUNS',tmp_path)
    monkeypatch.setattr(experiment,'CODE',Path(__file__).resolve().parents[1])
    monkeypatch.setattr(experiment,'best_checkpoint',lambda:tmp_path/'parent.pt')
    calls=[]
    def training_stub(name,gpu,options):
        calls.append((name,gpu,options));return len(calls)>1
    monkeypatch.setattr(experiment,'run',training_stub)
    with pytest.raises(RuntimeError,match='exhausted'):
        experiment.main()
    experiment.main()
    assert len(calls)==2
    experiment.main()
    assert len(calls)==2
