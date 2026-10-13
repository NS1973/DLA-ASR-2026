from ops import supervisor


def test_oom_reduces_explicit_batch_and_preserves_effective_batch(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(supervisor, 'RUNS', tmp_path)
    class Child:
        pid = 123
        def __init__(self, command, **kwargs):
            self.args = command
            calls.append(command)
            if len(calls) == 1:
                kwargs['stdout'].write('CUDA out of memory\n')
                kwargs['stdout'].flush()
        def wait(self):
            return 1 if len(calls) == 1 else 0
    monkeypatch.setattr(supervisor.subprocess, 'Popen', Child)
    assert supervisor.run('lr_candidate_finetune460', 1, ['batch_size=128'])
    assert 'batch_size=128' in calls[0]
    assert 'batch_size=64' in calls[1]
    assert 'batch_size=128' not in calls[1]
    assert 'accumulation=2' in calls[1]
