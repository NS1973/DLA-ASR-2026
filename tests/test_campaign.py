from datetime import datetime, timezone
from pathlib import Path
import os
from ops import campaign


def test_finished_initial_schedule_still_starts_new_training(tmp_path):
    calls = []
    current=[datetime(2026,10,13,2,45,tzinfo=timezone.utc)]
    def run(name, gpu, options):
        calls.append((name, options))
        current[0]=datetime.fromisoformat(next(x.split('=',1)[1] for x in options if x.startswith('end_utc=')))
        return True
    campaign.continue_training(run, tmp_path, lambda: Path('/best.pt'),
                               now=lambda:current[0])
    assert [name for name,_ in calls] == ['refine100','finetune100']
    assert 'end_utc=2026-10-13T03:00:00+00:00' in calls[0][1]
    assert 'end_utc=2026-10-13T09:30:00+00:00' in calls[1][1]
    assert all(not any(option.startswith('resume=') for option in opts) for _,opts in calls)


def test_extra_data_must_be_completely_ready_before_cutoff(tmp_path):
    marker = tmp_path/'train-clean-360.ready'
    marker.touch()
    cutoff = datetime(2026,10,13,3,tzinfo=timezone.utc).timestamp()
    for ready_time, expected in [(cutoff-1,'finetune460'),(cutoff,'finetune100')]:
        os.utime(marker,(ready_time,ready_time))
        calls=[]
        current=[datetime(2026,10,13,3,1,tzinfo=timezone.utc)]
        def run(name,gpu,opts):
            calls.append((name,opts))
            current[0]=datetime(2026,10,13,9,30,tzinfo=timezone.utc)
        campaign.continue_training(run,tmp_path,
                                   lambda: Path('/best.pt'),
                                   now=lambda:current[0])
        assert calls[0][0] == expected
        assert ('train_parts=[train-clean-100,train-clean-360]' in calls[0][1]) == (expected=='finetune460')


def test_final_epoch_budget_ending_early_repeats_until_deadline(tmp_path):
    current=[datetime(2026,10,13,3,1,tzinfo=timezone.utc)]
    calls=[]
    def run(name,gpu,options):
        calls.append(name)
        current[0]=datetime(2026,10,13,4 if len(calls)==1 else 9,30,tzinfo=timezone.utc)
    campaign.continue_training(run,tmp_path,lambda:Path('/best.pt'),now=lambda:current[0])
    assert calls==['finetune100','finetune100_continue1']
