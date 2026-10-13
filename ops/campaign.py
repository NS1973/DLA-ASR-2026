"""Continue the campaign even when the initial epoch budget ends early."""
from datetime import datetime, timezone

CUTOFF = datetime(2026,10,13,3,tzinfo=timezone.utc)
END = '2026-10-13T09:30:00+00:00'


def continue_training(run, data_directory, best_checkpoint, now=None, *,
                      gpu=0, prefix='', initial_lr=1e-4):
    now = now or (lambda:datetime.now(timezone.utc))
    def phase(name, deadline, parts, epochs):
        iteration=0
        result=True
        while now() < deadline:
            run_name=prefix+(name if iteration==0 else f'{name}_continue{iteration}')
            result=run(run_name,gpu,['data_root=/work/data/datasets',f'train_parts={parts}',
                                  f'init_checkpoint={best_checkpoint()}',f'epochs={epochs}',
                                  f'lr={initial_lr if iteration==0 else 0.00005}',
                                  'warmup_steps=100',f'end_utc={deadline.isoformat()}'])
            iteration+=1
        return result
    phase('refine100',CUTOFF,'[train-clean-100]',120)
    marker = data_directory/'train-clean-360.ready'
    extra_ready = marker.exists() and marker.stat().st_mtime < CUTOFF.timestamp()
    name = 'finetune460' if extra_ready else 'finetune100'
    parts = '[train-clean-100,train-clean-360]' if extra_ready else '[train-clean-100]'
    return phase(name,datetime.fromisoformat(END),parts,40 if extra_ready else 120)
