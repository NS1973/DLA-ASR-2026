"""Continue the campaign even when the initial epoch budget ends early."""
from datetime import datetime, timezone

CUTOFF = datetime(2026,10,13,3,tzinfo=timezone.utc)
END = '2026-10-13T09:30:00+00:00'


def continue_training(run, data_directory, best_checkpoint, now=None):
    now = now or (lambda:datetime.now(timezone.utc))
    if now() < CUTOFF:
        run('refine100',0,['data_root=/work/data/datasets',
                         f'init_checkpoint={best_checkpoint()}',
                         'epochs=120','lr=0.0001','warmup_steps=100',
                         f'end_utc={CUTOFF.isoformat()}'])
    marker = data_directory/'train-clean-360.ready'
    extra_ready = marker.exists() and marker.stat().st_mtime < CUTOFF.timestamp()
    name = 'finetune460' if extra_ready else 'finetune100'
    parts = '[train-clean-100,train-clean-360]' if extra_ready else '[train-clean-100]'
    # Fresh optimizer/scheduler: never resume an already-finished 60-epoch run.
    return run(name,0,['data_root=/work/data/datasets',f'train_parts={parts}',
                       f'init_checkpoint={best_checkpoint()}',
                       f'epochs={40 if extra_ready else 120}',
                       'lr=0.0001','warmup_steps=100',f'end_utc={END}'])
