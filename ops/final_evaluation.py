"""Freeze the completed campaign, select on full dev, then evaluate test once.

Runs independently of the SSH/chat session. Only this project's files are read
or written; no process is stopped and no service configuration is changed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

DEV = ('dev-clean', 'dev-other')
CUTOFF = datetime(2026, 10, 13, 9, 30, tzinfo=timezone.utc)


def atomic_write(path, content):
    path = Path(path)
    temporary = Path(str(path)+'.tmp')
    try:
        with temporary.open('wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def freeze_model(source, destination):
    if destination.exists():
        try:
            content = destination.read_bytes()
            payload = torch.load(io.BytesIO(content), map_location='cpu', weights_only=True)
            if 'model' not in payload or 'state' not in payload:
                raise ValueError('Incomplete model snapshot')
            return content
        except Exception:
            # This derived snapshot can always be rebuilt from the original.
            pass
    content = source.read_bytes()
    torch.load(io.BytesIO(content), map_location='cpu', weights_only=True)
    atomic_write(destination, content)
    return content


def word_rate(metrics):
    if metrics['words'] <= 0:
        raise ValueError('Dev reference word count must be positive')
    return metrics['word_errors'] / metrics['words']


def choose_checkpoint(scores):
    if not scores or any(set(parts) != set(DEV) for parts in scores.values()):
        raise ValueError('Checkpoint selection requires both dev splits only')
    for parts in scores.values():
        for metrics in parts.values():
            word_rate(metrics)
    return min(scores, key=lambda name: (sum(scores[name][part]['word_errors'] for part in DEV) /
                                        sum(scores[name][part]['words'] for part in DEV),
                                        word_rate(scores[name]['dev-clean']),
                                        word_rate(scores[name]['dev-other']), name))


def choose_decoder(scores):
    if set(scores) != set(DEV):
        raise ValueError('Decoder selection requires both dev splits only')
    if len({part['checkpoint_sha256'] for part in scores.values()}) != 1:
        raise ValueError('Decoder comparison mixes checkpoint hashes')
    names = set(scores['dev-clean']['decoders'])
    if names != set(scores['dev-other']['decoders']) or not names:
        raise ValueError('Both dev splits must have identical decoder settings')
    pooled = {}
    for name in names:
        metrics = [scores[part]['decoders'][name] for part in DEV]
        for item in metrics:
            word_rate(item)
        pooled[name] = sum(x['word_errors'] for x in metrics) / sum(x['words'] for x in metrics)
    name = min(pooled, key=lambda x: (pooled[x], x))
    result = {'name': name, 'wer': pooled[name], 'selection': 'pooled full dev-clean/dev-other corpus WER'}
    if name == 'greedy':
        plain = [setting for setting in names if re.fullmatch(r'beam\d+', setting)]
        width = int(min(plain, key=lambda x: (pooled[x], x))[4:]) if plain else 20
        result.update(decoder='greedy', width=width, lm_weight=None)
    else:
        match = re.fullmatch(r'beam(\d+)(?:_lm([\d.]+)_bonus([\d.]+))?', name)
        if match is None:
            raise ValueError(f'Unknown decoder setting: {name}')
        result.update(decoder='beam', width=int(match[1]),
                      lm_weight=float(match[2]) if match[2] else None,
                      word_bonus=float(match[3]) if match[3] else 1.0)
    return result


def copy_matching_resume(model_path, resume_path, destination):
    """Only load trusted full checkpoints made by this project."""
    model = torch.load(model_path, map_location='cpu', weights_only=True)
    content = Path(resume_path).read_bytes()
    resume = torch.load(io.BytesIO(content), map_location='cpu', weights_only=False)
    if (model['state'] != resume['state'] or model['model'].keys() != resume['model'].keys()
            or any(not torch.equal(value, resume['model'][name]) for name, value in model['model'].items())):
        raise ValueError('Full resume checkpoint does not match frozen model')
    atomic_write(destination, content)


def emit(output, event, **values):
    record = {'utc': datetime.now(timezone.utc).isoformat(), 'event': event, **values}
    text = json.dumps(record)
    with (output/'events.jsonl').open('a') as stream:
        stream.write(text+'\n')
    atomic_write(output/'status.json', text.encode())
    print(text, flush=True)


def own_trainers():
    pids = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            command = (process/'cmdline').read_bytes().split(b'\0')
            if (b'train.py' in command and any(x.startswith(b'run_dir=/work/runs/') for x in command)
                    and (process/'stat').read_text().split(') ', 1)[1].split()[0] != 'Z'):
                pids.append(int(process.name))
        except (FileNotFoundError, PermissionError):
            pass
    return pids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/work'))
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root/'runs/final-evaluation'
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output/'controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output/'complete.json').exists():
            print('Final evaluation is already complete; preserving it.', flush=True)
            return
        emit(output, 'waiting_for_training_cutoff', pid=os.getpid(), cutoff=CUTOFF.isoformat())
        while datetime.now(timezone.utc) < CUTOFF:
            time.sleep(30)
        # Allow the current batch, validation and checkpoint writes to finish.
        while pids := own_trainers():
            if datetime.now(timezone.utc).timestamp() > CUTOFF.timestamp()+300:
                raise RuntimeError(f'Own trainers still active after cutoff grace period: {pids}')
            time.sleep(10)
        torch.set_num_threads(2)
        from scripts.evaluate_decoders import cache_emissions, compare
        runs = root/'runs'
        candidates = [p for p in runs.iterdir() if p.is_dir() and
                      (p.name in ('main', 'no_aug') or p.name.startswith(
                          ('refine100', 'finetune100', 'finetune460', 'lr_candidate_')))
                      and (p/'best_model.pt').exists()]
        scores, sources = {}, {}
        for candidate in sorted(candidates):
            destination = output/'candidates'/candidate.name
            destination.mkdir(parents=True, exist_ok=True)
            model = destination/'model.pt'
            frozen = freeze_model(candidate/'best_model.pt', model)
            sources[candidate.name] = {'source': str(candidate),
                                        'sha256': hashlib.sha256(frozen).hexdigest()}
            scores[candidate.name] = {}
            for part in DEV:
                cache = cache_emissions(frozen, root/'data/datasets', part, 0,
                                        destination, torch.device('cuda'), 16, 4)
                summary = compare(cache, destination/part, [], 8, 1)
                scores[candidate.name][part] = summary['decoders']['greedy']
            emit(output, 'candidate_evaluated', candidate=candidate.name, scores=scores[candidate.name])
        selected = choose_checkpoint(scores)
        model = output/'candidates'/selected/'model.pt'
        frozen = model.read_bytes()
        atomic_write(output/'selected_model.pt', frozen)
        copy_matching_resume(model, Path(sources[selected]['source'])/'best.pt', output/'selected_resume.pt')
        selection = {'selected': selected, 'selection': 'pooled full dev-clean/dev-other greedy corpus WER',
                     'candidates': sources, 'scores': scores}
        atomic_write(output/'selection.json', json.dumps(selection, indent=2).encode())
        emit(output, 'model_frozen', selected=selected, sha256=hashlib.sha256(frozen).hexdigest())
        lm = root/'incoming/3-gram.pruned.1e-7.arpa.gz'
        decoded = {}
        for part in DEV:
            cache = cache_emissions(frozen, root/'data/datasets', part, 0,
                                    output, torch.device('cuda'), 16, 4)
            decoded[part] = compare(cache, output/'dev-decoders'/part,
                                    [10, 20, 50], 8, args.workers,
                                    str(lm), [.1, .2, .3], 1.0)
        decoder = choose_decoder(decoded)
        decoder['checkpoint_sha256'] = hashlib.sha256(frozen).hexdigest()
        atomic_write(output/'frozen_decoder.json', json.dumps(decoder, indent=2).encode())
        emit(output, 'decoder_frozen', **decoder)
        for part in ('test-clean', 'test-other'):
            cache = cache_emissions(frozen, root/'data/datasets', part, 0,
                                    output, torch.device('cuda'), 16, 4)
            summary = compare(cache, output/'test'/part, [decoder['width']], 8, args.workers,
                              str(lm) if decoder['lm_weight'] is not None else None,
                              [decoder['lm_weight']] if decoder['lm_weight'] is not None else [], 1.0)
            emit(output, 'test_evaluated', split=part, decoders=summary['decoders'])
        complete = {'model_sha256': hashlib.sha256(frozen).hexdigest(), 'selected': selected,
                    'decoder': decoder, 'completed_utc': datetime.now(timezone.utc).isoformat()}
        atomic_write(output/'complete.json', json.dumps(complete, indent=2).encode())
        emit(output, 'final_evaluation_complete', **complete)


if __name__ == '__main__':
    main()
