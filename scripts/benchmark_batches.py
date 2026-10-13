"""Measure real-audio BF16 train-step throughput in an isolated GPU process.

The temporary model is never exported or used as a quality candidate.
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.asr.data import SpeechDataset, collate, librispeech_entries
from src.asr.runtime import load_model, seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data-root', default='/work/data/datasets')
    parser.add_argument('--batches', nargs='+', type=int, default=[32,64,128,256])
    args = parser.parse_args()
    seed_all(42)
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    # Random train records only; sort the fixed cohort as in duration buckets.
    entries = librispeech_entries(args.data_root, 'train-clean-100')
    entries = random.Random(42).sample(entries, max(args.batches))
    entries.sort(key=lambda x: x['duration'])
    dataset = SpeechDataset(entries)
    items = [dataset[index] for index in range(len(dataset))]
    model = load_model(args.checkpoint, torch.device('cuda')).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=False)
    results = []
    for size in args.batches:
        # Evenly spread the same duration range across all tested batch sizes.
        subset = [items[index*len(items)//size] for index in range(size)]
        batch = collate(subset)
        spectrogram = batch['spectrogram'].cuda()
        targets = batch['targets'].cuda()
        elapsed = []
        torch.cuda.reset_peak_memory_stats()
        for step in range(10):
            torch.cuda.synchronize()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(spectrogram, batch['spectrogram_length'])
            loss = criterion(output['log_probs'].float().transpose(0,1), targets,
                             output['log_probs_length'], batch['target_lengths'])
            if not torch.isfinite(loss):
                raise ValueError('Non-finite benchmark CTC loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            if not torch.isfinite(norm):
                raise ValueError('Non-finite benchmark gradient')
            optimizer.step()
            torch.cuda.synchronize()
            if step >= 2:
                elapsed.append(time.perf_counter()-start)
        mean = sum(elapsed)/len(elapsed)
        row = {'batch_size': size, 'mean_seconds_per_step': mean,
               'utterances_per_second': size/mean,
               'max_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
               'max_duration_seconds': max(x['duration'] for x in entries),
               'final_loss': loss.item(), 'gradient_norm': norm.item(),
               'timed_steps': len(elapsed), 'warmup_steps': 2,
               'includes_audio_loading': False}
        results.append(row)
        Path(args.output).write_text(json.dumps({'results': results,
            'note': 'Train-step-only measurement on a fixed real train cohort; not end-to-end epoch speed.'},indent=2))
        print(json.dumps(row), flush=True)
        del spectrogram, targets, output, loss, batch


if __name__ == '__main__':
    main()
