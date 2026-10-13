"""Bounded real-data CUDA training benchmark; optimizer updates stay in RAM."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.asr.runtime import load_model, make_loader, supports_native_bf16, seed_all
from ops.build_mixed_manifest import load_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--steps', type=int, default=100)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    cfg.update(batch_size=args.batch_size, workers=args.workers)
    seed_all(cfg['seed'])
    torch.set_num_threads(cfg.get('cpu_threads', 4))
    torch.backends.cudnn.benchmark = True
    entries = load_manifest(cfg['training_manifest_path'], cfg['data_root'], cfg['training_manifest_sha256'])
    # Same fixed, stratified duration probe for every batch/worker setting.
    ordered = sorted(entries, key=lambda row: row['duration'])
    count = min(len(ordered), 12000)
    entries = [ordered[i * len(ordered) // count] for i in range(count)]
    loader = make_loader(entries, cfg, augment=True, shuffle=True)
    device = torch.device('cuda')
    model = load_model(cfg['init_checkpoint'], device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
    loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=False)
    records = []
    torch.cuda.reset_peak_memory_stats()
    iterator = iter(loader)
    for i in range(args.warmup + args.steps):
        start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        loaded = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        spec = batch['spectrogram'].to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=supports_native_bf16(device)):
            out = model(spec, batch['spectrogram_length'])
        loss = loss_fn(out['log_probs'].transpose(0, 1), batch['targets'].to(device),
                       out['log_probs_length'], batch['target_lengths'])
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise RuntimeError('Nonfinite benchmark loss/gradient')
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if i >= args.warmup:
            records.append(dict(seconds=elapsed, data_wait=loaded-start,
                audio_seconds=float(batch['spectrogram_length'].sum())*.01,
                padding_fraction=1-float(batch['spectrogram_length'].sum())/spec.shape[0]/spec.shape[-1],
                loss=float(loss), gradient_norm=float(norm)))
    durations = sorted(r['seconds'] for r in records)
    result = dict(config=cfg, warmup=args.warmup, measured_steps=len(records),
        audio_seconds_per_second=sum(r['audio_seconds'] for r in records)/sum(durations),
        step_median=statistics.median(durations), step_p95=durations[min(len(durations)-1, int(.95*len(durations)))],
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), records=records)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in {'records','config'}}), flush=True)


if __name__ == '__main__':
    main()
