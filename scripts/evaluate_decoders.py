"""Cache one checkpoint's emissions and compare own decoders on identical audio.

Use dev splits to select width. Only run the selected width on test splits.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.asr.codec import Codec
from src.asr.data import SpeechDataset, collate, librispeech_entries
from src.asr.metrics import corpus_metrics
from src.asr.runtime import load_model, worker_seed, supports_native_bf16
from src.asr.infer import initialize_decoder, decode_beam


def decode_job(job):
    emissions, width, topk = job
    return Codec().beam_search(emissions, width, topk)


@torch.inference_mode()
def cache_emissions(checkpoint_bytes, root, part, limit, destination, device, batch_size, workers):
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    cache = destination / f'{part}-{digest[:16]}-{limit}.npz'
    if not cache.exists():
        entries = librispeech_entries(root, part)
        if limit:
            import random
            entries = random.Random(42).sample(entries, min(limit, len(entries)))
        dataset = SpeechDataset(entries)
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate,
                            num_workers=workers, worker_init_fn=worker_seed)
        model = load_model(io.BytesIO(checkpoint_bytes), device)
        arrays, uids, references, lengths = [], [], [], []
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=supports_native_bf16(device)):
                result = model(batch['spectrogram'].to(device), batch['spectrogram_length'])
            for probabilities, length in zip(result['log_probs'].float().cpu(), result['log_probs_length']):
                arrays.append(probabilities[:int(length)].numpy())
                lengths.append(int(length))
            uids.extend(batch['uids'])
            references.extend(batch['texts'])
            print(json.dumps({'event':'cache_progress','split':part,'decoded':len(uids),'total':len(entries)}), flush=True)
        np.savez_compressed(cache, emissions=np.concatenate(arrays), lengths=lengths,
                            uids=uids, references=references, checkpoint_sha256=digest)
    return cache


def compare(cache, destination, widths, topk, workers, lm_path=None, lm_weights=(.5,), word_bonus=1.0):
    data = np.load(cache, allow_pickle=False)
    arrays = np.split(data['emissions'], np.cumsum(data['lengths'])[:-1])
    references, uids = data['references'].tolist(), data['uids'].tolist()
    summary = {'checkpoint_sha256':str(data['checkpoint_sha256']), 'utterances':len(uids), 'decoders':{}}
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                               initializer=initialize_decoder,initargs=(lm_path,)) if workers > 1 else None
    if pool is None:
        initialize_decoder(lm_path)
    settings = [(0,None), *[(width,None) for width in widths]]
    if lm_path:
        settings.extend((width,weight) for width in widths for weight in lm_weights)
        with Path(lm_path).open('rb') as stream:
            summary['lm_sha256'] = hashlib.file_digest(stream,'sha256').hexdigest()
    try:
        for width, lm_weight in settings:
            start = time.monotonic()
            name = 'greedy' if width == 0 else f'beam{width}'
            if lm_weight is not None:
                name += f'_lm{lm_weight}_bonus{word_bonus}'
            if width == 0:
                predictions = [Codec().greedy(p.argmax(-1)) for p in arrays]
            else:
                if lm_weight is None:
                    jobs = ((p, width, topk) for p in arrays)
                    function = decode_job
                else:
                    jobs = ((p,width,topk,lm_weight,word_bonus) for p in arrays)
                    function = decode_beam
                predictions = list(pool.map(function, jobs, chunksize=4)) if pool else list(map(function, jobs))
            folder = destination / name
            folder.mkdir(parents=True, exist_ok=True)
            for uid, prediction in zip(uids, predictions):
                (folder / f'{uid}.txt').write_text(prediction + '\n', encoding='utf-8')
            summary['decoders'][name] = {**corpus_metrics(references, predictions), 'seconds':time.monotonic()-start}
            (destination / 'comparison.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            print(json.dumps({'decoder':name, **summary['decoders'][name]}), flush=True)
    finally:
        if pool:
            pool.shutdown()
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-root', default='/work/data/datasets')
    p.add_argument('--parts', nargs='+', default=['dev-clean','dev-other'])
    p.add_argument('--output-dir', required=True)
    p.add_argument('--limit', type=int, default=256)
    p.add_argument('--widths', nargs='+', type=int, default=[5,10,20])
    p.add_argument('--token-topk', type=int, default=8)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--device', default='cuda')
    p.add_argument('--lm-path')
    p.add_argument('--lm-weights',nargs='+',type=float,default=[.3,.5,.7])
    p.add_argument('--word-bonus',type=float,default=1.0)
    args = p.parse_args()
    torch.set_num_threads(2)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(part.startswith('test') for part in args.parts) and (len(args.widths) != 1 or (args.lm_path and len(args.lm_weights)!=1)):
        p.error('Select a single width on dev before evaluating test splits')
    # Freeze once for the entire sweep, including every split. Training can
    # replace best_model.pt while CPU beam search is still processing split one.
    checkpoint_bytes = Path(args.checkpoint).read_bytes()
    for part in args.parts:
        cache = cache_emissions(checkpoint_bytes, args.data_root, part, args.limit, output,
                                torch.device(args.device), args.batch_size, min(4,args.workers))
        compare(cache, output / part, args.widths, args.token_topk, args.workers,args.lm_path,args.lm_weights,args.word_bonus)


if __name__ == '__main__':
    main()
