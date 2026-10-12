from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from src.asr.codec import Codec
from src.asr.data import CustomDirDataset, SpeechDataset, collate, librispeech_entries
from src.asr.metrics import corpus_metrics
from src.asr.runtime import load_model, worker_seed


def decode_beam(argument):
    probabilities, width, topk = argument
    return Codec().beam_search(probabilities, width, topk)


@torch.inference_mode()
def infer(config):
    config = OmegaConf.to_container(config, resolve=True) if not isinstance(config, dict) else config
    device = torch.device(config["device"] if config["device"] != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(config["cpu_threads"])
    model = load_model(config["checkpoint"], device)
    if config["custom_dir"]:
        dataset = CustomDirDataset(config["custom_dir"])
    else:
        dataset = SpeechDataset(librispeech_entries(config["data_root"], config["part"]))
    if config["limit"]:
        dataset.entries = dataset.entries[:config["limit"]]
    loader = DataLoader(dataset, batch_size=config["batch_size"], collate_fn=collate, num_workers=config["workers"],
                        worker_init_fn=worker_seed)
    destination = Path(config["output_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    codec = Codec()
    references, predictions, records = [], [], []
    start = time.monotonic()
    pool = ProcessPoolExecutor(max_workers=config["beam_workers"], mp_context=multiprocessing.get_context("spawn")) if config["decoder"] == "beam" and config["beam_workers"] > 1 else None
    try:
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config["bf16"] and device.type == "cuda"):
                result = model(batch["spectrogram"].to(device), batch["spectrogram_length"])
            log_probs = result["log_probs"].float().cpu()
            arrays = [p[:int(n)].numpy() for p, n in zip(log_probs, result["log_probs_length"])]
            if config["decoder"] == "greedy":
                texts = [codec.greedy(p.argmax(-1)) for p in arrays]
            elif config["decoder"] == "beam":
                arguments = [(p, config["beam_width"], config["token_topk"]) for p in arrays]
                texts = list(pool.map(decode_beam, arguments)) if pool else [decode_beam(a) for a in arguments]
            else:
                raise ValueError(f"Unknown decoder {config['decoder']}")
            for uid, text, target in zip(batch["uids"], texts, batch["texts"]):
                (destination / (uid + ".txt")).write_text(text + "\n", encoding="utf-8")
                record = {"uid": uid, "prediction": text, "target": target}
                if target is not None:
                    references.append(target)
                    predictions.append(text)
                    record.update(corpus_metrics([target], [text]))
                records.append(record)
            print(json.dumps({"decoded": len(records), "total": len(dataset), "elapsed_seconds": time.monotonic()-start}), flush=True)
    finally:
        if pool:
            pool.shutdown()
    metrics = corpus_metrics(references, predictions) if references else None
    summary = {"config": config, "metrics": metrics, "decoded": len(records), "elapsed_seconds": time.monotonic()-start}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (destination / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return summary
