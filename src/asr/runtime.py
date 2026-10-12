"""Training, trusted resume checkpoints and evaluation."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from src.asr.codec import Codec
from src.asr.data import SpeechDataset, CustomDirDataset, LengthBatchSampler, collate, librispeech_entries
from src.asr.metrics import corpus_metrics
from src.asr.model import DeepSpeech2CTC


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, model, optimizer, scheduler, state, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "model_spec": getattr(model, "spec", config.get("model", {})),
               "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "state": state.copy(),
               "config": config, "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
               "gradients": {name: p.grad.detach().cpu() for name, p in model.named_parameters() if p.grad is not None},
               "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    temp = Path(str(path) + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def restore_checkpoint(path, model, optimizer=None, scheduler=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    for name, parameter in model.named_parameters():
        gradient = payload.get("gradients", {}).get(name)
        parameter.grad = gradient.to(parameter.device) if gradient is not None else None
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    random.setstate(payload["rng_python"])
    np.random.set_state(payload["rng_numpy"])
    torch.set_rng_state(payload["rng_torch"])
    if torch.cuda.is_available() and payload.get("rng_cuda"):
        torch.cuda.set_rng_state_all(payload["rng_cuda"])
    return payload["state"]


def export_model(model, path, state):
    temp = Path(str(path) + ".tmp")
    torch.save({"model": model.state_dict(), "model_spec": model.spec, "state": state.copy()}, temp)
    os.replace(temp, path)


def load_model(checkpoint, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = DeepSpeech2CTC(**payload["model_spec"]).to(device)
    model.load_state_dict(payload["model"])
    return model.eval()


class RunLog:
    def __init__(self, directory, config):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "metrics.jsonl"
        self.wandb = None
        try:
            import wandb
            os.environ["WANDB_MODE"] = "offline"
            self.wandb = wandb.init(project="dla-asr-2026", name=self.directory.name, config=config,
                                    dir=str(self.directory), mode="offline")
        except Exception as exc:
            self.emit({"event": "wandb_unavailable", "error": str(exc)})

    def emit(self, record):
        record = {"utc": datetime.now(timezone.utc).isoformat(), **record}
        line = json.dumps(record, ensure_ascii=True, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        temp = self.directory / "status.json.tmp"
        temp.write_text(line, encoding="utf-8")
        temp.replace(self.directory / "status.json")
        print(line, flush=True)
        if self.wandb is not None:
            try:
                self.wandb.log({key: value for key, value in record.items() if isinstance(value, (int, float))})
            except Exception as exc:
                print(f"W&B logging error; JSONL preserved: {type(exc).__name__}", flush=True)


def worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def make_loader(entries, config, augment=False, shuffle=False):
    sampler = LengthBatchSampler(entries, config["batch_size"], config["seed"], shuffle)
    return DataLoader(SpeechDataset(entries, augment, config["seed"]), batch_sampler=sampler, collate_fn=collate,
                      generator=torch.Generator().manual_seed(config["seed"]),
                      num_workers=config["workers"], pin_memory=True, worker_init_fn=worker_seed,
                      persistent_workers=config["workers"] > 0)


@torch.inference_mode()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    codec = Codec()
    references, predictions, examples = [], [], []
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            result = model(batch["spectrogram"].to(device), batch["spectrogram_length"])
        paths = result["log_probs"].argmax(-1).cpu()
        for uid, reference, path, length in zip(batch["uids"], batch["texts"], paths, result["log_probs_length"]):
            prediction = codec.greedy(path[:int(length)])
            if reference is not None:
                references.append(reference)
                predictions.append(prediction)
                if len(examples) < 10:
                    examples.append({"uid": uid, "target": reference, "prediction": prediction,
                                     **corpus_metrics([reference], [prediction])})
    return corpus_metrics(references, predictions), examples


def log_media(entries, directory, logger, namespace="train"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import soundfile as sf
    from src.asr.data import load_audio, augment_waveform
    media = Path(directory) / "media" / namespace
    media.mkdir(parents=True, exist_ok=True)
    mel_transform = SpeechDataset([]).mel
    for i, entry in enumerate(entries[:3]):
        original = load_audio(entry["path"])
        for kind in ["original", "gain", "noise", "speed", "reverb"]:
            wave = original if kind == "original" else augment_waveform(original, kind)
            wav_path, image_path = media / f"{i}_{kind}.wav", media / f"{i}_{kind}.png"
            sf.write(wav_path, wave.numpy(), 16000)
            mel = mel_transform(wave).clamp_min(1e-10).log().numpy()
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.imshow(mel, origin="lower", aspect="auto", extent=[0, len(wave)/16000, 0, 80])
            ax.set(xlabel="Time (s)", ylabel="Mel band", title=f"{kind}: {entry['uid']}")
            fig.tight_layout()
            fig.savefig(image_path)
            plt.close(fig)
            if logger.wandb is not None:
                try:
                    import wandb
                    logger.wandb.log({f"audio/{namespace}/{i}_{kind}": wandb.Audio(str(wav_path), sample_rate=16000),
                                      f"spectrogram/{namespace}/{i}_{kind}": wandb.Image(str(image_path))})
                except Exception:
                    pass
    logger.emit({"event": "augmentation_media_saved", "path": str(media), "waveform_augmentations": 4})


def train(config):
    config = OmegaConf.to_container(config, resolve=True) if not isinstance(config, dict) else config
    seed_all(config["seed"])
    torch.set_num_threads(config.get("cpu_threads", 4))
    torch.backends.cudnn.benchmark = True
    device = torch.device(config["device"] if config["device"] != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    directory = Path(config["run_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    logger = RunLog(directory, config)
    if config.get("custom_train"):
        entries = CustomDirDataset(config["custom_train"]).entries
    else:
        entries = sum([librispeech_entries(config["data_root"], p) for p in config["train_parts"]], [])
    if config["overfit"]:
        entries = sorted(entries, key=lambda e: e.get("duration", 0))[:config["batch_size"]]
    if config.get("limit_train"):
        entries = entries[:config["limit_train"]]
    (directory / "train_manifest.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    loader = make_loader(entries, config, config["augment"] and not config["overfit"], not config["overfit"])
    validations = {}
    for part in config["valid_parts"]:
        try:
            valid_entries = librispeech_entries(config["data_root"], part)
            if config["valid_limit"]:
                valid_entries = random.Random(config["seed"]).sample(valid_entries, min(config["valid_limit"], len(valid_entries)))
            validations[part] = make_loader(valid_entries, config)
        except FileNotFoundError:
            logger.emit({"event": "validation_pending", "split": part})
    model = DeepSpeech2CTC(**config["model"]).to(device)
    if config.get("init_checkpoint"):
        initial = torch.load(config["init_checkpoint"], map_location="cpu", weights_only=True)
        model.load_state_dict(initial["model"])
        logger.emit({"event": "initialize_from_model", "path": config["init_checkpoint"], "source_state": initial.get("state")})
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    total_steps = config["epochs"] * max(1, math.ceil(len(loader) / config["accumulation"]))
    def lr_factor(step):
        warmup = config["warmup_steps"]
        if step < warmup:
            return max(.01, (step + 1) / max(1, warmup))
        progress = min(1, (step - warmup) / max(1, total_steps - warmup))
        return .1 + .9 * .5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    state = {"step": 0, "epoch": 0, "batch_index": 0, "micro_count": 0, "best_wer": 1e9}
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=False)
    logger.emit({"event": "train_start", "parameters": sum(p.numel() for p in model.parameters()),
                 "utterances": len(entries), "batches": len(loader), "device": str(device), "step": state["step"]})
    log_media(entries, directory, logger)
    for part, valid_loader in validations.items():
        log_media(valid_loader.dataset.entries[:1], directory, logger, namespace=part)
    optimizer.zero_grad(set_to_none=True)
    if config.get("resume"):
        state.update(restore_checkpoint(config["resume"], model, optimizer, scheduler))
        previous = torch.load(config["resume"], map_location="cpu", weights_only=False)["config"]
        if previous.get("batch_size") != config["batch_size"] or previous.get("accumulation") != config["accumulation"]:
            state["batch_index"], state["micro_count"] = 0, 0
            optimizer.zero_grad(set_to_none=True)
            logger.emit({"event": "resume_changed_batch_restart_epoch", "epoch": state["epoch"]})
        logger.emit({"event": "resume_restored", "step": state["step"], "batch_index": state["batch_index"]})
    if config.get("recovery_lr_scale", 1.0) != 1.0:
        scale = config["recovery_lr_scale"]
        scheduler.base_lrs = [lr * scale for lr in scheduler.base_lrs]
        for group in optimizer.param_groups:
            group["lr"] *= scale
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    start = last_save = time.monotonic()
    mixed = config["bf16"] and device.type == "cuda"
    deadline = datetime.fromisoformat(config["end_utc"]).timestamp() if config.get("end_utc") else math.inf
    last = directory / "last.pt"
    save_checkpoint(last, model, optimizer, scheduler, state, config)
    completed_normally = False
    try:
        for epoch in range(state["epoch"], config["epochs"]):
            if time.time() >= deadline or (config["max_steps"] and state["step"] >= config["max_steps"]):
                break
            loader.batch_sampler.epoch = epoch
            loader.dataset.epoch[0] = epoch
            model.train()
            epoch_loss = count = 0
            resume_batch = state.get("batch_index", 0) if epoch == state["epoch"] else 0
            for batch_index, batch in enumerate(loader):
                if batch_index < resume_batch:
                    continue
                state["epoch"] = epoch
                output_lengths = model.transform_input_lengths(batch["spectrogram_length"])
                repeats, offset = [], 0
                for n in batch["target_lengths"].tolist():
                    seq = batch["targets"][offset:offset+n]
                    repeats.append(int((seq[1:] == seq[:-1]).sum()))
                    offset += n
                if (output_lengths < batch["target_lengths"] + torch.tensor(repeats)).any():
                    raise ValueError("CTC target requires more timesteps than output")
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=mixed):
                    result = model(batch["spectrogram"].to(device, non_blocking=True), batch["spectrogram_length"])
                loss = criterion(result["log_probs"].transpose(0, 1).float(), batch["targets"].to(device),
                                 result["log_probs_length"], batch["target_lengths"])
                if not torch.isfinite(loss):
                    logger.emit({"event": "nonfinite_loss", "step": state["step"]})
                    raise FloatingPointError("Nonfinite CTC loss; restore the last safe checkpoint")
                (loss / config["accumulation"]).backward()
                epoch_loss += loss.item()
                count += 1
                state["batch_index"] = batch_index + 1
                state["micro_count"] += 1
                if state["micro_count"] == config["accumulation"] or batch_index + 1 == len(loader):
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
                    if not torch.isfinite(norm):
                        logger.emit({"event": "nonfinite_gradient", "step": state["step"]})
                        raise FloatingPointError("Nonfinite gradient; restore the last safe checkpoint")
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    state["micro_count"] = 0
                    state["step"] += 1
                    if state["step"] == 1 or state["step"] % config["log_every"] == 0:
                        paths = result["log_probs"].detach().argmax(-1).cpu()
                        codec = Codec()
                        predictions = [codec.greedy(p[:int(n)]) for p, n in zip(paths, output_lengths)]
                        metrics = corpus_metrics(batch["texts"], predictions)
                        logger.emit({"event": "train", "epoch": epoch, "step": state["step"], "loss": loss.item(),
                                     "lr": optimizer.param_groups[0]["lr"], "gradient_norm": float(norm),
                                     "train_wer": metrics["wer"], "train_cer": metrics["cer"], "elapsed_seconds": time.monotonic()-start,
                                     "samples": [{"target": r, "prediction": p, **corpus_metrics([r],[p])} for r,p in list(zip(batch["texts"], predictions))[:3]]})
                if time.monotonic()-last_save >= config["checkpoint_seconds"]:
                    save_checkpoint(last, model, optimizer, scheduler, state, config)
                    export_model(model, directory / "last_model.pt", state)
                    last_save = time.monotonic()
                if stopping or time.time() >= deadline or (config["max_steps"] and state["step"] >= config["max_steps"]):
                    break
            if config["overfit"] and (epoch % 20 == 0 or stopping):
                metrics, examples = evaluate(model, loader, device, max_batches=1)
                logger.emit({"event": "overfit_eval", "step": state["step"], **metrics, "examples": examples})
                if metrics["wer"] <= config["overfit_target"]:
                    stopping = True
            for part, valid_loader in validations.items():
                metrics, examples = evaluate(model, valid_loader, device)
                logger.emit({"event": "validation", "split": part, "epoch": epoch, "step": state["step"], **metrics, "examples": examples})
                if part == "dev-clean" and metrics["wer"] < state["best_wer"]:
                    state["best_wer"] = metrics["wer"]
                    save_checkpoint(directory / "best.pt", model, optimizer, scheduler, state, config)
                    export_model(model, directory / "best_model.pt", state)
            if state["batch_index"] >= len(loader):
                state["epoch"], state["batch_index"] = epoch + 1, 0
            if not config["overfit"] or epoch % 25 == 0:
                save_checkpoint(last, model, optimizer, scheduler, state, config)
                export_model(model, directory / "last_model.pt", state)
                logger.emit({"event": "epoch_end", "epoch": epoch, "step": state["step"], "mean_loss": epoch_loss/max(1,count)})
            if stopping or time.time() >= deadline or (config["max_steps"] and state["step"] >= config["max_steps"]):
                break
        completed_normally = True
    finally:
        if completed_normally:
            save_checkpoint(last, model, optimizer, scheduler, state, config)
            export_model(model, directory / "last_model.pt", state)
        logger.emit({"event": "train_exit" if completed_normally else "train_failed_preserved_checkpoint", "step": state["step"], "epoch": state["epoch"]})
        if logger.wandb is not None:
            logger.wandb.finish()
