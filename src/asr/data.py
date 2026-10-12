import json
from pathlib import Path
import random
import numpy as np
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler
import torchaudio
from src.asr.codec import Codec


def load_audio(path):
    try:
        data, rate = sf.read(path, dtype="float32", always_2d=True)
        wave = torch.from_numpy(data.mean(axis=1).copy())
    except Exception:
        wave, rate = torchaudio.load(str(path))
        wave = wave.mean(0)
    if not wave.numel():
        raise ValueError(f"Empty audio file: {path}")
    if rate != 16000:
        wave = torchaudio.functional.resample(wave, rate, 16000)
    return wave.clamp(-1, 1)


def augment_waveform(wave, kind, rng=None, generator=None):
    rng = rng or random
    if kind == "gain":
        result = wave * rng.uniform(.75, 1.25)
    elif kind == "noise":
        snr = rng.uniform(20, 35)
        noise = torch.randn(wave.shape, generator=generator)
        result = wave + noise * wave.square().mean().sqrt() / (10 ** (snr / 20))
    elif kind == "speed":
        speed = rng.uniform(.90, 1.10)
        # Interpolation avoids thousands of resampling kernels for random ratios.
        result = torch.nn.functional.interpolate(wave[None, None], size=max(1, int(len(wave) / speed)), mode="linear", align_corners=False)[0, 0]
    elif kind == "reverb":
        duration = rng.uniform(.1, .3)
        n = int(duration * 16000)
        impulse = torch.randn(n, generator=generator) * torch.exp(-torch.arange(n) / (n / 6))
        impulse[0] = 0
        impulse /= impulse.abs().sum().clamp_min(1e-8)
        size = 1 << (len(wave) + n - 2).bit_length()
        reflected = torch.fft.irfft(torch.fft.rfft(wave, size) * torch.fft.rfft(impulse, size), size)[:len(wave)]
        result = wave + rng.uniform(.1, .2) * reflected
    else:
        raise ValueError(kind)
    return result.clamp(-1, 1)


class SpeechDataset(Dataset):
    def __init__(self, entries, augment=False, seed=1):
        self.entries = entries
        self.augment = augment
        self.seed = seed
        self.epoch = torch.zeros(1, dtype=torch.int64).share_memory_()
        self.codec = Codec()
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, win_length=400,
                                                       hop_length=160, n_mels=80, f_min=20, f_max=7600)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        wave = load_audio(entry["path"])
        if self.augment:
            seed = self.seed + int(self.epoch[0]) * 1000003 + index
            rng = random.Random(seed)
            generator = torch.Generator().manual_seed(seed)
            for kind in ("gain", "noise", "speed", "reverb"):
                if rng.random() < .35:
                    wave = augment_waveform(wave, kind, rng, generator)
        # For very short files reflect-padding can fail, so pad to an FFT window.
        if len(wave) < 400:
            wave = torch.nn.functional.pad(wave, (0, 400 - len(wave)))
        spec = self.mel(wave).clamp_min(1e-10).log()
        spec = (spec - spec.mean()) / spec.std().clamp_min(1e-5)
        text = entry.get("text")
        return {"spectrogram": spec, "waveform": wave, "text": text, "uid": entry["uid"],
                "encoded": torch.tensor(self.codec.encode(text), dtype=torch.long) if text is not None else torch.empty(0, dtype=torch.long)}


class CustomDirDataset(SpeechDataset):
    def __init__(self, directory, augment=False):
        root = Path(directory)
        if not (root / "audio").is_dir():
            raise ValueError(f"Missing audio directory: {root / 'audio'}")
        entries = []
        seen = set()
        for path in sorted((root / "audio").iterdir()):
            if path.suffix.lower() not in {".wav", ".flac", ".mp3"}:
                continue
            if path.stem in seen:
                raise ValueError(f"Duplicate utterance ID: {path.stem}")
            seen.add(path.stem)
            target = root / "transcriptions" / (path.stem + ".txt")
            entries.append({"uid": path.stem, "path": str(path), "text": target.read_text(encoding="utf-8").strip() if target.exists() else None,
                            "duration": sf.info(path).duration})
        if not entries:
            raise ValueError(f"No supported audio in {root}")
        super().__init__(entries, augment)


def librispeech_entries(root, part):
    directory = Path(root) / "LibriSpeech" / part
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    entries = []
    for transcript in sorted(directory.rglob("*.trans.txt")):
        for line in transcript.read_text(encoding="utf-8").splitlines():
            uid, text = line.split(" ", 1)
            path = transcript.parent / (uid + ".flac")
            info = sf.info(path)
            entries.append({"uid": uid, "text": Codec.normalize(text), "path": str(path), "duration": info.duration})
    return entries


def collate(items):
    return {"spectrogram": pad_sequence([x["spectrogram"].T for x in items], batch_first=True).transpose(1, 2),
            "spectrogram_length": torch.tensor([x["spectrogram"].shape[1] for x in items], dtype=torch.long),
            "targets": torch.cat([x["encoded"] for x in items]),
            "target_lengths": torch.tensor([len(x["encoded"]) for x in items], dtype=torch.long),
            "uids": [x["uid"] for x in items], "texts": [x["text"] for x in items]}


class LengthBatchSampler(Sampler):
    def __init__(self, entries, batch_size=32, seed=1, shuffle=True):
        self.entries, self.batch_size, self.seed, self.shuffle = entries, batch_size, seed, shuffle
        self.epoch = 0

    def __len__(self):
        return (len(self.entries) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.entries)))
        if self.shuffle:
            rng.shuffle(indices)
        batches = []
        chunk_size = self.batch_size * 50
        for start in range(0, len(indices), chunk_size):
            bucket = sorted(indices[start:start + chunk_size], key=lambda i: self.entries[i].get("duration", 0))
            batches.extend(bucket[i:i + self.batch_size] for i in range(0, len(bucket), self.batch_size))
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches
