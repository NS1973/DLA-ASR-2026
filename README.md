# DLA 2026 — HW1 ASR

Own DeepSpeech2-style CTC recognizer based on the official course project template.
Template branch: `example/asr`; template history is preserved in Git. The new pipeline is in `src/asr/`; legacy template examples are retained but are not runtime dependencies.

**Status: training and final evaluation are in progress. No held-out quality claim is made yet.**

## Installation

Python 3.12 is the reference environment. Linux with an NVIDIA GPU is recommended for training. CPU inference is supported.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest tests -q
```

Windows: activate with `.venv\Scripts\Activate.ps1`.

## Resources and inference

The final release `resources.json` contains exact artifact URLs and SHA-256 digests. Once the release is finalized:

```bash
python scripts/download_resources.py
python inference.py checkpoint=artifacts/model.pt custom_dir=/path/to/dataset output_dir=predictions decoder=greedy
python calc_metrics.py --reference-dir /path/to/dataset/transcriptions --prediction-dir predictions
```

Custom dataset layout:

```text
dataset/
  audio/utterance-id.wav       # WAV, FLAC or MP3, any sample rate, mono/stereo
  transcriptions/utterance-id.txt  # optional
```

Predictions keep the utterance IDs. Ground truth is optional for inference. Metric calculation requires a complete matching ID set and fails explicitly for missing predictions.

Own CTC prefix beam search:

```bash
python inference.py checkpoint=artifacts/model.pt custom_dir=/path/to/dataset output_dir=predictions_beam decoder=beam beam_width=20 beam_workers=4
```

## Training

Download official LibriSpeech archives:

```bash
python ops/download_data.py --parts train-clean-100 dev-clean dev-other test-clean test-other
mkdir -p data/datasets
for archive in data/downloads/*.tar.gz; do tar -xzf "$archive" -C data/datasets; done
python train.py run_dir=runs/main
```

The reference configuration uses 80 log-Mel features, 16 kHz audio, 25 ms windows and 10 ms hops; two convolutions with 4x temporal reduction, five bidirectional GRU layers (512 units per direction), 29 CTC symbols; AdamW, BF16 with FP32 CTC loss, warmup/cosine scheduling and gradient clipping.

The configuration has a campaign cutoff. For reproduction after the submission date, explicitly disable it:

```bash
python train.py run_dir=runs/reproduction end_utc=null
```

The exact original configuration and training manifests are saved inside each run. Seeds, selected checkpoint, epochs and final results will be documented in `REPORT.md`.

Resume a trusted training checkpoint:

```bash
python train.py resume=runs/main/last.pt run_dir=runs/main end_utc=null
```

One-batch test on real training audio:

```bash
python ops/download_data.py --bootstrap
python train.py custom_train=data/downloads/bootstrap overfit=true augment=false valid_parts=[] batch_size=4 workers=0 epochs=2000 max_steps=1500 warmup_steps=10 run_dir=runs/overfit end_utc=null
```

Four waveform augmentations are gain, additive noise, speed change and weak reverberation. Each experiment saves original/transformed audio and spectrograms under `media/`. Training logs include loss, LR, gradient norm, predictions, WER and CER from the beginning. W&B is offline; `metrics.jsonl` remains authoritative if W&B is unavailable.

## Colab demo

Open `DLA_ASR_Demo.ipynb` in Google Colab, select a GPU, enter only your dataset archive link, and run all cells. It installs the project, downloads and verifies model resources, runs a built-in example, processes your dataset and calculates metrics when ground truth exists. The repository must be publicly accessible for anonymous cloning after the deadline.

## Reproducibility and artifacts

The delivery includes source history, model-only weights for safe `weights_only=True` loading, trusted full resume checkpoints, configuration files, logs, predictions, measured metrics, augmentation evidence, report and SHA-256 manifests. Training checkpoints contain Python/NumPy RNG state and must only be loaded from trusted sources.

On GMC, training uses a separate container and an isolated virtual environment based on the already-installed CUDA image. Host system packages and other services are not modified. `ops/gmc.py` is a local operator utility that uses an existing protected credential loader; it contains no credentials and is not needed to run the submitted project.

## Evaluation integrity

Model/checkpoint selection and decoder tuning use dev data. Test data are reserved for final evaluation. No pretrained acoustic model or downloaded ASR implementation is used. Course instructions prohibit LLM use; technical reproducibility does not establish compliance with that rule.
