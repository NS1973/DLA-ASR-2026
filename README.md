# DLA 2026 — HW1 ASR

Own DeepSpeech2-style CTC recognizer based on the official course project template.
Template branch: `example/asr`; template history is preserved in Git. The new pipeline is in `src/asr/`; legacy template examples are retained but are not runtime dependencies.

**Measured full-set result of the preserved dev-selected candidate: test-clean WER 14.16996% (2620 utterances), test-other WER 40.46195% (2939 utterances). Final demo and packaging acceptance are in progress. Two additional training stages are separate candidates; they do not replace this result without full dev selection and acceptance.**

## Installation

Python 3.12 is the reference environment. Linux with an NVIDIA GPU is recommended for training. CPU inference is supported.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest tests -q
```

Windows: activate with `.venv\Scripts\Activate.ps1`.

The fully pinned CPU verification environment is in `requirements-lock-cpu.txt` (install its PyTorch wheels from the official CPU index). H200 training uses native BF16; T4 and CPU inference automatically use FP32.

## Resources and inference

The final release `resources.json` contains exact artifact URLs and SHA-256 digests. The download script verifies every resource hash:

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
python inference.py checkpoint=artifacts/model.pt custom_dir=/path/to/dataset output_dir=predictions_beam decoder=beam beam_width=50 beam_workers=2 lm_path=artifacts/3-gram.pruned.1e-7.arpa.gz lm_weight=0.1 word_bonus=1.0
```

## Training

Download official LibriSpeech archives:

```bash
python ops/download_data.py --parts train-clean-100 dev-clean dev-other test-clean test-other
mkdir -p data/datasets
for archive in data/downloads/*.tar.gz; do tar -xzf "$archive" -C data/datasets; done
python train.py run_dir=runs/main end_utc=null
```

The reference configuration uses 80 log-Mel features, 16 kHz audio, 25 ms windows and 10 ms hops; two convolutions with 4x temporal reduction, five bidirectional GRU layers (512 units per direction), 29 CTC symbols; AdamW, BF16 with FP32 CTC loss, warmup/cosine scheduling and gradient clipping.

The configuration has a campaign cutoff. For reproduction after the submission date, explicitly disable it:

```bash
python train.py run_dir=runs/reproduction end_utc=null
```

The exact original configuration and training manifests are saved inside each run. See [REPRODUCTION.md](REPRODUCTION.md) for the own staging initialization and scheduler-preserving training commands. Seeds, selected checkpoint, epochs and final results will be documented in `REPORT.md`.

Resume a trusted training checkpoint:

```bash
python train.py resume=runs/main/last.pt run_dir=runs/main end_utc=null
```

One-batch test on real training audio:

```bash
python ops/download_data.py --bootstrap
python train.py custom_train=data/downloads/bootstrap overfit=true augment=false valid_parts=[] batch_size=4 workers=0 epochs=2000 max_steps=1500 warmup_steps=10 run_dir=runs/overfit end_utc=null
```

Four waveform augmentations are gain, additive noise, speed change and weak reverberation. The default `augmentation_profile=legacy` preserves prior experiments exactly. The separate `effective_v2` profile adds a smooth gain envelope and energy-controlled reverberation; it retains noise, speed and the CTC-length guard. `data_epoch_offset=80` changes training order/augmentation epochs while keeping the fixed dev subset unchanged. These settings are recorded in each config; use the saved phase config when reproducing a run. Each experiment saves original/transformed audio and spectrograms under `media/`. Training logs include loss, LR, gradient norm, predictions, WER and CER from the beginning. W&B is offline; `metrics.jsonl` remains authoritative if W&B is unavailable.

`ops/augmentation_proof.py` creates a separate offline demonstration run on the GMC project data. W&B text tables contain random target/prediction samples with CER/WER; dev-clean and dev-other have distinct metric namespaces.

Optional word LM fusion is implemented in the same own prefix beam. The source, license and checksum of the OpenSLR trigram are in `lm_sources.json`. It is experimental until the final dev comparison; no bonus is claimed from implementation alone.

```bash
python inference.py checkpoint=artifacts/model.pt custom_dir=/path/to/dataset decoder=beam lm_path=/path/to/3-gram.pruned.1e-7.arpa.gz lm_weight=0.5 word_bonus=1.0
```

## Colab demo

Open `DLA_ASR_Demo.ipynb` in Google Colab, select a GPU, enter only your dataset archive link, and run all cells. It installs the project, downloads and verifies model resources, runs a built-in example, processes your dataset and calculates metrics when ground truth exists. The repository must be publicly accessible for anonymous cloning after the deadline.

## Reproducibility and artifacts

The delivery includes source history, model-only weights for safe `weights_only=True` loading, trusted full resume checkpoints, configuration files, logs, predictions, measured metrics, augmentation evidence, report and SHA-256 manifests. Training checkpoints contain Python/NumPy RNG state and must only be loaded from trusted sources.

The current report is a draft. To rebuild its PDF from measured logs, install `requirements-report.txt` separately and run `python scripts/build_report_pdf.py --metrics /path/to/run/metrics.jsonl`. A Cyrillic Arial or DejaVu Sans font must be available. `scripts/package_delivery.py` packages a committed source tree and explicitly selected, locally collected artifacts with SHA-256 digests.

On GMC, training uses a separate container and an isolated virtual environment based on the already-installed CUDA image. Host system packages and other services are not modified. `ops/gmc.py` is a local operator utility that uses an existing protected credential loader; it contains no credentials and is not needed to run the submitted project.

## Evaluation integrity

Model/checkpoint selection and decoder tuning use dev data. Test data are reserved for final evaluation. No pretrained acoustic model or downloaded ASR implementation is used. Course instructions prohibit LLM use; technical reproducibility does not establish compliance with that rule.


Final selection uses best **and** last checkpoints of the campaign runs, validates each corresponding trusted full checkpoint, and deduplicates exact model-file SHA-256 values. The ranking is pooled full dev-clean/dev-other word errors divided by pooled reference words, with clean WER, other WER and candidate name as deterministic ties. Decoder widths 10/20/50 and LM weights 0.1/0.2/0.3 (word bonus 1) are compared on both full dev sets. Test sets are evaluated only after model and decoder are frozen.

Before changing a trainer, preserve the campaign with `python scripts/preserve_runs.py --runs /work/runs --output /work/incoming/NEW_UNIQUE_SNAPSHOT --parts 4`. This command creates content-addressed TAR parts, a logical file manifest and exact model/full checkpoint pair evidence, while retaining a 20-GiB free-space reserve. An existing destination is rejected. Download each archive, verify its SHA-256 and each content member, and recheck model/full tensor and state equality. Active log snapshots must be supplemented after a trainer exits. Full backup archives are separate from the compact submission logs and must not be deleted when packaging.
