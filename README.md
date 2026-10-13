# DLA 2026 — HW1 ASR

Own DeepSpeech2-style CTC recognizer trained from scratch. Official example/asr history is retained in Git; implementation: src/asr.

Final full-set WER: **test-clean 13.55181%, test-other 35.46797%**. Dev-selected model: othermix100 best; own beam50 + OpenSLR trigram LM0.1. Other30% target was not reached. Exact counts, selection and comparisons: FINAL_RESULTS.json; report: REPORT.md / REPORT.pdf.

## Demo

[Open final Colab](https://colab.research.google.com/drive/1F1PlTt1mqMsS6AMHmO5qhl5epAieQ_4f). The public notebook bytes match the tested canonical source.

Open DLA_ASR_Demo.ipynb in Colab, select T4, paste your dataset archive URL and Run all. Its frozen anonymous source ZIP and resource manifest are SHA-verified automatically. No GitHub authentication is needed. Original12 cells passed with and without references; executed proofs are delivered in acceptance-final2040.zip.

## Local inference

Use Python3.12, install requirements.txt, then:

```bash
python scripts/download_resources.py
python inference.py checkpoint=artifacts/model.pt custom_dir=/path/to/dataset output_dir=predictions decoder=beam beam_width=50 token_topk=8 beam_workers=2 lm_path=artifacts/3-gram.pruned.1e-7.arpa.gz lm_weight=0.1 word_bonus=1.0
python calc_metrics.py --reference-dir /path/to/dataset/transcriptions --prediction-dir predictions
```

Dataset: audio/ID.wav (WAV/FLAC/MP3, any sample rate, mono/stereo), optional transcriptions/ID.txt. Without references, inference writes predictions and metrics=null. IDs are preserved. Resource downloading fetches model/example/LM; full training logs are separate archive members.

## Restore source and training

```bash
git clone --branch codex/hw1-asr source.bundle DLA-ASR-2026
cd DLA-ASR-2026
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest tests -q
```

Activate the environment first (Windows: .venv\Scripts\Activate.ps1; Linux: source .venv/bin/activate). Source ZIP is an alternative to bundle cloning. FINAL_TRAINING_RECIPE.json contains the selected saved cfg and all four extended-stage configs. Early own initialization: REPRODUCTION.md. Exact resume: RESUME_RECIPE.md; preserve the whole cfg, dataset/order, optimizer/scheduler/RNG/gradients. Set end_utc=None for reproduction after the campaign date. Full checkpoints are trusted project artifacts; never load an unknown pickle.

The mixed recipe uses clean100+360 and the exact100.1166h other subset repeated5 times. The preceding50h subset is included in100h. Exact IDs/hashes/texts and acquisition provenance are supplied separately; original operator machine paths require adaptation. No external acoustic pretrained model was used.

Delivery manifest lists source commit, weights, matching resume, all closed logs, parent weights, evaluation, acceptance proofs and SHA-256. Check final archive and bundle-restoration verification before treating a staging directory as a release. Historical baseline and e640 artifacts remain separately preserved.

Course rules prohibit LLM assistance. This work used AI assistance; technical verification does not establish compliance with that rule.
