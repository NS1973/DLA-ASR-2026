# DLA HW1 ASR execution ledger

Approved plan: chat 01a13245-8e70-7373-8dae-fbd7d26072df, deadline 2026-10-13 16:00 Europe/Moscow.

## Objective
Own CTC ASR, measured WER/CER, own beam, four audio augmentations, complete logs, reproducible training/inference and working Colab demo.

## Decisions
- Separate clone of the official `example/asr` template, branch `codex/hw1-asr`; preserve template history.
- Ruling: a separate user-approved clone supplies isolation; do not create a second checkout inside it.
- GMC host has no external egress; transfer official data and offline wheels from workstation.
- Existing GPU models were stopped on explicit user request. Xorg is untouched; GPU reset blocked by Xorg, memory free.
- Two free H200 NVL; existing image has Python 3.12.3, PyTorch 2.11.0+cu129, torchaudio, numpy.
- Goal active; ten-minute heartbeat automation `1-asr-10` active.
- Course explicitly prohibits LLM use. No representation of independent human authorship is made.

## Progress
- Template cloned; goal and heartbeat created.
- Own DeepSpeech2-CTC (24,471,453 parameters), audio pipeline, four augmentations, own prefix beam, corpus metrics, Hydra training/inference and offline logs implemented.
- GMC offline environment and detached supervisors ready. Real H200 one-batch overfit reached WER/CER 0 at step 81; this is not held-out quality.
- Review fixes: save partial gradients for accumulation resume; stateless seed/epoch/sample augmentation; preserve safe checkpoint on exceptions; explicit epoch restart when batch size changes. Exact next-step resume regression passes.
- 15 tests pass on workstation and in the H200 container. Official datasets are downloading/transferring with checksums; dev transfer has an independent queue.
- Colab notebook drafted; resource links, actual fresh Colab run and final report are pending.

## Verification
- SSH authenticated host `kv-ai-00.gmc.local`; GPU memory used 14 MiB per device.
- Local gh token invalid. Publishing must be separately verified; Git bundle is the offline fallback.

## Remaining
Full-corpus training, dev/test evaluation, full artifact download, fresh Colab verification, publication, packaging.
