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
- 23 tests pass on workstation and in the isolated clean environment on GMC (23 passed in 1.95s at 03:24 MSK). Clean CPU environment has no inherited site packages; pip check and checkpoint inference passed.
- Staging training: 1,024 real train records, 3,200 optimizer steps, 100 epochs completed. These are preliminary weights, not the final quality claim.
- Own beam and optional pruned trigram ARPA LM were evaluated on identical immutable staging emissions. On 32 dev-clean records: greedy 467/567 word errors, beam-10 466/567, beam-10 + LM weight 0.5 449/567. CER became worse with LM; final decoder remains undecided pending main-model dev evaluation.
- Separate augmentation-proof run completed: 3 real train records, all 4 transformations, original/transformed audio and spectrograms, offline W&B.
- Both dev splits are transferred and verified. Train-clean-100 transfer is nearing completion; train-clean-360 has a resumable streaming transfer gated by train-clean-100 readiness and the 06:00 cutoff.
- Main and no-augmentation supervisors are detached and waiting for train-clean-100; their initial weights, scheduler horizon and seed match. Ablation stops after 8,920 steps or 07:00.
- Inference detects native BF16 support; T4 uses FP32. Offline W&B now includes random per-record text tables and numeric train/dev metrics.
- Colab notebook drafted; resource links, actual fresh Colab run and final report are pending.

## Verification
- SSH authenticated host `kv-ai-00.gmc.local`; GPU memory used 14 MiB per device.
- Private repository NS1973/DLA-ASR-2026 was created and its private visibility verified. Git CLI authentication requires the user's GitHub sudo confirmation; local Git bundle remains available while publication is pending.
- An authenticated Google Colab page is accessible in Edge; a fresh complete notebook execution remains pending final weights and resource links.

## Remaining
Full-corpus training, dev/test evaluation, full artifact download, fresh Colab verification, publication, packaging.
