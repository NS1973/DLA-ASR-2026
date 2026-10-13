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
- 29 tests pass on the workstation (5.99s at 03:59 MSK). The clean GMC environment passed 27 tests before the two Drive-resource tests were added. Clean CPU environment has no inherited site packages; pip check and checkpoint inference passed.
- Staging training: 1,024 real train records, 3,200 optimizer steps, 100 epochs completed. These are preliminary weights, not the final quality claim.
- Own beam and optional pruned trigram ARPA LM were evaluated on identical immutable staging emissions. On 32 dev-clean records: greedy 467/567 word errors, beam-10 466/567, beam-10 + LM weight 0.5 449/567. CER became worse with LM; final decoder remains undecided pending main-model dev evaluation.
- Separate augmentation-proof run completed: 3 real train records, all 4 transformations, original/transformed audio and spectrograms, offline W&B.
- Both dev splits are transferred and verified. Train-clean-100 and both test splits are also transferred, hash-verified and extracted; train-clean-360 has a resumable streaming transfer gated by train-clean-100 readiness and the 06:00 cutoff.
- Main training is active and the no-augmentation run completed ten epochs (8,920 steps); their initial weights, scheduler horizon and seed match. Ten-epoch checkpoints and logs are frozen under runs/ablation-10epoch; no_aug now continues as an additional training candidate until 06:00, preserving the ten-epoch result.
- Inference detects native BF16 support; T4 uses FP32. Offline W&B now includes random per-record text tables and numeric train/dev metrics.
- Actual fresh Colab preflight on Tesla T4 / Python 3.13.15 / torch 2.11+cu130 passed: installation, 23 tests, greedy and own beam inference, and corpus metrics. Executed notebook is saved under logs/. Source and preliminary weights were uploaded manually for this preflight; final resource downloads and full demo Run All remain pending.
- Main training failed on a fast utterance shortened below CTC alignment length by speed augmentation. All 28,539 original train100 records are feasible. Speed-up now respects the target-plus-repeat alignment length; regression passes. The run was gracefully restored from its safe checkpoint and passed the formerly failing step 3,541, reaching step 4,580 with finite loss. No samples were silently skipped.
- Dev at main step 4,460: WER 44.58% clean / 66.65% other on 256 records per split. No-augmentation step 8,920: 38.33% / 64.79%; these first values were not a matched-step comparison or final test results. At exactly 8,920 steps main reached dev-clean WER 35.82%, versus no_aug 38.33% on the identical 256 validation records.
- Train360 downloader now uses four exact bounded ranges, appends in order and preserves resumability. The streaming SFTP mirror is active; inclusion still requires complete transfer before 06:00.

- Campaign manager v8 is active (PID 9203, child trainer PID 9204 at 03:57). It starts a new refinement schedule if 60 epochs finish early; at 06:00 it chooses the best dev checkpoint from main/no_aug/refine100 and starts finetune460 only when train360 was fully ready before 06:00, otherwise finetune100. The final training cutoff remains 12:30. Extra-data phase uses 40 epochs bounded by the cutoff; 100h refinement uses 120, also bounded.
- Decoder comparison for frozen no_aug10: on the fixed seed-42 dev-clean subset greedy WER 38.82%, beam20 37.61%; LM weights 0.3/0.5 worsened clean. A matching frozen main10 decoder sweep is running/completing; inspect runs/decoder-main10 and decoder-noaug10.
- Example ZIP is published read-only on Drive and anonymously downloaded/hash-verified using the resource downloader. URL and checksum are in data/publication.json. It is a real training example for I/O demonstration, not a held-out quality measurement.

## Verification
- SSH authenticated host `kv-ai-00.gmc.local`; GPU memory used 14 MiB per device.
- Private repository NS1973/DLA-ASR-2026 was created and its private visibility verified. Git CLI authentication remains unavailable. The connected GitHub app published all 116 source/history files on private codex/hw1-asr; remote train.py and the bundle blob SHA were verified. API snapshot commit 8c9b44904c0a211b9916fa0f0e0357f6ea0125b8 represents local d8c9398; full original ancestry is preserved in history/DLA-ASR-2026.bundle.
- Fresh Colab preflight was actually executed; the final demo remains pending final weights and resource links.

## Remaining
Full-corpus training, dev/test evaluation, full artifact download, fresh Colab verification, publication, packaging.

### 2026-10-13 04:23 MSK checkpoint
- Campaign v9 supervisor PID 12072 is active. Both initial refinement and final finetuning repeat fresh schedules when an epoch budget ends early, preserving the 06:00/12:30 cutoffs. Main step 17,840: dev-clean 32.07%, dev-other 56.29% WER on the fixed 256-record monitoring subsets. No_aug step 21,408: 33.61% / 61.83%.
- 34 regression tests passed locally and in the clean GMC environment. Transfer recovery now handles missing temporary files, an already-renamed final archive, and acknowledged-prefix truncation. A real remote retry proof preserved the original readiness timestamp rather than moving it past the data cutoff.
- Four-channel resumable train360 transfer PID 7200 is active; 8.69 GB acknowledged at 04:21, local download 20.59/23.05 GB. Data inclusion remains conditional on complete verification/extraction before 06:00.
- Full intermediate dev evaluation on the immutable main10 model is saved locally: all 2,703 dev-clean and 2,864 dev-other records, same emissions for every decoder. Dev-other greedy WER 60.75%, beam20 60.05%, beam20 + LM weight 0.1 WER 58.87%. These are development results, not final test results.
