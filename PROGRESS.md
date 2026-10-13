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

### 2026-10-13 04:38 MSK acceptance checkpoint
- Fresh user datasets in the demo now receive separate workspace and prediction folders on every execution. A regression first reproduced contamination from two sequential archives, then passed after the notebook fix. 35 tests pass locally (8.45s) and in the clean offline GMC environment (1.79s).
- Official train-clean-360 download completed and archive validation passed: 23,049,477,885 bytes, SHA-256 146a56496217e96c14334a160df97fffedd6e0a04e66b9c5af0d40be3c792ecf. Resumable upload remains active and must finish before 06:00 to join training.
- Completed training/evaluation evidence was downloaded as data/completed-evidence-0432.tar.gz, 125,275,596 bytes, SHA-256 fde411d3ede865ed828e5fa88debabf66959e047ea0de41721ad12f49cbe4a62. Root-owned W&B debug symlinks are excluded from local extraction; real metrics, offline run files and media are preserved.

### 2026-10-13 04:55 MSK source history
- GitHub Actions run 38334166104 completed successfully in 11 seconds. It verified and imported the original-history bundle into codex/original-history. The exact original fc607d9 commit is now available remotely; codex/hw1-asr was moved to that original commit after preserving the API snapshot head as codex/api-snapshots. Future verified bundles can be imported the same way, then the source branch advanced normally.
- Main/no_aug initialization was read directly from actual logs: staging best step 2784, epoch 86, same source_state for both runs. REPRODUCTION.md preserves the original 100-epoch scheduler horizon when truncating that prefix.
- Current main best monitoring dev-clean WER is 27.55% at step 28544. Both trainer processes were confirmed live. Train360 upload acknowledged 14.06/23.05 GB and continues.

### 2026-10-13 05:10 MSK campaign checkpoint
- Main/no_aug trainers remain live. Main monitoring dev-clean WER at step 36572 was 25.22%; latest step 37464 is 26.27% clean / 52.57% other. No_aug step 44600: 28.88% / 58.20%. Each monitoring split contains the same 256 seed-1 sampled records; these are not final test metrics.
- Train360 transfer acknowledged 16.75/23.05 GB at 05:06 and continues with four channels; local download is fully hash-verified. Inclusion still requires extraction/readiness before 06:00.
- Detached LR-candidate controller PID 12814 was verified live. It waits until 06:00 and until the own no_aug trainer exits, then trains on GPU 1 at initial LR 3e-4 with the same data gate and 12:30 cutoff. It does not terminate any process. Candidate selection records actual parent checkpoints; this is a further dev candidate, not an assertion of an isolated LR effect.
- 36 tests passed locally (5.14s) and in the clean offline GMC environment (2.37s). The routing regression first failed on the missing API, then a Windows path expectation was corrected; the complete rerun passed.

### 2026-10-13 05:23 MSK final-evaluation controller
- Detached controller PID 13185 is live. It waits until 12:30 MSK, then snapshots run candidates, chooses by full dev-clean greedy WER, selects decoder on pooled full-dev corpus WER, and only then evaluates full test-clean/test-other. Final results have not been produced yet.
- Independent review caught non-atomic snapshot/cache writes that would poison restart after truncation. A regression reproduced the failure; atomic replace and validation/rebuild now pass. Matching full resume is checked against all frozen model tensors and training state. Focused re-review found no remaining substantial issue.
- Complete suite: 41 local tests passed in 5.19s; 41 clean offline GMC tests passed in 1.74s. The pipeline-order test uses isolated fixtures, not real test data.

### 2026-10-13 05:47 MSK data and batch checkpoint
- Train360 archive SHA-256 verified remotely and fully extracted. Ready marker actual mtime2026-10-13T02:43:45.177223Z, strictly before06:00MSK. Metadata/CTC audit passed all104014 records; min margin2, mono16kHz. Train100 audit passed all28539, min margin6. Free remote space50.30GiB after extraction.
- Batch benchmark completed on GPU1, finite BF16 CTC backward/optimizer at32/64/128/256:310/495/579/621 utterances/sec, allocated1.83/3.28/6.22/12.12GiB. Audio I/O excluded. Candidate uses128; main stays32. Own waiting controller12814 was replaced safely before06:00; new controller14917 is live. No trainer was stopped.
- OOM regression reproduced duplicate overrides preventing batch backoff; parsing the requested batch once now gives128 ->64 and accumulation1 ->2. Complete suite42tests passed locally5.32s and clean offline1.92s.
- Main completed60epochs/53520steps normally at05:45:50 and automatically started refinement trainer14995. Exact main60 snapshot SHA4d93c19132ca8530b28dd18eccbd50c68db5de57bf68fd03f7aabcbb9c2c6b4a saved beside no_aug60. Full-dev greedy ablation comparison is running on free GPU1. Initial upload to root-owned run directory failed; upload via user-owned incoming then copy inside project container succeeded.
- Working report rebuilt and all5pages visually inspected; graph heading/caption now stays with the figure. .gitattributes preserves PDF bytes across Windows/Linux checkouts.

### Full 60-epoch augmentation ablation
- All2703 dev-clean and2864 dev-other records evaluated with exact last checkpoints from53520 steps. Main WER24.4495%/51.3798%; no_aug28.8151%/57.8884%. Main CER7.9752%/22.2142%; no_aug9.8920%/26.0608%. Actual model hashes in logs/ablation60-model-hashes.json; configs/counts in four logs/ablation60-*.json files. One seed only; no seed variance claim. Test sets remain unused.

### 2026-10-13 06:01 MSK expanded-corpus startup
- Both detached trainers started on132553 utterances. GPU0finetune460 PID15949 batch32/LR1e-4 (4143batches); GPU1lr_candidate_finetune460 PID16298 batch128/LR3e-4 (1036batches). Same own main best parent step52628/epoch58. Initial checkpoints exist; first loss/gradients finite. Full epoch speed and dev results pending.
- Source d220b62655489fe137d9f28e719e8394dab11378 is available on private codex/hw1-asr after successful Actions run38339521521. Original template ancestry retained. Git PDF bytes identical to inspected working artifact.

### 2026-10-13 06:06 MSK first expanded-data epoch and no-reference fixture
- GPU1 candidate completed one full132553-record epoch: train_start03:00:24Z, epoch_end03:05:12Z (~288s including initialization/media). Monitoring256-record seed1 dev: cleanWER25.2773%,other50.4364%; not full-dev/test. GPU0 remains in epoch0, step2580; both last checkpoints younger than45s at06:06.
- Public no-reference fixture prepared from own train example374-180298-0000: mono16kFLAC,stereo8kWAV,stereo44.1kMP3. Archive694252bytes SHA80693d40e670df1df7c24c4d5c92931960a3fe3fe912f16ce1ab9bb6f2ccbf51. Anonymous Drive download verified; preflight own saved model decoded all3 preserved IDs, metricsnull. Final-checkpoint/fresh-Colab repeat remains pending.

### 2026-10-13 06:09 MSK measured epoch budget
- Main completed first full expanded-data epoch at06:09:26 (~546s from train_start includinginit/media), monitoring256 cleanWER24.5184%/other50.1148%. Candidate completed second epoch at06:09:24 (252s since prior epoch_end), clean24.2460%/other50.2297%. Both actual PIDs15949/16298 live and continuing. About380minutes remain until12:30, sufficient for many measured full passes.
- GPU claims refreshed at06:08:42/43, expire07:08. Heartbeat ACTIVE until16:00 verified. Full/test/finaldecoder selection remains scheduled12:30.

### 2026-10-13 06:15 MSK measured error analysis
- New scripts/analyze_errors.py decomposes minimum word edit alignments, validates corpus sums and optional inference summary, rejects missing references/duplicateIDs. Three initialtests failed on missingmodule; implemented ->3passed0.16s. Documented full command python -m pytest tests -q ->45passed5.46s local and45passed1.81s cleanGMC. A broad pytest without tests scope collected the restored packaging copy and failed on duplicate module imports; rerun at documented tests scope succeeded without deleting artifacts.
- Exact main60 full-dev predictions: clean S11145/D1055/I1101 =13301errors/54402words; other S22005/D2236/I1936 =26177/50948. Analysis JSONs match original summaries includingCER. Frequentpairs and deterministic high-error examples retained, rawfrequency limitations stated.
- Working report now6pages with erroranalysis; all6rendered pages inspected. Optional pypdf page-count command unavailable locally; actual rendered PNG count6 verified, rendering succeeded. Finalreport/analysis remainpending selectedmodel.

### 2026-10-13 06:40 MSK safety snapshot
- Candidate completed10 expanded-data epochs; monitoring256seed1 cleanWER19.8482%,other45.7051%. These are dev subset values, not target test acceptance. Immutable model snapshot copied locally, SHA517323eb77fb6e01614ad38158b472e1a95582f3de7e3c409d122e840e96b0fb, step10360/epoch9. Hash and safe load state verified. CPU ownbeam10 inference on three no-reference format fixtures passed with exact IDs, metricsnull.
- Main completed4 expanded epochs, monitoringclean21.4439%/other47.3588%. Both owntrainers continue. Median actualepoch intervalsmain498.93sec,candidate209.68sec.
- Commit ec6efd6 adds Docker context exclusions for localdatasets/runs/checkpoints/env; source/dependency files remainincluded. This is only build-context configuration; no running container changed.

### 2026-10-13 07:10 MSK decoder probe and parent backups
- Detached probe18275 completed normally at07:09:21 on fixed256seed42 dev-clean/other. SnapshotSHA550acdfcb28a055be50b49dfc9d2d30a1ead8811b5748706895bf1b7bcdef54d (candidatebeststep18648/epoch17). Width10, LMweights0.1/0.2/0.3/0.5, wordbonuses0/1/2:14distinctsettings per split. Greedy19.2044%/44.4492%; ownplainbeam10 18.8125%/44.1056%; bestLM0.2bonus1 18.1266%/42.5381%. Pooledsubsetbest remains existingbonus1; finalcontroller unchanged and full-devselection stillpending. CompleteJSON/log/metadata downloaded; allhashes/counts verified.
- Both unique legacyparentmodels copied to data/initialization-models andSHAverified (staging0513d0...,main4f0d97...). Selectedchainlaterparents stilltoinclude in finalinitializations archive.
- Both ownGPUtrainerslive07:10: mainstep32580, candidate20060. Candidate19completedepochs monitoring256seed1WER17.8634%/43.9596%; different subset from decoderprobe. No test used.

### 2026-10-13 draft archive recovery
- User chose autonomous Gitbundle/archive fallback; no further GitHub authentication requested. Bundle0c39299 verified complete. Captured66279regularlogfiles: TAR444277262bytes SHA4e9a32b8f06c4018035434ed60b0a1f2abc2ca1a13cc4ba67708e8b3a7a941d8. Download/hash/extraction passed.
- First archive packaging failed on metadataJSONmtime0 (ZIP default rejectsbefore1980), after slow directory traversal. Ownership-guardedstop found no live matchingprocess; no process was stopped. Failedoutputcheckpoint-0720 is not delivery. New --logs-archive streams TAR directly, clamps ZIPheaderdates and atomically replaces logsZIP onlyaftercompletion. Tests firstfailedmissinghelper ->3targetedpassed0.11s/all49passed6.12s. Successfulrealrerun pending; trainingneverstopped.

### 2026-10-13 07:51 MSK verified draft and report curves
- Intermediate autonomous delivery/checkpoint-0720-v2 successfully rebuilt from captured TAR; 15 manifest hashes, 4 ZIP CRC checks, all66279 log-entry SHA comparisons, both parent-weight hashes verified. Archive750288964bytes SHA7c543c53c88b8f0730cda0fad0600d020c8edeef593fbe746af29bdc8c2a1e1e. Sourcebundle restores exactc6d0450 with template ancestry;49tests and CPU beam+LM3-format no-reference inference pass. This is not final submission; fulltest/matchingresume/finalColab remainpending.
- Report curves now support actual shared UTC elapsedhours; resets/repeated optimizer steps split traces. Two regression tests firstfailedmissingAPI ->passed. All51tests pass locally6.33s and cleanGMC1.82s. Independentreview found no substantialissues; chronological graph from freshlycaptured actual5runlogs rendered and visuallyinspected (logs/campaign-curves-0750.png). Trainingprocesses unchanged.

### 2026-10-13 08:10 MSK immutable parent collection
- Added scripts/collect_initializations.py: campaign-only own parent snapshots, original initialization events, explicit retrospective evidence, SHA/state/path checks, deduplication and atomic archive replacement. Four regressiontests firstfailedmissingmodule ->passed;55complete tests pass locally5.48s and cleanGMC1.91s. Independent review found no substantialissues.
- First real collection rejected main historical retry events because legacy proof covered only firsttimestamp. Verified actual sourceSHA unchanged, exact loggedsource states, parentcompletion/mtime beforeeach of9events; separately wrote legacy-parents-expanded.json, original logs/proof untouched. Exactfloatmtimecomparison firstfailed on submicrosecond serialization difference; explicit1us tolerance passed with actualdeltas -0.238/-0.477us. Retry initialization can precede resume loading, not a new trainingstart.
- Successful realrerun:9events,2uniqueparents. Local data/initializations-0800.zip181523827bytes SHAa6cbb3274f6fac7876f8c6a1920fa6cf6792bb203ed017edfa4a2182bb670801 matchesremote; allZIPCRC/modelhashes verified. Metadata logs/legacy-initialization-parents-expanded.json and logs/initialization-collection-0800-verification.json. Mustrecollectafterfinalcontinuationphases.
- 08:07bothH200trainersreallylive; candidate36completedepochs monitoring256seed1 cleanWER15.3921%,other41.1576%. Checkpoints100/75sec. Finaltest selectionstill12:30, no testresults.

### 2026-10-13 08:23 MSK first continuation and matching backup
- Candidate40 completed41440steps normally08:19:38; fixeddev256seed1 finalWER14.9056%/40.8360%. Automatic continue1 PID22624 starts08:19:55, batch128 LR5e-5, same464.20h. At08:20 actualprocesslive step60 finite loss/grad, lastcheckpoint29sec. Exact parentSHAa51d1268802190790766736a6e97f8f23a17de78787dec8e7d349f8a21204fa5 and stateverified from immutable snapshot. Startup folder/metrics creation race resolved by re-poll, no restart.
- Both completed candidate40 model97898508bytes and fullresume293733105bytes downloaded to data/safety-snapshots/candidate40-*.pt. AllremotelocalSHA/size checks and everytensor/state matchpass. Resume contains optimizer/scheduler/RNG/gradients. Evidence logs/candidate40-backup-verification.json. No final selection/testclaim.
