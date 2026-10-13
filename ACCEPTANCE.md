# HW1 acceptance ledger

This ledger records verified evidence and remaining gates. It is not a declaration of completion.

| Requirement | Current evidence | Final gate |
|---|---|---|
| Course template and Git provenance | Official example/asr history in local Git and published history bundle; TEMPLATE_PROVENANCE.md documents API snapshot ancestry | Verify final remote branch and bundle against frozen source commit |
| Private repository before deadline | NS1973/DLA-ASR-2026 private visibility verified by GitHub API | Change public visibility only after 16:00 MSK; verify anonymous access |
| Own CTC recognizer | 24,471,453-parameter model; padding/length and CTC regressions pass | Freeze selected checkpoint and exact model/config hashes |
| Own training from scratch | Real one-batch WER/CER 0; own staging, main and no_aug logs from step 0 retained | Collect complete selected initialization chain and commands |
| Four augmentations and separate evidence | augmentation-proof run: three real records, original plus four transformed WAV/PNG pairs, offline W&B | Include run and media in downloadable final logs |
| Greedy and own prefix beam | Exhaustive small-case tests; main10 full dev-clean greedy 35.73%, beam20 34.89%; dev-other 60.75% versus 60.05% | Repeat comparison on final immutable checkpoint |
| Optional external LM | Own ARPA fusion; same main10 emissions: clean 33.73%, other 58.87% with weight 0.1 | Choose final LM settings on dev and repeat measured comparison |
| CustomDirDataset / Hydra / IDs | WAV/FLAC/MP3, sample-rate/stereo, optional references, short audio tests pass | Run final checkpoint on fresh custom input and preserve IDs |
| Corpus WER/CER script | Known edit-distance examples, strict ID matching, sum(errors)/sum(reference units) tests pass | Final dev/test metric JSON and counts |
| Checkpoint / resume | Exact next-step regression including dropout, partial gradients and RNG; real H200 resume | Publish matching safe model-only and trusted full resume checkpoints |
| Clean environment | Isolated CPU environment with no inherited packages: install, pip check, train/inference checks; 41 current tests pass | Repeat final installation and final checkpoint inference from source archive |
| Downloadable resources | Public example anonymously downloaded and SHA-verified; Drive/HTTP downloader tests pass | Populate resources.json with final model, LM when selected and complete logs, verify anonymous downloads |
| Fresh Colab | Actual fresh T4 preflight: installation, 23 tests, greedy/beam, metrics; executed notebook retained | Execute all final demo cells with final resources and a user link; record any source-access setup accurately |
| Repeated notebook inputs | Regression with two sequential archives passes; each run has independent data and prediction workspace | Final notebook run with and without references |
| Report and logs | Working four-page report rendered and inspected; full intermediate dev tables and real curves | Final measured dev/test values, training recipe, ablation analysis and limitations; inspect every final PDF page |
| Reproduction archive | Packaging check passed 13 SHA-256 entries, ZIP CRC, 29,200 log files and exact Git-bundle restoration | Build and verify final archive from frozen model/source/configuration |
| Test isolation | Test-clean/test-other transferred but never used for training or selection | Evaluate only the frozen dev-selected model/decoder; no tuning on test |
| Quality targets | WER <=20% clean / <=30% other not yet established | Report actual test results even if targets are missed |
| Course LLM rule | AI assistance explicitly disclosed in README/report/provenance | Technical acceptance does not establish compliance with this rule |

The packaging check in delivery/packaging-check is a repair/verification artifact, not the final submission. Its evidence is in logs/packaging-check.json. Final delivery will use a separate directory and its own manifest.
