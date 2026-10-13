# HW1 acceptance ledger

This ledger records verified evidence and remaining gates. It is not a declaration of completion.

| Requirement | Current evidence | Final gate |
|---|---|---|
| Course template and Git provenance | Official example/asr history in local Git and published history bundle; TEMPLATE_PROVENANCE.md documents API snapshot ancestry | Verify final remote branch and bundle against frozen source commit |
| Private repository before deadline | NS1973/DLA-ASR-2026 private visibility verified by GitHub API | Change public visibility only after 16:00 MSK; verify anonymous access |
| Own CTC recognizer | 24,471,453-parameter model; padding/length and CTC regressions pass | Selected baseline continue2 last; model d919aed4 and full resume578761bf hashes/tensors/state verified on GMC and E |
| Own training from scratch | Real one-batch WER/CER 0; own staging, main and no_aug logs from step 0 retained | 12 initialization events/5 unique immutable parents collected and all SHA verified; final recipe to include in report |
| Four augmentations and separate evidence | augmentation-proof run: three real records, original plus four transformed WAV/PNG pairs, offline W&B | Include run and media in downloadable final logs |
| Greedy and own prefix beam | Exhaustive small-case tests; main10 full dev-clean greedy 35.73%, beam20 34.89%; dev-other 60.75% versus 60.05% | Full test-clean: greedy14.80523% ->ownbeam50 14.51423%; full test-other41.96359% ->41.42292%; same frozen d919aed4 model |
| Optional external LM | Own ARPA fusion; same main10 emissions: clean 33.73%, other 58.87% with weight 0.1 | 13 full-dev decoder settings evaluated; beam50 LM0.1/bonus1 selected before test. Full test WER14.16996%/40.46195% improves own plainbeam on both |
| CustomDirDataset / Hydra / IDs | WAV/FLAC/MP3, sample-rate/stereo, optional references, short audio tests pass | Actual final CPU CLI passed with GT (one train demo WER3/36), without GT (three WAV/FLAC/MP3 IDs, metricsnull); fresh final Colab repeat passed with both input cases |
| Corpus WER/CER script | Known edit-distance examples, strict ID matching, sum(errors)/sum(reference units) tests pass | FINAL_RESULTS.json records all four full split counts, errors and WER/CER; recomputed and checked |
| Checkpoint / resume | Exact next-step regression including dropout, partial gradients and RNG; real H200 resume; immutable initialization regression and CPU resume without parent path | Selected full checkpoint restores exact model/AdamW/scheduler/RNG/gradients; finite diagnostic update succeeds without modifying final weights; archive publication pending |
| Clean environment | Isolated CPU environment with no inherited packages: install, pip check, train/inference checks; 80 current tests pass | Source cb39bb3 ZIP SHA00ae68b0 installed in isolated clean-env; pipcheck,80tests3.49s, chosen inference with/noGT, actual train+resume: all8commands passed |
| Downloadable resources | Final selected model97898508bytes/d919aed4 and example235054bytes/52efa373 anonymously downloaded with SHA verification; inherited anyone-reader verified | resources.json populated model/example/LM/fullclosedlogs; model/example/logs anonymously downloaded and SHAverified; logs707817314bytes/fcaee6a1 |
| Fresh Colab | Actual fresh T4 preflight: installation, 23 tests, greedy/beam, metrics; executed notebook retained | Fresh T4/Python3.13.15/Torch2.11+cu130: all12originalcells/6code passed; ownprivateZIP preloaded and SHAverified; anonymous clone pending public visibility |
| Repeated notebook inputs | Regression with two sequential archives passes; each run has independent data and prediction workspace | Actual original demo RunAll withGT (WER3/36 trainfixture), then noGT3format IDs, metricsnull/no stale IDs; both executed notebooks/ZIPs verified |
| Report and logs | Working six-page report rendered and inspected; full intermediate dev tables and real curves | Final measured dev/test values, training recipe, ablation analysis and limitations; inspect every final PDF page |
| Reproduction archive | Verified intermediate draft:15 SHA entries,4 ZIP CRC checks,66279 log members and exactc6d0450 Git-bundle restore; not final delivery | Build and verify final archive from frozen model/source/configuration |
| Test isolation | All16 best/last candidates ranked on pooled full dev before any test; final model frozen12:39 | Evaluate only the frozen dev-selected model/decoder; no tuning on test |
| Quality targets | Full test-clean14.16996% meets20; full test-other40.46195% exceeds30. These are quality-score thresholds7/8, not mandatory acceptance gates | Report actual test results even if targets are missed |
| Course LLM rule | AI assistance explicitly disclosed in README/report/provenance | Technical acceptance does not establish compliance with this rule |

The packaging check in delivery/packaging-check is a repair/verification artifact, not the final submission. Its evidence is in logs/packaging-check.json. Final delivery will use a separate directory and its own manifest.
