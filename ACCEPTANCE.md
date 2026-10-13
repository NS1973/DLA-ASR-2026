# HW1 final acceptance evidence

The measured final model is e05cdea1, beam50 LM0.1. This ledger identifies evidence; final archive hashes and restore proof are external to avoid circular hashes.

| Requirement | Verified evidence |
|---|---|
| Course template | Full Git history; original template ancestor e9ca57d; TEMPLATE_PROVENANCE.md |
| Own acoustic model / own training | 24,471,453 parameters; staging and complete own parent chain; REPRODUCTION.md and FINAL_TRAINING_RECIPE.json |
| CTC / lengths / metrics / beam / data formats | 105 current local tests and105 independent clean-export tests; real overfit and resume logs |
| Four augmentations / media / W&B | Original and transformed WAV/PNG; offline logs retained from step0 in closed-log ZIPs |
| Model selection and test isolation | 28 best/last model SHA, full-dev other-primary clean<=20%;13 decoder settings; timestamped freeze before test |
| Exact full corpus results | FINAL_RESULTS.json all four splits, reference counts, errors, WER/CER and comparisons |
| Checkpoint restore | Matching model/full SHA, all tensors/state, optimizer/scheduler/RNG/gradients; finite real CPU probe |
| Anonymous inference resources | Model, source ZIP and frozen manifest downloaded without cookies and SHA checked |
| Fresh final Colab | Project initially absent, T4; original12 cells with/noGT, exactsource checks, independent workspaces, noGT3IDs and metrics=null |
| Full preservation | Closed4pairs control/LR3 plus2pairs mixed100 deepverified onE; prior mixed50/baseline/e640 unchanged |
| Report | REPORT.md contains actual final comparison/limits; REPORT.pdf must be accompanied by visual-page verification |
| Delivery archive and Git bundle | Final delivery verification must prove manifest/memberSHA/CRC, exact branch commit restore and template ancestor |
| Repository publication | Source ZIP and bundle fallback are authorized; anonymous Git clone must only be claimed after actual verification |
| Quality | test-clean13.55181% meets20%; test-other35.46797% exceeds30%; no grade guarantee |
| LLM rule | AI assistance disclosed; technical acceptance does not prove rule compliance |
