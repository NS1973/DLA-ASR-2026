# Template provenance and Git history

This project started from the official Blinorot/pytorch_project_template `example/asr` branch, as required by the DLA 2026 HW1 instructions. The workstation checkout retains that history and uses branch `codex/hw1-asr`. The upstream remote is named `template` and its push URL is disabled.

The implementation branch retains the original template and local implementation ancestry. While workstation Git CLI authentication was unavailable, a project-only GitHub Actions workflow imported the verified Git bundle into the private repository. The API snapshot commits are retained separately under `codex/api-snapshots`; they do not replace the source history.

The latest published original-history bundle is available at `history/DLA-ASR-2026.bundle` on `codex/api-snapshots`, and the reproducible delivery contains `source.bundle`. Restore the latter using `git clone --branch codex/hw1-asr source.bundle restored-project`. Verify the restored HEAD against the delivery manifest.

The course prohibits LLM use. This project was prepared with an AI coding assistant; technical completeness does not establish compliance with that rule.
