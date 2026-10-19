# Sentinel judge-versus-probe alignment experiment

This is the six-cell check frozen by the literature-inheritance review. It does not repeat the broad
question of whether hidden-state probes can ever outperform prompted outputs. It tests whether the
two IntentProbe CPU anchors behave consistently with that literature on three representative local
tasks.

## Frozen cells

| Task | Quality anchor | Cost anchor |
| --- | --- | --- |
| Authorization to execute | Qwen3.5-0.8B | LFM2.5-230M |
| Toxicity | Qwen3.5-0.8B | LFM2.5-230M |
| Sentiment | Qwen3.5-0.8B | LFM2.5-230M |

The model pair was chosen before sentinel scoring. Both weights were already local, both expose a
chat template, and both passed a two-row fixed-label generation smoke test. Qwen3.5 is the quality
anchor among the locally cached post-trained candidates with completed public-control readouts. LFM
is the smallest completed activation-screen backbone.

## Integrity boundary

- `build_eval_manifest.py` freezes exact evaluation IDs and source hashes before any sentinel model
  scoring.
- ToxicChat and SST-2 use the same 512 deterministic stratified test IDs for both models and all
  comparators.
- DelegationCheck uses all 156 human-labelled rows and preserves `group_id` for uncertainty.
- Prompt text and parser rules live in `config.json` and `run_judge.py`; changing either creates a new
  experiment rather than overwriting this one.
- The existing activation and TF-IDF score artifacts are reused for ToxicChat and SST-2. The
  authorization activation heads must be fitted on the silver development source without using the
  DelegationCheck labels, then evaluated on all 156 external rows.
- Raw predictions, invalid outputs, timing observations, package versions, model revisions, and file
  hashes are retained.

`run_authorization_probe.py` refits only the validation-selected EXP002 head from the preserved
silver-training activation cache, verifies exact validation AUPRC reproduction, and extracts the
selected layers for all 156 human-labelled DelegationCheck rows. Those human labels are
evaluation-only and never enter the head fit.

## Expected outputs

```text
evaluation_manifest.json
predictions/<model-alias>__<task>__<prompt>__<method>.jsonl
results/<model-alias>__<task>.json
SENTINEL-SUMMARY.csv
SENTINEL-SUMMARY.md
```

The one-forward/many-head scaling experiment is separate. It consumes compatible frozen readout
manifests after this alignment check and measures `K = 1, 2, 4, 8, 16` without changing task
definitions or prompts.
