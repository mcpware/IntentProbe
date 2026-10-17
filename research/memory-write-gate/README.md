# Memory Write Gate

Research and engineering workspace for deciding what an AI assistant should save, skip, forget, or
retrieve under the user's privacy and persistence preferences.

This folder is deliberately separate from the authorization project. The activation benchmark engine
may later be shared, but label semantics, source provenance, product policy, and external tests remain
independent.

## Current status

- Living plan, source registry, label policy, and benchmark design: written.
- Raw source acquisition: reproducible through `scripts/fetch_sources.py`.
- Source QA: expanded to human and competitor sources. Current LUFY, historical LUFY annotations,
  REALTALK, MemUse, AgentMemGate, CareCall, and NUANCED are pinned for research; `memory-traces`
  remains gated and untouched.
- MEM-EXP001 surface baseline: complete; the official test is consumed.
- TF-IDF test result: AUPRC 0.8241 and AUROC 0.8890 over 2,600 rows in 26 seed families; the
  validation-selected 1%-false-save threshold transfers to 2.69% false saves and 11.38% SAVE recall.
- MEM-EXP002 activation screen: complete on development data. All 216 candidates were fitted using
  14,200 train rows and selected on 3,200 validation rows from 32 held-out seed families.
- Activation winner: Qwen2.5-0.5B masked mean at hidden-state output 11, with validation AUPRC 0.8897
  and AUROC 0.9030. It leads TF-IDF by 0.0531 AUPRC and 0.0419 AUROC, although both paired 95%
  seed-bootstrap intervals cross zero.
- Frozen artifact: `experiments/EXP002-qwen-activation-screen/selected_probe.npz`, accompanied by all
  candidate results, row-aligned validation scores, category/language slices, and integrity tests.
- Public release package: `../../hf/intentprobe-qwen2.5-0.5b-memory-write/` with a local scorer,
  model card, hashes, and frozen benchmark summary.
- Human benchmark: LUFY historical release provides 6,470 aligned user turns with three independent
  importance votes each. The reproducible audit is in `HUMAN_DATASET_AUDIT_2026-10-17.md`; activation
  transfer scoring is the next locked experiment.
- Model weights: pinned by the authorization benchmark's shared model matrix; not bulk-downloaded.

## First commands

```bash
python research/memory-write-gate/scripts/fetch_sources.py
python research/memory-write-gate/scripts/qa_sources.py
pytest -q research/memory-write-gate/tests
```
