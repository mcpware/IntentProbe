---
license: apache-2.0
language:
  - en
  - zh
library_name: intentprobe
base_model: Qwen/Qwen2.5-0.5B
datasets:
  - chris0809/memory-write-gate-20k
tags:
  - activation-probe
  - linear-probe
  - memory
  - long-term-memory
  - memory-write
  - ai-agents
  - intent-detection
  - local
  - qwen
metrics:
  - roc_auc
---

# IntentProbe: 15 KB activation probe for memory-write decisions

[![GitHub](https://img.shields.io/badge/GitHub-mcpware%2Fintentprobe-181717?logo=github)](https://github.com/mcpware/intentprobe)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](./LICENSE)

**A local research probe for ranking whether one conversational utterance is worth carrying into
long-term memory.** The Qwen2.5-0.5B backbone stays frozen. IntentProbe reads masked-mean hidden-state
output **11** and applies a 15 KB class-weighted logistic head.

This is the second public IntentProbe task after tool-use authorization. It demonstrates that the
same activation-probe mechanism can be specialized for a different agent boundary by changing the
labelled data and selecting the strongest internal representation for that task.

Highlights:

- **216 representations compared:** every individual hidden output, adjacent width-2/3 windows, and
  three pooling rules.
- **Intermediate representation selected:** masked mean at output 11 (normalized depth 0.458).
- **Development AUPRC 0.8897 / AUROC 0.9030** across 3,200 rows in 32 held-out seed families.
- **Stronger point estimate than word+character TF-IDF:** +0.0531 AUPRC and +0.0419 AUROC.
- **Small learned component:** the numeric probe is 15,305 bytes; the base model is not bundled.

## What is included

| Component | Value |
| --- | --- |
| Frozen backbone | [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B) |
| Pinned revision | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| Input | one user utterance |
| Representation | attention-mask-aware mean of hidden-state output 11 |
| Maximum input | 256 tokens, right truncation |
| Learned component | class-weighted logistic regression |
| Probe size | 15,305 bytes |
| Labels | `SKIP=0`, `SAVE=1` |
| Release type | open research preview with reproducible local scorer |

## Quick start

    python3 -m pip install -r requirements.txt
    python3 score.py --text "I prefer concise answers in future conversations."

Example output:

    {
      "save_score": 0.9992653463346063,
      "decision_at_0_5": "SAVE",
      "decision_at_development_1pct_false_save_threshold": "SKIP",
      "status": "research_preview"
    }

The output above was reproduced on CPU with the pinned backbone and artifact. `save_score` is a
ranking signal, not permission to persist personal data. A product should still apply explicit consent,
sensitivity, retention, deletion, and user-control policy.

## Representation search

| Pooling | Best representation | Validation AUPRC | Validation AUROC |
| --- | --- | ---: | ---: |
| **masked mean** | **output 11** | **0.8897** | **0.9030** |
| last non-padding token | output 22 | 0.8803 | 0.8918 |
| masked max | outputs 11–13 | 0.8656 | 0.8739 |

The winning layer differs from IntentProbe's authorization detector. That is the engineering point:
the useful signal location is task- and backbone-dependent rather than automatically the last layer.

## Development comparison

The search used a synthetic silver dataset with 200 conceptual seed families. Train and validation
were grouped by seed; no seed crossed splits.

| 3,200 validation rows / 32 held-out seed families | AUPRC | AUROC |
| --- | ---: | ---: |
| Qwen activation probe | **0.8897** | **0.9030** |
| Word+character TF-IDF logistic | 0.8366 | 0.8611 |

Paired seed-family bootstrap differences:

- AUPRC: +0.0531, 95% CI −0.0199 to +0.1469.
- AUROC: +0.0419, 95% CI −0.0194 to +0.0998.

These are development estimates after representation selection. An independent human-written LUFY
evaluation is being run and will be added as a versioned benchmark receipt rather than retrofitted
into the selection process.

## Intended use

- Research on which internal representations encode durable conversational information.
- Ranking candidate memories before a separate consent and retention policy.
- Comparing activation probes with surface classifiers and prompted LLM judges.
- Reproducing the layer/pooling search on other small, locally runnable backbones.

## Scope

- The training labels are silver and synthetic; human-written transfer is a separate evaluation.
- The 1% false-save threshold was selected on development data and is not a universal calibrated
  policy threshold.
- Inputs after 256 tokens are invisible to this release.
- The artifact is tied to the pinned Qwen architecture and renderer.
- Future utility, consent to store, permission to retrieve, and a request to forget are separate
  decisions.

## Integrity and provenance

| File | SHA-256 |
| --- | --- |
| `probe.npz` | `2a988fbfa76e36d4fe0b1aa4fcca1b8b19a9a5c90d702452f8d36e1defb6ca92` |
| frozen selection receipt | `ca441ea5dc5e98d34eded1080142c729c66b96754577c7fb5a884042f8d69f4c` |
| experiment config | `1680ab22347d67c0bc619fb37597dee5c4a88920a49d3f6a33eb243d65f540d3` |

Training source: [`chris0809/memory-write-gate-20k`](https://huggingface.co/datasets/chris0809/memory-write-gate-20k)
at revision `004add9de074a96e1d2eeddf8c3f695f9e121b2a` (Apache-2.0).

The NumPy artifact contains numeric arrays only and is loaded with `allow_pickle=False`.

## Links

- [IntentProbe source and full experiment](https://github.com/mcpware/intentprobe/tree/main/research/memory-write-gate)
- [Qwen2.5-0.5B backbone](https://huggingface.co/Qwen/Qwen2.5-0.5B)
