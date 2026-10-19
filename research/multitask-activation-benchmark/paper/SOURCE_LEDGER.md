# Multi-task paper source and evidence ledger

Last reconciled: 2026-10-18 (America/Vancouver).

This ledger distinguishes external prior art, external datasets, and IntentProbe's own computed
results. Full retrieval queries and adverse findings remain in the parent benchmark ledgers.

## External scholarly sources

| Status | Source | Primary link | Paper use | Boundary/adverse point |
| --- | --- | --- | --- | --- |
| VERIFIED PRIMARY | Tillman & Mossing, *Investigating Task-Specific Prompts and Sparse Autoencoders for Activation Monitoring* | <https://arxiv.org/abs/2504.20271> | Probe versus generated-output precedent | Conditional results; does not establish universal probe superiority |
| VERIFIED PRIMARY | Meyoyan & Del Corro, *A BERTology View of LLM Orchestrations* | <https://aclanthology.org/2026.acl-long.1955/> | Closest token/layer selection and single-pass systems precedent | Uses a larger serving model and different hardware/tasks |
| VERIFIED PRIMARY | Li et al., *Representation-as-a-Judge* | <https://openreview.net/forum?id=VAISvCsrvG> | Small-model representation-versus-judge evidence | Reasoning-quality evaluation, not heterogeneous workflow heads |
| VERIFIED PRIMARY | Ratnakar et al., *LatentGate* | <https://aclanthology.org/2026.acl-industry.153/> | Non-generative small-model routing precedent | Routing only; one readout pipeline rather than 16 attachable tasks |
| VERIFIED PRIMARY | Karamolegkou et al., *Trick or Neat* | <https://aclanthology.org/2025.findings-acl.954/> | Prompt/probe extraction gap on ambiguity | Linguistic classification, not systems safety |
| VERIFIED PRIMARY | Schumacher et al., *Prompting Underestimates LLM Capability for Time Series Classification* | <https://arxiv.org/abs/2601.03464> | Cross-domain extraction-gap evidence | Time-series inputs, not the text runtime studied here |
| VERIFIED INDEXED FULL TEXT; PDF ACCESS FAILED | Rammouz et al., *Embarrassingly Simple Access Control for Large Language Models* | <https://openreview.net/pdf?id=Re0fXzBeXH> | Close authorization/access-control prior art | Direct PDF returned HTTP 403 during review; reopen final proceedings source before submission |
| VERIFIED PRIMARY | Xi et al., *Calibration, Not Answer Selection* | <https://arxiv.org/abs/2609.33290> | Narrow-signal and calibration boundary | A useful latent signal may fail on an adjacent decision |
| VERIFIED PRIMARY | Yang et al., *You Only Judge Once* | <https://arxiv.org/abs/2604.10966> | Adjacent one-pass scaling evidence | Multi-response reward scoring, not heterogeneous task heads |

## External dataset sources

| Status | Source | Canonical link | Local evidence use |
| --- | --- | --- | --- |
| PINNED | Stanford Sentiment Treebank | <https://nlp.stanford.edu/sentiment/> | Leakage-clean binary sentiment and fixed 1,821-row test |
| PINNED | UCI SMS Spam Collection | <https://archive.ics.uci.edu/dataset/228/sms+spam+collection> | Exact-deduplicated spam and fixed 774-row test |
| PINNED | LMSYS ToxicChat 0124 | <https://huggingface.co/datasets/lmsys/toxic-chat> | Human-only toxicity and jailbreak labels on identical prompts |
| PINNED | Google Research GoEmotions | <https://github.com/google-research/google-research/tree/master/goemotions> | Five human-labelled emotion heads after conflict/overlap repair |
| PINNED | Conversation AI Unhealthy Comments Corpus | <https://github.com/conversationai/unhealthy-conversations> | Seven human-labelled pragmatics heads; CC BY-NC-SA 4.0 |
| VERIFIED PRIMARY | UCC paper | <https://aclanthology.org/2020.alw-1.15/> | Dataset/annotation context |
| PINNED | DelegationCheck | <https://github.com/PieLord757/delegation-check> | Human-labelled authorization sentinel with 26 bootstrap groups |

## IntentProbe empirical artifacts

| Status | Artifact | Exact role |
| --- | --- | --- |
| COMPLETE | `SENTIMENT-SST2-MODEL-MATRIX.json` | Six-backbone sentiment readouts and planned paired TF-IDF comparisons |
| COMPLETE | `SPAM-SMS-UCI-MODEL-MATRIX.json` | Six-backbone spam readouts and planned paired TF-IDF comparisons |
| COMPLETE | `TOXICCHAT0124-MODEL-TASK-MATRIX.json` | Six-backbone × toxicity/jailbreak crossed readout analysis |
| COMPLETE | `SENTINEL-JUDGE-VS-PROBE-EXP001/result.json` | Six same-backbone judge/probe cells, 18 paired comparisons, 12-test BH family |
| COMPLETE | `ONE-FORWARD-MULTIHEAD-EXP001/result.json` | K scaling, 16 native quality/calibration records, 80 score-shift cells, live replay |
| COMPLETE | `ONE-FORWARD-MULTIHEAD-EXP001/HEAD_REGISTRY.json` | Unique probe hashes, exact backbone revision, layers, pooling, native metrics |
| COMPLETE | `ONE-FORWARD-MULTIHEAD-EXP001/WORKLOAD_MANIFEST.json` | Fixed real heldout latency and score-shift IDs |
| COMPLETE | `PRAGMATICS-UCC-QWEN-EXP001/result.json` | Four primary UCC heads |
| COMPLETE | `PRAGMATICS-UCC-EXPANSION-QWEN-EXP002/result.json` | Three preregistered UCC expansion heads |
| GENERATED | `paper/DERIVED_RESULTS.json` | Paper-level counts and ranges rebuilt from the complete artifacts above |

## Searches and exclusions carried forward

The companion literature review checked 36 additional scholarly queries after the broader prior-art
review and found no exact combination of a sub-2B frozen local decoder, independently attachable
heads with different readout manifests, heterogeneous operational tasks, and measured
K=`1,2,4,8,16` CPU scaling. This is a search result, not proof that no such work exists. The draft
therefore describes the exact measured combination and avoids a categorical “first ever” claim.

No external source supplies IntentProbe's latency, memory, accuracy, head count, or scaling ratios.
Every such number in the draft traces to an IntentProbe machine-readable artifact above.
