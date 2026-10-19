# Multi-task activation benchmark

**Public architecture release:** [IntentProbe: One Local Model Pass, Many Task-Specific
Probes](../../docs/ONE_LOCAL_MODEL_PASS_MANY_TASK_SPECIFIC_PROBES.md)<br>
**Author:** [Wan Sheng Leung](https://github.com/ithiria894)<br>
**Publisher:** MCPware, a product and research brand of GEASS LABS INC.

![Panel a, runtime: a request and its context enter a frozen local model that runs one forward pass over layers 1 to N; the shared hidden states fan out in parallel to toxicity, sentiment and jailbreak readouts, each with its own layer selection and pooling, feeding independent heads that each return a score to a host policy of allow, review or block. Panel b, offline: labelled data feeds a readout selection and head fit that produces a versioned head, loaded later as a compatible head without retraining the others.](../../docs/intentprobe-architecture-overview.png)

The simple idea is that a model may already react differently to toxicity, sentiment, spam,
authorization and other concepts even when prompting it for `YES` or `NO` does not recover those
signals reliably. IntentProbe runs the frozen model once, exposes the hidden states, and lets each
small supervised head read the layers and token summary that work for its own label contract.

The first systems benchmark now measures that architecture with 16 heads. Adding the other 15 heads
raised mean CPU latency from 68.63 ms to 73.34 ms. Sixteen repeated base-model forwards took 956.71
ms versus 62.70 ms for the shared path on the matched ten-prompt comparison. Quality did not move in
one direction across tasks:
toxicity favoured the trained activation heads, Qwen3.5 sentiment was tied, and the current
authorization transfer favoured the direct judge.

This is the primary paper track for IntentProbe.

Project hierarchy: **GEASS LABS INC. → MCPware → IntentProbe → papers, runtime, and probe
products/heads**. MCPware is the publisher and GitHub/Hugging Face organization; IntentProbe is the
umbrella under which the research and products stay connected.

The public implementable specification for the shared system is
[`ARCHITECTURE_DISCLOSURE_V0_1.md`](ARCHITECTURE_DISCLOSURE_V0_1.md). It distinguishes current code
and results from planned extensions. The first measured one-forward, many-probe CPU experiment is
now complete with 16 independent Qwen2.5-0.5B heads. The first machine-readable contracts are the
[`readout manifest`](../schemas/intentprobe_readout_manifest.schema.json) and
[`multi-head receipt`](../schemas/intentprobe_multihead_receipt.schema.json) JSON Schemas; they are
specifications with validation tests, not a claim that the fan-out runtime is already complete.

Individual probes are useful products, but the paper asks a broader engineering question:

> Across small locally runnable language models and semantically different tasks, which hidden-state
> readouts remain useful after source leakage, surface-text baselines, calibration, and grouped
> uncertainty are controlled?

The paper's object of study is the complete **activation-readout pipeline**, not logistic regression
and not any one probe. The engineering matrix is:

`input/renderer -> token region -> token pooling -> hidden-state depth -> layer combination ->`
`fold-local normalization -> classifier head -> calibration/threshold -> runtime deployment`

IntentProbe therefore studies how to cut and read a frozen model's hidden states: which tokens to
retain, which hidden-state outputs to use, whether to mean/max/otherwise pool them, whether adjacent
outputs should be averaged or concatenated, whether depth should be expressed absolutely or
relative to architecture depth, how scaling/whitening changes the signal, and only then which head
best converts that representation into a decision. A released probe is one frozen point in this
larger design space.

## Prior art and claim boundary

The complete patent and academic landscape review is in
[`PRIOR_ART_REVIEW_2026-10-18.md`](PRIOR_ART_REVIEW_2026-10-18.md), with the database coverage,
queries, failures, and canonical sources preserved in
[`PRIOR_ART_SOURCE_LEDGER_2026-10-18.md`](PRIOR_ART_SOURCE_LEDGER_2026-10-18.md).
The focused
[`probe-versus-judge literature inheritance map`](LITERATURE_INHERITANCE_MAP_2026-10-18.md)
records which general findings are cited from prior work and reduces our own generated-judge work to
a six-cell sentinel panel plus the novel many-head scaling experiment.

The paper does not claim to invent activation probes or one-pass multi-head classification. Its
target contribution is the crossed small-model benchmark and measured one-prefill/many-probe
runtime: task-specific internal readouts, direct same-backbone judge baselines, OOD specificity, and
the accuracy--latency--RAM frontier against separate classifiers.

## Topic-open task registry

The matrix is no longer restricted to five safety topics. Admission depends on real human data,
clear labels, provenance, and leakage control—not topic. Sentiment, emotion, spam, politeness,
ordinary intent classification, and other clean semantic tasks are first-class generality tests.

The rules are frozen in [`TASK_ADMISSION_POLICY.md`](TASK_ADMISSION_POLICY.md), and the expandable
execution queue is in [`TASK_REGISTRY.json`](TASK_REGISTRY.json). The current queue includes
sentiment, toxicity, jailbreak intent, SMS spam, fine-grained emotion, banking/assistant intent,
unhealthy-conversation attributes, hate/offensive speech, politeness, topic, truthfulness,
authorization, prompt injection, and memory write.
The latest human-data search, admission decisions, and source ledger are in
[`TOPIC_EXPANSION_2026-10-18.md`](TOPIC_EXPANSION_2026-10-18.md).

## Shared factors

- backbone architecture and size;
- absolute and normalized layer depth;
- masked mean, last-token, max, and valid field/span pooling;
- single layers, bounded adjacent windows, and selected layer combinations;
- a fixed logistic probe for linear-accessibility comparisons, followed by a frozen-representation
  head-family ablation covering calibrated linear, covariance-aware, and bounded nonlinear heads;
- matched TF-IDF and text-encoder baselines across the matrix, with same-backbone direct judges on
  the preregistered sentinel panel;
- grouped external transfer, calibration, low-error operating points, CPU latency, RAM, and artifact
  size.

## Evidence boundary

Naturally occurring or human-authored, human-labelled held-out data is the main external evidence.
The confirmatory test contains no synthetic rows. Model-generated data is capped at 10% of headline
training/development, remains source-tagged, and must be removed in a sensitivity analysis.
Every task keeps its own label definition and independent grouping unit; scores with incompatible
semantics are not pooled into a misleading universal safety number.

## Current state

- Authorization: six-backbone pilot and one frozen held-out evaluation complete.
- Memory write: public 15 KB synthetic-trained probe released; human-label experiments complete for
  Qwen2.5-0.5B. Direct human training raises fixed-output-11 AUROC to 0.7032. A 216-candidate screen on
  12 users selects masked-mean outputs 7--8; on five held-out users it reaches 0.1575 AUPRC / 0.7842
  AUROC versus 0.1726 / 0.7892 for TF-IDF, with paired intervals including zero. A post-test anchor
  comparison measures +0.0607 AUPRC and +0.1088 AUROC over output 11, showing why readout selection
  is a central engineering variable rather than a cosmetic detail.
- Prompt injection and tool/skill poisoning: historical evidence exists, but the shared protocol must
  be rerun.
- ToxicChat human-only tasks: acquisition, leakage repair, frozen readout search, and official-test
  evaluation are complete for Qwen2.5-0.5B. Toxicity selects masked-mean outputs 18--20 and reaches
  0.7093 AUPRC / 0.9091 AUROC versus TF-IDF at 0.7262 / 0.8998; paired intervals include zero.
  Jailbreaking selects masked-max outputs 5--7 and reaches 0.6650 / 0.9505 versus TF-IDF at
  0.8264 / 0.9817; paired intervals favour TF-IDF. The different winners on identical prompts are
  strong evidence that layer and pooling choice is task-specific. See the
  [result summary](experiments/TOXICCHAT0124-QWEN-EXP001/RESULT_SUMMARY.md).
- Sentiment: the second complete six-backbone matrix is finished on leakage-clean SST-2. All six
  runs use the same 1,821 untouched official-test sentences. Masked mean wins for five backbones;
  Qwen3.5 instead selects last-nonpadding outputs 11--13. Selected normalized depth ranges from
  0.500 to 0.607 and window width from one to three outputs. Every backbone's planned
  activation-minus-TF-IDF 95% paired interval is above zero on both AUPRC and AUROC. Cross-backbone
  test ranking remains exploratory. See the validated
  [six-model matrix](experiments/SENTIMENT-SST2-MODEL-MATRIX.md).
- SMS spam: the first complete six-backbone matrix is finished on the official UCI corpus, cleaned
  from 5,574 to 5,159 exact-unique messages. All six runs use the same 774 untouched test messages.
  The validation-selected readout changes by architecture: masked mean wins for Granite, Qwen3, and
  MiniCPM5; masked max wins for LFM2.5, Qwen2.5, and Qwen3.5. Selected normalized depth ranges from
  0.438 to 0.750 and selected window width from one to three outputs. Granite and MiniCPM5 have
  activation-minus-TF-IDF 95% paired intervals above zero on both AUPRC and AUROC; four other
  backbones remain unresolved. Cross-backbone test ranking is exploratory. See the validated
  [six-model matrix](experiments/SPAM-SMS-UCI-MODEL-MATRIX.md).
- GoEmotions neutral-label presence: official filtered files cleaned from 54,263 to 53,905 comments
  after removing label-conflict groups and exact overlap. A fixed 5,000-row screening budget selects
  masked-mean outputs 15--16. On 5,409 untouched test comments, activation reaches 0.6675 AUPRC /
  0.8175 AUROC versus TF-IDF at 0.6578 / 0.8228; paired intervals include zero. See the
  [result summary](experiments/EMOTION-GOEMOTIONS-NEUTRAL-QWEN-EXP001/RESULT_SUMMARY.md).
- GoEmotions four-emotion panel: admiration, anger, curiosity, and sadness were locked before
  screening, then evaluated on the same frozen rows and reused Qwen activations. Masked mean wins
  all four targets, but selected depth is not universal: admiration 0.125, anger 0.208, curiosity
  0.125, and sadness 0.667. Activation does not beat the matched TF-IDF baseline on these four
  targets under the planned paired intervals; anger favours TF-IDF on both AUPRC and AUROC, while
  curiosity remains unresolved. See the validated
  [four-emotion matrix](experiments/EMOTION-GOEMOTIONS-PANEL-QWEN-EXP001/EMOTION-PANEL-SUMMARY.md).
- Unhealthy Comments pragmatics: the official 44,355-row corpus is pinned and cleaned to 44,323
  rows after excluding one missing-text row, two exact-text label-conflict groups, and remaining
  exact overlap with test-first retention. Seven Qwen2.5 heads are complete on the same untouched
  4,424-comment test: condescension, dismissiveness, sarcasm, unfair generalisation, antagonizing,
  generalisation, and hostility. Their winners span masked mean and masked max, normalized depths
  0.208--0.708, and widths one to three. Heldout AUPRC ranges from 0.0836 to 0.1898 against class
  prevalences of 0.0206--0.0608. Only dismissiveness has a paired AUPRC interval excluding zero, and
  it favours TF-IDF; the other six activation-minus-TF-IDF AUPRC intervals include zero. Calibration
  is materially worse than TF-IDF for all seven heads.
- BANKING77 fine-grained intent: official data cleaned from 13,083 to 13,071 exact-unique queries.
  A fixed-budget multiclass screen selects masked-mean output 1. On 3,079 official-test queries,
  activation reaches 0.6967 accuracy / 0.7815 macro-F1 versus TF-IDF at 0.9022 / 0.9022; paired
  intervals favour TF-IDF. This is a useful negative result for activation probes on highly lexical
  77-way intent classification. See the
  [result summary](experiments/INTENT-BANKING77-QWEN-EXP001/RESULT_SUMMARY.md).
- Direct-judge sentinel: all six model-by-task cells are complete for LFM2.5-230M and Qwen3.5-0.8B
  across authorization-to-execute, toxicity, and sentiment. Activation heads beat both constrained
  direct-judge prompt variants on toxicity after Benjamini--Hochberg correction. On sentiment,
  Qwen3.5 activation and direct choice-likelihood judging are tied. On human-labelled authorization,
  the Qwen3.5 activation head loses to zero-shot choice likelihood by 0.2137 AUPRC (group-bootstrap
  95% CI -0.3973 to -0.0335; BH q=0.048). The result is task/model interaction, not universal probe
  superiority. See the [sentinel summary](experiments/SENTINEL-JUDGE-VS-PROBE-EXP001/RESULT_SUMMARY.md).
- One-forward scaling: the complete Qwen2.5-0.5B suite contains 16 independently trained heads. Mean
  CPU latency rises from 68.63 ms at K=1 to 73.34 ms at K=16, a paired-bootstrap ratio of 1.069
  (95% CI 1.056--1.078). On the matched ten prompts, the one-forward/16-head path takes 62.70 ms and
  16 repeated base forwards take 956.71 ms, a ratio of 15.26 (95% CI 15.04--15.49). This repeated-forward
  comparator is a lower bound for separate prompted judges. Peak RSS changes by +0.73 MiB and the 16
  saved probes occupy 386,121 bytes. See the
  [systems summary](experiments/ONE-FORWARD-MULTIHEAD-EXP001/RESULT_SUMMARY.md).

The detailed prospective protocol currently lives in
[`../authorization-send-detector/paper/MULTITASK_BENCHMARK_PLAN.md`](../authorization-send-detector/paper/MULTITASK_BENCHMARK_PLAN.md).
This index is the canonical entry point so the paper is not mistaken for an authorization-only or
memory-only study.

## Next execution order

1. add the one multi-label judge-call arm and a measured compact-classifier fleet to complete the
   three-way systems comparison;
2. run deterministic repeatability and cold-start/throughput measurements for the shared runtime;
3. replicate the completed GoEmotions and UCC targets across the five non-anchor backbones;
4. acquire and audit the Stanford Politeness files, then run the same frozen contract;
5. expand each admitted task across the six-model panel without touching its external test;
6. run selection-budget, seed, solver-convergence, and head-capacity sensitivity, then fit the
   cross-task analysis only after the matrix and missing-cell rules are frozen.
