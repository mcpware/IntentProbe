# One Forward, Sixteen Decisions: Task-Specific Hidden-State Readouts from a Small Local Language Model

**Status:** public research manuscript v0.1.0; not peer reviewed<br>
**Author:** Nicole BeWater<br>
**Publisher:** MCPware, a product and research brand of GEASS LABS INC.

![IntentProbe runs one frozen local model forward pass and reuses the hidden states across many
small task heads.](../../../docs/one-forward-many-decisions.svg)

## Abstract

Agent workflows increasingly call language models repeatedly to classify the same input for safety,
routing, sentiment, and policy decisions. We test a different architecture: run one frozen local
language model once, then apply independently trained task-specific heads to selected hidden states.
Our empirical programme has three parts. First, a 24-cell screen crosses six 230M--1B backbones with
sentiment, spam, toxicity, and jailbreak detection. Selected pooling and depth vary by model and
task; even toxicity and jailbreak labels on identical ToxicChat prompts select different readouts.
Second, a six-cell sentinel compares activation heads, matched TF-IDF, and direct output judgments
from two local backbones on authorization-to-execute, toxicity, and sentiment. Activation heads beat
both choice-likelihood judge prompts on toxicity after Benjamini--Hochberg correction, tie the
strongest Qwen3.5 sentiment judge, and lose to Qwen3.5 zero-shot choice likelihood on human-labelled
authorization by 0.2137 AUPRC (group-bootstrap 95% CI 0.0335--0.3973 in the judge's favour). The
result is task-by-model interaction, not universal probe superiority.

Third, we freeze 16 binary heads on one exact Qwen2.5-0.5B revision across five real-text sources:
toxicity, jailbreak, spam, sentiment, five emotion labels, and seven conversational-pragmatics
labels. Native heldout AUPRC ranges from 0.0836 to 0.9804, reflecting different tasks and
prevalences. On one six-core CPU, mean latency rises from 68.63 ms for one forward plus one head to
73.34 ms for the same forward plus 16 heads, a ratio of 1.069 (paired-bootstrap 95% CI
1.056--1.078). Sixteen repeated base-model forwards take 956.71 ms and are 15.26 times slower than
the shared path on matched prompts (95% CI 15.04--15.49). Peak process memory changes by +0.73 MiB;
the 16 saved probe artifacts occupy 386,121 bytes. The repeated-forward arm omits prompting,
LM-head scoring, and generation, so it is a lower bound rather than a complete judge implementation.
These results support a modular one-forward/many-readout runtime, while also showing that head
quality, calibration, and domain specificity remain separate engineering problems.

## 1. Introduction

A workflow may need to answer several questions about one user turn. Is it toxic? Is it a jailbreak?
Does it authorize an external action? Is the conversation becoming hostile? Which route should
handle it? A straightforward agentic implementation asks a language model each question separately
or loads one classifier per question. Both approaches repeat substantial computation.

Frozen-model probing offers another interface. A local decoder already computes one hidden vector
per token and layer. A small classifier can read those activations without generating an answer.
Multiple classifiers can also share the same forward pass. The systems promise is simple: the
backbone is the expensive sensor, while each task head is a small, replaceable readout.

That promise has two conditions. First, the hidden-state head must be useful enough for its own task.
Second, serving many heads must actually be cheaper than rerunning the backbone or maintaining a
fleet of models. Prior work establishes that hidden-state probes can outperform prompted answers in
some settings and that selected tokens and layers matter. It does not make every task, backbone, or
readout interchangeable.

We therefore separate four questions:

1. How much does the selected readout vary across models and tasks?
2. When does a hidden-state head beat, match, or lose to directly asking the same local model?
3. How does CPU latency and memory scale when one forward serves 1, 2, 4, 8, or 16 independent
   heads?
4. Do the same heads remain numerically stable and well behaved when texts come from other domains?

Our contributions are:

- a 24-cell, six-backbone readout screen over four real-labelled binary tasks;
- a six-cell same-backbone probe-versus-judge sentinel with paired uncertainty and a 12-test AUPRC
  false-discovery family;
- a 16-head Qwen2.5-0.5B suite whose heads use different pooling methods and hidden-state windows;
- a paired CPU scaling experiment at K=`1,2,4,8,16`, including peak process memory, saved-artifact
  size, and a deliberately favourable repeated-forward lower bound;
- native heldout discrimination and calibration for every served head, plus an explicitly unlabelled
  five-source score-shift diagnostic; and
- complete manifests, hashes, row-level timing artifacts without source text, and tests that enforce
  row alignment and interpretation boundaries.

We do not claim to invent activation probing, layer search, or multi-head inference. The contribution
is the measured combination: independently selected readout manifests for heterogeneous operational
tasks, served from one sub-1B local backbone pass.

## 2. Related work

### 2.1 Hidden representations versus generated judgments

[Tillman and Mossing](https://arxiv.org/abs/2504.20271) compare prompted outputs with raw, prompted,
and sparse-autoencoder activation probes for moderation, hallucination, and sentiment. Their results
show that generated labels are a serious baseline but do not exhaust information linearly available
inside the same model. [Representation-as-a-Judge](https://openreview.net/forum?id=VAISvCsrvG)
reports a similar extraction gap for reasoning-quality evaluation with frozen small models.
[Trick or Neat](https://aclanthology.org/2025.findings-acl.954/) finds direct prompting fragile for
several ambiguity classes while layer-wise probes can be strong. [Prompting Underestimates LLM
Capability for Time Series Classification](https://arxiv.org/abs/2601.03464) reports the same broad
pattern in a different modality.

These papers support the premise that a model can encode a distinction without reliably verbalizing
it. They do not support a universal claim that probes beat prompting. Task, prompt, training volume,
backbone, readout, and distribution shift remain material.

### 2.2 Single-pass classification and routing

[A BERTology View of LLM Orchestrations](https://aclanthology.org/2026.acl-long.1955/) is the closest
systems precedent. It searches token- and layer-selective probes on a frozen serving model, compares
against prompted and standalone classifiers, and measures the latency and memory saved by reusing a
serving pass. [LatentGate](https://aclanthology.org/2026.acl-industry.153/) uses a frozen small model,
mean pooling, PCA whitening, and a linear probe for semantic routing. [You Only Judge
Once](https://arxiv.org/abs/2604.10966) shows near-constant latency when a discriminative reward model
scores multiple responses in one pass. Those systems establish the value of shared computation.
Our experiment narrows the question to a 0.5B CPU sidecar whose heads are independently trained,
independently attachable, and allowed to read different layer windows and pooling rules.

### 2.3 Authorization and narrow signal contracts

[Embarrassingly Simple Access Control for Large Language Models](https://openreview.net/pdf?id=Re0fXzBeXH)
probes final-token, final-layer representations for access-control decisions and compares them with
direct and two-step prompting. It is close prior art for permission classification, although its task
is organizational access control rather than whether a user's language authorizes executing a
specific action. [Calibration, Not Answer Selection](https://arxiv.org/abs/2609.33290) supplies an
important boundary: a latent signal can be useful for one declared target while failing on a nearby
decision. We apply that boundary throughout. Decodability is not causal understanding, and a
probabilistic authorization score is not an execution permission.

## 3. Data and tasks

### 3.1 Cross-backbone matrix

The six-backbone matrix uses LFM2.5-230M, Granite 4.0 350M, Qwen2.5-0.5B, Qwen3-0.6B,
Qwen3.5-0.8B, and MiniCPM5-1B. It contains four binary tasks:

- binary sentiment from the official Stanford Sentiment Treebank root sentences;
- spam from the UCI SMS Spam Collection;
- toxicity from the human-authored portion of ToxicChat 0124; and
- jailbreak intent from the same ToxicChat prompts.

The cleaned SST-2 experiment retains 9,602 rows and an untouched 1,821-row official test. SMS Spam
is reduced from 5,574 messages to 5,159 exact-unique messages, with 774 held out. The leakage-repaired
human ToxicChat slice contains 5,441 prompts, with 2,790 official-test rows. Exact duplicates,
cross-split overlaps, and label-conflict groups are handled before activation extraction according to
each source manifest.

### 3.2 Shared 16-head suite

The shared Qwen2.5-0.5B suite adds five GoEmotions labels and seven Conversation AI Unhealthy
Comments attributes to sentiment, spam, toxicity, and jailbreak detection:

| Source | Served heads | Native heldout rows |
| --- | --- | ---: |
| ToxicChat 0124 | toxicity; jailbreaking | 2,790 each |
| UCI SMS Spam | spam | 774 |
| SST-2 | positive sentiment | 1,821 |
| GoEmotions | neutral; anger; sadness; admiration; curiosity | 5,409 each |
| Unhealthy Comments | condescending; dismissive; unfair generalisation; sarcastic; antagonize; generalisation; hostile | 4,424 each |

GoEmotions is pinned to the official filtered files and cleaned to 53,905 comments after removing
exact-text conflict and cross-split overlap groups. Unhealthy Comments is pinned to repository
revision `55934d0079073095af8b69807d235e40c8aceff4` and cleaned from 44,355 to 44,323 rows. Its labels
come from human judgments, but the released files omit author and article identifiers. UCC
uncertainty is therefore row-level and cannot establish unseen-author or unseen-thread performance.

### 3.3 Sentinel panels

The direct-judge sentinel fixes two backbones: LFM2.5-230M as the cost anchor and Qwen3.5-0.8B as the
quality anchor. It evaluates 512 heldout toxicity rows, 512 heldout sentiment rows, and all 156
human-labelled DelegationCheck authorization rows in 26 construction families. The authorization
activation heads are trained on the earlier silver authorization corpus and transferred without
using human labels for fitting.

## 4. Methods

### 4.1 Activation readout search

Backbone weights remain frozen. For Qwen2.5-0.5B, each input produces 25 hidden-state outputs of
width 896. The simple candidate family combines:

- masked mean, last non-padding token, or featurewise masked maximum over input tokens;
- every individual hidden output and every adjacent width-two or width-three window; and
- fold-local featurewise standardization followed by a class-weighted logistic head.

This gives 216 candidates per binary Qwen task. Window representations average the standardized
layer features. Selection uses the task's frozen validation metric: AUROC for SST-2 and AUPRC for
the imbalanced binary tasks. The winner is refit on train plus validation once, then scored on the
untouched test. The matched surface baseline is class-weighted word-and-character TF-IDF logistic
regression on the same fit rows. Other backbones use the same conceptual family with candidate count
adjusted for depth.

### 4.2 Direct same-backbone judges

For each sentinel cell, we compare the selected activation head and TF-IDF with four direct-judge
arms: zero-shot and rubric prompts, each read either by normalized YES/NO next-token likelihood or
strict deterministic generation. Choice likelihood supplies a continuous score without depending on
the model emitting a parseable word. Generated outputs use strict parsing, and invalid outputs are
retained and counted.

The primary probe-versus-judge family contains 12 paired AUPRC comparisons: two backbones, three
tasks, and two choice-likelihood prompts. Each cell uses 2,000 paired bootstraps. Authorization
resamples whole construction families; toxicity and sentiment resample rows. Benjamini--Hochberg
controls false discovery at 0.05 across the 12 AUPRC tests.

### 4.3 One-forward scaling protocol

The systems experiment freezes the exact Qwen2.5-0.5B revision
`060db6499f32faf8b98477b0a26969ef7d8b9987` and 16 unique probe hashes. K is cumulative and fixed at
`1,2,4,8,16`. Each K cell runs in a separate process. The latency panel contains the same 80 real
heldout texts at every K: 16 from each of the five sources. Batch size is one, two warm-up inference
calls precede timing and are not recorded, tokenization is included, and row identity and token count
must match exactly across K.

The shared path tokenizes once, calls `AutoModel` once with all hidden states, applies only the first
K frozen normalization/readout recipes, and records tokenization, forward, head, and total time. Peak
RSS is process `ru_maxrss`. Paired ratios use 2,000 row-bootstrap resamples.

The comparator tokenizes once and repeats the base `AutoModel` forward K times on ten fixed prompts.
It omits a task prompt, language-model head, choice-likelihood calculation, parser, and generation.
It therefore understates the cost of K separate LLM judges and is labelled a repeated-forward lower
bound throughout.

### 4.4 Quality, calibration, and score shift

Every head keeps its native heldout AUPRC, AUROC, accuracy, balanced accuracy, F1, Brier score, and
10-bin expected calibration error. We do not average incompatible labels into a single product
score. A separate 320-row panel applies all 16 heads to 64 texts from each source. Outside a head's
native label contract, these scores have no error label. We report means, upper tails, threshold
rates, and Kolmogorov--Smirnov distance from the full native score distribution as descriptive shift
diagnostics only.

## 5. Results

### 5.1 Readout choice varies across model and task

The four-task, six-backbone matrix contains 24 completed binary cells.

| Task | Selected pooling across six backbones | Normalized-depth range | Activation vs TF-IDF, both planned metrics |
| --- | --- | ---: | --- |
| Sentiment | masked mean 5; last non-padding 1 | 0.500--0.607 | activation advantage in 6/6 cells |
| SMS spam | masked mean 3; masked max 3 | 0.438--0.750 | activation advantage in 2/6; unresolved in 4/6 |
| Toxicity | masked mean 6 | within combined ToxicChat range 0.250--0.833 | activation advantage in 2/6 |
| Jailbreak | masked mean 3; masked max 3 | within combined ToxicChat range 0.250--0.833 | activation advantage in 0/6 |

The strongest within-input evidence comes from ToxicChat: toxicity and jailbreak use identical text
rows but select different depth and pooling patterns. Across its 12 cells, the median absolute
within-model task difference in normalized depth is 0.1875. A descriptive variance decomposition of
selected depth attributes 54.8% to model-by-task interaction, 43.5% to model, and 1.7% to task. With
only two tasks, these shares have no replicated interaction error term and are not population
estimates.

A universal ToxicChat readout—masked mean, width two, normalized depth 0.5—has mean validation AUPRC
0.7872. Cell-specific selection reaches 0.8220, a mean regret of 0.0348 for the universal rule on the
same validation cells. A model-specific rule reduces mean regret to 0.0142. These are descriptive
selection-set comparisons, but they show why a shared runtime still needs task-specific manifests.

### 5.2 Six-cell direct-judge sentinel

| Backbone | Task | Activation AUPRC | TF-IDF AUPRC | Zero-shot choice AUPRC | Rubric choice AUPRC |
| --- | --- | ---: | ---: | ---: | ---: |
| LFM2.5-230M | Authorization | 0.5529 | 0.4533 | 0.4423 | 0.5336 |
| LFM2.5-230M | Toxicity | 0.7571 | 0.7544 | 0.1300 | 0.1875 |
| LFM2.5-230M | Sentiment | 0.9149 | 0.9068 | 0.8896 | 0.6932 |
| Qwen3.5-0.8B | Authorization | 0.3567 | 0.4533 | 0.5704 | 0.4920 |
| Qwen3.5-0.8B | Toxicity | 0.8332 | 0.7544 | 0.1572 | 0.1013 |
| Qwen3.5-0.8B | Sentiment | 0.9424 | 0.9068 | 0.9369 | 0.9487 |

Six of the 12 activation-versus-choice-likelihood AUPRC comparisons reject after correction. Both
backbones' toxicity activation heads beat both prompts, with corrected q=0.0024 in all four cells.
For Qwen3.5 sentiment, activation differs by +0.0055 AUPRC from zero-shot choice likelihood and
-0.0063 from rubric choice likelihood; both intervals include zero and both corrected q-values are
0.7971. The strongest adverse cell is Qwen3.5 authorization: activation minus zero-shot choice
likelihood is -0.2137 AUPRC, with 95% group-bootstrap CI [-0.3973, -0.0335] and q=0.0480.

Generated labels are also operationally brittle for the smaller cost anchor on toxicity: 91 of 512
zero-shot generations and 163 of 512 rubric generations fail strict YES/NO parsing. Choice
likelihood avoids invalid strings but does not remove the task interaction. Toxicity strongly favours
the trained hidden-state readout, sentiment is competitive either way on Qwen3.5, and authorization
exposes the cost of silver-to-human transfer mismatch.

### 5.3 Sixteen native heldout heads

The shared Qwen suite contains 11 masked-mean and five masked-max winners. Selected normalized depth
spans 0.125--0.792. Native AUPRC spans 0.0836--0.9804 and AUROC 0.7330--0.9959; positive prevalence
spans 0.0206--0.4992.

| Head | Pooling / outputs | Prevalence | AUPRC | AUROC | Balanced accuracy | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Toxicity | mean 18--20 | 0.127 | 0.709 | 0.909 | 0.828 | 0.091 | 0.087 |
| Jailbreaking | max 5--7 | 0.032 | 0.665 | 0.950 | 0.811 | 0.029 | 0.030 |
| Spam | max 15--17 | 0.124 | 0.980 | 0.996 | 0.964 | 0.012 | 0.012 |
| Positive sentiment | mean 12 | 0.499 | 0.960 | 0.959 | 0.892 | 0.083 | 0.052 |
| Emotion: neutral | mean 15--16 | 0.329 | 0.668 | 0.817 | 0.738 | 0.177 | 0.117 |
| Emotion: anger | mean 5 | 0.037 | 0.323 | 0.879 | 0.782 | 0.102 | 0.160 |
| Emotion: sadness | mean 16 | 0.029 | 0.337 | 0.911 | 0.830 | 0.089 | 0.136 |
| Emotion: admiration | mean 2--4 | 0.093 | 0.601 | 0.917 | 0.851 | 0.105 | 0.158 |
| Emotion: curiosity | mean 3 | 0.052 | 0.414 | 0.928 | 0.857 | 0.081 | 0.121 |
| Condescending | mean 11--13 | 0.061 | 0.190 | 0.765 | 0.696 | 0.175 | 0.269 |
| Dismissive | max 13 | 0.034 | 0.084 | 0.745 | 0.677 | 0.169 | 0.268 |
| Unfair generalisation | max 15--17 | 0.021 | 0.100 | 0.830 | 0.717 | 0.126 | 0.185 |
| Sarcastic | mean 4--6 | 0.045 | 0.112 | 0.733 | 0.664 | 0.187 | 0.299 |
| Antagonize | mean 15 | 0.046 | 0.172 | 0.787 | 0.727 | 0.159 | 0.251 |
| Generalisation | max 17 | 0.022 | 0.101 | 0.812 | 0.720 | 0.130 | 0.196 |
| Hostile | mean 16--17 | 0.024 | 0.094 | 0.783 | 0.688 | 0.134 | 0.199 |

The rare UCC labels show why AUROC alone is not sufficient. Several heads rank positives above
negatives reasonably well while retaining low AUPRC and poor 0.5-scale calibration. Serving a head
cheaply does not make its threshold production-ready.

### 5.4 One forward plus K heads stays nearly flat

| K | Mean total ms | Mean head ms | Ratio to K=1 | Repeated-forward lower bound ms | Repeated / shared ratio | Peak RSS MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 68.63 | 0.38 | 1.000 | 57.88 | 0.94 | 3616.7 |
| 2 | 69.86 | 0.61 | 1.018 | 120.15 | 1.99 | 3615.5 |
| 4 | 70.73 | 0.73 | 1.031 | 238.73 | 3.95 | 3616.1 |
| 8 | 71.78 | 1.08 | 1.046 | 475.64 | 7.71 | 3614.0 |
| 16 | 73.34 | 1.85 | 1.069 | 956.71 | 15.26 | 3617.4 |

At K=16, the paired mean-latency ratio to K=1 is 1.069, 95% CI [1.056, 1.078]. The repeated-forward
ratio to its own K=1 baseline is 16.53, 95% CI [16.39, 16.69]. On the matched ten prompts, 16
repeated forwards are 15.26 times slower than one forward plus 16 heads, 95% CI [15.04, 15.49].

Head application itself takes 1.85 ms on average at K=16. The recorded K=16 peak RSS is 3617.4 MiB,
only 0.73 MiB above K=1. Saved probe files total 386,121 bytes. Since the shared path requests all
hidden outputs even at K=1, these measurements describe the intended attachable-head bus rather than
an optimized single-head special case.

### 5.5 Cross-domain scores move enough to require monitoring

The 320-row diagnostic finds meaningful distribution movement. The largest absolute mean shift is
+0.301 for the curiosity head on ToxicChat relative to its full native GoEmotions score distribution;
the KS distance is 0.406. This is not an error estimate because ToxicChat has no curiosity label. It
does show that a fixed threshold can fire at a very different rate when the source changes.

Live runtime scoring reproduces the archived float16-cache behavior closely on each head's 64 native
rows: the worst mean absolute score difference is 0.00645 and the minimum Pearson correlation is
0.97986. The worst single-row absolute difference is larger, 0.33849, on jailbreak detection. Live
deployment should therefore validate numerical precision and batch behavior instead of assuming bit
identity with an offline cache.

## 6. Discussion

### 6.1 The useful unit is a readout manifest, not a universal layer

The results reject two easy defaults. The final hidden output is not consistently best, and one
pooling/depth recipe does not serve every label. Even when inputs and backbone are identical,
toxicity and jailbreak can select different locations. A shared runtime should therefore expose a
stable activation bus while allowing each head to declare its own renderer, token pooling, layer
window, normalization, classifier, calibration, and threshold.

### 6.2 One pass changes the economics of workflow classification

The backbone dominates cost. Adding 15 more linear heads increases mean latency by about 6.9%, while
repeating only the base forward 16 times increases latency approximately sixteenfold. This supports
the IntentProbe one-forward, many-decisions architecture: one small local sidecar can inspect an
input once and fan the same activations into many narrow functions.

This does not require access to Claude, Codex, or another hosted model's hidden states. In that
deployment, IntentProbe is a local sidecar. If the primary model is itself local and exposes hidden
states, the same readout design can attach to that host and avoid the extra sidecar forward.

### 6.3 Cheap is not the same as correct

The sentinel prevents a misleading product story. Activation heads are dramatically better than
direct judges on the studied toxicity cells, essentially tied with strong Qwen3.5 choice-likelihood
judging on sentiment, and worse on human-labelled authorization after silver-data training. The
engineering advantage is therefore conditional: shared heads are attractive when labelled data and
readout selection produce an adequate task head. A poor head stays poor even when it is almost free.

Authorization especially needs a separate deterministic boundary. A probe may estimate whether the
context looks like authority to act. Exact payload review, recipient binding, scope, expiry, and the
right to execute remain explicit state and policy facts outside the classifier.

### 6.4 What remains for a full systems frontier

This experiment completes the shared-path curve and a conservative repeated-forward lower bound. It
does not yet measure two planned arms: one prompted multi-label judge call and a measured fleet of
compact task-specific classifiers on the same hardware. Those arms are required before claiming the complete
quality--latency--RAM frontier. The current result is narrower and still useful: given 16 existing
heads, their marginal serving cost is small relative to one 0.5B forward.

## 7. Limitations

1. The 16-head systems result uses one Qwen2.5-0.5B revision and one CPU machine.
2. K uses one fixed cumulative head order. Random head combinations are not evaluated.
3. K cells run in ascending order without randomized interleaving or CPU-frequency pinning. Paired
   intervals capture prompt variation, not run-order uncertainty.
4. Peak RSS is one process observation per K and includes the Python, PyTorch, Transformers, model,
   and inference stack.
5. The repeated-forward arm is intentionally incomplete and cannot be reported as a full prompted
   LLM-judge benchmark.
6. The direct-judge sentinel uses LFM2.5 and Qwen3.5, while the systems suite uses Qwen2.5. The two
   experiments answer complementary rather than perfectly matched questions.
7. The one-call multi-label judge and measured compact-classifier fleet remain incomplete.
8. Native heldout tasks differ in prevalence, size, source, and label definition. Their metrics must
   not be averaged into a universal quality number.
9. UCC lacks author and article grouping. Several other datasets also lack the independent unit
   required for unseen-user or unseen-conversation claims.
10. Cross-domain scoring is unlabelled outside each native task and cannot measure OOD accuracy.
11. The simple screen covers mean, max, and last-token pooling with adjacent windows and logistic
    heads. Learned token attention, nonlinear heads, layer mixtures, and causal interventions remain
    outside the confirmatory family.
12. Strong probe performance establishes predictive decodability under the sampled distribution,
    not a causal mechanism or human-like understanding.
13. Some source licences permit research but constrain redistribution. Public weight/data release is
    a separate licence decision from reporting aggregate internal research results.
14. Cold start, throughput under concurrency, energy, repeated-run variance, hot-loading semantics,
    and GPU performance remain unmeasured.

## 8. Conclusion

A frozen 0.5B local language model can serve 16 independently trained activation heads from one
forward pass with modest marginal CPU latency and almost no measured process-memory growth. The
result is not that one hidden layer solves every classifier. The 24-cell matrix, the 16-head suite,
and the direct-judge sentinel all show task-by-model-by-readout interaction. Sentiment, spam,
toxicity, jailbreak, emotion, pragmatics, and authorization do not share one accuracy story.

The defensible architecture is therefore modular. Run the backbone once; expose a versioned hidden-
state bus; let each task install its own tested readout manifest; keep calibration and policy outside
the backbone; and refuse to promote a cheap head whose native or shifted-data evidence is weak. In
the measured CPU setting, that architecture turns 16 repeated model forwards into one forward plus
1.85 ms of head work. Completing the multi-label-judge and compact-classifier arms will determine
where the full quality--latency--RAM frontier lies.

## Reproducibility and evidence map

- Paper-derived ledger: [`DERIVED_RESULTS.json`](DERIVED_RESULTS.json)
- Cross-backbone sentiment matrix: [`../experiments/SENTIMENT-SST2-MODEL-MATRIX.json`](../experiments/SENTIMENT-SST2-MODEL-MATRIX.json)
- Cross-backbone spam matrix: [`../experiments/SPAM-SMS-UCI-MODEL-MATRIX.json`](../experiments/SPAM-SMS-UCI-MODEL-MATRIX.json)
- Crossed ToxicChat matrix: [`../experiments/TOXICCHAT0124-MODEL-TASK-MATRIX.json`](../experiments/TOXICCHAT0124-MODEL-TASK-MATRIX.json)
- Six-cell sentinel: [`../experiments/SENTINEL-JUDGE-VS-PROBE-EXP001/result.json`](../experiments/SENTINEL-JUDGE-VS-PROBE-EXP001/result.json)
- Sentinel human-readable summary: [`../experiments/SENTINEL-JUDGE-VS-PROBE-EXP001/RESULT_SUMMARY.md`](../experiments/SENTINEL-JUDGE-VS-PROBE-EXP001/RESULT_SUMMARY.md)
- One-forward K-scaling result: [`../experiments/ONE-FORWARD-MULTIHEAD-EXP001/result.json`](../experiments/ONE-FORWARD-MULTIHEAD-EXP001/result.json)
- Systems summary and interpretation boundaries: [`../experiments/ONE-FORWARD-MULTIHEAD-EXP001/RESULT_SUMMARY.md`](../experiments/ONE-FORWARD-MULTIHEAD-EXP001/RESULT_SUMMARY.md)
- UCC primary result: [`../experiments/PRAGMATICS-UCC-QWEN-EXP001/result.json`](../experiments/PRAGMATICS-UCC-QWEN-EXP001/result.json)
- UCC expansion result: [`../experiments/PRAGMATICS-UCC-EXPANSION-QWEN-EXP002/result.json`](../experiments/PRAGMATICS-UCC-EXPANSION-QWEN-EXP002/result.json)
- Dataset/task source ledger: [`../SOURCE_LEDGER_2026-10-17.md`](../SOURCE_LEDGER_2026-10-17.md)
- Literature inheritance map: [`../LITERATURE_INHERITANCE_MAP_2026-10-18.md`](../LITERATURE_INHERITANCE_MAP_2026-10-18.md)
- Paper assumptions and remaining gates: [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md)
- Tests: [`../../../tests/test_one_forward_multihead.py`](../../../tests/test_one_forward_multihead.py) and [`../../../tests/test_sentinel_alignment.py`](../../../tests/test_sentinel_alignment.py)

## Appendix A. Environment

The systems run records Linux 6.8, Python 3.10.12, PyTorch 2.11.0, Transformers 5.8.0, six Torch
threads, and no NVIDIA GPU. Each metadata file preserves the exact model revision, timestamps,
package versions, input-manifest hashes, output hashes, elapsed time, and peak RSS.

## Appendix B. Evidence not combined into headline claims

The broader repository also contains an authorization-specific six-backbone pilot, memory-write
experiments, and a 77-way BANKING77 negative result. BANKING77 selects masked-mean output 1 on
Qwen2.5-0.5B but reaches 0.6967 accuracy and 0.7815 macro-F1, versus 0.9022 and 0.9022 for TF-IDF.
That result is useful evidence against universal probe superiority but is excluded from the 16-head
binary runtime because one multiclass softmax artifact is not equivalent to 77 independently trained
binary heads.

## Author and assistance disclosure

Nicole BeWater defined the research question, architecture, experiment priorities, interpretation,
and release boundary, and approved this manuscript. Coding assistants were used to implement and
check experiments, generate figures, and edit the manuscript. Quantitative claims are regenerated
from the versioned artifacts linked above and are covered by repository tests.
