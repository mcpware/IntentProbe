# Authorization Activation Benchmark — Experimental Design

## Primary estimands

1. Scenario-held-out AUROC and AUPRC for `authorized` versus `not authorized` on public binary data.
2. Macro one-vs-rest AUROC/AUPRC and cost-weighted error for `ALLOW/CONFIRM/BLOCK` on SEND-state gold data.
3. False-approve rate at predeclared recall targets, with scenario-clustered 95% bootstrap intervals.
4. Regret of a universal normalized-depth rule relative to each model's validation-selected optimum.
5. Task-by-model interaction across authorization, memory-write, and control datasets.

## Splits

- Public source-provided splits are preserved but audited for overlap.
- Contrast pairs and paraphrase families are grouped.
- Where IDs are absent, a documented normalized fingerprint and source-specific family extractor is used.
- One external benchmark remains untouched until the final candidate is frozen.
- Cross-source transfer is reported separately from within-source generalization.

## Representations

Each frozen backbone is run once per renderer version with `output_hidden_states=True`. Cache:

- last non-padding token;
- masked token mean;
- full-token mean where this intentionally includes template tokens;
- featurewise maximum;
- optional user-span and action-span means when span boundaries are deterministic.

Layerwise features are standardized using training data only. PCA/whitening, if tested, is fitted inside
each training fold and treated as a separate hypothesis.

## Layer search

- all individual hidden-state indices;
- adjacent windows of width 2 and 3;
- a bounded forward sparse-combination search;
- normalized depth `q = layer_index / num_hidden_layers`;
- universal windows defined before the final test;
- block-type-aware analysis for hybrid architectures.

Model/layer/pooling/head selection uses nested group cross-validation. A random row split is prohibited.

## Heads

Primary: class-weighted logistic regression.

Sensitivity:

- difference-in-means direction;
- linear SVM with calibrated probabilities;
- small MLP with a fixed parameter budget.

The project calls logistic regression a linear probe. Ordinary least-squares linear regression is not
used as the primary classifier.

## Baselines

- majority / label-prior;
- action-only and request-only ablations;
- TF-IDF word+character n-gram logistic regression;
- dedicated encoders `ProCreations/auto-0.4b` and `auto-1b`;
- zero-shot/output-logit classifier when the backbone interface permits it;
- deterministic state machine on cases containing complete structured state.

## Metrics

- AUROC, AUPRC, balanced accuracy, macro F1;
- false approve, false deny, recall at 1% and 5% false-positive rates;
- Brier score and expected calibration error with reliability plots;
- coverage/risk curve when `CONFIRM` is treated as abstention;
- CPU p50/p95 latency, peak RSS, model load size, cached-feature size;
- per-source, per-language, per-action, and per-failure-mode slices.

## Repetition and uncertainty

- Fixed split seed plus at least five model-head seeds for stochastic heads.
- Clustered bootstrap over scenario groups, minimum 2,000 replicates for published intervals.
- Paired bootstrap for model comparisons on the same test groups.
- Declare the multiple-comparison family before layer results are viewed; publish the complete layer
  curve, not only the maximum.

## Initial experiment queue

| ID | Hypothesis | Data | Models | Gate |
|---|---|---|---|---|
| EXP001 | Surface text already explains most public authorization labels | approve-or-deny | TF-IDF, action-only | establish floor |
| EXP002 | Qwen2.5 activation beats surface baseline under group holdout | approve-or-deny | Qwen2.5-0.5B | CI excludes no-gain margin |
| EXP003 | Best normalized depth transfers across canonical backbones | approve-or-deny | Phase-2 four | universal-depth regret |
| EXP004 | Signal transfers to same-operation/different-permission cases | AuthorizationPairs | frozen candidates | untouched external test |
| EXP005 | One backbone is generally strong across tasks | authorization + memory + controls | promoted models | interaction analysis |

Each experiment gets its own directory and immutable config, environment snapshot, logs, metrics,
failure record, and lessons learned.
