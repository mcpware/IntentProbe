# Task admission policy

The benchmark is **data-quality first and topic-open**. A task is not admitted because it is an
agent-safety task, and a useful task is not rejected because it is sentiment, emotion, spam,
politeness, topic, or another ordinary semantic distinction.

## Evidence tiers

1. **Tier A — naturally occurring human text.** The text came from a real review, comment, message,
   query, conversation, competition submission, or other human event. Labels may be direct human
   judgments or objective event metadata with a documented rule.
2. **Tier B — controlled human-authored text.** People wrote the examples for a study or benchmark.
   These rows are human data, but they are reported separately from naturally occurring data.
3. **Tier C — synthetic or model-generated text.** These rows may be used for development,
   augmentation, ablation, or stress testing. They are never the confirmatory evidence.

## Required admission checks

Every task entering the main matrix must have:

- a direct label definition that can be implemented without an LLM judge;
- human-authored input and a traceable source revision or immutable download hash;
- an explicit split and leakage audit, including exact normalized-text overlap;
- source, author, user, conversation, pair, prompt family, or another defensible grouping unit where
  the source exposes one;
- enough examples in every evaluated class to support the declared metric and uncertainty unit;
- a provenance and licence record that distinguishes internal research from public redistribution;
- a renderer that excludes labels, rationales, moderation scores, dataset names, and annotation
  metadata from model input.

If a source exposes no defensible group identifier, its official test may support a standard
benchmark replication, but not a claim of unseen-user or unseen-source generalization.

## Synthetic-data ceiling

- The confirmatory test is **0% synthetic/model-generated**.
- The headline training and development pool must be at least **90% Tier A or Tier B**. Synthetic
  rows are capped at 10%, tagged by source, and removed in a mandatory sensitivity analysis.
- Synthetic rows may not supply the only positive or only negative class, determine a split, or
  increase a task's weight in the cross-task summary.
- Attack-only human competition submissions remain valid real stress tests, but they cannot produce
  AUROC, AUPRC, or false-positive estimates without an independently justified negative set.

## One protocol per admitted task

Each task runs the same frozen Tier-1 representation family:

- masked mean, last non-padding token, and masked featurewise maximum;
- every hidden-state output plus adjacent windows of width two and three;
- train-fold-only scaling and a matched logistic head;
- word-plus-character TF-IDF and a matched encoder baseline;
- untouched test data after all readout and hyperparameter selection;
- task-appropriate discrimination, calibration, uncertainty, and resource measurements.

Multi-class tasks use macro-F1 and accuracy as primary discrimination metrics. Binary tasks use
AUPRC and AUROC, with balanced accuracy and operating-point metrics. Cross-task analysis compares
ranks and normalized depth; it does not average incompatible raw metrics into one safety score.

## Stop rules

A dataset is excluded from the main matrix when its labels are only model scores, its provenance is
untraceable, synthetic rows dominate, cross-split leakage cannot be repaired, or the label is merely
inferred from the dataset filename. It may remain in the registry as `STRESS_ONLY`, `DEVELOPMENT_ONLY`,
or `REJECTED` so the exclusion is reproducible.
