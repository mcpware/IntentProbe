# Memory Write Gate — Experimental Design

The first experiment predicts `SAVE/SKIP`. It uses the same frozen-backbone extraction contract as
the authorization benchmark so model/task interactions are interpretable.

## Split unit

Primary unit: seed family, not paraphrase row. A split is invalid if any `seed_id`, source template,
or near-duplicate family crosses partitions. Human-gold user-level tests group by author/user.

## Baselines

- majority class;
- explicit memory-keyword rule;
- TF-IDF word+character logistic regression;
- frozen encoder baseline;
- Qwen2.5-0.5B activation probe before any expanded model search.

## Activation matrix

- every hidden layer cached once;
- last token, masked token mean, full-token mean, max pooling;
- individual, adjacent 2–3 layer, and bounded sparse combinations;
- absolute and normalized relative depth;
- training-fold-only feature scaling and selection.

## Metrics

- AUROC, AUPRC, balanced accuracy and macro F1;
- false-save and false-skip rates;
- recall at predeclared false-save rates;
- Brier score, ECE, reliability curves;
- seed-clustered 95% bootstrap CIs;
- CPU latency, peak RSS, and artifact size;
- language, difficulty, category, sensitivity, and explicit-cue slices.

## Cross-task analysis

After both project matrices exist, compare:

- average rank of each backbone across tasks;
- task-by-model interaction with uncertainty;
- overlap of validation-selected normalized-depth intervals;
- regret of one shared backbone/depth window versus task-specific choices.

Do not declare a model “generally stronger” from two point estimates. The matrix must include repeated
group-aware splits or external tests, and the interaction analysis must report uncertainty.
