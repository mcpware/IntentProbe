# MEM-EXP001 — SAVE/SKIP surface baselines

## Frozen question

How much of `memory-write-gate-20k` can be predicted from surface text when seed families, rather
than paraphrase rows, define the independent unit?

## Protocol

- Official train, validation, and test splits are preserved.
- Only `text` enters a model.
- Validation selects the lowest threshold with at most 1% false saves.
- The test split is opened once after `config.json` is frozen.
- Confidence intervals resample the 26 test seed families, not 2,600 paraphrases.

Run with:

`python3 research/memory-write-gate/experiments/EXP001-surface-baselines/run.py`

The result is a silver-source baseline. It cannot establish real-user or human-written
generalization.

## Result

The one-shot test contains 2,600 rows in 26 seed families, balanced as 13 `SAVE` and 13 `SKIP`
families. Word-plus-character TF-IDF reaches AUPRC 0.8241 (seed-clustered 95% CI 0.6424–0.9618) and
AUROC 0.8890 (0.8067–0.9510).

At threshold 0.5, its false-save rate is 18.31% and false-skip rate is 17.46%. The validation-selected
threshold targeting at most 1% false saves transfers to 2.69% false saves on test and retains only
11.38% of `SAVE` rows. The 95% seed-clustered intervals are 0–8.08% false saves and 7.00–16.08% SAVE
recall.

The surface baseline therefore ranks this silver source reasonably well but does not provide a
stable low-false-save operating point. The test is permanently consumed and cannot select the next
activation layer, pooling rule, or threshold.
