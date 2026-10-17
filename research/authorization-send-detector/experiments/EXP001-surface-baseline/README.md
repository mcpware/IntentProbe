# EXP001 — Surface Baseline

Hypothesis: surface text explains a substantial portion of the public binary authorization labels.

This is a full-data word unigram/bigram TF-IDF logistic classifier trained on the pinned
`auto-1b-data` train file. Hyperparameter selection uses only the pinned validation file. The
separate `approve-or-deny` test file is transformed and scored only after the vectorizer, alpha, and
operating threshold have been selected.

The operational point maximizes validation true-positive rate subject to false-approve rate at or
below 1%. Because these public sources do not expose a scenario-group key, its uncertainty intervals
are row-bootstrap/Wilson intervals and do **not** support a scenario-independent product claim.

Run:

```bash
python3 research/authorization-send-detector/experiments/EXP001-surface-baseline/run.py
```

`metrics.json` and `environment.txt` are generated only after successful completion.

## Result

Completed on 2026-10-16 (America/Vancouver) using 712,000 train rows, 13,000 validation rows,
and 3,000 held-out test rows. Normalized 64-bit fingerprints found zero cross-split exact-text
overlap.

| Metric | Held-out result |
|---|---:|
| AUROC | 0.9171 (row-bootstrap 95% CI 0.9067–0.9264) |
| AUPRC | 0.9336 (row-bootstrap 95% CI 0.9243–0.9424) |
| Balanced accuracy at 0.5 | 0.8314 |
| False-approve rate at 0.5 | 9.71% (Wilson 95% CI 8.27%–11.37%) |
| False-deny rate at 0.5 | 24.02% (Wilson 95% CI 21.99%–26.17%) |
| False-approve rate at validation-selected threshold | 0.86% (Wilson 95% CI 0.49%–1.49%) |
| False-deny rate at validation-selected threshold | 52.03% (Wilson 95% CI 49.58%–54.47%) |

The conservative operating point transferred below the declared 1% false-approve target, but rejected
more than half of the approved test cases. That trade-off is now the concrete floor for EXP002: an
activation detector must improve recall at a matched false-approve rate, not merely post a high AUROC.

Peak RSS was 15,496,276 KiB and elapsed time was 396.5 seconds on the current CPU host. These public
files do not expose scenario-group IDs, so the intervals above treat rows as independent and must not
be presented as product-grade generalization evidence.
