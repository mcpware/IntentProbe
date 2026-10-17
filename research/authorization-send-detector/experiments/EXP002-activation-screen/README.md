# EXP002 — Frozen Activation Screen and One-Shot Held-Out Test

Question: on the same authorization rows, does a linear probe over frozen hidden states provide more
useful signal than a surface-text classifier?

The selection screen used only the official `auto-1b-data` train and validation files. Six pinned
small backbones completed the same rendered runtime fields, 192-token cap, two pooling rules, fixed
logistic head, and complete individual/adjacent-width-2/adjacent-width-3 layer family. Falcon-H1 was
attempted but was CPU-infeasible without its optional fast kernels: one 16-row batch did not finish in
151 seconds and used about 10.8 GiB RSS. The two `auto-*` encoders are publisher-trained supervised
baselines, not evidence that a generic frozen activation representation works.

## Screen result

| Frozen backbone | Best pooled outputs | Validation AUPRC | Recall at <=5% false approve | Rows/s |
|---|---:|---:|---:|---:|
| MiniCPM5-1B-Base | masked mean, 13–14 | 0.9228 | 0.6441 | 2.48 |
| Qwen3.5-0.8B | masked mean, 14–16 | 0.9205 | 0.6140 | 1.96 |
| Qwen2.5-0.5B | masked mean, 16–18 | 0.9136 | 0.6416 | 4.25 |
| LFM2.5-230M | masked mean, 10–12 | 0.9090 | 0.5689 | 8.68 |
| Granite-4.0-350M | masked mean, 19 | 0.9081 | 0.6090 | 5.33 |
| Qwen3-0.6B-Base | masked mean, 14 | 0.9045 | 0.5965 | 3.04 |

The matched 1,600-row TF-IDF baseline reached 0.8866 AUPRC. The publisher-trained `auto-0.4b` and
`auto-1b` encoders reached 0.9567 and 0.9914 respectively, but their training budgets are not matched
to the 1,600-row probe.

The predeclared highest-AUPRC rule froze MiniCPM5, masked-mean pooling, and hidden outputs 13–14 at
`2026-10-17T04:44:25Z`, before the held-out file was read. The freeze record is
`FROZEN_SELECTION.json`.

## Held-out result

The one-shot 3,000-row `approve-or-deny` test has zero exact rendered overlap with the screen train or
validation rows. It is now consumed and must never again be used to select a model, pooling rule,
layer, regularization value, calibration method, or threshold.

| Same 1,600 training rows | Held-out AUPRC | Held-out AUROC | Oracle recall at <=5% false approve |
|---|---:|---:|---:|
| Frozen MiniCPM activation probe | 0.8223 | 0.8137 | 0.3058 |
| Word/bigram TF-IDF logistic | 0.8079 | 0.7897 | 0.3189 |

The paired AUPRC difference was +0.0144 with row-bootstrap 95% CI `[-0.0053, +0.0336]`
(`p=0.144`), so this test does not establish an AUPRC gain. The paired AUROC difference was +0.0240,
95% CI `[+0.0056, +0.0429]` (`p=0.015`), supporting a better overall ranking signal. At the
safety-critical low-false-approve end of the curve, TF-IDF had slightly higher recall.

Calibration did not transfer. The activation threshold chosen for <=5% validation false approves
produced 9.49% on held-out; its <=1% threshold produced 5.07%. The matched TF-IDF thresholds produced
2.50% and 0.64%. The frozen activation artifact therefore fails the current standalone send-gate
criterion. It may be studied as an attached risk sensor, but deterministic authorization state and
payload binding remain mandatory.

## Limits

- Both train and held-out labels are silver and come from the same publisher family.
- Public rows lack scenario-group IDs, so uncertainty is row-level rather than clustered by scenario.
- The screen selected one maximum among many layer candidates; its validation score is optimistic.
- Max pooling, sparse/non-adjacent layer combinations, output-logit controls, group holdouts, and
  multiple seeds remain undone.
- A larger development/calibration stage should have preceded the held-out test. Because it did not,
  future calibration or layer-mixing work requires a new untouched external test.
