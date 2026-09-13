# Metric verification record

Run on 2026-09-11 with the retained IntentProbe datasets and embeddings.

## Fixed-config prompt-injection LOSO

- Frozen sensor: Qwen2.5-0.5B.
- Features: mean-pooled hidden states, concatenated layers 13--15.
- Probe: standardized logistic regression, seed 42.
- Baseline: TF--IDF unigram/bigram logistic regression, same rows, seed 42.
- Per-class cap: 350 rows per public source.
- Held-out sizes: deepset 613; SafeGuard 700; SPML 700; jayavibhav 537; total 2,550.
- Uncertainty: 2,000 paired bootstrap resamples per source, bootstrap seed 20260911.

| Held-out source | Probe AUROC | TF--IDF AUROC | Difference | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|
| deepset | 0.9332 | 0.7319 | +0.2013 | [0.1572, 0.2437] |
| SafeGuard | 0.9991 | 0.9928 | +0.0063 | [0.0025, 0.0116] |
| SPML | 0.9897 | 0.9351 | +0.0545 | [0.0390, 0.0726] |
| jayavibhav | 0.9997 | 0.9974 | +0.0023 | [0.0005, 0.0050] |

The intervals are uncorrected for four comparisons. The unweighted point-estimate means are 0.9805 for the probe and 0.9143 for TF--IDF.

## HackAPrompt positive-only stress test

- Uniform-random-offset sample: n = 3,866.
- At 5% training-clean FPR: probe recall 0.903; TF--IDF recall 0.528.
- At 1% training-clean FPR: probe recall 0.883; TF--IDF recall 0.303.
- Approximate Wilson 95% intervals, reconstructed from n and the retained three-decimal recall values:
  - Probe at 5% FPR: [0.8933, 0.9119].
  - TF--IDF at 5% FPR: [0.5122, 0.5436].
  - Probe at 1% FPR: [0.8726, 0.8928].
  - TF--IDF at 1% FPR: [0.2886, 0.3176].

These intervals do not include threshold-calibration uncertainty or dataset shift.

## Tool-poisoning leave-one-corpus-out

Copied from `research/_results_published/toolpoison_crosscorpus.json`:

| Held-out corpus | n | Probe AUROC | TF--IDF AUROC | Difference 95% CI |
|---|---:|---:|---:|---:|
| MCPTox | 692 | 0.738 | 0.545 | [0.145, 0.241] |
| RouteGuard | 700 | 0.640 | 0.582 | [-0.002, 0.120] |
| Synthetic minimal pairs | 186 | 0.494 | 0.498 | [-0.104, 0.089] |

The tool-poisoning attacked rows are constructed, not naturally occurring human attacks.
