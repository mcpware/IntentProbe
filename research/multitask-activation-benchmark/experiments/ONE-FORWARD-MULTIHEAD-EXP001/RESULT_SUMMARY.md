# One-forward multi-head result

Status: **COMPLETE**. All five frozen K cells, the 16-head score-shift panel, and all 16 native heldout quality/calibration records passed alignment checks.

## Main result

Across the 80-row shared-path panel, one Qwen2.5-0.5B forward plus 16 independently trained activation heads took 73.34 ms per example on average. That was 1.069× the one-head time (paired-bootstrap 95% CI 1.056–1.078). In the separate matched ten-prompt comparator, the one-forward/16-head mean was 62.70 ms and 16 repeated base-model forwards took 956.71 ms. The repeated/shared ratio was 15.26 (95% CI 15.04–15.49). This repeated-forward comparator omits task prompts, LM-head scoring, and generation, so it is a lower bound for 16 separate LLM-judge calls.

Peak process RSS was 3617.4 MiB at K=16, a +0.7 MiB change from K=1. The saved probe artifacts total 386,121 bytes at K=16.

### Shared-path scaling on all 80 inputs

| k | mean_total_ms | mean_heads_ms | one_forward_mean_ratio_vs_k1 | peak_rss_mib |
| --- | --- | --- | --- | --- |
| 1.000 | 68.632 | 0.383 | 1.000 | 3616.699 |
| 2.000 | 69.857 | 0.609 | 1.018 | 3615.488 |
| 4.000 | 70.729 | 0.729 | 1.031 | 3616.137 |
| 8.000 | 71.780 | 1.078 | 1.046 | 3614.008 |
| 16.000 | 73.337 | 1.853 | 1.069 | 3617.430 |

### Both paths on the same ten prompts

| k | matched_shared_mean_ms | repeated_forward_mean_ms | repeated_over_one_forward_ratio |
| --- | --- | --- | --- |
| 1.000 | 61.834 | 57.884 | 0.936 |
| 2.000 | 60.299 | 120.147 | 1.993 |
| 4.000 | 60.487 | 238.731 | 3.947 |
| 8.000 | 61.710 | 475.643 | 7.708 |
| 16.000 | 62.698 | 956.706 | 15.259 |

`mean_total_ms` uses all 80 shared-path rows. `matched_shared_mean_ms`, `repeated_forward_mean_ms`, and `repeated_over_one_forward_ratio` use the matched ten-prompt comparator. The 80-row and 10-row means belong to different panels and should not be divided to reconstruct a ratio.

## Quality is task-specific

The 16 heads retain their own native heldout labels. AUPRC ranges from 0.084 (pragmatics_dismissive, prevalence 0.034) to 0.980 (spam, prevalence 0.124). The systems result therefore does not imply that every head is equally useful; it shows that many validated readouts can share one backbone pass.

| head | native_source | pooling | start_layer | end_layer | rows | positive_prevalence | auprc | auroc | balanced_accuracy | brier | ece_10_bin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toxicity | toxicchat | masked_mean | 18.000 | 20.000 | 2790.000 | 0.127 | 0.709 | 0.909 | 0.828 | 0.091 | 0.087 |
| jailbreaking | toxicchat | masked_max | 5.000 | 7.000 | 2790.000 | 0.032 | 0.665 | 0.950 | 0.811 | 0.029 | 0.030 |
| spam | sms_spam | masked_max | 15.000 | 17.000 | 774.000 | 0.124 | 0.980 | 0.996 | 0.964 | 0.012 | 0.012 |
| sentiment_positive | sst2 | masked_mean | 12.000 | 12.000 | 1821.000 | 0.499 | 0.960 | 0.959 | 0.892 | 0.083 | 0.052 |
| emotion_neutral | goemotions | masked_mean | 15.000 | 16.000 | 5409.000 | 0.329 | 0.668 | 0.817 | 0.738 | 0.177 | 0.117 |
| emotion_anger | goemotions | masked_mean | 5.000 | 5.000 | 5409.000 | 0.037 | 0.323 | 0.879 | 0.782 | 0.102 | 0.160 |
| emotion_sadness | goemotions | masked_mean | 16.000 | 16.000 | 5409.000 | 0.029 | 0.337 | 0.911 | 0.830 | 0.089 | 0.136 |
| emotion_admiration | goemotions | masked_mean | 2.000 | 4.000 | 5409.000 | 0.093 | 0.601 | 0.917 | 0.851 | 0.105 | 0.158 |
| emotion_curiosity | goemotions | masked_mean | 3.000 | 3.000 | 5409.000 | 0.052 | 0.414 | 0.928 | 0.857 | 0.081 | 0.121 |
| pragmatics_condescending | ucc | masked_mean | 11.000 | 13.000 | 4424.000 | 0.061 | 0.190 | 0.765 | 0.696 | 0.175 | 0.269 |
| pragmatics_dismissive | ucc | masked_max | 13.000 | 13.000 | 4424.000 | 0.034 | 0.084 | 0.745 | 0.677 | 0.169 | 0.268 |
| pragmatics_unfair_generalisation | ucc | masked_max | 15.000 | 17.000 | 4424.000 | 0.021 | 0.100 | 0.830 | 0.717 | 0.126 | 0.185 |
| pragmatics_sarcastic | ucc | masked_mean | 4.000 | 6.000 | 4424.000 | 0.045 | 0.112 | 0.733 | 0.664 | 0.187 | 0.299 |
| pragmatics_antagonize | ucc | masked_mean | 15.000 | 15.000 | 4424.000 | 0.046 | 0.172 | 0.787 | 0.727 | 0.159 | 0.251 |
| pragmatics_generalisation | ucc | masked_max | 17.000 | 17.000 | 4424.000 | 0.022 | 0.101 | 0.812 | 0.720 | 0.130 | 0.196 |
| pragmatics_hostile | ucc | masked_mean | 16.000 | 17.000 | 4424.000 | 0.024 | 0.094 | 0.783 | 0.688 | 0.134 | 0.199 |

## Cross-domain diagnostic

The 320-row panel is deliberately unlabelled outside each head's native task. It measures score movement, not cross-domain error. The largest absolute mean shift was +0.301 for `emotion_curiosity` on `toxicchat` relative to that head's full native heldout score distribution (KS distance 0.406). Treat these shifts as monitoring/calibration evidence, not accuracy claims.

Runtime replay against the saved float16-cache evaluation scores had a worst mean absolute difference of 0.006451, a worst maximum absolute difference of 0.338490, and a minimum Pearson correlation of 0.979861 across the 16 heads. This quantifies the sensitivity introduced by using live full-precision hidden states rather than cached float16 activations.

## Interpretation boundary

- Supported: one base-model pass can serve 16 real, independently trained classification heads with nearly flat latency and memory in this fixed CPU environment.
- Supported: direct same-weight judging is not uniformly equivalent to activation readouts; the separate six-cell sentinel found strong task/model interactions.
- Not supported: universal probe superiority, production calibration on every deployment domain, or accuracy for the unlabelled cross-domain panel.
- Comparator boundary: repeated base forwards are a conservative systems lower bound, not a full implementation of 16 prompted LLM judges.

Machine-readable detail is in `result.json`; row-level artifacts contain IDs, timings, and scores but no source text.
