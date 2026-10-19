# One forward, many activation heads

This systems experiment freezes 16 independently trained binary readouts on the exact same pinned
Qwen2.5-0.5B backbone. Every head consumes raw text, uses one or more saved hidden outputs, and has
native heldout labels and saved per-row scores.

The latency panel uses 80 real heldout texts: 16 each from SST-2, ToxicChat, SMS Spam, GoEmotions,
and the UCC unhealthy-comments corpus. The cross-domain diagnostic uses 64 texts per source. It is
descriptive outside each head's labelled native source; an out-of-domain score is not called a
false positive without a target label.

The repeated-forward comparator is deliberately favourable to separate judge calls: it measures
only `K` repeated backbone forwards on the same tokenized input, without an LM head, task prompt,
label likelihood calculation, or generation. It is therefore a lower bound, not a full judge
implementation.

## Result

All frozen cells are complete. One Qwen2.5-0.5B forward plus 16 heads takes 73.34 ms per example on
average on the recorded six-core CPU environment, 1.069 times the one-head time (paired-bootstrap
95% CI 1.056--1.078). On the matched ten prompts, the one-forward/16-head path takes 62.70 ms and 16
repeated base-model forwards take 956.71 ms, a ratio of 15.26 (95% CI 15.04--15.49). Peak process RSS changes by
+0.73 MiB from K=1 to K=16, and the 16 saved probe artifacts occupy 386,121 bytes.

This is a systems result, not a claim that every head has equal quality. Native heldout AUPRC ranges
from 0.084 to 0.980 across labels with prevalences from 0.021 to 0.499. Complete discrimination,
calibration, score-shift, live-replay, timing, and uncertainty records are in:

- [`RESULT_SUMMARY.md`](RESULT_SUMMARY.md)
- [`result.json`](result.json)
- [`SCALING-SUMMARY.csv`](SCALING-SUMMARY.csv)
- [`NATIVE-HEAD-QUALITY.csv`](NATIVE-HEAD-QUALITY.csv)
- [`OOD-SCORE-SHIFTS.csv`](OOD-SCORE-SHIFTS.csv)
- [`ASSUMPTION_LEDGER.md`](ASSUMPTION_LEDGER.md)
- [`ATTEMPT_LOG.md`](ATTEMPT_LOG.md)

Row-level artifacts contain IDs, scores, and timings but no source text.
