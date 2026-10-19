# UCC pragmatics expansion

This frozen expansion adds three independent binary readouts—antagonize, generalisation, and
hostile—before the one-forward `K=16` systems benchmark. It reuses the immutable, label-blind
Qwen2.5-0.5B activation cache from `PRAGMATICS-UCC-QWEN-EXP001`; no model forward is repeated and
no EXP001 artifact is overwritten.

Each target screens the same 216 pooling/layer/window candidates on a deterministic 5,000-row
stratified training sample and the complete validation split. The selected readout is refitted on
train plus validation exactly once, then evaluated on the untouched official test comments.

| Task | Selected readout | Test prevalence | Activation AUPRC / AUROC | TF-IDF AUPRC / AUROC | Paired AUPRC conclusion |
| --- | --- | ---: | ---: | ---: | --- |
| Antagonize | masked mean 15 | 0.0459 | 0.1725 / 0.7873 | 0.1801 / 0.7824 | interval includes zero |
| Generalisation | masked max 17 | 0.0217 | 0.1011 / 0.8120 | 0.1005 / 0.8177 | interval includes zero |
| Hostile | masked mean 16--17 | 0.0244 | 0.0939 / 0.7831 | 0.0979 / 0.7833 | interval includes zero |

All three activation heads have worse Brier score and 10-bin ECE than the matched TF-IDF baseline.
Exact values and 2,000 paired row-bootstrap intervals are in [`result.json`](result.json).
The interrupted pre-result attempt and deterministic resume are recorded in
[`ATTEMPT_LOG.md`](ATTEMPT_LOG.md).
