# SST-2 six-backbone execution log

The Qwen2.5-0.5B anchor experiment was completed before the shared model-matrix runner. On
2026-10-17/18, the same leakage-clean 6,909/872/1,821 train/validation/official-test contract was
adapted to the other five frozen backbones. Validation AUROC remained the selection metric, and the
unweighted TF-IDF logistic baseline remained unchanged from the anchor experiment.

## Completed runs

| Backbone | Candidate count | Extraction status | Screen/test status |
| --- | ---: | --- | --- |
| LFM2.5-230M | 126 | complete; resumed after batch-size tuning | complete |
| Granite-4.0-350M | 252 | complete | complete |
| Qwen3-0.6B | 252 | complete | complete |
| Qwen3.5-0.8B | 216 | complete; resumed after batch-size tuning | complete |
| MiniCPM5-1B | 216 | complete | complete |

LFM extraction began at batch size 16, resumed at 64, and finished at 128. Qwen3.5 extraction began
at 32, briefly tested 64, then resumed and finished at 32 because the local torch fallback was faster
at the smaller batch. These were compute-only changes: cached rows were retained, and model revision,
token cap, pooling, labels, splits, selection metric, and statistical protocol did not change.

Qwen3.5 reported that optional fast-path libraries were absent and used the torch implementation.
No package was installed. MiniCPM reported an unexpected `lm_head.weight` while loading the base
transformer body with `AutoModel`; the hidden-state count and width matched the frozen manifest, and
extraction completed with finite arrays.

Because the final extraction manifests record only their last resumed segment, their elapsed-time
fields are not directly comparable for LFM and Qwen3.5. A clean timing rerun is required before any
cross-backbone latency claim.
