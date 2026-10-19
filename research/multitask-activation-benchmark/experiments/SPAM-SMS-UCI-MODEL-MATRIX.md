# UCI SMS spam: six-backbone activation-readout matrix

## Contract and integrity checks

All six frozen backbones use the same 5,159 processed examples, the same
3,611/774/774 train/validation/test
split, and the same untouched test rows (96 positive). Example IDs, labels,
and TF-IDF test probabilities are identical across all six score artifacts. Every reported AUPRC
and AUROC below was independently recomputed from the saved test probabilities.

Each backbone selected its readout by validation AUPRC only. Its one-shot test comparison against
the matched word-plus-character TF-IDF logistic baseline was prospectively planned. Cross-backbone
test ranking is descriptive and exploratory: the test results were already observed before this
matrix summary, so it is not used for a new significance claim.

## Result

| Backbone | Validation-selected readout | Depth | Val AUPRC | Test AUPRC | Test AUROC | Δ AUPRC vs TF-IDF (95% CI) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MiniCPM5-1B | masked_mean 10--11 | 0.438 | 0.9983 | 0.9963 | 0.9994 | +0.0267 [+0.0082, +0.0504] |
| Granite-4.0-350M | masked_mean 21--21 | 0.750 | 0.9973 | 0.9937 | 0.9990 | +0.0241 [+0.0064, +0.0471] |
| Qwen3-0.6B | masked_mean 14--14 | 0.500 | 0.9992 | 0.9858 | 0.9971 | +0.0162 [-0.0023, +0.0367] |
| Qwen3.5-0.8B | masked_max 10--11 | 0.438 | 0.9985 | 0.9822 | 0.9957 | +0.0126 [-0.0083, +0.0378] |
| Qwen2.5-0.5B | masked_max 15--17 | 0.667 | 0.9990 | 0.9804 | 0.9959 | +0.0108 [-0.0092, +0.0334] |
| LFM2.5-230M | masked_max 9--9 | 0.643 | 0.9937 | 0.9730 | 0.9936 | +0.0034 [-0.0119, +0.0184] |

The common TF-IDF baseline is 0.9696 AUPRC and 0.9895 AUROC. Two of six planned within-backbone comparisons have activation-minus-TF-IDF 95% paired row-bootstrap intervals above zero on both AUPRC and AUROC: MiniCPM5-1B, Granite-4.0-350M. The other four remain unresolved.

The main engineering observation is heterogeneity, not a universal winning cut. The selected pooling
counts are masked mean 3, last non-padding 0, and masked max 3. Selected normalized depth spans 0.438 to 0.750, while selected windows span 1 to 3 adjacent outputs. MiniCPM5-1B has the highest observed test AUPRC, while Qwen3-0.6B has the highest validation AUPRC. This one task cannot establish a globally best backbone.

## Evidence boundary

This is one binary task and one fixed split. The available metadata does not support a stronger
clustered bootstrap, so uncertainty is row-level. The matrix supports
the study design—task × backbone × layer × pooling—but the paper claim requires the same six-backbone
protocol across the remaining real-data tasks. No cross-model superiority claim is made here.
