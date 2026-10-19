# SST-2 binary sentiment: six-backbone activation-readout matrix

## Contract and integrity checks

All six frozen backbones use the same 9,602 processed examples, the same
6,909/872/1,821 train/validation/test
split, and the same untouched test rows (909 positive). Example IDs, labels,
and TF-IDF test probabilities are identical across all six score artifacts. Every reported AUPRC
and AUROC below was independently recomputed from the saved test probabilities.

Each backbone selected its readout by validation AUROC only. Its one-shot test comparison against
the matched word-plus-character TF-IDF logistic baseline was prospectively planned. Cross-backbone
test ranking is descriptive and exploratory: the test results were already observed before this
matrix summary, so it is not used for a new significance claim.

## Result

| Backbone | Validation-selected readout | Depth | Val AUROC | Test AUPRC | Test AUROC | Δ AUPRC vs TF-IDF (95% CI) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Granite-4.0-350M | masked_mean 15--15 | 0.536 | 0.9447 | 0.9669 | 0.9658 | +0.0545 [+0.0428, +0.0678] |
| Qwen2.5-0.5B | masked_mean 12--12 | 0.500 | 0.9493 | 0.9599 | 0.9588 | +0.0474 [+0.0345, +0.0617] |
| Qwen3-0.6B | masked_mean 14--15 | 0.518 | 0.9472 | 0.9544 | 0.9558 | +0.0420 [+0.0288, +0.0574] |
| MiniCPM5-1B | masked_mean 11--13 | 0.500 | 0.9543 | 0.9532 | 0.9531 | +0.0407 [+0.0278, +0.0554] |
| Qwen3.5-0.8B | last_nonpadding 11--13 | 0.500 | 0.9489 | 0.9421 | 0.9446 | +0.0297 [+0.0118, +0.0469] |
| LFM2.5-230M | masked_mean 8--9 | 0.607 | 0.9292 | 0.9329 | 0.9291 | +0.0205 [+0.0079, +0.0345] |

The common TF-IDF baseline is 0.9124 AUPRC and 0.9051 AUROC. All six planned within-backbone comparisons have activation-minus-TF-IDF 95% paired row-bootstrap intervals above zero on both AUPRC and AUROC: Granite-4.0-350M, Qwen2.5-0.5B, Qwen3-0.6B, MiniCPM5-1B, Qwen3.5-0.8B, LFM2.5-230M.

The main engineering observation is heterogeneity, not a universal winning cut. The selected pooling
counts are masked mean 5, last non-padding 1, and masked max 0. Selected normalized depth spans 0.500 to 0.607, while selected windows span 1 to 3 adjacent outputs. Granite-4.0-350M has the highest observed test AUROC, while MiniCPM5-1B has the highest validation AUROC. This one task cannot establish a globally best backbone.

## Evidence boundary

This is one binary task and one fixed split. The available metadata does not support a stronger
clustered bootstrap, so uncertainty is row-level. The matrix supports
the study design—task × backbone × layer × pooling—but the paper claim requires the same six-backbone
protocol across the remaining real-data tasks. No cross-model superiority claim is made here.
