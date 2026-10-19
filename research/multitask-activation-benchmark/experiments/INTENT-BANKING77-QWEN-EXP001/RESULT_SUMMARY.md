# BANKING77 fine-grained intent activation benchmark

## Data and design

The pinned official files contain 13,083 human banking queries across 77 documented intents. Twelve
normalized exact-text duplicate groups contain 24 rows, seven groups cross the original train/test
boundary, and no duplicate group has a conflicting intent. Preserving official test first and then
de-duplicating leaves 13,071 utterances: 8,493 train, 1,499 validation, and 3,079 official test.

The 216-way screen compares every individual Qwen2.5-0.5B hidden output and adjacent width-two/three
window under masked mean, last non-padding token, and masked max. A deterministic stratified sample
of 3,000 clean training rows and the complete validation split select by macro-F1. Every candidate
uses the same fixed-budget averaged one-vs-rest SGD logistic head: at most 40 epochs. After selection,
the winner is refit on all 9,992 train plus validation utterances and evaluated once on the complete
official test. TF-IDF uses the identical multiclass head and fit rows.

## Result

Masked-mean **output 1** wins validation. Fourteen of the top 20 candidates use masked mean and six
use masked max. The best masked-max candidate is output 1 at rank 5. The best last-token candidate is
the final output at rank 66. The final output ranks 55, 66, and 93 under masked mean, last-token, and
masked max respectively.

| 3,079 official-test queries / 77 intents | Accuracy | Macro-F1 | Top-3 accuracy | Macro OVR AUPRC / AUROC |
| --- | ---: | ---: | ---: | ---: |
| Activation, masked-mean output 1 | 0.6967 | 0.7815 | 0.7246 | 0.7027 / 0.9503 |
| Word + character TF-IDF | **0.9022** | **0.9022** | **0.9734** | **0.9391 / 0.9980** |

Paired activation-minus-TF-IDF differences are -0.2056 accuracy (95% row-bootstrap CI -0.2209 to
-0.1903) and -0.1207 macro-F1 (95% CI -0.1348 to -0.1089). Both intervals exclude zero in favour
of TF-IDF. Activation remains far above 1/77 chance and carries substantial one-vs-rest ranking
signal, but it is not the stronger representation for this fine-grained lexical intent task.

Activation macro-F1 exceeds its accuracy because one intent absorbs 815 predictions while the other
predicted intents tend to have high precision and lower recall: macro precision is 0.9208 and macro
recall is 0.6967. The metric was independently recomputed from saved probabilities.

## Solver and post-test diagnostic

All screened SGD heads reached the frozen 40-epoch budget before the convergence tolerance. The
result is a fixed-compute linear-probe comparison, not a fully converged optimum. Multiclass OVR
probabilities use clipped sigmoid scores followed by row normalization to prevent numerical
underflow; this does not change the decision-score argmax.

After the test was opened, the selected readout trained only on the original 3,000 screening rows was
scored as a diagnostic. It reaches 0.6077 accuracy and 0.7135 macro-F1, below the full-data refit's
0.6967 and 0.7815. Full-data refitting therefore improves rather than causes the test drop. The
diagnostic is not a second confirmatory result and does not separate external-test difficulty,
validation-selection optimism, and fixed-budget solver effects.

The release has no user or source-group IDs, so uncertainty is row-level and cannot support an
unseen-user claim. Public release, if any, is a separate project-owner decision; this is an internal
research experiment.
