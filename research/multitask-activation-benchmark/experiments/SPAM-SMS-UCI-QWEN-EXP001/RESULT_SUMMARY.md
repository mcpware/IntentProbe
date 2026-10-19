# UCI SMS Spam activation benchmark

## Data and design

The official archive contains 5,574 labelled real SMS messages: 4,827 ham and 747 spam. Normalized
exact-text analysis finds 290 duplicate groups containing 705 rows and zero label conflicts. Keeping
one row per exact-text group leaves 5,159 messages, split 70/15/15 with label stratification after
de-duplication. No exact-text group crosses splits.

The frozen Qwen2.5-0.5B experiment compares 216 readouts: masked mean, last non-padding token, and
masked max across every single hidden output and all adjacent width-two and width-three windows.
Validation AUPRC selects the winner. The matched baseline is class-weighted word and character
TF-IDF logistic regression on identical fit rows.

## Result

Masked-max outputs **15--17** win validation. Sixteen of the top 20 candidates use masked max. The
best masked-mean candidate is outputs 14--15 at rank 10, and the best last-token candidate is outputs
19--20 at rank 24. The single final output ranks 27, 50, and 187 under masked mean, last-token, and
masked max respectively.

| 774 untouched test messages / 96 spam | AUPRC | AUROC | Balanced accuracy @ 0.5 | Macro-F1 @ 0.5 |
| --- | ---: | ---: | ---: | ---: |
| Activation, masked-max outputs 15--17 | **0.9804** | **0.9959** | **0.9643** | 0.9643 |
| Word + character TF-IDF | 0.9696 | 0.9895 | 0.9621 | **0.9726** |

Paired activation-minus-TF-IDF differences are +0.0108 AUPRC (95% row-bootstrap CI -0.0092 to
+0.0334) and +0.0064 AUROC (95% CI -0.0025 to +0.0198). The point estimates favour activation, but
the intervals include zero, so the official test does not resolve a ranking winner.

## Sensitivity and limits

Only one of 5,159 messages exceeds the frozen 256-token cap, and it is not in the test split. Test
metrics therefore require no truncation sensitivity adjustment.

The historical merged collection does not retain per-message source-corpus, sender, or author IDs.
Exact-text leakage is controlled, but the row-level bootstrap does not support an unseen-sender or
unseen-source claim. Public release, if any, is a separate project-owner decision; this is an
internal research experiment.
