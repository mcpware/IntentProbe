# MEM-EXP004 — train the output-11 head on human labels

MEM-EXP003 measured zero-shot synthetic-to-human transfer. MEM-EXP004 asks the more direct product
question: what happens when the logistic head itself is trained on the human LUFY labels?

The representation remains frozen from the earlier synthetic development experiment: Qwen2.5-0.5B,
masked-mean hidden output 11. Seventeen leave-one-user-out folds generate one out-of-fold prediction
for every one of the 6,470 turns. The matched TF-IDF baseline refits its vocabulary and classifier
inside every fold.

| 6,470 rows / 17 held-out-user folds | AUPRC | AUROC | Balanced accuracy @ 0.5 |
| --- | ---: | ---: | ---: |
| Human-trained activation head | 0.1180 | 0.7032 | 0.6160 |
| Human-trained word + character TF-IDF | 0.1544 | 0.7808 | 0.6273 |

Training on human labels materially improves the fixed activation readout over the zero-shot
MEM-EXP003 result. TF-IDF still wins this fixed-output comparison: paired user-bootstrap differences
for activation minus TF-IDF are −0.0364 AUPRC (95% CI −0.0680 to −0.0076) and −0.0777 AUROC (95% CI
−0.0993 to −0.0578).

This experiment motivates MEM-EXP005: output 11 was selected on synthetic data and should not be
assumed optimal for the human task.

The final all-human head remains local because the historical annotation revision is still in
licence quarantine. `result.json` contains the publishable aggregate receipt; features, per-row
scores, and the numeric head are ignored.
