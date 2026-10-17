# MEM-EXP005 — human-trained layer and pooling selection

## Design

This experiment trains directly on LUFY's triple-annotated human turns and asks which Qwen2.5-0.5B
readout is most useful.

- 17 users total;
- five users frozen as the final test using anonymized user hashes, without using their labels;
- 12 development users for all representation selection;
- 216 candidates: three token-pooling rules × every individual output × adjacent width-2/3 windows;
- four-fold GroupKFold by user for development selection;
- one final 1,748-row / five-user test after selecting the winner by development AUPRC;
- matched TF-IDF trained on the same 12 development users;
- 2,000 held-out-user bootstrap resamples for uncertainty.

Every candidate standardizes each layer using only the training users in that fold, then averages
the standardized outputs inside an adjacent window.

## Development result

Masked-mean outputs **7--8** win with mean grouped-development AUPRC 0.1824. The strongest single
output is masked-mean output 7 (rank 7). The previously used output 11 ranks 11. The best last-token
candidate ranks 34 and the best masked-max candidate ranks 40; all top 20 candidates use masked mean.

## One-shot human test

| 1,748 turns / 5 held-out users / 91 SAVE labels | AUPRC | AUROC | Balanced accuracy @ 0.5 |
| --- | ---: | ---: | ---: |
| Human-trained activation, outputs 7--8 | 0.1575 | 0.7842 | **0.6603** |
| Human-trained word + character TF-IDF | **0.1726** | **0.7892** | 0.6250 |

Paired activation-minus-TF-IDF differences are −0.0151 AUPRC (95% user-bootstrap CI −0.0802 to
+0.0016) and −0.0050 AUROC (95% CI −0.0416 to +0.0309). The test does not distinguish their ranking
performance; activation has the higher fixed-threshold balanced accuracy.

## Readout-engineering diagnostic

After the primary test was opened, the pre-existing output-11 recipe was run on the same development
and test split. Outputs 7--8 improve over output 11 by +0.0607 AUPRC (95% CI +0.0302 to +0.1200) and
+0.1088 AUROC (95% CI +0.0675 to +0.1483). This is a labelled post-test diagnostic, not a second
confirmatory result. It quantifies why layer and pooling selection are central to the multi-task
paper.

## Release boundary

Aggregate results and reproducible code are suitable for the research record. The cached features,
per-row scores, reconstructed data, and human-trained numeric probe remain local until release rights
for the deleted historical LUFY annotation files are confirmed.
