# Result summary

Status: complete validation-only readout selection and one-shot official test.

## Data and leakage QA

- Official source: Stanford Sentiment Treebank v1.0 ZIP, SHA-256
  `3f5209483b46bbf129cacbbbe6ae02fe780407034f61cf6342b7833257c3f1db`.
- Binary label rule: negative at score <= 0.4, positive at score > 0.6; 2,242 neutral root
  sentences removed.
- The official binary source contained repeated root strings. Test and validation were preserved;
  11 repeated training rows were removed.
- Final rows: 6,909 train, 872 validation, 1,821 test. Exact normalized-text overlap across splits:
  zero.
- Token audit: maximum 62, p95 41, and zero rows truncated at the frozen 256-token cap.

## Readout selection

The 216-candidate screen used the train split only for fitting and the validation split only for
selection. The winner was:

- pooling: masked mean;
- hidden-state output: 12;
- width: one output;
- normalized depth: 0.50;
- validation AUROC: 0.94935;
- validation AUPRC: 0.95106.

Eighteen of the top 20 candidates used masked mean. The best last-token candidate ranked 13th; the
best masked-max candidate ranked 75th. The final output ranked 71st under last-token pooling, 123rd
under masked mean, and 150th under masked max.

## Untouched official test

| Model | AUROC | AUPRC | Balanced accuracy | Macro-F1 | Brier | ECE-10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen activation, masked-mean output 12 | 0.95880 | 0.95985 | 0.89183 | 0.89182 | 0.08275 | 0.05200 |
| Word+character TF-IDF logistic | 0.90508 | 0.91241 | 0.82758 | 0.82756 | 0.13825 | 0.11489 |

Paired activation-minus-TF-IDF differences from 2,000 test-row bootstrap resamples:

- AUROC: +0.05371, 95% interval [0.04076, 0.06738];
- AUPRC: +0.04744, 95% interval [0.03447, 0.06175].

The intervals exclude zero. On this task and baseline, the selected activation readout discriminates
better and is better calibrated than TF-IDF.

## Interpretation boundary

This is strong evidence for a Qwen2.5-0.5B sentiment signal at an intermediate layer and for masked
mean pooling on this task. It does not yet establish a universal layer, cross-backbone replication,
or superiority over a matched fine-tuned encoder. SST exposes no author or review identifier, so the
uncertainty unit is the test sentence rather than an independent user or source group.
