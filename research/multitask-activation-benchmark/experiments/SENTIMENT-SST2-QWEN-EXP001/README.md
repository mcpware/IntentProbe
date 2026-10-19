# SENTIMENT-SST2-QWEN-EXP001

First newly admitted task under the topic-open, real-data benchmark policy.

- Data: Stanford Sentiment Treebank root sentences, official split, neutral labels removed.
- Rows: 6,909 leakage-cleaned train; 872 validation; 1,821 untouched test. The official binary
  source had 6,920 train rows; 11 repeated training rows were removed after preserving test and dev.
- Backbone: frozen Qwen2.5-0.5B.
- Search: 216 readouts = three token pooling rules times every single output and adjacent windows of
  width two or three.
- Selection: official validation AUROC, with AUPRC as a co-primary reported measure.
- Baseline: matched word-plus-character TF-IDF logistic regression.
- Test: opened once after readout selection; paired row bootstrap because SST exposes no author or
  review grouping identifier.

This task measures sentiment, not safety. Its role is to test whether the same activation-engineering
conclusions appear on a clean, ordinary semantic label.
