# Qwen2.5 UCC pragmatics panel

This frozen experiment evaluates four human-labelled pragmatic attributes from the Conversation AI
Unhealthy Comments Corpus using Qwen2.5-0.5B activations and a matched word-plus-character TF-IDF
baseline. The official corpus is pinned, exact label-conflict groups are removed, and official-test
rows take precedence during exact de-duplication. No test label selects pooling, layers, or width.

| Task | Selected readout | Test prevalence | Activation AUPRC / AUROC | TF-IDF AUPRC / AUROC | Paired AUPRC conclusion |
| --- | --- | ---: | ---: | ---: | --- |
| Condescending | masked mean 11--13 | 0.0608 | 0.1898 / 0.7648 | 0.1694 / 0.7391 | interval includes zero |
| Dismissive | masked max 13 | 0.0339 | 0.0836 / 0.7448 | 0.1217 / 0.7680 | favours TF-IDF |
| Unfair generalisation | masked max 15--17 | 0.0206 | 0.0995 / 0.8297 | 0.0920 / 0.8176 | interval includes zero |
| Sarcastic | masked mean 4--6 | 0.0454 | 0.1118 / 0.7330 | 0.1032 / 0.6929 | interval includes zero |

All four activation heads have worse Brier score and 10-bin ECE than TF-IDF at the uncalibrated
0.5 output scale. Exact values, 2,000 paired row-bootstrap intervals, winners, and hashes are in
[`result.json`](result.json) and the task-specific result/score files.

The released files omit author and article/thread identifiers. Uncertainty is therefore row-level;
this experiment does not make an unseen-author or unseen-article claim.
