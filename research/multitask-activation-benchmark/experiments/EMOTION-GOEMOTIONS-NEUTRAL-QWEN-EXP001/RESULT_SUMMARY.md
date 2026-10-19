# GoEmotions neutral-label activation benchmark

## Target and data

This task predicts whether the human-filtered GoEmotions label set contains `neutral`. It does not
relabel mixed examples as exclusively neutral: 1,751 source rows carry `neutral` together with at
least one other emotion label. The target is therefore **neutral label present**, not a claim that a
comment expresses no other emotion.

The pinned official filtered files contain 54,263 Reddit comments. An exact-text audit finds 82
cross-split duplicate groups and 38 normalized-text groups whose neutral label conflicts. Removing
all 116 rows in those conflicting groups, then preserving test before validation before train for
remaining exact duplicates, leaves 53,905 comments: 43,095 train, 5,401 validation, and 5,409 test.

The 216-way readout screen uses a deterministic label-stratified 5,000-row sample from official
train and the complete validation split. This fixed screening budget was frozen after a label-blind
compute pilot and before the test was opened. After selection, the winner is refit on all 48,496
train plus validation comments, then evaluated once on the full test.

## Result

Masked-mean outputs **15--16** win validation. All top 20 candidates use masked mean. The best
masked-max candidate is outputs 1--2 at rank 61, while the best last-token candidate is outputs
15--16 at rank 102. The final single output ranks 69, 154, and 168 under masked mean, last-token,
and masked max respectively.

| 5,409 untouched test comments / 1,778 neutral-positive | AUPRC | AUROC | Balanced accuracy @ 0.5 | Macro-F1 @ 0.5 |
| --- | ---: | ---: | ---: | ---: |
| Activation, masked-mean outputs 15--16 | **0.6675** | 0.8175 | 0.7379 | 0.7128 |
| Word + character TF-IDF | 0.6578 | **0.8228** | **0.7523** | **0.7293** |

Paired activation-minus-TF-IDF differences are +0.0098 AUPRC (95% row-bootstrap CI -0.0115 to
+0.0303) and -0.0054 AUROC (95% CI -0.0151 to +0.0046). Both intervals include zero, so this test
does not resolve a ranking winner.

## Sensitivity and limits

Three of 53,905 comments exceed the frozen 256-token cap, but none is in the official test. Test
metrics therefore require no truncation sensitivity adjustment.

The filtered files preserve a unique Reddit comment ID but omit author and thread IDs. Exact-text
leakage is controlled; uncertainty remains a paired comment-level bootstrap and cannot support an
unseen-user claim. The 5,000-row screening budget needs a later seed/budget stability analysis before
the cross-task paper treats exact winner rank as fixed. Public release, if any, is a separate
project-owner decision; this is an internal research experiment.
