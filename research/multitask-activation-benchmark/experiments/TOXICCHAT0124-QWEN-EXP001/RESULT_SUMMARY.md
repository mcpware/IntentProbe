# ToxicChat 0124 human-only activation benchmark

## Data and frozen design

The pinned ToxicChat 0124 files contain 10,165 prompts. Of those, 5,654 have
`human_annotation=true`. Exact normalized-text de-duplication preserves the official test and leaves
5,441 unique human-labelled prompts: 2,120 train, 531 validation, and 2,790 official test.

The two labels share one Qwen2.5-0.5B activation extraction but run independent readout searches.
Each search compares 216 candidates: three pooling rules across every individual hidden output and
all adjacent width-two and width-three windows. Validation AUPRC selects the winner. The official
test is opened only after selection. The matched surface baseline is class-weighted word and
character TF-IDF logistic regression on identical fit rows.

The label-blind token audit selected a 512-token cap. Nine of 5,441 prompts exceed the cap; five are
in the official test. Removing those five test prompts produces effectively unchanged results.

## Results

| Human-labelled task | Selected activation readout | Test positives / rows | Activation AUPRC / AUROC | TF-IDF AUPRC / AUROC |
| --- | --- | ---: | ---: | ---: |
| Toxicity | masked mean, outputs 18--20 | 354 / 2,790 | 0.7093 / **0.9091** | **0.7262** / 0.8998 |
| Jailbreaking | masked max, outputs 5--7 | 89 / 2,790 | 0.6650 / 0.9505 | **0.8264** / **0.9817** |

For toxicity, paired activation-minus-TF-IDF differences are -0.0169 AUPRC (95% row-bootstrap CI
-0.0570 to +0.0236) and +0.0093 AUROC (95% CI -0.0091 to +0.0291). The test does not resolve a
ranking winner. Activation has higher balanced accuracy at threshold 0.5 (0.8283 versus 0.8088),
while TF-IDF has slightly higher macro-F1 (0.7914 versus 0.7838).

For jailbreaking, paired activation-minus-TF-IDF differences are -0.1614 AUPRC (95% CI -0.2310 to
-0.0945) and -0.0312 AUROC (95% CI -0.0475 to -0.0170). Both intervals exclude zero in favour of
TF-IDF. Activation still carries substantial signal relative to the 0.0319 positive prevalence, but
it does not beat the matched surface baseline on this task.

## Readout finding

The winner changes materially by label even though the prompts, split, backbone, and candidate
budget are identical:

- toxicity: late masked-mean outputs 18--20; all top 20 candidates use masked mean;
- jailbreaking: early masked-max outputs 5--7; the top 20 contain 11 masked-max and nine masked-mean
  candidates;
- the final single output is not automatically best: its ranks are 55/74/142 across toxicity's
  masked-max/masked-mean/last-token candidates and 25/46/138 for jailbreaking.

Together with sentiment's masked-mean output 12 and memory write's masked-mean outputs 7--8, this is
direct evidence against a universal final-layer or universal-pooling recipe. Readout selection is a
task-specific engineering variable.

## Sensitivity and limits

Excluding the five official-test prompts longer than 512 tokens changes toxicity activation AUPRC
from 0.709289 to 0.709552 and jailbreak activation AUPRC from 0.664997 to 0.665013. Truncation does
not explain the headline results.

Exact-text groups do not cross splits. ToxicChat does not expose a stable user or source-cluster ID,
so uncertainty is a paired prompt-level bootstrap and does not support an unseen-user claim. Public
release, if any, is a separate project-owner decision; this experiment is an internal research run.
