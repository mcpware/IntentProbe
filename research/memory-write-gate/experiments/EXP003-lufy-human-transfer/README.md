# MEM-EXP003 — frozen transfer to human-written LUFY turns

## Question

Does the Qwen2.5-0.5B representation and logistic probe selected entirely on the synthetic
`memory-write-gate-20k` development source transfer to human-written conversational turns labelled
for future usefulness?

No LUFY row selected the renderer, layer, pooling rule, classifier weight, or threshold.

## Data

- 6,470 user turns from 17 users and three memory systems;
- three independent binary `important?` annotations per turn;
- 332 majority-vote positives and 6,138 negatives;
- 5,070 unanimous rows, including 52 unanimous positives;
- main label: majority vote;
- uncertainty unit: user, not individual turn.

The historical annotation files are used from the official LUFY repository at commit
`e0695216d52a38ef39ec2f91708d324877e328bd`. They remain in local quarantine because the licence
scope of the deleted historical files has not been confirmed. Raw text and per-row labels are not
redistributed by this experiment.

## Frozen comparison

| 6,470 majority-vote rows | AUPRC | AUROC |
| --- | ---: | ---: |
| Qwen activation probe | 0.0737 | 0.5811 |
| Word + character TF-IDF | 0.0666 | 0.5599 |
| Positive prevalence | 0.0513 | — |

Paired activation-minus-TF-IDF differences under 2,000 user-cluster bootstrap resamples:

- AUPRC: +0.0071, 95% CI −0.0058 to +0.0198;
- AUROC: +0.0212, 95% CI −0.0226 to +0.0644.

The activation point estimates are higher, but neither paired interval excludes zero. On the
unanimous-only view, activation reaches 0.0214 AUPRC and 0.6278 AUROC across 5,070 rows with only 52
positives; its uncertainty is correspondingly wide.

## Interpretation

The frozen activation probe retains a measurable ranking signal after transfer from synthetic
utterances to human-written dialogue, but the domain shift is large and superiority over the matched
surface baseline is unresolved. This result supports keeping Memory Write Gate as a research preview
and as one column of the multi-task activation benchmark. It does not support treating the score as
storage consent or as a deployment-ready automatic write decision.

## Reproduction

1. `build_dataset.py` reconstructs the quarantined majority-vote table and asserts row counts.
2. `score_activation.py` applies the frozen MEM-EXP002 artifact.
3. `analyze.py` refits the frozen TF-IDF recipe on the original synthetic training split and computes
   point estimates plus vectorized user-cluster bootstrap intervals.

Public aggregate receipt: `result.json`. Per-row scores and reconstructed LUFY data are ignored and
remain local.
