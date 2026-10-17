# MEM-EXP002 — Qwen2.5-0.5B activation screen

This experiment searches every Qwen2.5-0.5B hidden output using masked-mean, last-nonpadding, and
masked-max pooling, plus adjacent width-2 and width-3 windows. Selection uses the 32 validation seed
families only.

The 256-token cap was frozen after a tokenizer-only audit: raw development utterances have maxima of
196 train tokens and 171 validation tokens. The renderer adds a short fixed template and must receive
a second post-render coverage assertion before feature extraction is accepted.

The official test was consumed by MEM-EXP001. It cannot select any MEM-EXP002 representation,
classifier, or threshold.

## Completed development result

The full frozen search completed on 2026-10-17: 3 pooling rules × 72 layer/window candidates = 216
candidates. The selected representation is masked mean over all non-padding rendered tokens at
hidden-state output index 11 of 24 (normalized depth 0.4583), followed by the preregistered
class-weighted logistic head.

| Pooling | Best candidate | Validation AUPRC | Validation AUROC |
|---|---:|---:|---:|
| masked mean | output 11 | 0.8897 | 0.9030 |
| last non-padding | output 22 | 0.8803 | 0.8918 |
| masked max | outputs 11–13 mean | 0.8656 | 0.8739 |

The selected activation probe exceeds the frozen word-plus-character TF-IDF validation baseline by
0.0531 AUPRC and 0.0419 AUROC. Paired seed-family bootstrap intervals are −0.0199 to 0.1469 for the
AUPRC difference and −0.0194 to 0.0998 for the AUROC difference. The direction is promising, but the
32-seed development comparison is not statistically resolved.

At the validation-selected 1% false-save operating point, SAVE recall is 22.56%. The 95% seed-clustered
intervals are 0.07%–2.57% for false-save rate and 12.44%–34.72% for SAVE recall. At the ordinary 0.5
threshold, false-save rate is 19.50% and SAVE recall is 83.69%.

The exploratory category audit exposes the next data target. At threshold 0.5, `general_advice`,
`one_off_document`, and `secret` have the highest false-save rates (88%, 64%, and 52%).
`project_constraint` has the lowest SAVE recall (46%). These categories are single-label generator
families, so the report uses error rates rather than invalid within-category AUPRC values.

## Reproducible outputs

- `pooling_results/*.json`: all 216 candidate results, not only the winners.
- `candidate_table.csv`: flat paper-ready table of the complete candidate screen.
- `selection_result.json`: selection, clustered uncertainty, paired baseline comparison, and slices.
- `selected_probe.npz`: frozen linear head, train-only normalization, selected output, and threshold.
- `validation_scores.npz`: row-aligned scores used for the paired analysis.
- `select_candidate.py`: deterministic selection and reporting pipeline.

All reported intervals resample `seed_id`, not individual paraphrases. Because the same validation
split selected the winner and measured it, these are explicitly post-selection development results.
No MEM-EXP002 code loaded the previously consumed official test.
