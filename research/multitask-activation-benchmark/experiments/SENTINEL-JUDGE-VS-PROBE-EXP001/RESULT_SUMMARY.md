# Sentinel judge-versus-probe result summary

Status: **COMPLETE** — 6/6 six-cell panels have all four judge arms.

Point estimates and 2,000 paired row-bootstrap intervals are computed from saved per-example scores. Generated-label arms use strict YES/NO parsing; their hard scores are shown without rank-metric bootstrap intervals.

| model          | task                     | method                         |   n |   invalid_outputs |   auprc |   auroc |   balanced_accuracy |   brier | seconds_per_example_batch   |
|:---------------|:-------------------------|:-------------------------------|----:|------------------:|--------:|--------:|--------------------:|--------:|:----------------------------|
| cost_anchor    | authorization_to_execute | activation_head                | 156 |                 0 |  0.5529 |  0.5187 |              0.5217 |  0.4146 |                             |
| cost_anchor    | authorization_to_execute | tfidf_word_char                | 156 |                 0 |  0.4533 |  0.5276 |              0.469  |  0.2494 |                             |
| cost_anchor    | authorization_to_execute | judge_zero_shot_choice_logprob | 156 |                 0 |  0.4423 |  0.4736 |              0.4928 |  0.4115 | 0.5899                      |
| cost_anchor    | authorization_to_execute | judge_zero_shot_generate       | 156 |                 0 |  0.4416 |  0.4985 |              0.4985 |  0.5577 | 0.2102                      |
| cost_anchor    | authorization_to_execute | judge_rubric_choice_logprob    | 156 |                 0 |  0.5336 |  0.5456 |              0.522  |  0.2653 | 0.7055                      |
| cost_anchor    | authorization_to_execute | judge_rubric_generate          | 156 |                 2 |  0.4448 |  0.5045 |              0.5045 |  0.4423 | 0.2484                      |
| cost_anchor    | toxicity                 | activation_head                | 512 |                 0 |  0.7571 |  0.9324 |              0.8972 |  0.0673 |                             |
| cost_anchor    | toxicity                 | tfidf_word_char                | 512 |                 0 |  0.7544 |  0.8979 |              0.8203 |  0.0822 |                             |
| cost_anchor    | toxicity                 | judge_zero_shot_choice_logprob | 512 |                 0 |  0.13   |  0.4482 |              0.4163 |  0.5668 | 0.2435                      |
| cost_anchor    | toxicity                 | judge_zero_shot_generate       | 512 |                91 |  0.104  |  0.3389 |              0.3389 |  0.7871 | 0.1068                      |
| cost_anchor    | toxicity                 | judge_rubric_choice_logprob    | 512 |                 0 |  0.1875 |  0.6061 |              0.5677 |  0.1881 | 0.3203                      |
| cost_anchor    | toxicity                 | judge_rubric_generate          | 512 |               163 |  0.127  |  0.34   |              0.34   |  0.4062 | 0.1314                      |
| cost_anchor    | sentiment                | activation_head                | 512 |                 0 |  0.9149 |  0.9114 |              0.8242 |  0.1329 |                             |
| cost_anchor    | sentiment                | tfidf_word_char                | 512 |                 0 |  0.9068 |  0.9026 |              0.8164 |  0.1379 |                             |
| cost_anchor    | sentiment                | judge_zero_shot_choice_logprob | 512 |                 0 |  0.8896 |  0.8839 |              0.5    |  0.417  | 0.1163                      |
| cost_anchor    | sentiment                | judge_zero_shot_generate       | 512 |                 2 |  0.501  |  0.502  |              0.502  |  0.498  | 0.0455                      |
| cost_anchor    | sentiment                | judge_rubric_choice_logprob    | 512 |                 0 |  0.6932 |  0.7387 |              0.5273 |  0.3522 | 0.1795                      |
| cost_anchor    | sentiment                | judge_rubric_generate          | 512 |                 0 |  0.5214 |  0.541  |              0.541  |  0.459  | 0.0651                      |
| quality_anchor | authorization_to_execute | activation_head                | 156 |                 0 |  0.3567 |  0.3721 |              0.4828 |  0.4519 |                             |
| quality_anchor | authorization_to_execute | tfidf_word_char                | 156 |                 0 |  0.4533 |  0.5276 |              0.469  |  0.2494 |                             |
| quality_anchor | authorization_to_execute | judge_zero_shot_choice_logprob | 156 |                 0 |  0.5704 |  0.6168 |              0.51   |  0.3977 | 2.3241                      |
| quality_anchor | authorization_to_execute | judge_zero_shot_generate       | 156 |                 0 |  0.4532 |  0.5215 |              0.5215 |  0.5321 | 0.9265                      |
| quality_anchor | authorization_to_execute | judge_rubric_choice_logprob    | 156 |                 0 |  0.492  |  0.6011 |              0.5885 |  0.2592 | 2.9229                      |
| quality_anchor | authorization_to_execute | judge_rubric_generate          | 156 |                 0 |  0.4796 |  0.5637 |              0.5637 |  0.4231 | 1.0970                      |
| quality_anchor | toxicity                 | activation_head                | 512 |                 0 |  0.8332 |  0.9537 |              0.8884 |  0.0591 |                             |
| quality_anchor | toxicity                 | tfidf_word_char                | 512 |                 0 |  0.7544 |  0.8979 |              0.8203 |  0.0822 |                             |
| quality_anchor | toxicity                 | judge_zero_shot_choice_logprob | 512 |                 0 |  0.1572 |  0.5619 |              0.4888 |  0.1276 | 0.8960                      |
| quality_anchor | toxicity                 | judge_zero_shot_generate       | 512 |                 6 |  0.127  |  0.4855 |              0.4855 |  0.1523 | 0.3937                      |
| quality_anchor | toxicity                 | judge_rubric_choice_logprob    | 512 |                 0 |  0.1013 |  0.376  |              0.4502 |  0.2241 | 1.2400                      |
| quality_anchor | toxicity                 | judge_rubric_generate          | 512 |                 7 |  0.1217 |  0.4673 |              0.4673 |  0.2988 | 0.5663                      |
| quality_anchor | sentiment                | activation_head                | 512 |                 0 |  0.9424 |  0.9481 |              0.877  |  0.0974 |                             |
| quality_anchor | sentiment                | tfidf_word_char                | 512 |                 0 |  0.9068 |  0.9026 |              0.8164 |  0.1379 |                             |
| quality_anchor | sentiment                | judge_zero_shot_choice_logprob | 512 |                 0 |  0.9369 |  0.9343 |              0.6953 |  0.2065 | 0.5400                      |
| quality_anchor | sentiment                | judge_zero_shot_generate       | 512 |                 0 |  0.6234 |  0.6973 |              0.6973 |  0.3027 | 0.2698                      |
| quality_anchor | sentiment                | judge_rubric_choice_logprob    | 512 |                 0 |  0.9487 |  0.946  |              0.8672 |  0.0952 | 0.8415                      |
| quality_anchor | sentiment                | judge_rubric_generate          | 512 |                 0 |  0.8152 |  0.8711 |              0.8711 |  0.1289 | 0.3535                      |

## Paired activation differences

- cost_anchor / authorization_to_execute / activation_head-minus-tfidf_word_char: ΔAUPRC +0.0996 [-0.1387, +0.3277]; ΔAUROC -0.0088 [-0.2821, +0.2694].
- cost_anchor / authorization_to_execute / activation_head-minus-judge_zero_shot_choice_logprob: ΔAUPRC +0.1106 [-0.0857, +0.2796]; ΔAUROC +0.0451 [-0.1989, +0.2750]. AUPRC sign p=0.3348, BH q=0.4464, reject=False.
- cost_anchor / authorization_to_execute / activation_head-minus-judge_rubric_choice_logprob: ΔAUPRC +0.0193 [-0.2096, +0.1876]; ΔAUROC -0.0268 [-0.2647, +0.2114]. AUPRC sign p=0.9195, BH q=0.9195, reject=False.
- cost_anchor / toxicity / activation_head-minus-tfidf_word_char: ΔAUPRC +0.0027 [-0.1085, +0.1131]; ΔAUROC +0.0345 [-0.0139, +0.0860].
- cost_anchor / toxicity / activation_head-minus-judge_zero_shot_choice_logprob: ΔAUPRC +0.6271 [+0.5115, +0.7240]; ΔAUROC +0.4842 [+0.3997, +0.5710]. AUPRC sign p=0.0010, BH q=0.0024, reject=True.
- cost_anchor / toxicity / activation_head-minus-judge_rubric_choice_logprob: ΔAUPRC +0.5696 [+0.4546, +0.6687]; ΔAUROC +0.3263 [+0.2456, +0.4059]. AUPRC sign p=0.0010, BH q=0.0024, reject=True.
- cost_anchor / sentiment / activation_head-minus-tfidf_word_char: ΔAUPRC +0.0081 [-0.0193, +0.0359]; ΔAUROC +0.0088 [-0.0188, +0.0365].
- cost_anchor / sentiment / activation_head-minus-judge_zero_shot_choice_logprob: ΔAUPRC +0.0253 [-0.0050, +0.0572]; ΔAUROC +0.0275 [-0.0037, +0.0567]. AUPRC sign p=0.1089, BH q=0.1694, reject=False.
- cost_anchor / sentiment / activation_head-minus-judge_rubric_choice_logprob: ΔAUPRC +0.2217 [+0.1657, +0.2785]; ΔAUROC +0.1727 [+0.1313, +0.2185]. AUPRC sign p=0.0010, BH q=0.0024, reject=True.
- quality_anchor / authorization_to_execute / activation_head-minus-tfidf_word_char: ΔAUPRC -0.0966 [-0.2078, +0.0163]; ΔAUROC -0.1554 [-0.3334, +0.0507].
- quality_anchor / authorization_to_execute / activation_head-minus-judge_zero_shot_choice_logprob: ΔAUPRC -0.2137 [-0.3973, -0.0335]; ΔAUROC -0.2446 [-0.4794, +0.0144]. AUPRC sign p=0.0240, BH q=0.0480, reject=True.
- quality_anchor / authorization_to_execute / activation_head-minus-judge_rubric_choice_logprob: ΔAUPRC -0.1354 [-0.3475, +0.0347]; ΔAUROC -0.2290 [-0.5144, +0.0596]. AUPRC sign p=0.1129, BH q=0.1694, reject=False.
- quality_anchor / toxicity / activation_head-minus-tfidf_word_char: ΔAUPRC +0.0788 [-0.0091, +0.1859]; ΔAUROC +0.0558 [+0.0066, +0.1126].
- quality_anchor / toxicity / activation_head-minus-judge_zero_shot_choice_logprob: ΔAUPRC +0.6759 [+0.5827, +0.7614]; ΔAUROC +0.3919 [+0.3108, +0.4797]. AUPRC sign p=0.0010, BH q=0.0024, reject=True.
- quality_anchor / toxicity / activation_head-minus-judge_rubric_choice_logprob: ΔAUPRC +0.7319 [+0.6471, +0.8088]; ΔAUROC +0.5777 [+0.4939, +0.6628]. AUPRC sign p=0.0010, BH q=0.0024, reject=True.
- quality_anchor / sentiment / activation_head-minus-tfidf_word_char: ΔAUPRC +0.0356 [-0.0046, +0.0736]; ΔAUROC +0.0455 [+0.0171, +0.0747].
- quality_anchor / sentiment / activation_head-minus-judge_zero_shot_choice_logprob: ΔAUPRC +0.0055 [-0.0299, +0.0379]; ΔAUROC +0.0138 [-0.0076, +0.0380]. AUPRC sign p=0.6767, BH q=0.7971, reject=False.
- quality_anchor / sentiment / activation_head-minus-judge_rubric_choice_logprob: ΔAUPRC -0.0063 [-0.0431, +0.0246]; ΔAUROC +0.0021 [-0.0199, +0.0242]. AUPRC sign p=0.7306, BH q=0.7971, reject=False.
