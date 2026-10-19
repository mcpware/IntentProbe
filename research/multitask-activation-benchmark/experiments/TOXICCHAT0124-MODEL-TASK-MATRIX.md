# ToxicChat: six-backbone crossed activation-readout matrix

The matrix uses the same 5,441 human-annotated prompts and frozen train/validation/test rows for
both toxicity and jailbreak. All six score artifacts have identical held-out IDs, labels, and
matched TF-IDF probabilities within each task.

| Task | Backbone | Validation-selected readout | Depth | Test AUPRC | TF-IDF AUPRC | Difference (95% CI) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| jailbreaking | LiquidAI/LFM2.5-230M | masked_max 4--6 | 0.357 | 0.7471 | 0.8264 | -0.0793 [-0.1369, -0.0196] |
| jailbreaking | Qwen/Qwen2.5-0.5B | masked_max 5--7 | 0.250 | 0.6650 | 0.8264 | -0.1614 [-0.2310, -0.0945] |
| jailbreaking | Qwen/Qwen3-0.6B-Base | masked_max 8--9 | 0.304 | 0.7723 | 0.8264 | -0.0541 [-0.1091, +0.0029] |
| jailbreaking | Qwen/Qwen3.5-0.8B | masked_mean 9--9 | 0.375 | 0.8422 | 0.8264 | +0.0158 [-0.0295, +0.0629] |
| jailbreaking | ibm-granite/granite-4.0-350m-base | masked_mean 18--20 | 0.679 | 0.8514 | 0.8264 | +0.0250 [-0.0187, +0.0726] |
| jailbreaking | openbmb/MiniCPM5-1B-Base | masked_mean 19--21 | 0.833 | 0.8499 | 0.8264 | +0.0235 [-0.0222, +0.0752] |
| toxicity | LiquidAI/LFM2.5-230M | masked_mean 3--5 | 0.286 | 0.7491 | 0.7262 | +0.0229 [-0.0250, +0.0710] |
| toxicity | Qwen/Qwen2.5-0.5B | masked_mean 18--20 | 0.792 | 0.7093 | 0.7262 | -0.0169 [-0.0570, +0.0236] |
| toxicity | Qwen/Qwen3-0.6B-Base | masked_mean 15--17 | 0.571 | 0.7546 | 0.7262 | +0.0284 [-0.0114, +0.0712] |
| toxicity | Qwen/Qwen3.5-0.8B | masked_mean 9--9 | 0.375 | 0.7665 | 0.7262 | +0.0403 [-0.0003, +0.0845] |
| toxicity | ibm-granite/granite-4.0-350m-base | masked_mean 15--17 | 0.571 | 0.7863 | 0.7262 | +0.0601 [+0.0168, +0.1047] |
| toxicity | openbmb/MiniCPM5-1B-Base | masked_mean 12--12 | 0.500 | 0.7816 | 0.7262 | +0.0554 [+0.0131, +0.0978] |

## Readout pattern

Toxicity selected masked mean in all six backbones. Jailbreak split evenly between masked mean
and masked max. No task selected last-nonpadding pooling. Selected normalized depths span
0.250 to 0.833; the median is 0.438.

The best single validation policy is masked mean, width 2, centered at normalized depth 0.50.
Its mean validation AUPRC is 0.7872, compared with 0.8220 for per-task-per-backbone selection. Model-specific policies reduce mean regret from 0.0348 to 0.0142; task-specific policies alone leave 0.0338 regret.

## Evidence boundary

The descriptive normalized-depth decomposition assigns 43.5% of variation to backbone, 1.7% to the two-task main effect, and 54.8% to task-by-backbone interaction. With only two tasks and no replicated interaction error term, these shares are descriptive rather than population inference.
The policy comparison uses the same validation cells to select and evaluate policies, so its regret
estimates are also descriptive. Held-out activation-versus-TF-IDF claims use the prospectively
specified paired row bootstrap; two of twelve cells have intervals above zero on both AUPRC and
AUROC, both on toxicity (Granite-4.0-350M and MiniCPM5-1B).
