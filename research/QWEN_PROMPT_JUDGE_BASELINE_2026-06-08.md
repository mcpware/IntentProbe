# Qwen Direct-Prompt Judge Baseline - 2026-06-08

## Question

Can the same local Qwen sensor model simply be prompted to classify tool
poisoning, instead of reading its hidden activations and training a probe?

This tests a common LLM-as-judge shape against the current activation-probe
lane. It does not replace the product scanner.

## Setup

- Model: `Qwen/Qwen2.5-0.5B`
- Device: Apple MPS
- Prompt: fixed zero-shot `CLEAN` vs `POISONED` tool-security prompt
- Script: `research/benchmark_qwen_prompt_judge.py`
- Seed: `42`
- Methods:
  - `choice-logprob`: compare the model probability of the next label
    ` CLEAN` vs ` POISONED`;
  - `generate`: ask the model to generate an answer, then parse the generated
    text.

## Commands

```bash
.venv/bin/python -m research.benchmark_qwen_prompt_judge \
  --dataset family_curated_v0 \
  --max-samples 9999 \
  --method choice-logprob \
  --device auto \
  --pretty

.venv/bin/python -m research.benchmark_qwen_prompt_judge \
  --dataset family_curated_v0 \
  --max-samples 9999 \
  --method generate \
  --device auto \
  --pretty

.venv/bin/python -m research.benchmark_qwen_prompt_judge \
  --dataset routeguard_external_v0 \
  --max-samples 200 \
  --method choice-logprob \
  --device auto \
  --pretty

.venv/bin/python -m research.benchmark_qwen_prompt_judge \
  --dataset routeguard_external_v0 \
  --max-samples 100 \
  --method generate \
  --device auto \
  --pretty
```

## Results

| Dataset | Method | n | Accuracy | Precision | Recall | F1 | Clean FPR | Unknown |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `family_curated_v0` | `choice-logprob` | 76 | 0.500 | 0.500 | 1.000 | 0.667 | 1.000 | 0 |
| `family_curated_v0` | `generate` | 76 | 0.500 | 0.500 | 0.789 | 0.612 | 0.789 | 16 |
| `routeguard_external_v0` sample | `choice-logprob` | 200 | 0.500 | 0.500 | 1.000 | 0.667 | 1.000 | 0 |
| `routeguard_external_v0` sample | `generate` | 100 | 0.590 | 0.696 | 0.320 | 0.438 | 0.140 | 77 |

Saved outputs:

- `research/_results/qwen_prompt_judge/20260608T072516Z-family-curated-qwen-choice-logprob-full.json`
- `research/_results/qwen_prompt_judge/20260608T072539Z-family-curated-qwen-generate-full.json`
- `research/_results/qwen_prompt_judge/20260608T072602Z-routeguard-qwen-choice-logprob-sample200.json`
- `research/_results/qwen_prompt_judge/20260608T072711Z-routeguard-qwen-generate-sample100.json`

## Read

The direct-prompt baseline is not competitive with the activation scanner.

The label-logprob version is deterministic and parse-free, but it collapses into
an all-poison policy on both tested gates. That gives perfect poison recall but
blocks every clean item, so it is not usable as a scanner default.

The generation version reduces false positives on the RouteGuard sample, but it
misses most poisoned rows and produces many unparseable outputs. That is a poor
hot-path security contract: the scanner has to decide whether to allow, warn, or
block, not interpret arbitrary text continuations.

Important nuance: LLM-as-judge can be made more deterministic with fixed
decoding, especially at `temperature=0`. The bigger product problem is prompt
fragility, parseability, latency, and model/version drift. A fixed activation
artifact has a simpler contract: same input, same model, same probe, same
threshold, same score.

## Current Position

Keep raw Qwen activation probing as the v0 product default. Keep direct-prompt
LLM judge baselines in the benchmark suite as a comparison lane, not as the
scanner implementation.
