# Erratum / Corrigendum

**Paper:** *Can Model Internals Detect MCP Tool Poisoning That Text Analysis Cannot?*
**DOI:** [10.5281/zenodo.19990741](https://doi.org/10.5281/zenodo.19990741)
**Date:** 2026-06-18

This erratum corrects errors in the preliminary study and clarifies how its numbers
relate to the shipped IntentProbe product. The paper remains a preliminary GPT-2
proof-of-concept. The current, defensible evidence is the Qwen2.5-0.5B cross-source
generalization results in this repository (`research/_results_published/`).

## Corrections

1. **The matched-pair headline F1 (97.5% / 98.5%) was inflated by pair leakage.** The
   original matched-pair evaluation did not fully separate the two halves of each
   minimal pair across train/test. Under leak-free GroupKFold by pair, the
   within-distribution result is not a probe win. On the **shipped Qwen2.5-0.5B config**
   the TF-IDF text baseline slightly **beats** the probe (~0.74 vs ~0.82 AUROC); a nested
   CV free to pick a larger 1.5B sensor closes that to roughly a tie (~0.79 vs ~0.82), but
   that is not the shipped config. The "structurally blind to same-vocabulary attacks"
   framing is withdrawn.

2. **The same-words result was confounded by length/framing at low layers.** A
   length-only baseline is at chance, but controlling for length and framing does not
   by itself establish the probe reads "intent." The surviving claim is generalization
   to novel-source / novel-vocabulary attacks, not same-vocabulary detection.

3. **The statistics were underpowered and did not respect pairing** (n ≈ 100; a
   permutation test that did not respect the pair structure). The corrected repository
   results use bootstrap confidence intervals and grouped/paired resampling.

4. **"485 MCPTox descriptions" is a self-selected subset.** The MCPTox benchmark
   contains 1,312 poisoned descriptions; the paper used a 485-item subset. It should
   read "a 485-item subset of MCPTox."

5. **"2,059 SAE neurons" should read 1,758 SAE features** (Safe-SAIL).

6. **The 96.5% / 96.6% / 100% figures are GPT-2 research-probe numbers, not
   shipped-product performance.** The shipped product uses Qwen2.5-0.5B (mean-pooled
   concat L13-15). Its current evidence is recall on a held-out real-attack source
   (HackAPrompt: 90.3% at a 5% clean false-positive rate, vs a same-data TF-IDF
   baseline's 52.8%) and curated leave-one-source-out cross-source AUROC of **0.980 vs
   0.914** for that shipped 0.5B config (per-source: deepset 0.933, safeguard 0.999, spml
   0.990, jayavibhav 1.000). A nested cross-validation that is additionally free to pick a
   larger **1.5B** sensor per fold (deepset and spml folds picked Qwen2.5-1.5B) reaches a
   **research upper bound** of 0.984, with 95% bootstrap CIs on the probe-minus-TF-IDF
   difference (deepset +0.209, spml +0.059, both significant); that is an upper bound, not
   the shipped 0.5B artifact. These shipped numbers supersede the paper's headline figures
   for any product claim.

## What stands

The core hypothesis — that a linear probe on a frozen model's mid-layer activations
generalizes to attacks worded in ways a text classifier never saw, better than a
same-data text classifier — is supported by the corrected, CI-backed cross-source
results in the repository. The contribution is generalization in a specific deployment
shape; it is not a new state of the art and not first-of-technique (PIShield,
TaskTracker, RouteGuard, MindGuard, and frontier-lab production probes predate or
parallel it).

All corrected benchmarks and scripts: <https://github.com/mcpware/IntentProbe> (`research/`).
