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
   within-distribution result is a **tie** with a TF-IDF text baseline (~0.79 vs ~0.82
   AUROC), not a decisive probe win. The "structurally blind to same-vocabulary
   attacks" framing is withdrawn.

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
   shipped-product performance.** The shipped product uses Qwen2.5-0.5B. Its current
   evidence is recall on a held-out real-attack source (HackAPrompt: 90.3% at a 5%
   clean false-positive rate, vs a same-data TF-IDF baseline's 52.8%) and curated
   cross-source AUROC (0.984 vs 0.914, CI-backed). These supersede the paper's headline
   numbers for any product claim.

## What stands

The core hypothesis — that a linear probe on a frozen model's mid-layer activations
generalizes to attacks worded in ways a text classifier never saw, better than a
same-data text classifier — is supported by the corrected, CI-backed cross-source
results in the repository. The contribution is generalization in a specific deployment
shape; it is not a new state of the art and not first-of-technique (PIShield,
TaskTracker, RouteGuard, MindGuard, and frontier-lab production probes predate or
parallel it).

All corrected benchmarks and scripts: <https://github.com/mcpware/IntentProbe> (`research/`).
