<!-- FINAL honest draft for Nicole's review — NOT pushed to live README/GitHub/PyPI.
     All numbers verified against research/_results_published/*.json. Adversarially reviewed against
     the comparison-fairness audit's indefensible-claims list. Real-human numbers use the
     UNIFORM-RANDOM HackAPrompt sample (not the order-biased first-N). -->

# IntentProbe

<p align="center">
  <strong>A tiny linear probe on a frozen local model's activations. It generalizes to attacks from sources it never trained on, especially attacks worded in ways it never saw, better than a same-data text classifier.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="License" /></a>
  <a href="https://doi.org/10.5281/zenodo.19990741"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19990741-blue" alt="DOI" /></a>
  <img src="https://img.shields.io/badge/runs-100%25%20local-brightgreen" alt="Runs locally" />
  <img src="https://img.shields.io/badge/telemetry-zero-blue" alt="Zero telemetry" />
  <img src="https://img.shields.io/badge/status-research%20preview-orange" alt="Research preview" />
</p>

<p align="center">
  <img src="docs/honest_generalization.png" width="640" alt="Recall on real human attacks at 1% false-positive rate. HackAPrompt (unfamiliar wording): IntentProbe 88%, TF-IDF 30%. Gandalf (familiar wording): IntentProbe 94%, TF-IDF 92%. The edge is unfamiliar wording, not real attacks in general." />
</p>

## What it is

IntentProbe runs a tool description or prompt through a frozen local model (Qwen2.5-0.5B), reads a few mid-layers, and scores the mean-pooled activation vector with a small logistic probe (about 10.5 KB). Most scanners read the text itself: patterns, classifiers, rules, or "ask an LLM". This reads the host model's internal state instead.

The reason to do it this way is generalization to the unknown. Train a text classifier on attacks and it learns the words of those attacks. Face a new attack worded differently and its features stop transferring, so recall drops. The probe keys off how the model represents the input, not the exact tokens, so it holds up better when the wording is new. That is the whole claim, and it is the case that matters most: the attack you could not enumerate in advance.

This is a **research preview**, a local single-pass review signal, not a hard security boundary. It runs 100% local on any CPU with nothing uploaded. To our knowledge it is the only shipped tool in this exact shape (local, standalone tool/skill descriptions at registration time, no clean reference, no response generation). It is **not** the first probe-based detector; there is substantial prior and parallel work (TaskTracker, PIShield, MindGuard, RouteGuard, and others).

## How it works

The probe is the primary signal for `allow` and `warn`. The `block` tier additionally requires a static-keyword hit to hold down false positives, so a novel no-keyword input the probe flags surfaces as `warn`, not `block`. "Reads activations, not text" describes the warn signal, not the whole product.

The probe needs the frozen 0.5B model to produce the activations, so inference is **heavier** than a standalone text classifier, not lighter. The ~10.5 KB is a training-and-storage advantage (no base fine-tuning), not a runtime one.

## Benchmarks

Reproducible from `research/`. The honest summary first: the probe's edge is **generalization to attacks worded in unfamiliar ways**. Where the wording is new, it wins clearly. Where the new source reuses familiar attack wording, a text classifier transfers fine and the two are comparable. It is not better in general, and within one distribution it is not better at all.

**1. Real human attacks, two sources neither detector trained on.** Both are positive-only (attacks, no benign), so we report recall at a clean false-positive rate fixed on the training data.

```
                              HackAPrompt (n=3,866,        Gandalf (n=1,000,
                              diverse/disguised wording)   wording overlaps training)
  recall @ 1% clean-FPR
    Probe (Qwen2.5-0.5B)            88.3%                       93.8%
    TF-IDF (same train data)       30.3%                       92.1%
  recall @ 5% clean-FPR
    Probe                          90.3%                       94.4%
    TF-IDF                         52.8%                       97.4%
```

Read this honestly. On HackAPrompt, where real people phrase attacks in ways the training sources never used, the text classifier's vocabulary does not transfer and it misses most attacks (30% at a 1% FPR); the probe still catches 88%. On Gandalf, where the attack wording overlaps the training data, the text classifier transfers fine and is comparable or better at looser thresholds. The probe's advantage is specifically about unfamiliar wording, not real attacks in general.

**2. Curated cross-source generalization: leave-one-source-out, nested CV, 4 real PI datasets.** Train on three of {deepset, safeguard, spml, jayavibhav}, test on the held-out fourth. Model and layer are chosen inside the nested loop, never on the held-out source. 95% bootstrap CIs on the probe-minus-TF-IDF AUROC difference.

```
  held-out      probe AUROC   TF-IDF AUROC   difference (95% CI)
  deepset          0.941         0.732        +0.209 [0.168, 0.250]  significant
  spml             0.995         0.935        +0.059 [0.044, 0.077]  significant
  safeguard        0.999         0.993        +0.006 [0.002, 0.011]  significant, at ceiling
  jayavibhav       1.000         0.997        +0.002 [0.000, 0.005]  tie
  mean             0.984         0.914        +0.070
```

The gap is driven by deepset, where the held-out vocabulary differs most. The shipped fixed config (0.5B, mean-pooled layers 13-15) reaches mean 0.980 with no per-input layer choice, and the result is robust to layer choice (an exhaustive single+pair search lands at 0.982; combining more layers does not beat one good mid-layer).

**3. Within one distribution, text is not blind.** On matched-vocabulary minimal pairs (same topic, near-identical words, GroupKFold, best layer) the probe ties TF-IDF, roughly 0.79 vs 0.82, with overlapping CIs. A length-only baseline is at chance (0.50) there, so length is not the story. Do not expect the probe to catch a same-words poison a bag-of-words model would also catch.

**4. Tool-poisoning, cross-corpus: partial.** Train on two of {routeguard skill-text, MCPTox, synthetic minimal-pairs}, test on the third. On held-out MCPTox the probe leads (0.738 vs TF-IDF 0.545, +0.19 [0.145, 0.241]). On routeguard the lead is not significant. On the synthetic minimal-pairs both detectors are at chance, which says that synthetic set is out of distribution, not that either method works on it.

## Limitations

- **The edge is conditional.** It shows up on attacks worded unlike the training data (HackAPrompt). On familiar-wording attacks (Gandalf) and within one distribution (minimal pairs) a text classifier is comparable or better.
- **Real-human evidence is two sources, positive-only.** Recall at a matched FPR, not AUROC, and not an exhaustive panel. HackAPrompt numbers use a uniform-random sample of the corpus.
- **It needs a frozen ~0.5B host model.** Inference is heavier than a standalone text classifier. The small size is a train/store advantage only.
- **Single model family.** Numbers are Qwen2.5. Every base model needs its own retrained probe.
- **The `block` tier is not pure activations.** It needs static-keyword corroboration, so a novel no-keyword attack surfaces as `warn`.
- **Not first, not only on accuracy.** It is one point in a deployment shape, not a new state of the art. Not a security boundary; treat it as a review tripwire.

## Prior and related work

Reading model internals to flag malicious input is an established area. TaskTracker (activation deltas, 2024), PIShield (frozen-model residual stream + linear classifier, pip-installable, 2025), MindGuard (attention, 2025), RouteGuard (hidden-state + attention for skill poisoning, 2026), and deployed probes at frontier labs all predate or parallel this. IntentProbe's contribution is the specific deployment shape and the cross-source generalization evidence, not primacy.
