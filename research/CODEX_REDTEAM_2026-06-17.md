# Codex Red-Team Audit - 2026-06-17

Scope: independent pre-launch audit of the IntentProbe README headline claims:

- "Snyk's shipped scanner catches 0%, IntentProbe catches 96.5%" on matched-vocabulary MCP tool poisoning.
- "100% MCPTox recall."
- "Reads what the model understood, not what the text says."

Bottom line: the README headline is not defensible as written for Hacker News. The strongest defensible claim is narrower: a GPT-2 research probe reaches about 96.6% F1 under shuffled 5-fold CV on a small synthetic n=86 hand-written pool, and the shipped Qwen product artifact is a separate model trained on that pool plus MCPTox. The shipped hard-block tier does not block on activations alone; every block path requires static regex evidence.

## Executive Verdict

**Launch verdict: rescope before HN.**

The current README conflates at least three different things:

1. A **GPT-2 research benchmark** that produces the 96.5% / 96.6% headline number.
2. A **shipped Qwen2.5-0.5B artifact** packaged under `intentprobe/scanner/artifacts`.
3. A **runtime product decision policy** whose `block` tier is static-corroborated, not activation-only.

The most serious issue is provenance. The benchmark script that produces the headline matched-vocabulary numbers says the probe is GPT-2 layer 3 (`research/benchmark-probe-vs-deberta-samesplit.py:3`, `research/benchmark-probe-vs-deberta-samesplit.py:8`, `research/benchmark-probe-vs-deberta-samesplit.py:23`, `research/benchmark-probe-vs-deberta-samesplit.py:57`, `research/benchmark-probe-vs-deberta-samesplit.py:60`, `research/benchmark-probe-vs-deberta-samesplit.py:135`). The shipped CLI default artifact is `qwen-pooled-curated-core-l13-15-v2` (`intentprobe/scanner/cli.py:31`, `intentprobe/scanner/cli.py:35`), and that artifact's `model_id` is `Qwen/Qwen2.5-0.5B` (`intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2/metadata.json:141`).

## Finding 1 - Critical: The Headline Numbers Are GPT-2 Research Numbers, Not The Shipped Product Model

The README claims:

- "Snyk's shipped scanner catches **0%**. IntentProbe catches **96.5%**" (`README.md:22`).
- "Matched-vocabulary F1 (n=86) ... IntentProbe ... **96.6%**" (`README.md:89`, `README.md:90`).
- "MCPTox poisoned recall (n=249) ... IntentProbe ... **100.0%**" (`README.md:85`, `README.md:86`).
- The limitations block repeats "Matched-vocabulary poisoning 96.5%" and "Template attacks (MCPTox) 100%" (`README.md:269`, `README.md:270`).

Those numbers match the same-split GPT-2 research benchmark:

- The script header says "activation probe (GPT-2) vs Snyk's shipped detector" (`research/benchmark-probe-vs-deberta-samesplit.py:3`).
- It says the probe is "mean-pooled GPT-2 residual-stream activations at layer 3" (`research/benchmark-probe-vs-deberta-samesplit.py:8`, `research/benchmark-probe-vs-deberta-samesplit.py:9`).
- It sets `LAYER = 3` (`research/benchmark-probe-vs-deberta-samesplit.py:23`).
- It loads `gpt2` tokenizer and model (`research/benchmark-probe-vs-deberta-samesplit.py:57`, `research/benchmark-probe-vs-deberta-samesplit.py:58`, `research/benchmark-probe-vs-deberta-samesplit.py:60`).
- It prints the small-pool row as "Activation probe (GPT-2 L3, 5-fold CV)" (`research/benchmark-probe-vs-deberta-samesplit.py:135`).
- The linked benchmark result table says "GPT-2 layer-3 mean-pooled residual stream" (`research/benchmark-results-deberta-vs-probe-2026-05-31.md:30`, `research/benchmark-results-deberta-vs-probe-2026-05-31.md:31`) and reports 96.5% accuracy / 96.6% F1 (`research/benchmark-results-deberta-vs-probe-2026-05-31.md:39`, `research/benchmark-results-deberta-vs-probe-2026-05-31.md:42`).

The shipped artifact is different:

- The default CLI artifact is `qwen-pooled-curated-core-l13-15-v2` (`intentprobe/scanner/cli.py:31`, `intentprobe/scanner/cli.py:35`).
- The artifact is packaged by `pyproject.toml` (`pyproject.toml:36`, `pyproject.toml:42`, `pyproject.toml:43`) and `MANIFEST.in` (`MANIFEST.in:7`).
- The artifact metadata says `model_id: Qwen/Qwen2.5-0.5B` (`intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2/metadata.json:141`).
- The sensor model is `qwen2.5-0.5b` (`intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2/metadata.json:171`).
- The artifact uses concatenated Qwen layers 13, 14, and 15 (`intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2/metadata.json:124`, `intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2/metadata.json:133`, `intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2/metadata.json:137`).

The product calibration document also frames the shipped Qwen artifact differently:

- It says the current pooled scanner artifact is Qwen (`research/THRESHOLD_CALIBRATION_QWEN_POOLED_2026-06-03.md:5`, `research/THRESHOLD_CALIBRATION_QWEN_POOLED_2026-06-03.md:9`).
- It promotes `qwen-pooled-curated-core-l13-15-v2` as the runtime candidate (`research/THRESHOLD_CALIBRATION_QWEN_POOLED_2026-06-03.md:132`, `research/THRESHOLD_CALIBRATION_QWEN_POOLED_2026-06-03.md:134`, `research/THRESHOLD_CALIBRATION_QWEN_POOLED_2026-06-03.md:135`).
- Its honest product claim is a local warn/review layer with conservative hard block, not 96.5% matched-vocabulary product performance (`research/THRESHOLD_CALIBRATION_QWEN_POOLED_2026-06-03.md:137`, `research/THRESHOLD_CALIBRATION_QWEN_POOLED_2026-06-03.md:139`, `research/THRESHOLD_CALIBRATION_QWEN_POOLED_2026-06-03.md:141`).

Adversarial conclusion: the README implies the shipped IntentProbe product model produces the headline benchmark numbers. The code shows those numbers are from a GPT-2 research probe. That is a critical launch-risk mismatch.

## Finding 2 - Critical: The Shipped Artifact Metric Is A Train Metric, And The Matched Set Is In Training

The training script computes training metrics on the same matrix it fits:

- `train_probe()` fits `LogisticRegression` on `matrix, y` (`research/train_probe_artifact.py:191`, `research/train_probe_artifact.py:193`, `research/train_probe_artifact.py:194`).
- It predicts on the same `matrix` (`research/train_probe_artifact.py:195`).
- It computes `binary_metrics(y, pred)` (`research/train_probe_artifact.py:196`).
- It saves the result under `train_metrics` (`research/train_probe_artifact.py:197`, `research/train_probe_artifact.py:201`).

The shipped metadata stores exactly that train metric:

- `train_metrics.accuracy = 0.994944...` (`intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2/metadata.json:149`, `intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2/metadata.json:150`).
- `train_metrics.f1 = 0.995563...` (`intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2/metadata.json:151`).
- `train_metrics.recall = 0.991166...` (`intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2/metadata.json:153`).

The matched set is in the shipped artifact's training data:

- The artifact `train_source` is `pooled-curated-core` (`intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2/metadata.json:109`).
- `pooled-curated-core` loads MCPTox, the hand core pool, and `family_curated_v0` (`research/train_probe_artifact.py:104`, `research/train_probe_artifact.py:105`).
- The hand core styles are `matched`, `neutral`, and `hard_v2` (`research/benchmarks/datasets.py:86`).
- The matched style maps to `hard_v3_matched_clean.json` and `hard_v3_matched_poisoned.json` (`research/benchmarks/datasets.py:71`, `research/benchmarks/datasets.py:72`, `research/benchmarks/datasets.py:73`).
- The artifact summary explicitly lists `hard_v3_matched_clean.json: 8` and `hard_v3_matched_poisoned.json: 8` as training sources (`intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2/metadata.json:68`, `intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2/metadata.json:69`).
- The artifact summary also lists MCPTox clean and poisoned files as training sources (`intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2/metadata.json:70`, `intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2/metadata.json:71`).

Adversarial conclusion: do not describe the shipped artifact's stored 0.9956 F1 as held-out performance. It is a training metric, and both the matched set and MCPTox are inside the training pool.

## Finding 3 - High: The n=86 CV Uses Row-Level Shuffled Folds With Pair Leakage And Duplicate Rows

The same-split script builds the n=86 pool from:

- `hard_v3_matched_clean.json` / `hard_v3_matched_poisoned.json` (`research/benchmark-probe-vs-deberta-samesplit.py:123`).
- `neutral_clean.json` / `neutral_poisoned.json` (`research/benchmark-probe-vs-deberta-samesplit.py:124`).
- `hard_v2_clean.json` / `hard_v2_poisoned.json` (`research/benchmark-probe-vs-deberta-samesplit.py:125`).
- It does not include MCPTox in that n=86 pool (`research/benchmark-probe-vs-deberta-samesplit.py:121`, `research/benchmark-probe-vs-deberta-samesplit.py:126`).

The CV is row-level stratified shuffle:

- It uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)` (`research/benchmark-probe-vs-deberta-samesplit.py:129`).
- It fits on row indices and predicts on row indices without grouping (`research/benchmark-probe-vs-deberta-samesplit.py:131`, `research/benchmark-probe-vs-deberta-samesplit.py:132`).

I reran the GPT-2 layer-3 activation CV in `research/.venv-audit` with cached local GPT-2. Results:

```text
POOL_N 86
LABEL_COUNTS {0: 43, 1: 43}
EXACT_DUP_GROUPS 8
EXACT_DUP_ROWS_TOTAL 16
EXACT_DUP_ROWS_BONUS 8
ORIGINAL_PAIR_LEAK_ROWS 68
CV_RESULT original_raw_stratified_shuffle n= 86 acc= 0.9651 prec= 0.9545 recall= 0.9767 f1= 0.9655 errors= 3
CV_RESULT raw_grouped_by_pair_or_duplicate n= 86 acc= 0.9419 prec= 0.9318 recall= 0.9535 f1= 0.9425 errors= 5
DEDUP_N 78 REMOVED 8 LABEL_COUNTS {0: 39, 1: 39}
CV_RESULT exact_dedup_grouped_by_pair n= 78 acc= 0.9487 prec= 0.9268 recall= 0.9744 f1= 0.9500 errors= 4
```

Interpretation:

- The original 96.6% F1 reproduces.
- Grouping by paired topic or exact duplicate drops F1 to 94.25%.
- Removing exact duplicate rows and grouping by pair drops F1 to 95.00%.
- The delta is not a collapse, but the original headline should not be sold as a clean independent held-out result.

Exact duplicate examples exist across `hard_v3_matched_*` and `hard_v2_*`. For example:

- The hard_v3 matched SSH clean row (`research/datasets/hard_v3_matched_clean.json:31`) duplicates the hard_v2 SSH clean row (`research/datasets/hard_v2_clean.json:3`).
- The hard_v3 matched SSH poisoned row (`research/datasets/hard_v3_matched_poisoned.json:23`) duplicates the hard_v2 SSH poisoned row (`research/datasets/hard_v2_poisoned.json:3`).
- The hard_v3 matched file-permissions clean row (`research/datasets/hard_v3_matched_clean.json:23`) duplicates the hard_v2 file-permissions clean row (`research/datasets/hard_v2_clean.json:7`).
- The hard_v3 matched random-password poisoned row (`research/datasets/hard_v3_matched_poisoned.json:19`) duplicates the hard_v2 random-password poisoned row (`research/datasets/hard_v2_poisoned.json:55`).

Adversarial conclusion: the n=86 result is a small in-distribution CV result with row-level pair leakage and exact duplicate rows. It still looks strong after grouping/dedup, but the README headline should say "GPT-2 research probe, small synthetic grouped/dedup CV ~94-95% F1" if it wants to survive hostile review.

## Finding 4 - High: `block` Structurally Requires Static Regex Evidence

The shipped scanner does use activations to compute scores:

- It extracts features and computes activation probabilities (`intentprobe/scanner/core.py:453`, `intentprobe/scanner/core.py:454`, `intentprobe/scanner/core.py:455`).
- It sets `risk_score = max(activation_score, static_risk_score)` (`intentprobe/scanner/core.py:368`, `intentprobe/scanner/core.py:369`, `intentprobe/scanner/core.py:370`).

But the `block` decision has only two paths, and both require static findings:

### Block path A - Static-only bundle

- `choose_runtime_decision()` first checks `static_block_bundle(static_findings)` (`intentprobe/scanner/core.py:285`, `intentprobe/scanner/core.py:286`).
- If present, it returns `block` (`intentprobe/scanner/core.py:287`, `intentprobe/scanner/core.py:289`).
- The static block bundle is `static-secret-file + static-exfiltration + static-hidden-action` (`intentprobe/scanner/core.py:29`, `intentprobe/scanner/core.py:30`, `intentprobe/scanner/core.py:31`).
- `static_block_bundle()` requires that bundle's IDs be present and nearby (`intentprobe/scanner/core.py:249`, `intentprobe/scanner/core.py:252`, `intentprobe/scanner/core.py:257`, `intentprobe/scanner/core.py:258`).

### Block path B - Activation plus static corroborator

- The second `block` path requires `activation_score >= block_threshold` (`intentprobe/scanner/core.py:291`, `intentprobe/scanner/core.py:292`).
- It also requires `static_risk_score >= warn_threshold` (`intentprobe/scanner/core.py:293`).
- It also requires `activation_static_corroborated(static_findings)` (`intentprobe/scanner/core.py:294`).
- The corroborating IDs are `static-exfiltration`, `static-hidden-action`, and `static-host-modification` (`intentprobe/scanner/core.py:32`, `intentprobe/scanner/core.py:36`).
- `activation_static_corroborated()` is a set-intersection over static finding IDs (`intentprobe/scanner/core.py:262`, `intentprobe/scanner/core.py:264`, `intentprobe/scanner/core.py:266`).

If activation is high but static corroboration is missing, the product downgrades to warn:

- `if risk_score >= warn_threshold` then `if risk_score >= block_threshold` it appends "downgraded to warn" (`intentprobe/scanner/core.py:302`, `intentprobe/scanner/core.py:303`, `intentprobe/scanner/core.py:304`).
- It then returns `warn` (`intentprobe/scanner/core.py:307`).

I verified the structure directly:

```text
activation_only_high warn ['high risk score without block corroboration; downgraded to warn']
static_bundle_only block ['static block bundle: static-exfiltration,static-hidden-action,static-secret-file']
activation_plus_static_corroborator block ['activation block score corroborated by static finding (activation=0.990, static=0.800)']
activation_plus_non_corroborating_static warn ['high risk score without block corroboration; downgraded to warn']
```

The product calibration doc says the same thing:

- `block` if the static bundle appears (`research/THRESHOLD_CALIBRATION_QWEN_POOLED_2026-06-03.md:42`, `research/THRESHOLD_CALIBRATION_QWEN_POOLED_2026-06-03.md:44`, `research/THRESHOLD_CALIBRATION_QWEN_POOLED_2026-06-03.md:46`).
- `block` if activation reaches 0.85, static reaches 0.30, and static evidence includes a corroborator (`research/THRESHOLD_CALIBRATION_QWEN_POOLED_2026-06-03.md:47`, `research/THRESHOLD_CALIBRATION_QWEN_POOLED_2026-06-03.md:50`).
- `warn` if risk reaches 0.30 but hard-block corroboration is missing (`research/THRESHOLD_CALIBRATION_QWEN_POOLED_2026-06-03.md:51`, `research/THRESHOLD_CALIBRATION_QWEN_POOLED_2026-06-03.md:53`).

Adversarial conclusion: "reads what the model understood" is partly true for scoring and warn/review, but false if readers infer that the shipped hard gate blocks based on model internals alone. The shipped `block` decision structurally requires a static regex hit.

## Finding 5 - Medium: `hard_v3_matched_*` Is Not A Set Of True Same-Vocabulary Minimal Pairs

The `hard_v3_matched_*` files contain 8 clean rows and 8 poisoned rows:

- Clean file has 8 objects (`research/datasets/hard_v3_matched_clean.json:1`, `research/datasets/hard_v3_matched_clean.json:33`).
- Poisoned file has 8 objects (`research/datasets/hard_v3_matched_poisoned.json:1`, `research/datasets/hard_v3_matched_poisoned.json:33`).

I found zero strict same-vocabulary pairs by token-set equality:

```text
HARD_V3_MATCHED_COUNTS 8 8
HARD_V3_EXACT_SAME_VOCAB_PAIRS 0 []
HARD_V3_BEST_JACCARD_BY_CLEAN
 - clean#0 best_poison#5 jaccard=0.128 shared=5 union=39
 - clean#1 best_poison#2 jaccard=0.071 shared=2 union=28
 - clean#2 best_poison#7 jaccard=0.116 shared=5 union=43
 - clean#3 best_poison#0 jaccard=0.179 shared=5 union=28
 - clean#4 best_poison#6 jaccard=0.137 shared=7 union=51
 - clean#5 best_poison#7 jaccard=0.109 shared=6 union=55
 - clean#6 best_poison#7 jaccard=0.078 shared=5 union=64
 - clean#7 best_poison#5 jaccard=0.439 shared=18 union=41
```

Manual inspection supports the token-set result:

- The clean file starts with CSV summary statistics (`research/datasets/hard_v3_matched_clean.json:2`, `research/datasets/hard_v3_matched_clean.json:4`), memory usage (`research/datasets/hard_v3_matched_clean.json:6`, `research/datasets/hard_v3_matched_clean.json:8`), passwd database user info (`research/datasets/hard_v3_matched_clean.json:10`, `research/datasets/hard_v3_matched_clean.json:12`), and system uptime (`research/datasets/hard_v3_matched_clean.json:14`, `research/datasets/hard_v3_matched_clean.json:16`).
- The poisoned file starts with hostname and `/etc/` identity profiling (`research/datasets/hard_v3_matched_poisoned.json:2`, `research/datasets/hard_v3_matched_poisoned.json:4`), QR tracking URL generation (`research/datasets/hard_v3_matched_poisoned.json:6`, `research/datasets/hard_v3_matched_poisoned.json:8`), git config tokens (`research/datasets/hard_v3_matched_poisoned.json:10`, `research/datasets/hard_v3_matched_poisoned.json:12`), and terminal encoding files (`research/datasets/hard_v3_matched_poisoned.json:14`, `research/datasets/hard_v3_matched_poisoned.json:16`).
- The only obvious same-topic near-minimal pair is SSH config clean (`research/datasets/hard_v3_matched_clean.json:31`, `research/datasets/hard_v3_matched_clean.json:32`) versus SSH private-key poisoned (`research/datasets/hard_v3_matched_poisoned.json:23`, `research/datasets/hard_v3_matched_poisoned.json:24`). Even that pair is not same-vocabulary; it adds "private key files", "fingerprints", and "configured identity" vocabulary.

Adversarial conclusion: `hard_v3_matched_*` should not be described as 8 true same-vocabulary minimal pairs. At strict same-vocabulary level the count is 0. At same-topic near-minimal level the count is 1 obvious pair. The rest are better described as length-ish matched or mixed-topic subtle examples.

## Finding 6 - Medium: The n=86 Pool Is Not Sourced From MCPTox, But The Shipped Artifact Trains On MCPTox

The n=86 benchmark pool is constructed only from matched, neutral, and hard_v2 files (`research/benchmark-probe-vs-deberta-samesplit.py:121`, `research/benchmark-probe-vs-deberta-samesplit.py:123`, `research/benchmark-probe-vs-deberta-samesplit.py:126`). My exact-text overlap check found zero overlap with MCPTox:

```text
SMALL_SOURCE_FILES Counter({'hard_v2_clean.json': 20, 'hard_v2_poisoned.json': 20, 'neutral_clean.json': 15, 'neutral_poisoned.json': 15, 'hard_v3_matched_clean.json': 8, 'hard_v3_matched_poisoned.json': 8})
SMALL_N 86
MCPTox_N 827
EXACT_TEXT_OVERLAP_SMALL_WITH_MCPTox 0
```

However, the shipped Qwen artifact trains on MCPTox:

- `mcptox_clean_descriptions_labeled.json` contributes 342 training rows (`intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2/metadata.json:70`).
- `mcptox_poisoned_descriptions_labeled.json` contributes 485 training rows (`intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2/metadata.json:71`).

Adversarial conclusion: the n=86 GPT-2 headline is not MCPTox-sourced, but the shipped product artifact's MCPTox behavior is in-training unless evaluated on a separate holdout.

## Finding 7 - Medium: The Opponent Wording Is Internally Inconsistent

The README says "Snyk's shipped scanner catches 0%" (`README.md:22`). The benchmark script says the compared opponent is `protectai/deberta-v3-base-prompt-injection-v2` (`research/benchmark-probe-vs-deberta-samesplit.py:24`). The older benchmark result calls that "the exact classifier Snyk Agent-Scan ships" (`research/benchmark-results-deberta-vs-probe-2026-05-31.md:3`).

But the competitive landscape doc is more cautious:

- It says Snyk's public client exposes API validation and the remote detector is opaque from the user's machine (`docs/COMPETITIVE_LANDSCAPE.md:60`, `docs/COMPETITIVE_LANDSCAPE.md:62`).
- It says the public repo does not provide a user-reproducible benchmark proving tool-poisoning accuracy on these matched cases (`docs/COMPETITIVE_LANDSCAPE.md:63`, `docs/COMPETITIVE_LANDSCAPE.md:65`).
- It frames the head-to-head as ProtectAI DeBERTa "Snyk/Invariant-style text-classifier baseline" (`docs/COMPETITIVE_LANDSCAPE.md:67`, `docs/COMPETITIVE_LANDSCAPE.md:69`).

Adversarial conclusion: unless there is current source proof that Snyk's actual shipped backend decision is exactly this DeBERTa threshold, say "ProtectAI DeBERTa prompt-injection classifier baseline" or "Snyk/Invariant-style DeBERTa baseline", not "Snyk's shipped scanner."

## Product Sanity Check: Shipped Qwen Artifact On Its Training Pools

I also ran the packaged Qwen artifact locally on the small n=86 pool and MCPTox. This is **not held-out evidence** because the artifact metadata includes those files as training sources (`intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2/metadata.json:68`, `intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2/metadata.json:71`).

```text
SCAN_DONE small_n86_training_pool n= 86 seconds= 13.07 labels= {0: 43, 1: 43}
PRODUCT_METRIC small_n86_training_pool warn_or_block_decision acc= 0.9419 prec= 0.8958 recall= 1.0000 f1= 0.9451 pred_pos= 48
PRODUCT_METRIC small_n86_training_pool block_decision acc= 0.5349 prec= 1.0000 recall= 0.0698 f1= 0.1304 pred_pos= 3
PRODUCT_METRIC small_n86_training_pool activation_ge_warn030 acc= 1.0000 prec= 1.0000 recall= 1.0000 f1= 1.0000 pred_pos= 43
PRODUCT_METRIC small_n86_training_pool activation_ge_block085 acc= 1.0000 prec= 1.0000 recall= 1.0000 f1= 1.0000 pred_pos= 43
DECISIONS small_n86_training_pool {'allow': 38, 'warn': 45, 'block': 3}

SCAN_DONE mcptox_training_pool n= 827 seconds= 252.1 labels= {0: 342, 1: 485}
PRODUCT_METRIC mcptox_training_pool warn_or_block_decision acc= 1.0000 prec= 1.0000 recall= 1.0000 f1= 1.0000 pred_pos= 485
PRODUCT_METRIC mcptox_training_pool block_decision acc= 0.4353 prec= 1.0000 recall= 0.0371 f1= 0.0716 pred_pos= 18
PRODUCT_METRIC mcptox_training_pool activation_ge_warn030 acc= 1.0000 prec= 1.0000 recall= 1.0000 f1= 1.0000 pred_pos= 485
PRODUCT_METRIC mcptox_training_pool activation_ge_block085 acc= 1.0000 prec= 1.0000 recall= 1.0000 f1= 1.0000 pred_pos= 485
DECISIONS mcptox_training_pool {'allow': 342, 'warn': 467, 'block': 18}
```

This supports two launch cautions:

- If "catches" means warn/review, Qwen catches these training-pool examples.
- If "catches" means hard block, the shipped block tier catches only 3/43 small-pool poisoned rows and 18/485 MCPTox poisoned rows, because hard block needs static corroboration.

## Recommended Public Rescope

Replace the README headline with something closer to:

> A GPT-2 activation-probe research benchmark reaches 96.6% F1 on a small synthetic same-topic tool-poisoning pool where a DeBERTa prompt-injection classifier baseline gets 0% recall. The packaged Qwen scanner is a local preview: activation scores power the warn/review tier, while hard block is deliberately conservative and requires static corroboration.

For product metrics, use the Qwen calibration doc rather than the GPT-2 research headline:

- RouteGuard external warn/review F1 is 0.747 with recall 0.800 (`research/THRESHOLD_CALIBRATION_QWEN_POOLED_2026-06-03.md:67`).
- RouteGuard external block F1 is 0.071 with precision 1.000 and recall 0.037 (`research/THRESHOLD_CALIBRATION_QWEN_POOLED_2026-06-03.md:68`).
- The product doc already says hard-block recall is preview-only and not solved (`research/THRESHOLD_CALIBRATION_QWEN_POOLED_2026-06-03.md:81`, `research/THRESHOLD_CALIBRATION_QWEN_POOLED_2026-06-03.md:85`).

## Final Answer To The Five Audit Questions

1. **Which model produces 96.5% / 96.6% F1 / 100% MCPTox?** The headline same-split numbers are GPT-2 layer-3 research-probe numbers (`research/benchmark-probe-vs-deberta-samesplit.py:3`, `research/benchmark-probe-vs-deberta-samesplit.py:8`, `research/benchmark-probe-vs-deberta-samesplit.py:135`). The shipped artifact is Qwen2.5-0.5B (`intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2/metadata.json:141`). Critical mismatch.

2. **Is there CV leakage?** Yes. The n=86 CV uses shuffled row-level `StratifiedKFold` (`research/benchmark-probe-vs-deberta-samesplit.py:129`, `research/benchmark-probe-vs-deberta-samesplit.py:132`). I counted 8 exact duplicate groups involving 16 rows and 68/86 row-level test placements with the paired opposite-label item in train. Rerun F1 dropped from 0.9655 to 0.9425 grouped, or 0.9500 exact-dedup grouped.

3. **Does shipped BLOCK rely on activations alone?** No. Every `block` path requires static regex evidence: either a static-only bundle (`intentprobe/scanner/core.py:286`, `intentprobe/scanner/core.py:289`) or activation plus static corroboration (`intentprobe/scanner/core.py:291`, `intentprobe/scanner/core.py:300`). Activation-only high scores downgrade to warn (`intentprobe/scanner/core.py:302`, `intentprobe/scanner/core.py:307`).

4. **How many true matched-vocabulary minimal pairs exist in `hard_v3_matched_*`?** Strict same-vocabulary count: 0. Obvious same-topic near-minimal count: 1 SSH pair (`research/datasets/hard_v3_matched_clean.json:31`, `research/datasets/hard_v3_matched_poisoned.json:23`). The files are mostly different topics (`research/datasets/hard_v3_matched_clean.json:2`, `research/datasets/hard_v3_matched_poisoned.json:2`). The n=86 pool is not sourced from MCPTox (`research/benchmark-probe-vs-deberta-samesplit.py:123`, `research/benchmark-probe-vs-deberta-samesplit.py:126`), and exact-text overlap with MCPTox is 0 in my audit run.

5. **Is the shipped artifact's metric held-out or train? Is matched in training?** Train metric. `train_probe()` fits and predicts on the same matrix (`research/train_probe_artifact.py:194`, `research/train_probe_artifact.py:195`) and stores `train_metrics` (`research/train_probe_artifact.py:201`). Matched and MCPTox files are in the shipped artifact's training summary (`intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2/metadata.json:68`, `intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2/metadata.json:71`).

