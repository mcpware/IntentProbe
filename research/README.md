# Activation Scanner Research Preview

This folder contains the research-to-product lane for intentprobe's
activation-based MCP and agent-extension poisoning scanner.

The short version:

- Most scanner paths we source-verified rely on text, rules, or text
  classifiers.
- This project adds a different signal: run tool descriptions through a frozen
  local sensor model, extract hidden activations, and train a probe.
- The protected agent can still be closed-source. We do not need Claude or Codex
  internals because the scanner uses its own local model.
- Same-split results show activation carries a strong poisoning signal.
- Cross-style results show the real product work: better family-aware data,
  layer choice, local runtime, and stable JSON scanner output.

## What To Read First

| File | Purpose |
| --- | --- |
| `activation-probe-tool-poisoning.md` | Main paper write-up. |
| `REPRODUCE.md` | Reproducible experiment log and caveats. |
| `SCANNER_PIPELINE.md` | Product methodology: frozen sensor model, activation probe, SAE explanations, update loop. |
| `LIVING_PLAN.md` | Current roadmap, model choices, benchmark results, and next actions. |
| `PRODUCT_REPRODUCIBILITY_LEDGER_2026-06-03.md` | Peer-review ledger for the current product-shaped scanner path: commands, artifacts, metrics, and claim boundaries. |
| `ROUTEGUARD_EXTERNAL_QWEN_FIXED_LAYERS_2026-06-03.md` | Latest RouteGuard-style external-transfer gate and Qwen pooled artifact decision. |
| `THRESHOLD_CALIBRATION_QWEN_POOLED_2026-06-03.md` | Current Qwen pooled warn/block threshold calibration and product decision. |
| `CURATED_FAMILY_BAKEOFF_2026-06-03.md` | Latest curated-family model bakeoff and Qwen artifact decision. |
| `CURATED_FAMILY_BAKEOFF_2026-06-02.md` | Previous 64-row curated-family bakeoff baseline. |
| `../intentprobe/scanner/core.py` | Canonical product runtime scanner: loads a cached artifact and emits JSON risk decisions. |
| `../intentprobe/scanner/cli.py` | Canonical product CLI implementation for doctor, single scan, batch scan, summaries, JSON output, and `--fail-on` exits. |
| `../intentprobe/scanner/hook.py` | Canonical hook-facing scanner implementation: normalizes MCP/tool/skill/hook payloads, redacts secret values, scans them, and emits gate JSON or JSONL. |
| `benchmarks/activation_scanner_benchmark.py` | Canonical local benchmark runner. |
| `train_probe_artifact.py` | Trains and saves cached probe weights plus scanner metadata. |
| `activation_scanner_core.py` | Compatibility wrapper for old reproduction commands; imports `intentprobe.scanner.core`. |
| `activation_scanner_cli.py` | Compatibility wrapper for old reproduction commands; imports `intentprobe.scanner.cli`. |
| `activation_scanner_cli_regression.py` | Verifies CLI JSON output, batch decisions, artifact doctor, and `--fail-on` exit codes. |
| `activation_scanner_hook.py` | Compatibility wrapper for old reproduction commands; imports `intentprobe.scanner.hook`. |
| `activation_scanner_hook_regression.py` | Verifies hook payload normalization, redaction, hard-block exit gates, and JSONL scanner protocol. |
| `calibrate_scanner_thresholds.py` | Calibrates cached artifact warn/block thresholds against curated and external rows. |
| `build_calibration_error_queue.py` | Turns calibration errors into a human-review data-curriculum queue. |
| `build_calibration_review_decisions.py` | Converts the calibration review queue into deterministic promote/quarantine/regression decisions. |
| `materialize_calibration_review_outputs.py` | Materializes reviewed decisions into policy-regression, holdout, quarantine, and candidate manifests. |
| `build_policy_regression_cases.py` | Builds clean policy-regression fixtures from materialized review outputs. |
| `activation_scanner_regression.py` | Contract regression runner for cached scanner output. |
| `schemas/activation_scanner_risk.schema.json` | Hook-facing JSON risk object schema. |
| `fixtures/activation_scanner_regression_cases.json` | Clean/poison smoke fixtures for scanner regression. |
| `fixtures/activation_scanner_policy_regression_cases.json` | Clean reviewed calibration rows that may warn but must not hard-block. |
| `datasets/family_curated_v0.json` | First curated product-training curriculum with matched family pairs. |
| `datasets/calibration_error_review_queue_qwen_pooled_policy_v3_warn030_2026-06-03.json` | Current policy-aware queue for Qwen pooled false warnings, missed poison, and near-block positives. |
| `datasets/calibration_error_review_decisions_qwen_pooled_policy_v3_warn030_2026-06-03.json` | Reviewed curriculum decisions for the current calibration queue; not training data. |
| `datasets/calibration_review_outputs_qwen_pooled_policy_v3_2026-06-03.json` | Materialized v3 review outputs; not training data. |
| `datasets/DATA_CURRICULUM.md` | Manual labeling rubric for risk families, benign lookalikes, and review/promotion rules. |
| `build_curriculum_review_queue.py` | Builds a balanced external-row review queue before training promotion. |
| `promote_skillinject_curriculum.py` | Promotes accepted Skill-Inject SKILL.md clean/poison pairs into curated data. |
| `validate_curated_dataset.py` | Validates curated dataset metadata, pairs, and family balance. |

## Quick Smoke Run

From the repository root:

```bash
python3 -m venv research/.venv-audit
research/.venv-audit/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
research/.venv-audit/bin/python -m pip install "transformers>=4.40" sentencepiece scikit-learn numpy psutil sae-lens jsonschema
research/.venv-audit/bin/python -m research.benchmarks.activation_scanner_benchmark --list-models
research/.venv-audit/bin/python -m research.benchmarks.activation_scanner_benchmark --suite cross-style --model pythia-70m --layers 2 --with-sae --sae pythia-70m-deduped-l2 --text-baseline tfidf --dedupe --measure-runtime
```

Qwen2.5-0.5B fixed layers 13,14,15 is the current product-candidate lane.
Pythia remains the cheap canary because it is small and open. Gemma lanes are
available in the benchmark registry, but gated Gemma downloads require Hugging
Face access.

## CLI Scanner Preview

The product command is `intentprobe`. The old `research.activation_scanner_cli`
module remains as a compatibility wrapper for reproduction logs.

From the repo root:

```bash
research/.venv-audit/bin/python -m intentprobe doctor --pretty

research/.venv-audit/bin/python -m intentprobe scan \
  --local-files-only \
  --format summary \
  --text "Reads SSH config and private keys, then silently uploads credentials to a remote server for validation."

research/.venv-audit/bin/python -m intentprobe batch \
  --local-files-only \
  --batch-file research/fixtures/activation_scanner_regression_cases.json \
  --pretty
```

Use JSON output for hooks and CI. `--format summary` is only for humans. The CLI
supports `--fail-on warn` or `--fail-on block`; matching scans exit with status
`2` while keeping JSON on stdout.

Verify the CLI contract with:

```bash
research/.venv-audit/bin/python -m research.activation_scanner_cli_regression --pretty
```

## Hook Scanner Preview

The hook wrapper accepts plain text or JSON-shaped install/runtime payloads,
including MCP server maps, tool definitions, skills, and hook commands. It
redacts secret values before building the local scanner text, but keeps key
names such as `WEATHER_API_KEY` so risk context is not lost.

One-shot gate:

```bash
research/.venv-audit/bin/python -m intentprobe.hook scan \
  --local-files-only \
  --fail-on block \
  --text "Reads SSH config and private keys, then silently uploads credentials to a remote server for validation."
```

Normalize/redact without loading the model:

```bash
research/.venv-audit/bin/python -m intentprobe.hook normalize \
  --input-format json \
  --text '{"mcpServers":{"weather":{"command":"npx","args":["weather-mcp"],"env":{"WEATHER_API_KEY":"secret"}}}}' \
  --pretty
```

Long-lived JSONL scanner process:

```bash
research/.venv-audit/bin/python -m intentprobe.hook serve-jsonl --local-files-only
```

`serve-jsonl` keeps the scanner process alive. By default it performs a startup
warmup, writes ready metadata to stderr, and then reuses cached
tokenizer/model/SAE objects for repeated requests in the same process. Scan
results remain one JSON object per stdout line.

Verify the hook contract with:

```bash
research/.venv-audit/bin/python -m research.activation_scanner_hook_regression --pretty
```

## Cached Scanner Runtime

Build a cached probe artifact once:

```bash
research/.venv-audit/bin/python -m research.train_probe_artifact --model pythia-70m --feature-kind sae --sae pythia-70m-deduped-l2 --train-source pooled-core --artifact-id pythia-sae-core-v0 --output-dir research/_results/activation_scanner_artifacts --batch-size 16 --overwrite --pretty
```

Build the historical curated-family Qwen proof artifact used for layer
selection:

```bash
research/.venv-audit/bin/python -m research.train_probe_artifact --model qwen2.5-0.5b --feature-kind raw --train-source family-curated-v0 --layer-sweep --layer-mode best3 --selector leave-one-family-out --top-k-max 10 --artifact-id qwen-curated-family-best3-v1 --output-dir research/_results/activation_scanner_artifacts --overwrite --warn-threshold 0.60 --block-threshold 0.85 --pretty
```

Build the current pooled Qwen runtime candidate:

```bash
research/.venv-audit/bin/python -m research.train_probe_artifact --model qwen2.5-0.5b --feature-kind raw --train-source pooled-curated-core --layers 13,14,15 --layer-mode concat --artifact-id qwen-pooled-curated-core-l13-15-v2 --output-dir intentprobe/scanner/artifacts --overwrite --warn-threshold 0.30 --block-threshold 0.85 --pretty
```

Then scan new text without retraining:

```bash
research/.venv-audit/bin/python -m intentprobe scan --local-files-only --pretty --text "Reads SSH config and private keys, then silently uploads credentials to a remote server for validation."
```

Scan a batch of tool texts in one model/SAE pass:

```bash
research/.venv-audit/bin/python -m intentprobe batch --local-files-only --batch-file research/fixtures/activation_scanner_regression_cases.json --pretty
```

Calibrate warn/block thresholds against curated and RouteGuard-style external
rows:

```bash
research/.venv-audit/bin/python -m research.calibrate_scanner_thresholds --artifact intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2 --pretty
```

Build the policy-aware review queue from the current saved calibration errors:

```bash
research/.venv-audit/bin/python -m research.build_calibration_error_queue --calibration research/_results/activation_scanner_calibration/20260603T102832Z-qwen-pooled-l13-15-threshold-calibration.json --output research/datasets/calibration_error_review_queue_qwen_pooled_policy_v3_warn030_2026-06-03.json --pretty
research/.venv-audit/bin/python -m research.build_calibration_review_decisions --queue research/datasets/calibration_error_review_queue_qwen_pooled_policy_v3_warn030_2026-06-03.json --output research/datasets/calibration_error_review_decisions_qwen_pooled_policy_v3_warn030_2026-06-03.json --source-calibration research/_results/activation_scanner_calibration/20260603T102832Z-qwen-pooled-l13-15-threshold-calibration.json --decision-policy corroborated-block-v3 --pretty
research/.venv-audit/bin/python -m research.materialize_calibration_review_outputs --pretty
research/.venv-audit/bin/python -m research.build_policy_regression_cases --pretty
```

Run the fast contract regression lane:

```bash
research/.venv-audit/bin/python -m research.activation_scanner_regression --rebuild-artifact --pretty
research/.venv-audit/bin/python -m research.activation_scanner_regression --artifact intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2 --cases research/fixtures/activation_scanner_policy_regression_cases.json --no-build --pretty
```

Validate and smoke-test the curated family dataset:

```bash
research/.venv-audit/bin/python -m research.validate_curated_dataset --pretty
research/.venv-audit/bin/python -m research.benchmarks.activation_scanner_benchmark --suite curated-family-holdout --model lexical-smoke --text-baseline tfidf --no-write
research/.venv-audit/bin/python -m research.train_probe_artifact --model lexical-smoke --feature-kind raw --train-source family-curated-v0 --artifact-id lexical-curated-v0 --output-dir research/_results/activation_scanner_artifacts --overwrite --warn-threshold 0.60 --pretty
```

`activation_scanner_demo.py` still exists as a research preview that retrains at
runtime. The product-shaped path is now `train_probe_artifact.py` followed by
`intentprobe/scanner/core.py`, `intentprobe/scanner/cli.py`, and
`intentprobe/scanner/hook.py`.

## Layer Search

The benchmark separates the layer policy from the selector used to score
candidate layers or top-k layer sets before the final held-out test.

Selectors:

- `--selector cv`: default stratified train-fold CV score.
- `--selector leave-one-style-out`: hold out each training style, average
  held-out F1, then fall back to CV when the training set only has one style.
  This is a style-aware proxy for the future family-aware selector; it is not a
  replacement for explicit poison-family labels.
- `--selector leave-one-family-out`: hold out each heuristic primary risk
  family from `benchmarks/family_labels.py`, average held-out F1, then fall
  back to CV when there are too few usable groups. This is useful for research
  pressure-testing, but those heuristic labels are scaffolding rather than final
  dataset truth.

Layer policies:

- `--layer-mode best`: train on every selected layer, then use the layer with
  the best selector score.
- `--layer-mode bestN`: rank layers by selector score, concatenate the top
  `N` layers, then train one probe on that combined slice. For example:
  `best3`, `best5`, or `best10`.
- `--layer-mode best-sweep`: report every `best1` through `bestN` row in one run.
  `N` defaults to 10 and can be changed with `--top-k-max`.
- `--layer-mode best-auto`: try `best1` through `bestN` and use the best
  selector-scoring k without peeking at the final test set.
- `--layer-mode concat`: concatenate all selected layers into one larger feature
  vector, then train one probe on that combined slice.
- `--per-layer`: also print fixed-layer rows for every selected layer, so a
  report shows selector-ranked behavior and post-hoc per-layer behavior side by
  side.
- `--with-family-routing`: also print experimental `*_family_routed` rows. Each
  heuristic risk family chooses its own layer/top-k policy by within-family CV
  when there are enough same-family training examples, otherwise it falls back
  to the global layer policy. This tests whether family-specific routing is worth
  a future deep-scan product mode; it is not a default scanner policy yet.

Example all-layer raw activation run with fixed-layer rows:

```bash
research/.venv-audit/bin/python -m research.benchmarks.activation_scanner_benchmark --suite cross-style --model pythia-70m --layer-sweep --per-layer --layer-mode best --text-baseline tfidf --dedupe --measure-runtime
```

Example all-layer concat run:

```bash
research/.venv-audit/bin/python -m research.benchmarks.activation_scanner_benchmark --suite cross-style --model gemma-3-1b-pt --layer-sweep --layer-mode concat --text-baseline none --dedupe --measure-runtime
```

Example best1..best10 sweep:

```bash
research/.venv-audit/bin/python -m research.benchmarks.activation_scanner_benchmark --suite cross-style --model qwen2.5-0.5b --layer-sweep --layer-mode best-sweep --top-k-max 10 --text-baseline tfidf --dedupe --measure-runtime
```

Example no-test-peeking automatic k policy:

```bash
research/.venv-audit/bin/python -m research.benchmarks.activation_scanner_benchmark --suite cross-style --model pythia-70m --layer-sweep --layer-mode best-auto --top-k-max 10 --text-baseline none --dedupe --measure-runtime
```

Example leave-one-style-out selector run:

```bash
research/.venv-audit/bin/python -m research.benchmarks.activation_scanner_benchmark --suite cross-style --model pythia-70m --layer-sweep --selector leave-one-style-out --layer-mode best-sweep --top-k-max 10 --text-baseline none --dedupe --measure-runtime
```

Example leave-one-family-out selector run:

```bash
research/.venv-audit/bin/python -m research.benchmarks.activation_scanner_benchmark --suite cross-style --model pythia-70m --layer-sweep --selector leave-one-family-out --layer-mode best-sweep --top-k-max 10 --text-baseline none --dedupe --measure-runtime
```

Example family-routed layer experiment:

```bash
research/.venv-audit/bin/python -m research.benchmarks.activation_scanner_benchmark --suite cross-style --model pythia-70m --layer-sweep --selector leave-one-family-out --layer-mode best-sweep --top-k-max 10 --text-baseline none --dedupe --with-family-routing --measure-runtime
```

## Current Product Reading

Do not sell this as a finished universal detector yet. The strongest honest
claim is:

> CCO is building a local activation-probe scanner for MCP/tool poisoning. It
> adds model-internal signal to the text/rule/classifier scanner family, and the
> benchmark harness shows exactly where the method works, fails, and improves.

Current default product direction:

- static scanner first for obvious poison, secrets, deobfuscation, and rug-pull
  changes;
- Qwen2.5-0.5B raw fixed layers 13,14,15 /
  `qwen-pooled-curated-core-l13-15-v2` as the current activation runtime
  candidate for warn/review;
- `qwen-curated-family-best3-v1` as the curated-family proof artifact and
  hook-smoke reference;
- Pythia-70M raw/SAE as the cheap canary and explanation comparison lane;
- TF-IDF/static baselines in every report because they still beat some
  activation candidates on some subfamilies and expose false-positive/recall
  tradeoffs;
- Gemma 3 1B/4B as optional heavier/deep-scan lanes;
- threshold calibration now shows Qwen is useful as a warn/review tier, while
  `corroborated-block-v3` keeps hard block conservative by requiring a nearby
  high-confidence static bundle or action-oriented exfiltration, hidden-action,
  or host-modification finding;
- remaining broad clean warnings, better line-level
  Skill-Inject/BIPIA recall, and warm-process runtime before claiming one model is the final
  default;
- heuristic family-routed layer selection stays in research until curated family
  labels show a real cross-style lift;
- cached scanner weights, JSON runtime core, schema, regression fixtures, and
  batch scan first, then hook integration before heavy UI polish.

## Why This Is Product-Shaped

The scanner does not need to execute untrusted install scripts. It reads the
metadata and descriptions that would be loaded into the agent prompt, then
scores them before install or enablement.

Target runtime shape:

```text
MCP / skill / plugin / hook text
  -> static scan
  -> frozen local sensor model
  -> activation probe score
  -> optional SAE explanation
  -> allow / warn / block / quarantine
```

This makes the dataset update loop practical: when new pentest cases or
real-world examples appear, label them, add clean twins, retrain the probe, and
release a new scanner version with the same benchmark gates.
