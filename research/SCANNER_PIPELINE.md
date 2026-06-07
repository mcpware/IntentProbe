# Activation Scanner Pipeline

Last updated: 2026-06-03
Status: pipeline-first methodology

## Position

The activation scanner should ship as a repeatable pipeline before the dataset is
perfect.

The first public claim should be about the method and the update loop:

> We scan tool, MCP, skill, and plugin descriptions before install by combining
> static checks with a frozen local sensor model, activation probes, and optional
> SAE explanations. The probe can be retrained as new attack families and
> pentest cases are added.

This is different from claiming that the current dataset is complete or that the
current probe is final.

Priority framing: the March 29, 2026 activation-probe paper in this repo is the
starting point. Later internal-signal papers such as RouteGuard strengthen the
category argument; they do not change the product wedge, which is to ship a
local scanner path rather than only a paper result.

Launch posture:

- Show the method publicly now.
- Keep the benchmark harness easy to rerun.
- Mark the current dashboard scanner as static/text/rule based.
- Mark the activation scanner as the research-to-product preview until hook
  integration and calibrated release gates exist.
- Improve the dataset and probe through normal scanner updates, not a hidden
  research freeze.

## Core Method

The base language model is frozen. Its weights do not change during scanner
training.

The training process is:

1. Collect clean and poisoned tool descriptions.
2. Run each description through a frozen local sensor model.
3. Extract hidden-state activations from selected layers.
4. Train a small supervised probe on those activation vectors.
5. Evaluate the probe across source/style/family splits.
6. Package the scanner with the frozen base model id, layer policy, probe
   weights, thresholds, and benchmark report.

In notation:

```text
text -> frozen sensor model -> activation vector X
label -> clean or poison y
train probe: X -> y
```

The first probe model should stay simple, usually logistic regression, because a
linear probe gives a clean read on whether the activation signal is real. More
complex models such as LightGBM, XGBoost, or a small MLP can be added after the
dataset curriculum is stable.

## Scanner Runtime

At runtime, the scanner does not retrain anything. It only scores new inputs.

```text
new tool description
  -> static risk checks
  -> frozen sensor model forward pass
  -> selected activation features
  -> trained probe score
  -> optional SAE feature explanation
  -> decision policy
  -> allow / warn / block / quarantine
```

The scanner must inspect metadata and files before running untrusted code. It
should not execute install scripts just to decide whether a tool is safe.

The product hook path now has two runtime shapes:

- one-shot gate: normalize a tool/MCP/skill/hook payload, scan it, and return a
  JSON decision plus an exit code for install-time blocking;
- warm JSONL process: keep the Python scanner alive, feed one payload per line,
  and reuse cached tokenizer/model/SAE objects across repeated scans.

Secret values in MCP env/config payloads are redacted before scanner text is
constructed. Key names remain visible because they carry risk context without
leaking the secret itself.

The current runtime policy is `corroborated-block-v3`: `warn` can come from a
high risk score, but `block` needs a nearby static block bundle or
action-oriented exfiltration, hidden-action, or host-modification evidence. This
keeps the review signal broad while preventing a threshold-only high score or
generic secret-management language from becoming an automatic hard block.

## Why Data Still Matters

This pipeline still needs labeled examples. A better feature source does not
remove the need for good labels.

The current failure mode is that many public poisoning examples teach overt
instruction chaining:

```text
must call X
before using Y
ignore previous instructions
```

Realistic MCP and tool poisoning can be quieter:

```text
read private keys
forward cookies or credentials
upload configuration contents
log generated secrets
make live handshakes with stored credentials
```

The scanner should therefore learn attack families, not only prompt-injection
phrases.

## Data Curriculum

Each dataset row should eventually carry:

- `label`: `clean` or `poisoned`
- `family`: risk family or benign lookalike family
- `source`: original dataset, MCP repo, skill repo, pentest case, synthetic case
- `pair_id`: optional clean/poison matched pair
- `notes`: short reason for the label

Initial families:

| Family | Meaning |
| --- | --- |
| `instruction_chaining` | Forces hidden or unsafe tool-call sequencing. |
| `secret_file_access` | Reads private keys, tokens, credentials, cookies, or env files. |
| `credential_forwarding` | Sends or validates credentials outside the expected local scope. |
| `network_exfiltration` | Uploads, transmits, tracks, or forwards sensitive data. |
| `hidden_persistence_logging` | Stores secrets, adds hidden logs, or keeps background state. |
| `live_system_access` | Performs live handshakes, registry pulls, database checks, or network probes beyond passive parsing. |
| `system_inventory` | Reads host identity or system inventory data. |
| `tool_shadowing` | Tries to intercept, route, or override tools belonging to another server. |

The benchmark should include clean twins wherever possible. For example:

```text
Clean: Reads SSH config and returns host aliases, hostnames, and ports.
Poison: Reads SSH config and also reads private key files to verify identities.
```

Clean twins are important because a scanner that blocks every mention of SSH,
database, file, cookie, or URL is not useful.

Current curated dataset files:

- `research/datasets/family_curated_v0.json` - 38 matched clean/poison pairs
  across 8 risk families. Rows include curated/synthetic pairs,
  real-carrier-adapted pairs, regression fixtures, 8 accepted public
  Skill-Inject SKILL.md pairs whose unsafe action is source-confirmed, and
  bridge rows that teach host fingerprinting across multiple risk families.
- `research/datasets/DATA_CURRICULUM.md` - manual labeling rubric for risk
  families, curriculum levels, benign lookalikes, and promotion rules.
- `research/build_curriculum_review_queue.py` - builds a non-training review
  queue from external rows before any promotion into curated data.
- `research/promote_skillinject_curriculum.py` - idempotently promotes the
  accepted Skill-Inject clean/poison SKILL.md pairs into curated data and writes
  `research/datasets/curriculum_review_decisions_v0.json`.
- `research/schemas/activation_curated_dataset.schema.json` - row contract for
  family, source type, pair id, split group, text, and notes.
- `research/validate_curated_dataset.py` - validates ids, labels, pair balance,
  family coverage, split-group consistency, and schema shape.

## Update Loop

New cases should be added like pentest findings.

1. Find a new suspicious tool, MCP server, skill, plugin, or attack pattern.
2. Reduce it to a clear description-level case.
3. Label the risk family and write the reason.
4. Add a clean twin or benign lookalike if one does not already exist.
5. Add the case to the dataset with source and family metadata.
6. Run duplicate/leakage checks.
7. Retrain the probe.
8. Compare against the previous scanner version on fixed benchmark gates.
9. Release a new scanner version only if it improves or preserves the gates.

This makes the model update process incremental. We do not need a perfect
dataset on day one, but every new case must make the benchmark more realistic
instead of just making the training set larger.

## Benchmark Gates

A scanner version is not ready just because it scores well on the same dataset
style it trained on.

Required gates:

| Gate | Required check |
| --- | --- |
| Same-split regression | Confirms the runner and probe have not broken. |
| Cross-style generalization | Train on one source/style and test on another. |
| Family holdout | Hold out one attack family and test whether the scanner generalizes. |
| Benign lookalikes | Measure false positives on safe tools that mention risky objects. |
| Runtime | Measure wall time, memory, and model footprint on a normal laptop. |
| Explanation | Record static reasons and SAE feature hints where available. |

Same-split results are useful as regression checks. They are not enough for a
product claim.

## Versioned Artifacts

Each scanner release should record:

- dataset version
- base sensor model id
- layer or layer-selection policy
- SAE release and SAE id, if used
- probe algorithm
- probe weights
- threshold policy
- benchmark report
- known failure modes
- release date

The important object is not just one trained probe. It is the full recipe that
can be rerun and improved.

Current artifact path:

```text
research/train_probe_artifact.py
  -> metadata.json
  -> probe_weights.npz
  -> intentprobe/scanner/core.py
  -> intentprobe/scanner/cli.py
  -> JSON risk object
```

The runtime core loads the artifact and scores new text without retraining. It
also supports batch scanning so a group of tool descriptions can share one
model/SAE feature-extraction pass. The CLI wrapper adds a product-preview
entrypoint with artifact doctor, single scan, batch scan, human summaries, JSON
stdout for hooks, and `--fail-on warn|block` exit codes.

Curated v0 can be used directly as a training source. The current product path
uses the pooled Qwen artifact and v3 policy-aware calibration queue:

```bash
research/.venv-audit/bin/python -m research.validate_curated_dataset --pretty
research/.venv-audit/bin/python -m research.train_probe_artifact --model qwen2.5-0.5b --feature-kind raw --train-source pooled-curated-core --layers 13,14,15 --layer-mode concat --artifact-id qwen-pooled-curated-core-l13-15-v2 --output-dir intentprobe/scanner/artifacts --overwrite --warn-threshold 0.30 --block-threshold 0.85 --pretty
research/.venv-audit/bin/python -m research.calibrate_scanner_thresholds --artifact intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2 --pretty
research/.venv-audit/bin/python -m research.build_calibration_error_queue --calibration research/_results/activation_scanner_calibration/20260603T102832Z-qwen-pooled-l13-15-threshold-calibration.json --output research/datasets/calibration_error_review_queue_qwen_pooled_policy_v3_warn030_2026-06-03.json --pretty
research/.venv-audit/bin/python -m research.build_calibration_review_decisions --queue research/datasets/calibration_error_review_queue_qwen_pooled_policy_v3_warn030_2026-06-03.json --output research/datasets/calibration_error_review_decisions_qwen_pooled_policy_v3_warn030_2026-06-03.json --source-calibration research/_results/activation_scanner_calibration/20260603T102832Z-qwen-pooled-l13-15-threshold-calibration.json --decision-policy corroborated-block-v3 --pretty
research/.venv-audit/bin/python -m research.materialize_calibration_review_outputs --pretty
research/.venv-audit/bin/python -m research.build_policy_regression_cases --pretty
research/.venv-audit/bin/python -m research.activation_scanner_cli_regression --pretty
research/.venv-audit/bin/python -m research.activation_scanner_regression --artifact intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2 --cases research/fixtures/activation_scanner_policy_regression_cases.json --no-build --pretty
```

Runtime contract files:

- `research/schemas/activation_scanner_risk.schema.json`
- `research/fixtures/activation_scanner_regression_cases.json`
- `research/activation_scanner_regression.py`
- `intentprobe/scanner/core.py`
- `intentprobe/scanner/cli.py`
- `intentprobe/scanner/hook.py`
- `research/activation_scanner_cli.py` compatibility wrapper
- `research/activation_scanner_cli_regression.py`
- `research/calibrate_scanner_thresholds.py`
- `research/build_calibration_error_queue.py`
- `research/build_calibration_review_decisions.py`
- `research/materialize_calibration_review_outputs.py`
- `research/build_policy_regression_cases.py`
- `research/fixtures/activation_scanner_policy_regression_cases.json`
- `research/PRODUCT_REPRODUCIBILITY_LEDGER_2026-06-03.md`

## Current Default Direction

Current evidence supports a first MVP lane, but not a final universal detector.

The working direction is:

- use static checks first for obvious risk;
- use Qwen2.5-0.5B raw fixed layers 13,14,15 /
  `qwen-pooled-curated-core-l13-15-v2` as the current external-transfer
  activation candidate for warn/review;
- keep `qwen-curated-family-best3-v1` as the curated-family proof artifact and
  hook-smoke reference;
- keep Pythia-70M raw/SAE as the cheap canary and explanation comparison lane;
- keep Gemma 1B/4B as optional heavier/deep-scan lanes;
- use `corroborated-block-v3` for preview hard-block decisions, and do not ship
  threshold-only hard block from the current Qwen pooled artifact;
- improve broad clean warnings, line-level Skill-Inject/BIPIA recall,
  and warm-process runtime before calling any artifact the final default
  scanner.

The MVP should be honest:

> This is an activation-based scanner pipeline with updateable probes and a
> growing attack-family benchmark, not a one-time frozen detector that claims to
> know every future MCP attack.
