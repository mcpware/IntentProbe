# Release Checklist

This checklist is the local gate before telling strangers to download and try
intentprobe.

## Clean checkout

```bash
git status --short --branch
python -m pip install -e .
```

## CLI smoke tests

```bash
intentprobe doctor --pretty
intentprobe scan --format summary --local-files-only \
  --text "Reads SSH config and private keys, then silently uploads credentials to a remote server."
intentprobe scan-path research/fixtures/scan_path/poisoned-skill \
  --local-files-only --format summary --fail-on block
intentprobe-hook normalize --input-json '{"tool":{"name":"demo","description":"Adds two numbers."}}'
```

The poisoned scan-path command should exit with code `2`.

## Regression suites

```bash
python -m research.activation_scanner_cli_regression --pretty
python -m research.activation_scanner_hook_regression --pretty
python -m research.activation_scanner_regression \
  --artifact intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2 \
  --no-build --pretty
python -m research.activation_scanner_regression \
  --artifact intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2 \
  --cases research/fixtures/activation_scanner_policy_regression_cases.json \
  --no-build \
  --pretty
```

## Package build

```bash
python -m pip install build
python -m build --sdist --wheel
python -m zipfile -l dist/intentprobe-0.1.0-py3-none-any.whl | \
  rg 'probe_weights|metadata.json|targets.py|entry_points'
python -m tarfile -l dist/intentprobe-0.1.0.tar.gz | \
  rg 'probe_weights|metadata.json|targets.py|SECURITY.md|SAMPLE_REPORTING|RELEASE_CHECKLIST'
```

## Hygiene

```bash
git diff --check
git diff --cached --check
git diff --cached | rg -n 'hf_[A-Za-z0-9]{20,}|gh[oprsu]_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{35}|AKIA[0-9A-Z]{16}'
rg -n --hidden --glob '!research/datasets/**' --glob '!build/**' \
  --glob '!dist/**' --glob '!intentprobe.egg-info/**' --glob '!.git/**' \
  'hf_[A-Za-z0-9]{20,}|gh[oprsu]_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{35}|AKIA[0-9A-Z]{16}' .
```

The staged secret scan should return no matches. The repo-wide scan excludes
`research/datasets/` because those files intentionally contain synthetic and
public-example secret-like strings used as scanner data; any hit outside that
directory needs manual triage.

## Launch claim boundary

Safe to say:

- intentprobe is a local research-preview scanner.
- it uses model activations, not only text patterns.
- it can scan text, package folders, MCP configs, and Claude Code skill folders.
- current benchmarks show strong wins on matched-vocabulary tool poisoning.
- novel attack-family generalization is still the open frontier.

Do not say:

- "production proven".
- "catches all poisoned tools".
- "zero false positives".
- "replaces sandboxing, permissions, or code review".
