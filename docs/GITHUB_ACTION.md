# GitHub Action

IntentProbe can run as a GitHub Action so a repository can scan MCP configs,
skills, and tool manifests before a pull request merges.

This is the v0.2 preview action. It installs the public PyPI package, runs
`intentprobe scan-path`, and fails the job when the verdict reaches your
configured `fail-on` level.

## Minimal workflow

Create `.github/workflows/intentprobe.yml`:

```yaml
name: IntentProbe scan

on:
  pull_request:
  push:
    branches: ["main"]
  workflow_dispatch:

jobs:
  scan-ai-tools:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: mcpware/IntentProbe@main
        with:
          paths: |
            .
          fail-on: block
          hf-token: ${{ secrets.HF_TOKEN }}
```

Use `paths` to narrow the scan once you know where your agent tooling lives:

```yaml
      - uses: mcpware/IntentProbe@main
        with:
          paths: |
            .mcp.json
            mcp.json
            mcp/**/*.json
            skills/**
            packages/**/package.json
          fail-on: block
          hf-token: ${{ secrets.HF_TOKEN }}
```

## Inputs

| Input | Default | Meaning |
|---|---:|---|
| `paths` | `.` | Newline-separated paths or shell globs to scan. |
| `fail-on` | `block` | Minimum decision that fails the job: `never`, `warn`, `block`, or `quarantine`. |
| `format` | `summary` | `summary` for humans, `json` for machine logs. |
| `intentprobe-version` | `0.1.4` | PyPI version installed by the action. |
| `python-version` | `3.11` | Python version for the runner. |
| `max-files` | `200` | Maximum candidate files read under each scanned directory. |
| `max-file-bytes` | `200000` | Maximum bytes read from each candidate file. |
| `local-files-only` | `false` | Set `true` only when the runner cache already has the model. |
| `hf-token` | empty | Optional Hugging Face token passed as a GitHub secret to avoid anonymous download rate limits. |

## Optional Hugging Face token

The scanner runs locally in the GitHub Actions runner, but the first
model-backed scan downloads Qwen2.5-0.5B from Hugging Face. Public runners can
hit anonymous rate limits, so CI workflows should add an `HF_TOKEN` repository
secret and pass it with `hf-token: ${{ secrets.HF_TOKEN }}`.

## What gets uploaded?

Nothing is uploaded to an IntentProbe server. There is no IntentProbe server.
The scanner runs inside the GitHub Actions runner. The first model-backed scan
downloads Qwen2.5-0.5B from Hugging Face and caches it for later runs.

## Exit behavior

`allow` passes. `warn` passes unless `fail-on: warn`. `block` fails when
`fail-on: block` or stricter. The scanner prints the same evidence chain as the
CLI: decision, activation score, static findings, thresholds, and reasons.
