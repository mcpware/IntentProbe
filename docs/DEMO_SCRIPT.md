# Demo Script

This is the no-video demo. It gives people the same proof a terminal video
would show, but it is easier to reproduce and harder to dismiss as editing.

## Local CLI demo

```bash
python3 -m pip install intentprobe
git clone https://github.com/mcpware/IntentProbe.git
cd IntentProbe
```

Scan the safe package fixture:

```bash
intentprobe scan-path research/fixtures/scan_path/safe-weather-package \
  --format summary \
  --fail-on block
```

Expected result: both candidate files are `allow`.

Scan the poisoned skill fixture:

```bash
intentprobe scan-path research/fixtures/scan_path/poisoned-skill \
  --format summary \
  --fail-on block
```

Expected result: `SKILL.md` is `block`, and the command exits with code `2`.
The evidence should mention both the activation score and static findings for
private keys plus exfiltration.

## GitHub Action demo

Use this workflow in a repo that contains MCP configs, skills, or tool
manifests:

```yaml
name: IntentProbe scan

on:
  pull_request:
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

Live proof repo:

- Demo repo: https://github.com/mcpware/intentprobe-demo
- Safe PR passes: https://github.com/mcpware/intentprobe-demo/pull/1
- Poisoned PR is blocked: https://github.com/mcpware/intentprobe-demo/pull/2

For a narrower scan:

```yaml
          paths: |
            .mcp.json
            mcp.json
            mcp/**/*.json
            skills/**
            packages/**/package.json
```

## What this proves

- The scanner can be installed by a stranger.
- Safe MCP-style package docs pass.
- A poisoned skill that reads private keys and uploads them gets blocked.
- The same scanner can run as a GitHub Action gate before a pull request
  merges.
- Scan targets and results stay inside the local machine or GitHub runner.
- CI can pass an optional Hugging Face secret so first-run model downloads do
  not depend on anonymous rate limits.

## What this does not prove

- It does not prove every poisoned tool will be caught.
- It does not replace sandboxing, permissions, code review, or runtime controls.
- It does not prove enterprise readiness without more deployment calibration.

The point of v0.2 is product proof: CLI, CI gate, runtime hook, reproducible
benchmarks, and a clean feedback loop for missed detections.
