# IntentProbe v0.2 Demo Pack Living Plan

Last updated: 2026-06-08 PT
Owner: Nicole + Codex
Status: active

This is the master plan for turning IntentProbe from a working scanner into a
buyer-grade proof pack. The public story is simple: IntentProbe is not another
text scanner. It is a local activation-probe admission gate for MCP servers,
agent tools, skills, and runtime tool events.

## What v0.2 has to prove

The bar is not "cool paper" or "nice README." The bar is:

> A stranger can install IntentProbe, scan real agent tooling, understand the
> verdict, and wire the result into CI or runtime policy without asking us.

If that works, the project starts looking less like a repo and more like a
category wedge: admission control for AI-agent capabilities.

## Current product truth

Done:

- One-command PyPI install: `python3 -m pip install intentprobe`.
- Local CLI scans for text, folders, MCP configs, and discovered local configs.
- Runtime JSONL hook that keeps the model warm and emits machine-readable
  allow/warn/block verdicts.
- Reproducible benchmark reports comparing activation probing with text
  classifier baselines.
- GitHub issue templates for missed detections and false positives.
- Public README, FAQ, evidence packet, operator decision docs, and competitive
  landscape.
- GitHub Action metadata and workflow docs added for CI gating.
- No-video demo script added for copy-paste CLI and Action proof.
- Action supports an optional Hugging Face token secret for reliable first-run
  model downloads in CI.
- Public demo repo added:
  https://github.com/mcpware/intentprobe-demo
- Live demo PRs added:
  safe PR passes, poisoned PR blocks.

Not done yet:

- No broad external user feedback loop yet.
- No GitHub Action Marketplace listing yet.
- No calibrated enterprise policy pack for allow/warn/redact/block/review.
- No public download or usage dashboard.
- No short video yet. This is deliberately optional, not a v0.2 blocker.

## v0.2 milestones

| Milestone | Status | Evidence target |
|---|---|---|
| M0. One-command install | Done | PyPI package installs and scans locally. |
| M1. CI gate | Done | `action.yml`, docs, optional HF token, and green action smoke: https://github.com/mcpware/IntentProbe/actions/runs/27186986017 |
| M2. No-video demo | Done | `docs/DEMO_SCRIPT.md` plus action smoke workflow. |
| M3. Demo repo | Done | https://github.com/mcpware/intentprobe-demo with safe PR #1 and poisoned PR #2. |
| M4. Runtime receipt demo | Next | `serve-jsonl` demo showing allow/warn/block receipts. |
| M5. Public challenge loop | Open | Issues convert misses and false positives into labeled samples. |
| M6. v0.2 release page | Open | Release notes with action usage, demo links, and benchmark links. |
| M7. Buyer/integration pack | Open | One-page artifact: problem, signal, traction, integrations, asks. |

## Immediate build sequence

1. Finish the GitHub Action and README/docs entrypoint.
2. Add and run an action smoke workflow:
   safe fixture passes, poisoned fixture blocks.
3. Push and verify the action metadata is visible from GitHub.
4. Create a tiny demo repository or fixture workflow:
   `safe-weather-mcp` passes, `credential-health-check` blocks.
5. Record exact command/output in a text demo script anyone can follow without
   knowing activation probing.
6. Cut a v0.2 release once the action, demo script, and smoke evidence are all
   in one place.

## Next 72 hours

- Make the CI install path copy-pasteable.
- Add a "scan this repo in GitHub Actions" section to README.
- Produce no-video proof: clone, install, scan safe, scan poisoned, show block,
  show JSON, and link the green action smoke.
- Reply to technical comments with concrete product behavior, not research
  abstraction.
- Track every external comment, star burst, install question, and failure case
  in a small launch log.

## Next 2 weeks

- Add a public demo repo and wire IntentProbe as a PR gate.
- Add Marketplace-facing metadata and docs when the action tag is cut.
- Build a small gallery of real MCP/skill scans: clean, warn, block.
- Add a calibration page that explains when to allow, warn, block, or review.
- Add a lightweight usage metric that does not collect scan contents.
- Publish one technical post aimed at security engineers and one simple post
  aimed at agent/MCP builders.

## Buyer-grade proof checklist

Technical proof:

- Reproducible benchmark commands and reports.
- One green GitHub Action run on a public demo repo.
- Runtime JSONL receipt with replayable decision evidence.
- Clear false-positive and missed-detection reporting loop.
- Small model, local scan, no customer content uploaded to us.

Market proof:

- Stars, forks, PyPI downloads, and action usage moving upward.
- External issues or comments from people scanning real tools.
- At least one integration conversation with an MCP/runtime/control-plane
  builder.
- At least one security person challenging the benchmark and getting a
  reproducible answer.

Acquirer proof:

- A new signal class: representation-level scanning, not another regex pack.
- A clear attach point: pre-install gate, CI gate, and runtime hook.
- A data flywheel: every miss becomes curriculum, every curriculum update
  improves the probe.
- A small artifact that can be embedded into a bigger security platform.

## Weekly operating loop

Monday: triage external comments, issues, false positives, missed detections.

Wednesday: update data curriculum, regression fixtures, and benchmark report.

Friday: ship one visible improvement: action, demo, docs, release, or
integration example.

Do not spend a week polishing wording while the product surface stands still.
The repo needs fresh proof more than perfect prose.

Video is optional. For scanners, a terminal video is weaker than a runnable
workflow because most scanner demos look the same. Prioritize reproducible proof
first: green CI, expected block, expected allow, clear JSON, and public issue
loop.

## Claim boundaries

Safe to say:

- Local activation-probe scanner for MCP/tool/skill poisoning.
- Reads model internals rather than only surface text.
- Reproducible public benchmarks show wins on matched-vocabulary tool
  poisoning.
- Usable as CLI, install-time scanner, CI gate, and runtime JSONL hook.

Do not say:

- Production proven.
- Catches every poisoned tool.
- Replaces sandboxing, permissions, code review, or runtime control planes.
- Enterprise-ready without calibration and operator policy work.

## Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-06-08 | Build GitHub Action before more posting. | CI turns IntentProbe from "try this CLI" into an admission gate people can wire into repos. |
| 2026-06-08 | Keep the public pack framed as a demo/evidence pack. | It still serves the acquisition path, but reads better to users, partners, and buyers. |
| 2026-06-08 | Defer video and prioritize runnable proof. | Security scanner videos are low-signal; action smoke, demo script, and reproducible commands create stronger trust. |
| 2026-06-08 | Add optional `hf-token` input to the Action. | The first public action smoke hit Hugging Face anonymous rate limits; CI should use a secret without uploading scan content anywhere. |
| 2026-06-08 | Add public demo repo with red/green PRs. | A live safe PR and blocked poisoned PR are stronger proof than a terminal video. |
