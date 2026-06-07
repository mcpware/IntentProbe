# intentprobe — Roadmap & Project State

> Single source of truth for where this project is. Update after each milestone so any new session can resume from this file alone.

## What this is

An activation-probe security scanner for AI agent tooling. Before an agent installs an MCP server, a Claude Code skill, or an npm package, intentprobe reads a small local model's internal activations to detect tool poisoning that text and regex scanners miss. It ships as a pre-install hook, so the agent runs it automatically before adding any tool.

It is the first scanner to use model internals instead of regex / AST / LLM-as-judge. Apache-2.0, research preview.

Backed by research: a 60-rule regex scanner catches **0 / 485** on the MCPTox benchmark, while an activation probe scores **97%** on matched pairs where the safe and poisoned descriptions share the same vocabulary.

## Roadmap

- [x] **1. Name + repo.** `intentprobe` (npm unscoped + `@mcpware` both free; survived competitive / SEO / trademark research against trueye, klyro, probescan, latens, etc). Repo scaffolded, Apache-2.0.
- [x] **2. Spike: validate a modern sub-2B model's probe.** Qwen2.5-0.5B activations on the hard_v3 (100+100) and neutral (15+15) sets vs a TF-IDF baseline. Goal: confirm the pipeline works and a modern small model matches or beats the GPT-2 reference (paper: GPT-2 probe 98.5% on hard_v3 where TF-IDF is 79.5%).
- [x] **3. Pick current scanner-candidate lane.** Qwen2.5-0.5B pooled raw activations, fixed layers 13/14/15, calibrated policy v3, with Pythia as the cheap open canary. This is the current default research-preview lane, not the final published artifact.
- [x] **4. intentprobe Python CLI preview.** `intentprobe` now wraps the scanner runtime with doctor, single scan, batch scan, JSON output, summaries, and `--fail-on` exit codes.
- [x] **5. Hook wrapper preview.** `intentprobe-hook` normalizes MCP/tool/skill/hook payloads, redacts secret values, emits gate JSON, and supports a warm JSONL process for runtime scanning.
- [ ] **6. Research-preview README + crowdsource + publish.** Honest about the ~71-73% cross-phrasing frontier; collect real poisoned samples from users (feeds the underlying research). npm publish + GitHub push (org: mcpware, tentative).
- [ ] **7. Product packaging.** Move the research-preview CLI/hook into an installable package, add a release artifact download path, and wire install-time scanner hooks for common agent workflows.

## Key technical facts (do not relearn these)

- The probe consumes **activations, not text**. Every scan runs the base model's forward pass (stop at a middle layer, no generation, no GPU; a sub-2B model on CPU is roughly sub-second per description).
- The probe weights are tiny (a few KB) but **cannot run standalone** — they need the base model to turn text into activations. So the base model must be small enough to run on any user machine (sub-2B). A large model excludes users with no GPU.
- Training and runtime are both **Python** (torch + transformers + sklearn). A hook just spawns the command, so users never see the runtime — `npx` vs `pipx` does not matter to them.
- **Research-preview positioning**: lead with the cases where text scanners hit 0 and the probe hits 97. Cross-phrasing generalization (~71-73%) is the open frontier — stay honest, and crowdsource real samples.
- The probe **detects and flags for a human**. It never tries to silently "fix" a tool from the inside.

## Where things live

- Training / eval data: `research/datasets/`.
- Product CLI wrappers: `intentprobe/cli.py` and `intentprobe/hook.py`.
- Scanner runtime: `research/activation_scanner_core.py`, `research/activation_scanner_cli.py`, and `research/activation_scanner_hook.py`.
- Benchmark harness: `research/benchmarks/`.
- Reproducibility ledger: `research/PRODUCT_REPRODUCIBILITY_LEDGER_2026-06-03.md`.
- Spike history: `training/model_compare.py` and `training/model_compare_v2.py`.
- Research paper: "Can Model Internals Detect MCP Tool Poisoning That Text Analysis Cannot?"

## Current status

Step 5 done in research-preview form. The original v1 spike
(in-distribution, optimistic) ran 5 of 8 models before OOM on 30GB Linux
machine:

| model | params | ms/each | hard_v3 (in-dist 5-fold) | neutral |
|---|---|---|---|---|
| gpt2-124m | 124M | 59ms | 0.985 | 0.967 |
| smollm2-135m | 135M | 64ms | 0.990 | 0.900 |
| smollm2-360m | 362M | 165ms | 0.995 | 0.967 |
| gemma3-270m | 268M | 74ms | 0.990 | 0.900 |
| qwen2.5-0.5b | 494M | 179ms | 0.985 | 0.933 |

TF-IDF baseline: 0.790 (matches paper's 0.795). GPT-2 0.985 matches paper exactly.

**These in-dist numbers are optimistic (overfit risk).** model_compare_v2.py tests cross-set generalization: trains on hard_v3, tests on independent held-out sets (hard_v1/v2/neutral/matched). Selection metric = cross-set, not in-dist.

The current product-candidate scanner is documented in
`research/LIVING_PLAN.md` and `research/SCANNER_PIPELINE.md`. Next big
milestone: package the CLI/hook into a real installable scanner, then run the
release report from the reproducibility ledger before public launch.
