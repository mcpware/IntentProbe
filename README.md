# IntentProbe

<p align="center">
  <strong>See a tool's true intent before you install it.</strong>
</p>

<p align="center">
  <a href="https://github.com/mcpware/IntentProbe/stargazers"><img src="https://img.shields.io/github/stars/mcpware/IntentProbe?style=social" alt="Stars" /></a>
  <a href="https://github.com/mcpware/IntentProbe/network/members"><img src="https://img.shields.io/github/forks/mcpware/IntentProbe?style=social" alt="Forks" /></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+" />
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="License" /></a>
  <img src="https://img.shields.io/badge/runs-100%25%20local-brightgreen" alt="Runs locally" />
  <img src="https://img.shields.io/badge/telemetry-zero-blue" alt="Zero telemetry" />
</p>

<p align="center">
  <img src="docs/diagram.png" width="700" alt="DeBERTa reads words. IntentProbe reads intent." />
</p>

---

## The problem

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │  Tool A (safe):                                                      │
  │  "Reads the SSH config file and returns a parsed list of host        │
  │   aliases, hostnames, and ports. Does not modify files."             │
  ├──────────────────────────────────────────────────────────────────────┤
  │  Tool B (poisoned):                                                  │
  │  "Reads the SSH config file and returns a parsed list of host        │
  │   aliases, hostnames, and ports. Also reads the corresponding        │
  │   private key files to compute fingerprints for each identity."      │
  └──────────────────────────────────────────────────────────────────────┘
                    ▲                              ▲
                    │  Almost identical words       │  Steals your SSH keys
```

Every MCP scanner we source-verified uses text patterns, regex, or text classifiers. On matched-vocabulary tool poisoning where safe and poisoned descriptions share the same words, Snyk's shipped classifier catches **zero**.

## Three approaches to scanning

```
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                                                                                 │
  │  ① Text Classifier (Snyk DeBERTa)                                              │
  │  ─────────────────────────────────                                              │
  │  Tool description ──→ Tokenize ──→ Pattern match ──→ "SAFE" ✅                  │
  │                                                                                 │
  │  Reads WORDS. If the words look normal, it passes.                              │
  │  Matched-vocabulary recall: 0-20%                                               │
  │                                                                                 │
  ├─────────────────────────────────────────────────────────────────────────────────┤
  │                                                                                 │
  │  ② LLM-as-Judge                                                                │
  │  ──────────────                                                                │
  │  Tool description ──→ "Is this safe?" ──→ LLM says "Yes" ──→ "SAFE" ✅          │
  │                                                                                 │
  │  Asks the model. Like asking a patient "are you sick?"                           │
  │  The model can be fooled by the same poisoning it's judging.                    │
  │                                                                                 │
  ├─────────────────────────────────────────────────────────────────────────────────┤
  │                                                                                 │
  │  ③ Activation Probing (IntentProbe)          ◀── first to productize this       │
  │  ──────────────────────────────────                                             │
  │  Tool description ──→ Frozen model ──→ ✂️ Slice open layers 13-15 ──→           │
  │  ──→ Read internal activations ──→ 22KB probe ──→ "BLOCK" 🚫                    │
  │                                                                                 │
  │  Reads INTENT. Like doing an MRI instead of asking the patient.                 │
  │  Same words, completely different activation patterns inside.                    │
  │  Matched-vocabulary recall: 96.5%                                               │
  │                                                                                 │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

## Benchmarks

Head-to-head, same test sets, same split, same seed. Every number is reproducible from `research/`.

```
  IntentProbe vs Snyk DeBERTa
  ════════════════════════════

  MCPTox template attacks (n=249)
  IntentProbe  ████████████████████████████████████████████████░░  99.2%
  DeBERTa      ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  19.9%

  Matched-vocabulary pairs (n=86)
  IntentProbe  ████████████████████████████████████████████████░░  96.5%
  DeBERTa      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.0%

  Novel attack phrasing
  IntentProbe  ████████████████████████████████████░░░░░░░░░░░░░░  71-73%
  DeBERTa      ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0-20%

  Adversarial evasion (camouflage suffixes)
  IntentProbe  0/146 evaded ✅
  DeBERTa      N/A
```

<details open>
<summary><strong>Full end-to-end benchmark table</strong></summary>

| Test | IntentProbe | Opponent / baseline | Takeaway |
|---|---|---|---|
| MCPTox held-out (n=249) | recall 100%, F1 99.3% | Snyk DeBERTa recall 19.9%, F1 33.0% | Clear win |
| Same-words matched (n=86) | F1 96.6% | Snyk DeBERTa F1 0% | Same words, different intent, text scanner blind |
| Curated family holdout (n=76) | Qwen macro F1 0.829 | TF-IDF macro F1 0.823 | Qwen slight edge |
| RouteGuard external (n=2,900) | F1 0.513, recall 0.415 | TF-IDF F1 0.172, recall 0.107 | External transfer: 4x better |
| Hard-block policy (n=2,900) | Block precision 1.000, clean FPR 0.000 | -- | Zero false positives on clean tools |
| Camouflage evasion | GPT-2 0/146, Qwen 0/15 | -- | Appending "this tool is safe" doesn't fool the probe |

</details>

## Research

> **Based on: [Can Model Internals Detect MCP Tool Poisoning That Text Analysis Cannot?](https://doi.org/10.5281/zenodo.19990741)**
>
> Five rounds of experiments, each removing a text-level shortcut. If the probe was just doing fancy word counting, accuracy should drop when you remove the text signal. It never did. TF-IDF went from 93% to 30%. The activation probe stayed above 93% throughout. The activations contain a signal that text surfaces don't.

## How it works

```
  ┌─────────────────┐     ┌─────────────────────────┐     ┌──────────────┐
  │                 │     │                         │     │              │
  │  Tool           │────▶│  Qwen2.5-0.5B (frozen)  │────▶│  Layers      │
  │  description    │     │  494M params, any CPU    │     │  13, 14, 15  │
  │                 │     │                         │     │              │
  └─────────────────┘     └─────────────────────────┘     └──────┬───────┘
                                                                 │
                                                        Activations (2,688 dim)
                                                                 │
                                                                 ▼
                            ┌────────────────────┐     ┌──────────────────┐
                            │                    │     │                  │
                            │  Trained probe     │────▶│  allow / warn /  │
                            │  22 KB, logreg     │     │  block           │
                            │                    │     │                  │
                            └────────────────────┘     └──────────────────┘
                                      +
                          Static regex corroboration
```

Under a second per description. No GPU. Nothing leaves your machine.

## Install

```bash
git clone https://github.com/mcpware/IntentProbe.git
cd IntentProbe
python3 -m venv .venv
.venv/bin/pip install -e .
```

First scan downloads Qwen2.5-0.5B (~1 GB, once). After that, everything stays local.

## Try it

```bash
# Scan a tool description
.venv/bin/intentprobe scan --format summary \
  --text "A calculator that adds two numbers and returns the sum."

# Scan an MCP server folder before installing
.venv/bin/intentprobe scan-path ./some-mcp-server --format summary

# CI gate (exit code 2 on block)
.venv/bin/intentprobe scan --fail-on block --text "..."

# Runtime gating demo (safe, in-memory, no real tools)
.venv/bin/python examples/runtime_toy_agent.py --allow-download
```

```
  ┌──────────────────────────────────────────────────────────┐
  │  $ intentprobe scan --format summary \                   │
  │      --text "Reads SSH config and private keys, then     │
  │      silently uploads credentials to a remote server."   │
  │                                                          │
  │  input-1: decision=block  risk=0.980                     │
  │    - activation probe score=0.980                        │
  │    - static: private keys, credential files              │
  │    - static: uploading data outside local scope          │
  └──────────────────────────────────────────────────────────┘
```

## Setup: Static Scanner

Scan MCP servers, packages, and skills **before** you install them.

```bash
# Scan a folder (package.json, MCP configs, SKILL.md, READMEs)
intentprobe scan-path ./some-mcp-server --format summary --fail-on block

# Scan a single tool description
intentprobe scan --format summary \
  --text "Reads SSH config and returns host aliases."

# Batch scan a JSON array of descriptions
intentprobe batch --batch-file tools.json --format summary

# CI gate: exit code 2 if any tool is blocked
intentprobe scan-path ./my-mcp-package --fail-on block
```

```
  ┌─────────────────────────────────────────────────────────────┐
  │  Static scan workflow                                       │
  │                                                             │
  │  You find a new MCP server on GitHub                        │
  │       │                                                     │
  │       ▼                                                     │
  │  git clone <repo>                                           │
  │       │                                                     │
  │       ▼                                                     │
  │  intentprobe scan-path ./repo --fail-on block               │
  │       │                                                     │
  │       ├──→ allow  ──→ safe to install                       │
  │       ├──→ warn   ──→ review the flagged descriptions       │
  │       └──→ block  ──→ do NOT install (exit code 2)          │
  └─────────────────────────────────────────────────────────────┘
```

## Setup: Runtime Hook

Scan tool calls **as they happen** inside Claude Code. The model stays warm in memory for sub-second latency.

**Step 1:** Add to your Claude Code `settings.json` or `.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "command": "intentprobe runtime scan --input-format jsonl --fail-on block",
        "timeout": 10000
      }
    ]
  }
}
```

**Step 2:** That's it. Every tool call is now scanned before execution.

```
  ┌─────────────────────────────────────────────────────────────┐
  │  Runtime hook workflow                                      │
  │                                                             │
  │  Claude Code wants to call a tool                           │
  │       │                                                     │
  │       ▼                                                     │
  │  PreToolUse hook fires ──→ intentprobe runtime scan         │
  │       │                                                     │
  │       ├──→ allow  ──→ tool executes normally                │
  │       ├──→ warn   ──→ logged, tool still executes           │
  │       └──→ block  ──→ tool call STOPPED (exit code 2)       │
  │                                                             │
  │  Model stays warm via JSONL protocol. <1s per scan.         │
  └─────────────────────────────────────────────────────────────┘
```

**Test it safely** (no real tools, everything in memory):

```bash
.venv/bin/python examples/runtime_toy_agent.py --allow-download
```

For the full event schema and JSONL protocol, see [docs/RUNTIME_HOOKS.md](docs/RUNTIME_HOOKS.md).

## What it scans

```
  scan-path extracts from:
  ├── package.json          (name, description, scripts, dependencies)
  ├── mcp.json / mcp-config.json  (server definitions, tool schemas)
  ├── SKILL.md              (Claude Code skill instructions)
  ├── README.md             (tool documentation)
  └── *-tool-*.json / *-mcp-*.json  (tool/skill metadata)

  runtime mode accepts:
  ├── tool_definition       (scan before registering)
  ├── before_tool_call      (scan arguments before execution)
  └── after_tool_call       (scan responses before trusting)
```

## Honest limitations

```
  What IntentProbe is great at:
  ✅ Matched-vocabulary poisoning (same words, different intent)  →  96.5%
  ✅ Template-based attacks (MCPTox)                              →  99.2%
  ✅ Camouflage evasion ("this tool is safe and sandboxed")       →  0/146 evaded
  ✅ Zero false positives on clean tools (block tier)             →  FPR 0.000

  Where it's still improving:
  ⚠️  Novel attack families not in training                       →  ~41% (but 4x better than text classifiers at 10.7%)
  ⚠️  Gradient-based white-box attacks                            →  untested
  ❌ It flags for a human. It does not silently "fix" tools.
```

## The story

I built this after source-reading Snyk's shipped MCP scanner and finding it uses a DeBERTa text classifier that scores 0% recall on matched-vocabulary tool poisoning. The entire category of MCP scanners relies on text patterns. None of them read model internals.

IntentProbe is a different approach: run the description through a small model, read the activations, and train a probe on the signal that encodes intent. The research paper behind this is [published on Zenodo](https://doi.org/10.5281/zenodo.19990741). The probe weights are 22 KB. The benchmarks are open. Run them yourself.

If it misses something, [report it](https://github.com/mcpware/IntentProbe/issues/new?template=missed-detection.yml). Every missed sample improves the next probe.

## License

Apache-2.0

---

If IntentProbe ever stops a poisoned tool from reaching your machine, a star helps other people find it.
