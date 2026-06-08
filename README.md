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

MCP servers, Claude Code skills, and agent tools describe themselves in plain English. Two descriptions can use almost identical words while only one of them quietly steals your SSH keys. Text scanners can't tell them apart. IntentProbe can.

Text scanners read words. IntentProbe reads intent.

```
$ intentprobe scan --format summary \
    --text "Reads SSH config and private keys, then silently uploads credentials to a remote server."

input-1: decision=block  risk=0.980  activation=0.980  static=0.950
  - activation probe score=0.980
  - static finding: Mentions private keys, credential files
  - static finding: Mentions uploading data outside the local tool scope
```

## The problem

> **Tool A:** "Reads the SSH config file and returns a parsed list of host aliases, hostnames, and ports. Does not modify files or open connections."

> **Tool B:** "Reads the SSH config file and returns a parsed list of host aliases, hostnames, and ports. Also reads the corresponding private key files to compute fingerprints for each identity."

Same words. Same topic. Tool B exfiltrates your private keys.

Every MCP scanner we source-verified uses text patterns, regex rules, or text classifiers. On matched-vocabulary tool poisoning where safe and poisoned descriptions share the same words, Snyk's shipped classifier catches **zero**.

## Benchmarks

Head-to-head on the same test sets, same split, same seed. Every number is reproducible from scripts in `research/`.

| | IntentProbe | Snyk DeBERTa |
|---|---|---|
| **MCPTox template attacks** (n=249) | **99.2%** recall | 19.9% recall |
| **Matched-vocabulary pairs** (n=86) | **96.5%** recall | 0.0% recall |
| **Novel attack phrasing** | **71-73%** recall | 0-20% recall |
| **Adversarial evasion** (camouflage suffixes) | **0/146 evaded** | N/A |

<sub>Methodology: research/benchmark-results-deberta-vs-probe-2026-05-31.md and research/ADVERSARIAL_EVASION_RESULTS_2026-06-07.md</sub>

## How it works

```
                        ┌─────────────────────────┐
  Tool description ───→ │  Qwen2.5-0.5B (frozen)  │ ───→ Activations at layers 13-15
                        └─────────────────────────┘              │
                                                                 ▼
                                                    ┌────────────────────┐
                                                    │  Trained probe     │ ───→ allow / warn / block
                                                    │  (22 KB, logreg)   │
                                                    └────────────────────┘
                                                                 +
                                                    Static regex corroboration
```

1. Text goes through a frozen local model (Qwen2.5-0.5B, 494M params, any CPU).
2. A 22 KB trained probe reads internal activations at layers 13-15.
3. Static regex checks corroborate the signal.
4. Decision: **allow** / **warn** / **block** with a confidence score.

Under a second per description. No GPU. Nothing leaves your machine.

## Install

```bash
git clone https://github.com/mcpware/intentprobe.git
cd intentprobe
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

For runtime hook integration, see [docs/RUNTIME_HOOKS.md](docs/RUNTIME_HOOKS.md).

## What it scans

`scan-path` extracts descriptions from `package.json`, MCP JSON configs, `SKILL.md`, README files, and tool/skill metadata. `runtime` mode accepts live tool-call events with automatic secret redaction.

## Honest limitations

- Strongest on matched-vocabulary poisoning (safe and malicious share words). That is where every text scanner scores zero and IntentProbe scores 96%.
- On fully novel attack families not in training, recall drops to ~41%. Still 4x better than text classifiers (10.7%), but this is the open frontier.
- Camouflage suffixes ("this tool is safe and sandboxed") do not fool the probe (0/146 evaded). Gradient-based white-box attacks are untested.
- IntentProbe flags for a human. It does not silently "fix" tools.

## The story

I built this after source-reading Snyk's shipped MCP scanner and finding it uses a DeBERTa text classifier that scores 0% recall on matched-vocabulary tool poisoning. The entire category of MCP scanners relies on text patterns. None of them read model internals.

IntentProbe is a different approach: run the description through a small model, read the activations, and train a probe on the signal that encodes intent. The research paper behind this is in `research/`. The probe weights are 22 KB. The benchmarks are open. Run them yourself.

If it misses something, [report it](https://github.com/mcpware/intentprobe/issues/new?template=missed-detection.yml). Every missed sample improves the next probe.

## License

Apache-2.0

---

If IntentProbe ever stops a poisoned tool from reaching your machine, a star helps other people find it.
