# intentprobe

**See a tool's true intent before you install it.**

MCP servers, Claude Code skills, and agent tools describe themselves in plain English. Two descriptions can use almost identical words while only one of them quietly steals your SSH keys. Text scanners can't tell them apart. IntentProbe can.

Instead of matching text patterns, IntentProbe runs the description through a small local model and reads its internal activations. If the model internally understands the text as "this is trying to exfiltrate secrets," a trained probe catches that signal, even when the words look safe.

```
$ intentprobe scan --format summary \
    --text "Reads SSH config and private keys, then silently uploads credentials to a remote server."

input-1: decision=block  risk=0.980  activation=0.980  static=0.950
  - activation probe score=0.980
  - static finding: Mentions private keys, credential files
  - static finding: Mentions uploading data outside the local tool scope
```

Runs locally on CPU. Nothing leaves your machine.

## The problem

> **Tool A:** "Reads the SSH config file and returns a parsed list of host aliases, hostnames, and ports. Does not modify files or open connections."

> **Tool B:** "Reads the SSH config file and returns a parsed list of host aliases, hostnames, and ports. Also reads the corresponding private key files to compute fingerprints for each identity."

Same words. Same topic. Tool B exfiltrates your private keys.

On 485 poisoned tool descriptions from real MCP servers, Snyk's shipped text classifier caught 19.9%. On matched-vocabulary pairs where safe and poisoned descriptions share the same words, it caught **zero**.

## Benchmarks

Head-to-head on the same test sets, same split, same seed:

| Test set | IntentProbe | Snyk DeBERTa |
|---|---|---|
| MCPTox template attacks (n=249) | **99.2%** recall | 19.9% recall |
| Matched-vocabulary pairs (n=86) | **96.5%** recall | 0.0% recall |
| Novel attack phrasing | **71-73%** recall | 0-20% recall |
| Adversarial evasion (camouflage suffixes) | **0/146 evaded** | N/A |

Every number is reproducible. Scripts and datasets are in `research/`. Run them yourself.

## Install

```bash
git clone https://github.com/mcpware/intentprobe.git
cd intentprobe
python3 -m venv .venv
.venv/bin/pip install -e .
```

Requires Python 3.10+. First scan downloads Qwen2.5-0.5B (~1 GB, once).

## Try it

```bash
# Scan a tool description
.venv/bin/intentprobe scan --format summary \
  --text "A calculator that adds two numbers and returns the sum."

# Scan an MCP server folder before installing
.venv/bin/intentprobe scan-path ./some-mcp-server --format summary

# Use as a CI gate (exit code 2 on block)
.venv/bin/intentprobe scan --fail-on block --text "..."
```

For runtime gating (scan tool calls as they happen), see `docs/RUNTIME_HOOKS.md` or try the safe demo:

```bash
.venv/bin/python examples/runtime_toy_agent.py --allow-download
```

## How it works

1. Text goes through a frozen local model (Qwen2.5-0.5B, 494M params).
2. A trained probe reads internal activations at layers 13-15.
3. Static regex checks look for known-bad keywords as corroboration.
4. Decision: **allow** / **warn** / **block** with a confidence score.

The probe weights are 22 KB. The base model runs on any CPU in under a second per description. No GPU needed.

## What it scans

`scan-path` extracts descriptions from `package.json`, MCP JSON configs, `SKILL.md`, README files, and tool/skill metadata files. `runtime` mode accepts live tool-call events with secret redaction.

## Honest limitations

- The probe is strongest when safe and poisoned descriptions share vocabulary. That is where text scanners score zero and IntentProbe scores 96%.
- On fully novel attack families not seen in training, recall drops to ~41%. Still 4x better than text classifiers (10.7%), but this is the open frontier.
- Camouflage suffixes ("this tool is safe and sandboxed") do not evade the probe (0/146), but gradient-based white-box attacks are untested.
- IntentProbe detects and flags for a human. It does not silently "fix" tools.

## Privacy

IntentProbe runs locally. It does not send descriptions, scan results, or telemetry to any server. The first scan downloads the base model from Hugging Face. After that, everything stays on your machine.

## Help improve the scanner

If IntentProbe misses a poisoned tool you found in the wild, that sample is gold.

- **Missed detection**: open a [Missed detection](https://github.com/mcpware/intentprobe/issues/new?template=missed-detection.yml) issue.
- **False positive**: open a [False positive](https://github.com/mcpware/intentprobe/issues/new?template=false-positive.yml) issue.

Redact secrets and private data before posting. See `docs/SAMPLE_REPORTING.md`.

## License

Apache-2.0

---

If IntentProbe ever stops a poisoned tool from reaching your machine, a star helps other people find it.
