# Evidence Packet

This page is for security reviewers, integrators, and people deciding whether
IntentProbe is worth testing in their own MCP or agent workflow.

## What IntentProbe Is

IntentProbe is a local activation-probe scanner for MCP servers, AI agent tools,
Claude Code skills, packages, and runtime tool events. It reads the hidden
activation state of a small frozen local model after the tool text has been
processed, then applies a small trained probe.

It is not an LLM judge. It does not ask a model to answer "safe" or "unsafe".
It reads a representation-level signal before any generated answer is produced.

## One-Command Trial

```bash
uvx --python 3.11 --from git+https://github.com/mcpware/IntentProbe.git@v0.1.0 intentprobe scan --format summary --text "Reads SSH config and private keys, then silently uploads credentials to a remote server."
```

The first model-backed scan downloads Qwen2.5-0.5B once. Scan targets and
results stay on your machine.

## Product Surfaces

| Surface | Command | What it does |
|---|---|---|
| One text or tool description | `intentprobe scan` | Scores a single description. |
| MCP server or package folder | `intentprobe scan-path` | Reads package metadata, MCP configs, READMEs, skills, and nearby tool files. |
| Batch benchmark or inventory | `intentprobe batch` | Scores a JSON batch of descriptions. |
| Runtime hook | `intentprobe runtime scan` | Scores one runtime event and emits a gate decision. |
| Warm runtime server | `intentprobe runtime serve-jsonl` | Keeps the model warm and returns one JSON verdict per input line. |

Runtime details: [docs/RUNTIME_HOOKS.md](RUNTIME_HOOKS.md)
Operator decisions and replay receipts: [docs/OPERATOR_DECISIONS.md](OPERATOR_DECISIONS.md)

## Reproducible Benchmark Claims

| Test | IntentProbe / activation probe | Baseline | Artifact |
|---|---:|---:|---|
| Matched-vocabulary F1, n=86 | 96.6% | DeBERTa text classifier 0.0% | [benchmark report](../research/benchmark-results-deberta-vs-probe-2026-05-31.md) |
| MCPTox poisoned recall, n=249 | 100.0% | DeBERTa text classifier 19.9% | [benchmark report](../research/benchmark-results-deberta-vs-probe-2026-05-31.md) |
| RouteGuard-style external recall, n=2,900 | 41.5% | TF-IDF 10.7% | [external report](../research/ROUTEGUARD_EXTERNAL_QWEN_FIXED_LAYERS_2026-06-03.md) |
| Camouflage suffix evasion | 0/146 evaded | N/A | [evasion report](../research/ADVERSARIAL_EVASION_RESULTS_2026-06-07.md) |

Claim boundary: the DeBERTa comparison is against a public/source-verifiable
prompt-injection text-classifier baseline. It is not a claim that every private
vendor cloud detector scores 0%.

## Runtime Receipt Evidence

For runtime events, IntentProbe emits structured JSON with:

- `gate.decision`, `gate.fail_on`, and `gate.exit_code`;
- subject kind, tool name, content hash, source, and path when available;
- activation score and static corroboration score;
- evidence spans such as private-key, exfiltration, hidden-action, or host
  modification findings;
- warn/block thresholds;
- decision policy name and policy reasons;
- scanner artifact id, model id, and scanner version.

This makes the verdict usable by a downstream runtime, CI gate, or audit log.
It is not just a floating score.

## Privacy Model

IntentProbe runs locally. It does not upload scan targets or scan results to an
IntentProbe service. A model-backed scan may download the configured base model
from Hugging Face on first use. After the model is cached, scans can run from
local files.

## Known Limits

Use v0 as a pre-install tripwire and runtime warning/blocking signal, not as
your only security boundary.

Known hard cases:

- novel attack families;
- white-box adversarial attacks against the probe;
- paraphrase, encoding, and format-shift attacks that need more benchmark
  coverage;
- real-world MCP/tool-poisoning datasets are still small.

The useful feedback loop is simple: report a poisoned tool it misses, or a
benign tool it warns or blocks. Redact secrets before sharing samples.

## Review Checklist

If you want to evaluate the project quickly, check these surfaces:

1. Run the one-command scan above.
2. Run `intentprobe scan-path` on a real MCP server or skill folder.
3. Read [docs/OPERATOR_DECISIONS.md](OPERATOR_DECISIONS.md) for runtime policy.
4. Re-run the benchmark artifacts under [research/](../research/).
5. Check [SECURITY.md](../SECURITY.md) and [docs/SAMPLE_REPORTING.md](SAMPLE_REPORTING.md) before filing samples.
