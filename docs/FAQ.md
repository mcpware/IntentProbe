# IntentProbe FAQ

## What is IntentProbe?

IntentProbe is a local scanner for MCP servers, AI agent tools, Claude Code
skills, packages, and runtime tool events. It looks for poisoned intent such as
credential access, secret exfiltration, hidden persistence, forced tool
chaining, or tool hijacking.

## What does "activation probing" mean?

IntentProbe runs a tool description through a small frozen local model and reads
the hidden activation state inside the model. Then a small trained probe scores
whether that internal state looks poisoned.

Simple version: text scanners read the words. LLM judges ask for an answer.
IntentProbe checks the internal signal before the model says anything.

## Is this the same as asking Qwen if a tool is safe?

No. Asking Qwen "is this safe?" is an LLM-as-judge approach. IntentProbe uses
Qwen2.5-0.5B as a fixed feature extractor and reads hidden activations instead
of trusting the model's generated answer.

We tested the direct-prompt approach. The deterministic label-score version
flagged every clean curated item as poisoned, while the generated-answer
version missed poison and produced many unparseable outputs. See
[`research/QWEN_PROMPT_JUDGE_BASELINE_2026-06-08.md`](../research/QWEN_PROMPT_JUDGE_BASELINE_2026-06-08.md).

## Why not just use a text classifier?

Text classifiers are useful when unsafe samples have obvious words or patterns.
They struggle when a safe tool and a poisoned tool use almost the same
vocabulary.

On the matched-vocabulary benchmark, the public/source-verifiable DeBERTa
prompt-injection text-classifier baseline scored 0.0% F1, while the
activation-probe method scored 96.6% F1.

## Does IntentProbe upload my tool descriptions?

No. IntentProbe runs locally. Scan targets and scan results stay on your
machine.

The first model-backed scan may download Qwen2.5-0.5B once from Hugging Face.
After the model is cached, scans can run from local files.

## What model does v0 use?

The released v0 scanner uses Qwen2.5-0.5B as the frozen local sensor model and
reads layers 13-15. The shipped probe artifact is about 22 KB.

## Does IntentProbe change or train the base model?

No. The base model stays frozen. IntentProbe trains a small classifier on top of
extracted activation features. At scan time, the model is only used to produce
features.

## What can it scan today?

IntentProbe can scan:

- one text/tool description;
- package folders through `scan-path`;
- `package.json`;
- MCP configs and tool JSON;
- Claude Code `SKILL.md` folders;
- README files and nearby tool metadata;
- runtime events such as tool definitions, before-tool-call arguments, and
  after-tool-call responses.

## Can I use it as a runtime hook?

Yes. See [`docs/RUNTIME_HOOKS.md`](RUNTIME_HOOKS.md). Runtime scanning is
event-boundary scanning: tool definitions before trust, tool arguments before
execution, and tool responses before the agent trusts them.

The runtime output is structured JSON, so a host can consume it directly. See
[`docs/OPERATOR_DECISIONS.md`](OPERATOR_DECISIONS.md) for `allow`, `warn`,
`block`, replay receipts, and suggested operator mappings.

## What are the headline benchmarks?

The highest-signal public numbers are:

| Test | IntentProbe / activation probe | Baseline |
|---|---:|---:|
| Matched-vocabulary F1, n=86 | 96.6% | DeBERTa text classifier 0.0% |
| MCPTox poisoned recall, n=249 | 100.0% | DeBERTa text classifier 19.9% |
| RouteGuard-style external recall, n=2,900 | 41.5% | TF-IDF 10.7% |
| Camouflage suffix evasion | 0/146 evaded | N/A |

Benchmark artifacts are under [`research/`](../research/).

For a compact reviewer packet, see
[`docs/EVIDENCE_PACKET.md`](EVIDENCE_PACKET.md).

## Is this a claim about every private cloud scanner?

No. The DeBERTa result is a reproducible comparison against a
public/source-verifiable text-classifier baseline. Private cloud/API scanners
may work well, but their detector artifacts and MCP/tool-poisoning benchmarks
are usually not reproducible by users.

## Have you tried SAE features?

Yes. SAE features are useful for interpretability and may improve future recall.
The v0 product ships raw Qwen activations because the current raw-activation
artifact is the most complete, lightweight, and reproducible product path today.

SAE is planned as an optional layer for recall improvements and human-readable
explanations.

## Is v0 production-ready?

Use v0 as a pre-install tripwire and runtime warning/blocking layer, not as your
only security boundary. It already catches important same-vocabulary poisoning
cases that text scanners miss, but novel attack families and white-box
adversarial attacks still need more work.

## How do I try it quickly?

```bash
python3 -m pip install intentprobe
intentprobe scan --format summary --text "Reads SSH config and private keys, then silently uploads credentials to a remote server."
```

## How do I report a miss or false positive?

Please submit the smallest redacted sample that reproduces the result.

- [Missed detection](https://github.com/mcpware/IntentProbe/issues/new?template=missed-detection.yml)
- [False positive](https://github.com/mcpware/IntentProbe/issues/new?template=false-positive.yml)
- [Sample reporting guide](SAMPLE_REPORTING.md)
