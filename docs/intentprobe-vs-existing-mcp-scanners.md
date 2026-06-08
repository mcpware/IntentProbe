# IntentProbe vs Existing MCP Scanners

IntentProbe is a local activation-probe scanner for MCP servers, AI agent tools,
Claude Code skills, packages, and runtime tool events.

The practical difference is mechanism:

- text/rule/policy scanners read the surface;
- LLM-as-judge scanners ask another model for an answer;
- cloud/API scanners return a vendor-side decision;
- IntentProbe reads a local model's internal activation state.

## Quick comparison

| Scanner approach | Examples | What it reads | What users can verify | Main gap | IntentProbe difference |
|---|---|---|---|---|---|
| Text, rule, and policy scanning | Regex, metadata checks, static MCP scanners, DeBERTa-style text classifiers | Words, permissions, schemas, known suspicious patterns | Usually local rules or model name; sometimes source code | Same-vocabulary poisoning can look normal at the text layer | Reads hidden activations after the model processed the description |
| LLM-as-judge | Prompting GPT, Claude, Qwen, Promptfoo-style graders | A generated answer such as safe/unsafe | Prompt and model choice, if disclosed | Prompt-sensitive, parse-sensitive, token-costly, and output-level | Uses a fixed local probe score before any verbal answer is generated |
| Enterprise cloud/API guardrails | Lakera, Azure Prompt Shields, Google Model Armor, AWS Bedrock Guardrails, Pangea/CrowdStrike AI Guard, Cisco AI Defense, HiddenLayer | Vendor-side detector over prompts, documents, tool calls, or outputs | Usually API response and product docs | Detector artifact and MCP/tool-poisoning benchmark are usually not reproducible by users | Runs locally with public data, scripts, and probe artifact |
| Red-team frameworks | garak, Giskard, Promptfoo red team | Generated attacks and app behavior | Test harness and attack set | Audit workflow, not a cheap pre-install scanner | CLI and runtime hook for install-time and tool-boundary scanning |
| IntentProbe | IntentProbe | Internal activations from Qwen2.5-0.5B layers 13-15 | Repo, benchmark scripts, datasets, probe artifact | v0 still needs more wild data and white-box adversarial testing | Different detector class: representation-level, local, reproducible |

## Why activation probing matters

A poisoned tool does not always announce itself with obvious words. It can look
like a normal helper while quietly adding credential access, remote upload,
hidden chaining, or persistence.

Text scanner:

```text
Tool description -> read words -> "looks safe"
```

LLM-as-judge:

```text
Tool description -> ask "is this safe?" -> generated answer
```

IntentProbe:

```text
Tool description -> frozen local model -> hidden layers -> probe score
```

The bet is not that Qwen is a better judge. The bet is that a model's internal
representation can carry intent signal that the surface text and generated
answer do not reliably expose.

## Benchmark snapshot

| Test | IntentProbe / activation probe | Baseline | Source |
|---|---:|---:|---|
| Matched-vocabulary F1, n=86 | 96.6% | DeBERTa text-classifier baseline 0.0% | [`research/benchmark-results-deberta-vs-probe-2026-05-31.md`](../research/benchmark-results-deberta-vs-probe-2026-05-31.md) |
| MCPTox poisoned recall, n=249 | 100.0% | DeBERTa text-classifier baseline 19.9% | [`research/benchmark-results-deberta-vs-probe-2026-05-31.md`](../research/benchmark-results-deberta-vs-probe-2026-05-31.md) |
| RouteGuard-style external recall, n=2,900 | 41.5% | TF-IDF logistic baseline 10.7% | [`README.md`](../README.md#benchmarks) |
| Camouflage suffix evasion | 0/146 evaded | N/A | [`research/ADVERSARIAL_EVASION_RESULTS_2026-06-07.md`](../research/ADVERSARIAL_EVASION_RESULTS_2026-06-07.md) |
| Direct Qwen judge baseline | Not used as scanner default | Clean FPR 1.000 in label-logprob mode | [`research/QWEN_PROMPT_JUDGE_BASELINE_2026-06-08.md`](../research/QWEN_PROMPT_JUDGE_BASELINE_2026-06-08.md) |

## What this does not claim

IntentProbe does not claim that every vendor cloud detector fails. Most private
cloud/API detectors are not reproducible from public artifacts, so they cannot
be fairly scored by this repo unless a public benchmark path exists.

IntentProbe does not claim text scanning is useless. Rules, signatures, policy
checks, and credential scanners are good hygiene. IntentProbe adds a different
signal for the cases where the words look normal but the intent is not.

IntentProbe does not claim v0 catches everything. Novel families, paraphrase
attacks, encoding attacks, and white-box attacks remain active work.

## When to use IntentProbe

Use IntentProbe when you want a local, inspectable tripwire before installing
or trusting an MCP server, skill, package, or tool definition.

Use it with other controls, not instead of them:

- static policy checks for obvious dangerous capabilities;
- dependency and secret scanning for package hygiene;
- sandboxing and least privilege for runtime containment;
- IntentProbe for representation-level intent signal.

## Start here

Try a one-command scan:

```bash
python3 -m pip install intentprobe
intentprobe scan --format summary --text "Reads SSH config and private keys, then silently uploads credentials to a remote server."
```

Then read:

- [`README.md`](../README.md)
- [`docs/FAQ.md`](FAQ.md)
- [`docs/COMPETITIVE_LANDSCAPE.md`](COMPETITIVE_LANDSCAPE.md)
- [`docs/RUNTIME_HOOKS.md`](RUNTIME_HOOKS.md)
