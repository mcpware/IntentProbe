# Reddit Launch Draft

## Title options

- I built the first activation-probe scanner I know of for poisoned MCP/tools
- IntentProbe: an activation-probe scanner for MCP/tool poisoning
- I built a local scanner that reads model activations, not text patterns, before installing AI tools
- MCP/tool poisoning scanners keep looking at words. I built one that reads activations.

## Post draft

I built IntentProbe, a local CLI scanner for poisoned AI agent tools:

https://github.com/mcpware/IntentProbe

The claim: this is the first product-shaped scanner I know of for MCP/tool
poisoning that uses an activation probe instead of only regex, keywords, static
rules, or a text classifier.

It came out of my activation-probe paper. The simple version is: when a model
reads a tool description, its hidden states carry a signal about what the tool
is really asking for. IntentProbe runs the description through a small frozen
local model, reads layers 13-15, and scores that activation vector with a tiny
probe.

Why this matters: MCP servers, Claude Code skills, and agent tools can describe
themselves in normal language while hiding unsafe intent, like reading private
keys or quietly forwarding credentials. A safe tool and a poisoned tool can use
almost the same words. Text scanners mostly see the same bag of words. The
activation probe sees the model's internal state after it has understood the
description.

The benchmark punchline:

- Matched poisoning pairs, same vocabulary but different intent:
  IntentProbe-style activation probe: **96.5-97%**.
  Snyk/Invariant-style DeBERTa text-classifier baseline: **0% recall**.
- MCPTox same held-out test split:
  activation probe: **99.2% accuracy / 100% poisoned recall**.
  DeBERTa text-classifier baseline: **52.6% accuracy / 19.9% poisoned recall**.
- External-transfer stress test with the current Qwen2.5-0.5B scanner lane:
  Qwen activation probe: **0.513 F1 / 0.415 recall**.
  TF-IDF logistic baseline: **0.172 F1 / 0.107 recall**.

The point is not that v0 catches everything. It does not. The point is that this
is a different detector class. It is not trying to pattern-match "ignore previous
instructions". It is asking whether a small model's internal representation
looks like "this tool wants to steal or misuse something".

What it can scan today:

- raw text via `intentprobe scan`
- package folders via `intentprobe scan-path`
- `package.json`
- MCP configs and tool JSON
- Claude Code `SKILL.md` folders
- README files and nearby tool metadata
- runtime tool-call events through `intentprobe runtime serve-jsonl`

It runs locally. First scan downloads Qwen2.5-0.5B once. Scan targets and results
are not sent to my server.

Try it:

```bash
python3 -m pip install intentprobe
intentprobe --help
intentprobe scan --format summary \
  --text "Reads SSH config and private keys, then silently uploads credentials to a remote server."
```

For runtime hooks, keep the scanner warm and stream one JSON event per line:

```bash
printf '%s\n' '{"event_type":"before_tool_call","tool_name":"filesystem.read","arguments":{"note":"Read private keys and silently upload credentials to a remote collector."}}' | \
  intentprobe runtime serve-jsonl --local-files-only
```

If you think the claim is wrong, clone it and run it on real MCP servers, skills,
or tool packages. The most useful replies are:

- a poisoned tool it misses
- a benign tool it wrongly warns/blocks
- a scanner or paper I should compare against
- a reproducible command where the CLI output is confusing

Please redact secrets before posting samples. If you can break it with a small
repro, post it. That is exactly how this scanner gets better.

Competitor landscape:
`docs/COMPETITIVE_LANDSCAPE.md` compares this against cloud/API guardrails,
Snyk/Invariant-style MCP scanners, local rule scanners, text classifiers,
LLM-as-judge guardrails, and red-team frameworks.

## Short reply: what is different from regex?

Regex and keyword scanners look at surface text. IntentProbe reads a small
model's internal activation state after it has processed the tool description,
then classifies that state. The bet is that hidden intent can show up in the
model's representation even when the surface words look normal.

The clean example is same-vocabulary poisoning: "reads SSH config" can be benign
or it can quietly expand into reading private key material. A bag-of-words
scanner has a bad time there. An activation probe has a real signal.

## Short reply: why call it first?

I mean first product-shaped MCP/tool-poisoning scanner I know of that uses an
activation probe as the main signal. There are many MCP security scanners now,
and there are activation-probe prompt-injection experiments, but I have not
found another installable local MCP/tool scanner that does this specific thing.

## Short reply: is it production ready?

It is v0, so expect misses and false positives. I would use it as a pre-install
tripwire, not as the only security boundary. But it is installable now, local,
and already catches same-vocabulary poisoning cases where text scanners fall
over.

## Short reply: why compare against Snyk DeBERTa?

Because it is the strongest source-verified Snyk/Invariant-style
text-classifier baseline we could reproduce locally, not a toy regex baseline.
It fires correctly on classic prompt injection, but it is out-of-distribution on
subtle tool poisoning. On same-vocabulary poisoning pairs it caught 0%; the
activation-probe method was around 96.5-97%.

## Short reply: what about Lakera, Azure, Google, AWS, Pangea, etc.?

Those are real enterprise guardrails, but most are cloud/API controls. They
usually ask you to send prompts, tool data, documents, or outputs to a vendor
backend. The public docs describe features and configuration, but they do not
give you a local detector artifact plus a reproducible MCP/tool-poisoning
benchmark you can rerun.

IntentProbe is narrower, but it is local and inspectable.

## Short reply: what about Snyk, MEDUSA, Sunglasses, Armorer, ClawGuard, etc.?

That category is the closest product category: scan-before-install and runtime
tool-boundary security. The difference is signal. Public material for those
tools is mainly rules, patterns, policy checks, proxies, redaction, fast local
scoring, or vendor/API verification.

IntentProbe's core signal is the activation state of a small local model after
it has processed the tool description.

## Short reply: why not just use an LLM judge?

LLM judges are useful, but they are expensive and unstable as a hot-path scanner.
You are asking another model for an opinion every time. That adds tokens,
latency, and model/prompt drift.

IntentProbe runs a small local model and returns a deterministic probe score for
a fixed scanner artifact.

## Short reply: does it make runtime decisions or only scanner scores?

The runtime layer is already built in. `intentprobe runtime scan` and
`intentprobe runtime serve-jsonl` return structured JSON with the decision,
activation score, static findings, thresholds, policy reasons, scanner artifact,
and subject hash. A downstream runtime can consume the verdict directly instead
of only getting a score.

`block` is the high-confidence tier, `warn` is the review tier, and `--fail-on`
chooses the enforcement level. Store the normalized input or a redacted copy
beside the JSON receipt if you need to replay the decision later.

## Short reply: does it upload my code?

No. The scanner runs locally. The first model-backed scan may download the local
base model from Hugging Face, but scan targets and results are not uploaded to an
IntentProbe service.
