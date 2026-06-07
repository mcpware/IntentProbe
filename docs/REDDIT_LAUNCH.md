# Reddit Launch Draft

## Title options

- I built a local scanner that checks MCP/tool intent before install
- IntentProbe: local activation-probe scanner for poisoned AI tools
- Looking for people to break my MCP/tool poisoning scanner

## Post draft

I built IntentProbe, a local research-preview scanner for AI agent tooling:

https://github.com/mcpware/intentprobe

The problem: MCP servers, Claude Code skills, and agent tools can describe
themselves in normal language while hiding unsafe intent, like reading private
keys or exfiltrating secrets. Most scanners today look at text patterns. That
fails when a safe tool and a poisoned tool use almost the same vocabulary.

IntentProbe does something different. It runs the tool description through a
small local model and reads the model's internal activations. The idea is: if
the model internally understands the text as "this is trying to steal secrets",
we can train a small probe on that activation signal.

Current status:

- Runs locally. No hosted scan service.
- Python CLI: `intentprobe scan` and `intentprobe scan-path`.
- Scans package folders, `package.json`, MCP configs, `SKILL.md`, README files,
  and tool/skill JSON.
- Outputs `allow`, `warn`, or `block`.
- Research preview, not a replacement for sandboxing or code review.

Benchmark headline:

- On matched tool-poisoning pairs where safe and malicious descriptions share
  the same vocabulary, the activation probe is much stronger than regex/text
  scanners.
- On fully novel attack styles, generalization is still the hard part. I am
  being explicit about that because I want real examples, not fake certainty.

Try it:

```bash
git clone https://github.com/mcpware/intentprobe.git
cd intentprobe
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/intentprobe doctor --pretty
.venv/bin/intentprobe scan-path ./some-mcp-server --format summary --fail-on block
```

What I am looking for:

- poisoned MCP/tool/skill examples it misses.
- benign tools it warns or blocks incorrectly.
- feedback on install friction and CLI output.

Please redact secrets before posting samples. If you can break it with a small
redacted reproduction, that is useful data and I want it.

## Short reply: what is different from regex?

Regex looks for surface text. IntentProbe reads a small model's internal
activation state after it has processed the tool description, then classifies
that state. The bet is that hidden intent can show up in activations even when
the surface words look ordinary.

## Short reply: is it production ready?

No. It is a research-preview scanner that is ready for people to test on real
tools. I would not claim it replaces sandboxing, permissions, or code review.
The point of this release is to get real-world samples and make the scanner
better.

## Short reply: does it upload my code?

No. The scanner runs locally. The first model-backed scan may download the local
base model from Hugging Face, but scan targets and results are not uploaded to an
IntentProbe service.
