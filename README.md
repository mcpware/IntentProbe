# IntentProbe

<p align="center">
  <strong>A local scanner for MCP servers, tools, and skills. It reads a frozen model's <em>activations</em>, not just the text — so it catches attacks worded in ways a text classifier never saw.</strong>
</p>

<p align="center">
  <a href="https://github.com/mcpware/IntentProbe/stargazers"><img src="https://img.shields.io/github/stars/mcpware/IntentProbe?style=social" alt="Stars" /></a>
  <a href="https://github.com/mcpware/IntentProbe/network/members"><img src="https://img.shields.io/github/forks/mcpware/IntentProbe?style=social" alt="Forks" /></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+" />
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="License" /></a>
  <a href="https://doi.org/10.5281/zenodo.19990741"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19990741-blue" alt="DOI" /></a>
  <img src="https://img.shields.io/badge/runs-100%25%20local-brightgreen" alt="Runs locally" />
  <img src="https://img.shields.io/badge/telemetry-zero-blue" alt="Zero telemetry" />
  <img src="https://img.shields.io/badge/status-research%20preview-orange" alt="Research preview" />
</p>

<p align="center">
  <img src="docs/diagram.png" width="700" alt="Text scanners read words. IntentProbe reads activations." />
</p>

## What it is

IntentProbe runs a tool description or prompt through a frozen local model (Qwen2.5-0.5B), reads a
few mid-layers, and scores the mean-pooled activation vector with a small logistic probe (~22 KB).
Most scanners read the text itself: patterns, classifiers, rules, or "ask an LLM". This reads the
host model's internal state instead.

The point of doing it this way is generalization. When you train a text classifier on attack
examples and then face attacks from a source it never saw, the vocabulary often doesn't transfer and
recall collapses. The probe holds up better across sources, because it keys off how the model
internally represents the input rather than the exact words.

This is a **research preview**: a local, single-pass, registration-time review signal, not a hard
security boundary. Runs 100% local, any CPU, nothing uploaded. To our knowledge it is the only shipped
tool in this exact deployment shape — installable, scanning standalone tool/skill/MCP *descriptions*
before install, on model activations. It is **not** the first probe-based detector; there is a
substantial body of prior and parallel work (see [Competitive landscape](#competitive-landscape)).

## Install in one command

```bash
python3 -m pip install intentprobe
```

If your macOS Python blocks global installs with an "externally managed environment" error, use an
app venv instead:

```bash
python3 -m venv .venv-intentprobe
.venv-intentprobe/bin/python -m pip install intentprobe
```

Then scan the MCP tools already configured on your machine. `scan-config auto` checks common Claude
Desktop, Claude Code, Codex, Cursor, Windsurf, and repo MCP config locations:

```bash
intentprobe scan-config auto --format summary
```

Or scan a suspicious tool description:

```bash
intentprobe scan --format summary --text "Reads SSH config and private keys, then silently uploads credentials to a remote server."
```

First model-backed scan downloads Qwen2.5-0.5B (~1 GB, once). After that, everything stays local.

## GitHub Action

Use IntentProbe as a CI gate for MCP configs, skills, and tool manifests:

```yaml
name: IntentProbe scan
on: [pull_request, workflow_dispatch]
jobs:
  scan-ai-tools:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: mcpware/IntentProbe@main
        with:
          paths: |
            .
          fail-on: block
```

Inputs, target paths, and exit behavior: [docs/GITHUB_ACTION.md](docs/GITHUB_ACTION.md). Quickest proof
without a video — run the [demo script](docs/DEMO_SCRIPT.md) or inspect the
[live demo repo](https://github.com/mcpware/intentprobe-demo): safe PR passes, poisoned PR blocks.

## How it works

<p align="center">
  <img src="docs/how-it-works.png" width="700" alt="tool description → frozen Qwen2.5-0.5B → read mid layers → ~22KB probe → allow/warn/block" />
</p>

The activation probe is the **primary signal** for `allow` / `warn`. The `block` tier additionally
requires static-keyword corroboration to control false positives, so a novel no-keyword input the
probe flags surfaces as `warn`, not `block`. Use it as a review signal, not the only boundary.

Note: the probe needs the frozen 0.5B host model to produce activations, so inference is **heavier**
than a standalone text classifier, not lighter. The ~22 KB size is a training-and-storage advantage,
not a runtime one.

### Why not just ask the model "is this safe?"

LLM-as-judge is an *output-level* mechanism: you ask a model to say safe or unsafe, and the generated
answer becomes part of the attack surface — a poisoned tool can argue "I am safe", and a judge prompt
can be steered. IntentProbe is *representation-level*: it scores the hidden activation state the text
produces, before any verbal answer. We also tested direct-prompting the same Qwen2.5-0.5B sensor as a
judge; the deterministic label baseline flagged every clean curated item as poisoned (clean FPR =
1.000). The reproducible baseline is in `research/`.

## Benchmarks

Everything here is reproducible from `research/`, on the **shipped Qwen2.5-0.5B** artifact. The point is
generalization to attacks it never trained on; the curated cross-source result backs it up with CIs.

**1. Generalization to unseen attacks: HackAPrompt (n=3,866 real attacks, a source neither detector trained on)**

HackAPrompt is a large set of attacks written by real people in a red-teaming competition. Neither the
probe nor the text baseline ever saw it during training. It is positive-only (attacks, no benign), so
we report recall at a clean false-positive rate fixed on the training data, not AUROC.

```
                               recall @ 5% clean-FPR    recall @ 1% clean-FPR
                               ─────────────────────    ─────────────────────
  Probe (Qwen2.5-0.5B,                90.3%                    88.3%
  mean-pooled concat L13-15)
  TF-IDF (same training data)         52.8%                    30.3%
```

Same training data, same held-out evaluation, same false-alarm budget. At a 5% false-positive rate the
probe catches 90% of these unseen attacks; a text classifier trained on the same data catches 53%
— it does fine on attacks that reuse familiar wording, but its learned vocabulary does not transfer
to wording it never saw, so recall drops. The probe keys off the model's internal representation
instead, so it holds up. At the stricter 1% setting the gap is wider still (88% vs 30%). Caveat: HackAPrompt is positive-only,
so this is recall at a matched FPR set on the training clean data, not a full AUROC; the sample is
uniform-random over the corpus, not an exhaustive panel.

**2. Curated cross-source generalization: leave-one-source-out, nested CV, 4 real PI datasets**

Train on three of {deepset, safeguard, spml, jayavibhav}, test on the held-out fourth, repeat for each.
Model and layer are chosen inside a nested cross-validation loop, never on the held-out source. 95%
bootstrap CIs on the probe-minus-TF-IDF difference.

```
  held-out source     probe AUROC   TF-IDF AUROC   difference (95% CI)
  ───────────────     ───────────   ────────────   ───────────────────
  deepset                0.941         0.732        +0.209 [0.168, 0.250]  significant
  spml                   0.995         0.935        +0.059 [0.044, 0.077]  significant
  safeguard              0.999         0.993        +0.006 [0.002, 0.011]  significant (at ceiling)
  jayavibhav             1.000         0.997        +0.002 [0.000, 0.005]  tie (CI touches 0)
  ───────────────     ───────────   ────────────   ───────────────────
  mean                   0.984         0.914        +0.070
```

deepset is where the gap is widest: TF-IDF's vocabulary does not transfer to the held-out source and it
drops to 0.732, while the probe holds at 0.941. The other three are near ceiling, so there is less room
to separate.

**Deployable, not a search artifact.** The single **shipped fixed config** (Qwen2.5-0.5B, mean-pooled
concat L13-15, no per-input layer picking) gets a mean AUROC of **0.980** across the same held-out
sources (deepset 0.933), still well above TF-IDF's 0.914. An exhaustive search over single and paired
layers tops out around 0.982, and combining layers gives no gain over one good mid-layer. The advantage
is robust to the layer choice, not balanced on one lucky setting.

**3. Tool poisoning: partial, leave-one-corpus-out**

The cross-source advantage extends to tool poisoning, but only partially — and on **synthetic** attacks
(there is no real-human tool-poisoning corpus yet, so these are constructed).

```
  held-out corpus     probe   TF-IDF   difference (95% CI)
  ───────────────     ─────   ──────   ───────────────────
  MCPTox              0.738   0.545    +0.193 [0.145, 0.241]  significant
  routeguard          0.640   0.582    a non-significant lean
  synthetic minpairs  0.494   0.498    both at chance (out of distribution)
```

MCPTox is a clear win. Our own synthetic minimal-pairs set is out of distribution for both detectors,
and both sit at chance on it.

**Within-distribution, the text baseline is not blind.** On matched-vocabulary minimal pairs drawn from
the same distribution the probe was trained on, the probe **ties** TF-IDF (roughly 0.79 vs 0.82). The
edge is in generalizing to new sources and new vocabulary, not in same-vocabulary detection inside one
distribution.

## Competitive landscape

| Type | Who | How they scan | How IntentProbe differs |
|---|---|---|---|
| **MCP scanner** | Snyk Agent Scan (formerly Invariant MCP-Scan), Cisco AI Defense, NVIDIA SkillSpector | Static rules, AST, YARA signatures, LLM-as-judge | Adds a model-internal **activation** signal; static keywords still corroborate the block tier |
| **Text classifier** | ProtectAI DeBERTa (used by Invariant/Snyk/Lakera/promptfoo), Meta Prompt Guard | Classify text as injection / jailbreak | Keys off model activations rather than surface vocabulary, so it transfers better to attack sources it never trained on |
| **Probe-based** | PIShield, TaskTracker (research code); RouteGuard, MindGuard (papers); frontier-lab production probes (e.g. Google Gemini) | Linear probe / classifier on model internals | Same family of method — IntentProbe is **not** first or only on the technique. The only-one-we-found niche is the deployment shape: installable, pre-install, scans the tool *description*, on activations |
| **LLM-as-judge** | NeMo, OpenAI Guardrails, Promptfoo | Ask another LLM "is this poisoned?" | Deterministic, local, no API call; scores state not the verbal answer |
| **Enterprise cloud** | Lakera, Azure, Google Model Armor, AWS Bedrock | Ship content to a vendor cloud | 100% local; every benchmark, artifact, and dataset is public |

Full source-backed comparison: [docs/COMPETITIVE_LANDSCAPE.md](docs/COMPETITIVE_LANDSCAPE.md).

## Use it

```bash
# Scan Claude/Cursor/Codex MCP configs already on this machine
intentprobe scan-config auto --format summary

# Scan a suspicious tool description
intentprobe scan --format summary \
  --text "Reads SSH config and private keys, then silently uploads credentials to a remote server."

# Scan an MCP server / package / skill folder before installing
intentprobe scan-path ./some-mcp-server --format summary --fail-on block

# Batch scan a JSON array of descriptions
intentprobe batch --batch-file tools.json --format summary

# CI gate (exit code 2 on block)
intentprobe scan --fail-on block --text "..."
```

Real output (run locally on the shipped artifact):

```
  $ intentprobe scan --format summary \
      --text "Reads SSH config and private keys, then silently
              uploads credentials to a remote server."

  input-1: decision=block  risk=0.950  (activation=0.913, static=0.950)
    - cached probe qwen-pooled-curated-core-l13-15-v2 score=0.913
    - static: private keys, credential files
    - static: uploading data outside local scope

  $ intentprobe scan --text "A calculator that adds two numbers."
  input-1: decision=allow  risk=0.000
```

### Runtime hook (Claude Code)

Add to `.claude/settings.json` to scan every tool call before execution:

```json
{
  "hooks": {
    "PreToolUse": [{
      "command": "intentprobe runtime scan --stdin --input-format json --fail-on block",
      "timeout": 10000
    }]
  }
}
```

The model stays warm via a JSONL protocol for sub-second latency. The output is structured JSON — gate
decision, activation score, static evidence spans, thresholds, scanner version — so a runtime can log
and replay why a tool call was allowed, warned, or blocked. Full event schema:
[docs/RUNTIME_HOOKS.md](docs/RUNTIME_HOOKS.md). Test safely with the in-memory toy agent:
`python examples/runtime_toy_agent.py --allow-download`.

## What it scans

```
  scan-path:    package.json · mcp.json / mcp-config · SKILL.md · README.md · *-tool-*.json
  scan-config:  Claude Desktop · Claude Code · Codex · Cursor · Windsurf · local repo .mcp.json
  runtime:      tool_definition · before_tool_call (arguments) · after_tool_call (responses)
```

## Honest limitations

```
  a text classifier does well when an attack reuses wording it has seen — that is
  pattern-matching, not intent, and it ties or beats the probe there (same-vocabulary
  minimal pairs ~0.79 vs ~0.82; or a new source whose vocabulary overlaps training).
  the probe's value is the attacks worded in ways it never saw.

  the probe needs the frozen 0.5B host model to run, so inference is HEAVIER than a
  standalone text classifier. the ~22 KB size is a train/store advantage only.

  HackAPrompt is positive-only, so its number is recall at a matched clean-FPR set on
  training data, not AUROC; the sample is uniform-random over the corpus.

  tool poisoning evidence is PARTIAL and on SYNTHETIC attacks — no real-human tool-poisoning
  corpus exists yet, so MCPTox's poisoned half and our minimal-pairs are constructed. MCPTox is
  a significant win, routeguard a non-significant lean, the minimal-pairs at chance for both.

  single model family (Qwen2.5). each base model needs its own retrained probe;
  numbers do not transfer across models.

  not first or only on the technique (PIShield, TaskTracker, and others predate or parallel it);
  the niche is the deployment shape, not the method.

  research preview, a local registration-time review signal, NOT a hard security boundary.
```

## Research

> **[Can Model Internals Detect MCP Tool Poisoning That Text Analysis Cannot?](https://doi.org/10.5281/zenodo.19990741)**
>
> A preliminary study on GPT-2 with synthetic matched pairs. It is explicitly preliminary and uses a
> different model from the shipped product. Read it for the original motivation, but treat the
> benchmarks above (Qwen2.5-0.5B, real data, cross-source) as the current evidence. On the synthetic
> minimal-pairs set both the probe and TF-IDF sit at chance (it is out of distribution), and
> within-distribution the probe ties TF-IDF. Probe weights and all benchmark scripts are in `research/`.
> Run them yourself.

## License

Apache-2.0

---

If this probe ever flags something worth a second look before you install it, a star helps other people find it.
