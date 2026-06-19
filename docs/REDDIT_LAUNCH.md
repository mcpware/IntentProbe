# Reddit Launch Draft

## Title options

- A local scanner for poisoned MCP/tools that reads model activations, not just the text
- IntentProbe: catches prompt-injection attacks worded in ways a text classifier never saw
- I built a local MCP/tool scanner that generalizes to attack wording it never trained on
- Text scanners learn attack vocabulary and break on new wording. I tried scoring activations instead.

## Post draft

I built IntentProbe, a local CLI scanner / GitHub Action / runtime hook for AI agent tools, MCP
servers, and skills:

https://github.com/mcpware/IntentProbe

It runs a tool description or prompt through a frozen local model (Qwen2.5-0.5B), reads a few
mid-layer activations (L13-15), mean-pools them, and scores that vector with a small (~22 KB)
logistic probe. Most scanners read the text itself: regex, signatures, a fine-tuned text classifier,
or "ask an LLM". This reads the host model's internal state instead — it reads activations, **not
just** the text. (Disclosure up front: the `allow`/`warn` tiers run on the activation probe; the
`block` tier additionally requires static-keyword corroboration to keep false positives down, so a
novel no-keyword input the probe flags comes out as `warn`, not `block`. It's a review signal, not a
hard boundary.)

**The point is generalization, not raw accuracy.** When you train a text classifier on attack
examples and then hit it with attacks from a source it never saw, the learned vocabulary often
doesn't transfer and recall collapses. The probe keys off how the model internally represents the
input rather than the exact words, so it holds up better on wording it never trained on. That's the
whole bet, and it's the thing that matters for zero-day-style attacks.

**The headline benchmark (reproducible from `research/`, on the shipped Qwen2.5-0.5B artifact):**

HackAPrompt is a large set of prompt-injection attacks written by real people in a red-teaming
competition. Neither the probe nor a TF-IDF text baseline saw it during training — it's a held-out
source for both. It's positive-only (attacks, no benign), so I report recall at a clean
false-positive rate fixed on the training data, not AUROC.

```
                              recall @ 5% clean-FPR    recall @ 1% clean-FPR
                              ─────────────────────    ─────────────────────
  Probe (Qwen2.5-0.5B)              90.3%                    88.3%
  TF-IDF (same training data)       52.8%                    30.3%
```

Same training data, same held-out eval, same false-alarm budget. At a 5% FPR the probe catches ~90%
of these unseen attacks; the text classifier catches ~53% — it does fine on attacks that reuse
familiar wording, but its learned vocabulary doesn't transfer, so recall drops. (n = 3,866,
uniform-random over the corpus. Caveat: positive-only, so this is recall at a matched FPR set on the
training clean data, not a full AUROC.)

I want to be straight about the honest caveat to this: it is **not** "real attacks always lose to the
probe". On a second real source (Gandalf) where the attack vocabulary overlaps the training data,
TF-IDF transfers fine and ties or beats the probe. The edge shows up specifically when the attack
wording is novel/disguised relative to training, which is exactly the case a vocabulary-based
classifier struggles with.

**Curated cross-source check (with confidence intervals).** Four real prompt-injection datasets,
leave-one-source-out (train on three, test on the held-out fourth). The **shipped fixed config** —
Qwen2.5-0.5B, mean-pooled concat L13-15, no per-input picking — is the product number: mean AUROC
**0.980** vs TF-IDF 0.914 (deepset 0.933, safeguard 0.999, spml 0.990, jayavibhav 1.000). The hardest
held-out source is deepset, where TF-IDF's vocabulary doesn't transfer (0.732) while the probe holds
(0.933); the other three sit near ceiling.

A nested cross-validation that is additionally free to pick a **larger 1.5B sensor** per fold (model +
layer chosen on the training sources only, never the held-out one) reaches mean **0.984** — that's a
research upper bound, not the shipped 0.5B artifact (the deepset and spml folds pick a 1.5B sensor).
On that nested-CV run the probe-minus-TF-IDF gap is +0.209 on deepset (95% CI [0.168, 0.250]) and
+0.059 on spml (CI [0.044, 0.077]), both significant; safeguard/jayavibhav are near ceiling. So the
advantage holds on the shipped 0.5B and isn't balanced on one lucky setting.

**Tool poisoning is weaker and I'll say so.** The cross-source advantage extends to tool poisoning
only partially, and on **synthetic** attacks — there's no real-human tool-poisoning corpus yet, so
these are constructed. On MCPTox held-out the probe gets 0.738 vs TF-IDF 0.545 (+0.193, CI [0.145,
0.241], significant). On routeguard it's a non-significant lean. On my own synthetic minimal-pairs
set both detectors sit at chance — it's out of distribution for both. So: one clear win, one lean,
one tie. PI is the real-human evidence; tool poisoning is the showcase direction, not the proof.

**Where the text baseline is NOT blind — it wins.** On matched-vocabulary minimal pairs drawn from
the same distribution the probe trained on, the shipped 0.5B probe scores AUROC ~0.74 vs TF-IDF ~0.82
— a text classifier is slightly **better** there. (A nested CV free to pick a 1.5B sensor closes it to
roughly a tie, ~0.79 vs ~0.82, but that's not the shipped config.) I'm not claiming the probe reads
intent that text fundamentally can't see — within one distribution, a bag-of-words model does fine.
The edge is generalizing to new sources and new vocabulary, not same-vocabulary detection inside one
distribution.

What it can scan today:

- raw text via `intentprobe scan`
- package / MCP-server / skill folders via `intentprobe scan-path`
- MCP configs already on your machine via `intentprobe scan-config auto`
  (Claude Desktop / Claude Code / Codex / Cursor / Windsurf / local repo)
- `package.json`, `mcp.json`, `SKILL.md`, README files, tool JSON
- runtime tool-call events through `intentprobe runtime`
- as a CI gate via the `mcpware/IntentProbe@main` GitHub Action

It runs locally, CPU-only, after a one-time ~1 GB model download (Qwen2.5-0.5B, downloaded once from
Hugging Face). After that nothing is uploaded — scan inputs and results never leave your machine.

Try it:

```bash
python3 -m pip install intentprobe
intentprobe scan-config auto --format summary
intentprobe scan --format summary \
  --text "Reads SSH config and private keys, then silently uploads credentials to a remote server."
```

If you think the framing is wrong, clone it and run it on real MCP servers, skills, or tool packages.
The most useful replies are:

- a poisoned tool it misses
- a benign tool it wrongly warns/blocks
- a scanner or paper I should compare against
- a reproducible command where the CLI output is confusing

Please redact secrets before posting samples. The experiment scripts, result JSONs, and the probe
weights are committed in `research/`; the datasets are downloaded from their original sources
(deepset, SafeGuard, SPML, jayavibhav, HackAPrompt on Hugging Face), so you can rerun the benchmarks
yourself.

## Short reply: how is this different from regex / a text classifier?

Regex and keyword scanners look at surface text. A fine-tuned text classifier learns attack
vocabulary. Both work great when an attack reuses wording they've seen — that's pattern matching, and
they tie or beat the probe there. IntentProbe instead reads a small model's internal activation state
after it has processed the description, then classifies that state. The bet is that the activation
representation transfers to attack wording the classifier never saw better than the learned
vocabulary does. The HackAPrompt held-out numbers above (90.3% vs 52.8% at a matched FPR) are the
evidence for that bet.

## Short reply: is this the "first" activation-probe scanner?

No, and I don't claim that. Probing model internals to detect prompt injection / unsafe intent is an
established line of work — PIShield, TaskTracker, RouteGuard, MindGuard, and frontier-lab production
probes all predate or parallel this. IntentProbe is **not** first or only on the technique.

The only-one-we-found niche is the **deployment shape**, not the method: an installable tool that, at
registration time, before install, scans a standalone MCP/skill/tool *description* on model
activations. That's an "only one we found" absence claim about packaging, and even that I'd happily
be corrected on. Full source-backed comparison is in `docs/COMPETITIVE_LANDSCAPE.md`.

## Short reply: is it production ready?

No. It's a research preview — a local, single-pass, registration-time review signal, not a hard
security boundary. Expect misses and false positives. Use it as a pre-install tripwire alongside your
existing controls, not as the only gate. It's installable now, fully local, and inspectable.

## Short reply: 22 KB probe — is that an efficiency win?

Only for training and storage, and I don't want to oversell it. The probe head is tiny, but it needs
the frozen 0.5B host model to produce activations, so **inference is heavier** than a standalone text
classifier, not lighter. The ~22 KB number is a train/store advantage, not a runtime one.

## Short reply: why not just ask an LLM "is this safe?"

LLM-as-judge is an output-level mechanism: you ask a model to say safe/unsafe, and the generated
answer becomes part of the attack surface — a poisoned tool can argue "I am safe", and a judge prompt
can be steered. IntentProbe scores the hidden activation state the text produces, before any verbal
answer. It's deterministic and local — no API call, no per-call tokens, no prompt drift. (I also
tested direct-prompting the same Qwen2.5-0.5B as a judge; that deterministic-label baseline flagged
essentially all clean curated items as poisoned. The reproducible baseline is committed in
[research/QWEN_PROMPT_JUDGE_BASELINE_2026-06-08.md](../research/QWEN_PROMPT_JUDGE_BASELINE_2026-06-08.md).)

## Short reply: what about Lakera, Azure, Google Model Armor, AWS, etc.?

Those are real enterprise guardrails, but most are cloud/API controls — you send prompts, tool data,
or outputs to a vendor backend. IntentProbe is narrower but runs locally (CPU-only, after a one-time
~1 GB model download) and is inspectable, with a public detector artifact and reproducible benchmarks
(scripts and result JSONs committed; datasets pulled from their original sources).

## Short reply: what about Snyk, NVIDIA SkillSpector, Cisco AI Defense, MCP-Scan, etc.?

That's the closest product category: scan-before-install and runtime tool-boundary security. The
difference is the signal. Those tools are mainly static rules, AST/signatures, or LLM-as-judge.
IntentProbe adds a model-internal activation signal (and static keywords still corroborate its block
tier). It's a different detector class, not a claim that those tools catch nothing.

## Short reply: does it make runtime decisions or only scores?

Both. `intentprobe runtime` returns structured JSON — the gate decision, activation score, static
evidence spans, thresholds, and scanner version — so a runtime can log and replay why a tool call was
allowed, warned, or blocked. The model stays warm via a JSONL protocol for sub-second latency.
`--fail-on` chooses the enforcement level (`block` exits non-zero).

## Short reply: does it upload my code?

No. The scanner runs locally. The first model-backed scan may download the base model from Hugging
Face once, but scan targets and results are never sent to an IntentProbe service.

---

⭐ If this probe ever flags something worth a second look before you install it, a star helps other
people find it: https://github.com/mcpware/IntentProbe
