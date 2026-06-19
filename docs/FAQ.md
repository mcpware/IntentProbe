# IntentProbe FAQ

## What is IntentProbe?

IntentProbe is a local scanner for MCP servers, AI agent tools, Claude Code
skills, packages, and runtime tool events. It looks for poisoned intent such as
credential access, secret exfiltration, hidden persistence, forced tool
chaining, or tool hijacking.

It is a **research preview**: a local, single-pass, registration-time review
signal, not a hard security boundary.

## What does "activation probing" mean?

IntentProbe runs a tool description through a small frozen local model
(Qwen2.5-0.5B) and reads the hidden activation state inside the model. Then a
small trained probe scores whether that internal state looks poisoned.

Simple version: most text scanners read the words; an LLM judge asks the model
for a verbal answer. IntentProbe scores the model's internal representation
instead — it reads activations, **not just** the text. That keys the signal off
how the model represents the input rather than the exact surface vocabulary.

One honest caveat about the product, not just the method: the activation probe
is the primary signal for `allow` / `warn`, but the `block` tier additionally
requires static-keyword corroboration to control false positives. So a novel,
no-keyword input the probe flags surfaces as `warn`, not `block`. The block tier
is not activation-only.

## Is this the same as asking Qwen if a tool is safe?

No. Asking Qwen "is this safe?" is an LLM-as-judge approach. IntentProbe uses
Qwen2.5-0.5B as a fixed feature extractor and reads hidden activations instead
of trusting the model's generated answer.

We tested the direct-prompt approach against the same Qwen2.5-0.5B sensor. The
deterministic label-score baseline flagged every clean curated item as poisoned
(clean false-positive rate = 1.000), and the generated-answer version missed
poison and produced many unparseable outputs. The reproducible baseline is in
[`research/QWEN_PROMPT_JUDGE_BASELINE_2026-06-08.md`](../research/QWEN_PROMPT_JUDGE_BASELINE_2026-06-08.md).

## Why not just use a text classifier?

A text classifier does well when an attack reuses wording it has already seen.
That is the common case, and there it ties or beats the probe — it is fast and
effective on familiar vocabulary.

The probe's value is generalization. When a text classifier trained on attack
examples then faces attacks from a source it never saw, the learned vocabulary
often does not transfer and recall drops. The probe keys off the model's
internal representation, so it holds up better on attack *sources* and *wording*
it never trained on. See the benchmarks below for the size of that effect (and
where it does not appear).

## Does IntentProbe upload my tool descriptions?

No. IntentProbe runs locally. Scan targets and scan results stay on your
machine.

The first model-backed scan may download Qwen2.5-0.5B once from Hugging Face
(~1 GB). After the model is cached, scans can run from local files.

## What model does v0 use?

The released v0 scanner uses Qwen2.5-0.5B as the frozen local sensor model and
reads mean-pooled mid-layer activations (layers 13-15). The shipped probe
artifact is about 22 KB (float64 logistic-regression weights).

Note on that 22 KB: it is a training-and-storage advantage, not a runtime one.
The probe needs the frozen 0.5B host model to produce activations, so inference
is **heavier** than a standalone text classifier, not lighter. The small probe
head is cheap to train and store; the host model is the cost at scan time.

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

## What are the benchmarks?

Everything here is reproducible from [`research/`](../research/), on the
**shipped Qwen2.5-0.5B** artifact. The thesis is generalization to attacks the
probe never trained on — and the results show both where that holds and where it
does not.

**1. Generalization to unseen real attacks (HackAPrompt, n=3,866 uniform-random,
a source neither detector trained on):**

HackAPrompt is a large set of attacks written by real people in a red-teaming
competition. Neither the probe nor the text baseline saw it during training. It
is positive-only (attacks, no benign), so we report recall at a clean
false-positive rate fixed on the training data — recall at a matched FPR, not
AUROC.

| Detector | recall @ 5% clean-FPR | recall @ 1% clean-FPR |
|---|---:|---:|
| Probe (Qwen2.5-0.5B, mean-pooled L13-15) | 90.3% | 88.3% |
| TF-IDF (same training data) | 52.8% | 30.3% |

Same training data, same held-out evaluation, same false-alarm budget. The text
classifier's learned vocabulary does not transfer to wording it never saw, so
recall drops; the probe holds up. Caveat: this is recall at a matched FPR set on
the training clean data, not a full AUROC, and the sample is uniform-random over
the corpus.

**2. Curated cross-source generalization (leave-one-source-out, nested CV, 4 PI
datasets):**

Train on three of {deepset, safeguard, spml, jayavibhav}, test on the held-out
fourth, repeat for each. Model and layer are chosen inside a nested
cross-validation loop, never on the held-out source. 95% bootstrap CIs on the
probe-minus-TF-IDF difference.

| held-out source | probe AUROC | TF-IDF AUROC | difference (95% CI) |
|---|---:|---:|---|
| deepset | 0.941 | 0.732 | +0.209 [0.168, 0.250] significant |
| spml | 0.995 | 0.935 | +0.059 [0.044, 0.077] significant |
| safeguard | 0.999 | 0.993 | +0.006 [0.002, 0.011] significant (at ceiling) |
| jayavibhav | 1.000 | 0.997 | +0.002 [0.000, 0.005] tie (CI touches 0) |
| **mean** | **0.984** | **0.914** | **+0.070** |

deepset is where the gap is widest: TF-IDF's vocabulary does not transfer and it
drops to 0.732, while the probe holds at 0.941. The single **shipped fixed
config** (Qwen2.5-0.5B, mean-pooled concat L13-15, no per-input layer picking)
reaches mean AUROC **0.980** across the same held-out sources (deepset 0.933) —
still above TF-IDF's 0.914, so the advantage is not balanced on one lucky
setting.

**3. Tool poisoning — partial, and on synthetic attacks:**

The cross-source advantage extends to tool poisoning only partially, and on
**synthetic** attacks (no real-human tool-poisoning corpus exists yet, so these
are constructed). Leave-one-corpus-out:

| held-out corpus | probe AUROC | TF-IDF AUROC | difference (95% CI) |
|---|---:|---:|---|
| MCPTox | 0.738 | 0.545 | +0.193 [0.145, 0.241] significant |
| routeguard | 0.640 | 0.582 | non-significant lean |
| synthetic minpairs | 0.494 | 0.498 | both at chance (out of distribution) |

MCPTox is a clear win. The synthetic minimal-pairs set is out of distribution for
both detectors, and both sit at chance on it.

**4. Within-distribution, the text baseline is not blind.** On matched-vocabulary
minimal pairs drawn from the same distribution the probe was trained on, the
probe **ties** TF-IDF (roughly 0.79 vs 0.82). The edge is generalizing to new
sources and new vocabulary, not same-vocabulary detection inside one
distribution.

## Is this a claim about every private cloud scanner?

No. The comparison baseline is a TF-IDF text classifier trained on the same data,
plus the public/source-verifiable PI datasets above. Private cloud/API scanners
may work well, but their detector artifacts and MCP/tool-poisoning benchmarks are
usually not reproducible by users, so we do not benchmark against them. The
honest comparison is "activation probe vs same-data text classifier," reproducible
end to end.

## Is this the first activation-probe scanner?

No. Probing model internals for safety is an established line of work: PIShield,
TaskTracker, RouteGuard, MindGuard, and frontier-lab production probes predate or
parallel IntentProbe. IntentProbe is **not** first or only on the technique.

The only-one-we-found niche is the deployment shape, not the method: a tool that
is installable, runs before install, scans the standalone tool/skill/MCP
*description*, and does it on model activations. That is an absence claim ("the
only one we found in this exact shape"), not "first ever." Full source-backed
comparison: [`docs/COMPETITIVE_LANDSCAPE.md`](COMPETITIVE_LANDSCAPE.md).

## Have you tried SAE features?

Yes. SAE features are useful for interpretability and may improve future recall.
The v0 product ships raw Qwen activations because the current raw-activation
artifact is the most complete and reproducible product path today.

SAE is planned as an optional layer for recall improvements and human-readable
explanations.

## Is v0 production-ready?

Use v0 as a pre-install review signal and runtime warning/blocking layer, not as
your only security boundary. Its value is generalizing to attacks worded in ways
it never trained on, where a same-data text classifier's recall drops. On
familiar-vocabulary attacks it ties or loses to a text classifier, and on the
synthetic minimal-pairs set both sit at chance. Novel attack families and
white-box adversarial attacks still need more work.

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
