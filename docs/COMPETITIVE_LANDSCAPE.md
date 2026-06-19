# Competitive Landscape

Last checked: 2026-06-18.

This is the full, source-backed version of the comparison table in the
[README](../README.md#competitive-landscape). Everything here mirrors the
README's framing, numbers, and claim boundaries. The product numbers below come
from `research/_results_published/` and are reproducible from `research/`.

IntentProbe is a local scanner for MCP servers, skills, packages, and runtime
tool events. It runs a small frozen model (Qwen2.5-0.5B) locally, reads
mean-pooled mid-layer activations (layers 13-15), and scores them with a small
(~22 KB) logistic probe. The activation probe is the primary signal for
`allow` / `warn`; the `block` tier additionally requires static-keyword
corroboration to control false positives.

It reads activations, **not just the text** — that is the part that transfers to
attack sources and wording it never trained on. It is a **research preview**: a
local, single-pass, registration-time review signal, not a hard security
boundary.

## What IntentProbe is not

Two corrections up front, because the rest of this page only makes sense with
them.

- **It is not the first or only probe-based detector.** There is a substantial
  body of prior and parallel work that puts a linear probe / classifier on model
  internals: PIShield, TaskTracker (research code), RouteGuard, MindGuard
  (papers), and frontier-lab production probes (e.g. Google running activation
  probes on Gemini in production). Some of these predate IntentProbe; some are
  parallel. The technique is not new.
- **The ~22 KB probe is not a runtime efficiency win.** The probe needs the
  frozen 0.5B host model to produce activations, so inference is *heavier* than
  a standalone text classifier, not lighter. The ~22 KB (float64) is a
  training-and-storage advantage only.

The narrow niche we can actually defend is a **deployment shape**, not a method
or an accuracy crown:

> The only tool we found that is **installable**, scans a standalone
> tool/skill/MCP **description before install**, in a **single pass with no
> reference context**, and does it on the host model's **activations**.

That is an absence claim ("the only one we found"), not "first ever" and not
"most accurate". Static/regex/LLM-judge products are installable and scan
descriptions but read text; the activation/hidden-state probes in the research
literature read internals but ship as research code that runs as a runtime
monitor, not a pre-install description scanner.

## One-screen map

This is the same table as the README, expanded.

| Type | Who | How they scan | How IntentProbe differs |
|---|---|---|---|
| **MCP / agent scanner** | Snyk Agent Scan (formerly Invariant MCP-Scan), Cisco AI Defense, NVIDIA SkillSpector, MCP Scanner, MEDUSA, Sunglasses, Armorer Guard, ClawGuard, SkillsSafe, AgentSeal, mcpwn, MCPRadar | Static rules, AST, signatures, policy checks, proxies, optional LLM-as-judge or cloud verification | Adds a model-internal **activation** signal; static keywords still corroborate the block tier |
| **Text classifier** | ProtectAI / LLM Guard DeBERTa (used by Invariant/Snyk/Lakera/promptfoo), Meta Prompt Guard / Llama Prompt Guard 2 | Classify text as benign / prompt injection / jailbreak | Keys off model activations rather than surface vocabulary, so it transfers better to attack **sources** it never trained on. Within one distribution, on matched vocabulary, the text classifier is not blind — it ties or beats the probe |
| **Probe-based** | PIShield, TaskTracker (research code); RouteGuard, MindGuard (papers); frontier-lab production probes (e.g. Google Gemini) | Linear probe / classifier on model internals | Same family of method — IntentProbe is **not** first or only on the technique. The only-one-we-found niche is the deployment shape (installable, pre-install, scans the tool *description*, single pass, on activations) |
| **LLM-as-judge** | NeMo self-check, OpenAI Guardrails, Promptfoo graders | Ask another LLM "is this poisoned?" | Deterministic for a fixed artifact, local, no API call; scores the hidden state, not a generated verbal answer that is itself part of the attack surface |
| **Enterprise cloud** | Lakera Guard, Azure Prompt Shields, Google Model Armor, AWS Bedrock Guardrails, Pangea / CrowdStrike AI Guard, Cisco AI Defense, HiddenLayer | Ship prompts / tool calls / responses to a vendor cloud | 100% local; every benchmark, artifact, and dataset is public and reproducible from this repo |

## Short positioning

Enterprise guardrails ask a vendor backend. Text classifiers read surface
patterns. LLM judges ask another model for an opinion, and the generated answer
becomes part of the attack surface. Local rule scanners match known suspicious
patterns.

IntentProbe reads the local model's internal state after it has processed the
tool description — and then, for the `block` tier, asks static keywords to
corroborate before it hard-blocks.

## What the benchmarks actually show

These are the numbers from the README, reproducible from `research/`, on the
**shipped Qwen2.5-0.5B** artifact. The thesis is generalization: a tiny
activation probe that transfers to attacks worded in ways it never trained on
(held-out sources, novel vocabulary) better than a same-data text classifier. It
ties or loses on familiar-vocabulary attacks.

### 1. Generalization to unseen attacks — HackAPrompt (real human attacks)

HackAPrompt is a large set of attacks written by real people in a red-teaming
competition. Neither the probe nor the TF-IDF text baseline saw it during
training. It is positive-only (attacks, no benign), so we report recall at a
clean false-positive rate fixed on the training data, not AUROC.

```
  n = 3,866 (uniform-random over the corpus, held-out source)

                                 recall @ 5% clean-FPR    recall @ 1% clean-FPR
                                 ─────────────────────    ─────────────────────
  Probe (Qwen2.5-0.5B, L13-15)          90.3%                    88.3%
  TF-IDF (same training data)           52.8%                    30.3%
```

Same training data, same held-out evaluation, same false-alarm budget. The
text classifier does fine on attacks that reuse familiar wording but its learned
vocabulary does not transfer to wording it never saw, so recall drops. The probe
keys off the model's internal representation, so it holds up. Caveat:
positive-only, so this is recall at a matched FPR, not a full AUROC, and the
sample is uniform-random, not an exhaustive panel.

### 2. Curated cross-source generalization — leave-one-source-out, nested CV

Train on three of {deepset, safeguard, spml, jayavibhav}, test on the held-out
fourth, repeat. Model and layer are chosen inside a nested cross-validation
loop, never on the held-out source. 95% bootstrap CIs on the probe-minus-TF-IDF
difference.

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

deepset is where the gap is widest: TF-IDF's vocabulary does not transfer to the
held-out source and it drops to 0.732, while the probe holds at 0.941. The other
three are near ceiling, so there is less room to separate.

The single **shipped fixed config** (Qwen2.5-0.5B, mean-pooled concat L13-15, no
per-input layer picking) gets a mean AUROC of **0.980** across the same held-out
sources (deepset 0.933), still well above TF-IDF's 0.914. The advantage is
robust to the layer choice, not balanced on one lucky setting.

### 3. Tool poisoning — partial, and on synthetic attacks

The cross-source advantage extends to tool poisoning, but only partially — and
on **synthetic** attacks. There is no real-human tool-poisoning corpus yet, so
these are constructed (MCPTox's clean half points at real MCP repos, but the
poisoned half is template-injected; minimal pairs are ours).

```
  held-out corpus     probe   TF-IDF   difference (95% CI)
  ───────────────     ─────   ──────   ───────────────────
  MCPTox              0.738   0.545    +0.193 [0.145, 0.241]  significant
  routeguard          0.640   0.582    a non-significant lean
  synthetic minpairs  0.494   0.498    both at chance (out of distribution)
```

MCPTox is a clear win. routeguard leans the same way but the CI touches zero.
Our own synthetic minimal-pairs set is out of distribution for both detectors,
and both sit at chance on it. One of three corpora is a significant win.

### 4. Within-distribution / same-vocabulary — the text baseline is not blind

On matched-vocabulary minimal pairs drawn from the same distribution the probe
was trained on, the probe **ties** TF-IDF: roughly 0.79 vs 0.82 AUROC overall
(on the strongest "innocuous-word-swap" subset, 0.797 vs 0.853, with the
difference CI crossing zero). The edge is generalizing to new sources and new
vocabulary, not detecting same-vocabulary attacks inside one distribution. We do
**not** claim the probe "catches the words a text classifier can't" — within a
distribution, it does not.

## Direct MCP / agent scanner competitors

### Snyk Agent Scan

Snyk Agent Scan is the closest public product-shaped competitor. Its README says
it discovers and scans agent components, MCP servers, and skills for prompt
injections and vulnerabilities. It supports Claude, Cursor, Windsurf, Gemini
CLI, VS Code, Claude Code, and other agent surfaces.

Important public details:

- It can execute stdio MCP server commands to retrieve tool descriptions, with
  interactive consent by default.
- It validates components with local checks and by invoking the Agent Scan API.
- Its README states that skills, agent applications, tool names, and
  descriptions are shared with Snyk for analysis.
- Its background mode reports results to a Snyk Evo instance for enterprise
  monitoring.

Source: <https://github.com/snyk/agent-scan>

Comparison:

- Snyk is a real agent scanner, not a toy baseline.
- Its public client exposes a scan-and-upload / API validation shape; the remote
  detector is opaque from the user's machine.
- Its public material does not describe an activation-probe method.
- We did not benchmark against Snyk's hosted detector directly; its public repo
  does not provide a user-reproducible accuracy benchmark we could measure
  against. Our reproducible head-to-head is against the TF-IDF text baseline
  trained on the same data and the ProtectAI DeBERTa classifier (see text
  classifiers below), not against Snyk's backend.

### Former Invariant MCP-Scan

Invariant MCP-Scan was an MCP-focused scanner with static scan and proxy modes.
The public Invariant docs describe scanning Claude, Cursor, Windsurf, and other
MCP client configurations; checking tool descriptions for prompt injection and
tool poisoning; monitoring MCP traffic; enforcing tool restrictions; detecting
tool shadowing; and pinning tools to detect rug pulls.

Sources:

- <https://invariantlabs-ai.github.io/docs/mcp-scan/>
- <https://explorer.invariantlabs.ai/docs/mcp-scan/>
- <https://github.com/invariantlabs-ai/explorer>

Comparison:

- Strong MCP product shape and operational scanner / proxy concept.
- Public docs describe rules, guardrails, hashing, proxying, and external
  verification, not activation-probe internals.
- The Invariant GitHub route now redirects toward Snyk Agent Scan; hosted
  Explorer material points users toward Snyk AI Security.

### NVIDIA SkillSpector and other recent local scanners

This space is filling quickly. NVIDIA SkillSpector (`pip install skillspector`,
released 2026-06-17) is a recent installable entrant in the same install-time
moment. The most relevant current public scanners we found are below.

| Product / project | Public positioning | Method shape from public material | How IntentProbe differs |
|---|---|---|---|
| NVIDIA SkillSpector | Installable scanner for agent skills / tool definitions | Static / pattern checks before use | Same install-time moment; IntentProbe's signal is a model activation probe |
| MCP Scanner | Open-source MCP scanner for tool poisoning, prompt injection, rug pulls, cross-origin escalation | Rule categories and MCP security checks | IntentProbe adds a model-internal activation signal |
| MEDUSA | AI security scanner with 9,600+ detection rules | Large rule / pattern catalog, SAST-style | IntentProbe is narrower but model-internal rather than rule-count driven |
| Sunglasses | Local open-source agent scanner / filter | Pattern catalog, keywords, normalization | IntentProbe's differentiator is activation state, not text-pattern coverage |
| Armorer Guard | Local Rust scanner for prompts, outputs, tool args, MCP proxying | Fast local structured rule / scoring boundary | IntentProbe is slower but uses a learned activation probe |
| ClawGuard | Security scanning for AI agent skills; CLI / registry / hooks | Scanner plus hooks / proxy / registry | IntentProbe focuses the core signal on activation-probed intent |
| SkillsSafe | Skill scanner for SKILL.md, MCP configs, system prompts | Skill / MCP pattern scanning before install | Same install-time moment, different signal class |
| AgentSeal | Open-source security scanner for AI agents and MCP / tool poisoning | Red-team / scanner positioning | Public material does not show activation-probe scanning |
| mcpwn / MCPRadar / MEOK-style | MCP security scanners for injection, poisoning, path traversal, SSRF | MCP-specific rules, probes, protocol checks | Good complementary hygiene; not activation probing |

Representative sources:

- <https://mcpscanner.cloud/>
- <https://pantheonsecurity.io/>
- <https://sunglasses.dev/open-source-ai-agent-security-scanner>
- <https://armorerlabs.com/blog/armorer-guard-inline-prompt-injection-defense>
- <https://www.clawguard.sh/>
- <https://skillssafe.com/en>
- <https://agentseal.org/>
- <https://safematix.com/mcpwn/>
- <https://mcpradar.dev/>
- <https://mcpservers.org/es/servers/csoai-org/meok-mcp-injection-scan-mcp>

Comparison:

- These products make the category more real, which helps IntentProbe — they
  prove scan-before-install and runtime tool-boundary scanning are becoming
  normal expectations.
- Most public descriptions emphasize pattern / rule coverage, latency, MCP proxy
  placement, credential redaction, and known attack categories.
- We did not find another installable local scanner whose primary signal is a
  model activation probe and that scans a standalone tool description before
  install. That is the deployment-shape niche, not an accuracy claim.

## Probe-based detectors (same method family)

IntentProbe is not first or only on the technique. The work below puts a probe
or classifier on model internals; some predates IntentProbe, some is parallel.

- **PIShield** (arXiv:2510.14005) — a frozen LLM with a linear probe on residual
  activations, runtime input filter (Llama-3.1-8B). Near-twin on method; it runs
  as a runtime monitor and does not scan a standalone tool description before
  install, and it is not on PyPI as `pishield` (that name resolves to an
  unrelated package). Dated 2025-10, ahead of IntentProbe.
- **TaskTracker** (Microsoft, 2024) — task-drift detection on model internals.
- **RouteGuard** (arXiv:2604.22888) — hidden-state plus attention features for
  skill poisoning, on a Skill-Inject benchmark; same niche, runtime monitor.
- **MindGuard** — hidden-state probing (paper / research code).
- **Frontier-lab production probes** — Google has reported running activation
  probes on Gemini in production (arXiv:2509.03888, 2601.11516). So "first to use
  activation probing for safety" is false.

What is different about IntentProbe is only the deployment shape: it is
packaged, installable (`pip install intentprobe`), runs locally, and scans a
standalone tool / skill / MCP **description** at registration time in a single
pass. The research probes above ship as clone-and-`pip install -e .` research
code and run as runtime monitors over a live session, not as a pre-install
description scanner.

## Enterprise cloud / API guardrails

These are serious enterprise controls. The issue is not that they are useless —
it is that a developer often cannot independently verify the detector internals
or reproduce the advertised accuracy on a local MCP / tool-poisoning benchmark,
and SaaS / API modes mean prompts, tool data, or outputs are sent to a provider.

### Lakera Guard

Lakera Guard documents real-time visibility, threat detection, prompt-attack
detection, data-leakage controls, centralized policy management, and SaaS /
self-hosted deployment. Its integration docs describe calling the Lakera Guard
API for user interactions or agent steps.

Sources:

- <https://docs.lakera.ai/guard>
- <https://docs.lakera.ai/docs/api/guard>

Comparison: good enterprise control plane; the detector and benchmark harness
are not reproducible by the user from the docs; SaaS mode sends prompts to
Lakera's API. IntentProbe runs locally and exposes its benchmark scripts.

### Microsoft Azure Prompt Shields

Azure Prompt Shields targets user prompt attacks and document / indirect prompt
injection (system-rule changes, conversation mockups, role-play, encoding
attacks).

Source:
<https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/content-filter-prompt-shields>

Comparison: strong cloud-platform integration; detector internals and benchmark
set are not exposed as a reproducible scanner artifact; built for prompt /
document filtering around Azure AI workloads, not a local scan-before-install
MCP / skill scanner.

### Google Cloud Model Armor

Model Armor screens prompts and responses, supports prompt injection / jailbreak
detection, sensitive-data protection, malicious-URL detection, and confidence
thresholds.

Source: <https://docs.cloud.google.com/model-armor/overview>

Comparison: useful cloud AI security layer; a Google Cloud service, not a local
open scanner; public docs describe configuration and thresholds, not a
reproducible MCP / tool-poisoning benchmark.

### Amazon Bedrock Guardrails

Bedrock Guardrails supports prompt-attack filters through the console or API.
AWS requires tagging user input for prompt-attack filtering in InvokeModel and
InvokeModelWithResponseStream; without tags the filter does not apply.

Source:
<https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-prompt-attack.html>

Comparison: strong inside Bedrock workflows; cloud / API rather than local
scanner; depends on app-side tagging; public docs do not provide a reproducible
MCP / tool-poisoning benchmark.

### Pangea / CrowdStrike AI Guard and Prompt Guard

Pangea's docs describe AI Guard and Prompt Guard as API / SDK services for
detecting direct and indirect prompt injection, malicious content, PII, and
other AI traffic risks.

Sources:

- <https://pangea.cloud/docs/ai-guard>
- <https://pangea.cloud/docs/prompt-guard/>

Comparison: enterprise API guardrail; the detection backend and benchmark
details are vendor-side; useful for production app traffic, not a local
activation-probe scanner for install-time MCP / tool descriptions.

### Cisco AI Defense and HiddenLayer

Cisco AI Defense documents runtime protection and an Inspection API for prompt
injection, denial-of-service, and data leakage. HiddenLayer documents AI runtime
security for prompt attacks, jailbreaks, unsafe outputs, and malicious tool use.

Sources:

- <https://developer.cisco.com/docs/ai-defense-inspection/>
- <https://docs.hiddenlayer.ai/docs/products/aidr-g/overview>
- <https://www.hiddenlayer.com/platform/ai-runtime-security>

Comparison: serious enterprise AI security stacks; public docs do not disclose
enough detector / benchmark detail to reproduce MCP / tool-poisoning accuracy.
IntentProbe is narrower, but local and inspectable.

## Text classifier competitors

### ProtectAI / LLM Guard DeBERTa

LLM Guard's prompt-injection scanner uses a fine-tuned DeBERTa classifier: a
binary prompt-injection model (`0` no injection, `1` injection), not recommended
for system prompts.

Sources:

- <https://github.com/protectai/llm-guard/blob/main/docs/input_scanners/prompt_injection.md>
- <https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2>

Comparison:

- A real local text classifier. It detects classic prompt-injection strings
  correctly.
- In our cross-dataset prompt-injection tests, a fine-tuned text classifier of
  this kind generalizes well when the attack vocabulary overlaps what it has
  seen, and can match or beat the probe there. The probe's advantage is on
  held-out sources and novel vocabulary, not on familiar-vocabulary attacks.
- We do not claim the probe "ties" or "beats" any specific fine-tuned DeBERTa
  number — our reproducible head-to-head is the probe vs a **TF-IDF baseline
  trained on the same data**, reported in section 2 above and in `research/`.

### Meta Prompt Guard

Meta's Prompt Guard and Llama Prompt Guard 2 are text-classification models for
benign / injection / jailbreak. Llama Prompt Guard 2 is a fine-tuned
BERT / DeBERTa-style classifier for direct jailbreak and prompt-injection
attacks.

Sources:

- <https://huggingface.co/meta-llama/Prompt-Guard-86M>
- <https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M>
- <https://meta-llama.github.io/PurpleLlama/LlamaFirewall/docs/documentation/scanners/prompt-guard-2>

Comparison: small and local-friendly; useful as a prompt / jailbreak classifier;
still a text classifier, and its model cards do not make it an MCP tool-intent
activation scanner.

## LLM-as-judge and red-team frameworks

### NVIDIA NeMo Guardrails self-checking

NeMo Guardrails documents `self_check_input`, where the LLM is prompted to answer
whether the input should be allowed. NVIDIA notes performance depends strongly on
the LLM's ability to follow the self-check prompt.

Source:
<https://docs.nvidia.com/nemo/guardrails/latest/configure-rails/guardrail-catalog/self-check.html>

Comparison: flexible and easy to understand; it asks another LLM to judge, which
costs tokens, adds latency, and varies with model / version / prompt; the
generated answer is itself part of the attack surface. IntentProbe's probe score
is deterministic for a fixed artifact and scores the hidden state, not a verbal
answer.

### OpenAI Guardrails prompt-injection check

OpenAI Guardrails documents a prompt-injection detection check using LLM-based
analysis on function calls and tool-call outputs, with a configurable model,
confidence threshold, and token usage.

Source:
<https://openai.github.io/openai-guardrails-js/ref/checks/prompt_injection_detection/>

Comparison: strong agent-flow alignment check; explicitly LLM-based analysis,
not a local activation scanner; better compared to runtime judge guardrails than
to install-time MCP scanner artifacts.

### Promptfoo red-team graders

Promptfoo is a strong eval / red-team framework with red-team attack generation
and grading; graders can be LLM-based and configurable.

Sources:

- <https://www.promptfoo.dev/docs/red-team/configuration/>
- <https://www.promptfoo.dev/docs/red-team/troubleshooting/grading-results/>

Comparison: excellent for testing an app or agent; not the same job as a local
scanner that runs before installing a tool; LLM-based grading is useful for
audits but not ideal as a cheap deterministic runtime hook.

### garak and Giskard

garak is an LLM vulnerability scanner with prompt-injection probes. Giskard
provides LLM vulnerability scanning and detectors for injection-style failures.

Sources:

- <https://docs.garak.ai/garak/examples/prompt-injection>
- <https://docs.giskard.ai/hub/sdk/scan/index.html>
- <https://docs.giskard.ai/en/latest/reference/scan/llm_detectors.html>

Comparison: useful for red-team campaigns and vulnerability assessment; they
test whether a target LLM / app can be made to fail. IntentProbe is aimed at a
different moment — before trusting a tool, skill, MCP server, package, or runtime
tool event.

## What we can say publicly

The defensible claim, with all three qualifiers:

> The only tool we found that is installable, scans a standalone tool / skill /
> MCP description before install in a single pass with no reference context, and
> does it on the host model's activations. It runs locally and ships with
> reproducible benchmark artifacts.

Why this is defensible:

- We found public MCP / agent scanners, cloud guardrails, prompt-injection text
  classifiers, LLM-as-judge guardrails, and red-team frameworks — all read text,
  rules, or ask another model.
- We found activation / hidden-state probes in the research literature
  (PIShield, TaskTracker, RouteGuard, MindGuard) and in production (Google /
  Gemini) — none ships as an installable pre-install description scanner; they
  run as runtime monitors.
- IntentProbe publishes the scanner artifact, methodology files, benchmark
  scripts, and results in this repo.

What we do **not** claim:

- Not the first or only probe-based / activation-based detector. The technique
  predates and parallels us.
- Not "more accurate than incumbents." Within a distribution, on matched
  vocabulary, a text classifier ties or beats the probe. The edge is cross-source
  / novel-vocabulary generalization, and on tool poisoning it is partial and on
  synthetic data.
- Not a runtime efficiency win. The ~22 KB head needs the frozen 0.5B host, so
  inference is heavier than a standalone text classifier.
- Not a hard security boundary. It is a research-preview registration-time review
  signal; the block tier needs static-keyword corroboration.
- The tool-poisoning evidence is partial and on synthetic attacks — no
  real-human tool-poisoning corpus exists yet.

## Clean public soundbite

Most scanners ask: "Does this text look suspicious?"

IntentProbe asks: "When a small model reads this tool description, does its
internal state look like it understood a malicious capability?" — and on attacks
worded in ways it never trained on, that question transfers where surface
vocabulary does not.
