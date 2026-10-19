# IntentProbe title-positioning review — 2026-10-18

**Status:** superseded by the final public wording adopted later on 18 October 2026:
*IntentProbe: One Local Model Pass, Many Task-Specific Probes*. The alternatives below are retained
as the decision record rather than current recommendations.

## Source coverage

| Status | Source | Actual coverage | Use |
|---|---|---|---|
| ✅ GOT DATA | Local IntentProbe release | README, public note, manuscript, disclosure, citation metadata, figure source, release records, and prior-art review | Located every public title and separated the architecture claim from the 16-head measurement. |
| ✅ GOT DATA | arXiv primary records | TaskTracker paper, PIShield, RouteGuard, *A BERTology View of LLM Orchestrations*, *Representation-as-a-Judge*, and *You Only Judge Once* | Verified the exact titles, dates, abstracts, and title-to-contribution relationship. |
| ✅ GOT DATA | ACL Anthology | LatentGate official record and paper | Verified the published title and its benefit + task + method structure. |
| ✅ GOT DATA | Official GitHub repositories | Microsoft TaskTracker, PIShield, and LatentGate | Checked how the paper name is carried into a project heading and plain repository description. |
| ✅ GOT DATA | Adjacent primary-source discovery | Single-pass classification, frozen-backbone probing, shared hidden states, and multi-task readout queries | No materially different title pattern appeared after the named peers and closest system papers were opened. |
| ⏭️ SKIPPED | Reddit, Hacker News, Product Hunt, news, and vendor blogs | Not needed for an academic/project title comparison | These sources could measure reception, but they are not authoritative for the works' actual titles or methods. |
| ⏭️ SKIPPED | Paid academic and SEO databases | Not needed | Primary paper and repository records answered the naming question directly. |

## What the closest projects actually do

| Work | Public title | What the title leads with |
|---|---|---|
| [PIShield](https://arxiv.org/abs/2510.14005) | *PIShield: Detecting Prompt Injection Attacks via Intrinsic LLM Features* | Brand, exact threat, then the internal signal. |
| [RouteGuard](https://arxiv.org/abs/2604.22888) | *RouteGuard: Internal-Signal Detection of Skill Poisoning in LLM Agents* | Brand, method category, exact threat, and deployment setting. |
| [LatentGate](https://aclanthology.org/2026.acl-industry.153/) | *LatentGate: Low-Latency Semantic Routing via Frozen-Backbone Probing of Small Language Models* | Brand, user-visible benefit and task, then the mechanism. |
| [TaskTracker](https://arxiv.org/abs/2406.00799v6) | *Get my drift? Catching LLM Task Drift with Activation Deltas* | A memorable hook followed by the phenomenon and signal. Verified again on 19 October: current v6 uses this title; [v4](https://arxiv.org/abs/2406.00799v4) used *Are you still on track!? Catching LLM Task Drift with Activations*. The external review had mistaken that intermediate title for the current one. The [repository](https://github.com/microsoft/TaskTracker) keeps the product name and explains the same job plainly. |
| [A BERTology View of LLM Orchestrations](https://arxiv.org/abs/2601.13288) | *A BERTology View of LLM Orchestrations: Token- and Layer-Selective Probes for Efficient Single-Pass Classification* | System setting, exact readout design, and the single-pass benefit. |
| [Representation-as-a-Judge](https://arxiv.org/abs/2601.22588) | *Rethinking LLM-as-a-Judge: Representation-as-a-Judge with Small Language Models via Semantic Capacity Asymmetry* | Names the old paradigm, the replacement paradigm, and the explanation. |
| [You Only Judge Once](https://arxiv.org/abs/2604.10966) | *You Only Judge Once: Multi-response Reward Modeling in a Single Forward Pass* | A memorable systems hook followed by the exact one-pass mechanism. |

The repeated pattern is not “project name + the number in the first experiment.” It is “memorable
name or hook + enduring capability + method or deployment setting.” Result counts belong in the
abstract, badge, figure, or evidence line.

## Decision

### Public architecture title

> **IntentProbe: One Model Pass, Many Task-Specific Readouts**

This states the durable architecture. It does not cap the system at 16 heads, promise an unlimited
number of heads, or imply that every score is already an action decision.

### Research manuscript title

> **IntentProbe: Many Task-Specific Hidden-State Readouts from One Small Language Model Forward Pass**

This follows the peer pattern: project name, contribution, signal, and system shape. The title still
fits the current experiment, but it will remain accurate when the head registry grows beyond the
first 16-head workload.

### Headline proof

> **Measured proof: serving 16 heads added 6.9% mean CPU latency over serving one.**

Sixteen remains prominent because it is the measured evidence. It is no longer presented as the
definition or limit of IntentProbe.

## Why “readouts,” not “decisions”

An activation head returns a task-specific score. Calibration turns that score into a usable signal.
Host policy decides whether to allow, warn, review, route, save, or block. Calling every head output
a “decision” collapses those layers and conflicts with the architecture's own policy boundary.
“Readout” describes the shared mechanism across toxicity, jailbreak, sentiment, routing,
authorization, memory, and future admitted tasks without claiming that one fixed set of 16 heads is
the product.

## Claims deliberately avoided

- “Any number of heads” would imply an unmeasured capacity guarantee.
- “Unlimited decisions” would be false and would hide memory, compatibility, and admission limits.
- “Sixteen learned decisions” turns the first benchmark configuration into an apparent product cap.
- A latency number in the title makes a versioned measurement look like the identity of the method.

## Retrieval ledger

| Canonical query or direct record | Primary destination | Result |
|---|---|---|
| `PIShield prompt injection intrinsic LLM features` | `arxiv.org/abs/2510.14005`, `github.com/weizou52/PIShield` | Exact paper and repository wording verified. |
| `RouteGuard hidden state alignment response-conditioned attention` | `arxiv.org/abs/2604.22888` | Exact paper wording verified. |
| `TaskTracker activation deltas prompt injection` | `arxiv.org/abs/2406.00799`, `github.com/microsoft/TaskTracker` | Exact paper hook, technical subtitle, and repository framing verified. |
| `LatentGate frozen-backbone probing semantic routing` | `aclanthology.org/2026.acl-industry.153`, `github.com/shivoxaham/latentgate` | Published title and repository identity verified. |
| `single forward pass hidden states multiple classifiers language model` | arXiv and ACL primary records | Recovered the closest orchestration, representation-judge, and single-pass reward-model titles; no new title structure changed the decision. |

