---
license: apache-2.0
language:
  - en
library_name: intentprobe
base_model: openbmb/MiniCPM5-1B-Base
datasets:
  - ProCreations/auto-1b-data
  - ProCreations/approve-or-deny
tags:
  - activation-probe
  - hidden-states
  - cpu-inference
  - local-llm
  - linear-probe
  - ai-agents
  - agent-security
  - authorization
  - permission
  - approval-detection
  - intent-detection
  - tool-use
  - function-calling
  - mcp
  - security
  - llm-security
  - ai-safety
  - guardrails
  - runtime-safety
  - local
  - minicpm
metrics:
  - roc_auc
---

# IntentProbe: 28 KB activation probe for AI-agent action authorization

**Author:** [Wan Sheng Leung](https://github.com/ithiria894) · **Publisher:** MCPware

[Architecture: one local model pass, many task-specific probes](https://mcpware.github.io/IntentProbe/architecture/)
· [All IntentProbe models and research](https://github.com/mcpware/IntentProbe)

[![GitHub](https://img.shields.io/badge/GitHub-mcpware%2Fintentprobe-181717?logo=github)](https://github.com/mcpware/intentprobe)
[![PyPI](https://img.shields.io/pypi/v/intentprobe?label=PyPI)](https://pypi.org/project/intentprobe/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](./LICENSE)

**A local AI-agent permission detector for tool-use authorization.** This 28 KB cross-layer
activation probe estimates whether a proposed structured action is supported by the user's request
and conversation history before an MCP server, function-calling agent, or automation runtime acts.

It is designed as a secondary runtime safety signal for approval detection, risky tool-call
preflight, and agent permission gates. MiniCPM5 ranked first among six completed frozen activation
backbones in our CPU benchmark.

The MiniCPM5 backbone stays frozen. IntentProbe reads attention-mask-aware mean-pooled hidden outputs
**13 and 14**, standardizes each layer with train-only statistics, averages the two representations,
and applies a class-weighted logistic head.

Highlights:

- **Best of six frozen activation backbones** by validation AUPRC: 0.9228.
- **Intermediate layers beat the default last-layer assumption:** outputs 13–14 were selected.
- **Positive separate-test ranking result:** AUROC 0.8137 versus 0.7897 for matched-budget TF-IDF.
- **Tiny task-specific artifact:** the learned probe and normalization statistics fit in 28 KB.
- **Reproducible release:** pinned model, data revisions, scorer, uncertainty intervals, and hashes.

## AI-agent authorization use cases

- Score whether a proposed tool call matches the user's request and conversation history.
- Add an independent approval signal before send, submit, modify, or other consequential actions.
- Compare activation-based authorization detection with surface-text rules and classifiers.
- Study where permission and approval intent appears inside small, locally runnable language models.

This is especially relevant to MCP security, function calling, AI-agent guardrails, and local-first
runtime control. The score is designed to complement an exact, deterministic authorization state
machine rather than replace it.

## What is included

| Component | Value |
| --- | --- |
| Frozen backbone | [`openbmb/MiniCPM5-1B-Base`](https://huggingface.co/openbmb/MiniCPM5-1B-Base) |
| Pinned backbone revision | `156170697656c48f69915b33a2fb44110242187c` |
| Representation | masked mean of hidden outputs 13–14 after per-layer standardization |
| Maximum input | 192 tokens, right truncation |
| Learned component | class-weighted logistic regression |
| Probe size | 28,708 bytes |
| Training budget | 1,600 silver-labelled rows |
| Labels | `deny=0`, `approve=1` |
| Release type | Open research artifact with reproducible local scorer |

The MiniCPM5 weights are not included. The first run downloads the roughly 1B-parameter base model;
28 KB is the size of the learned probe, not the total runtime footprint.

## Quick start

```bash
python3 -m pip install "transformers>=5.8,<6" "torch>=2.2" "numpy>=1.26" huggingface_hub
hf download mcpware/intentprobe-minicpm5-1b-authorization --local-dir ./intentprobe-auth
python3 ./intentprobe-auth/score.py \
  --artifact ./intentprobe-auth/probe.npz \
  --call-json '{"tool":"send_email","args":{"to":"sam@example.com","body":"The draft"}}' \
  --user-request 'Show me the draft before sending.' \
  --history 'The assistant displayed a draft but the user has not approved it.'
```

The output is an `approve_score`, not an execution decision. A production system should combine the
score with a deterministic authorization state machine that binds the exact payload, recipient,
scope, preview, approval, and any changes made after approval.

The benchmark renderer is fixed as:

```text
PROPOSED ACTION:
{call}
USER REQUEST:
{user_request}
HISTORY:
{history}
```

`call` must use the structured JSON tool-call contract shown above. Free-form forms such as
`send_email(...)` are out of distribution and can produce radically different scores. The CLI
normalizes `args` to the nested JSON string used by the benchmark before rendering the context.

## Results

### Separate held-out set

The model, pooling rule, layers, and head configuration were frozen before a one-shot evaluation on
3,000 separate examples. There was no exact rendered-text overlap with the 1,600 training or 800
validation rows.

| Same 1,600-row training budget | AUPRC | 95% CI | AUROC | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| MiniCPM5 activation probe | **0.8223** | 0.8024–0.8414 | **0.8137** | 0.7986–0.8290 |
| Word/bigram TF-IDF logistic | 0.8079 | 0.7876–0.8281 | 0.7897 | 0.7729–0.8051 |

In a paired 2,000-resample row bootstrap:

- AUROC difference: **+0.0240**, 95% CI **+0.0056 to +0.0429**, two-sided bootstrap
  `p=0.0150`.
- AUPRC difference: **+0.0144**, 95% CI **-0.0053 to +0.0336**, `p=0.1439`.

### Cross-backbone screen

On the frozen 800-row validation screen, MiniCPM5 ranked first by AUPRC among the six completed
frozen activation backbones. Every winner used masked-mean pooling, while the best depth and window
width varied by architecture.

| Frozen backbone | Best hidden output(s) | Normalized depth | Validation AUPRC |
| --- | ---: | ---: | ---: |
| **MiniCPM5-1B** | **13–14** | **0.5625** | **0.9228** |
| Qwen3.5-0.8B | 14–16 | 0.6250 | 0.9205 |
| Qwen2.5-0.5B | 16–18 | 0.7083 | 0.9136 |
| LFM2.5-230M | 10–12 | 0.7857 | 0.9090 |
| Granite 4.0 350M | 19 | 0.6786 | 0.9081 |
| Qwen3-0.6B | 14 | 0.5000 | 0.9045 |
| Word/bigram TF-IDF logistic | n/a | n/a | 0.8866 |

Falcon-H1-0.5B was registered but excluded from the completed comparison because CPU extraction was
not operationally feasible without its fast Mamba kernels. It was not silently counted as a loss.

## How this differs from a fine-tuned encoder

Both approaches begin with text, and both Transformers contain activations. The difference is where
the task-specific learning happens.

| Approach | Task-specific training | Signal used at inference |
| --- | --- | --- |
| TF-IDF text classifier | logistic head over word features | surface words and bigrams |
| Fine-tuned encoder such as `auto-0.4b` / `auto-1b` | backbone plus classification head trained end to end | final task logits |
| This activation probe | frozen MiniCPM5 plus a tiny logistic head | selected intermediate hidden states |

Publisher-trained dedicated encoders use a much larger training corpus and are not matched-budget
comparisons. This release investigates a complementary question: how much authorization signal can
be recovered from a frozen, reusable local backbone with only 1,600 labels and a tiny head?

## Intended use

- Research on where authorization intent appears inside small language models.
- A secondary runtime risk signal beside deterministic permission state.
- Reproduction of the layer/pooling result and comparison with new backbones.
- Local experiments in which prompts and activations remain on the user's machine.

## Scope and responsible use

- Labels are silver, primarily model-generated rather than fully human-gold.
- The held-out set has now been consumed. Future tuning requires a new untouched test source.
- Row-bootstrap intervals do not correct for every latent template or scenario family.
- Subgroup sample sizes are too small for strong language-specific claims.
- Inputs after 192 tokens are invisible to the probe.
- The input contract is narrow: changing the tool-call serialization can materially change a score.
- The probe is tied to the pinned MiniCPM5 architecture and hidden-state indexing.
- `trust_remote_code=True` is required by the supplied example; review and pin model code before use
  in a sensitive environment.

The held-out AUPRC interval for the activation-minus-TF-IDF difference crossed zero. Validation
thresholds targeting 5% and 1% false approves produced 9.49% and 5.07% respectively on the separate
test, so the scores should be used as research ranking signals rather than calibrated permission
probabilities. Consequential actions should still be bound to deterministic facts such as the exact
payload, recipient, scope, preview, and approval state.

## Artifact integrity and provenance

| File | SHA-256 |
| --- | --- |
| `probe.npz` | `b105bfd0026716d25b393b6337e0252f7331afab60ed5b6a72ac76a13a5aa87f` |
| source `metadata.json` | `66b49a4e3441de30c802d9292cbb64420940dbc134be0d0c60368e0a7d425a1d` |
| frozen selection record | `1293df7db9a5d984faffd11d59217bd2651fbf4a3efc66527eae83dbf74c0181` |
| held-out result | `19354ce783ac0cac3da7cab69a9326d5dd8f0d491528058dc3332eca3ccc2b0e` |

Training source: `ProCreations/auto-1b-data` at revision
`d265bbf7bac76cd55daaf81a1d185dfe089a412b` (Apache-2.0). Held-out source:
`ProCreations/approve-or-deny` at revision `a38b625913dd46ca9702063f1597c979e6ace34e`
(Apache-2.0).

The NumPy artifact contains numeric arrays only and is loaded with `allow_pickle=False`.

## License and links

The probe and example code are Apache-2.0. MiniCPM5-1B-Base is also published under Apache-2.0 and
remains governed by its own repository.

- [IntentProbe source](https://github.com/mcpware/IntentProbe)
- [MiniCPM5-1B-Base](https://huggingface.co/openbmb/MiniCPM5-1B-Base)
- [AUTO training data](https://huggingface.co/datasets/ProCreations/auto-1b-data)
- [Approve-or-deny held-out data](https://huggingface.co/datasets/ProCreations/approve-or-deny)

