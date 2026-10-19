# IntentProbe: One Forward, Many Decisions — architecture disclosure v0.1.1

Document date: **2026-10-18 (America/Vancouver)**<br>
Status: **public technical disclosure v0.1.1**<br>
Author: **Nicole BeWater**<br>
Publisher: **MCPware, a product and research brand of GEASS LABS INC.**

This document specifies the IntentProbe research and product architecture in implementable detail.
It separates components that exist today from proposed components that still require implementation
and measurement. This version is the citable public technical bridge between the IntentProbe papers,
runtime, and probe catalog.

The architecture starts from one economic fact: the local-model forward pass dominates the cost.
IntentProbe performs that pass once, exposes a versioned hidden-state bus, and lets independently
trained task readouts reuse it. Calibration, host policy and irreversible action remain outside the
sensor. The measured research runtime serves 16 heads with 6.9% more mean latency than one head in
the recorded CPU setting.

| Release boundary | v0.1.1 state |
|---|---|
| Public product | Single-head PyPI tool-poisoning path; standalone Hugging Face authorization and memory-write scorers |
| Measured research implementation | One-forward fan-out across 16 frozen heads with timing artifacts and tests |
| Specified next implementation | General registry, hot-loading and multi-head receipt emission |

For a shorter explanation, start with [the public architecture
note](../../docs/ONE_FORWARD_MANY_DECISIONS.md).

![One input goes through one frozen local model. The same hidden states feed many small task heads,
which produce calibrated scores and a runtime receipt.](../../docs/one-forward-many-decisions.svg)

## 1. Publisher and project hierarchy

```text
GEASS LABS INC.                         legal company
  └── MCPware                           public publisher; GitHub/Hugging Face organization
      └── IntentProbe                   research and product umbrella
          ├── Papers                    readout benchmark and shared-runtime architecture
          ├── Runtime                   local activation extraction, head registry, policy, receipts
          └── Probe products / heads
              ├── Tool poisoning
              ├── Action authorization
              ├── Memory write
              └── Future admitted tasks
```

MCPware is the publisher. IntentProbe is the project. A tool-poisoning model, authorization model,
or memory-write model is one IntentProbe probe product, not a separate top-level brand or research
programme. The papers explain and test the common method behind the products.

The scholarly author is Nicole BeWater. MCPware publishes the work as a product and research brand
of GEASS LABS INC. This identity should remain consistent in the paper PDF, DOI metadata, GitHub
release, Hugging Face Collection, model cards, citation file, and project site.

## 2. Problem

Agent workflows repeatedly classify the same request and context for different purposes. Examples
include whether the request authorizes an irreversible action, whether content is toxic or contains
a prompt injection, whether a turn should enter long-term memory, and which workflow or model should
handle the request.

Three common approaches have different costs:

1. a rule or text classifier is fast, but only uses the observable input representation chosen for
   that classifier;
2. an LLM judge can reason in generated text, but each separate question can require another prompt
   and generation pass;
3. separate task encoders can work well, but a workflow checking many properties may load or invoke
   many models.

IntentProbe studies a fourth design: run one frozen, locally available language model forward pass,
read selected hidden activations, and reuse those activations through small independently versioned
task heads. Each head is trained for a defined label contract and may read different tokens, layers,
layer combinations, normalizations, and feature transformations from the shared pass.

### 2.1 Stable architecture, replaceable components

IntentProbe is the complete serving architecture, not one backbone or one set of probe weights. Its
stable structure is the versioned input renderer, shared local-model pass, hidden-state activation
bus, readout-manifest contract, calibration, policy boundary, and receipt. The model, labelled
dataset, layer selection, pooling method, normalization, classifier, and threshold are replaceable
implementations inside that structure. The current research runtime demonstrates fan-out; the
general independently loadable registry remains a specified implementation step.

This distinction lets the system improve without being rebuilt. New data can update one head. A new
readout search can move that task to different layers. A stronger local model can replace the
backbone after compatible heads are trained. None of those changes requires unrelated detectors to
be retrained or the host workflow to adopt a different control interface.

## 3. System boundary

The specified full system is:

```text
request + conversation + proposed action + runtime event
                           |
                    versioned renderer
                           |
              frozen local backbone forward pass
                           |
              hidden-state activation envelope
                           |
        +------------------+------------------+
        |                  |                  |
 authorization readout  memory readout   poisoning readout   ...
        |                  |                  |
 calibrated head       calibrated head    calibrated head
        +------------------+------------------+
                           |
              decisions + abstentions + receipt
                           |
              host workflow / policy enforcement
```

The activation layer says what the model representation indicates under a task's learned label
contract. The host application's hard rules still decide what the software may do. For example, a
send control may require a final user confirmation even when the authorization head scores the
action highly. These layers can be combined without treating a hard gate as proof that the model
understood the user's intention.

## 4. Component contracts

### 4.1 Input renderer

The renderer converts task inputs into one deterministic text or token sequence. Depending on the
task, its fields can include:

- the current user request;
- selected conversation history;
- the assistant's proposed action;
- tool name and canonicalized arguments;
- the candidate memory;
- a tool, MCP server, or agent-skill description;
- source and field boundary markers.

The renderer has its own identifier and version. Its manifest records field order, separators,
escaping, maximum context length, truncation side, special-token policy, and any fields excluded for
privacy. Training and runtime must use the same renderer contract.

### 4.2 Frozen backbone adapter

The backbone adapter loads a named open-weight model and tokenizer revision, performs a forward pass
with hidden-state output enabled, and returns the hidden states needed by all compatible installed
heads. The adapter records:

- model repository and immutable revision;
- tokenizer repository and immutable revision;
- hidden-state indexing convention, including whether output `0` is the embedding output;
- precision, device, attention implementation, and maximum length;
- padding, truncation, and chat-template behaviour;
- software versions that can affect extraction.

The backbone is frozen during probe training. Adding or updating a probe does not update the
backbone weights.

### 4.3 Activation envelope and bus

For one rendered input, the runtime creates an activation envelope containing the valid-token mask,
requested hidden-state outputs, optional token spans, extraction metadata, and timing. The activation
bus exposes this envelope only to compatible readout manifests.

A compatible multi-head execution groups all admitted heads by backbone and renderer contract,
computes the union of their requested hidden-state outputs, and performs one backbone forward pass
per compatible group. A one-backbone installation therefore needs one pass for all compatible heads;
a two-backbone portfolio needs at most one pass for each participating backbone.

### 4.4 Readout manifest

Every probe product is defined by a versioned readout manifest. At minimum, it binds:

| Field | Meaning |
| --- | --- |
| `probe_id` and `probe_version` | Stable task-head identity |
| `task_contract` | Labels, positive class, intended decision, and excluded interpretations |
| `backbone` and `revision` | Exact frozen representation source |
| `renderer_id` and `renderer_version` | Exact input construction |
| `max_length` and truncation policy | Runtime token boundary |
| `token_region` | Valid tokens, last non-padding token, named field/span, or another declared region |
| `pooling` | Mean, max, top-k, last-token, attention-weighted, or another declared operation |
| `hidden_outputs` | Absolute indices and normalized-depth equivalents |
| `layer_combination` | Single output, mean, concatenation, learned mixture, or trajectory operation |
| `normalization` | None, training-fold standardization, PCA/whitening, or another fitted transform |
| `head_family` | Logistic, calibrated linear/SVM, shallow MLP, boosting, or another bounded head |
| `artifact_hashes` | Integrity hashes for every fitted parameter file |
| `calibration` | Method, fitting split, and calibration parameters |
| `operating_points` | Thresholds, abstention region, and their validation provenance |
| `compatibility` | Runtime schema and backbone-adapter versions |

Normalization, dimensionality reduction, head fitting, calibration, and threshold selection must be
fitted using training/development data only. Test examples cannot affect the manifest.

### 4.5 Task head

A head accepts only the feature vector declared by its manifest and returns raw score, calibrated
probability or class scores, predicted label, confidence/abstention state, and head timing. Binary,
multiclass, ordinal, and multilabel heads are permitted when their task contracts and metrics are
declared.

Logistic regression is the current principal baseline because it directly measures linear
accessibility and produces very small artifacts. It is not an architectural requirement. A more
complex head is admitted only through a frozen head-family protocol, sample-size safeguards, a
matched held-out comparison, and a measured deployment cost.

### 4.6 Calibration and decision policy

The calibrated head result and the host policy are separate objects. A head may expose probability,
uncertainty, and abstention. A policy maps those values, corroborating evidence, and the action's
consequence into `allow`, `warn`, `review`, or `block` without changing the head artifact.

This separation allows the same authorization score to support different policies for drafting a
document and sending it to another person. It also permits conservative hard gates to remain a final
boundary while the activation head improves semantic intent understanding.

### 4.7 Receipt

Each runtime decision emits a machine-readable receipt sufficient to reproduce the route from input
class to decision. It should include:

- timestamp and event identifier;
- renderer and runtime versions;
- backbone, tokenizer, and revision;
- probe ID/version and artifact hashes;
- requested hidden outputs, token region, pooling, layer combination, and normalization;
- raw and calibrated score, threshold, predicted decision, and abstention state;
- policy version and any non-activation corroboration;
- extraction, head, and total elapsed time;
- redaction-safe input fingerprint rather than secret payloads.

The current scanner already records artifact identity, activation details, layer policy, score,
decision policy, and elapsed time in its risk record schema. The shared runtime extends that contract
to multiple heads and backbone groups.

## 5. One-forward, many-decision execution

Reference algorithm:

```text
INPUT: runtime event E, installed manifests M, policy set P

1. Select manifests whose task contracts apply to E.
2. Group selected manifests by compatible backbone + renderer execution contract.
3. For each group G:
     a. Render E once according to G's shared renderer contract.
     b. Compute the union U of hidden-state outputs required by manifests in G.
     c. Run the frozen backbone once and retain U plus the valid-token mask.
     d. For each manifest m in G:
          i.   select m.token_region;
          ii.  apply m.pooling and m.layer_combination;
          iii. apply m.normalization;
          iv.  execute m.head_family and m.calibration;
          v.   apply the relevant policy and abstention rule;
          vi.  append the result and provenance to the receipt.
4. Return the decision vector and the complete receipt.
```

The activation envelope is request-scoped and need not be stored after the receipt is produced.
Deployments may retain only hashes and redacted metadata.

## 6. Head lifecycle

1. Define the task and label contract.
2. Admit datasets through provenance, licence, label-quality, leakage, and split checks.
3. Pin the renderer, backbone/tokenizer revision, and extraction environment.
4. Extract or cache activation tensors with source IDs and integrity hashes.
5. Search the permitted readout space on training/development data.
6. Fit normalization and the candidate head inside each training fold.
7. Select the readout and head under accuracy, calibration, latency, memory, and stability
   constraints.
8. Freeze the manifest before opening the final test.
9. Evaluate once on the untouched or source-held-out test and report uncertainty and failure slices.
10. Package the artifact, manifest, benchmark, licence/provenance record, hashes, scorer, and runtime
    compatibility declaration.
11. Install, update, disable, or roll back the head independently of the frozen backbone and other
    heads.

## 7. Deployment-constrained readout compiler

IntentProbe treats layer choice as an engineering search problem rather than assuming the last layer
is best. The compiler's candidate space can include:

- valid-token masked mean, masked max, last non-padding token, named spans, top-k pooling, and learned
  attention pooling;
- one hidden-state output, adjacent windows, concatenation, averaging, learned mixtures, and
  layer-to-layer trajectory features;
- absolute output indices and normalized depth so architectures with different layer counts can be
  compared;
- no scaling, fold-local standardization, PCA, whitening, and covariance-aware transforms;
- linear and bounded nonlinear heads;
- single-backbone and resource-constrained backbone-portfolio selection.

For a task `t`, backbone `b`, and candidate manifest `m`, selection optimizes a declared objective
over task quality, calibration, latency, peak memory, artifact size, and out-of-source stability.
Constraints are frozen before the final test. The output is not merely a score; it is an executable
manifest plus the evidence ledger that produced it.

The compiler may select different readouts for two tasks on the same prompts. Existing ToxicChat
experiments already provide such a case: toxicity selected masked-mean hidden outputs 18–20, while
jailbreak detection selected masked-max outputs 5–7 on the same Qwen2.5-0.5B prompt corpus. This is
evidence against treating one universal readout as an assumption; broader matrices are required to
determine whether stable conditional rules exist.

## 8. Product embodiments

### 8.1 Tool-poisoning probe

Input: MCP tool, server, package, or agent-skill descriptions and selected runtime fields.<br>
Decision: whether the description contains instructions or intent inconsistent with the tool's
stated function, including data theft, control takeover, or action outside the declared job.<br>
Current readout: Qwen2.5-0.5B, mean-pooled hidden states 13–15, logistic head.<br>
Current state: integrated into the Python package, CLI, hook wrapper, warm JSONL service, and policy
receipt path.

### 8.2 Action-authorization probe

Input: user request, relevant history, and exact proposed action/tool call.<br>
Decision: whether the available context supports carrying out that action now, rather than only
preparing, explaining, or drafting it.<br>
Current readout: MiniCPM5-1B-Base, standardized masked-mean hidden outputs 13–14, logistic head.<br>
Current state: standalone research release and scorer; shared-runtime integration remains pending.

### 8.3 Memory-write probe

Input: an utterance or candidate memory plus the context declared by the renderer.<br>
Decision: task-specific value or eligibility for long-term memory write.<br>
Current public readout: Qwen2.5-0.5B, masked-mean hidden output 11, logistic head.<br>
Current state: standalone public research release. Human-labelled experiments have subsequently
tested alternative readouts; the public product artifact has not silently been replaced by those
post-release experiments.

### 8.4 Prompt-injection and jailbreak head

Potential inputs include untrusted documents, retrieved content, tool results, or direct prompts.
The task contract must distinguish prompt injection, jailbreak attempt, harmful content, and mere
discussion of those topics. Existing historical experiments are not yet a shared-protocol release.

### 8.5 Toxicity and harmful-intent heads

These are separate possible label contracts. A toxicity head can classify language properties; a
harmful-intent head can classify the intended action. They should not be merged merely because both
are safety related. ToxicChat experiments currently supply a human-data benchmark lane.

### 8.6 Sentiment and conversation-state heads

These heads can expose sentiment, emotion, escalation, uncertainty, or semantic trajectory to a
workflow without generating a verbal judgment for each attribute. Current SST-2 and GoEmotions
experiments test whether those states are accessible under different readout choices.

### 8.7 Routing and task-complexity heads

A routing head can choose a workflow, specialist agent, tool class, or compute tier. A task-complexity
head can route a simple request to a cheaper path and an uncertain request to a stronger model or
human review. BANKING77 is a current negative result for one highly lexical 77-way task and therefore
helps define where activation readouts may not be the best component.

## 9. Multi-backbone portfolio

One backbone need not be best for every task. The runtime may support one to three resident small
backbones and assign each head to the smallest suitable backbone. Portfolio selection minimizes the
number and total cost of backbone passes subject to per-task quality and calibration constraints.

This produces three deployment modes:

- **single-backbone:** maximum activation reuse and simplest local deployment;
- **small portfolio:** two or three backbones covering tasks with materially different representation
  needs;
- **task-specialized:** one backbone per head when quality outweighs sharing, equivalent to giving up
  most of the shared-runtime advantage.

The benchmark must measure the frontier rather than assume single-backbone sharing always wins.

## 10. Evaluation programme

### 10.1 Paper A: readout benchmark

Paper A crosses task, backbone, token region, pooling, hidden-state depth/window, layer combination,
normalization, and head family. Its confirmatory evidence uses real or naturally occurring,
human-labelled held-out data wherever available. Required controls include:

- source-grouped or leave-one-dataset-out evaluation;
- exact and near-duplicate leakage checks;
- matched TF-IDF and compact text-encoder baselines;
- direct yes/no prompting of the same local backbone;
- repeated seeds or grouped resampling with uncertainty;
- calibration and low-error operating points;
- CPU latency, memory, and artifact size;
- negative tasks and failure slices, not only successful heads.

The supported current finding is an interaction among task, backbone, and readout. A universal best
layer, pooling method, or model is not yet established.

### 10.2 Paper B: one-prefill, many-decision architecture

Paper B compares three matched arms on the same hardware, input set, precision, context budget, and
task definitions:

| Arm | Execution for `K` decisions |
| --- | --- |
| Same-backbone LLM judge | Prompt and generate a decision separately `K` times |
| Standalone classifier fleet | Invoke or keep resident `K` task-specific classifiers |
| IntentProbe shared runtime | One compatible backbone pass, then `K` manifests and heads |

Measure `K = 1, 2, 4, 8, 16`; cold and warm startup; p50/p95 latency; throughput; peak RSS; load
cost; energy where measurement is reliable; run-to-run stability; AUROC/F1 or task-appropriate
quality; ECE/Brier calibration; and abstention coverage. Quality comparisons must use a defined
decision parser for generated judges and disclose failed/invalid judge outputs.

The shared-path `K=1,2,4,8,16` curve and repeated-base-forward lower bound are complete. Across the
80-row shared-path panel, one Qwen2.5-0.5B forward plus one head takes 68.63 ms per input on average;
one forward plus 16 heads takes 73.34 ms. In the separate matched ten-prompt comparator, the
one-forward/16-head mean is 62.70 ms and the 16-repeated-forward mean is 956.71 ms. The matched
repeated/shared ratio is 15.26, paired-bootstrap 95% CI `[15.04, 15.49]`. The one-call multi-label
judge and measured standalone-classifier fleet remain open, so this release does not claim the
complete systems frontier. It also does not claim that every activation head will match a generated
judge or specialized classifier on every task.

## 11. Current implementation map

| Architecture element | Evidence in this repository | Status |
| --- | --- | --- |
| Frozen-model registry and layer selection | [`intentprobe/scanner/model_registry.py`](../../intentprobe/scanner/model_registry.py) | Implemented for the current scanner path |
| Artifact validation and activation scoring | [`intentprobe/scanner/core.py`](../../intentprobe/scanner/core.py) | Implemented for one loaded probe |
| Runtime-event normalization and warm JSONL process | [`intentprobe/scanner/hook.py`](../../intentprobe/scanner/hook.py) | Implemented for the current scanner |
| Risk/receipt fields | [`research/schemas/activation_scanner_risk.schema.json`](../schemas/activation_scanner_risk.schema.json) | Implemented current schema; multi-head extension pending |
| Dataset provenance record | [`research/schemas/activation_curated_dataset.schema.json`](../schemas/activation_curated_dataset.schema.json) | Implemented current schema |
| Versioned readout-manifest contract | [`research/schemas/intentprobe_readout_manifest.schema.json`](../schemas/intentprobe_readout_manifest.schema.json) | Draft v1 schema and validation examples implemented |
| Multi-head decision receipt | [`research/schemas/intentprobe_multihead_receipt.schema.json`](../schemas/intentprobe_multihead_receipt.schema.json) | Draft v1 schema and validation examples implemented; runtime emission pending |
| Product catalog | [`probes/`](../../probes/) | Three documented probe products |
| Multi-task readout evidence | [`research/multitask-activation-benchmark/`](./) | Multiple completed and in-progress task matrices |
| Shared activation bus and head fan-out | [`experiments/ONE-FORWARD-MULTIHEAD-EXP001/`](experiments/ONE-FORWARD-MULTIHEAD-EXP001/) | Implemented and measured as a research runtime; production registry and hot-loading remain pending |
| `K`-head systems comparison | [`experiments/ONE-FORWARD-MULTIHEAD-EXP001/RESULT_SUMMARY.md`](experiments/ONE-FORWARD-MULTIHEAD-EXP001/RESULT_SUMMARY.md) | Shared `K` curve and repeated-forward lower bound complete; multi-label judge and standalone fleet pending |

The shipped scanner still concatenates selected layer features for one artifact. The research
runtime now demonstrates one-forward fan-out across 16 frozen heads, but the production package does
not yet provide the general manifest registry, hot-loading, or multi-head receipt emission described
above. This distinction remains part of the release boundary.

## 12. Replaceable implementations

The architecture is not tied to one vendor, model family, classifier, or task. Implementations may
replace:

- Qwen or MiniCPM with another open-weight local backbone;
- a decoder-only backbone with another architecture exposing deterministic internal states;
- logistic regression with another fitted head;
- mean pooling with span, max, learned, or trajectory pooling;
- one-backbone serving with a resource-aware portfolio;
- a fixed threshold with calibrated abstention and consequence-aware policy;
- Python transport with another local process or library interface.

Compatibility depends on the manifest and evidence contract, not on reproducing one current artifact.

## 13. Prior-art and claim boundary

This document does not claim to invent transformers, hidden-state access, activation probing, linear
probing, layer sweeps, shared trunks, multi-head classifiers, early exit, or model routing. Earlier
work exists for each broad component.

The work to be tested and associated with IntentProbe is the complete system and engineering method:

- task-specific activation readout manifests over replaceable small local backbones;
- one compatible forward pass reused by independently installable, calibrated heads;
- a deployment-constrained compiler for choosing token region, pooling, layer depth/window,
  combination, normalization, and head;
- head lifecycle, compatibility, calibration, abstention, policy separation, and versioned receipts;
- measured task-quality and systems trade-offs against same-backbone judges and separate classifier
  fleets.

The detailed academic and patent landscape is maintained in
[`PRIOR_ART_REVIEW_2026-10-18.md`](PRIOR_ART_REVIEW_2026-10-18.md). Public wording should identify the
specific disclosed architecture and date. It should not say “first activation detector” or claim to
prove what no one else privately thought.

## 14. Public priority package

This public version freezes a canonical release bundle containing:

```text
paper and architecture disclosure
source and code snapshot
readout and head manifests
dataset provenance and licence records
machine-readable experiment results
figures and diagrams
CITATION.cff and release metadata
SHA256SUMS
```

The first distribution point is the versioned GitHub source and release. Hash-linked records can
also be deposited through Zenodo/DataCite, an OSF Registration, OpenTimestamps, and Software
Heritage. TechRxiv can provide an additional moderated engineering-preprint route. Every identifier,
server timestamp, commit, archive hash, DOI, and SWHID belongs in one public priority ledger.

Those records establish what was publicly disclosed by a specific date and make it discoverable and
citable. They do not create an exclusive patent right.

## 15. What this version adds to the existing public record

Existing public IntentProbe material already describes individual activation probes, selected
layers, artifacts, benchmarks, and the local tool-poisoning runtime. The v0.1.0 release consolidated
and specifies the following architecture-level material in one coherent disclosure:

- the GEASS LABS INC. → MCPware → IntentProbe → papers/runtime/products hierarchy;
- the activation-envelope and manifest contracts;
- compatibility grouping and one-forward fan-out across independently installable heads;
- the head lifecycle and deployment-constrained readout compiler;
- resource-aware multi-backbone portfolio variants;
- explicit separation of activation inference, calibration, and host policy;
- the matched `K`-decision evaluation against generated judges and classifier fleets.

The prior-art review and source ledger separate earlier public material from details first
consolidated in this release.

## 16. Open release work

The public author and publisher identity, schema names, first paper title, shared-path benchmark, and
repeated-forward lower bound were frozen in v0.1.0. Version 0.1.1 improves the public explanation,
figures, and panel labelling without changing those measurements. Later versions may add an ORCID,
DOI, SWHID,
one-call multi-label judge, standalone-classifier fleet, cold-start and throughput results, and a
production head registry. Those additions must receive new versioned evidence rather than silently
changing this record.

## 17. Version history

| Version | Date | State | Change |
| --- | --- | --- | --- |
| 0.1-draft | 2026-10-18 | Local only | First integrated architecture, hierarchy, component contracts, embodiments, evaluation protocol, and public-priority plan |
| 0.1.0 | 2026-10-18 | Public release | Names Nicole BeWater as author; freezes the IntentProbe one-forward, many-decisions architecture; adds the completed 16-head shared-path result, direct-judge sentinel, schemas, figures, paper draft, and machine-readable evidence |
| 0.1.1 | 2026-10-18 | Public presentation patch | Leads with the measured architecture, draws the actual multi-head fan-out and policy boundary, separates the 80-row and matched ten-prompt timing panels, and states the shipped/research/specified boundary explicitly |
