# Patent and academic prior-art review

**Project:** IntentProbe multi-task activation benchmark and one-prefill/many-head runtime

**Review date:** 2026-10-18

**Companion retrieval record:**
[`PRIOR_ART_SOURCE_LEDGER_2026-10-18.md`](PRIOR_ART_SOURCE_LEDGER_2026-10-18.md)

**Probe-versus-judge evidence map:**
[`LITERATURE_INHERITANCE_MAP_2026-10-18.md`](LITERATURE_INHERITANCE_MAP_2026-10-18.md)

## Source coverage

| Status | Actually checked |
| --- | --- |
| ✅ GOT DATA | Local IntentProbe repo and git history; OpenAlex; arXiv API and 15 full papers; successful Crossref calls; ACL Anthology; public OpenReview records; Google Patents; USPTO Patent Public Search; CIPO; Hugging Face; GitHub; Zenodo; SSRN metadata |
| ⬜ EMPTY | Four successful exact CIPO searches for the two core WO families and two core titles returned no Canadian patent record as of 2026-10-18 |
| ❌ FAILED | Semantic Scholar API: 6 HTTP 429 responses; DBLP API: 5 rate-limited requests; 2 Crossref queries: HTTP 429; OpenReview API: 4 HTTP 403 responses; WIPO PATENTSCOPE: inaccessible/403; EPO Register, Espacenet, and OPS: HTTP 403 |
| ⏭️ SKIPPED | Paid professional patent platforms, unpublished applications, and attorney prosecution-history services |

The exact query families, request counts, canonical documents, access failures, and fallback sources
are in the [source ledger](PRIOR_ART_SOURCE_LEDGER_2026-10-18.md).

## Bottom line

### 1. The broad product idea is not new

The broad architecture — one language model, one inference pass, shared representations, many small
classification heads, and policy or routing decisions — is already disclosed in several independent
forms:

- a 2022 Chinese patent uses one pretrained text feature extractor's hidden states with several
  task-specific classifiers, including emotion, sensitive-word, semantic, and dialogue-operation
  tasks;
- WitnessAI's pending US claim 17 expressly recites a single base model, a static multiheaded behavior
  classifier, binary verdicts for multiple behaviors in one inference pass, and a granular control
  policy;
- the 2022 *All Birds with One Stone* paper performs multiple text-classification tasks in one forward
  pass;
- Patronus Lion Warden publicly ships one encoder with seven security heads in one forward pass;
- a 2026 orchestration paper reuses a serving LLM's hidden states for efficient single-pass
  classification.

We should therefore **not** claim that IntentProbe invented activation probes, multi-head
classification, or one-pass multi-task classification.

### 2. Prompt-injection activation gating has a real US FTO risk

HiddenLayer owns active US grants whose independent claims cover receiving a prompt, capturing an
intermediate-layer result or residual-stream activation from a generative model, applying a
prompt-injection classifier, and giving the determination to another application for prevention or
remediation. The earliest reviewed family claims 2024-04-26 priority and produced grants in October
and November 2024.

That is materially closer to a commercial prompt-injection gate than ordinary academic prior art.
It does not automatically cover every IntentProbe use case, but a US launch that performs automatic
prompt-injection blocking from model activations needs a lawyer-built, element-by-element claim chart.

### 3. There is still a publishable research contribution

The defensible paper is an empirical systems/benchmark paper, not a first-invention paper:

> When one frozen, locally runnable generative SLM is used as a shared sensor, how many heterogeneous
> operational classifiers can independently trained activation heads replace, and what layer,
> pooling, normalization, calibration, latency, memory, and OOD trade-offs determine the answer?

That exact crossed study remains useful if it compares:

- task × backbone × layer/window × token pooling × normalization;
- activation head versus matched surface-text models;
- activation head versus the same backbone answering a direct yes/no or label prompt;
- one prefill plus *N* heads versus *N* judge calls, one multi-label judge call, *N* separate encoders,
  and a jointly trained shared encoder;
- CPU p50/p95 latency, throughput, resident memory, cold start, artifact size, calibration, OOD/LODO,
  and error specificity.

### 4. A narrow patent candidate may exist, but it is not yet a strong filing recommendation

The only plausible candidate is the **specific modular implementation combination**, not the broad
idea:

1. a frozen, decoder-only, locally runnable SLM used only as a reusable semantic sensor;
2. independently trained heads that can be added or removed without retraining the backbone or other
   heads;
3. a per-head manifest that freezes model identity, normalized layer coordinate or adjacent-layer
   window, token pooling, feature normalization, classifier, and calibration;
4. exactly one prefill that exposes a bounded set of hidden tensors and fans them out to all compatible
   heads;
5. a portfolio selector that chooses one to three backbones under joint accuracy, RAM, and latency
   constraints; and
6. a calibrated multi-task receipt for routing, warning, or human review.

Every component has nearby prior art. The combination could still be argued as a concrete systems
method, but obviousness risk is high. The commercially sensible default is: publish the benchmark and
use the implementation/data flywheel as the moat; pay for a patent-agent claim chart only if the final
runtime design contains a concrete mechanism beyond this current combination.

## The invention actually being reviewed

The review used the following proposed system, not the narrower original tool-poisoning probe:

1. A small open-weight generative language model processes each user or agent input once.
2. The backbone remains frozen.
3. Selected intermediate hidden states are read from one layer, an adjacent-layer window, or a
   normalized cross-model depth.
4. Token states are summarized by a readout such as masked mean, masked max, last non-padding, span
   pooling, or a learned lightweight aggregator.
5. Multiple independently trained heads consume the reusable representation. Example heads include
   authorization-to-send, memory-write, toxicity, jailbreak, prompt injection, sentiment, routing,
   threat, tool class, and dialogue state.
6. The system produces several calibrated decisions from one backbone prefill instead of loading many
   task-specific encoders or issuing many generative judge calls.
7. A benchmark searches the model × task × layer × pooling × normalization space and determines
   whether a small backbone portfolio dominates a universal readout.

This definition matters. “One model with many heads” is old. “A frozen decoder SLM with independently
attachable, task-specific, validation-frozen internal readouts and a measured one-prefill runtime” is
narrower, but still assembled from known pieces.

## Verified project chronology

| Event | Verified date | What the date proves |
| --- | --- | --- |
| Original paper says “Date Written” | 2026-03-29 | SSRN author metadata says the paper was written then. It does not by itself establish public availability on that day. |
| LLM Router preprint | 2026-03-21 | Public activation-based routing work predates the paper's stated writing date. |
| RouteGuard preprint | 2026-04-24 | Public hidden-state/attention runtime monitor appeared before the first verified IntentProbe public record. |
| IntentProbe Zenodo record created | 2026-05-02 23:43 UTC | Earliest machine-verifiable repository timestamp found in this review. |
| IntentProbe Zenodo publication date | 2026-05-03 | Earliest verified public publication date found for the original paper. DOI: `10.5281/zenodo.19990741`. |
| WitnessAI application published | 2026-05-07 | Publication came four days after the Zenodo date, but the application claims 2024-11-07 priority. Entitlement of claim 17 to that priority depends on support in the priority document, which this review could not retrieve. |
| `tool-probe` GitHub repository created | 2026-05-08 | The public repository postdates the Zenodo publication by five days. It does not establish the author's private conception date. |
| SSRN paper posted | 2026-05-12 | SSRN's page says “7 Pages Posted: 12 May 2026” and separately “Date Written: March 29, 2026.” |
| Local first git commit | 2026-05-31 | First commit in the local repository history. |
| GitHub repository created | 2026-06-01 04:44 UTC | First verified GitHub repository creation timestamp. |

The accurate public statement is therefore:

> The original IntentProbe paper was written on 2026-03-29 and publicly archived no later than
> 2026-05-03. Its public archive predates the `tool-probe` GitHub repository, but it does not predate
> TaskTracker, PIShield, HiddenLayer's patents, WitnessAI's claimed priority, or activation probing as
> a field.

## Patent claim landscape

### Risk scale

- **Red:** active granted claim with a plausible element-by-element overlap in an intended product path.
- **Amber:** pending claim, jurisdiction-specific right, or close disclosure that needs monitoring or
  claim construction.
- **Yellow:** strong novelty/obviousness prior art but less direct present FTO concern.
- **Green:** adjacent background; useful for completeness, not a present blocking claim on the defined
  product.

### Core patents and applications

| Risk | Document | Priority / status | What is actually claimed or disclosed | Effect on IntentProbe |
| --- | --- | --- | --- | --- |
| **Red** | [HiddenLayer US12107885B1](https://patents.google.com/patent/US12107885B1/en) | 2024-04-26; active US grant, 2024-10-01 | Independent claim 1 receives a prompt for a GenAI model, captures an intermediate-layer result, uses a prompt-injection classifier to decide whether it is malicious or elicits malicious action, and supplies the decision to a consuming process to prevent undesired behavior. Claim 2 specifies transformer residual-stream activations. | Direct risk for US prompt-injection activation gating, including a separate consuming application. |
| **Red** | [HiddenLayer US12137118B1](https://patents.google.com/patent/US12137118B1/en) | 2024-04-26; active US grant, 2024-11-05 | Independent claim 1 expressly captures residual-stream activations from an intermediate transformer layer, classifies prompt injection/undesired behavior, starts remediation, and returns the model output when safe. | Direct risk for allow/block/remediate implementations. |
| **Red/Amber** | [HiddenLayer US12572777B2](https://patents.google.com/patent/US12572777B2/en) | 2024-05-20; active US grant, 2026-03-10 | Independent claims are narrower: multimodal input, compact quantized proxy replicating model topology, forward hooks, residual activations, policy-labelled prohibited actions, and allow/block gating. | High risk only when the implementation uses this complete proxy/multimodal/policy combination; still important for design review. |
| **Amber** | [WitnessAI US20260127380A1](https://patents.google.com/patent/US20260127380A1/en) | Claims 2024-11-07 priority; pending US application, published 2026-05-07; WO2026101639A1 family | Claim 17 recites one base model, a static multiheaded behavior classifier, embeddings from one input, binary verdicts for various behaviors in one inference pass, and a granular control policy. | The closest literal claim to the broad “one SLM, many heads, one pass, policy” architecture. It is pending, but it is a major novelty and monitoring item. |
| **Amber** | [CN114416989B](https://patents.google.com/patent/CN114416989B/en) | 2022-01-17; active Chinese grant, 2024-08-09 | Claims one text feature extractor and multiple task classifiers trained from labelled sentences. The description connects hidden states to emotion, sensitive-word, semantic, and dialogue-operation classifiers and describes one input recognizing multiple tasks. | Strong early novelty/obviousness prior art; possible China-specific FTO issue, but its jointly trained encoder/last-layer structure differs from the proposed modular decoder-probe registry. |
| **Amber** | [HiddenLayer US12549598B2](https://patents.google.com/patent/US12549598B2/en) and [US20260143004A1](https://patents.google.com/patent/US20260143004A1/en) | 2024-04-26 family; B2 granted 2026-02-10; continuation pending | Multimodal activation analysis, per-layer classifiers, dimensionality reduction, layer ensembles, quantized proxies, and gating. | Important if IntentProbe expands from text to multimodal or combines multiple per-layer security verdicts. |
| **Yellow** | [Microsoft US20250200333A1](https://patents.google.com/patent/US20250200333A1/en) | 2023-12-14; pending US application | Hidden-state probes for grounding errors; a linear classifier at a layer; time pooling; per-layer ensembles; combining layers; attention/FFN states. | Strong prior art against broad claims on choosing middle layers, pooling, or combining layer probes. Its target is generated-token grounding error rather than heterogeneous input classification. |
| **Yellow** | [WitnessAI US20260127216A1](https://patents.google.com/patent/US20260127216A1/en) | 2024-11-07 family; pending | Description gives a shared transformer encoder with simultaneous intent, jailbreak, and risk heads in one forward pass. Claims focus more on configurable intention definitions and dynamic guardrails. | Strong disclosure against the efficiency story even where the issued claim scope may differ. |
| **Yellow** | [WitnessAI US20260278411A1](https://patents.justia.com/patent/20260278411) | Provisional priority 2024-12-23; pending continuation-in-part | Description uses one shared encoder and three simultaneous heads; claims focus multi-prompt organizational behavior and temporal control. | Close system disclosure; less direct fixed-head claim evidence than claim 17 above. |
| **Yellow** | [Bitfast EP4621615A1](https://patents.google.com/patent/EP4621615A1/en) | 2024-03-18; pending EP application | A first LLM with a probabilistic classifier scores maliciousness and routes or rejects a prompt before a second LLM. Description uses a final hidden-state classifier head. | Prior art for a small sidecar and routing gate; not multi-head and not intermediate-layer search. |
| **Yellow** | [Articul8 US20260134293A1](https://patents.google.com/patent/US20260134293A1/en) | 2024-11-08; pending US/WO family | Description exposes intermediate hooks, masked mean/CLS/attention pooling, fixed encoders, task-specific heads, calibration, uncertainty, and routing. Claims focus a Gaussian-process probabilistic head. | Strong disclosure against novelty of the readout menu and calibrated routing; current claims are materially narrower. |
| **Yellow** | [Salesforce US11915471B2](https://patents.google.com/patent/US11915471B2/en) | 2021-02-05; active US grant | Shared transformer with multiple task-specific classification heads. Claims are tied to visual-linguistic product attributes and dynamic parameter allocation. | Establishes that shared-backbone/task-head architecture is old; low direct overlap with the defined text runtime. |
| **Green/Yellow** | [Amazon US12437058B1](https://patents.google.com/patent/US12437058B1/en) | 2023-11-15; active US grant | A separate sequence encoder plus binary classifier inspects tool/API results for indirect prompt injection and can terminate an action plan. | Adjacent agent-security gate; not activation probing of the generative model. |
| **Green/Yellow** | [Amazon US12292915B1](https://patents.google.com/patent/US12292915B1/en) | 2023-12-04; active US grant | Trust scoring and attention analysis between input spans and an action plan, followed by termination or confirmation. | Internal attention is used, but it is not the proposed post-hoc activation-head design. |
| **Green** | [US20250124296A1](https://patents.google.com/patent/US20250124296A1/en) | Pending | Transformer serving with task parameters and task-specific heads; claims focus token adaptation, merging, and latency/utility control. | General serving/obviousness background, not a close security claim. |

### Element-by-element novelty chart

This chart compares disclosures; it is not a court claim construction.

| Proposed element | CN114416989B | WitnessAI claim 17 | HiddenLayer grants | Microsoft application | Academic/product record |
| --- | --- | --- | --- | --- | --- |
| One language backbone | Direct | Direct | Direct GenAI model or proxy | Direct | Direct in many papers/products |
| One forward/prefill pass | Description: one input/multiple tasks | **Direct claim language** | A forward pass is inherent; not a many-task claim | Probe during model processing | Direct in *All Birds*, orchestration paper, Lion Warden |
| Multiple task heads | Direct multiple classifiers | **Direct multiheaded classifier** | Multiple models/types appear, but core independent claims can use one classifier | Per-layer probe ensemble | Direct in multi-task NLP and Lion Warden |
| Intermediate hidden activations | Hidden states, usually last layer | “Embeddings”; hidden layer not required by claim text | **Direct intermediate residual activations** | **Direct hidden states** | Direct across activation-probe literature |
| Frozen backbone | Not the central claim | Not required | Proxy/model can be quantized; freezing not the novelty | Probe can be lightweight | Direct in most probing papers |
| Heads independently added later | Not clearly claimed | “Afterwards, N heads are built,” but modular lifecycle is not detailed | Not central | Not central | Common transfer-learning practice; Lion Warden heads are jointly packaged |
| Different layer/window per task | Not central | No | Per-layer models/ensembles appear in continuations | Direct layer probes and combinations | Task-specific layer selection in several papers |
| Different token pooling per task | Some task-specific token positions | No | Dimensional reduction/activation vectors | Time pooling | Direct in high-stakes probes, PoolBench, orchestration paper |
| Normalized cross-model depth | No clear claim found | No | No clear claim found | No clear claim found | Common evaluation normalization; not found as a standalone product claim |
| One to three backbone portfolio optimized jointly | No | No | Quantized proxy selection is adjacent | No | Router/model-selection literature is adjacent |
| Multi-task calibrated receipt | No | Policy output | Determination/remediation | Action generator | Calibration and decision logs exist separately |

No single reviewed reference was found that cleanly discloses **all** of the last five rows in one
system. That is why a narrow combination remains a candidate. However, combining known layer search,
pooling, frozen probes, calibration, and multi-head serving may be considered an obvious engineering
combination unless the implementation produces a demonstrated technical effect not expected from the
pieces.

## Academic prior-art map

### Foundational probing and multi-task architecture

| Work | Date | Relevant contribution | Consequence |
| --- | --- | --- | --- |
| [Alain & Bengio, *Understanding intermediate layers using linear classifier probes*](https://arxiv.org/abs/1610.01644) | 2016 | Independent linear classifiers reveal information across intermediate layers. | Linear probing and layer sweeps are foundational, not IntentProbe inventions. |
| [Tenney et al., BERT layer studies](https://aclanthology.org/P19-1452/) | 2019 | Different linguistic tasks emerge at different depths. | “Different tasks prefer different layers” is expected background; our contribution must be the measured operational matrix, not the general idea. |
| [*All Birds with One Stone*](https://arxiv.org/abs/2205.10744) | 2022-05 | Multiple text-classification tasks with one forward pass using task prompts and conditional poolers/heads. | Direct prior art to the one-pass efficiency concept. |
| [Representation Engineering](https://arxiv.org/abs/2310.01405) | 2023-10 | Population-level representations in LLMs for concepts and safety-related control. | Broad internal-representation framing is established. |

### Security, agent, and runtime activation probes

| Work | Date | Relevant contribution | Difference from the proposed paper |
| --- | --- | --- | --- |
| [TaskTracker](https://arxiv.org/abs/2406.00799) | 2024-06 | Linear probing of activation changes before/after external data for prompt injection, task drift, and malicious instruction; large synthetic/constructed evaluation and OOD tests. | Mostly last-token/delta style and security-focused, not a heterogeneous small-model readout benchmark. |
| [Task-specific prompts and SAEs](https://arxiv.org/abs/2504.20271) | 2025-04 | Compares prompted probes, max pooling, SAE probes, and direct zero/few-shot model-output classification. | Already establishes that direct model answers are a necessary baseline and that readout choice matters. |
| [Trick or Neat](https://aclanthology.org/2025.findings-acl.954/) | 2025-06 | Direct prompting versus layer-wise linear probes for adversarial linguistic ambiguity; probes sometimes exceed 90% accuracy where direct prompting is not robust. | Direct evidence of an extraction gap, but on ambiguity rather than operational agent decisions. |
| [Detecting High-Stakes Interactions](https://arxiv.org/abs/2506.10805) | 2025-06 | Activation probes rival medium LLM monitors; compares mean, max, last, attention, and softmax pooling; proposes cascades. | Strong overlap with monitor and pooling evaluation, but not the same multi-backbone/multi-task CPU replacement question. |
| [PIShield](https://arxiv.org/abs/2510.14005) | 2025-10 | Selects an injection-critical layer, uses final-token residual activations and a linear classifier, evaluates multiple datasets/attacks, and reports large cost reduction. | Direct prior art to prompt-injection layer selection and runtime probing. |
| [Production-Ready Probes for Gemini](https://arxiv.org/abs/2601.11516) | 2026-01 | Production activation probes; long-context shifts; multiple-position readouts; rolling mean/attention/MultiMax; architecture search; cascade with prompted classifier. | Removes any credible “first production activation probe” claim. |
| [A BERTology View of LLM Orchestrations](https://arxiv.org/abs/2601.13288) | 2026-01 | Reuses the serving LLM's hidden states in the same forward pass; token- and layer-selective aggregation; safety and sentiment; compares prompted methods, standalone classifiers, latency, throughput, and memory. | Closest academic system. It does not appear to benchmark a registry of many independently attachable heterogeneous heads executed together on sub-2B CPU backbones. |
| [Representation-as-a-Judge](https://arxiv.org/abs/2601.22588) | 2026-01 | Frozen Qwen3-0.6B/1.7B and Llama hidden-state probes substantially outperform direct small-model prompting on reasoning-quality evaluation and approach larger judges in some cells. | Directly establishes the representation-versus-generated-judgment premise. IntentProbe must contribute a different task and systems scope. |
| [Prompting Underestimates LLM Capability](https://arxiv.org/abs/2601.03464) | 2026-01 | Same-model prompting versus linear probes on time-series classification; average F1 rises from 0.15–0.26 to 0.61–0.67 and prompt variants are unstable. | Strong cross-domain evidence for the extraction gap, but not natural-language agent operations. |
| [Embarrassingly Simple Access Control](https://openreview.net/pdf?id=Re0fXzBeXH) | 2026 | Last-layer, last-token logistic probes versus direct and two-step prompting for RBAC decisions on Spider, BIRD, and OrgAccess. | Very close to authorization classification, but it decides policy access rather than whether a user's language authorizes an agent action. |
| [RouteGuard](https://arxiv.org/abs/2604.22888) | 2026-04 | Hidden-state alignment plus response-conditioned attention for skill-poisoning monitoring. | Different signal construction and threat model; still prior art to runtime internal-state security monitors. |
| [LatentGate](https://aclanthology.org/2026.acl-industry.153/) | ACL 2026 | Frozen SLM, mean-pooled hidden states, PCA whitening, linear probe routing, five backbones, OOD results, and large latency advantage over prompt routers. | Closest routing and normalization prior art. The proposed paper must include routing only as one task in a broader systems benchmark. |
| [`tool-probe`](https://github.com/oleksandr-shyshchuk/tool-probe) | Public repo 2026-05-08 | Single-layer last-token linear probe predicts tool invocation before generation; five model families; layer sweep; latency comparisons. | Independent/parallel work on a different target. IntentProbe's verified Zenodo archive predates the repository by five days. |

### Pooling, layer choice, routing, and multi-head control

| Work | Date | Relevant contribution | Consequence |
| --- | --- | --- | --- |
| [LLM Router: Prefill is All You Need](https://arxiv.org/abs/2603.20895) | 2026-03-21 | Uses open-model prefill activations and one shared trunk to output simultaneous correctness estimates for many target models; includes layer selection, z-score normalization, and PCA. | Predates the original paper's stated writing date and directly covers one-pass multi-output activation routing. |
| [Multi-Head Latent Control](https://arxiv.org/abs/2607.14277) | 2026-07 | Frozen hidden-state trajectories plus separate capability and resolution/control heads. | Postdates the original paper but predates any filing now; close to multi-head activation control. |
| [You Only Judge Once](https://arxiv.org/abs/2604.10966) | 2026-04 | Scores multiple candidate responses with one discriminative forward pass; reports up to 3.9× latency and 4.0× FLOP reduction at four candidates. | Supports shared-pass scaling logic, but it is a multimodal reward model rather than independently attachable heterogeneous task heads. |
| [PoolBench](https://arxiv.org/abs/2608.05162) | 2026-08 | 17 concepts, 19 pooling strategies, three decoder-only models, 37,693 real passages, and statistical comparison of pooling rankings. | A broad “first pooling benchmark” claim is unavailable. Our value is small-model operational classification and head fan-out, not pooling alone. |
| [Prompted to Discriminate](https://arxiv.org/abs/2610.02413) | 2026-10 | Classification suffixes plus activation probes across 13 safety datasets and three model families; LODO evaluation; model/task/readout dependence. | Very close adverse/concurrent evidence; reinforces the need for strict task-generalization controls. |

### Adverse evidence that the paper must answer

| Work | Finding | Required response in our design |
| --- | --- | --- |
| [False Sense of Security](https://aclanthology.org/2026.findings-acl.1300/) | In-domain probe accuracy exceeded 98%, yet OOD accuracy dropped by 15–99 percentage points in tested cells; n-gram baselines were often competitive; controlled tests implicated instruction patterns and trigger words rather than stable harmfulness. | Add source-held-out/LODO evaluation, surface baselines, lexical counterfactuals, semantically matched minimal pairs, and a claim boundary that distinguishes predictive utility from semantic “understanding.” |
| [When Benchmarks Lie](https://arxiv.org/abs/2602.14161) | Activation safety results change under true distribution shift; matched benign-FPR and leave-one-dataset-out evaluation matter. | Pre-register held-out sources and operating points; do not headline random-split AUROC alone. |
| [Before the Last Token](https://arxiv.org/abs/2605.12726) | Last-token probes can miss earlier evidence; max pooling can overfire; trajectory models recover some cases. | Retain multiple pooling/trajectory candidates and report false-positive mechanisms, not only aggregate rank. |
| [Internal-State Probes Read the Situation, Not the Action](https://arxiv.org/abs/2606.30449) | Probes may decode domain or prompt situation rather than the future action of interest. | For authorization and agent actions, test action-generalization, prompt-form swaps, held-out verbs, and context-preserving intent changes. |

This adverse literature is not a reason to stop. It identifies the experiment that could make the
paper stronger than ordinary probe demonstrations.

## Closest released product comparison

### Patronus Lion Warden

[Lion Warden](https://huggingface.co/patronus-studio/lion-warden-ai-security-classifier) is the
closest released product shape found:

- one mmBERT-small encoder;
- seven independent outputs in one forward pass: injection, threat, routing, sensitive document,
  tool class, tool action, and tool tags;
- local CPU and ONNX deployment;
- a reported mean per-head F1 of 0.937 on its model card; and
- a public Hugging Face creation date of 2026-07-15.

The important distinction is architectural, not rhetorical. Lion Warden is a dedicated encoder-only
multi-task classifier whose package contains a fixed jointly trained head set. IntentProbe's proposed
system uses a frozen generative decoder as a reusable sensor, searches different internal readouts per
task, and adds heads post hoc without retraining the backbone. The benchmark should test whether that
extra modularity is worth its cost.

### Commercial/patent actors

- **HiddenLayer:** strongest reviewed activation-security patent position; intermediate residual
  activations, prompt-injection classification, proxy models, and remediation.
- **WitnessAI:** closest pending claim to single-SLM/many-head/policy architecture.
- **Patronus:** closest public multi-head security product.
- **Google/Gemini research:** strongest evidence that probes can be engineered for production, with
  context-shift and cascade work.
- **Microsoft:** patent and research footprint on layer probes, pooling, grounding errors, and
  TaskTracker-style agent monitoring.

## What is academically new enough

### Claims we should not make

- first activation detector;
- first prompt-injection or tool-poisoning activation probe;
- first runtime activation monitor;
- first multi-head language classifier;
- first one-pass multi-task classifier;
- first layer sweep;
- first mean/max/last pooling comparison;
- first activation-based router; or
- first production-ready probe.

### Defensible contribution package

The paper can make the following narrower contributions if the experiments support them:

1. **Crossed small-model design map.** A preregistered matrix across six or more locally runnable
   230M–1B backbones, heterogeneous real-labelled tasks, normalized depth, adjacent-layer windows,
   token pooling, and feature normalization.
2. **One-prefill replacement test.** A direct measurement of when one frozen SLM plus *N* tiny heads
   can replace *N* text classifiers or *N* judge calls at comparable calibrated performance.
3. **Task-specific readout manifests.** Evidence that each task requires a different internal cut even
   when the prompts and backbone are identical, with selection performed only on validation data.
4. **Backbone portfolio result.** A constrained optimization showing whether one universal backbone,
   a two/three-model portfolio, or task-specific backbones provide the best accuracy–RAM–latency
   frontier.
5. **Probe specificity stress test.** Surface baselines, source-held-out transfer, semantic
   counterfactuals, action-generalization, lexical triggers, label-null controls, and calibration at
   fixed error rates.
6. **Reproducible CPU system evidence.** p50/p95 latency, throughput, peak/resident memory, cold start,
   artifact bytes per head, and marginal cost as heads increase from 1 to 2, 4, 8, 16, and 32.

Suggested title:

> **One Prefill, Many Decisions: Benchmarking Frozen Small-LLM Activation Heads Across Tasks,
> Backbones, and Readouts**

The headline question is clearer in ordinary language:

> **How many classifiers can one frozen small language model replace?**

## What our completed evidence already supports

The current repository does not support a universal “activations beat text” claim. It supports a more
interesting interaction claim.

### Six-backbone SST-2 sentiment

All six backbones used the same 1,821 untouched official-test sentences. All six validation-selected
activation readouts beat the matched TF-IDF baseline with 95% paired bootstrap intervals above zero on
AUPRC and AUROC. Five selected masked mean; Qwen3.5 selected last non-padding. The selected normalized
depth was comparatively tight, 0.500–0.607.

Source: [SST-2 model matrix](experiments/SENTIMENT-SST2-MODEL-MATRIX.md).

### Six-backbone SMS spam

Two of six backbones — MiniCPM5-1B and Granite-4.0-350M — had intervals above zero against TF-IDF on
both AUPRC and AUROC. Three models selected masked mean and three selected masked max. Normalized depth
ranged from 0.438 to 0.750.

Source: [SMS model matrix](experiments/SPAM-SMS-UCI-MODEL-MATRIX.md).

### Six-backbone ToxicChat toxicity and jailbreak

The same 5,441 human-annotated prompts and identical splits were used for two labels. Toxicity selected
masked mean in all six backbones. Jailbreak split three masked mean and three masked max. Only two of
twelve activation-versus-TF-IDF cells had paired intervals above zero on both metrics, both for
toxicity. The descriptive normalized-depth decomposition attributed 54.8% of variation to the
task-by-backbone interaction, 43.5% to backbone, and 1.7% to the two-task main effect; with two tasks
and no replicated interaction error term, those shares are descriptive, not population inference.

Source: [ToxicChat crossed matrix](experiments/TOXICCHAT0124-MODEL-TASK-MATRIX.md).

### Current finding

The evidence so far says:

> There is no universal best layer or pooling rule. Readout choice depends on both task and backbone.
> Masked mean is a strong default for several semantic tasks, but max pooling can be selected for
> jailbreak and spam, and highly lexical fine-grained intent can strongly favour TF-IDF.

This finding is useful because it supports the per-head manifest and portfolio design. It is not yet a
claim that the activation head understands intent more deeply than text features.

## Experiments required before submission

### A. Direct-judge comparison

The generic proposition that probes can outperform prompted output is inherited from prior work and
will not be rerun over the full task×backbone matrix. Run a preregistered **sentinel alignment panel**
on authorization-to-execute, one safety task, and one mature positive-control task, using the selected
best-quality and cheapest viable sub-2B backbones. Compare each selected frozen backbone under:

1. activation head;
2. direct zero-shot label prompt with constrained decoding;
3. a strong frozen rubric or chain-of-thought label prompt;
4. one multi-label prompt returning all task labels at once; and
5. *N* separate judge calls.

Measure accuracy and calibration, but also prompt-format failure, invalid outputs, nondeterminism across
seeds/decoding settings, and time to first/final token. The activation system only has a product
advantage if performance is close enough while latency and marginal-head cost are materially lower.
Expand beyond the sentinel panel only if a result contradicts the cited literature or changes the
backbone/product decision.

### B. Shared-encoder comparison

Compare against:

- matched TF-IDF logistic regression;
- one strong fine-tuned encoder per task;
- one jointly trained shared encoder with several heads;
- a Lion-Warden-style fixed multi-head encoder; and
- an embedding-only linear head using the same backbone's final representation.

This separates the value of **internal readout selection** from the ordinary benefit of shared
representations and many heads.

### C. OOD and semantic-specificity controls

At minimum:

- leave-one-source/dataset-out;
- held-out authors/users/templates where available;
- lexical trigger insertion into benign examples;
- harmful-to-benign semantic replacement while preserving instruction form;
- paraphrases that preserve meaning but change surface form;
- context changes that flip authorization while keeping the requested action words fixed;
- label-null and random-direction probes;
- length-only, bag-of-words, character n-gram, and embedding baselines; and
- post-hoc failure clusters with confidence intervals.

### D. Runtime scaling

Run 1, 2, 4, 8, 16, and 32 heads on each candidate backbone. Record:

- backbone prefill time;
- hidden-state capture/copy overhead;
- total head time and marginal time per head;
- p50, p95, and p99 latency;
- examples/second at batch sizes 1 and a practical batch;
- resident and peak RAM;
- model and head artifact bytes;
- cold start and warm start; and
- energy or CPU time where measurable.

The system claim should be based on the **measured slope as heads are added**, not the assumption that
logistic heads are cheap.

### E. Selection and statistical lock

- Freeze task admission, datasets, group units, model panel, candidate readouts, primary metrics, and
  missing-cell rules before looking at the final cross-task matrix.
- Choose readouts on validation only.
- Report paired uncertainty against every matched baseline.
- Treat cross-backbone ranking observed after test access as exploratory.
- Use hierarchical or grouped resampling when the dataset has user/source clusters.
- Correct or clearly partition confirmatory versus exploratory multiple comparisons.
- Publish failed cells and convergence failures.

## FTO-oriented design options

These are design distinctions to investigate, not guaranteed workarounds.

1. **Lead with authorization, memory, routing, sentiment, and state tracking.** HiddenLayer's closest
   independent claims are specifically directed to prompt injection or undesired behavior and
   remediation. Non-security semantic classifiers are farther from those claims.
2. **Separate scoring from automatic remediation.** A research/diagnostic receipt is factually
   different from an automatic allow/block gate, although some claims cover providing the decision to
   another process and may still reach the architecture.
3. **Use a separate generic sensor rather than a topology-matched quantized proxy.** This may avoid
   elements of the narrower multimodal proxy claims, but not necessarily the broader prompt-injection
   claims.
4. **Avoid multimodal topology replication unless needed.** US12572777B2 recites a multimodal prompt,
   quantized proxy, replicated layer topology, forward hooks, policy mapping, and gating.
5. **Keep a dated feature-to-claim matrix for every release.** Record whether the release performs
   prompt-injection classification, automatic remediation, same-model versus separate-sensor
   activation capture, multimodal processing, quantization, and policy mapping.
6. **Monitor WitnessAI prosecution.** Claim 17 is pending and may be amended, rejected, divided, or
   granted. Recheck its file wrapper and continuations before a commercial US launch.
7. **Recheck Canadian national-phase entries.** CIPO returned no current matches, but the WO families
   are recent enough that later Canadian entries remain possible.

## Filing timing and public disclosure

The verified original publication chronology matters:

- Canada officially warns that an inventor who has disclosed an invention should move quickly and
  identifies a 12-month filing window.
- The USPTO describes a one-year grace period for inventor-originated public disclosures, subject to
  the statutory conditions and proof.
- EPO Article 55 has only narrow six-month exceptions for evident abuse or qualifying international
  exhibitions; an ordinary voluntary paper or repository publication is not a general grace-period
  route.

Sources: [CIPO patent guidance](https://ised-isde.canada.ca/site/canadian-intellectual-property-office/en/learn-patents-canadian-intellectual-property-office),
[USPTO grace-period guidance](https://www.uspto.gov/patents/basics/international-protection/filing-patents-abroad),
and [EPO Article 55 / 2026 Guidelines](https://www.epo.org/en/legal/guidelines-epc/2026/g_v_1.html).

For subject matter actually disclosed in the Zenodo paper on 2026-05-03, a US or Canadian grace-period
analysis should be completed well before 2027-05-03. That date does **not** automatically apply to a
later, genuinely new multi-head runtime mechanism that the paper did not disclose; the first public
GitHub, Hugging Face, Reddit, demo, or paper disclosure of that specific mechanism must be identified
separately. European rights may already be impaired for any claim fully disclosed before filing.

Do not publish a detailed new runtime mechanism while deciding whether to file. A paper abstract can
state the research question without disclosing implementation details that are not yet in a filing.

## Recommended decision

### Paper

**Proceed.** Reframe it as a benchmark and systems paper. The strongest question is whether a single
frozen local SLM plus many independent activation heads can replace a collection of judge calls or
task-specific classifiers under a real CPU/RAM/latency budget. The literature makes the baseline suite
and OOD controls stricter, but it does not make that experiment pointless.

### Product

**Proceed with the modular head registry and one-prefill runtime prototype.** Treat authorization,
memory-write, routing, sentiment, and dialogue-state heads as the first product surface. Keep
prompt-injection automatic gating behind an FTO review for the US market.

### Patent

**Do not file a broad “activation probe” or “one model, many heads” application.** If patent protection
is still desired, first write a short invention disclosure around the concrete head manifest, one-pass
fan-out ABI, compatibility validation, portfolio selection, and calibrated receipt mechanism. Then pay
a registered patent agent to:

1. chart the proposed independent claim against WitnessAI claim 17, HiddenLayer's active claims,
   CN114416989B, Microsoft, and the academic references;
2. inspect every priority document and US prosecution file wrapper;
3. search continuations and Canadian/PCT national-phase entries;
4. determine inventorship and the first public disclosure of each claimed feature; and
5. decide whether the demonstrated runtime effect is enough to support non-obviousness.

If that chart cannot identify at least one concrete mechanism absent from the cited combination, use a
defensive publication and keep the evolving calibration corpus, task-head data, runtime receipts, and
integration layer as the commercial moat.

## One-sentence position

IntentProbe is not the first activation detector, but it can still produce a valuable paper and
product by being the most rigorous small-model benchmark and modular one-prefill/many-decision runtime —
provided it proves when the heads generalize, when they merely reproduce surface cues, and when their
speed advantage survives a fair comparison with direct judges and shared encoders.
