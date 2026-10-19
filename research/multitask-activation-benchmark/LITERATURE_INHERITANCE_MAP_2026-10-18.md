# Literature inheritance map: activation probes versus generated judgments

**Review date:** 2026-10-18<br>
**Decision:** cite established results; run only the comparisons needed to validate IntentProbe's
specific small-model, local-runtime, and many-head claims.

**Execution update:** the frozen six-cell sentinel and the first K=`1,2,4,8,16` shared-head CPU
experiment are complete. The outcome preserves the literature boundary: activation heads dominate
the studied toxicity judge cells, tie the strongest Qwen3.5 sentiment judge, and lose one corrected
human-authorization comparison. One Qwen2.5 forward plus 16 heads is 1.069× the K=1 latency, while
16 repeated base forwards are 15.26× slower than the shared path on matched prompts. The
repeated-forward arm is a lower bound, so the full multi-label-judge and compact-classifier arms
remain open.

## Source coverage

| Status | Actually checked |
| --- | --- |
| ✅ GOT DATA | Existing IntentProbe prior-art review and retrieval ledger, including 15 previously inspected academic papers |
| ✅ GOT DATA | 36 additional scholarly web queries across arXiv, ACL Anthology, OpenReview-indexed papers, direct prompting, probing, representation-as-judge, access control, ambiguity, routing, single-pass judging, and multi-head synonyms |
| ✅ GOT DATA | Primary paper pages or full papers for *Task-specific prompts and SAEs*, *A BERTology View of LLM Orchestrations*, *Representation-as-a-Judge*, *Prompting Underestimates LLM Capability*, *Trick or Neat*, *LatentGate*, *Calibration, Not Answer Selection*, and *You Only Judge Once* |
| ✅ GOT DATA | OpenReview-indexed full-paper text for *Embarrassingly Simple Access Control for Large Language Models*, including method and result tables; author and forthcoming-venue metadata cross-checked on Anthony Rios's publication page |
| ⬜ EMPTY | No paper was found that combines the complete IntentProbe target: one sub-2B frozen local decoder, independently attachable task heads with different readout manifests, one shared prefill, heterogeneous operational tasks, and measured `K = 1, 2, 4, 8, 16` CPU scaling against both repeated and multi-label judges |
| ❌ FAILED | Direct HTTP retrieval of the OpenReview access-control PDF returned HTTP 403; the indexed full-paper text was usable, but the paper should be reopened from the final proceedings record before submission |
| ⏭️ SKIPPED | Patent databases were not rerun for this narrow evidence question; they were already covered in the companion prior-art review |

## Answer

Yes. The broad experiment has already been done several times:

> A classifier trained on a frozen language model's hidden states can match or outperform the same
> model's generated or prompted judgment on some classification tasks, while avoiding autoregressive
> decoding or a separate model call.

This is established prior evidence, not a new IntentProbe claim. IntentProbe should cite it and test
only whether the result survives in our exact deployment setting.

The papers do **not** establish that activation probes always beat prompting. Their results depend on
the task, backbone, training-data volume, prompt strength, distribution shift, readout, and metric.
The defensible inherited claim is therefore conditional: internal representations can expose useful
classification signals more accurately or cheaply than generated answers in multiple studied
settings.

## Direct evidence we can inherit

| Evidence tier | Work | What it already tested | What we can cite instead of rediscovering |
| --- | --- | --- | --- |
| **Direct** | [Tillman & Mossing, *Investigating task-specific prompts and sparse autoencoders for activation monitoring*](https://arxiv.org/abs/2504.20271) | Same chat model; prompted model-output classification versus raw, prompted, and SAE activation probes on expert-labelled harassment/violence, hallucination, and sentiment data | Zero-shot output is a serious baseline; with enough labelled data, activation probes can outperform it. Prompted probing is data-efficient. A suffix-only design can share most of one forward pass across categories. |
| **Direct** | [Meyoyan & Del Corro, *A BERTology View of LLM Orchestrations*](https://aclanthology.org/2026.acl-long.1955/) | Frozen serving LLM; token×layer hidden-state readouts versus zero-shot, few-shot, chain-of-thought prompting, logit reuse, and standalone classifiers on safety, sentiment, and emotion | The same serving pass can support competitive classification without a separate guard call. Learned token/layer aggregation can beat fixed pooling, and probes can substantially outperform prompted output on the same backbone. |
| **Direct** | [Li et al., *Rethinking LLM-as-a-Judge: Representation-as-a-Judge*](https://arxiv.org/abs/2601.22588) | Frozen Qwen3-0.6B/1.7B and Llama models; prompted small-model evaluation versus layer/pooling/classifier searches for five reasoning-quality aspects | Small-model hidden representations can outperform the small model's generated judgments. The paper reports average F1 improvements above 20 percentage points on most studied cells; mean pooling and logistic regression were frequent winners, but selected layers and readouts varied. |
| **Direct, authorization-adjacent** | [Rammouz et al., *Embarrassingly Simple Access Control for Large Language Models*](https://openreview.net/pdf?id=Re0fXzBeXH) | Last-layer, last-token logistic probes versus direct and two-step prompting on Spider, BIRD, and OrgAccess permission decisions with Llama-3.1-8B, Qwen2.5-7B, and Mistral-7B | Access-control decisions can be linearly decoded and can remain more stable than prompted decisions as policy complexity grows. This is close prior art to authorization classification, although it is RBAC rather than user authorization to execute an action. |
| **Direct, routing** | [Ratnakar et al., *LatentGate*](https://aclanthology.org/2026.acl-industry.153/) | Frozen SLM, mean pooling, PCA whitening, and a linear probe versus embedding and prompted routers across five backbones and 100 agents | A non-generative probe can beat prompted routers on the studied routing task while being much faster. The reported setting achieved 98.8% in-domain and 80.0% OOD accuracy at about 28 ms on a T4; prompted routers reported 70–77% at 1.5–2.0 seconds. These are external-system numbers, not an IntentProbe benchmark. |
| **Direct, other domain** | [Schumacher et al., *Prompting Underestimates LLM Capability for Time Series Classification*](https://arxiv.org/abs/2601.03464) | Direct prompt outputs versus linear probes over the same models across six time-series datasets | Prompt output can understate linearly accessible information. Average F1 rose from 0.15–0.26 for prompting to 0.61–0.67 for probes in that study. It also measured prompt-variant sensitivity. |
| **Direct, ambiguity** | [Karamolegkou et al., *Trick or Neat*](https://aclanthology.org/2025.findings-acl.954/) | Direct prompting versus layer-wise linear probes for syntactic, lexical, and phonological ambiguity | Direct prompting was not robust, while probes sometimes exceeded 90% accuracy. This supports the extraction-gap premise on a linguistic property but is not a systems or safety result. |
| **Boundary/adverse** | [Xi et al., *Calibration, Not Answer Selection*](https://arxiv.org/abs/2609.33290) | Hidden-state confidence probe versus verbalized confidence on reasoning models | A probe's expected calibration error was 5–38 times lower in the studied cells, yet the same signal was much less useful for selecting the correct answer. This is evidence that a useful latent signal can be narrow and must not be generalized beyond its label contract. |
| **Adjacent one-pass evidence** | [Yang et al., *You Only Judge Once*](https://arxiv.org/abs/2604.10966) | One-pass multi-response discriminative reward model versus scoring each response separately | At `N=4`, the paper reports up to 3.9× latency and 4.0× FLOP reduction, and near-constant multi-response latency as `N` grows in its setting. This supports the systems logic of sharing a pass, but it scores responses rather than running heterogeneous task heads. |

## The most useful already-published numbers

The ACL orchestration paper gives the closest controlled result to our product argument. On a frozen
Llama-3.2-3B backbone, its Table 4 reports:

| Task | Zero-shot prompt | Few-shot prompt | Chain of thought | Direct pooling probe | Learned MHA probe |
| --- | ---: | ---: | ---: | ---: | ---: |
| IMDB | 77.59 | 76.06 | 91.54 | 94.07 | 95.15 |
| SST-2 | 83.97 | 85.28 | 93.06 | 92.29 | 95.39 |
| Emotion | 44.55 | 33.40 | 56.05 | 69.18 | 87.68 |

The paper labels these sentiment/emotion values as accuracy. They are its measurements on its data,
prompts, hardware, and implementation. We may cite them as prior evidence; we must not present them as
IntentProbe results.

Its ToxicChat runtime test on an RTX 3090, batch size 1, and maximum length 512 reports:

| Pipeline | Average latency | Peak GPU memory |
| --- | ---: | ---: |
| Serving 3B model only | 26.43 ms | 6,497.63 MB |
| + direct pooling probe | 29.66 ms | 6,497.63 MB |
| + scoring-attention probe | 30.90 ms | 6,749.00 MB |
| + MHA probe | 40.27 ms | 6,968.92 MB |
| + separate ToxicChat-T5 | 88.33 ms | 7,992 MB |
| + separate WildGuard | 100.32 ms | 21,084 MB |
| + separate Llama Guard 3 | 123.21 ms | 22,769 MB |

That already supports the generic claim that hidden-state reuse can avoid the latency and memory of a
separate guard pipeline. It does **not** measure a sub-2B CPU sidecar or the marginal cost of many
independently attachable heterogeneous heads.

## Claims we should inherit, with citations

These statements belong in Related Work and Motivation. They do not need a new full experiment:

1. Hidden states can contain classification information that the same model does not reliably express
   through generated labels.
2. Activation probes can outperform direct zero-shot/few-shot output on safety, sentiment, ambiguity,
   reasoning-evaluation, time-series, routing, and access-control tasks in published settings.
3. Reusing a serving model's forward pass can avoid an additional guard model and autoregressive
   decoding.
4. Linear/logistic heads and mean pooling are strong baselines.
5. The best layer, pooling method, and classifier are not universal; they vary by task and model.
6. Prompted output is sensitive to prompt wording and decoding, while a fixed discriminative head has
   a constrained output space.
7. Strong in-domain probe performance does not by itself prove semantic understanding or OOD
   robustness.

## Claims we cannot inherit

These remain IntentProbe experiments because the reviewed literature did not answer the exact
question:

1. whether the result holds on our sub-2B backbones on ordinary local CPU hardware;
2. whether one **unchanged activation extraction** can feed authorization, memory-write, safety,
   sentiment, routing, and dialogue-state heads together;
3. the quality and marginal cost curve as compatible heads increase from `K=1` to `2`, `4`, `8`, and
   `16`;
4. the fair comparison among repeated same-backbone judge calls, one multi-label judge call, one
   prefill plus many heads, and a fleet of compact encoders;
5. whether independently trained, hot-swappable head manifests remain calibrated when served
   together;
6. the model×task×readout interaction on our admitted real-labelled datasets;
7. source-held-out generalization, semantic counterfactuals, and action-intent flips for the actual
   authorization and memory-write definitions; and
8. whether a one-, two-, or three-backbone portfolio dominates one universal sensor under a joint
   accuracy, RAM, and latency budget.

## Revised experiment budget

### Do not rerun

- Do not build another broad experiment whose only conclusion is “hidden-state probes can beat
  prompting.”
- Do not run direct-judge baselines for every readout candidate. Judge results do not select layers or
  pooling methods.
- Do not use every backbone for the expensive generated-judge arm merely because every backbone was
  used in the cheap activation screen.
- Do not present sentiment or toxicity as novel applications. Keep them as positive controls and
  comparability anchors.

### Minimal alignment study

Run the same-backbone prompt/probe comparison only on a preregistered sentinel panel:

- tasks: authorization-to-execute, one safety task, and one mature positive-control task;
- backbones: the selected best-quality sub-2B candidate and the selected cheapest viable candidate;
- methods: selected activation head, zero-shot constrained label prompt, a strong frozen
  rubric/chain-of-thought prompt, and one compact text-classifier baseline;
- repetitions: deterministic decoding plus repeated sampling only where stochasticity is part of the
  comparison;
- metrics: task metric, invalid-output rate, calibration, p50/p95 latency, peak RSS, and uncertainty.

This is a six-cell alignment check, not a new field-wide proof. Expand a cell only if it contradicts
the cited literature or changes the product decision.

### Full IntentProbe experiment

Spend the main compute on the part not already established:

1. select the head manifests using validation data only;
2. freeze one compatible task suite and one or two candidate backbones;
3. measure `K = 1, 2, 4, 8, 16` for one prefill plus `K` heads;
4. compare with `K` separate same-backbone judge calls, one multi-label judge call, and `K` compact
   classifiers;
5. report the full quality–latency–RAM–calibration frontier and marginal slope per added decision;
6. repeat the key operating points on source-held-out data and semantic counterfactuals; and
7. publish every manifest, prompt, parser, invalid output, and raw timing record.

## Paper positioning after this review

The paper should not ask:

> Are activation probes better than directly asking an LLM?

That question is already answered conditionally by prior work.

The paper should ask:

> Given that hidden-state classifiers can outperform or undercut generated judgments, how many
> heterogeneous operational decisions can one frozen, locally runnable small language model serve
> from one prefill, and which task-specific readout manifests preserve quality under a CPU/RAM
> budget?

That framing uses the literature as the foundation and keeps the contribution on the crossed
small-model design map, modular head runtime, and measured many-decision scaling law.

## Citation boundary

External paper numbers belong in the related-work table with their exact model, task, metric, and
hardware. IntentProbe's abstract, model card, and product page may cite those papers for motivation,
but any statement about IntentProbe accuracy, latency, memory, stability, or number of supported heads
must come from an IntentProbe result manifest.

## Completed local evidence

| Question from this map | Result artifact | Answer now supported |
| --- | --- | --- |
| Does probe-versus-judge behavior survive in our selected small backbones? | `experiments/SENTINEL-JUDGE-VS-PROBE-EXP001/result.json` | Conditionally. Direction changes across toxicity, sentiment, authorization, and backbone; six of 12 corrected AUPRC tests reject. |
| Can heterogeneous heads share one unchanged local extraction? | `experiments/ONE-FORWARD-MULTIHEAD-EXP001/HEAD_REGISTRY.json` | Yes for 16 independently trained binary heads on one pinned Qwen2.5-0.5B revision. |
| What is the K scaling curve? | `experiments/ONE-FORWARD-MULTIHEAD-EXP001/result.json` | Mean shared latency rises 6.9% from K=1 to K=16; head work is 1.85 ms at K=16; peak RSS delta is +0.73 MiB in this environment. |
| Is repeated judging fully benchmarked? | Same systems result and assumption ledger | No. Repeated base forwards provide a conservative lower bound; prompted multi-label judging and a compact-classifier fleet remain unmeasured. |
| Are the 16 heads equally reliable? | `experiments/ONE-FORWARD-MULTIHEAD-EXP001/NATIVE-HEAD-QUALITY.csv` | No. Native AUPRC spans 0.0836--0.9804 and calibration varies materially. Serving scalability and task quality are separate findings. |
