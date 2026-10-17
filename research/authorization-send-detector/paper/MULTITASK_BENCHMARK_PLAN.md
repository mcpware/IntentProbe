# Multi-task benchmark plan — where to read agent-safety signals

## Decision and mode

**Mode: SCOPE EXPANSION.** The authorization experiment becomes the completed pilot of a broader
paper. The target study asks whether readout choices transfer across distinct agent-safety tasks, and
whether any small frozen backbone is consistently strong after task-specific data leakage and
calibration are controlled.

This document is a prospective protocol. Only the authorization column currently contains a complete
six-backbone screen and one-shot held-out result. Historical prompt-injection and tool-poisoning
results are prior evidence, not cells in the new matrix until rerun under this common protocol.

## Research question

For locally usable frozen language models, how do **task**, **backbone**, **token pooling**, and
**layer readout** interact when detecting agent-safety boundaries, and do the same readout choices
transfer to an ordinary semantic classification task?

The study separates four questions that should not be collapsed:

1. **Representation:** Is the target label linearly decodable from frozen hidden states?
2. **Readout:** Which token and layer aggregation exposes the strongest validation signal?
3. **Generalization:** Does the selected probe transfer across scenario, source, and template groups?
4. **Deployment:** Do its probabilities and low-error operating thresholds transfer?

## Core tasks

| Task | Decision target | Primary data | Independent unit | Primary test design | Current status |
| --- | --- | --- | --- | --- | --- |
| Authorization | Is this exact proposed action authorized by the request and history? | Existing pilot for historical comparison; DelegationCheck majority labels as the direct external target | matched pair, scenario, and permission-contrast group | matched-pair-preserving external evaluation | Six-backbone pilot complete; current 3,000-row test consumed; no natural-log direct-label source found |
| Prompt injection / task drift | Does untrusted content contain or induce instruction takeover? | Existing four-source development pool; LLMail-Inject human attacks as the primary action-shaped positive external source; a separately reviewed natural-email negative set; SaTML CTF and HackAPrompt as stress tests | team, source, defense, scenario, and attack/template family | participant/source/defense-held-out; natural-email FPR is unavailable until the negative audit is frozen | Historical Qwen mean-pooling results exist; LLMail's 203 benign emails are synthetic fixtures, not real-inbox negatives; common matrix not run |
| Tool / skill poisoning | Is a tool description or skill package trying to expand authority, exfiltrate, persist, or redirect execution? | Existing development data; MASB confirmed positives plus a newly reviewed matched benign marketplace set | repository, author/account, campaign, corpus, and structural family | repository/campaign/source-disjoint | Historical results exist; real-world negative gold must be built before the final test |
| Memory write | Should this conversational fact be saved durably rather than used only in the current turn? | `memory-write-gate-20k` for silver development; historical LUFY triple annotations for human-trained grouped evaluation; AlpsBench v4 Task 2 for a later update-policy view | `seed_id` for silver data; user for LUFY; user/session/update lineage for later real data | readout selection on 12 users, then one test on five held-out users | Qwen human screen complete: masked-mean outputs 7--8 win development and reach 0.1575 AUPRC / 0.7842 AUROC on 1,748 test turns, statistically unresolved versus TF-IDF at 0.1726 / 0.7892; historical files remain in licence quarantine |
| Toxicity generality control | Is this real user message toxic under the source's human-label definition? | ToxicChat 0124 human-annotated rows for the primary task; Civil Comments for separately reported domain transfer; HateXplain only as a later rationale/target robustness set | conversation/source cluster for ToxicChat; article/publication/time for Civil Comments | human-gold-only grouped evaluation; external domain transfer reported separately because the label definitions differ | Official sources verified; exact ToxicChat file counts and licence gate must be frozen before acquisition; common matrix not run |

The toxicity row is a **predeclared generality control**, not part of an agent-safety macro-average.
It tests whether the engineering conclusion about where and how to read activations extends beyond
attack and permission labels. A positive or negative transfer result is informative; toxicity must
not be renamed as authorization, intent, or prompt injection.

### External and stress-only sets

- **LUFY historical annotations:** 6,470 human user turns from 17 users across three memory systems,
  each carrying three binary future-use votes. Majority vote is the primary external label;
  unanimous rows form a high-agreement sensitivity view and 2--1 rows form a disagreement view.
  Bootstrap users, not turns. These labels measure predicted future utility, not consent to store.
  The deleted historical annotation files remain quarantined until their licence scope is confirmed.
- **AuthorizationPairs:** same operation with different permission evidence; external authorization
  evaluation. Preserve its group IDs.
- **DelegationCheck:** four-way `ACT / ASK_AUTHORIZATION / REQUEST_INFORMATION / REFUSE`, three
  annotators, and 48 matched pairs. Use the 149 majority-labelled scenarios as the confirmatory
  reference and the seven no-majority scenarios as ambiguity tests. This is human-authored rather
  than naturally occurring log data.
- **LLMail-Inject:** human competition submissions in a simulated email agent, with observable
  retrieval, detection, tool-call, destination, and content outcomes. Keep execution outcomes
  separate from LLM-judge `attack_attempt` annotations and never relabel a failed attack as benign.
  Its 203 Project Zenith/Orion/Apollo/Aurora emails are benchmark-authored fixtures: report them as
  protocol regressions, never as a natural-inbox false-positive set. A manually reviewed,
  source-matched Enron/SpamAssassin ham sample is required for the latter claim.
- **SaTML CTF and HackAPrompt:** human-adversarial stress tests. Positive-heavy or attack-only slices
  cannot produce AUROC or a false-positive rate by themselves.
- **MaliciousAgentSkillsBench:** use the 157 sandbox-confirmed, independently reviewed malicious
  marketplace skills as confirmed-positive evidence. The `safe` and `suspicious` tiers are not
  automatically binary gold.
- **SkillsAlign:** 552 human-verified in-the-wild skills for an adjacent alignment task; metadata-body
  misalignment is not the same label as maliciousness.
- **AlpsBench:** real WildChat dialogue with human-verified structured memories. Native Task 2 is a
  memory-update benchmark; a derived `ADD/MODIFY/RETAIN` view needs deterministic lineage and new
  human audit before it can become a write-gate test.
- **MemUse:** four-month real diary-companion deployment with 72 human-verified memory-cued moments,
  72 matched no-memory turns, and full histories. Use as a user-held-out auxiliary test of
  `RECALL NOW / DO NOT FORCE RECALL`; do not rename it a memory-write benchmark.
- **Enron action signals:** Lampert request labels, EPA task-recipient labels, Sappelli intent/task
  annotations, and Parakweet sentence labels support a product-facing `ACTION/REPLY NEEDED` appendix.
  They do not replace authorization labels. Licence and Enron privacy gates remain in force.
- **ToxicChat 0124:** real Vicuna-demo prompts. The current card reports 10,166 prompts and 5,634
  human annotations, while the Hub split API reports 5,082 train plus 5,083 test rows. Freeze the
  exact files and reconcile that one-row/card-count discrepancy before analysis. Use only rows with
  `human_annotation=true` for confirmatory scoring. This is the canonical generality control, but
  remains outside the agent-safety macro-average.
- **Civil Comments:** large real public-comment toxicity source with crowd-label fractions and CC0
  release. Use as a separately reported domain-transfer test; its label definition is not identical
  to ToxicChat's binary human annotation.
- **PIDS-Bench and benign-instruction-bench:** frozen/rebuildable prompt-injection stress suites for
  hard-benign, agent-tool-output, and carrier-format transfer. They are external evaluation sources,
  not extra rows to pool into development.
- **SafeClawBench / FORTIS:** scenario sources for later cross-boundary evaluation; do not silently
  remap their labels into binary training labels.
- **MemoryCtrl / SP-Mem:** retrieval, no-use, and forget evaluations. They do not become `SAVE/SKIP`
  training data.

The full provenance, licence, label-quality, revision, and acquisition decisions are frozen in
[the real-world gold dataset audit](../../datasets/REAL_WORLD_GOLD_DATASET_AUDIT_2026-10-16.md).

## Prior-art and novelty lock

The paper must **not** claim to be the first activation detector or the first activation-probe study
of prompt injection.

- [*When Benchmarks Lie*](https://arxiv.org/abs/2602.14161) (Fomin, 2026) already trains logistic and MLP probes on Llama-3.1-8B/70B
  activations over 18 malicious/benign sources, uses leave-one-dataset-out evaluation, and reports
  strong layer/position heterogeneity. Its primary readout is the last user token at the final layer;
  the sensitivity analysis checks selected late layers and two token positions.
- [*Prompted to Discriminate*](https://arxiv.org/abs/2610.02413) (David and Fomin, 2026) already compares single-position, mean,
  attention, multi-max, and MLP pooling under suffix prompting across Llama-3.1-8B, Qwen3.5-9B, and
  Gemma-4-12B on 13 safety benchmarks. It checks two late layers per family rather than a full
  lightweight-model layer sweep.
- [*Aligned Probing*](https://aclanthology.org/2026.tacl-1.14/) (Waldis et al., 2026) already probes
  mean-pooled input/output activations at every layer of 7B--8B OLMo, Llama, and Mistral models for
  six toxicity attributes and reports particularly strong encoding in lower layers. Toxicity is
  therefore a replication/generality control here, not a new application of activation probing.
- [*Lightweight Safety Classification Using Pruned Language Models*](https://arxiv.org/abs/2412.13435)
  already evaluates Qwen2.5-0.5B/1.5B/3B on content safety and SPML prompt injection, scans for an
  optimal intermediate layer, fits penalized linear classifiers, and tests concatenating preceding
  layers. Its main split is random 66/33, its content-safety labels combine unsafe SALAD rows with
  benign LMSYS rows, and its headline evaluation is weighted F1 rather than source/carrier-held-out
  transfer, calibration, or low-FPR operation. Small Qwen models, intermediate-layer search, and a
  linear probe are therefore prior art; our study must be framed as a replication and extension.
- [*Passing the Test You Trained On*](https://arxiv.org/abs/2610.03448) (Liu, 2026) shows that
  detector rankings transfer poorly from short/document benchmarks to serialized agent tool
  outputs, so carrier-format transfer and low-FPR evaluation are required rather than optional.
- [TaskTracker](https://arxiv.org/abs/2406.00799) and related task-drift work already establish
  activation-based runtime monitoring as a method family.

The defensible contribution is narrower and engineering-specific:

1. replicate and test the intermediate-layer finding across six independent, locally usable
   **230M--1B** backbones under one paired protocol and a normalized-depth coordinate system;
2. compare token pooling, bounded adjacent-layer windows, and layer concatenation under a fixed
   selection budget, rather than treating the hidden-state layer as the only readout choice;
3. test semantically distinct product tasks: authorization, prompt injection, tool/skill poisoning,
   memory write, plus toxicity as a non-agentic generality control;
4. use real or human-labelled data with source/carrier-held-out transfer, explicit format-confound
   controls, calibration, and low-FPR operating points instead of relying on random row splits;
5. compare against matched-data frozen and fine-tuned text encoders and report measured
   CPU/RAM/latency trade-offs.

The headline question is therefore not “do activations contain a safety signal?” It is: **for small
local models, which readout choices survive changes in task, source, and carrier, and what accuracy–
deployment trade-off do they buy over matched text classifiers?**

## What already exists

| Asset | What it answers | What it does not answer | Treatment in the expanded paper |
| --- | --- | --- | --- |
| [Authorization EXP002 screen](../experiments/EXP002-activation-screen/screen_leaderboard.json) and [held-out test](../experiments/EXP002-activation-screen/heldout_result.json) | Six-backbone simple readout screen; frozen MiniCPM test; same-budget TF-IDF | Cross-task transfer or independent scenario families | Completed pilot and first matrix column |
| [PI Monte Carlo / fixed config](../../_results_published/pi_montecarlo_analyzed.json) | Four-source outer holdout with mean-pooled Qwen activations; historical mean AUROC 0.984 versus TF-IDF 0.914 | Six-backbone comparison, alternative pooling, common renderer, group-level uncertainty | Historical result only until common-protocol rerun |
| [Tool-poison cross-corpus](../../_results_published/toolpoison_crosscorpus.json) | Three outer corpus folds; historical mean AUROC 0.624 versus 0.542 | Reliable minimal-pair thesis or a universal readout | Historical result; rerun after data repair |
| [Shipped Qwen probe](../../../intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2/metadata.json) | Qwen2.5-0.5B, pooled token means, concatenated outputs 13--15, logistic head | Honest out-of-sample product accuracy; metadata metric is training-set fit | Artifact identity only, never a performance estimate |
| [Memory Write Gate source QA](../../memory-write-gate/data/manifests/qa_summary.json) | 20,000 unique rows, balanced labels, 200 seed families, no seed crossing official splits | Human-written generalization or user preference change | Valid silver bootstrap with seed-clustered inference |
| [LUFY historical annotation audit](../../memory-write-gate/HUMAN_DATASET_AUDIT_2026-10-17.md) | 6,470 aligned human user turns, three importance votes per turn, user/system/session lineage | Storage consent, privacy policy, or a public redistribution right for deleted historical files | Frozen zero-shot external transfer with majority, unanimous, and disagreement views; raw data stays quarantined |

## Predeclared backbone panel

The canonical panel remains the six completed authorization backbones so the task comparison is
paired by model:

1. `LiquidAI/LFM2.5-230M`
2. `ibm-granite/granite-4.0-350m-base`
3. `Qwen/Qwen2.5-0.5B`
4. `Qwen/Qwen3-0.6B-Base`
5. `Qwen/Qwen3.5-0.8B`
6. `openbmb/MiniCPM5-1B-Base`

Qwen2.5-1.5B remains a historical/reference sensor because older prompt-injection experiments used
it. It does not enter the primary six-by-task matrix unless the entire matrix is expanded under a
separate preregistered compute tier. Falcon-H1 remains a recorded CPU-feasibility failure until a
comparable implementation can complete the extraction protocol.

## Representation search

**Interpretation lock:** the authorization pilot established that masked mean beat the tested
last-token alternatives for that dataset and renderer. The historical prompt-injection and
tool-poisoning runs *used* token means but did not compare the pooling families below. They therefore
cannot be cited as evidence that masked mean already wins across tasks. That claim is available only
after the common-protocol reruns.

### Tier 1 — confirmatory core family

The core family is deliberately bounded and identical across tasks.

#### Token readouts

1. **Masked mean:** featurewise mean over all non-padding tokens.
2. **Last non-padding token:** conventional decoder-only endpoint baseline.
3. **Featurewise maximum:** maximum activation per hidden dimension across non-padding tokens.
4. **Field/span mean:** separate means for semantically defined runtime fields, when deterministic
   boundaries exist. Examples: proposed action, user request, history/untrusted content, tool
   description, candidate memory.

Field/span pooling is omitted—not approximated—when a dataset cannot supply reliable boundaries.
This prevents a parser-quality difference from masquerading as a representation result.

#### Layer readouts

1. every individual hidden-state output;
2. adjacent standardized means of width two and three;
3. adjacent concatenations of width two and three, with the head's effective capacity reported;
4. one normalized-depth rule selected without using the target test task/model cell.

All feature scaling is fitted on training folds only. “Layer” tables must distinguish the embedding
output from transformer-block outputs.

### Tier 2 — exploratory representation engineering

These candidates answer useful engineering questions but do not join the confirmatory search family:

- **top-k token-score pooling:** train a token-level direction, then aggregate the largest k scores;
- **rolling token-score maximum:** detect a short malicious or authorization-bearing span inside long
  context;
- **mean--max concatenation:** retain both distributed context and a sparse activation spike;
- **moment pooling:** concatenate featurewise mean with standard deviation to retain activation
  dispersion that a mean erases;
- **smooth maximum pooling:** use a fixed-temperature LogSumExp or generalized mean so a few strong
  tokens can dominate without the instability of a hard maximum;
- **learned attention pooling:** a small scoring function over tokens, with a fixed parameter budget;
- **position-weighted or gated pooling:** learn a bounded mixture of prefix, middle, and suffix regions
  without giving every token a free attention parameter;
- **layer delta / trajectory:** use differences between adjacent hidden outputs rather than a static
  output;
- **trusted-prefix delta:** compare representations before and after untrusted content when the task
  has a valid paired prefix;
- **field interaction features:** differences and cosine similarities between action, request,
  untrusted-content, tool-description, or candidate-memory span means;
- **sparse non-adjacent output combinations:** bounded forward selection inside nested validation;
- **classification suffix readout:** append a frozen content-free classification instruction and
  measure whether it improves cross-source transfer;
- **anchor-token readout:** append one fixed neutral sentinel and read only its final hidden state,
  testing whether a common endpoint can gather the preceding context;
- **whitened or robust-scaled features:** preprocessing fitted inside the training fold only.

Tier 2 is reported separately. It cannot replace a disappointing Tier-1 result after the test is
opened.

The alternatives test different hypotheses rather than adding arbitrary knobs:

| Readout family | Signal hypothesis | Main failure mode to test |
| --- | --- | --- |
| Masked or span mean | The concept is distributed across many relevant tokens | Dilution by long clean or irrelevant context |
| Hard/smooth max or top-k | A short token span carries the decisive signal | Sensitivity to one noisy activation spike |
| Mean--max or moment pooling | Both level and dispersion matter | Larger feature/head budget creates an unfair gain |
| Attention/gated pooling | Relevant token positions vary by example | Extra learned parameters overfit templates |
| Anchor/suffix token | A fixed endpoint can integrate the whole causal prefix | Prompt wording creates the signal being measured |
| Layer mean/concat/delta | The concept is distributed or changes through depth | Extra dimensionality or scaling, not layer information, drives the gain |
| Field interactions | Authorization is a relation between action and evidence | Field parser or renderer leaks the label |

### Probe heads and output-level baselines

Primary head: class-weighted logistic regression.

Sensitivity heads:

- difference-in-means direction;
- linear SVM with validation-only probability calibration;
- small MLP with a fixed parameter budget;
- backbone output-logit or direct verbal classification baseline when its interface supports one.

The minimum matched surface baseline is word-plus-character TF-IDF logistic regression, trained on
the same rows. It is not strong enough to support a broad “activations beat text classifiers” claim
by itself. Each task therefore also receives:

- a frozen text-embedding/encoder plus logistic head trained on the identical folds;
- one sub-100M fine-tuned encoder trained on the identical rows, groups, selection budget, and test;
- task-relevant released detectors as **external product baselines** where available (for example,
  [Prompt Guard 2](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M) and
  [ProtectAI](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) on prompt
  injection, and an [Opir](https://arxiv.org/abs/2605.29659) edge model on toxicity after its licence
  and artifact are verified), with training-data overlap audited and clearly separated from
  matched-data comparisons.

Action-only, request-only, history-only, and explicit-keyword rules are required ablations when those
fields exist. A claim that activation features beat text classification is allowed only if they beat
the matched fine-tuned encoder on the untouched grouped test, not merely TF-IDF.

Every task also receives a shortcut-control package:

- a source/carrier classifier trained on the same representations, to quantify how easily the probe
  can identify dataset provenance;
- label-permutation/selectivity controls fitted with the identical nested procedure;
- length, language, renderer, field-count, and explicit-cue summaries by class and split;
- within-source metrics wherever both labels exist, plus leave-one-source-out metrics;
- matched or residualized sensitivity analyses when a format feature predicts the target;
- a rule that dataset name, label, rationale, moderation score, annotator status, and evaluation
  metadata never enter the rendered model input.

High held-out accuracy is not evidence of a concept signal if the same representation primarily
separates source or task format. This control package is mandatory for every headline cell.

## Study design

```text
raw source snapshots (immutable, hashed)
                |
                v
task-specific label + provenance audit
                |
                v
scenario/source/seed groups ------------------------------+
                |                                         |
                v                                         v
development groups                               untouched external groups
       |                                                  |
       +--> train folds: scaler + probe fit               |
       +--> inner validation: readout selection           |
       +--> outer development: estimate stability         |
                                                          |
frozen renderer + model + pooling + layers + head --------+
                                                          |
                                                          v
                                             one-shot paired evaluation
                                                          |
                                                          v
                                      discrimination + calibration + cost
```

Rows are never the independent unit when a source supplies scenario, pair, seed, author, or template
groups. If no defensible group can be reconstructed, the dataset can support descriptive row-level
results but not group-generalization claims.

## Measurement graph

This benchmark is predictive, not causal. The following graph records why a high probe score cannot
be interpreted as a discovered mechanism:

```text
true task concept ---------------------> hidden representation -------> probe score
      |                                      ^       ^                       ^
      |                                      |       |                       |
      +--> wording / action / context -------+       |                       |
                                                     |                       |
backbone + tokenizer + pretraining ------------------+                       |
                                                                             |
renderer + truncation + field order -----------------------------------------+
                                                                             |
source generator / template / annotation process ----------------------------+

pooling + layer rule + scaling + head + selection set -----------------------> measured metric
```

The study estimates predictive performance of a complete measurement pipeline. It does not identify
whether the backbone causally relies on the decoded direction.

## Primary estimands and hypotheses

### Per-task primary estimand

On the untouched grouped test for task t:

`Delta_t = AUPRC(selected activation probe) - AUPRC(selected matched-data text baseline)`.

The matched-data text comparator is selected wholly inside development data from word+character
TF-IDF, frozen encoder features, and the fixed-budget sub-100M fine-tuned encoder. All three remain
visible in the result table; the strongest development-selected text system is the confirmatory
comparator, so the activation result cannot win merely because TF-IDF is weak.

Report the paired scenario/group bootstrap estimate and 95% interval. The primary family contains
four agent-safety task-level comparisons; use Holm correction across the four primary p-values if
p-values are reported. Toxicity uses the same estimand as a separately named generality-control
hypothesis and does not enter the agent-safety family. Effect sizes and intervals remain primary.

### Secondary estimands

- paired AUROC difference;
- recall at predeclared 1% and 5% false-positive/false-approve/false-save rates;
- Brier score, expected calibration error, and reliability curves;
- final-output regret relative to the selected candidate;
- universal-readout regret relative to each task/model optimum;
- CPU p50/p95 latency, throughput, peak RSS, cached-feature size, and head size.

### Cross-task questions

1. How often does each pooling family win across the 30 task-by-backbone cells?
2. Does a common normalized-depth window stay within a preregistered regret margin of task-specific
   selection?
3. Is backbone rank stable across tasks, or is there a task-by-model interaction?
4. Does the model with the best discrimination also have the best calibration and low-error recall?
5. Does prompt injection transfer to tool poisoning, or are their internal signals distinct?

Selection counts over 30 cells are descriptive because cells share data, models, and training
procedures. A mixed-effects model is deferred until repeated grouped splits provide a defensible
error structure.

## Split and leakage locks by task

### Authorization

- Preserve permission-contrast and scenario groups.
- Proposed action, request, and prior history are runtime features.
- Labels, rationales, evaluator notes, and publisher-composed combined text are excluded.
- The consumed 3,000-row test remains a historical test; all new selection requires a new external
  test.

### Prompt injection

- Outer split is by source, not shuffled row.
- Attack templates and paraphrase families must remain within one fold.
- LLMail rows are grouped by team/participant, normalized attack family, phase, scenario, and
  level/defense. Repeated submissions across levels never cross folds.
- Observable LLMail flags and semantic attack annotations are separate endpoints.
- SaTML chats are grouped by attack team, defense ID, conversation, and strategy cluster.
- A positive-only set reports recall only at a threshold calibrated on a separate mixed-label source.
- Direct injection, indirect injection, task drift, jailbreak, and malicious instruction are reported
  separately before any macro average.
- Run a strict leave-one-source-out analysis and a carrier-format transfer analysis. At minimum,
  report short-prompt, embedded-document/email, and serialized tool-output carriers separately.
- Keep `benign-instruction-bench` AgentDojo/tau-bench outputs and PIDS-Bench hard-benign/domain-shift
  sets untouched until the renderer, layer/pooling selector, and threshold are frozen.
- Deepset's small mixed-label set may support development or comparability, but it cannot be the only
  test and its source/card label is not independent human gold.

### Tool / skill poisoning

- Corpus, risk family, and clean/poison pair stay grouped.
- Repository, publisher/account, campaign, and structural near-duplicate family also stay grouped.
- Real-world false-positive claims require independently reviewed benign rows sampled from the same
  registry/time/language/length/popularity strata as the malicious rows.
- MASB `suspicious` is a weak candidate label, not poisoned gold; MASB `safe` is not independently
  confirmed benign gold.
- Metadata summaries and redaction placeholders are never rendered as if they were original skill
  text.
- The historical `hard_v3_matched_*` length-matched set is not evidence of vocabulary-matched intent.
- Any new minimal-pair set needs pair IDs, build-time lexical-similarity checks, independent review,
  and complete separation from training.
- Tool name, description, manifest metadata, and executable content are separate modalities; missing
  modalities are not filled with label-correlated placeholders.

### Memory write

- Split by `seed_id`, not paraphrase row; report both rows and 200 independent seed families.
- `SAVE/SKIP`, `FORGET/KEEP`, and retrieval authorization remain separate targets.
- Explicit “remember” keywords receive a dedicated cue-stripped or cue-matched evaluation.
- Human-user tests group by author/user and never publish raw private conversation content.
- AlpsBench-derived rows group by user, session, canonical ID, and update lineage.
- A missing gold evidence link is not automatically a `SKIP` label. Derived transitions are
  `ADD`, `MODIFY`, `RETAIN`, or `UNCERTAIN`, with two-reviewer adjudication before binary mapping.
- Native AlpsBench Task 2 results are reported separately from the derived activation-classification
  task so transformation error cannot masquerade as model error.
- MemUse groups by user, session, and matched trigger pair; gold facts, sub-questions, and detector
  metadata are excluded from the rendered input. Report leave-one-user-out sensitivity because
  72 positive and 72 negative turns are too small for a row-random headline.

### Toxicity generality control

- Use ToxicChat 0124 only after file-level counts, hashes, fields, and licence are frozen. The
  confirmatory slice contains only `human_annotation=true`; `openai_moderation` and annotation-status
  fields never enter the renderer.
- Keep conversation IDs and recoverable source/duplicate clusters intact. Report class prevalence
  and exact human-gold counts after acquisition rather than copying the unresolved card/API totals.
- Select a binary toxicity threshold and ambiguity treatment before opening the held-out groups.
- Civil Comments is external domain transfer. Group by article, publication, and time; do not use
  `parent_text` with the official random split because the official card warns that parent/child
  leakage is possible.
- Do not pool ToxicChat and Civil Comments into one random split. Their carriers and annotation
  definitions differ, so cross-source performance is a transfer result, not an IID test.
- HateXplain and OLID remain optional robustness sets after data-rights and redistribution review;
  neither is needed for the first confirmatory matrix.

### Action-needed appendix

- Keep mailbox/user, message ID, thread, quoted-message chain, and task together.
- De-duplicate the same Enron message copied into multiple mailboxes before any split.
- Lampert's 505 unanimous request labels are the preferred untouched target if the authoritative
  annotation artifact can be obtained. The paper alone is not a dataset licence.
- EPA's unanimous email-task-recipient tuples test a relational target; Sappelli's two-annotator
  labels and Parakweet's single-reviewer labels are separate tasks and must not be merged into one
  binary class.
- This appendix enters the paper only after lineage, licence, and frozen-split review. Otherwise it
  becomes a follow-up product benchmark, not a missing requirement for the four-task paper.

## Assumption registry

| Method | Assumption | Diagnostic | Fallback |
| --- | --- | --- | --- |
| Logistic probe | Linear decision surface is adequate for the stated head | Compare fixed-budget linear SVM and small MLP on development groups | Report linear limitation; do not expand after test |
| Class weighting | Development prevalence differs from decision costs in a manageable way | Report unweighted metrics and prevalence by split | Cost-sensitive thresholds reported separately |
| Feature scaling | Train-fold mean/variance is stable and non-degenerate | Count zero/near-zero variance features; compare split moments | Floor scales; robust scaling as Tier 2 |
| Grouped CV | Groups contain dependence and groups are exchangeable enough for transfer | Audit duplicate/template leakage and group-size imbalance | Leave-one-source-out or report source-specific results |
| AUPRC | Class prevalence is meaningful and comparable within a test | Report prevalence with every AUPRC | Also report AUROC and prevalence-normalized lift |
| Cluster bootstrap | Groups approximate independent resampling units and are numerous enough | Report group count, size distribution, and influential groups | Exact/permutation analysis where feasible; label interval unstable |
| Probability calibration | Development-to-test score mapping is sufficiently stable | Reliability curves, ECE, Brier, score-distribution shift | Recalibrate only on new calibration data; abstain otherwise |
| Cross-task rank | Task cells are comparable after common metrics and protocol | Report per-task ranks and rank uncertainty | Avoid “generally best” claim |
| Paired baseline comparison | Both systems score identical test units | Assert identical IDs/order and paired bootstrap indices | Reject comparison if pairing cannot be verified |
| Field/span pooling | Boundaries are deterministic and label-independent | Parser tests and boundary coverage report | Omit field pooling for that dataset |

## Power and precision plan

Most datasets have fixed size. Before opening each final test:

1. use development-group predictions to simulate paired group-bootstrap precision;
2. report the minimum detectable paired AUPRC difference at 80% and 90% power for the available
   group count;
3. define a smallest practically useful difference before seeing test results;
4. if the available independent groups cannot resolve that difference, describe the test as
   estimation/exploration rather than a confirmatory win/loss test.

Rows never substitute for independent group count in the power statement.

## Multiple-comparison strategy

1. **Primary family:** four activation-versus-selected-matched-text AUPRC comparisons; Holm correction.
2. **Secondary held-out metrics:** intervals and effect sizes, explicitly secondary; no claim mining.
3. **Readout search:** nested entirely inside development data; publish every candidate curve and the
   candidate count.
4. **Subgroups:** false-discovery-rate-controlled exploratory analyses within each task; no pooling of
   tiny or single-class slices into headline metrics.
5. **Tier 2:** separate exploratory appendix and a new untouched test if it changes the frozen model.

## Calibration and shift diagnostics

Every task produces the same diagnostic package:

- train/validation/test score distributions by class;
- AUROC and AUPRC with paired group-bootstrap intervals;
- Brier score and ECE with reliability plot;
- threshold table showing development target versus observed test error;
- language, source, difficulty, action/risk family, and explicit-cue slices where supported;
- score-shift decomposition distinguishing ranking degradation from probability-scale drift;
- abstention/confirmation coverage-risk curve.

The authorization pilot's threshold drift makes this package mandatory, not optional.

## Planned figures

1. **Task × backbone heatmap:** held-out AUPRC lift over the development-selected matched-data text
   baseline, with unavailable cells shown explicitly rather than imputed.
2. **Normalized-depth ridge/line plot:** complete validation AUPRC curves by model and task.
3. **Pooling regret forest plot:** per-cell regret and grouped uncertainty relative to the selected
   readout.
4. **Universal-rule regret plot:** task/model-specific optimum versus shared readout.
5. **Calibration panels:** validation and test reliability plus frozen-threshold movement.
6. **Pareto frontier:** performance versus CPU latency, peak RSS, and head size.

All figures use colorblind-safe palettes, confidence intervals where inferential, and 300-DPI export.

## Execution order

### Phase A — freeze the common protocol

- complete the source/licence/label audit in
  `research/datasets/REAL_WORLD_GOLD_DATASET_AUDIT_2026-10-16.md`;
- pin repository revisions and file hashes before retrieval;
- reserve real-world external tests before any new readout search;
- finalize the core pooling/layer family and renderer contracts;
- specify group IDs, primary test source, primary metric, baseline, and correction family per task;
- hash all inputs and freeze the six-model revision table;
- calculate development-only power/precision expectations.

### Phase B — rebuild the two historical tasks

1. prompt injection under six backbones and the common representation family;
2. tool/skill poisoning after removing contaminated or falsely described evaluation sets;
3. run TF-IDF, frozen-encoder, and fixed-budget fine-tuned-encoder baselines on identical folds;
4. compare new outputs with historical results without silently treating them as identical experiments.

### Phase B2 — run the toxicity generality control

1. freeze ToxicChat 0124 metadata, exact file hashes, human-gold count, duplicate groups, and licence;
2. reserve the grouped ToxicChat test before any layer or pooling search;
3. run the same six-backbone/readout budget and all three matched-data text baselines;
4. after selection and calibration are frozen, open the Civil Comments transfer slice once;
5. report this row separately from the agent-safety macro-average.

### Phase C — run Memory Write Gate

- start with majority and keyword rules, then the three matched-data text baselines;
- run Qwen2.5-0.5B as the pilot;
- only then complete the six-backbone screen using seed-family splits;
- reserve human-written or external data before readout selection;
- evaluate AlpsBench native updates and audited transitions separately;
- run MemUse only as a frozen user-held-out memory-use auxiliary task.

### Phase D — optional real-email action appendix

- resolve Enron text lineage, annotation licence, privacy, and duplicate-message handling;
- run rule and all three matched-data text baselines before any activation extraction;
- add one clearly named task (`REQUEST PRESENT`, `REPLY EXPECTED`, or `TASK OWNER`) rather than
  collapsing all action labels;
- omit the appendix if no authoritative frozen test artifact clears the gates.

### Phase E — cross-task synthesis

- compute pooling and normalized-depth regret from frozen per-task development results;
- open each untouched external test once;
- generate paired uncertainty, calibration, and resource tables;
- rewrite the abstract and title from actual cross-task results.

## Release gates

- No result enters the paper without a config, data hashes, group audit, full candidate curve, and
  executable reproduction command.
- No “best model” claim uses validation maxima alone.
- No “masked mean is universally best” claim is permitted unless it wins across the new task matrix
  and survives external grouped tests; even then the claim is bounded to the evaluated tasks/models.
- No detector is called a standalone runtime gate unless frozen low-error thresholds transfer.
- No “activations beat text classification” claim is permitted unless the probe beats the
  development-selected matched-data text comparator, including the fixed-budget fine-tuned encoder,
  on an untouched grouped test.
- No dataset or weight is released until provenance, license, privacy, and derivative-use review is
  complete.

## Not in scope for the first expanded paper

- vision/audio activations;
- closed-weight APIs that do not expose hidden states;
- models larger than the declared local-compute tier;
- causal claims about the mechanism used by the backbone;
- automatic execution based solely on a probe score;
- merging write, delete, retrieval, authorization, and poisoning labels into one “safe/unsafe” head;
- unrestricted neural architecture search over pooling and layer combinations.

## Paper claim ladder

The final wording depends on the completed matrix:

1. **Minimum defensible:** readout selection materially affects multiple agent-safety tasks.
2. **Stronger:** masked token pooling and mid-to-late normalized-depth windows transfer with low regret
   across the evaluated tasks and small backbones.
3. **Strongest allowed by this design:** one bounded shared readout rule is competitive with
   task/model-specific selection while retaining held-out gains over the selected matched-data text
   baselines.

Failure is informative. If masked mean stops winning, that supports task-specific readouts. If
activation gains vanish under group/source holdout, the paper should say the earlier signal was
largely dataset-specific. If one backbone wins every task with stable calibration, that supports a
general sensor-backbone claim; otherwise the product should choose by task.

