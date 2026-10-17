# Authorization Send Detector — Living Plan

Last reconciled: 2026-10-17 after EXP002 held-out evaluation and Hugging Face research release

## Outcome

Build and honestly evaluate a local runtime component that distinguishes:

1. `ALLOW` — the proposed action is within an already reviewed and valid authorization scope;
2. `CONFIRM` — intent may be present, but an exact approval is absent, stale, ambiguous, or narrower
   than the proposed action;
3. `BLOCK` — the action conflicts with the user's instruction, exceeds scope, or must never be
   inferred from conversational pressure alone.

The immediate benchmark asks whether frozen small-model activations add useful authorization signal
over surface-text and dedicated encoder baselines. The product is allowed to use an activation sensor,
but it is not allowed to delegate final send authority to a probabilistic classifier.

## Product contract

### Activation sensor

Reads the user request, relevant history, proposed action, and compact execution-state summary. Emits
calibrated probabilities plus the layers/pooling/head version used.

### Deterministic authorization state machine

Checks exact facts that should never be guessed:

- whether the complete outbound payload was previewed;
- recipients, CC, subject, body, attachments, channel, and action type;
- whether approval covers one item or an explicitly enumerated batch;
- whether the current payload hash matches the preview hash;
- whether an intervening user message invalidated unused approval;
- whether the approval phrase and timing satisfy the configured policy;
- whether the proposed action is externally consequential or irreversible.

The runtime receipt records `sensor_score`, `state_facts`, `policy_rule`, and final decision separately.

## Non-goals

- A generic harmful-content detector.
- A classifier that silently authorizes an outbound message because it predicts likely consent.
- Replacing deterministic payload binding with an LLM judge.
- Claiming all agent actions share the same authorization policy.
- Optimizing on the final external test set.
- Presenting a public research probe as a production authorization gate before licensing, leakage,
  calibration, and baseline gates pass.

## Core research questions

1. Does an activation probe outperform TF-IDF and output-logit baselines on contrast pairs where the
   proposed operation is identical but authorization evidence changes?
2. Does it add value over dedicated encoders such as `auto-0.4b` and `auto-1b`?
3. Is the best authorization signal model-specific, or stable at a normalized relative depth?
4. Does one backbone perform generally well across authorization, memory-write, and control tasks, or
   do task-by-model interactions dominate?
5. Can a universal normalized-depth window stay within a predeclared regret tolerance of each model's
   individually selected best layer/window?
6. Does activation sensing still help when lexical cues, action names, tools, frameworks, languages,
   and scenario families are held out?

## Data tiers

| Tier | Meaning | Allowed use |
|---|---|---|
| Gold | deterministic state-machine label or human-adjudicated exact context | train, validation, final test if isolated |
| Silver | synthetic/model-labeled with documented generation and spot-checks | train and development only |
| External evaluation | benchmark authored outside this project | untouched final evaluation |
| Quarantine | unclear license, weak provenance, contradictory or ambiguous label | analysis only; never release-trained weight |

The initial public sources are not treated as one homogeneous pool. `auto-1b-data` is large silver
training data; `approve-or-deny` is held-out silver evaluation; AuthorizationPairs and AuthGateBench
are external contrastive evaluation; SafeClawBench is evaluation-only; FORTIS remains quarantined for
release training until its dataset license is explicit.

## Benchmark phases

### Phase 0 — provenance and QA (active)

- [x] Create independent project folder.
- [x] Freeze source and model registries.
- [x] Specify three-class product semantics.
- [x] Download all 14 registered non-model source artifacts.
- [x] Generate per-file SHA-256 and retrieval manifest.
- [x] Execute schema, null, label, duplicate, contradiction, and available group-leakage QA.
- [ ] Resolve each source's release-training license status.
- [x] Freeze external tests before any model/layer selection.

### Phase 1 — honest baselines

- [x] Build a canonical context renderer with versioned templates.
- [x] Full-data word TF-IDF logistic regression (EXP001).
- [x] Dedicated encoder baselines: `auto-0.4b`, `auto-1b` (EXP002 screen; publisher-trained budget
  is not matched to the probe).
- [ ] Backbone output-logit/zero-shot baseline where meaningful.
- [ ] Majority and action-only baselines.
- [x] Report calibration and false-approve/false-deny costs, not accuracy alone.

### Phase 2 — four-backbone activation screen

Initial matrix: Qwen2.5-0.5B anchor, LFM2.5-230M, Granite-4.0-350M, Qwen3-0.6B.

- [x] Cache every hidden layer once per completed model and renderer version (six complete; Falcon-H1
  CPU-infeasible without optional kernels).
- [ ] Compare last-token, masked/full-token mean, and max pooling (last and masked mean complete; max
  remains).
- [ ] Compare individual layers, adjacent windows of width 2–3, and sparse combinations (individual
  and adjacent complete; sparse remains).
- [x] Compare absolute layer index with normalized depth.
- [ ] Use nested group-aware selection; final test stays unopened.
- [ ] Track CPU latency, peak RSS, artifact size, and energy proxy alongside discrimination (all but
  energy proxy complete).

### Phase 3 — expansion and architecture tests

- [ ] Promote only models meeting Phase 2 quality/latency gates.
- [x] Add Falcon-H1-0.5B, Qwen3.5-0.8B, and MiniCPM5-1B (Falcon attempted and rejected for ordinary
  CPU deployment feasibility).
- [x] Treat attention, convolution/Mamba, and DeltaNet/MoE blocks as distinct block types in the
  registry and result interpretation.
- [ ] Test whether a block-aware relative-depth rule improves cross-model stability.

EXP002 audit note: the public silver rows do not expose reliable scenario-group IDs, so this first
screen used row-level selection and row bootstrap. The separately downloaded `approve-or-deny` test
was opened only after a timestamped selection freeze, but it was opened before a larger grouped or
nested-CV development stage existed. It is now permanently consumed. EXP002 is useful engineering
evidence and an adverse calibration result; it does not satisfy the stronger group-aware publication
lock below.

### Phase 4 — SEND-state gold set

- [ ] Write versioned state-machine rules before generating cases.
- [ ] Generate exact contrast families by varying one state fact at a time.
- [ ] Human-review ambiguous language and multilingual cases.
- [ ] Hold out complete scenario templates and action families.
- [ ] Target at least 600 gold contexts before product threshold claims.

### Phase 5 — product and Hugging Face release

- [x] Select and freeze one research-release backbone and head before opening the one-shot held-out
  test: MiniCPM5 hidden outputs 13–14 with masked-mean pooling.
- [ ] Produce calibrated `ALLOW/CONFIRM/BLOCK` receipt.
- [ ] Add runtime integration behind an explicit opt-in policy.
- [ ] Expose the same versioned receipt contract at the CCO control-plane attach point; CCO consumes
  the decision and evidence but does not silently widen authorization scope.
- [x] Publish the EXP002 research artifact, benchmark summary, model card, limitations, scorer, and
  reproducibility hashes at
  [`mcpware/intentprobe-minicpm5-1b-authorization`](https://huggingface.co/mcpware/intentprobe-minicpm5-1b-authorization),
  current public release commit `6db35552fd4836b5d201170073e244b5f61b00fe`.
- [x] Release only the MiniCPM probe trained from the pinned Apache-2.0 AUTO source; do not release
  the broader mixed-source dataset before row-level provenance and license QA.

## Statistical and leakage locks

- Unit of independence is scenario/contrast group, never rendered row.
- Contrast siblings, paraphrases, and shared source templates stay in the same split.
- Tool/framework/category/language holdouts are reported separately.
- Layer, pooling, head, regularization, and threshold are selected inside training/validation only.
- Multiple layer/window comparisons use a predeclared family and Benjamini-Hochberg sensitivity table;
  the primary result is nested-CV performance, not the maximum observed score.
- Report clustered bootstrap 95% confidence intervals by scenario group.
- Compare paired predictions with a paired bootstrap; McNemar is a sensitivity check for a fixed binary
  threshold, not a replacement for calibration analysis.
- Every claim states sample size and evaluation unit.

## Go / stop criteria

Proceed toward a standalone detector only if the activation system materially improves the declared
false-approve operating point or calibration over both TF-IDF and dedicated encoder baselines.

Proceed toward an attached runtime sensor if it adds signal while reusing activations already computed
by the agent backbone, even when a standalone encoder remains faster end-to-end.

Stop or pivot if:

- gains disappear under scenario-group or action-family holdout;
- the selected layer/window is unstable across folds and seeds;
- performance is explained by renderer or template artifacts;
- the sensor cannot beat action-only or TF-IDF controls;
- false-approve risk remains unacceptable at useful recall;
- license/provenance prevents a defensible released weight.

## Open decisions

- Whether product policy should expose three classes directly or represent `CONFIRM` as abstention.
- Whether separate heads share one cached representation or each chooses its own depth/window.
- Whether multilingual coverage is a release requirement or a post-v1 expansion.
- The exact predeclared practical margin over `auto-0.4b/1b` needed for a standalone claim.

## Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-10-16 | Benchmark before product UI | A strong existing encoder baseline can validate or kill the activation-only product claim cheaply. |
| 2026-10-16 | Separate sensor from execution authority | Preview/payload/batch binding is deterministic state, not a probability. |
| 2026-10-16 | Start with four canonical backbones | Broad architecture coverage without paying the full matrix cost before a baseline exists. |
| 2026-10-16 | Download all nine pinned model snapshots | Nicole requested a complete local CPU screen; 13.29 GB were cached with exact revisions and per-file hashes. |
| 2026-10-16 | Freeze MiniCPM5 outputs 13–14 for the EXP002 held-out test | It had the highest validation AUPRC under the predeclared selection rule. |
| 2026-10-16 | Stop the standalone activation-only send-gate path at EXP002 | Held-out AUPRC gain was inconclusive, low-FPR recall did not beat TF-IDF, and validation thresholds exceeded their held-out false-approve budgets. Continue only as an attached sensor or with a new calibration design and a new untouched test. |
| 2026-10-17 | Publish EXP002 as a Hugging Face research preview | The 28 KB artifact, exact scorer, benchmark receipt, adverse calibration result, and licenses were ready for reproducible public use. The release explicitly emits no execution decision. |
| 2026-10-17 | Publish under `ithiria137` pending an HF organization | The authenticated account has no HF organization membership and the `mcpware` organization endpoint returns 404; the repository can be transferred after that namespace exists. |
