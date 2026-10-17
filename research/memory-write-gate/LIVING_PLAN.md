# Memory Write Gate — Living Plan

Last reconciled: 2026-10-17

## Outcome

Build a local sensor and policy layer that decides whether a conversational fact should be:

- `SAVE` — eligible for durable memory;
- `SKIP` — usable in the current interaction but not persisted;
- `FORGET` — remove a previously stored item when the user directs it;
- `RETRIEVE_ALLOWED` / `RETRIEVE_DENIED` — whether a stored item may be surfaced in the current context.

The first benchmark is binary `SAVE/SKIP`, because that is where direct labeled data exists. Product
semantics must eventually keep write, delete, and retrieval authorization separate.

## Non-goals

- Treating every personal fact as useful memory.
- Inferring consent to persist merely because information appeared in conversation.
- Using a synthetic paraphrase split as proof of real-user generalization.
- Collapsing "should save" and "may retrieve now" into one classifier.
- Publishing weights trained on unclear or incompatible sources.

## Research questions

1. Which small backbone/layer/pooling combination best separates `SAVE` from `SKIP` under seed-family holdout?
2. Does the authorization winner also win memory, or is there a task-by-model interaction?
3. Is normalized depth stable across the two boundary tasks?
4. Can a probe distinguish durable preference from temporary state, sensitive fact, quoted third-party
   data, and explicit no-store instruction without relying on keywords?
5. Does a user-specific preference layer materially improve over a universal memory policy?

## Data strategy

- `memory-write-gate-20k`: silver bootstrap only. Its 20,000 rows derive from 200 seed families; all
  evaluation and resampling must group by `seed_id`.
- LUFY: first-choice human-written benchmark. Its historical release contains three independent
  binary importance annotations per user turn using the criterion that the information would be used
  in future conversations. Use majority vote for the main label, unanimous rows as a high-agreement
  core, and 2–1 rows as a disagreement set. Keep historical annotations quarantined until the author
  confirms that the current CC BY 4.0 grant covers those deleted files.
- REALTALK: human-human longitudinal conversations with human-authored QA evidence and significant
  life events. Use referenced evidence turns as positive outcome anchors; unreferenced turns are
  unlabeled, not negative examples.
- MemUse: real four-month human-chatbot diary study with human-verified memory-use moments. Use only
  for non-commercial external evaluation because the dataset is CC BY-NC 4.0.
- SP-Mem: external privacy-aware retrieval evaluation, not a clean write-gate training label.
- MemoryCtrl: evaluation/quarantine until license terms are clear.
- memory-traces: real conversations with model labels; privacy/provenance review required before any
  release-training use.
- Project gap set: human-authored contexts covering sensitive data, temporary instructions, durable
  preferences, third-party information, explicit save/no-save, corrections, and forget requests.
  LUFY supplies future-utility labels, but it does not settle storage consent or privacy policy.

## Phases

### Phase 0 — source and design freeze (partially complete)

- [x] Create independent folder and living plan.
- [x] Pin direct and adjacent sources.
- [x] Define write/delete/retrieval as separate decisions.
- [x] Download and hash every accessible registered source; record `memory-traces` as gated/failed.
- [x] Execute schema, duplicate, contradiction, seed-family, and split QA on acquired datasets.
- [x] Find a real human-written, multi-annotator write-side benchmark: historical LUFY provides 6,470
  aligned user turns with three annotations each; 332 are majority-positive.
- [x] Pin current LUFY text/QA release and quarantine the historical annotation commit pending license
  confirmation.
- [x] Acquire the historical LUFY annotation commit for internal research after Nicole's explicit
  2026-10-17 direction to use it; raw redistribution remains blocked pending license confirmation.
- [ ] Resolve MemoryCtrl and memory-traces release-use questions.
- [ ] Obtain written confirmation that CC BY 4.0 covers LUFY's historical per-turn annotation files.

### Phase 1 — SAVE/SKIP baseline

- [x] Preserve official splits and verify seed isolation. The 200 seeds contain exactly 100 rows each;
  train/validation/test contain 142/32/26 disjoint seed families.
- [x] Run label-prior, explicit-keyword, request-length, and word-plus-character TF-IDF controls
  (MEM-EXP001). On the consumed 2,600-row/26-seed test, TF-IDF reaches 0.8241 AUPRC and 0.8890 AUROC.
- [x] Complete the Qwen2.5-0.5B all-output activation sweep (MEM-EXP002): 216 frozen candidates across
  masked mean, last non-padding, masked max, and adjacent width-2/3 windows. Masked mean at output 11
  wins with development AUPRC 0.8897 and AUROC 0.9030.
- [x] Compare the development winner with the frozen TF-IDF baseline using paired seed-family
  resampling. Activation gains are +0.0531 AUPRC (95% CI −0.0199 to 0.1469) and +0.0419 AUROC
  (95% CI −0.0194 to 0.0998); promising, but not resolved with 32 validation seed families.
- [x] Report seed-clustered overall CIs. TF-IDF AUPRC 95% CI is 0.6424–0.9618 and AUROC CI is
  0.8067–0.9510.
- [x] Report exploratory category, language, and difficulty slices on development data without
  selecting slice-specific models or thresholds. The largest threshold-0.5 false-save rates are
  `general_advice` 88%, `one_off_document` 64%, and `secret` 52%; `project_constraint` SAVE recall is
  46%, making these the first human-gold targets.
- [x] Audit the validation-selected 1%-false-save operating point. It shifts to 2.69% false saves on
  test and retains 11.38% SAVE recall, so the silver score is not a standalone memory-write gate.

### Phase 2 — cross-model benchmark

- [ ] Reuse the same canonical backbone families and representation cache contract as authorization.
- [ ] Follow the shared confirmatory and exploratory readout protocol in
  [`../authorization-send-detector/paper/MULTITASK_BENCHMARK_PLAN.md`](../authorization-send-detector/paper/MULTITASK_BENCHMARK_PLAN.md)
  so memory results can enter the same task-by-backbone matrix.
- [ ] Compare model-specific optimum with normalized universal depth.
- [ ] Fit a task-by-model mixed-effects analysis only after the benchmark matrix is complete and the
  independence assumptions are defensible.

### Phase 3 — human gold and personalization

- [ ] Write annotation manual and consent/privacy handling process.
- [ ] Build a user-held-out LUFY benchmark with majority, unanimous-core, and disagreement views.
- [ ] Add human-authored consent/privacy gap cases with explicit author intent; do not relabel LUFY
  future utility as consent to persist.
- [ ] Separate universal safety rules from personal memory preferences.
- [ ] Test cold-start, per-user calibration, and preference changes over time.

### Phase 4 — productization

- [ ] Emit a receipt with decision, confidence, reason code, retention scope, and expiry.
- [ ] Offer the receipt and inspect/delete controls through the CCO control-plane attach point without
  merging write, delete, and retrieval authorization into one policy bit.
- [ ] Provide local inspect/delete/export controls.
- [ ] Publish only after source, privacy, and license review.

## Leakage and statistical locks

- Seed families and paraphrase families never cross splits.
- User IDs never cross train/test for user-level generalization claims.
- The test set cannot choose layer, pooling, threshold, category mapping, or renderer.
- Primary intervals bootstrap seed/user groups, not individual paraphrases.
- Compare against TF-IDF before interpreting activation gains.
- Every performance claim states rows and independent group count.

## Stop criteria

Pivot or stop if activation gains vanish under seed-family and human-written holdout; if the classifier
mainly reads explicit words such as "remember"; if false-save risk remains unacceptable; or if the
model cannot represent changing user preferences without retaining the sensitive input it was meant
to protect.

## Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-10-16 | Keep Memory Write Gate separate from authorization | Same benchmark mechanics, different policy semantics and privacy risks. |
| 2026-10-16 | Start after authorization EXP001/EXP002 | Reuse the honest baseline/cache framework without splitting the first implementation effort. |
| 2026-10-16 | Treat 20k rows as 200 conceptual seed families | Prevent synthetic paraphrase volume from masquerading as independent evidence. |
| 2026-10-17 | Permanently consume the MEM-EXP001 test after a frozen surface-baseline protocol | TF-IDF is predictive under seed holdout, but the low-false-save threshold shifts and recall remains poor. Future activation choices use train/validation only. |
| 2026-10-17 | Freeze MEM-EXP002 at 256 tokens before extraction | Tokenizer-only audit covers every rendered train/validation row; unlike the earlier authorization audit, no development contrast is silently removed by truncation. |
| 2026-10-17 | Select masked mean output 11 after the complete 216-candidate screen | It has the highest validation AUPRC (0.8897); neither adjacent windows nor the other pooling rules improve it. |
| 2026-10-17 | Treat the activation-over-TF-IDF result as promising, not confirmatory | The paired development gains are positive at the point estimate but both 95% seed-bootstrap intervals include zero after validation selection. |
| 2026-10-17 | Adopt LUFY as the first human-written benchmark candidate | The historical public release contains three direct importance votes per user turn under a future-use criterion, much closer to SAVE/SKIP than retrieval-only or synthetic sources. |
| 2026-10-17 | Keep historical LUFY annotations quarantined pending license confirmation | The current Hugging Face card is CC BY 4.0, but the deleted historical XLSX files have no explicit license file tying that grant to the old release. |
