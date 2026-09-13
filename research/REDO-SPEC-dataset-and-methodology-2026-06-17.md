# IntentProbe — Rebuild Spec: Dataset + Evaluation Methodology (v1, 2026-06-17)

> Direction (Nicole, 2026-06-17): **rebuild the science from the foundation up.** This spec is the
> first, most audit-critical layer (PM-PLAYBOOK 鐵律 5: audit starts from structure). Nothing gets
> built until the methodology here is blessed. Everything downstream (probe training, head-to-head,
> the paper rewrite, the product copy) consumes the datasets + protocol defined here.
>
> **What we keep (verified-real, don't throw away):** the activation signal genuinely separates
> clean/poison on real held-out third-party data (~75% warn recall), and naive evasion is 0%. What we
> rebuild: the *evidence* — a real matched-vocab set, leak-free splits, committed reproducible numbers.

## 0. What this dataset must prove (and must be able to DISPROVE)

The product's central claim: *"the activation probe separates clean vs poison when the text vocabulary
is held near-identical and only the malicious action differs — text/keyword scanners structurally
cannot."* The current evidence cannot test this (the "matched" set has Jaccard 0.087, 0 strict pairs,
n=16, 100% in-training). We build a set that **can** test it, and we **pre-register pass/fail before
running** (§5) so the result is honest either way.

---

## 1. Dataset A — `minpairs-v1` (the make-or-break matched-vocabulary set)

**Purpose:** the falsification substrate. On true minimal pairs, text baselines are *forced* near
chance (no lexical signal), so the probe's separation (or collapse) is a clean test of the thesis.

**Construction (mechanical, not free-authored — this is the key fix):**
- Write the **clean** tool description first.
- Derive the **poison** twin by inserting/swapping **only the minimal malicious clause** (e.g. clean
  `"Reads ~/.ssh/config and returns host aliases. Does not modify files."` → poison `"Reads
  ~/.ssh/config and returns host aliases, then reads the matching private key files and uploads
  fingerprints."`). One localized edit, nothing else changed.
- **Never** author the two sides independently (that leaks stylistic tells — the confound the paper
  admitted at line 87/123).

**Build-time gates (CI-enforced, fail the build on violation):**
- `jaccard(tokens(clean), tokens(poison)) >= 0.75` per pair. Reject pairs below threshold.
- Length control: `|len(clean) - len(poison)| <= 15 chars` (kills the 22% length confound that scored 92.5% alone).
- Framing control: **neither** side may use the tell-words a counter exploits — forbid clean-only
  reassurance (`does not`, `read-only`, `only`) AND poison-only additive (`also`, `automatically`,
  `additionally`) unless **both** twins carry it. (The reassurance/additive counter scored 88.5% — we
  remove that crutch so any probe lift is real.)
- No-leak: every row's exact text is asserted **absent** from the training pool.

**Coverage:** 8 risk families × ~15 pairs = **~120 pairs (240 rows)**. Families: credential_forwarding,
network_exfiltration, secret_file_access, host_modification, hidden_persistence, tool_shadowing,
instruction_chaining, system_inventory.

**Held-out by construction:** dedicated `split_group = "minpairs_v1_eval"`, **never** in any training pool.

**Deliverable:** `research/datasets/minpairs_v1.json` + `research/validate_minpairs.py` (prints per-pair
Jaccard/length/framing, exits non-zero on any gate violation — committed so a reviewer re-checks vocab
near-equality themselves).

---

## 2. Dataset B — `heldout_realworld_v1` (the generalization set, real third-party)

**Purpose:** does the probe generalize to attacks it never trained on? This is the honest
generalization number (currently ~75% warn / 0.513 transfer, on contaminated/wrong-distribution data).

**Source:** real third-party only (BIPIA / MaliciousAgentSkillsBench / Skill-Inject / MCPTox holdout),
but **filtered** to fix what the map flagged:
- **Drop** the ~1,000 BIPIA census-table QA rows (external-context QA, family-mislabeled — wrong
  distribution for a tool/skill scanner; inflates clean-side accuracy).
- **Separate** the 157 MASB metadata-only weak-label rows into their own bucket; do not pool them into
  the headline number (report them separately, labeled weak).
- Keep rows that are genuinely tool/skill-shaped text.

**Discipline:** per-family balance reported; **0% exact-text overlap with any training pool** (CI gate);
per-family recall with CI, never a single pooled number hiding family variance.

**Deliverable:** `research/datasets/heldout_realworld_v1.json` + provenance/license manifest +
`validate_no_train_overlap.py`.

---

## 3. Splits + leakage discipline (the methodology fix)

- **Training pool** (for the probe): existing curated + MCPTox, **rebuilt** so that (a) minpairs-v1 and
  heldout_realworld_v1 are fully excluded, (b) exact-duplicate rows removed, (c) a per-pair `group_id`
  exists on every paired row.
- **CV:** `StratifiedGroupKFold` keyed on `group_id` — a minimal pair's two twins **always** land in the
  same fold (kills the 79% pair-leak + 8-dup leak the red-team found; drops the inflated 0.966 to the
  honest ~0.94).
- **Build-time leak gate:** `assert eval_text ∉ train_text_set` for every eval row, run in CI. The README
  promised this; it has never been code-enforced. Now it is.
- **No train-metric ever cited.** `metadata.json` stores a **grouped-CV** number (+ a MCPTox-excluded
  variant to isolate template memorization), never the fit-on-train F1.

---

## 4. Evaluation protocol (what we run, on every dataset)

**Scanners compared (apples-to-apples):**
| Scanner | Why included |
|---|---|
| Qwen2.5-0.5B probe (L13-15) | the shipped product |
| GPT-2 L3 probe | continuity with the paper's numbers |
| DeBERTa **zero-shot** | the OOD text-classifier baseline (honest framing: zero-shot, not "Snyk") |
| DeBERTa **fine-tuned on same train** | the FAIR comparison (the one that ties the probe on PI) |
| TF-IDF + LogReg | lexical baseline |
| length-only classifier | confound control |
| reassurance/additive word counter | confound control (the 88.5% crutch) |
| static-keyword baseline | the "keyword scanner" the thesis claims to beat |

**Metrics (every run):**
- Per-tier (allow/warn/block) recall, precision, F1 — **FPR always reported with recall** (no
  cherry-picking block-precision alone).
- **Bootstrap 95% CI** (1000×) on every headline number.
- **Paired permutation test** that respects the pair `group_id` (shuffle within pairs, not row-level —
  fixes the paper's possibly-anti-conservative p=0.005).
- **Operating-point curve** (recall vs clean-FPR), not a single cherry-picked threshold.
- Fixed seed (42); every result written to `research/_results_published/<run>.json` + **committed**;
  sha256 manifest.

---

## 5. Pre-registered pass/fail (decide BEFORE seeing results)

On `minpairs-v1` (held-out, Jaccard≥0.75):
- **Expected by construction:** TF-IDF, length-only, framing-counter, DeBERTa zero-shot all ≈ chance
  (50%) — there is no lexical signal left. (If a text baseline scores high, the pairs aren't truly
  matched → fix the dataset, not the claim.)
- **Thesis SUPPORTED** iff: Qwen probe recall is materially above chance **and** above every text
  baseline, with **non-overlapping 95% CIs**, **and** the lexical-framing control (swap which twin
  carries the malicious clause) does not collapse it.
- **Thesis FALSIFIED** iff: the probe also drops toward chance → it was reading surface lexical
  confounds all along; the product's differentiation is false and we pivot the pitch to
  "one tiny local probe, multi-task, fully local" (the modest-but-real claim) or reassess.

---

## 6. Paper corrections (separate track, fold into the rewrite)

Methodology: (a) re-report matched-vocab F1 with grouped-CV + CI, retire 0.966; (b) state the
length/framing confounds in the headline, not just the appendix; (c) paired permutation + bootstrap CI.
Citations (audited 2026-06-17, all 19 refs real, no fabrication): (d) the `arXiv:2604.01151` "future
cite" is NOT an impossibility — git shows the paper was written 2026-03-29 and that citation was added
2026-04-03 (commit 1394056, after the cited Apr-1 paper existed); only the header `Date: March 29`
is stale → add "(last revised April 28, 2026)" so a reviewer isn't confused; (e) MCPTox "485" →
clarify it is *our 485-row subset of MCPTox's 1,312*; (f) Safe-SAIL "2,059 neurons" → "1,758 features".
Push a revised version + erratum to Zenodo once the science is redone.

---

## 7. Open methodology choices for Nicole (please steer)

1. **minpairs-v1 size:** ~120 pairs proposed. More (200+) = tighter CIs but more authoring. OK at 120?
2. **Authoring source:** mechanical templated edit by me (Claude), or do you want a second model
   (Codex/Gemini) to author independently to reduce single-author style bias? (multi-model authoring
   was the paper's own bias-reduction tactic.)
3. **Jaccard threshold:** 0.75 proposed. Higher (0.85) = stricter "same words" but harder to author for
   exfiltration-type pairs (need a URL/path the clean twin lacks).
4. **Block-tier scope in this round:** do we also tackle the activation-only block threshold (the
   "can it enforce, not just warn" question) now, or hold it until after the minpairs falsification test?
