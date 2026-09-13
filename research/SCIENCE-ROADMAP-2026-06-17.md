# IntentProbe — Strengthen-the-Science Roadmap (2026-06-17)

> Direction (Nicole, 2026-06-17): **strengthen the science before launch.** Launch held.
> Built from a 5-thread science-map workflow (datasets / experiments+CV / block-recall /
> generalization+adversarial / reproducibility+Snyk), every number tied to file:line or a
> this-session computation. Cross-checks the two red-teams (`PRELAUNCH_REDTEAM_2026-06-17.md`,
> `CODEX_REDTEAM_2026-06-17.md`). Confidence tags: [VERIFIED]=ran this session, [CHECKED]=read source.

## Honest state of the science

The differentiation thesis — *"a tiny local activation probe catches same-words/opposite-intent
tool poisoning that text/keyword scanners are structurally blind to, and also handles prompt
injection"* — currently rests on evidence that does not survive a careful read. Four load-bearing
facts, independently verified this session:

1. **The matched-vocabulary minimal-pair set does not contain matched-vocabulary minimal pairs.**
   `hard_v3_matched_*` = n=16 (8+8), **0 strict same-vocab pairs**, positional Jaccard mean **0.087**
   (max 0.167; single SSH near-pair at 0.425). Its own `research/datasets/README.md:16` calls it
   *"Length-matched subset (within 20 chars)"* — matched on character length, never vocabulary. It is
   **100% inside the Qwen training pool**, which also contaminates the "held-out" hard_v3 by **24%**
   (48/200 exact-text rows in training). [VERIFIED — my own recompute matches Codex + workflow agent exactly]
2. **The enforcement (block) tier is keyword-gated by design.** `choose_runtime_decision` (`core.py:285-307`)
   has only two block paths, both requiring a static-regex corroborator; everything else is "downgraded
   to warn". So a no-keyword activation-only attack can **never block**. Structural ceiling ~**14.7%**
   (185/1256 poison rows even eligible); **73.3%** emit zero static tokens; observed block recall **3.7%**.
   [CHECKED core.py + this-session computation; converges with CODEX finding 4]
3. **White-box GCG (the team's own "strongest threat", `REPRODUCE.md:184`) was never run**, and the
   existing script attacks GPT-2 L3 only, not the shipped Qwen probe. [VERIFIED — no committed output]
4. **No product-tier Qwen number is reproducible from a clean checkout.** `research/_results/` +
   `research/audit_logs/` are gitignored and absent; the 5 ledger-cited report paths all 404; the only
   committed metric (`metadata.json` F1 0.9956) is a **fit-on-train/predict-on-train** number that must
   never be cited. [VERIFIED — git ls-files = 0]

**What honestly holds:** the activation signal is real (warn-tier separation genuine, naive evasion 0%);
honest defensible numbers are **warn-tier recall ~75% @ ~19% clean FPR** (n=150, hand-run, uncommitted)
and **local→external transfer F1 0.513 / recall 0.415** — modest, not headline. **"Beats Snyk 0%" is
indefensible**: that 0% was protectai DeBERTa, not Snyk (Snyk = closed cloud API, no local DeBERTa); no
SNYK_TOKEN run was ever done.

## Key verified numbers (the honest ledger)

| Metric | Value | n | Model | Split | Source |
|---|---|---|---|---|---|
| matched-vocab strict same-vocab pairs | **0** | 16 | — | — | this-session recompute |
| matched-vocab positional Jaccard | mean **0.087** | 8 pairs | — | — | this-session recompute |
| matched-vocab F1 (headline) | 0.966 | 86 | GPT-2 L3 | **leaky CV** | benchmark-results md:42 (row-shuffle, 79% pair-leak + 8 dups) |
| matched-vocab F1 (grouped, honest) | ~0.94–0.95 | 86/78 | GPT-2 L3 | CV | CODEX:107,109 (not re-verified) |
| shipped artifact stored F1 | 0.9956 | 989 | Qwen2.5-0.5B | **TRAIN** | metadata.json (never citable) |
| local→external transfer | F1 0.513 / recall 0.415 | 2900 | Qwen | held-out | ROUTEGUARD md:37 (report MISSING on disk) |
| warn-tier | recall 0.800 / clean FPR 0.260 | 2900 | Qwen | held-out | THRESHOLD md:67 |
| block-tier | recall **0.037** / prec 1.0 / FPR 0.0 | 2900 | Qwen | held-out | THRESHOLD md:68 |
| block structural ceiling (perfect probe) | **0.147** (185/1256) | 1256 | policy gate | held-out | this-session compute |
| poison rows with zero static findings | **0.733** (921/1256) | 1256 | policy gate | held-out | this-session compute |
| white-box GCG vs probe | **UNVERIFIED — never run** | (20) | GPT-2 L3 | — | script exists, no output |
| SAE cross-family "14→84" | 14%→84.5% | MCPTox | GPT-2 L7 | held-out | REPRODUCE md:154 (GPT-2 OUTLIER; +8/+12 on capable models) |
| real Snyk head-to-head | **NONE produced** | 0 | Snyk cloud | — | no SNYK_TOKEN run ever |

## Prioritized roadmap (rank = thesis-impact × tractability)

| # | Item | Thread | Effort | Why it matters |
|---|---|---|---|---|
| **1** | **Build real matched-vocab minimal-pair set** (intentprobe-minpairs-v1, ~100 pairs, build-time Jaccard≥0.75 gate, fully held out) | datasets | medium | **IS the thesis.** No artifact today can support the headline; everything downstream needs this as eval substrate. |
| **2** | **Matched-vocab head-to-head** on the new set (Qwen+GPT-2, bootstrap CI, lexical-framing control) | datasets+exp | medium | The experiment that **earns or kills** the headline. ⚠️ thesis_risk (below). |
| 3 | Fix n=86 CV leakage (StratifiedGroupKFold by pair_id + dedup), retire 0.966→~0.94 | exp/CV | **cheap** | 5-minute debunk vector; most-cited number is leaky. |
| 4 | Calibrated **activation-only block threshold** + abstention band (gate: first plot clean activation tail) | block | **cheap** | Decides "blocker" vs "warn-only tripwire" — the product story. |
| 5 | Run white-box GCG (GPT-2 now) → port to shipped Qwen probe | adversarial | medium | "Strongest threat" untested; linear probe = easiest GCG target. |
| 6 | Decontaminate hard_v3 (drop 48 overlap rows) + add CI leakage gate | datasets | **cheap** | Closes the 24%-contamination objection; permanent guard. |
| 7 | Regenerate + COMMIT all Qwen product reports (whitelist published-reports dir + checksums) | repro | heavy | Makes headline numbers reproducible — "trust me" → "diff it yourself". |
| 8 | Conformal / risk-controlled block threshold (distribution-free clean-FPR guarantee) | block | medium | Answers "why 0.85?"; turns tuned number into a bound. (after #4) |
| 9 | Reduce warn-tier FP <19% @ recall ~75% (threshold sweep + hard-negative curriculum, grouped CV) | adversarial | medium | 19% FP → users disable it → catches nothing. |
| 10 | Real **Snyk head-to-head** at small honest n (SNYK_TOKEN run) | repro/compete | medium | "Beats Snyk 0%" is fabricated-by-confusion; get a real number or drop it. |
| 11 | Pin ONE reproducible venv (kill 5.9.0-vs-5.10.2 drift), add repro.sh | repro | **cheap** | "Reproduce it yourself" fails at step one today. (before #7) |
| 12 | Swap metadata.json train-metric → grouped-CV; reconcile mcptox 362-vs-342 | exp/repro | **cheap** | Removes committed 99.6% + "numbers don't match files". |

**Top recommendation:** start **rank 1** — everything is downstream of it. **Thesis risk:** rank 2 is the
experiment that, if it returns bad, kills the product story. On true minimal pairs (Jaccard≥0.75) text
scanners are *guaranteed* near chance (no lexical signal left); if the activation probe **also** collapses
to chance, the probe was reading the same surface confounds all along and the differentiation is false.
Every impressive current number (0.966, 14→84) was produced on data with vocabulary differences and/or
leakage, so **none of them actually test this.** Cheap-to-medium, fully under our control, run it early.

**Quick wins (no-brainer, needed under any path):** rank 3 (CV fix), rank 4 sub-step (plot activation tail),
rank 6 (decontaminate), rank 11 (pin venv), rank 5 first half (GPT-2 GCG), rank 12 (metric+count fix).

**Heavy bets:** rank 7 (commit all reproducible reports), rank 5 second half (GCG on Qwen probe),
broaden held-out families with per-family CIs.
