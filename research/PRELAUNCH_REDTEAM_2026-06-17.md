# Pre-Launch Red-Team — IntentProbe (2026-06-17)

**Verdict: DO NOT launch the current README/llms.txt/REDDIT_LAUNCH as written.**
The science is real and the `research/` docs are honest; the **public copy strips the hedges
the research docs explicitly demand** and will be debunked within an hour on HN. The fix is not
new science — it is making the public copy obey `REPRODUCE.md`'s own "Do NOT headline" rules.

How this was checked: 5 independent Opus sub-audits (line-by-line over the whole repo) + the
author's own-hand verification (read code, ran the venv) + an independent Codex (gpt-5.5, xhigh)
re-run. **All four converged.** Plus new numbers generated this session (DeBERTa run locally,
shipped-Qwen head-to-head, PI home-turf experiment).

---

## A. Confirmed defects in the public copy (4-way verified, file:line)

| # | Defect | Evidence | Fix |
|---|---|---|---|
| 1 | **Headline 96.5%/96.6%/100% are GPT-2 numbers; product ships Qwen2.5-0.5B** | `benchmark-probe-vs-deberta-samesplit.py:3,23,58,118` (GPT-2 L3); artifact `metadata.json` model_id=`Qwen/Qwen2.5-0.5B`. README:22/90/106 present them as the product's. | Re-run on Qwen, label every number's model, or drop them. |
| 2 | **No honest held-out matched-vocab number for shipped Qwen** | `matched` set is IN Qwen's training pool (`datasets.py:86`, artifact metadata). Eval on it = test-on-train. | Use held-out 41.5% (novel families) as the lead number. |
| 3 | **CV leakage inflates the matched number** | `samesplit.py:129` `StratifiedKFold(shuffle=True)`, no grouping; n=86 pool = 78 unique, **8 exact dup rows** leak across folds. Reproduced (mine + Codex): F1 0.966 → **0.90–0.92** dedup/grouped. | GroupKFold + dedup; report the lower number. |
| 4 | **"n=86 matched-vocab pairs from MCPTox" false 3 ways** | Actually **43 pairs**, only **8 truly matched**, **0 from MCPTox**; my prefix check: 8/8 "matched" pairs are different topics (length-matched, not vocab-matched). | "8 matched-vocabulary minimal pairs." |
| 5 | **"reads activations, not text" misleading for BLOCK tier** | `core.py:285-310`: a `block` is structurally impossible without a static-regex hit; activation-only highs (even 0.99) downgrade to `warn`. Verified live + Codex. | "activation = primary warn signal; block requires static corroboration." |
| 6 | **Shipped artifact metric is a TRAIN metric** | `train_probe_artifact.py:194-195` fit+predict on same 989 rows → metadata precision 1.0 / F1 0.996. | Never cite; only held-out numbers. |
| 7 | **"Snyk's shipped scanner catches 0%" — not what we tested** | We benchmarked `protectai/deberta-v3-base-prompt-injection-v2` (verified it IS what `invariantlabs-ai/invariant` analyzer uses, `prompt_injections.py:7`). But `snyk/agent-scan` (rebranded Invariant mcp-scan) does its detection via a **closed cloud API** (`cli.py:169` → `api.snyk.io/hidden/mcp-scan/...`); the DeBERTa is **not in agent-scan's code**, and its tool-poisoning rule (E001) is partly **keyword-based** (`issue-codes.md:31`). | Say "the industry-standard PI text classifier (protectai DeBERTa, used by Invariant/Snyk/Lakera/promptfoo)". Don't say "Snyk's scanner" unless we run the actual product (needs SNYK_TOKEN). |
| 8 | **"clean FPR 0.000" cherry-picks** | True only at block tier; **block recall is 3.7%** (`ledger:319`; Codex: hard-block caught 3/43 small-pool, 18/485 MCPTox). | State block precision and recall together. |
| 9 | **"First and Only" unhedged** | Author's own `COMPETITIVE_LANDSCAPE.md:33` hedges to "first product-shaped one we know of." | Restore the hedge. |
| 10 | **SAE "14→84" is a GPT-2 outlier, not shipped** | Shipped artifact `sae:null`, raw activations. Other models +8/+12; partly rides lexical-framing features. `REPRODUCE.md:18` "Do NOT headline 14→84." | Keep out of public copy (already is). |

**Corrected agent false-alarms (verified by hand):** the repo IS pushed (`mcpware/intentprobe`
+ `IntentProbe` both exist on GitHub) and **PyPI has `intentprobe` up to 0.1.4** — install works.
The "launch will 404" claim was based on a stale `CLAUDE.local.md` note ("未 push"). NOTE: this
means the indefensible version is **already shipped on PyPI** — consider a 0.1.5 with corrected README.

---

## B. Real numbers generated this session (all run on this machine, by hand)

### B1. Same 8 matched-vocabulary pairs — head-to-head (shipped Qwen artifact)
```
protectai DeBERTa (PI text classifier):  POISON 0/8 caught     CLEAN 0/8 FP
intentprobe shipped Qwen scanner:         POISON 8/8 warn       CLEAN 0/8 FP
   poison activation 0.89–1.00 (all warn) ; clean activation 0.00–0.08 (all allow)
```
Caveats: these 8 are **in-distribution** (in Qwen's training pool) → an illustration of clean
separation, NOT a generalization claim. Decision is **warn**, not block. n=8, no CI.

### B2. Prompt injection — DeBERTa's OWN home turf (deepset/prompt-injections, held-out test n=116)
```
Qwen activation probe (trained on deepset-train):  acc 0.966  F1 0.967  95% CI [0.930, 0.993]
protectai DeBERTa (zero-shot), F1 by threshold:    0.9→0.500  0.5→0.537  0.25→0.554  0.1→0.554
```
DeBERTa is **not** crippled by threshold — zero-shot recall caps at ~38% at any threshold.

**FAIR-C (resolved):** fine-tuning DeBERTa-v3 (184M) on the SAME deepset-train split → **F1 0.957**
on the held-out test. So vs the probe's **0.967**, it is a **tie** (within the probe's CI). The
zero-shot "0.50" was a **zero-shot strawman — the same error as the "Snyk 0%" claim**; the honest,
same-data number is 0.957. Honest claim: a **22 KB linear probe on a frozen 0.5B model matches a
fully fine-tuned 184M PI transformer** on the incumbent's home task, AND the same method extends to
tool poisoning DeBERTa is blind to. Lead with efficiency + generality, never "we beat DeBERTa."

### B3. Held-out novel families — FRESH n=150/150 three-way, with 95% CI (run this session)
Sampled 150 poison + 150 clean from `routeguard_external_v0` (real third-party: BIPIA /
MaliciousAgentSkillsBench / Skill-Inject), stood up a 300-tool MCP server, ran all three scanners.
Ordering sanity-checked (clean mean activation 0.142 < poison 0.497).
```
                          POISON recall (95% CI)    CLEAN FP
intentprobe warn+block      74.7% [67-81]            18.7%
intentprobe block-only       6.7% [3-11]              0.0%
DeBERTa PI (thr 0.9)         8.7% [5-14]              2.0%
Snyk Agent Scan (cloud)      unscorable per-tool at n=300 — see note
```
**Key methodology finding:** Snyk's CLI `--json` groups findings by issue-type and only enumerates
*example* affected tools in free-text evidence (W017/W019 name ~5 each), so a clean per-tool recall at
n=300 is not obtainable from this output (lower bound ≥20%). Snyk's per-tool number is reliable only at
small n (the 8 matched pairs → 5/8). The clean large-n head-to-head is intentprobe vs DeBERTa.

**Honest read:** IntentProbe has no single number — it's a tunable tripwire. The old repo "41.5%" is
one operating point between the block tier (~7%, 0% FP) and the warn tier (~75%, 19% FP). Report the
curve + CI, not a cherry-picked point. This n=150/CI result supersedes the n=8 illustration in B1.

---

## C. The honest, defensible positioning

> Activation probing is a **new detector class** that catches a failure mode text/keyword scanners
> are structurally blind to (same-words / opposite-intent tool poisoning), and the **same 22 KB probe
> on a 0.5B local model handles prompt injection too** (F1 0.967 on deepset, beating the zero-shot
> incumbent classifier). It is a **research-preview pre-install tripwire**, not a complete or more
> accurate scanner than Snyk. Generality — one tiny local probe, multiple attack classes — is the moat.

NOT defensible: "beats Snyk", "96.5%", "100%", "the first and only", "zero false positives".

---

## D. Pre-launch fix checklist
1. Rewrite README/llms.txt/npm/REDDIT_LAUNCH to obey `REPRODUCE.md` (model labels, lead with 41.5%,
   block precision+recall together, hedge "first", DeBERTa = "zero-shot/OOD industry-standard PI classifier").
2. Fix CV: GroupKFold + dedup; report the ~0.90 number.
3. Fix the fail-open hook: single-shot `runtime scan` crashes (exit 1, no gate) on malformed JSON —
   wrap try/except, fail-closed, exit 2 (`hook.py:394,544`).
4. Restore missing `research/audit_logs/` so "reproduce it yourself" is not an empty promise.
5. (Optional, for a real "vs Snyk" number) run `snyk-agent-scan` on the 8 pairs — needs SNYK_TOKEN.
6. Ship 0.1.5 with the corrected README (current PyPI 0.1.4 has the indefensible copy).

## E. Verification provenance
- 5× Opus line-by-line sub-audits (datasets, public-claims, runtime, benchmark, robustness).
- Author-hand verification: read `core.py`/`samesplit.py`, ran the venv, reproduced leakage + DeBERTa 0/8.
- Independent Codex gpt-5.5 (xhigh) re-run → `research/CODEX_REDTEAM_2026-06-17.md`. Converged.
- New experiments this session: B1 (head-to-head), B2 (PI home turf + self-audit), B3 (traced).

NOT re-run this session: n=2,900 RouteGuard, n=249 MCPTox (traced from code + saved reports only).
