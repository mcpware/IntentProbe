# Source ledger — real, clearly labelled task expansion

Retrieval date: 2026-10-17 (America/Vancouver). Local-result reconciliation: 2026-10-18.

| Status | Source/search surface | Actual coverage | Material result |
| --- | --- | ---: | --- |
| GOT DATA | Official/author dataset pages, papers, and repositories | 16 candidate families opened or verified | SST, ToxicChat, GoEmotions, BANKING77, MASSIVE, Civil Comments/Wikipedia toxicity, UCC, HateXplain, SMS Spam, Stanford Politeness, 20 Newsgroups, LIAR, LLMail-Inject, HackAPrompt, SaTML CTF, WildGuard |
| GOT DATA | Hugging Face dataset API | 9 canonical queries; 818 returned records (eight capped at 100, politeness returned 18) | Confirmed a broad candidate pool, then routed decisions back to first-party sources rather than trusting cards or search rank |
| GOT DATA | Stanford SST official download | 1 ZIP, 6,372,817 bytes, SHA-256 `3f5209483b46bbf129cacbbbe6ae02fe780407034f61cf6342b7833257c3f1db` | 11,855 root sentences; binary source has 6,920 train, 872 dev, and 1,821 test rows after dropping neutral labels; exact-dedup repair removes 11 repeated training rows and keeps test/dev intact |
| GOT DATA | LMSYS ToxicChat Hub API and pinned 0124 files | 1 repository metadata record, 11 tree entries, 2 CSV files / 10,165 rows | Revision `29df8e4d...`; CC BY-NC 4.0; observed 5,654 human rows versus 5,634 on the card; 99 train-test exact-overlap groups; leakage-clean human slice has 5,441 rows, 695 toxic and 182 jailbreak-positive |
| GOT DATA | ToxicChat human-only Qwen benchmark | 5,441 prompts, 3 pooling caches, 216 candidates per label, 2,790-row official test | Toxicity selects masked-mean outputs 18--20; jailbreaking selects masked-max outputs 5--7; paired bootstrap and matched TF-IDF comparison complete |
| GOT DATA | UCI SMS Spam Collection official archive and API metadata | 1 ZIP / 5,574 labelled messages; archive SHA-256 `1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3` | 290 normalized duplicate groups, 705 rows inside duplicate groups, zero label conflicts; 5,159 unique messages remain for a frozen split |
| GOT DATA | UCI SMS Spam Qwen benchmark | 5,159 unique messages, 3 pooling caches, 216 candidates, 774-row untouched test | Masked-max outputs 15--17 win; activation test AUPRC/AUROC 0.9804/0.9959 versus TF-IDF 0.9696/0.9895; paired intervals include zero |
| GOT DATA | Google Research GoEmotions pinned filtered files | 3 TSV files / 54,263 rows plus 28-label vocabulary at revision `3ef52111...` | 82 cross-split exact-text groups; 38 exact-text groups have conflicting neutral labels; conflict removal plus test-first exact de-duplication leaves 53,905 rows |
| GOT DATA | GoEmotions neutral-label Qwen benchmark | 53,905 clean comments, 3 pooling caches, 216 candidates, complete 5,401-row validation and 5,409-row test | Masked-mean outputs 15--16 win; activation test AUPRC/AUROC 0.6675/0.8175 versus TF-IDF 0.6578/0.8228; paired intervals include zero |
| GOT DATA | PolyAI BANKING77 official repository | 2 CSV files / 13,083 utterances plus 77-category vocabulary at revision `57ec275d...` | 12 exact-duplicate groups, zero label conflicts, seven cross-split groups; official-test-first de-duplication leaves 13,071 utterances |
| GOT DATA | BANKING77 Qwen multiclass benchmark | 13,071 clean queries, 3 pooling caches, 216 candidates, 3,079-row official test | Masked-mean output 1 wins; activation accuracy/macro-F1 0.6967/0.7815 versus TF-IDF 0.9022/0.9022; paired intervals favour TF-IDF |
| GOT DATA | Conversation AI Unhealthy Comments official repository and ACL paper | 44,355 natural comments, seven pragmatic attributes plus healthy, aggregated human labels; repository revision `55934d0079073095af8b69807d235e40c8aceff4` | Leakage repair removes one missing-text row and exact-text conflict/overlap groups, leaving 44,323 rows; source licence is CC BY-NC-SA 4.0 and author/article IDs are not released |
| GOT DATA | UCC Qwen pragmatics benchmark | 7 frozen binary tasks, 216 candidates per task, one shared label-blind activation cache, 4,424-row official test | Winners span masked mean/max and normalized depth 0.208--0.708; AUPRC 0.0836--0.1898 at prevalence 0.0206--0.0608; six AUPRC comparisons are unresolved and dismissiveness favours TF-IDF; all seven activation heads are less calibrated than TF-IDF |
| GOT DATA | IntentProbe six-cell direct-judge sentinel | 2 pinned local backbones × 3 tasks × activation, TF-IDF, zero-shot choice/generation, and rubric choice/generation; 2,000 paired bootstraps | Toxicity activation beats both judge prompts after BH correction; Qwen3.5 sentiment activation and choice-likelihood judging are tied; Qwen3.5 authorization activation loses to zero-shot choice likelihood by 0.2137 AUPRC with group-bootstrap 95% CI [-0.3973, -0.0335], BH q=0.048 |
| GOT DATA | IntentProbe one-forward multi-head systems experiment | 16 independently trained Qwen2.5 heads, K=1/2/4/8/16, 80 paired real heldout prompts, 320-row score-shift panel, 2,000 paired bootstraps | K=16 one-forward path is 1.069× K=1 latency (95% CI 1.056--1.078); on matched prompts, one-forward/16-head is 62.70 ms and 16 repeated base forwards are 956.71 ms, a ratio of 15.26 (95% CI 15.04--15.49); peak RSS delta +0.73 MiB; repeated-forward arm is explicitly a lower bound, not a full judge |
| EMPTY | Hugging Face multi-word API searches such as `sentiment human annotated` | 9 queries, 0 returned records | The endpoint treats multi-word search too narrowly; repeated with canonical single-term queries instead |
| SKIPPED | Reddit and other social discussion | 0 requests | Dataset admission depends on primary provenance, annotation, files, and licences rather than community popularity |
| SKIPPED | Paid research APIs | 0 requests | First-party sources and zero-key APIs were sufficient for this registry pass |

## Biggest blind spot

Several otherwise strong sources do not expose a natural independent unit such as user, author, or
conversation in the released files. Those tasks can reproduce their official test but cannot support
an unseen-user claim. Dataset-specific licence and file-revision audits are also still required before
raw text or trained weights from conditional sources can be redistributed.

## Exclusion principles applied

- RealToxicityPrompts is naturally occurring text but its toxicity target is a classifier score, so
  it is not human-gold confirmatory evidence.
- WildGuardMix, HarmBench, ToxiGen, and similar mixed/generated collections are not automatically
  admitted. A documented human/natural slice may be registered later, but synthetic rows stay tagged.
- LLMail-Inject, HackAPrompt, and SaTML CTF contain valuable real human attacks, but attack-heavy
  submissions alone cannot estimate false positives or balanced discrimination.
