# Human Dataset Audit — Memory Write Gate

Date: 2026-10-17

## Decision

Use LUFY as the first human-written benchmark candidate. It is the closest public source found to a
direct `SAVE/SKIP` decision: human participants wrote the conversations, and at least three annotators
marked whether each user utterance contained information that would be used in future conversations.

Nicole approved use of the historical commit for internal research and benchmarking on 2026-10-17.
Do not yet redistribute the raw historical XLSX files or publish a commercial-license claim for a
model trained on them. The current Hugging Face
dataset card grants CC BY 4.0, but the per-turn annotation workbooks survive only in an older Git
commit and that commit contains no explicit license file. Obtain written confirmation from the author
that the current license covers the historical annotation files.

## LUFY evidence

Primary sources:

- Paper: <https://arxiv.org/abs/2409.12524>
- Current dataset: <https://huggingface.co/datasets/RuiSumida/LUFY>
- Official repository: <https://github.com/ryuichi-sumida/LUFY>
- Historical annotation commit:
  <https://github.com/ryuichi-sumida/LUFY/tree/e0695216d52a38ef39ec2f91708d324877e328bd/Dataset>

The historical commit contains 612 XLSX workbooks: three systems, 17 users, four sessions per user,
and three annotation files per user/session/system. Each workbook has a `Conversation` sheet with
`important?`, `speaker`, and `utterance` fields and a `Questions` sheet.

Executed local parsing retained human-user rows with binary labels, normalized Unicode and
whitespace for alignment, and required the three annotators' utterance sequences to match. The
result was:

| System | Workbooks | User turns | Majority `SKIP` | Majority `SAVE` | Unanimous | 2–1 split | Recomputed Fleiss kappa |
|---|---:|---:|---:|---:|---:|---:|---:|
| LUFY | 204 | 2,095 | 1,989 | 106 | 1,623 | 472 | 0.146 |
| MemoryBank | 204 | 2,217 | 2,106 | 111 | 1,746 | 471 | 0.152 |
| Naive RAG | 204 | 2,158 | 2,043 | 115 | 1,701 | 457 | 0.175 |
| **Total** | **612** | **6,470** | **6,138** | **332** | **5,070** | **1,400** | **0.158** |

The positive prevalence is 5.13%. Accuracy is therefore not an acceptable headline metric. Report
AUPRC, precision-recall operating points, calibration, and group-bootstrap uncertainty.

The paper reports Fleiss kappa 0.35, while the value recomputed from the historical public files is
0.158 overall. This discrepancy may reflect a different release/version or a different inclusion
rule. It must be disclosed rather than silently replacing either number.

The current Hugging Face release contains 4,254 turns and 497 participant-created QA/evidence pairs,
but not the per-turn annotations. Text comparison found 1,913 exact multiset matches between the
2,095 historical LUFY user turns and the 2,092 current LUFY user turns, before accounting for the
release's pseudonymization and text changes. Any derived benchmark must map by system, user, session,
turn order, and normalized text; unmatched historical-only text stays quarantined.

## Benchmark contract

1. Split by user, never by row or utterance. The same person's style and facts must not cross
   development and evaluation.
2. Use majority vote as the primary binary target.
3. Report the unanimous subset separately as the high-agreement core.
4. Preserve vote counts or soft targets for 2–1 items; do not pretend disagreement is annotation noise
   with a known correct answer.
5. Compare activation probes against label prior and TF-IDF controls under the same user-held-out
   folds.
6. Keep the natural class ratio for evaluation. Class weighting or balanced batches may be used only
   for training and must be documented.
7. Treat LUFY's label as future conversational utility, not permission to store. Consent, sensitive
   information, explicit no-store instructions, third-party data, and deletion require a separate
   human-authored gap set.

## Other human-data candidates

### REALTALK — external positive-outcome validation

- Repository: <https://github.com/danny911kr/REALTALK>
- Paper: <https://arxiv.org/abs/2502.13270>
- License: Apache-2.0 in the official repository.
- Scale: 10 real human-human conversation pairs, 219 sessions, 8,944 turns, 728 human-authored QA
  pairs, and 600 annotated significant life events.

QA evidence references provide strong positive anchors for facts demonstrably needed later. Turns
without evidence references are unlabeled, so they must not be converted into `SKIP` negatives.

### MemUse — non-commercial external evaluation

- Repository: <https://github.com/ryuichi-sumida/memuse>
- Dataset: <https://huggingface.co/datasets/RuiSumida/memuse>
- Paper: <https://arxiv.org/abs/2608.24189>
- License: CC BY-NC 4.0.
- Scale: 40 users, four months, 1,872 sessions, 21,575 turns, 72 human-verified memory-use moments with
  72 matched negatives, and 316 fact questions.

This is strong real-user outcome evidence, but the non-commercial license prevents it from becoming
the default release-training source.

### SAMD — ideal task definition, unavailable data

- Paper: <https://arxiv.org/abs/2602.11607>

SAMD reports direct memorable/non-memorable annotation with three or more annotators over large real
interaction sets. No official downloadable data or code repository was found in the completed search,
so it is a method reference rather than an executable benchmark.

### AgentMemGate and MemRouter — method competitors, not human gold

- AgentMemGate: <https://github.com/DistylAI/agentmemgate-artifacts>
- MemRouter: <https://github.com/SongW-SW/MemRouter>

Both directly model write-side selection and are important comparison systems. Their released labels
are generated or teacher-labeled rather than independent human gold, so they cannot solve the current
data-validity problem.

## Source coverage ledger

| Status | Channel | Actual coverage | Result |
|---|---|---:|---|
| GOT DATA | Hugging Face API | 20 query families; 1,116 hits; 1,037 unique repositories | LUFY and MemUse verified; numerous synthetic/retrieval-only sets excluded |
| GOT DATA | OpenAlex | 13 query families; 1,300 hits; 1,201 unique works | primary papers and adjacent human-memory studies found |
| GOT DATA | Kaggle | 9 query families; 71 hits; 56 unique datasets | no better human direct-label source found |
| GOT DATA | GitHub systematic search | 12 repository queries; 51 unique repositories, plus 8 schema/code queries covering 102 files in 39 repositories | LUFY history recovered; REALTALK and method competitors verified |
| GOT DATA | LUFY history audit | 1 full official repository history; 612 XLSX workbooks | 6,470 aligned, triple-annotated human-user turns |
| EMPTY | SAMD public artifacts | exact title, dataset-name, and author/project searches | paper found; downloadable dataset/code not found |
| FAILED | `Cossale/memory-traces` | one pinned acquisition attempt | gated HTTP 401; no agreement text read or accepted |
| SKIPPED | Reddit, Hacker News, Product Hunt, news, patents | 0 queries | opinions and launch coverage do not establish downloadable provenance or label quality |
| SKIPPED | Naver | 0 requests | no Korean-market or Korean-source requirement |

Biggest remaining blind spot: the license scope of LUFY's deleted historical annotations. The second
is task coverage: available human labels measure future utility or later use, not explicit permission,
privacy, or deletion intent.
