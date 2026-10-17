# Memory Write Gate — Sources

Last searched: 2026-10-17

## Source coverage ledger

| Status | Source | Actual coverage | Result |
|---|---|---:|---|
| ✅ | Hugging Face | 20 new human-memory query families; 1,116 hits and 1,037 unique repositories, in addition to the prior pass | current LUFY and MemUse releases verified |
| ✅ | GitHub | 12 systematic repository queries, 8 schema/code queries, and complete LUFY history inspection | historical LUFY annotations, REALTALK, and method competitors found |
| ✅ | OpenAlex/arXiv | 13 OpenAlex query families with 1,300 hits/1,201 unique works; primary papers opened | LUFY, REALTALK, MemUse, SAMD, AgentMemGate, and MemRouter reviewed |
| ✅ | Kaggle | 9 query families; 71 hits and 56 unique datasets | no better human direct-label source found |
| ✅ | Local executed QA | `memory-write-gate-20k` full rows examined | 20,000 unique texts, 200 seed families, no seed crossing the provided splits |
| ✅ | LUFY historical QA | 612 XLSX workbooks from one official historical commit | 6,470 aligned human-user turns with three labels each |
| ✅ | Pinned acquisition | 6 accessible entries after adding current LUFY | dataset/code/paper snapshots and per-file SHA-256 recorded |
| ✅ | Human-written direct importance labels | LUFY current release plus historical official commit | a usable benchmark candidate was found; historical-label license scope still needs confirmation |
| ⬜ | SAMD downloadable artifacts | exact title, dataset-name, and repository searches | promising paper, but no public data/code found |
| ❌ | `Cossale/memory-traces` | one pinned Hugging Face acquisition attempt | gated repository returned HTTP 401; no agreement text inspected and no access accepted |
| ⏭️ | Reddit, Hacker News, Product Hunt, news, patents | not probative of downloadable label provenance | skipped |
| ⏭️ | Naver | no Korean-market question | 0 requests |

Coverage honesty: LUFY fixes the absence of human-written future-utility labels, but not explicit
storage-consent, privacy, deletion, or preference-change labels. The historical files remain
quarantined until their license scope is confirmed.

## Direct sources

- [memory-write-gate-20k](https://huggingface.co/datasets/chris0809/memory-write-gate-20k) — direct `SAVE/SKIP` bootstrap.
- [LUFY current dataset](https://huggingface.co/datasets/RuiSumida/LUFY),
  [paper](https://arxiv.org/abs/2409.12524), and
  [historical annotation commit](https://github.com/ryuichi-sumida/LUFY/tree/e0695216d52a38ef39ec2f91708d324877e328bd/Dataset)
  — human-written conversations with three binary future-importance votes per user turn. Current text
  is CC BY 4.0; historical annotation scope needs written confirmation.
- [REALTALK](https://github.com/danny911kr/REALTALK) and its
  [paper](https://arxiv.org/abs/2502.13270) — human-human longitudinal conversations with human QA
  evidence and life-event annotations; use as positive-outcome validation, not a source of assumed
  negatives.
- [MemUse](https://huggingface.co/datasets/RuiSumida/memuse) and its
  [paper](https://arxiv.org/abs/2608.24189) — real longitudinal human-chatbot conversations and
  human-verified memory-use moments; CC BY-NC 4.0, so external non-commercial evaluation only.
- [SP-Mem dataset](https://huggingface.co/datasets/wwj95/privacy-aware-memory-benchmark) and
  [code](https://github.com/Jensassss/SP-Mem), plus its
  [primary paper](https://arxiv.org/abs/2608.16551) — privacy-aware retrieval authorization.
- [MemoryCtrl](https://huggingface.co/datasets/ethz-spylab/MemoryCtrl) — no-store/forget/no-use evaluation; license unresolved at cutoff.
- [memory-traces](https://huggingface.co/datasets/Cossale/memory-traces) — real-conversation substrate with model-generated labels; privacy and upstream provenance review required.

## Source-use rules

- Raw snapshots are immutable and pinned by commit.
- LUFY splits are grouped by user. Majority vote is the primary label; unanimous and 2–1 cases are
  reported separately.
- Historical LUFY annotations and unmatched historical-only text stay quarantined until the author
  confirms the applicable license.
- Future conversational utility is not treated as user consent to persist data.
- Synthetic row count is never reported as independent scenario count.
- SP-Mem retrieval labels do not become write labels.
- MemoryCtrl remains quarantined while license terms are unclear.
- Real conversations do not enter release training without a separate privacy/provenance decision.

## Executed acquisition and QA

- Retrieval manifest: `data/manifests/retrieval_manifest.json` — six entries report `GOT_DATA`; the
  gated `memory-traces` entry reports `FAILED`.
- QA manifest: `data/manifests/qa_summary.json`.
- `memory-write-gate-20k`: 20,000 unique normalized texts, balanced 10,000 `SAVE` / 10,000 `SKIP`,
  zero contradictory text groups, 200 seed families, and zero seed families crossing splits.
- SP-Mem: 1,000 history files, 250 in each of education, finance, medical, and mental domains.
- MemoryCtrl: 10 source conversations, 60 whole-recall questions, and 362 slot-recall questions;
  license remains unresolved, so it stays quarantined.
- LUFY historical annotations: 612 workbooks, 6,470 aligned human-user turns, 332 majority-positive,
  6,138 majority-negative, 5,070 unanimous, and 1,400 with a 2–1 split. Full audit:
  [`HUMAN_DATASET_AUDIT_2026-10-17.md`](HUMAN_DATASET_AUDIT_2026-10-17.md).
