# Authorization Send Detector — Sources

Last searched: 2026-10-16

## Source coverage ledger

| Status | Source | Actual coverage | Result |
|---|---|---:|---|
| ✅ | Hugging Face datasets | 49 search queries; 43 card/schema/file checks across the authorization, privacy, policy, prompt-injection, and agent-safety families | direct and auxiliary datasets identified |
| ✅ | Hugging Face models | 16 size-query searches; 18 publisher scans; 35 direct model/config/card checks | canonical small-backbone matrix and dedicated authorization baselines identified |
| ✅ | Kaggle | 15 live dataset queries | mostly reuploads/weak provenance; no better direct gold authorization source found |
| ✅ | arXiv / primary paper pages | 28 query strings; primary pages opened for the named recent benchmarks | AuthGateBench and AuthorizationPairs found four days after submission |
| ✅ | GitHub | evaluator and competitor repositories inspected | AuthorizationPairs evaluator is public; activation authorization remains distinct from deterministic execution state |
| ✅ | Local executed QA | 7 datasets examined with code | label counts, duplicates, contradictions, and split risks recorded in the research notes and regenerated here where source snapshots permit |
| ✅ | Pinned acquisition | 14/14 registry entries | every registered artifact downloaded; per-file SHA-256 manifest written |
| ⬜ | Public three-class `ALLOW/CONFIRM/BLOCK` SEND-state gold set | broad direct and synonym searches | no complete direct-fit public source found |
| ❌ | `datasets.load_dataset` route | one compatibility failure (`KeyError: maxdepth`) | bypassed with pinned repository files/Parquet; not treated as absence |
| ⏭️ | Paid research APIs | not required for source acquisition | skipped |
| ⏭️ | Naver | no Korean-market question | 0 requests |

Coverage honesty: public data is strongest for binary approve/deny and synthetic policy cases; exact
payload-bound SEND approval remains the main dataset gap and must be built as project gold data.

## Direct sources

### Authorization datasets and baselines

- [auto-1b-data](https://huggingface.co/datasets/ProCreations/auto-1b-data) — large silver training set.
- [approve-or-deny](https://huggingface.co/datasets/ProCreations/approve-or-deny) — held-out silver benchmark.
- [auto-0.4b](https://huggingface.co/ProCreations/auto-0.4b) and
  [auto-1b](https://huggingface.co/ProCreations/auto-1b) — dedicated encoder baselines.
- [AuthorizationPairs dataset](https://huggingface.co/datasets/Egbertjing/authorizationpairs),
  [paper](https://arxiv.org/abs/2610.16838), and
  [evaluator](https://github.com/HKUST-KnowComp/authorizationpairs-eval).
- [AuthGateBench](https://arxiv.org/abs/2610.16426) — 1,400 contrastive authorization-gating instances;
  public dataset/code was not located at the 2026-10-16 cutoff.

### Auxiliary agent-policy and safety evaluations

- [SafeClawBench](https://huggingface.co/datasets/sairights/safeclawbench) — evaluation-only adversarial cases.
- [COMPASS](https://huggingface.co/datasets/AIM-Intelligence/COMPASS-Policy-Alignment-Testbed-Dataset) — policy compliance testbed.
- [AgentCIBench](https://huggingface.co/datasets/UKPLab/agentcibench) — agent privacy/confidentiality cases.
- [NVIDIA Agentic Indirect Prompt Injection](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1) — successful attack traces; positive pool only.
- [FORTIS](https://huggingface.co/datasets/ShawnLi02/FORTIS_Agent_Skill_Safety) — least-privilege tool/skill selection; dataset license unclear at cutoff.
- [Agent-SafetyBench](https://huggingface.co/datasets/thu-coai/Agent-SafetyBench) — auxiliary risk evaluation.

## Source-use rules

- Snapshot commit hashes live in `SOURCE_REGISTRY.json`.
- Raw snapshots are immutable and gitignored.
- Retrieval hashes and statuses live in `data/manifests/`.
- A downloaded file is not automatically licensed for release training.
- Evaluation-only and quarantined sources never enter training.
- Cards and papers are evidence about provenance, not substitutes for row-level QA.

## Executed acquisition and QA

- Retrieval manifest: `data/manifests/retrieval_manifest.json` — all 14 source entries report
  `GOT_DATA`.
- QA manifest: `data/manifests/qa_summary.json`.
- `auto-1b-data`: 725,000 rows across its train and validation files; 724,999 unique normalized texts,
  one duplicate group, and zero contradictory normalized-text groups.
- `approve-or-deny`: 3,000/3,000 unique normalized texts and zero contradictions.
- AuthorizationPairs: 1,794/1,794 unique normalized inputs, zero contradictions, two rows marked
  ambiguous, and zero `group_id` values crossing development/test files.
