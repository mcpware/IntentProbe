# Prior-art source ledger — 2026-10-18

## Publication follow-up — 2026-10-19

This follow-up verifies sources needed for the architecture article. It is not a new exhaustive
novelty search. Earlier retrieval records below retain their original dates and access limits.

| Source | Retrieval and use |
|---|---|
| [Goodfire × Rakuten](https://www.goodfire.com/research/rakuten-sae-probes-for-pii-detection) | Official article, methods and discussion: separate sidecar, PII probes, and the stated motivation for adding more probes. Added as prior work rather than an unmatched numerical benchmark |
| [SingProbe v2](https://arxiv.org/abs/2608.30703v2) | Primary title, abstract, and version history: intrinsic serving-model monitoring with intent, safety, and hallucination signals. No collapsed latency-overhead number is used in this update |
| [Lion Warden](https://huggingface.co/patronus-studio/lion-warden-ai-security-classifier) | Official model card: shared encoder, seven security heads, joint training, and CPU inference. Added as a future matched-CPU baseline |
| [Constitutional Classifiers++](https://arxiv.org/abs/2601.04603) | Primary abstract: cascade and linear probes combined with external classifiers. The earlier repo survey already listed the title; this update adds substantive discussion |
| [MindGuard](https://arxiv.org/abs/2508.20412) | Primary abstract identifies attention-based Decision Dependence Graph inspection. Corrected its classification as a linear hidden-state probe |
| [You Only Judge Once](https://arxiv.org/abs/2604.10966) | Primary abstract identifies a multimodal reward model and up-to-N-fold speedup, rather than a general near-constant-latency guarantee |
| [TaskTracker current record](https://arxiv.org/abs/2406.00799) | Current v6 title is *Get my drift? Catching LLM Task Drift with Activation Deltas*. Retained the existing title; the different v3–v5 wording does not make it a current-title error |
| [False Sense of Security](https://arxiv.org/abs/2509.03888) | Confirmed the ID is adverse probing-generalization work, not evidence for Gemini production deployment. Removed that ID from the Gemini bullet |
| [Representation-as-a-Judge](https://arxiv.org/abs/2601.22588) | The manuscript now links to the stable arXiv record rather than relying solely on an OpenReview forum URL |

The OpenReview-only access-control reference still needs a fetchable primary mirror before a
submission citation is frozen. This editorial update does not mark that reference verified.

This ledger records the actual retrieval coverage for the companion
[`PRIOR_ART_REVIEW_2026-10-18.md`](PRIOR_ART_REVIEW_2026-10-18.md). A successful search is not
treated as proof that a database is exhaustive. Counts from broad scholarly APIs are discovery
counts, not a deduplicated evidence set.

## Source coverage

| Status | Source | Actual coverage | Result and use |
| --- | --- | --- | --- |
| ✅ GOT DATA | Local IntentProbe repository and git history | Research index, result matrices, original paper metadata, package history, first five git commits | Used to separate completed evidence from planned claims and to reconstruct the project's verified public chronology. |
| ✅ GOT DATA | OpenAlex API | 6 query families; up to 100 works requested per query; reported broad-match totals of 4,503, 112,512, 63,693, 30,030, 8,725, and 14,845 | Discovery only. Relevant records were reopened at arXiv, ACL, OpenReview, publisher, or repository sources. |
| ❌ FAILED | Semantic Scholar Graph API | 6 query families; 6 HTTP 429 responses | No Semantic Scholar result is used as evidence. The same concepts were continued through OpenAlex, arXiv, Crossref, ACL, OpenReview, and primary pages. |
| ✅ GOT DATA / ❌ FAILED | Crossref REST API | 4 query families; first 2 succeeded with broad-match totals 3,746,732 and 869,249; last 2 returned HTTP 429 | Successful responses were discovery only. Failed queries are not counted as searched results. |
| ✅ GOT DATA | arXiv API and full PDFs | 4 systematic query families returned 35, 80, 2, and 3 records; 15 high-relevance PDFs were downloaded and converted to text for complete inspection | Primary source for preprints and full methods/results. |
| ❌ FAILED | DBLP API | 5 requests across 4 query families; endpoint returned empty/non-JSON rate-limit responses and then an explicit HTTP 429 with `Retry-After: 60` | No DBLP result is used as evidence. Venue records were checked at ACL and OpenReview where available. |
| ✅ GOT DATA / ❌ FAILED | OpenReview | Public indexed pages and full PDFs were available; 4 API requests returned HTTP 403 | Relevant public records were inspected directly. The API failure remains visible and no completeness claim is made for the API. |
| ✅ GOT DATA | ACL Anthology | 4 direct query-page requests plus official record/PDF inspection for relevant papers, including LatentGate and *False Sense of Security* | Official metadata, DOI, venue, full paper, and adverse results used. |
| ✅ GOT DATA | Google Patents | 12 query families spanning activation/intermediate state, prompt injection, single-pass multi-head classification, intent/policy, frozen heads, pooling, layer selection, routing, and memory; 18 high-relevance documents opened; claims, priority, status, and family sections inspected for the core set | Main international full-text and family discovery source. Google explicitly labels legal status and priority as non-legal assumptions; those fields were cross-checked where possible. |
| ✅ GOT DATA | USPTO Patent Public Search | 7 exact-number lookups; all 7 found: US20260127380A1, US20260127216A1, US20250200333A1, US12107885B1, US12137118B1, US12549598B2, US12572777B2 | Official US publication/grant, title, inventor, publication date, and page-count confirmation. Claims were read in the publication text and cross-checked against Google Patents rendering. |
| ✅ GOT DATA | Canadian Patent Database (CIPO) | 4 exact searches on 2026-10-18: WO2025226897, WO2026101639, “Prompt injection classifier using intermediate results,” and “Intent Based Observability and Control” | All four returned “No patents found.” This means no matching public Canadian record was found on the search date; it does not rule out a later national-phase entry. |
| ❌ FAILED | WIPO PATENTSCOPE | 2 indexed web queries produced no result; direct browser access returned HTTP 403; direct result URL redirected to the search home page without a retrievable result | WO family metadata is taken from Google Patents and must be rechecked in PATENTSCOPE or by counsel before filing or launch. |
| ❌ FAILED | EPO Register / Espacenet / OPS | EPO Register and Espacenet returned HTTP 403; two OPS bibliographic/family requests returned HTTP 403 with a fair-use rejection | EP4621615A1 claims and status are taken from the publication text on Google Patents. The live EPO register must be rechecked separately before an FTO decision. |
| ✅ GOT DATA | Hugging Face Hub | Patronus Lion Warden model card and repository metadata; IntentProbe public artifact context | Used for the closest released multi-head product comparison, not as proof of patent status. |
| ✅ GOT DATA | GitHub API and repositories | IntentProbe, `tool-probe`, Patronus security repository, and related public implementation pages | Used for verifiable repository creation dates and implementation shape. GitHub creation dates do not prove invention dates. |
| ✅ GOT DATA | Zenodo API and SSRN public page | Zenodo record 19990741 metadata; SSRN abstract 6698259 search result and public page metadata | Used to distinguish “date written” from verified public posting/publication. |
| ✅ GOT DATA | Probe-versus-judge follow-up search | 36 additional query strings across arXiv, ACL Anthology, OpenReview-indexed papers, and named/adjacent concepts; eight primary paper pages or full papers inspected; access-control authorship and forthcoming venue cross-checked on an author publication page | Produced the separate literature-inheritance map and reduced the planned direct-judge experiment from the full matrix to a six-cell sentinel panel. |
| ❌ FAILED | OpenReview direct PDF retrieval for *Embarrassingly Simple Access Control* | 1 direct browser open and 1 direct HTTP download returned a verification page/HTTP 403 | Indexed full-paper text exposed the method and result tables, but the final proceedings PDF must be reopened before a submission citation is frozen. |
| ⬜ EMPTY | CIPO exact family/title searches | 4 successful official searches, 0 records | Kept separate from failed access and from a legal conclusion of no Canadian rights. |
| ⏭️ SKIPPED | Paid legal databases (Derwent, LexisNexis TotalPatent One, Orbit, PatSnap full platform) | Not available in this workspace | A professional novelty/FTO search should add prosecution histories, assignment changes, continuations, unpublished applications, and non-English family normalization. |

## Canonical academic queries

The wording varied by API syntax, but the discovery concepts were:

1. `activation probe language model hidden state safety classifier`
2. `prompt injection hidden state classifier activation`
3. `single pass multi task classification transformer heads`
4. `LLM routing prefill activations latent routing`
5. `frozen language model task specific classification head`
6. `pooling layer selection normalization hidden state probe`
7. `runtime monitor residual stream prompt injection`
8. `same forward pass multiple classifiers language model`
9. `production activation probes guardrail`
10. `activation probe out of distribution surface cues`

Synonyms used across searches included `linear probe`, `representation probe`, `residual stream`,
`intermediate result`, `latent monitor`, `prefill representation`, `behavior head`, `intent head`,
`shared encoder`, `multi-head classifier`, `one forward pass`, `single inference pass`, `sidecar`,
`guardrail`, `gating`, `orchestration`, and `routing`.

## Canonical patent queries

1. `("single inference pass" OR "single forward pass") ("task-specific heads" OR "multiple classification heads") language model`
2. `"hidden states" "multiple classifiers" large language model`
3. `activation probe authorization intent classifier LLM`
4. `activation analysis memory write gate language model`
5. `("masked mean" OR "mean pooling") "hidden state" classifier transformer layer`
6. `"layer selection" "linear probe" language model hidden state`
7. `"one forward pass" "multiple probes" language model`
8. `"frozen" language model "classification heads" hidden states`
9. `prompt injection classifier intermediate layer residual stream`
10. `small language model multiheaded behavior classifier intent policy`
11. `hidden state probe grounding error multiple layers pooling`
12. `quantized proxy activation analysis policy gate`

## Primary documents retained in the review

### Patents and applications

- [US20260127380A1 — WitnessAI intent observability and control](https://patents.google.com/patent/US20260127380A1/en)
- [US20260127216A1 — WitnessAI dynamic guardrail](https://patents.google.com/patent/US20260127216A1/en)
- [US20260278411A1 — WitnessAI multi-prompt organizational guardrail](https://patents.justia.com/patent/20260278411)
- [US12107885B1 — HiddenLayer prompt-injection classifier](https://patents.google.com/patent/US12107885B1/en)
- [US12137118B1 — HiddenLayer residual-stream prompt-injection classifier](https://patents.google.com/patent/US12137118B1/en)
- [US12549598B2 — HiddenLayer multimodal activation analysis](https://patents.google.com/patent/US12549598B2/en)
- [US20260143004A1 — HiddenLayer multimodal activation analysis continuation](https://patents.google.com/patent/US20260143004A1/en)
- [US12572777B2 — HiddenLayer policy-based activation gate](https://patents.google.com/patent/US12572777B2/en)
- [US20250200333A1 — Microsoft hidden-state grounding-error probes](https://patents.google.com/patent/US20250200333A1/en)
- [CN114416989B — multi-task hidden-state text classification](https://patents.google.com/patent/CN114416989B/en)
- [EP4621615A1 — Bitfast LLM security sidecar](https://patents.google.com/patent/EP4621615A1/en)
- [US11915471B2 — Salesforce shared transformer/task heads](https://patents.google.com/patent/US11915471B2/en)
- [US12437058B1 — Amazon LLM threat mitigation](https://patents.google.com/patent/US12437058B1/en)
- [US12292915B1 — Amazon attention-analysis security](https://patents.google.com/patent/US12292915B1/en)
- [US20260134293A1 — Articul8 pooled hidden states/probabilistic heads](https://patents.google.com/patent/US20260134293A1/en)
- [US20250124296A1 — elastic transformer serving/task heads](https://patents.google.com/patent/US20250124296A1/en)

### Academic and implementation prior art

- [Alain & Bengio, 2016 — linear probes across intermediate layers](https://arxiv.org/abs/1610.01644)
- [All Birds with One Stone, 2022 — multi-task text classification in one forward pass](https://arxiv.org/abs/2205.10744)
- [Representation Engineering, 2023](https://arxiv.org/abs/2310.01405)
- [TaskTracker, 2024](https://arxiv.org/abs/2406.00799)
- [Task-specific prompts and SAEs for activation monitoring, 2025](https://arxiv.org/abs/2504.20271)
- [Detecting High-Stakes Interactions with Activation Probes, 2025](https://arxiv.org/abs/2506.10805)
- [PIShield, 2025](https://arxiv.org/abs/2510.14005)
- [Production-Ready Probes for Gemini, 2026](https://arxiv.org/abs/2601.11516)
- [A BERTology View of LLM Orchestrations, 2026](https://arxiv.org/abs/2601.13288)
- [When Benchmarks Lie, 2026](https://arxiv.org/abs/2602.14161)
- [LLM Router: Prefill is All You Need, 2026](https://arxiv.org/abs/2603.20895)
- [RouteGuard, 2026](https://arxiv.org/abs/2604.22888)
- [Multi-Head Latent Control, 2026](https://arxiv.org/abs/2607.14277)
- [PoolBench, 2026](https://arxiv.org/abs/2608.05162)
- [Prompted to Discriminate, 2026](https://arxiv.org/abs/2610.02413)
- [LatentGate, ACL Industry 2026](https://aclanthology.org/2026.acl-industry.153/)
- [False Sense of Security, Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.1300/)
- [tool-probe implementation](https://github.com/oleksandr-shyshchuk/tool-probe)
- [Patronus Lion Warden model card](https://huggingface.co/patronus-studio/lion-warden-ai-security-classifier)

## Retrieval boundary

This is a high-coverage research screen, not a patent-office search opinion. The unresolved database
failures, unpublished applications, priority-document support, prosecution history, claim amendments,
and jurisdiction-specific claim construction remain outside the retrieved record. Every FTO conclusion
in the companion review is therefore expressed as a risk tier and a next claim-chart action, not as a
guarantee of clearance.
