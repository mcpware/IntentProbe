# Human-data topic expansion — 2026-10-18

## Decision

The paper should count **dataset families**, not inflate its breadth by treating every label from one
corpus as an independent dataset. The next confirmatory expansion is:

1. **Fine-grained emotion — GoEmotions panel.** Four targets were locked before screening:
   admiration, anger, curiosity, and sadness. They span different affective families and each has at
   least 150 positive examples in the untouched test split. The frozen Qwen activation cache can be
   reused because text and row order are identical to the completed neutral-label experiment.
2. **Pragmatic/social behaviour — Unhealthy Comments Corpus.** Use condescension, dismissiveness,
   sarcasm, and unfair generalisation as the primary panel. The source contains natural news comments,
   aggregated human labels, confidence scores, and individual judgments. It is meaningfully subtler
   than a second generic toxicity dataset.
3. **Perceived politeness — Stanford Politeness Corpus.** Keep as the next acquisition target after a
   file-level licence and source-identifier audit. The paper reports 4,353 annotated Wikipedia
   requests and 6,604 annotated Stack Exchange requests, with five ratings per request.

Civil Comments remains the preferred large external-domain robustness set. HateXplain is useful for
hate/offensive distinctions and rationale analysis, but it is closer to the existing ToxicChat
toxicity question and therefore comes after the three families above. Dreaddit stress and TweetEval
irony/stance are useful later sensitivity tasks, but their smaller gold set or platform/licence
boundaries make them weaker immediate choices.

## Why these topics strengthen the paper

| Family | Semantic question | Natural human text | Human gold | Main leakage/control issue | Decision |
| --- | --- | --- | --- | --- | --- |
| GoEmotions | Which emotion is expressed? | Reddit comments | Multiple raters; filtered split requires agreement | Filtered files omit author/thread IDs | QWEN PANEL COMPLETE |
| UCC | Is the comment sarcastic, condescending, dismissive, or unfairly generalising? | News-site comments | Aggregated and individual judgments | No released author/article grouping; repair exact overlap | QWEN ACTIVATIONS COMPLETE |
| Stanford Politeness | How polite is a real request perceived to be? | Wikipedia and Stack Exchange requests | Five normalized ratings per request | Corpus/platform terms and source identifiers need file audit | NEXT ACQUISITION |
| Civil Comments | Does a public comment contain threats, insults, identity attack, or toxicity? | News-site comments | Crowd-label fractions | Group by publication/article/time; never use parent text with random splits | EXTERNAL ROBUSTNESS |
| HateXplain | Hate, offensive, or normal; which tokens explain the label? | Twitter/Gab posts | Three annotators plus rationales | Platform redistribution and author/source grouping | LATER |
| Dreaddit | Does a post segment express stress? | Reddit posts | AMT labels on 3,553 segments | Small gold set; group by post/domain and audit release terms | LATER |
| TweetEval irony/stance | Is a tweet ironic, or what stance does it take? | Tweets | Original shared-task labels | Individual task and Twitter restrictions; user/thread grouping | LATER |

## Executed data checks

### GoEmotions four-emotion panel

- Frozen source rows: 53,905 after the earlier conflict removal and exact-text de-duplication.
- Official split sizes retained: 43,095 train / 5,401 validation / 5,409 test.
- Positive counts:
  - admiration: 4,106 / 487 / 504;
  - anger: 1,561 / 194 / 198;
  - curiosity: 2,183 / 246 / 283;
  - sadness: 1,318 / 143 / 156.
- Snapshot SHA-256: `1bddf8692dde12f25304a51dc34e076d97dd33860ca70f795d3b8a9d15770e78`.
- The four targets were chosen before any activation result was inspected.
- All four validation-selected readouts use masked mean. Admiration selects outputs 2--4
  (normalized depth 0.125), anger output 5 (0.208), curiosity output 3 (0.125), and sadness output
  16 (0.667). The within-family range is direct evidence against one fixed layer for every emotion.
- Activation versus TF-IDF test AUPRC is 0.6014 versus 0.6593 for admiration, 0.3232 versus
  0.4254 for anger, 0.4143 versus 0.4309 for curiosity, and 0.3373 versus 0.4823 for sadness.
  Anger is the only target whose paired intervals favour TF-IDF on both AUPRC and AUROC. Curiosity
  remains unresolved on both metrics. Admiration and sadness favour TF-IDF on AUPRC but not on
  AUROC under the preregistered paired intervals.

### Unhealthy Comments Corpus

- Official source revision: `55934d0079073095af8b69807d235e40c8aceff4`.
- Raw rows: 44,355 across the official train/validation/test files.
- One row with no comment text was excluded.
- Seven normalized exact-text groups crossed splits.
- Two exact-text groups had a label conflict; all four rows in those groups were removed.
- Test-first exact de-duplication leaves 44,323 rows: 35,472 train / 4,427 validation /
  4,424 test.
- Primary-panel test positives: condescending 269, dismissive 150, sarcastic 201, unfair
  generalisation 91. The last target will need wider uncertainty intervals because it is rarer.
- Snapshot SHA-256: `617083fa3662d6371b98cc37cd74ab4fa10c79cc71d6f99cbcd31388c1251d9f`.
- Licence recorded by the official repository: CC BY-NC-SA 4.0. Public redistribution or a public
  weight release remains a separate decision from this internal research run.
- Qwen2.5-0.5B extraction is complete for all 44,323 rows, 25 hidden-state outputs, and three pooling
  methods. The three float16 caches each contain shape `[44323, 25, 896]`; zero rows exceed the
  128-token cap. The run resumed safely from 28,448 rows after one terminal SIGTERM, and the final
  hashes are recorded in the extraction manifest.

## Paper breadth after this slice

The completed and admitted matrix now covers materially different kinds of signal:

- valence: SST-2 sentiment;
- harmfulness and adversarial intent: ToxicChat toxicity and jailbreak;
- message purpose: SMS spam;
- fine-grained operational intent: BANKING77;
- affect: GoEmotions neutral plus the four-emotion panel;
- pragmatics and social behaviour: UCC;
- later external/domain checks: politeness and Civil Comments.

This is enough breadth for a multi-task readout paper once the same frozen model panel and baseline
contract are completed. Adding more datasets after that point is valuable only if it resolves a
specific uncertainty, not merely to increase the task count.

## Source coverage ledger

| Status | Canonical source/query | Coverage | Material result |
| --- | --- | ---: | --- |
| USED | Google Research GoEmotions paper, repository, model-card material, and executed local matrix | Official paper/page/repository plus 4 completed task runs | 58,009 natural Reddit comments, 27 emotions plus neutral; filtered release uses at least two-rater agreement; four locked targets select masked mean at depths 0.125--0.667 |
| USED | Conversation AI Unhealthy Comments official repository and paper | README, three split files, individual-annotation availability, pinned revision | 44,355 natural comments; seven attributes plus healthy; human labels and confidence; CC BY-NC-SA 4.0 |
| USED | Stanford Politeness paper and original corpus pointer | Full 10-page paper | 4,353 Wikipedia and 6,604 Stack Exchange requests; five normalized human ratings per request |
| USED | TensorFlow Datasets Civil Comments documentation and original Figshare record | Current dataset documentation plus CC0 source record | Seven crowd-label fractions; publication IDs and timestamps available; parent text can leak under random splits |
| USED | HateXplain official repository and AAAI paper | Repository, licence marker, paper | 20,148 reported posts; three annotators; hate/offensive/normal, targets, and rationales |
| USED | Dreaddit paper | Full paper text | 187,444 collected posts, with 3,553 human-labelled segments from 3,000 posts |
| USED | TweetEval official repository and paper references | Seven tasks and licence notice | Natural tweets for emotion, irony, hate, offensive, sentiment, and stance; restrictions may apply per source task |
| USED | LIAR official dataset README and paper | Format, terms, paper | 12.8K PolitiFact statements; research-only and metadata leakage risks make it conditional |
| REJECTED | Intel Polite Guard as confirmatory training data | Dataset card | 100,000 synthetic rows dominate; only 200 real annotated rows |
| REJECTED | RealToxicityPrompts and model-scored derivatives | Prior audit plus official descriptions | Natural text, but the target is a model score rather than human gold |
| DEFERRED | SARC sarcasm | Dataset description | Large and natural with author self-labels, but UCC provides independent human judgments and cleaner immediate comparability |
| SKIPPED | Reddit/community popularity search | 0 requests for admission decision | Primary provenance, annotation, split, and licence evidence decide admission; popularity does not |
| SKIPPED | Paid research APIs | 0 requests | Official repositories, papers, and public dataset documentation were sufficient |

## Primary references

- GoEmotions: <https://research.google/pubs/goemotions-a-dataset-of-fine-grained-emotions/>
- GoEmotions repository: <https://github.com/google-research/google-research/tree/master/goemotions>
- Unhealthy Comments Corpus: <https://github.com/conversationai/unhealthy-conversations>
- UCC paper: <https://aclanthology.org/2020.alw-1.15/>
- Stanford Politeness paper: <https://nlp.stanford.edu/pubs/politeness.pdf>
- Civil Comments documentation: <https://www.tensorflow.org/datasets/catalog/civil_comments>
- Wikipedia Talk Corpus source: <https://figshare.com/articles/dataset/Wikipedia_Talk_Corpus/4264973>
- HateXplain: <https://github.com/hate-alert/HateXplain>
- Dreaddit: <https://arxiv.org/abs/1911.00133>
- TweetEval: <https://github.com/cardiffnlp/tweeteval>
- LIAR paper: <https://arxiv.org/abs/1705.00648>
