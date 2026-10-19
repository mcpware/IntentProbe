# Multi-task paper analysis plan and assumption ledger

Status: reconciled after the six-cell sentinel and `ONE-FORWARD-MULTIHEAD-EXP001` completed.

## Frozen questions

1. Does the best simple hidden-state readout vary by task and backbone?
2. On two selected small backbones and three sentinel tasks, how do frozen activation heads compare
   with constrained same-backbone output judgments and matched TF-IDF?
3. For 16 independently trained heads on one pinned Qwen2.5-0.5B backbone, how does CPU latency and
   process memory change at K=`1,2,4,8,16`?
4. Do head scores visibly shift when the same 16 readouts are applied across five real-text sources?

## Confirmatory and descriptive boundaries

| Analysis | Status | Unit / uncertainty | Allowed conclusion |
| --- | --- | --- | --- |
| Per-task activation versus TF-IDF | Planned within each source experiment | 2,000 paired row bootstraps unless stronger group IDs exist | Task/backbone-specific difference only |
| Authorization sentinel | Planned | 2,000 bootstraps over 26 construction families | Group-aware comparison on the 156 human-labelled rows |
| Toxicity and sentiment sentinel | Planned | 2,000 paired row bootstraps | Same-row method comparison in the frozen 512-row panels |
| Sentinel multiplicity | Planned | Benjamini--Hochberg over 12 activation-versus-choice-likelihood AUPRC tests | Six rejections at q<=0.05; no family-wide claim from unadjusted p-values |
| Cross-backbone test ranking | Exploratory | Descriptive only | No post-hoc “best model” significance claim |
| ToxicChat variance shares | Exploratory | Descriptive two-task decomposition | Evidence of interaction in these cells, not a population variance estimate |
| K scaling | Planned | 2,000 paired row bootstraps over 80 fixed prompts | Latency ratios for this implementation and CPU environment |
| Peak RSS | Planned process observation | One process run per K | Environment-specific memory observation; no confidence interval |
| Cross-domain scores | Diagnostic | 64 fixed rows per source | Score shift only; no OOD error without labels |

## Assumptions that remain visible

- The shared K suite uses a fixed cumulative head order rather than random head combinations.
- Every K path requests all hidden states, matching an attachable readout bus rather than an
  optimized single-head special case.
- Runs occur in ascending K order without randomized interleaving or CPU-frequency pinning.
- The repeated-forward comparator is deliberately a lower bound: it omits task prompts, the LM head,
  choice likelihood, parsing, and generation.
- The direct-judge sentinel uses LFM2.5-230M and Qwen3.5-0.8B; the K-scaling suite uses
  Qwen2.5-0.5B. They are complementary experiments, not one matched end-to-end arm.
- Native heldout metrics have different class prevalences and sample sizes and are never averaged
  into a universal safety score.
- The UCC files omit author/article identifiers, so their intervals cannot support unseen-author or
  unseen-thread generalization.
- Live full-precision hidden states and the original float16 activation cache are not bit-identical;
  the 64-row replay quantifies the difference.

## Follow-up studies for the empirical submission

The initial architecture preprint uses the completed evidence and states its implementation and
comparison boundaries. The studies below strengthen a later empirical submission; they do not
hold the explanatory architecture preprint for results that have not yet been measured.

- Add a measured one-call multi-label judge arm.
- Add a measured fleet of compact task-specific classifiers on the same hardware and prompt panel.
- Measure cold start, throughput, and deterministic repeatability.
- Add at least one labelled source-held-out or semantic-counterfactual panel for several heads.
- Reopen and verify every cited primary paper immediately before submission.
- Freeze author identity, version, hashes, licences, and a citable release bundle.
