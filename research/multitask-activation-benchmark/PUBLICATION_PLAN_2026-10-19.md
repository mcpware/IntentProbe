# IntentProbe publication plan — 19 October 2026

## Goal

Publish a useful, reproducible account of the IntentProbe architecture. Make it easy for a reader
researching activation probes, local guardrails, or alternatives to LLM-as-a-judge to find the
project, try a probe, reproduce the evidence, and cite a fixed version. IntentProbe is the research
and product umbrella under MCPware. Public author fields use the name `Leung, Wan Sheng`, matching
the May 2026 Zenodo record so citation indexes resolve both to one author.

## Current publication candidate

- Architecture article: `docs/ONE_LOCAL_MODEL_PASS_MANY_TASK_SPECIFIC_PROBES.md`.
- Website entry point: `docs/index.md`; GitHub Pages already targets `main:/docs`.
- Citation metadata: `CITATION.cff` and `.zenodo.json`.
- Discoverability: natural search terms in opening text, site metadata, root/site LLM context,
  and the three Hugging Face card sources; all link back to one project.
- Evidence: the frozen 16-head CPU experiment and same-backbone sentinel, with existing contracts.
- Source reconciliation: the dated follow-up in `PRIOR_ART_SOURCE_LEDGER_2026-10-18.md`.

## Publication sequence

1. Keep the architecture candidate public on the review branch while preparing the separate
   architecture preprint. Its reserved DOI is `10.5281/zenodo.23467859`; its author is
   `Leung, Wan Sheng`, displayed as Wan Sheng Leung. Align the saved Zenodo draft with the approved
   six-page PDF, Markdown source and BibTeX, then show the complete final metadata and file preview.
2. After the exact preview and Nicole's standalone uppercase `SEND`, publish that new preprint
   first and verify the public DOI, author, files and checksums. The two older tool-poisoning records
   already publish the legal name; restore their stale handle-only edit drafts without republishing
   their unchanged articles. The May paper's DOI remains a reference to that earlier work.
3. Deploy the reviewed GitHub/Pages and model-card revisions. Check the website returns 200, the
   article and images load, and canonical, Open Graph, description, author and sitemap output are
   present. Check the new DOI and legal-name BibTeX across the article, README, CFF and model cards.
4. Prepare short, separate introductions for relevant communities and curated lists. Keep each
   draft concrete: the deployment problem, code/model link, and one measured result. Review the
   actual wording before posting. Track referrals, model downloads and concrete usage reports;
   downloads are not unique users.
5. Continue the benchmark without holding the initial explanatory article for every future test.
   Publish each subsequent result with its own source, protocol and version.

## Highest-value next experiments

- Same model, one prompted call returning all labels; record invalid outputs and per-task quality.
- Separate judge calls with a common cacheable prefix and caching enabled. Report the actual
  renderer/cache configuration rather than assuming an automatic speedup.
- Same-CPU compact-classifier fleet and an overlapping-task shared encoder such as Lion Warden.
  Measure the K crossover, quality, p50/p95 latency and resident memory on matched inputs.
- Source-held-out evaluation and matched benign negatives where sources permit them.
- CPU head-training and activation-extraction costs, using hardware logs before making a claim
  about the complete training/search workflow.

Freeze the analysis plan before running a new comparison. Keep the measured systems panel, task
quality results, implementation status and future designs identifiable in every release.

## Editorial checks

Meaning lock: explain the reusable architecture and engineering choices in plain English; let
readers inspect and use the work; use the approved legal-name byline; preserve all measured results. The
article keeps the model-as-sensor explanation and introduces deployment before terminology.

Green review: clarified the hosted-agent sidecar cost, moved model/reproduction links near the
opening, and shortened the introduction. Red review: reconcile claims with frozen artifacts,
run the local Humanizer observable-style audit, inspect links/metadata, and run the existing
publication contract tests. External AI-detector scores are not part of this revision's claim.

## Primary implementation references

- [GitHub Pages and Jekyll configuration](https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/about-github-pages-and-jekyll).
- [Zenodo record editing](https://help.zenodo.org/docs/deposit/manage-records/).
- [Zenodo creator metadata](https://help.zenodo.org/docs/deposit/describe-records/creators/).
- [Zenodo API edit/publish actions](https://developers.zenodo.org/).
