# IntentProbe architecture v0.1.3 — publication and discoverability

**Author:** [Wan Sheng Leung](https://github.com/ithiria894)<br>
**Publisher:** MCPware, a product and research brand of GEASS LABS INC.<br>
**Revision date:** 19 October 2026<br>
**Architecture first published:** 18 October 2026

This editorial revision explains the CPU sidecar architecture, links readers to usable probes and
reproduction code, and prepares a GitHub Pages entry point. It keeps the public
architecture title and presents the system through its implementation and measured evidence.

## Changes

- Public author fields and citation metadata use `Leung, Wan Sheng`, matching the May 2026 Zenodo
  record (`10.5281/zenodo.19990741`) for consistent author metadata.
- The article and README explain the extra local pass beside a hosted agent and the reuse across
  the sidecar's own heads.
- The shared interface is named the Activation Backplane, with a first-use explanation across
  the public release documents.
- The approved architecture illustration appears in the article, README and six-page PDF.
  Its unchanged pixels, prompt and provenance are recorded with the preprint sources.
- Related work adds Goodfire × Rakuten, SingProbe, Lion Warden, and Constitutional Classifiers++.
  Citation corrections distinguish attention-based MindGuard from hidden-state linear probes.
- The website adds a homepage, title and description metadata, canonical page configuration,
  sitemap support, and machine-readable project links.
- Hugging Face card sources link back to the architecture and add relevant CPU/hidden-state tags.
- `.zenodo.json` supplies explicit author metadata for a future software archive. It does not
  assign a DOI or repurpose the earlier tool-poisoning paper's DOI.
- Two separate preprints are prepared: the six-page architecture note, DOI
  `10.5281/zenodo.23467859`, and the fifteen-page empirical paper, DOI
  `10.5281/zenodo.23471117`. They cross-reference each other. Both Zenodo records remain
  unsubmitted until the author approves the complete exact publication preview with `SEND`.
- The architecture note now reports the existing ToxicChat validation-selection regret and
  16-head pooling/window census. Their scopes remain explicit; no new experiment was run.
- The empirical paper adds a reference list and operational-calibration/readout context.
  Detailed claims about the inaccessible OpenReview access-control PDF are narrowed to the
  title/venue/topic verified on author pages, rather than treated as newly verified methods.
- Empirical PDF evidence links are built with a full immutable source commit, not a mutable
  review branch. The source ref and reproducible command are recorded in `paper/BUILD.md`.
- The empirical abstract now leads with measured serving cost and is shortened from 363 to 220
  whitespace-delimited words. Broader project-task scope is retained in Section 3.2.
- The empirical paper keeps the architecture illustration and adds the unchanged matched-panel
  latency SVG plus a reproducible 16-head selected-window census. Captions distinguish the
  ten-input comparison from the 80-input scaling means and selection counts from performance.

No benchmark dataset, weights, predictions, frozen experiment result JSON, confidence interval, or measured value is
changed. The current 16-head runtime remains research code. General registry, hot-loading, and
multi-head receipt emission remain specified work.
The derived paper ledger changes only to list the planned cached-judge comparison arm; all
of its numerical content remains identical to the previous candidate.

[`SHA256SUMS_V0_1_3.txt`](SHA256SUMS_V0_1_3.txt) records this candidate's public surfaces and the
unchanged experiment results. Older checksum ledgers refer to their original release snapshots;
they are not regenerated to match the changing main branch.

## Candidate checks

The full test suite passes (104 tests; 5 skipped), including three figure-preservation checks,
the two-paper citation/durable-link/rendering checks, and four added abstract/readout-figure checks.
The CFF validates against the official 1.2.0 schema. YAML front matter and site configuration parse,
and the new GitHub blob links resolve to local source files. Benchmark artifacts and derived
result JSON have no diff. The derived-ledger numerical-content comparison also passes.

The public article's Humanizer observable-style audit reports no stock phrases or Unicode
anomalies. Its long-sentence leads include metadata, HTML image markup, link groups and enumerated lists; they were
inspected in context. Frozen article SHA-256:
`0ba94ba656c802cc11efd10f0be3c73086df232265d029cb9a768d3a2a160b3a`.
This is a style check, not an AI-authorship detector result.
Both papers underwent a meaning-locked composition/green review followed by frozen-candidate
factual/style review. The empirical manuscript has the same structural long-sentence leads
in metadata, tables and lists, with no stock-phrase or Unicode-anomaly finding. External grammar
services and AI-detector panels were not run; no independent human approval is implied.

Jekyll is not installed locally. No packages were installed for this revision. A real Pages build,
live metadata inspection and HTTP-200 verification remain deployment checks after publication.
