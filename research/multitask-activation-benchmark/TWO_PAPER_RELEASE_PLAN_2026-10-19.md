# Two-paper release preparation — 19 October 2026

## Owner decision and completion condition

The project owner approved two complementary papers on 19 October 2026: the empirical paper
reports the model/layer/pooling research; the architecture note explains the shared-pass design
and cites that evidence. The owner also approved adding the existing readout results to the
architecture note. This plan implements those decisions without commissioning new experiments.

Preparation is complete when both candidate PDFs and their sources are reproducible,
cross-referenced by their separate reserved DOIs, synchronized to the GitHub review
branch and unsubmitted Zenodo drafts, and presented to the owner with exact metadata and files.
Publication is a separate action, authorized only by standalone uppercase `SEND`
after the complete final preview. No main, Pages, HF, or outreach deployment is included.

## Locked content

- Author: `Leung, Wan Sheng`; display byline: `Wan Sheng Leung`.
- Publisher: MCPware, a product and research brand of GEASS LABS INC.
- Keep the approved architecture image byte-identical and retain its provenance.
- Preserve the frozen datasets, models, trained heads, measured results and analysis scope.
- Keep the plain model-as-sensor explanation and Activation Backplane first-use definition.
- Report the ToxicChat readout regret as a descriptive validation-selection comparison.
- Report 16-head pooling/window counts as a census of those selected heads, not a universal law.
- Distinguish the measured shared runtime from specified registry/hot-loading integrations.
- Keep unrelated employer material, private transcripts and personal email out of public files.

## Work queue

- [x] Read the latest Claude decisions and inspect the actual diff.
- [x] Independently check the 16-head readout census and rerun the existing test suite.
- [x] Reconcile every cited claim with its primary source or narrow it to what is verified.
- [x] Reserve the empirical paper DOI without publishing (`10.5281/zenodo.23471117`).
- [x] Cross-reference both papers and supply separate citation metadata.
- [x] Build and inspect both PDFs with the existing installed rendering dependencies.
- [x] Complete meaning-locked Humanizer composition and frozen-candidate review.
- [x] Reconcile source ledgers, release notes, plan, links and current-release hashes.
- [x] Push the reviewed candidate branch; leave main and deployments unchanged.
- [x] Upload exact candidate files to the two unsubmitted drafts and verify downloads.
- [x] Prepare the complete final publication preview for owner review; stop before Publish.

## Checks and boundaries

### Presentation follow-up

The owner approved a targeted empirical-paper presentation revision after the first two-paper
preview: keep the architecture illustration; add the existing matched-panel latency curve and a
new manifest-derived readout-window figure; shorten the abstract and lead with measured serving
cost. Do not change experiments or introduce a first-ever or universal-readout claim.

- [x] Revise the abstract and contribution order; move the broader task scope into Section 3.2.
- [x] Add the unchanged latency SVG with an explicit ten-input versus 80-input distinction.
- [x] Render and check the 16-head selected-window census with text pooling labels.
- [x] Complete code review, full tests, Humanizer review, PDF inspection and release checksums.
- [x] Synchronize the review branch and unsubmitted draft; replace the exact publication preview.

The revised empirical candidate has 15 pages and three figures; its abstract has 220
whitespace-delimited words. The suite passes 104 tests, with five skipped and three dependency
warnings. A fresh CodeRabbit review reported zero findings. Revised empirical source/PDF files
and description have been saved to the same unsubmitted draft and downloaded for byte checks;
the architecture candidate remains unchanged. The refreshed release ledger has 56 entries.

### Depth and two-panel follow-up

- [x] Bridge normalized midpoint depth and actual window coverage, including the 25-output count.
- [x] Build a separate two-panel latency SVG/PNG; keep companion architecture assets unchanged.
- [ ] Complete full tests, code/style review, PDF inspection and refreshed release hashes.
- [ ] Synchronize the reviewed PDF/source and replace the exact publication preview; stop before Publish.

### Initial two-paper checks

The baseline independent check passed 97 tests, with 5 skipped and 3 dependency warnings.
All 39 current release hashes passed before this work started. Those are baseline
results, not a substitute for final-candidate QA. New tests or files can change the counts.

Final candidate: 100 tests passed, 5 skipped, with 3 dependency warnings; all 50
current-release checksum entries passed. Both public PDF downloads match their local
candidates byte for byte, and all 19 immutable empirical evidence links return HTTP 200.
Both Zenodo drafts have three verified attachments and remain unsubmitted.

Zenodo documentation permits metadata edits and new versions; simultaneous publication
is an editorial convenience rather than a platform requirement. Reserving both DOIs
before rendering avoids a later PDF-only reference amendment.

Primary platform references:

- <https://help.zenodo.org/docs/deposit/describe-records/reserve-doi/>
- <https://help.zenodo.org/docs/deposit/manage-records/>
- <https://help.zenodo.org/docs/deposit/manage-files/>
- <https://help.zenodo.org/docs/deposit/manage-versions/>
