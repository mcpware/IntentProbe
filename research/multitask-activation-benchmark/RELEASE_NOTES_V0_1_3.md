# IntentProbe architecture v0.1.3 — publication and discoverability

**Author:** [ithiria894](https://github.com/ithiria894)<br>
**Publisher:** MCPware, a product and research brand of GEASS LABS INC.<br>
**Revision date:** 19 October 2026<br>
**Architecture first published:** 18 October 2026

This editorial revision explains the CPU sidecar architecture, links readers to usable probes and
reproduction code, and gives the project a working GitHub Pages entry point. It keeps the public
architecture title and presents the system through its implementation and measured evidence.

## Changes

- Public author fields and citation metadata use the GitHub handle `ithiria894`.
- The article and README explain the extra local pass beside a hosted agent and the reuse across
  the sidecar's own heads.
- Related work adds Goodfire × Rakuten, SingProbe, Lion Warden, and Constitutional Classifiers++.
  Citation corrections distinguish attention-based MindGuard from hidden-state linear probes.
- The website adds a homepage, title and description metadata, canonical page configuration,
  sitemap support, and machine-readable project links.
- Hugging Face card sources link back to the architecture and add relevant CPU/hidden-state tags.
- `.zenodo.json` supplies explicit author metadata for a future software archive. It does not
  assign a DOI or repurpose the earlier tool-poisoning paper's DOI.
- The separate six-page architecture preprint uses DOI `10.5281/zenodo.23467859`. Publication is
  prepared on Zenodo before the corresponding GitHub citation update is deployed. Its source,
  abstract and PDF renderer are included; its added experiments are follow-up work.

No benchmark dataset, weights, predictions, result JSON, confidence interval, or measured value is
changed. The current 16-head runtime remains research code. General registry, hot-loading, and
multi-head receipt emission remain specified work.

[`SHA256SUMS_V0_1_3.txt`](SHA256SUMS_V0_1_3.txt) records this candidate's public surfaces and the
unchanged experiment results. Older checksum ledgers refer to their original release snapshots;
they are not regenerated to match the changing main branch.

## Candidate checks

The four existing architecture, paper, multi-head and memory-release test modules pass (22 tests).
The CFF validates against the official 1.2.0 schema. YAML front matter and site configuration parse,
and the new GitHub blob links resolve to local source files. Benchmark artifacts and derived
result JSON have no diff.

The public article's Humanizer observable-style audit reports no stock phrases or Unicode
anomalies. Its long-sentence leads are metadata, link groups and enumerated lists; they were
inspected in context. Frozen article SHA-256:
`c78fb9bc953661ae34e8e17559c0a36122a06113d3158956cadfce7459b44cc7`.
This is a style check, not an AI-authorship detector result.

Jekyll is not installed locally. No packages were installed for this revision. A real Pages build,
live metadata inspection and HTTP-200 verification remain deployment checks after publication.
