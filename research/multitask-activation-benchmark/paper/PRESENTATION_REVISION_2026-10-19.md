# Empirical-paper presentation revision — 19 October 2026

The author approved a targeted revision after reviewing the first two-paper candidate.
This changes presentation, not experiments, weights, datasets, task scores or conclusions.

## Applied

- Lead the abstract with the shared-pass CPU result, then explain task-specific readouts and
  quality comparisons. Shorten it from 363 to 220 whitespace-delimited words.
- Put the scaling contribution first without reordering the Results sections.
- Move the broader project-task scope to Section 3.2; keep native head-quality variation in the abstract.
- Retain the architecture illustration so that the empirical paper stands alone.
- Add the existing latency SVG unchanged to Section 5.4. Its caption identifies the matched
  ten-input panel, the separate 80-input means, the incomplete repeated-forward comparator,
  and the uncertainty that row bootstrap does and does not represent.
- Add a vector selected-window census to Section 5.3, plus a 300-DPI PNG and a reproducible
  renderer. It reads the frozen 16-head registry: 11 masked mean, five masked maximum,
  selected outputs 2–20, and no selected window including final output 24.

## Not adopted

- Remove the architecture illustration from the empirical paper: both papers need to be readable independently.
- Claim worldwide firstness or a uniquely novel scaling curve: the presentation review supplies no
  new comprehensive priority evidence.
- Treat selected-window diversity alone as proof that no common recipe can work: the figure is a
  census. The existing validation-selection regret and task comparisons have their own stated scope.
- Turn the repeated-forward ratio into a full LLM-judge speedup claim: the missing comparator work remains explicit.

## Initial presentation checks

The full suite passes 104 tests with five skipped and three dependency warnings. The new renderer
checks manifest integrity and matrix encoding before plotting. Two fresh SVG renders produced
identical SHA-256 hashes in the installed environment. The unchanged architecture and latency
assets, frozen experiment tree and derived numerical ledger have no diff from the prior candidate.
One completed CodeRabbit review of the presentation change reported zero findings.

The final PDF has 15 pages. The abstract, illustration, readout census, latency curve, captions,
tables and public evidence links are checked during candidate preparation. Both records remain
unsubmitted; the prior publication preview is replaced with exact revised files before any Publish
authorization is requested. No new experiment, package installation or deployment is part of this revision.

## Depth-definition and dual-panel follow-up

A subsequent author-approved clarification defines normalized depth as the window midpoint divided
by final output index 24. Midpoints 3–19 yield 0.125–0.792; the actual windows span indices 2–20
among 25 outputs (0–24). These values use different summaries of the same selected windows.

The empirical latency figure now has two equally sized panels with true linear K axes. Panel (a)
shows the 80-input shared-path overhead in percent, with the archived paired mean-ratio intervals
transformed into percent. Panel (b) shows matched ten-input absolute times and mean-time intervals.
It does not reuse the 80-input endpoint as the matched comparator's denominator. The earlier
architecture latency asset is retained unchanged; only the empirical paper uses the new figure.

The full suite now passes 107 tests with five skipped and three dependency warnings, including
checks of the distinct samples, interval transformation, equal-sized linear axes, percentage
labels, absence of insets and explicit midpoint/coverage wording. The new SVG reproduces exactly
within the installed rendering environment. The paper remains 15 pages with three figures.
