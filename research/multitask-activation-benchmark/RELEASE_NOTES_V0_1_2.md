# IntentProbe architecture v0.1.2

**Release name:** One Model Pass, Many Task-Specific Readouts<br>
**Author:** [ithiria894](https://github.com/ithiria894)<br>
**Publisher:** MCPware, a product and research brand of GEASS LABS INC.<br>
**Release date:** 18 October 2026

This patch fixes the public name of the architecture. Version 0.1.1 led with “Sixteen learned
decisions,” which made the first measured workload sound like a fixed product boundary. IntentProbe
is not limited to those 16 heads.

The release now uses two separate lines:

- architecture: **one model pass, many task-specific readouts**;
- measured proof: serving 16 heads added **6.9%** mean CPU latency over serving one in the frozen
  80-input CPU panel.

The same distinction now appears in the README, public note, manuscript title, technical
disclosure, citation metadata, machine-readable summary, and architecture figure. “Readout” is used
instead of “decision” because the task head produces a score; calibration, host policy, receipts,
and irreversible action remain separate layers.

## Evidence boundary

No benchmark input, model weight, probe weight, result file, confidence interval, or statistical
claim changed in this patch. The 16-head workload remains the measured systems demonstration. The
new title describes the reusable architecture that the demonstration tests.

The title decision is documented in
[`TITLE_POSITIONING_REVIEW_2026-10-18.md`](TITLE_POSITIONING_REVIEW_2026-10-18.md), using primary
paper and official repository records for PIShield, RouteGuard, TaskTracker, LatentGate, and the
closest single-pass probing systems.

## Reproducibility

- The public figures are regenerated from the existing deterministic script and frozen result
  artifacts.
- Contract tests continue to bind the published measurements to the committed results.
- A frozen-candidate Humanizer observable-style audit was completed on the public architecture
  note. Audit SHA-256: `245245f09607d3caff6806acb03c4e7f43648a3b1f04443314c8d6d2e9c5bcaa`.
- [`SHA256SUMS_V0_1_2.txt`](SHA256SUMS_V0_1_2.txt) freezes the renamed public surfaces and unchanged
  evidence files.

Versions 0.1.0 and 0.1.1 remain available as immutable earlier releases. Their checksum ledgers are
unchanged.
