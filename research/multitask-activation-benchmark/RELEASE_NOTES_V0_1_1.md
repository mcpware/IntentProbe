# IntentProbe architecture v0.1.1

**Release name:** One Forward, Many Decisions<br>
**Author:** [Wan Sheng Leung](https://github.com/ithiria894)<br>
**Publisher:** MCPware, a product and research brand of GEASS LABS INC.<br>
**Release date:** 18 October 2026

This patch makes the architecture easier to see without changing the frozen experiment.

The first screen now states the result directly: one Qwen2.5-0.5B pass served 16 learned heads with
6.9% more mean latency than the one-head path on the recorded 80-input CPU panel. The architecture
figure now draws the hidden-state fan-out, task-specific layers, separate host-control boundary, and
offline training path instead of collapsing the heads into one box.

The latency figure and every public summary now keep two measurement panels separate:

- shared-path scaling uses 80 held-out inputs per K and rises from 68.63 ms at K=1 to 73.34 ms at
  K=16, a paired ratio of 1.069 (95% CI `[1.056, 1.078]`);
- the matched ten-prompt comparator measures 62.70 ms for one forward plus 16 heads. Sixteen
  repeated base-model forwards take 956.71 ms, a ratio of 15.26 (95% CI `[15.04, 15.49]`).

The repeated-forward arm still omits task prompts, LM-head scoring, parsing, and generation. It is a
conservative lower bound for separate LLM-judge calls, not a complete judge implementation.

## Release boundary

- Shipped product: the PyPI package exposes the single-head tool-poisoning path.
- Public standalone research probes: authorization and memory-write scorers on Hugging Face.
- Measured research code: the reproducible 16-head one-forward runtime in this repository.
- Specified next runtime: the general manifest registry, hot-loading, and multi-head receipt
  emitter. These are not yet the PyPI product surface.

The 16 served heads remain a heterogeneous research workload. Each task still needs its own held-out
quality, calibration, shift, and admission evidence. This release sells the measured systems
architecture; it does not turn every first-generation head into a production detector.

## Reproducibility

- The machine-readable paper ledger now records the matched ten-prompt denominator explicitly.
- The manuscript presents the 80-input and ten-prompt measurements in separate tables.
- The figure generator uses a fixed bootstrap seed and deterministic SVG metadata.
- Contract tests bind the public numbers to the frozen result artifacts.
- A frozen-candidate Humanizer observable-style audit was completed on the public architecture
  note. Audit SHA-256: `b9851023fd43efc9ccfcc31b7d5336a4cbcab31feec01d2a597d670b0e2de5d7`.
- [`SHA256SUMS_V0_1_1.txt`](SHA256SUMS_V0_1_1.txt) freezes the public note, figures, schemas,
  manuscript, principal results, and prior-art record for this patch.

Version 0.1.0 remains available as the original public architecture release. Its checksum ledger is
unchanged.
