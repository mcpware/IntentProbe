# IntentProbe architecture v0.1.0

**Release name:** One Forward, Many Decisions<br>
**Author:** Nicole BeWater<br>
**Publisher:** MCPware, a product and research brand of GEASS LABS INC.<br>
**Release date:** 18 October 2026

This release makes the IntentProbe shared-runtime architecture public in one coherent package. A
frozen local language model reads an input once. Independently trained task heads then reuse selected
hidden states to produce calibrated scores under their own label, renderer, readout and policy
contracts.

## Included

- the plain-English [public architecture note](../../docs/ONE_FORWARD_MANY_DECISIONS.md);
- the implementable [technical disclosure](ARCHITECTURE_DISCLOSURE_V0_1.md);
- the public research manuscript [*One Forward, Sixteen Decisions*](paper/DRAFT.md);
- the completed 24-cell model-by-task readout matrix;
- the six-cell activation-head versus same-backbone direct-judge sentinel;
- the completed Qwen2.5-0.5B `K=1,2,4,8,16` systems benchmark;
- machine-readable results, summaries, scripts, schemas and contract tests;
- publication-ready architecture and latency figures.
- a [SHA-256 release ledger](SHA256SUMS_V0_1_0.txt) covering the public note, figures, schemas,
  manuscript, principal results and prior-art record.

## Headline measured result

On the recorded six-core CPU, one Qwen2.5-0.5B forward plus one head took 68.63 ms per input. One
forward plus 16 heads took 73.34 ms, a ratio of 1.069 to the one-head path, paired-bootstrap 95% CI
`[1.056, 1.078]`. Sixteen repeated base-model forwards took 956.71 ms and were 15.26 times slower
than the shared path on matched prompts, 95% CI `[15.04, 15.49]`.

The repeated-forward arm excludes task prompts, LM-head scoring, parsing and generation. It is a
conservative lower bound for separate LLM-judge calls, not a complete judge implementation.

## Quality boundary

The architecture does not assume that every cheap head is good enough to deploy. In the direct
same-backbone sentinel, toxicity strongly favoured the trained activation heads, Qwen3.5 sentiment
was statistically tied, and human-labelled authorization favoured the direct judge after the
activation head's silver-data transfer. Every head keeps its own held-out quality, calibration,
threshold and distribution-shift evidence.

## Not included in the public source snapshot

Raw third-party datasets, multi-gigabyte activation caches, model weights and research artifacts
whose redistribution terms need separate review are not committed. Their source identifiers,
revisions, hashes, build scripts and aggregate results remain in the evidence record.

The next systems release will add the one-call multi-label judge and a measured compact-classifier
fleet on the same hardware and inputs.

## Publication QA

- The public explanation preserves Nicole BeWater's meaning and plain-language framing: a model can
  carry a useful hidden-state pattern without reliably turning it into the requested `YES` or `NO`.
- Every headline latency value is checked against the frozen machine-readable systems result by the
  repository test suite.
- A frozen-candidate Humanizer style audit was completed on the public architecture note. Audit
  SHA-256: `53f3bd36ba9cbca49fe671c058398a45631b780eab003c1eef45f72a4f7e3d79`.
- The manuscript records author responsibility and coding/language-assistance disclosure.
