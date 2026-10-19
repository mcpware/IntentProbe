# IntentProbe: Many Task-Specific Probes from a Single Frozen Small-Model Forward Pass

**Public research manuscript v0.1.3**<br>
**Author:** [Wan Sheng Leung](https://github.com/ithiria894)<br>
**Publisher:** MCPware, a product and research brand of GEASS LABS INC.

The empirical preprint has reserved DOI
[`10.5281/zenodo.23471117`](https://doi.org/10.5281/zenodo.23471117), separate from the
architecture note's [`10.5281/zenodo.23467859`](https://doi.org/10.5281/zenodo.23467859).
Both remain unsubmitted publication candidates until the author reviews the complete publication
preview and authorizes Publish. A reserved DOI will not resolve before publication.

The public manuscript source is [`DRAFT.md`](DRAFT.md); the PDF candidate is
[`intentprobe-empirical-v0.1.3.pdf`](intentprobe-empirical-v0.1.3.pdf), and its citation is
[`intentprobe-empirical.bib`](intentprobe-empirical.bib). Build it from the repository root with
`python3 scripts/build_multitask_preprint.py --output research/multitask-activation-benchmark/paper/intentprobe-empirical-v0.1.3.pdf --evidence-ref FULL_SOURCE_COMMIT`.
The final source commit and exact build command are recorded in `BUILD.md` when the PDF is frozen.
The renderer uses installed Markdown/WeasyPrint, the approved architecture figure, the two-panel
latency figure, the reproducible selected-readout figure, and public GitHub
evidence links pinned to that immutable commit; it does not install packages or expose local
filesystem links in the PDF.

Regenerate the selected-readout SVG and 300-DPI PNG with
`python3 scripts/render_multitask_readouts.py`. The figure's source, encoding and interpretation
boundaries are documented in [`outputs/figures/README.md`](outputs/figures/README.md).
Regenerate the separate 80-input scaling / ten-input cost figure with
`python3 scripts/render_multitask_latency.py`. It leaves the companion architecture assets unchanged.

Paper-level numbers are rebuilt from frozen
experiment artifacts by:

`python3 scripts/build_derived_results.py`

The script writes [`DERIVED_RESULTS.json`](DERIVED_RESULTS.json). Related-work and empirical sources
are separated in [`SOURCE_LEDGER.md`](SOURCE_LEDGER.md); decisions, assumptions, and remaining gates
are in [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md).

The manuscript is public but has not been peer reviewed. The shorter public explanation is
[*IntentProbe: One Local Model Pass, Many Task-Specific Probes*](../../../docs/ONE_LOCAL_MODEL_PASS_MANY_TASK_SPECIFIC_PROBES.md).
