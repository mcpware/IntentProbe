# Empirical paper figures

## Selected readout windows

Figure 2 answers one question: which hidden-state windows and token-pooling rules were selected
for the 16 frozen heads served by the shared Qwen2.5-0.5B runtime?

Source: [`HEAD_REGISTRY.json`](../../../experiments/ONE-FORWARD-MULTIHEAD-EXP001/HEAD_REGISTRY.json).
The script reads only the head metadata; no source text, activations, fitting or inference is needed.
From the repository root:

`python3 scripts/render_multitask_readouts.py`

The renderer validates the 16 unique heads and their frozen order, valid output indices, window
widths and the selected pooling census before drawing the figure. It writes
`selected-readout-windows.svg` and a 300-DPI `selected-readout-windows.png` here. The current-release
checksum ledger identifies the exact figure bytes. Fixed SVG date and hash salt make the SVG
reproducible within the recorded rendering environment.

Each row is a head, and each filled cell is an output in its selected window. Blue means masked
mean, orange means masked maximum; right-hand text repeats the pooling label for greyscale access.
The hatched column is final output 24. Output 0 is the embedding output. Selected windows span
outputs 2–20; 11 heads use masked mean and five use masked maximum. These are exact manifest
counts, not uncertain performance estimates, so error bars do not apply.

This is a selection census, not an accuracy heatmap, a causal result or proof that no common recipe
could work. The paper separately reports validation-selection regret and native heldout quality.
It reuses the repository's existing Matplotlib style rather than a generic bar-chart helper,
because this figure encodes selected discrete intervals instead of a magnitude per category.

## Two-panel latency figure

Figure 3 uses `latency-panels.svg` and a 300-DPI `latency-panels.png`, generated with:

`python3 scripts/render_multitask_latency.py`

The two equally sized panels answer different questions and use true linear K axes:

- Panel (a): how much does shared-path cost increase when more heads are attached? It uses all
  80 matched inputs per K and plots `100 * (mean_total_ms(K) / mean_total_ms(1) - 1)`. Error bars
  transform the archived paired mean-ratio 95% intervals in `SCALING-SUMMARY.csv` by the same
  subtraction and multiplication. They are not intervals for individual timing observations.
- Panel (b): how does shared inference compare with repeating the backbone? It uses the same
  ten inputs in both paths and plots mean milliseconds with the existing 2,000-resample
  row-bootstrap mean-time intervals and fixed seeds. The repeated/shared endpoint ratio is
  15.26, using 62.70 ms rather than the 80-input mean of 73.34 ms as its denominator.

The renderer verifies row identity, panel sizes, the summary means/ratios and ratio-interval
bounds before plotting. Both y axes are linear: percentage increase on the left, absolute
milliseconds from zero on the right. Marker shape and line style distinguish the right-hand
series as well as colour. There is no inset. These are reformatted frozen measurements, not a
new timing experiment. Intervals capture prompt variation, not repeated-run or run-order
uncertainty. The repeated-forward comparator omits prompting, output scoring and generation.

The architecture companion's earlier latency SVG under `docs/` is unchanged; this paper-only
asset does not modify or rebuild the architecture PDF.
