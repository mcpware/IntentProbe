# Assumption and interpretation ledger

This ledger separates measured facts from design assumptions for
`ONE-FORWARD-MULTIHEAD-EXP001`.

| ID | Assumption or boundary | Evidence/check | Consequence |
| --- | --- | --- | --- |
| S1 | A shared runtime extracts all hidden outputs once, even when K=1 uses only one head. | Every K cell uses `output_hidden_states=True` on the same pinned Qwen2.5-0.5B revision. | The experiment measures the intended attachable-head bus, not an optimized single-head special case. |
| S2 | The first K heads form a fixed cumulative order. | `config.json` freezes 16 unique probe hashes and K=`1,2,4,8,16`. | The latency slope supports marginal head cost for this suite; it is not a random-head-combination study. |
| S3 | A repeated base forward is a lower bound for a separate local LLM judge call. | The comparator omits a task prompt, LM head, choice likelihood, parsing, and generation. | A measured speed advantage may be called conservative relative to repeated judge calls, but the comparator must not be labelled a full judge implementation. |
| S4 | The latency panel is paired. | The same 80 frozen real heldout IDs and token counts appear in every K artifact; tests enforce exact alignment. | Paired row bootstrap is appropriate for K-versus-K=1 latency ratios. |
| S5 | Separate processes make peak-RSS comparisons interpretable. | Every K cell has its own process metadata and `ru_maxrss`; K=1 and K=16 differ by +0.73 MiB. | The delta is evidence for this environment, not a hardware-neutral memory law. |
| S6 | Runs executed in ascending K order on one machine without randomized interleaving or CPU-frequency pinning. | Timestamps in the five metadata files record the order. | Small latency differences may include temporal system noise; paired intervals quantify row variation, not run-order uncertainty. |
| S7 | Native heldout metrics remain the quality evidence for each head. | All 16 registry entries point to frozen labelled score archives; test sizes and prevalences are reported separately. | Metrics must not be averaged into a universal safety score or compared without accounting for task prevalence. |
| S8 | Cross-domain rows do not acquire new labels merely because every head scores them. | The 320-row artifact contains real heldout text IDs from five sources, but non-native labels are undefined for each head. | OOD results are score-shift and monitoring diagnostics, not false-positive or accuracy estimates. |
| S9 | Live runtime scores may differ slightly from the original float16 activation-cache evaluation. | The 64-row native replay per head has worst mean absolute difference 0.00645 and minimum Pearson correlation 0.97986. | Native archived metrics remain canonical; live replay quantifies deployment sensitivity rather than silently assuming bit identity. |
| S10 | The six-cell direct-judge sentinel and the K-scaling experiment answer different questions. | The sentinel uses LFM2.5-230M and Qwen3.5-0.8B; the scaling suite uses Qwen2.5-0.5B. | Sentinel evidence supports task/model interaction. It is not a same-run latency comparator for the Qwen2.5 shared runtime. |
| S11 | Saved probe file size approximates head-distribution cost. | The 16 complete `.npz` artifacts total 386,121 bytes. | This includes metadata and normalization arrays; it is not the exact in-memory tensor increment. |
| S12 | One CPU environment is sufficient for an implementation result, not a universal throughput claim. | Metadata records Linux, Python, PyTorch, Transformers, six Torch threads, model revision, and per-process timing. | Replication on other CPUs/precisions and a GPU remains future work. |

## Expected-then-check record

The frozen expectation was that head cost would be small relative to the backbone and repeated
forwards would grow approximately linearly. Observed one-forward mean latency rose from 68.63 ms at
K=1 to 73.34 ms at K=16, while repeated-forward mean latency rose from 57.88 ms to 956.71 ms. The
direction matched the expectation. The exact ratios and confidence intervals were computed only
after all artifacts were complete.
