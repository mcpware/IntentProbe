# Attempt log

## Pre-freeze registry validation

The first `build_registry.py` invocation stopped before writing either frozen manifest because the
SMS processed file uses `message_id`, while the draft config declared `example_id`. The parquet
schema was inspected, the source-specific `id_column` was corrected, and the builder was rerun. It
then wrote `HEAD_REGISTRY.json` and `WORKLOAD_MANIFEST.json` once. No row, head, result, or timing
artifact existed during the failed attempt.

## Measured runs

The five latency cells ran sequentially in separate processes in ascending K order: 1, 2, 4, 8,
16. Each completed once and wrote one timing artifact, one repeated-forward lower-bound artifact,
and one metadata file. The K=16 cross-domain scoring process also completed once. No completed
artifact was overwritten or selected after viewing a later K cell.

## Summary

`summarize_scaling.py` ran after all required artifacts existed. It wrote the machine-readable result
and four CSV tables once. Tests then checked exact row pairing across K, finite positive component
times, probability bounds, native quality bounds, the comparator label, and 64-row live/cache replay
for every head.
