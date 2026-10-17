# EXP002 Run Log

Environment: Ryzen 5 PRO 7540U, 6 physical CPU cores, 30 GiB RAM, no NVIDIA GPU; PyTorch 2.11.0,
Transformers 5.8.0. All model and dataset revisions are pinned in the registries and retrieval
manifests.

## Completed evidence

- Downloaded all nine registered model snapshots: 13,292,386,617 cached bytes in total.
- Smoke-tested hidden-state output for all nine models.
- Built a deterministic 1,600-row train and 800-row validation screen with zero exact rendered
  cross-split overlap.
- Completed full hidden-state extraction and exhaustive single/adjacent-width-2/adjacent-width-3
  fitting for six activation backbones.
- Completed `auto-0.4b` and `auto-1b` publisher-trained encoder baselines.
- Froze MiniCPM5 hidden outputs 13–14 before first reading the held-out schema.
- Completed the one-shot 3,000-row held-out comparison and packaged the resulting 28,708-byte probe.

## Interrupted or corrected runs

- The first surface-baseline command was launched from the experiment directory with repo-relative
  paths and failed immediately with `Errno 2`. It was rerun from the repository root; no artifact was
  produced by the failed command.
- Falcon-H1 reported missing optional Mamba/causal-convolution fast paths. A 16-row, 192-token-capped
  batch completed zero rows after 151 seconds, at about 443% CPU and 11,283,552 KiB RSS. The run was
  interrupted and the model was excluded from accuracy ranking as CPU-infeasible, not assigned a
  fabricated score.
- Qwen3.5 batch sizes 16 and 32 were stopped after 80 and 64 rows respectively because batch 8 had
  better throughput. The final complete extraction used batch 8 and overwrote partial arrays.
- `auto-1b` batch 8 was stopped after 40 rows to test batch 16. Throughput was effectively unchanged;
  the complete recorded run used batch 16.

## Freeze and test audit

- Frozen selection timestamp: `2026-10-17T04:44:25.426600+00:00`.
- Original frozen-selection SHA-256 before held-out read:
  `1293df7db9a5d984faffd11d59217bd2651fbf4a3efc66527eae83dbf74c0181`.
- Held-out source SHA-256:
  `13607db717323399fb394885fdbf80663e5262b219683361c79e148853bc2ec1`.
- Exact rendered overlap: train/test 0; validation/test 0; test unique 3,000/3,000.
- Held-out result SHA-256:
  `19354ce783ac0cac3da7cab69a9326d5dd8f0d491528058dc3332eca3ccc2b0e`.

The held-out file is now consumed. No result from it may be used to revise EXP002 and then be reported
as fresh held-out evidence.
