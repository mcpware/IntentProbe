# Multi-task activation benchmark

This is the primary paper track for IntentProbe.

Individual probes are useful products, but the paper asks a broader engineering question:

> Across small locally runnable language models and semantically different tasks, which hidden-state
> readouts remain useful after source leakage, surface-text baselines, calibration, and grouped
> uncertainty are controlled?

## Frozen task matrix

1. action authorization;
2. prompt injection / task drift;
3. tool or skill poisoning;
4. memory write;
5. toxicity as a non-agentic generality control.

## Shared factors

- backbone architecture and size;
- absolute and normalized layer depth;
- masked mean, last-token, max, and valid field/span pooling;
- single layers, bounded adjacent windows, and selected layer combinations;
- logistic activation probe versus matched TF-IDF and text-encoder baselines;
- grouped external transfer, calibration, low-error operating points, CPU latency, RAM, and artifact
  size.

## Evidence boundary

Human-labelled or human-written held-out data is the main external evidence. Model-generated labels
may support distillation and development, but never become an independent human-generalization test.
Every task keeps its own label definition and independent grouping unit; scores with incompatible
semantics are not pooled into a misleading universal safety number.

## Current state

- Authorization: six-backbone pilot and one frozen held-out evaluation complete.
- Memory write: public 15 KB synthetic-trained probe released; human-label experiments complete for
  Qwen2.5-0.5B. Direct human training raises fixed-output-11 AUROC to 0.7032. A 216-candidate screen on
  12 users selects masked-mean outputs 7--8; on five held-out users it reaches 0.1575 AUPRC / 0.7842
  AUROC versus 0.1726 / 0.7892 for TF-IDF, with paired intervals including zero. A post-test anchor
  comparison measures +0.0607 AUPRC and +0.1088 AUROC over output 11, showing why readout selection
  is a central engineering variable rather than a cosmetic detail.
- Prompt injection and tool/skill poisoning: historical evidence exists, but the shared protocol must
  be rerun.
- Toxicity: predeclared human-labelled generality control; acquisition and frozen split audit remain.

The detailed prospective protocol currently lives in
[`../authorization-send-detector/paper/MULTITASK_BENCHMARK_PLAN.md`](../authorization-send-detector/paper/MULTITASK_BENCHMARK_PLAN.md).
This index is the canonical entry point so the paper is not mistaken for an authorization-only or
memory-only study.

## Next execution order

1. freeze ToxicChat provenance, licence, grouping, and human-labelled slice;
2. implement one common representation-cache contract across the six backbones;
3. run the bounded readout matrix task by task without touching each task's external test;
4. replicate the memory outputs-7--8 result across other small backbones and fresh users;
5. fit the cross-task analysis only after the matrix and missing-cell rules are frozen.
