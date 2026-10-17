# MiniCPM5-1B authorization probe

This probe estimates whether a proposed structured action is supported by the user's request and
conversation history. It was selected through the same frozen-backbone screen used for five other
small local language models.

## Probe recipe

| Component | Value |
| --- | --- |
| Task | approve versus deny a proposed agent action |
| Frozen backbone | `openbmb/MiniCPM5-1B-Base` at revision `156170697656c48f69915b33a2fb44110242187c` |
| Activation recipe | per-layer standardized masked mean; hidden outputs 13 and 14 averaged |
| Learned head | class-weighted logistic regression |
| Training budget | 1,600 rows |
| Learned artifact | 28,708 bytes |
| Benchmark result | first of six completed frozen backbones by validation AUPRC |

The separate 3,000-row test measured AUROC 0.8137 for the activation probe and 0.7897 for the
matched-budget word/bigram TF-IDF baseline. The paired AUROC difference was +0.0240 with a 95%
bootstrap interval of +0.0056 to +0.0429.

## Canonical files

- [Probe, scorer, metadata, and Hugging Face release card](../../hf/intentprobe-minicpm5-1b-authorization/)
- [Research overview](../../research/authorization-send-detector/README.md)
- [End-to-end reproduction guide](../../research/authorization-send-detector/REPRODUCE.md)
- [Living plan](../../research/authorization-send-detector/LIVING_PLAN.md)
- [Experiment design](../../research/authorization-send-detector/EXPERIMENT_DESIGN.md)
- [Pinned source registry](../../research/authorization-send-detector/SOURCE_REGISTRY.json)
- [Pinned model registry](../../research/authorization-send-detector/MODEL_REGISTRY.json)
- [Six-backbone results and one-shot held-out test](../../research/authorization-send-detector/experiments/EXP002-activation-screen/README.md)
- [Frozen selection record](../../research/authorization-send-detector/experiments/EXP002-activation-screen/FROZEN_SELECTION.json)
- [Integrity tests](../../tests/test_authorization_exp002.py)

## Reproduce the research workspace

Follow the [end-to-end reproduction guide](../../research/authorization-send-detector/REPRODUCE.md)
to retrieve the pinned inputs, rebuild the deterministic screen, extract every model's hidden states,
fit the layer candidates, and verify the frozen evidence chain.

The fetch scripts download pinned third-party data and model snapshots into ignored local folders.
Committed registries, hashes, experiment settings, and result files provide the audit trail without
duplicating multi-gigabyte source material in Git.
