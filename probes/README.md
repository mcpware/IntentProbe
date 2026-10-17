# IntentProbe probe catalog

IntentProbe is an umbrella for small classifiers trained on internal activations from frozen local
language models. Each probe has its own task, backbone, layer recipe, data provenance, benchmark,
and reproducible research trail.

| Probe | Task | Frozen backbone | Internal signal | Learned artifact | Evidence |
| --- | --- | --- | --- | ---: | --- |
| [Qwen2.5-0.5B tool-poisoning](qwen2.5-0.5b-tool-poisoning/) | Review MCP tool and agent-skill descriptions | `Qwen/Qwen2.5-0.5B` | Mean-pooled hidden states 13–15 | 22 KB | [Release benchmark](../hf/intentprobe-qwen2.5-0.5b/README.md) |
| [MiniCPM5-1B authorization](minicpm5-1b-authorization/) | Score whether a proposed action is supported by the request and history | `openbmb/MiniCPM5-1B-Base` | Standardized masked-mean hidden outputs 13–14 | 28 KB | [Six-backbone screen and held-out test](../research/authorization-send-detector/experiments/EXP002-activation-screen/README.md) |

## What belongs to one probe

Every catalog entry points to one canonical copy of:

- the numeric probe artifact and integrity hash;
- the exact frozen backbone revision and activation recipe;
- source and model registries with immutable revisions;
- experiment configuration, scripts, results, uncertainty estimates, and tests;
- a runnable scorer or the IntentProbe package integration;
- the corresponding Hugging Face release.

Large base-model weights, downloaded third-party datasets, and generated activation caches are not
duplicated in Git. The registries and fetch scripts reconstruct them from pinned sources, while the
committed manifests preserve file hashes and the benchmark audit trail.

## Adding another detector

A new detector gets a new folder here. It may reuse the same frozen backbone or select a different
one, but it must publish its own layer/pooling recipe and task-specific evidence. Probe weights are
not interchangeable across backbones or activation layouts.
