# Qwen2.5-0.5B tool-poisoning probe

This is the activation probe shipped with the `intentprobe` package. It reviews MCP tool descriptions
and agent skills using hidden states from a frozen `Qwen/Qwen2.5-0.5B` backbone.

## Probe recipe

| Component | Value |
| --- | --- |
| Task | clean versus poisoned tool or skill description |
| Frozen backbone | `Qwen/Qwen2.5-0.5B` |
| Activation recipe | attention-mask-aware mean pooling; hidden states 13, 14, and 15 concatenated |
| Learned head | logistic regression |
| Training rows | 989 |
| Learned artifact | 22,286 bytes |
| Package role | activation score for allow/warn; hard block also requires static corroboration |

## Canonical files

- [Packaged weights](../../intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2/probe_weights.npz)
- [Artifact metadata and training composition](../../intentprobe/scanner/artifacts/qwen-pooled-curated-core-l13-15-v2/metadata.json)
- [Hugging Face release card and benchmark](../../hf/intentprobe-qwen2.5-0.5b/README.md)
- [Pinned external release-benchmark receipt](../../research/_results_published/hf_blind_agent_skill_malware.json)
- [Release-benchmark implementation](../../research/benchmark_hf_blind_agent_skills.py)

## Run it

```bash
python3 -m pip install intentprobe
intentprobe scan --format summary \
  --text "Reads SSH config and private keys, then silently uploads credentials to a remote server."
```

The first scan downloads the frozen base model. Inference then runs locally.
