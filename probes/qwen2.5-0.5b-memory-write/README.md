# Qwen2.5-0.5B memory-write probe

| Field | Value |
| --- | --- |
| Task | Rank one user utterance for possible long-term memory |
| Frozen backbone | `Qwen/Qwen2.5-0.5B` |
| Pinned revision | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| Activation recipe | masked mean of hidden-state output 11 |
| Learned artifact | 15,305 bytes |
| Development result | AUPRC 0.8897 / AUROC 0.9030 over 3,200 rows in 32 held-out seed families |

Canonical files:

- [Hugging Face release package](../../hf/intentprobe-qwen2.5-0.5b-memory-write/)
- [Frozen experiment and full representation search](../../research/memory-write-gate/experiments/EXP002-qwen-activation-screen/)
- [Human-source audit](../../research/memory-write-gate/HUMAN_DATASET_AUDIT_2026-10-17.md)
- [Living plan](../../research/memory-write-gate/LIVING_PLAN.md)

The same Qwen backbone used by the tool-poisoning detector selects a different internal recipe for
memory writes: output 11 instead of outputs 13–15. The learned heads are task-specific and are not
interchangeable.
