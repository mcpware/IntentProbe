# Paper plan — Where to probe small language models for runtime authorization

## Working thesis

For a fixed authorization task and lightweight linear head, the best hidden-state readout is
architecture-specific rather than simply the last layer. Masked-mean pooling was selected by all six
completed frozen backbones, but the best hidden output window ranged from one to three outputs and
from normalized depth 0.5000 to 0.7857.

## Claims supported by EXP002 now

1. Six frozen small backbones completed the same renderer, 1,600-row training budget, 800-row
   validation screen, pooling family, layer/window search, and logistic head.
2. Every selected activation candidate used masked-mean pooling.
3. The selected output location and window width varied by architecture.
4. MiniCPM5 outputs 13–14 ranked first by validation AUPRC among the completed activation probes.
5. On a separately opened 3,000-row test, MiniCPM improved AUROC over same-budget TF-IDF by 0.0240;
   paired row-bootstrap 95% CI `[0.0056, 0.0429]`, `p=0.0150`.
6. The held-out AUPRC difference of 0.0144 was inconclusive; 95% CI `[-0.0053, 0.0336]`, `p=0.1439`.
7. Validation-frozen MiniCPM thresholds targeting 5% and 1% false approves produced 9.49% and 5.07%
   on held-out data. The artifact is therefore not a standalone authorization gate.

These claims support an arXiv or workshop research preview now. They do not yet support a broad claim
that activation probes beat dedicated encoders, generalize across scenario families, or are calibrated
for autonomous execution.

## Secondary product-safety result

Better overall ranking does not automatically produce a transferable execution threshold. The
selected MiniCPM probe improved held-out AUROC over matched-budget TF-IDF, while its
validation-selected low-false-approve thresholds shifted on the separate test. This is a secondary
deployment finding, not the paper's headline contribution.

## Strong-paper experiments still required

- Build a gold `ALLOW / CONFIRM / BLOCK` contrast set from deterministic authorization state, with
  scenario-family IDs and exact payload/approval mutations.
- Reserve a new untouched external test. The 3,000-row EXP002 held-out source is permanently consumed.
- Run nested group-aware selection with multiple seeds and clustered uncertainty by scenario family.
- Complete max pooling, sparse/non-adjacent windows, action-only, output-logit, and matched-budget
  fine-tuned-encoder baselines.
- Report selection regret: per-model best readout versus one universal normalized-depth rule.
- Separate discrimination, calibration, abstention, latency, peak memory, and low-false-approve recall.
- Test tool, action-family, framework, language, and template holdouts.
- Replicate the layer-location analysis on the memory-write task to estimate task-by-model interaction.

## Product and paper share the same work

| Product artifact | Paper evidence |
| --- | --- |
| Versioned renderer and tool-call schema | Prevents serialization drift and defines the measured task |
| Layer/pooling search receipt | Reproducible representation-selection benchmark |
| Tiny probe plus pinned backbone | Concrete deployment cost and architecture identity |
| Deterministic authorization state machine | Source of clean contrast families and gold labels |
| Runtime receipts and user corrections | Future calibration and failure corpus |
| Hugging Face model card | Public artifact, limitations, provenance, and reproduction entry point |

## Proposed structure

1. Authorization failures in tool-using agents.
2. Frozen internal-state probes versus surface and fine-tuned baselines.
3. Cross-backbone layer, width, and pooling protocol.
4. EXP002 screen and one-shot held-out results.
5. Discrimination–calibration mismatch and product consequences.
6. Gold contrast benchmark and group-aware replication.
7. Limitations, licensing, privacy, and deterministic execution authority.

## Release anchor

Public artifact:
[`mcpware/intentprobe-minicpm5-1b-authorization`](https://huggingface.co/mcpware/intentprobe-minicpm5-1b-authorization)
at public release commit `6db35552fd4836b5d201170073e244b5f61b00fe`.

Primary local evidence:

- `experiments/EXP002-activation-screen/FROZEN_SELECTION.json`
- `experiments/EXP002-activation-screen/screen_leaderboard.json`
- `experiments/EXP002-activation-screen/heldout_result.json`
- `artifacts/minicpm5-1b-authorization-exp002/metadata.json`
