# Authorization Send Detector

Research and engineering workspace for a runtime authorization boundary sensor.

The product question is not merely "is this action dangerous?" It is:

> Given the user's request, conversation history, exact proposed action, preview state,
> and approval scope, should the runtime allow the action, ask for confirmation, or block it?

The activation probe is one sensor. Final execution authority remains in a deterministic
state machine that binds approval to an exact payload or explicitly enumerated batch.

## Current status

- Living plan and experimental design: written.
- Source registry: pinned to immutable Hugging Face/GitHub/arXiv revisions where available.
- Raw source acquisition: reproducible through `scripts/fetch_sources.py`.
- Dataset QA: completed for the direct training/evaluation sources; results are in
  `data/manifests/qa_summary.json`.
- Model weights: all nine pinned snapshots in `MODEL_REGISTRY.json` were retrieved and smoke-tested;
  six frozen activation backbones completed the CPU screen.
- Benchmarking: EXP001 TF-IDF and EXP002 cross-backbone activation, dedicated-encoder, and one-shot
  held-out comparisons are complete.
- End-to-end reproduction commands and the recorded environment are in
  [`REPRODUCE.md`](REPRODUCE.md).
- Public research artifact: the MiniCPM5 hidden-output 13–14 probe, scorer, benchmark summary, and
  limitations are published at
  [`mcpware/intentprobe-minicpm5-1b-authorization`](https://huggingface.co/mcpware/intentprobe-minicpm5-1b-authorization).
- Product gate: the public probe remains research-only because validation-selected false-approve
  thresholds did not transfer to held-out data.

## Layout

```text
authorization-send-detector/
├── LIVING_PLAN.md
├── EXPERIMENT_DESIGN.md
├── LABEL_POLICY.md
├── SOURCES.md
├── SOURCE_REGISTRY.json
├── MODEL_REGISTRY.json
├── data/
│   ├── raw/          # immutable, gitignored source snapshots
│   ├── processed/    # derived data, gitignored
│   └── manifests/    # hashes, retrieval records, QA summaries
├── experiments/      # one folder per hypothesis
├── outputs/          # generated metrics/figures, gitignored until publication
├── scripts/
└── tests/
```

## First commands

```bash
python research/authorization-send-detector/scripts/fetch_sources.py
python research/authorization-send-detector/scripts/qa_sources.py
pytest -q research/authorization-send-detector/tests
```

Raw files are immutable. A source update is a new pinned revision and retrieval record, not an
in-place reinterpretation of the old snapshot.
