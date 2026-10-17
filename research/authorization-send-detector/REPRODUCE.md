# Reproducing the authorization benchmark

EXP002 compares frozen hidden-state probes, a matched-budget word/bigram TF-IDF classifier, and two
publisher-trained authorization encoders. The model, pooling rule, and layers were frozen before the
separate test was opened.

## Environment

The recorded run used Python 3.10.12 on Linux, six physical CPU cores, and 30 GiB RAM without an
NVIDIA GPU. Install the recorded package versions in an isolated environment:

```bash
python3 -m venv .venv-authorization
. .venv-authorization/bin/activate
python3 -m pip install -r research/authorization-send-detector/requirements-research.txt
```

The full model screen downloads about 13.3 GB of pinned model files. Generated datasets and
activation arrays stay in gitignored local directories.

## 1. Retrieve and verify pinned inputs

```bash
python3 research/authorization-send-detector/scripts/fetch_sources.py
python3 research/authorization-send-detector/scripts/fetch_models.py
python3 research/authorization-send-detector/scripts/qa_sources.py
pytest -q research/authorization-send-detector/tests
```

The registries pin every Hugging Face or GitHub revision. Retrieval manifests record file size and
SHA-256 without recording a machine-specific cache path.

## 2. Build the deterministic screen

Run the following in a disposable clone or worktree because these commands regenerate the committed
screen evidence files:

```bash
python3 research/authorization-send-detector/experiments/EXP002-activation-screen/build_screen_dataset.py
python3 research/authorization-send-detector/experiments/EXP002-activation-screen/run_surface_baseline.py
```

The renderer uses only `call`, `user_request`, and `history`. Labels, rationales, and the publisher's
combined text field are excluded. The committed screen has 1,600 training rows, 800 validation rows,
and zero exact rendered-text overlap across the two splits.

## 3. Extract hidden states and fit every frozen probe

```bash
for model in \
  Qwen/Qwen2.5-0.5B \
  LiquidAI/LFM2.5-230M \
  ibm-granite/granite-4.0-350m-base \
  Qwen/Qwen3-0.6B-Base \
  Qwen/Qwen3.5-0.8B \
  openbmb/MiniCPM5-1B-Base
do
  python3 research/authorization-send-detector/experiments/EXP002-activation-screen/extract_activations.py --model "$model"
  python3 research/authorization-send-detector/experiments/EXP002-activation-screen/fit_probe.py --model "$model"
done
```

Falcon-H1 remains in the pinned model registry and smoke evidence. It is excluded from the completed
CPU ranking because its optional fast Mamba kernels were unavailable and the fallback did not finish
one 16-row batch in 151 seconds.

## 4. Run the publisher-trained reference encoders

```bash
python3 research/authorization-send-detector/experiments/EXP002-activation-screen/run_encoder_baseline.py --model ProCreations/auto-0.4b
python3 research/authorization-send-detector/experiments/EXP002-activation-screen/run_encoder_baseline.py --model ProCreations/auto-1b --batch-size 16
```

These encoders were trained by their publisher on a much larger corpus. They are useful reference
systems, not matched-budget activation-probe comparisons.

## 5. Rebuild the screen table

```bash
python3 research/authorization-send-detector/experiments/EXP002-activation-screen/aggregate_screen.py
```

The predeclared rule selects the completed frozen activation probe with the highest validation
AUPRC. In the recorded run that was MiniCPM5-1B, masked-mean hidden outputs 13–14.

## 6. Verify the published evidence bundle

The separate 3,000-row test is already consumed. It may be rerun to reproduce the committed result,
but it must not be used to change a model, layer, pooling rule, regularization value, calibration
method, or threshold and then be described as untouched evidence.

```bash
python3 research/authorization-send-detector/experiments/EXP002-activation-screen/run_frozen_heldout.py
python3 research/authorization-send-detector/experiments/EXP002-activation-screen/analyze_heldout.py
pytest -q tests/test_authorization_exp002.py
```

The integrity test checks the frozen selection, leaderboard, every hashed screen input, packaged
probe shape, held-out overlap audit, statistical conclusions, threshold transfer, and Hugging Face
release synchronization.
