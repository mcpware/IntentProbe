#!/usr/bin/env python3
"""Score one utterance with the frozen IntentProbe memory-write probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
MAX_TOKENS = 256
HIDDEN_OUTPUT = 11


def render_text(text: str) -> str:
    """Apply the exact renderer used for training and development evaluation."""
    if not text.strip():
        raise ValueError("--text must not be empty")
    return f"USER_UTTERANCE\n{text}\n\nMEMORY_WRITE_DECISION"


def load_probe(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate the numeric-only release artifact."""
    required = {
        "coefficient",
        "intercept",
        "train_means",
        "train_scales",
        "classes",
        "start_layer",
        "end_layer",
        "pooling",
        "safe_threshold",
    }
    with np.load(Path(path), allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Probe is missing arrays: {sorted(missing)}")
        probe = {key: np.asarray(archive[key]) for key in required}

    if probe["coefficient"].shape != (1, 896):
        raise ValueError(f"Unexpected coefficient shape: {probe['coefficient'].shape}")
    if probe["intercept"].shape != (1,):
        raise ValueError(f"Unexpected intercept shape: {probe['intercept'].shape}")
    if probe["train_means"].shape != (1, 896) or probe["train_scales"].shape != (1, 896):
        raise ValueError("Expected one 896-dimensional normalization vector")
    if int(probe["start_layer"]) != HIDDEN_OUTPUT or int(probe["end_layer"]) != HIDDEN_OUTPUT:
        raise ValueError("This release requires hidden-state output 11")
    if str(probe["pooling"]) != "masked_mean":
        raise ValueError("This release requires masked-mean pooling")
    numeric = [
        probe["coefficient"],
        probe["intercept"],
        probe["train_means"],
        probe["train_scales"],
    ]
    if np.any(probe["train_scales"] <= 0) or not all(np.isfinite(value).all() for value in numeric):
        raise ValueError("Probe arrays must be finite and scales must be positive")
    return probe


def score_pooled_activations(pooled: np.ndarray, probe: dict[str, np.ndarray]) -> np.ndarray:
    """Score masked-mean activations shaped ``[batch, 896]``."""
    values = np.asarray(pooled, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 896:
        raise ValueError(f"Expected pooled shape [batch, 896], got {values.shape}")
    features = (values - probe["train_means"]) / probe["train_scales"]
    logits = (features @ probe["coefficient"].T + probe["intercept"])[:, 0].astype(np.float64)
    scores = np.empty_like(logits)
    positive = logits >= 0
    scores[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    scores[~positive] = exp_logits / (1.0 + exp_logits)
    return scores


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def score_texts(texts: list[str], artifact: str | Path, device: str = "auto") -> np.ndarray:
    """Extract hidden output 11 from pinned Qwen and return SAVE scores."""
    if not texts:
        raise ValueError("At least one text is required")
    probe = load_probe(artifact)
    resolved_device = choose_device(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype="auto",
    ).to(resolved_device)
    model.eval()
    rendered = [render_text(text) for text in texts]
    tokens = tokenizer(
        rendered,
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors="pt",
    )
    tokens = {key: value.to(resolved_device) for key, value in tokens.items()}
    with torch.inference_mode():
        output = model(**tokens, output_hidden_states=True, return_dict=True)
    mask = tokens["attention_mask"].to(dtype=torch.float32)[:, :, None]
    hidden = output.hidden_states[HIDDEN_OUTPUT].float()
    pooled = ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)).cpu().numpy()
    return score_pooled_activations(pooled, probe)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path(__file__).with_name("probe.npz"))
    parser.add_argument("--text", required=True, help="One user utterance to evaluate")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    args = parser.parse_args()
    probe = load_probe(args.artifact)
    score = float(score_texts([args.text], args.artifact, args.device)[0])
    safe_threshold = float(probe["safe_threshold"])
    print(
        json.dumps(
            {
                "save_score": score,
                "decision_at_0_5": "SAVE" if score >= 0.5 else "SKIP",
                "decision_at_development_1pct_false_save_threshold": (
                    "SAVE" if score >= safe_threshold else "SKIP"
                ),
                "status": "research_preview",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
