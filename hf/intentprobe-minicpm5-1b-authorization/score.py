#!/usr/bin/env python3
"""Score structured authorization context with the frozen MiniCPM5 activation probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_ID = "openbmb/MiniCPM5-1B-Base"
MODEL_REVISION = "156170697656c48f69915b33a2fb44110242187c"
MAX_TOKENS = 192


def render_context(call: str, user_request: str, history: str) -> str:
    """Apply the exact renderer used during training and evaluation."""
    return (
        f"PROPOSED ACTION:\n{call}\n"
        f"USER REQUEST:\n{user_request}\n"
        f"HISTORY:\n{history}"
    )


def normalize_call_json(value: str) -> str:
    """Normalize a structured tool call to the benchmark's nested-JSON contract."""
    try:
        call = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("--call-json must be a valid JSON object") from error
    if not isinstance(call, dict) or not isinstance(call.get("tool"), str) or not call["tool"]:
        raise ValueError("--call-json must contain a non-empty string field named 'tool'")
    if "args" not in call:
        raise ValueError("--call-json must contain an 'args' field")
    args = call["args"]
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    normalized = dict(call)
    normalized["args"] = args
    return json.dumps(normalized, ensure_ascii=False)


def load_probe(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate the numeric-only probe artifact."""
    required = {"coefficient", "intercept", "mean", "scale", "start_layer", "end_layer"}
    with np.load(Path(path), allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Probe is missing arrays: {sorted(missing)}")
        probe = {key: np.asarray(archive[key]) for key in required}

    if probe["coefficient"].shape != (1, 1536):
        raise ValueError(f"Unexpected coefficient shape: {probe['coefficient'].shape}")
    if probe["intercept"].shape != (1,):
        raise ValueError(f"Unexpected intercept shape: {probe['intercept'].shape}")
    if probe["mean"].shape != (1, 2, 1536) or probe["scale"].shape != (1, 2, 1536):
        raise ValueError("Expected two layers of 1,536-dimensional normalization statistics")
    if int(probe["start_layer"][0]) != 13 or int(probe["end_layer"][0]) != 14:
        raise ValueError("This release requires hidden outputs 13 and 14")
    if np.any(probe["scale"] <= 0) or not all(np.isfinite(value).all() for value in probe.values()):
        raise ValueError("Probe arrays must be finite and scales must be positive")
    return probe


def score_pooled_activations(
    pooled: np.ndarray, probe: dict[str, np.ndarray]
) -> np.ndarray:
    """Score pooled layer activations shaped ``[batch, 2, 1536]``."""
    values = np.asarray(pooled, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (2, 1536):
        raise ValueError(f"Expected pooled shape [batch, 2, 1536], got {values.shape}")
    standardized = (values - probe["mean"]) / probe["scale"]
    features = standardized.mean(axis=1)
    logits = features @ probe["coefficient"].T + probe["intercept"]
    logits = logits[:, 0].astype(np.float64)
    scores = np.empty_like(logits)
    positive = logits >= 0
    scores[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    scores[~positive] = exp_logits / (1.0 + exp_logits)
    return scores


def choose_device(requested: str) -> str:
    """Resolve the requested local inference device."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def score_texts(
    texts: list[str], artifact: str | Path, device: str = "auto"
) -> np.ndarray:
    """Extract hidden outputs 13–14 from pinned MiniCPM5 and return approve scores."""
    if not texts:
        raise ValueError("At least one rendered context is required")
    probe = load_probe(artifact)
    resolved_device = choose_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        dtype="auto",
    ).to(resolved_device)
    model.eval()
    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors="pt",
    )
    tokens = {key: value.to(resolved_device) for key, value in tokens.items()}
    with torch.inference_mode():
        output = model(**tokens, output_hidden_states=True, return_dict=True)
    mask = tokens["attention_mask"].to(dtype=torch.float32)[:, :, None]
    pooled_layers = []
    for layer_index in range(13, 15):
        hidden = output.hidden_states[layer_index].float()
        pooled_layers.append((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0))
    pooled = torch.stack(pooled_layers, dim=1).cpu().numpy()
    return score_pooled_activations(pooled, probe)


def main() -> None:
    """Run one structured authorization example and print a research score."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path(__file__).with_name("probe.npz"))
    parser.add_argument(
        "--call-json",
        required=True,
        help="JSON object with tool and args fields; args may itself be an object or JSON string",
    )
    parser.add_argument("--user-request", required=True)
    parser.add_argument("--history", default="")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    args = parser.parse_args()
    rendered = render_context(normalize_call_json(args.call_json), args.user_request, args.history)
    score = float(score_texts([rendered], args.artifact, args.device)[0])
    print(
        json.dumps(
            {
                "approve_score": score,
                "status": "research_preview",
                "decision": None,
                "recommended_use": "Combine this ranking signal with explicit authorization state.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

