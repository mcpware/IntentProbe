#!/usr/bin/env python3
"""Score the frozen MEM-EXP002 probe on the LUFY human-written set."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = HERE / "config.json"
DATA = ROOT / "data" / "processed" / "exp003" / "lufy_majority.parquet"
PROBE = HERE.parent / "EXP002-qwen-activation-screen" / "selected_probe.npz"
SCORES = HERE / "activation_scores.npz"
MANIFEST = HERE / "activation_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sigmoid(logits: np.ndarray) -> np.ndarray:
    scores = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0
    scores[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    scores[~positive] = exp_logits / (1.0 + exp_logits)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    model_config = config["activation_probe"]
    frame = pd.read_parquet(DATA)
    with np.load(PROBE, allow_pickle=False) as archive:
        coefficient = archive["coefficient"].astype(np.float64)
        intercept = archive["intercept"].astype(np.float64)
        means = archive["train_means"].astype(np.float64)
        scales = archive["train_scales"].astype(np.float64)
        layer = int(archive["start_layer"])
    if layer != 11 or coefficient.shape != (1, 896):
        raise ValueError("Unexpected frozen probe layout")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["backbone"],
        revision=model_config["backbone_revision"],
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    lengths = tokenizer(
        frame["rendered"].tolist(),
        add_special_tokens=True,
        truncation=False,
        return_length=True,
    )["length"]
    rows_over_cap = int(np.sum(np.asarray(lengths) > model_config["max_tokens"]))

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    model = AutoModel.from_pretrained(
        model_config["backbone"],
        revision=model_config["backbone_revision"],
        local_files_only=True,
        dtype="auto",
    )
    model.eval()
    pooled_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(frame), args.batch_size):
            rendered = frame["rendered"].iloc[start : start + args.batch_size].tolist()
            tokens = tokenizer(
                rendered,
                padding=True,
                truncation=True,
                max_length=model_config["max_tokens"],
                return_tensors="pt",
            )
            output = model(**tokens, output_hidden_states=True, return_dict=True)
            mask = tokens["attention_mask"].to(dtype=torch.float32)[:, :, None]
            hidden = output.hidden_states[layer].float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled_batches.append(pooled.cpu().numpy())
            if start and start % (args.batch_size * 50) == 0:
                print(f"scored {start}/{len(frame)}", flush=True)

    pooled_all = np.concatenate(pooled_batches, axis=0).astype(np.float64)
    features = (pooled_all - means) / scales
    logits = (features @ coefficient.T + intercept)[:, 0]
    scores = sigmoid(logits)
    np.savez_compressed(
        SCORES,
        score=scores,
        label=frame["label"].to_numpy(dtype=np.int8),
        row_id=frame["row_id"].astype(str).to_numpy(),
        user_name=frame["user_name"].astype(str).to_numpy(),
        system=frame["system"].astype(str).to_numpy(),
        unanimous=frame["unanimous"].to_numpy(dtype=bool),
    )
    manifest = {
        "experiment_id": "MEM-EXP003",
        "status": "SCORED",
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(frame),
        "batch_size": args.batch_size,
        "maximum_rendered_tokens": int(max(lengths)),
        "rows_over_256_tokens": rows_over_cap,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "input_sha256": sha256(DATA),
        "probe_sha256": sha256(PROBE),
        "scores_sha256": sha256(SCORES),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
