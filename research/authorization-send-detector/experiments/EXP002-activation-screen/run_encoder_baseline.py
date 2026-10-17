#!/usr/bin/env python3
"""Evaluate one pinned dedicated authorization encoder on the EXP002 screen."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, disable_caching
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from metrics_common import binary_metrics


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
REGISTRY = PROJECT / "MODEL_REGISTRY.json"
CONFIG = HERE / "config.json"
DATA = PROJECT / "data" / "processed" / "exp002" / "validation.parquet"
OUTPUT = HERE / "encoder_baselines"


def resolve(model_id: str) -> dict[str, Any]:
    """Resolve one exact dedicated-encoder registry entry."""
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if model_id not in config["dedicated_encoder_baselines"]:
        raise ValueError(f"{model_id} is not an EXP002 dedicated encoder")
    matches = [model for model in registry["models"] if model["id"] == model_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one registry record for {model_id}, got {len(matches)}")
    return matches[0]


def slug(model_id: str) -> str:
    """Return a filesystem-safe stable model key."""
    return model_id.replace("/", "__")


def main() -> None:
    """Run vectorized batched inference and score approve probabilities."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    experiment = json.loads(CONFIG.read_text(encoding="utf-8"))
    record = resolve(args.model)
    frame = pd.read_parquet(DATA, columns=["rendered", "label"])
    expected = experiment["screen_rows"]["validation"]
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} validation rows, got {len(frame)}")
    y_true = frame["label"].map({"deny": 0, "approve": 1}).to_numpy(dtype=np.int8)
    scores = np.empty(expected, dtype=np.float32)

    tokenizer = AutoTokenizer.from_pretrained(
        record["id"], revision=record["revision"], local_files_only=True, trust_remote_code=True
    )
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = experiment["renderer"]["truncation_side"]
    config = AutoConfig.from_pretrained(
        record["id"], revision=record["revision"], local_files_only=True, trust_remote_code=True
    )
    approve_index = int(config.label2id["approve"])
    started = time.perf_counter()
    model = AutoModelForSequenceClassification.from_pretrained(
        record["id"],
        revision=record["revision"],
        local_files_only=True,
        trust_remote_code=True,
        dtype="auto",
    )
    model.eval()
    disable_caching()
    dataset = Dataset.from_dict({"rendered": frame["rendered"].tolist()})

    def score_batch(batch: dict[str, list[str]], indices: list[int]) -> dict[str, list[int]]:
        """Score one framework batch and write probabilities by stable row index."""
        tokens = tokenizer(
            batch["rendered"],
            padding=True,
            truncation=True,
            max_length=experiment["renderer"]["max_tokens"],
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = model(**tokens, return_dict=True).logits
        probabilities = torch.softmax(logits.float(), dim=-1)[:, approve_index].cpu().numpy()
        scores[np.asarray(indices, dtype=np.int64)] = probabilities
        return {"done": [1] * len(indices)}

    completed = dataset.map(
        score_batch,
        batched=True,
        batch_size=args.batch_size,
        with_indices=True,
        remove_columns=["rendered"],
        load_from_cache_file=False,
        keep_in_memory=True,
        desc=record["id"],
    )
    if len(completed) != expected or sum(completed["done"]) != expected:
        raise RuntimeError("Incomplete encoder inference")
    result = {
        "experiment_id": experiment["experiment_id"],
        "model": record,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "approve_logit_index": approve_index,
        "rows": expected,
        "max_tokens": experiment["renderer"]["max_tokens"],
        "metrics": binary_metrics(y_true, scores, experiment["sampling_seed"]),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / f"{slug(record['id'])}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
