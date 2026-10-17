#!/usr/bin/env python3
"""Fit every frozen layer/window candidate for one MEM-EXP002 pooling."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = HERE / "config.json"
DATA = ROOT / "data" / "processed" / "exp002"
CACHE = ROOT / ".cache" / "exp002" / "Qwen__Qwen2.5-0.5B"
OUTPUT = HERE / "pooling_results"


def labels(split: str) -> np.ndarray:
    """Load binary labels in frozen processed-row order."""
    return pd.read_parquet(DATA / f"{split}.parquet", columns=["target"])[
        "target"
    ].to_numpy(dtype=np.int8)


def metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Calculate frozen validation selection metrics."""
    predicted = scores >= 0.5
    return {
        "auprc": float(average_precision_score(y, scores)),
        "auroc": float(roc_auc_score(y, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro")),
    }


def fit_scores(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Fit the preregistered linear head and return validation scores."""
    head = config["head"]
    model = LogisticRegression(
        C=head["C"],
        class_weight="balanced",
        solver=head["solver"],
        max_iter=head["max_iter"],
        random_state=head["random_state"],
    )
    model.fit(train_x, train_y)
    return model.predict_proba(validation_x)[:, 1]


def candidate_key(kind: str, start: int, end: int) -> str:
    """Return a stable candidate identity."""
    return f"{kind}:{start}:{end}"


def main() -> None:
    """Fit or resume one pooling's complete candidate family."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooling", required=True)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if args.pooling not in config["pooling"]:
        raise ValueError(f"Unknown pooling: {args.pooling}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / f"{args.pooling}.json"
    if output_path.exists():
        result = json.loads(output_path.read_text(encoding="utf-8"))
        candidates = result["candidates"]
    else:
        result = {
            "experiment_id": "MEM-EXP002",
            "pooling": args.pooling,
            "status": "RUNNING",
            "candidates": [],
        }
        candidates = result["candidates"]
    completed = {candidate["key"] for candidate in candidates}

    train_y = labels("train")
    validation_y = labels("validation")
    train = np.load(CACHE / f"train_{args.pooling}.npy", mmap_mode="r").astype(np.float32)
    validation = np.load(
        CACHE / f"validation_{args.pooling}.npy", mmap_mode="r"
    ).astype(np.float32)
    assert train.shape == (len(train_y), 25, 896)
    assert validation.shape == (len(validation_y), 25, 896)
    means = train.mean(axis=0, keepdims=True)
    scales = train.std(axis=0, keepdims=True)
    scales = np.where(scales < 1e-6, 1.0, scales)
    train_z = (train - means) / scales
    validation_z = (validation - means) / scales
    hidden_count = train_z.shape[1]
    started = time.perf_counter()

    definitions = [
        ("individual", layer, layer) for layer in range(hidden_count)
    ] + [
        (f"adjacent_mean_{width}", start, start + width - 1)
        for width in config["layer_candidates"]["adjacent_window_widths"]
        for start in range(hidden_count - width + 1)
    ]
    for kind, start, end in definitions:
        key = candidate_key(kind, start, end)
        if key in completed:
            continue
        train_x = train_z[:, start : end + 1].mean(axis=1)
        validation_x = validation_z[:, start : end + 1].mean(axis=1)
        scores = fit_scores(train_x, train_y, validation_x, config)
        candidate = {
            "key": key,
            "kind": kind,
            "start_layer": start,
            "end_layer": end,
            "width": end - start + 1,
            "normalized_depth": ((start + end) / 2) / (hidden_count - 1),
            "metrics": metrics(validation_y, scores),
        }
        candidates.append(candidate)
        best = max(candidates, key=lambda item: item["metrics"]["auprc"])
        result.update(
            status="RUNNING",
            completed_candidates=len(candidates),
            expected_candidates=len(definitions),
            current_best=best,
            elapsed_seconds_this_run=time.perf_counter() - started,
            peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        )
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "pooling": args.pooling,
                    "completed": len(candidates),
                    "candidate": key,
                    "auprc": candidate["metrics"]["auprc"],
                    "best": best["key"],
                    "best_auprc": best["metrics"]["auprc"],
                }
            ),
            flush=True,
        )

    result.update(
        status="COMPLETED",
        completed_candidates=len(candidates),
        expected_candidates=len(definitions),
        elapsed_seconds_this_run=time.perf_counter() - started,
        peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    )
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
