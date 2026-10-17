#!/usr/bin/env python3
"""Fit all frozen EXP002 layer/pooling/window probe candidates for one model."""

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
from sklearn.linear_model import LogisticRegression

from metrics_common import binary_metrics


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
CONFIG = HERE / "config.json"
DATA = PROJECT / "data" / "processed" / "exp002"
CACHE = PROJECT / ".cache" / "exp002"
OUTPUT = HERE / "probe_results"


def slug(model_id: str) -> str:
    """Return a filesystem-safe stable model key."""
    return model_id.replace("/", "__")


def labels(split: str) -> np.ndarray:
    """Load binary labels for one frozen screen split."""
    frame = pd.read_parquet(DATA / f"{split}.parquet", columns=["label"])
    values = frame["label"].map({"deny": 0, "approve": 1})
    if values.isna().any():
        raise ValueError(f"{split}: unexpected label")
    return values.to_numpy(dtype=np.int8)


def fit_candidate(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    """Fit the fixed linear head and return metrics plus validation scores."""
    head = config["head"]
    model = LogisticRegression(
        C=head["C"],
        class_weight="balanced",
        solver=head["solver"],
        max_iter=head["max_iter"],
        random_state=head["random_state"],
    )
    model.fit(train_x, train_y)
    scores = model.predict_proba(validation_x)[:, 1]
    return binary_metrics(validation_y, scores, config["sampling_seed"], include_ci=False), scores


def main() -> None:
    """Evaluate all predeclared layer/window candidates for one model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if args.model not in config["activation_models"]:
        raise ValueError(f"{args.model} is not an EXP002 activation model")
    started = time.perf_counter()
    model_slug = slug(args.model)
    train_y = labels("train")
    validation_y = labels("validation")
    candidates: list[dict[str, Any]] = []

    for pooling in config["pooling"]:
        train = np.load(CACHE / model_slug / f"train_{pooling}.npy", mmap_mode="r").astype(np.float32)
        validation = np.load(CACHE / model_slug / f"validation_{pooling}.npy", mmap_mode="r").astype(np.float32)
        if train.shape[0] != len(train_y) or validation.shape[0] != len(validation_y):
            raise ValueError(f"{pooling}: feature/label row mismatch")
        means = train.mean(axis=0, keepdims=True)
        scales = train.std(axis=0, keepdims=True)
        scales = np.where(scales < 1e-6, 1.0, scales)
        train_z = (train - means) / scales
        validation_z = (validation - means) / scales
        hidden_count = train_z.shape[1]

        for layer in range(hidden_count):
            metric, _ = fit_candidate(
                train_z[:, layer], train_y, validation_z[:, layer], validation_y, config
            )
            candidate = {
                "pooling": pooling,
                "kind": "individual",
                "start_layer": layer,
                "end_layer": layer,
                "width": 1,
                "normalized_depth": layer / (hidden_count - 1),
                "metrics": metric,
            }
            candidates.append(candidate)

        train_cumsum = np.concatenate(
            [np.zeros((len(train_z), 1, train_z.shape[2]), dtype=np.float32), np.cumsum(train_z, axis=1)],
            axis=1,
        )
        validation_cumsum = np.concatenate(
            [
                np.zeros((len(validation_z), 1, validation_z.shape[2]), dtype=np.float32),
                np.cumsum(validation_z, axis=1),
            ],
            axis=1,
        )
        for width in config["layer_candidates"]["adjacent_window_widths"]:
            for start_layer in range(hidden_count - width + 1):
                end_exclusive = start_layer + width
                train_window = (train_cumsum[:, end_exclusive] - train_cumsum[:, start_layer]) / width
                validation_window = (
                    validation_cumsum[:, end_exclusive] - validation_cumsum[:, start_layer]
                ) / width
                metric, scores = fit_candidate(
                    train_window, train_y, validation_window, validation_y, config
                )
                candidates.append(
                    {
                        "pooling": pooling,
                        "kind": f"adjacent_mean_{width}",
                        "start_layer": start_layer,
                        "end_layer": end_exclusive - 1,
                        "width": width,
                        "normalized_depth": ((start_layer + end_exclusive - 1) / 2) / (hidden_count - 1),
                        "metrics": metric,
                    }
                )

    selection_key = config["selection_metric"]
    best_index = int(np.argmax(np.asarray([candidate["metrics"][selection_key] for candidate in candidates])))
    best = candidates[best_index]

    # Refit exactly the selected candidate to attach row-bootstrap intervals without searching them.
    pooling = best["pooling"]
    train = np.load(CACHE / model_slug / f"train_{pooling}.npy", mmap_mode="r").astype(np.float32)
    validation = np.load(CACHE / model_slug / f"validation_{pooling}.npy", mmap_mode="r").astype(np.float32)
    means = train.mean(axis=0, keepdims=True)
    raw_scales = train.std(axis=0, keepdims=True)
    scales = np.where(raw_scales < 1e-6, 1.0, raw_scales)
    train_z = (train - means) / scales
    validation_z = (validation - means) / scales
    train_best = train_z[:, best["start_layer"] : best["end_layer"] + 1].mean(axis=1)
    validation_best = validation_z[:, best["start_layer"] : best["end_layer"] + 1].mean(axis=1)
    _, best_scores = fit_candidate(train_best, train_y, validation_best, validation_y, config)
    best["metrics"] = binary_metrics(validation_y, best_scores, config["sampling_seed"], include_ci=True)

    result = {
        "experiment_id": config["experiment_id"],
        "model": args.model,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "selection_metric": config["selection_metric"],
        "candidate_count": len(candidates),
        "best": best,
        "candidates": candidates,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / f"{model_slug}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
