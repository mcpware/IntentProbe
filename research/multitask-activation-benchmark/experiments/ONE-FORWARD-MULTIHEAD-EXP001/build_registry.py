#!/usr/bin/env python3
"""Validate the 16 compatible readouts and freeze real-text workload IDs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
CONFIG = HERE / "config.json"
REGISTRY = HERE / "HEAD_REGISTRY.json"
WORKLOAD = HERE / "WORKLOAD_MANIFEST.json"


def sha256(path: Path) -> str:
    """Return the SHA-256 of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ece(y: np.ndarray, score: np.ndarray, bins: int = 10) -> float:
    """Return fixed-width expected calibration error."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignment = np.clip(np.digitize(score, edges[1:-1]), 0, bins - 1)
    counts = np.bincount(assignment, minlength=bins).astype(np.float64)
    positives = np.bincount(assignment, weights=y, minlength=bins)
    confidence = np.bincount(assignment, weights=score, minlength=bins)
    nonempty = counts > 0
    observed = np.divide(positives, counts, out=np.zeros(bins), where=nonempty)
    predicted = np.divide(confidence, counts, out=np.zeros(bins), where=nonempty)
    return float(np.sum(counts * np.abs(observed - predicted)) / len(y))


def native_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    """Compute the same binary quality and calibration metrics for every head."""
    prediction = score >= 0.5
    return {
        "rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "positive_prevalence": float(y.mean()),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "auprc": float(average_precision_score(y, score)),
        "auroc": float(roc_auc_score(y, score)),
        "brier": float(brier_score_loss(y, score)),
        "ece_10_bin": ece(y, score),
    }


def source_backbone(source_config: Path) -> dict[str, Any]:
    """Read the pinned backbone record from one source experiment."""
    source = json.loads(source_config.read_text(encoding="utf-8"))
    backbone = source.get("backbone") or source.get("model")
    if not isinstance(backbone, dict):
        raise RuntimeError(f"No backbone record in {source_config}")
    return backbone


def validate_head(
    specification: dict[str, Any], backbone: dict[str, Any], order: int
) -> dict[str, Any]:
    """Validate one probe and its native heldout score archive."""
    probe_path = PROJECT / specification["probe"]
    scores_path = PROJECT / specification["scores"]
    source_config_path = PROJECT / specification["source_config"]
    observed_backbone = source_backbone(source_config_path)
    if observed_backbone["id"] != backbone["id"]:
        raise RuntimeError(f"Backbone ID mismatch for {specification['name']}")
    if observed_backbone["revision"] != backbone["revision"]:
        raise RuntimeError(f"Backbone revision mismatch for {specification['name']}")

    with np.load(probe_path, allow_pickle=False) as probe:
        required = {
            "coefficient",
            "intercept",
            "train_means",
            "train_scales",
            "classes",
            "pooling",
            "start_layer",
            "end_layer",
        }
        if not required.issubset(probe.files):
            raise RuntimeError(f"Probe schema mismatch: {probe_path}")
        coefficient = np.asarray(probe["coefficient"], dtype=np.float64)
        intercept = np.asarray(probe["intercept"], dtype=np.float64)
        means = np.asarray(probe["train_means"], dtype=np.float64)
        scales = np.asarray(probe["train_scales"], dtype=np.float64)
        classes = np.asarray(probe["classes"])
        pooling = str(np.asarray(probe["pooling"]).item())
        start = int(np.asarray(probe["start_layer"]).item())
        end = int(np.asarray(probe["end_layer"]).item())
    width = end - start + 1
    if coefficient.shape != (1, int(backbone["hidden_size"])):
        raise RuntimeError(f"Coefficient shape mismatch: {probe_path}")
    if intercept.shape != (1,) or means.shape != (width, coefficient.shape[1]):
        raise RuntimeError(f"Normalization shape mismatch: {probe_path}")
    if scales.shape != means.shape or not np.array_equal(classes, [0, 1]):
        raise RuntimeError(f"Binary probe metadata mismatch: {probe_path}")
    if pooling not in {"masked_mean", "masked_max"}:
        raise RuntimeError(f"Unsupported shared pooling: {pooling}")
    arrays = (coefficient, intercept, means, scales)
    if not all(np.isfinite(array).all() for array in arrays) or bool((scales <= 0).any()):
        raise RuntimeError(f"Non-finite or non-positive probe parameter: {probe_path}")

    with np.load(scores_path, allow_pickle=False) as scores:
        y = np.asarray(scores["label"], dtype=np.int8)
        activation = np.asarray(scores["activation"], dtype=np.float64)
    if len(y) != len(activation) or np.unique(y).tolist() != [0, 1]:
        raise RuntimeError(f"Heldout score schema mismatch: {scores_path}")
    if not np.isfinite(activation).all() or bool(((activation < 0) | (activation > 1)).any()):
        raise RuntimeError(f"Invalid heldout activation probability: {scores_path}")

    return {
        "order": order,
        "name": specification["name"],
        "native_source": specification["native_source"],
        "model_id": backbone["id"],
        "model_revision": backbone["revision"],
        "pooling": pooling,
        "start_layer": start,
        "end_layer": end,
        "width": width,
        "normalized_depth": ((start + end) / 2) / (int(backbone["hidden_state_outputs"]) - 1),
        "probe_path": specification["probe"],
        "probe_sha256": sha256(probe_path),
        "probe_bytes": probe_path.stat().st_size,
        "scores_path": specification["scores"],
        "scores_sha256": sha256(scores_path),
        "source_config": specification["source_config"],
        "source_config_sha256": sha256(source_config_path),
        "native_heldout": native_metrics(y, activation),
    }


def build_workload(config: dict[str, Any]) -> dict[str, Any]:
    """Freeze exact real heldout IDs for latency and score-shift diagnostics."""
    source_records: dict[str, Any] = {}
    seed = int(config["seed"])
    ood_rows = int(config["ood_rows_per_source"])
    latency_rows = int(config["latency_rows_per_source"])
    for offset, (name, source) in enumerate(config["workload_sources"].items()):
        path = PROJECT / source["path"]
        frame = pd.read_parquet(
            path,
            columns=[source["id_column"], source["text_column"], source["split_column"]],
        )
        frame = frame.loc[
            frame[source["split_column"]].eq(source["split"]),
            [source["id_column"], source["text_column"]],
        ].copy()
        frame[source["id_column"]] = frame[source["id_column"]].astype(str)
        if frame[source["id_column"]].duplicated().any() or len(frame) < ood_rows:
            raise RuntimeError(f"Invalid heldout workload source: {name}")
        selected = frame.sample(n=ood_rows, random_state=seed + offset, replace=False)
        selected = selected.sort_values(source["id_column"]).reset_index(drop=True)
        ids = selected[source["id_column"]].tolist()
        source_records[name] = {
            **source,
            "file_sha256": sha256(path),
            "available_test_rows": int(len(frame)),
            "ood_ids": ids,
            "latency_ids": ids[:latency_rows],
        }
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "seed": seed,
        "config_sha256": sha256(CONFIG),
        "sources": source_records,
        "ood_rows": ood_rows * len(source_records),
        "latency_rows": latency_rows * len(source_records),
    }


def main() -> None:
    """Write the immutable registry and workload manifest."""
    if REGISTRY.exists() or WORKLOAD.exists():
        raise SystemExit("Refusing to overwrite the frozen registry or workload manifest")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if len(config["heads"]) != 16 or config["k_values"] != [1, 2, 4, 8, 16]:
        raise RuntimeError("Expected the frozen 16-head K schedule")
    backbone = config["backbone"]
    heads = [
        validate_head(specification, backbone, order)
        for order, specification in enumerate(config["heads"], start=1)
    ]
    registry = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256(CONFIG),
        "backbone": backbone,
        "head_count": len(heads),
        "heads": heads,
    }
    workload = build_workload(config)
    REGISTRY.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    WORKLOAD.write_text(json.dumps(workload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "heads": len(heads),
                "latency_rows": workload["latency_rows"],
                "ood_rows": workload["ood_rows"],
                "pooling_counts": pd.Series([head["pooling"] for head in heads])
                .value_counts()
                .to_dict(),
                "unique_layers": sorted(
                    {
                        layer
                        for head in heads
                        for layer in range(head["start_layer"], head["end_layer"] + 1)
                    }
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
