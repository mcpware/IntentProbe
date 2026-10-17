#!/usr/bin/env python3
"""Diagnose EXP002 held-out drift without tuning on the consumed test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SCREEN = PROJECT / "data" / "processed" / "exp002"
TEST = PROJECT / "data" / "raw" / "huggingface" / "approve-or-deny" / "test.parquet"
ARTIFACT = PROJECT / "artifacts" / "minicpm5-1b-authorization-exp002" / "probe.npz"
VALIDATION_FEATURES = (
    PROJECT / ".cache" / "exp002" / "openbmb__MiniCPM5-1B-Base" / "validation_masked_mean.npy"
)
TEST_FEATURES = PROJECT / ".cache" / "exp002" / "heldout" / "frozen_test_selected_layers.npy"
RUNTIME_FIELDS = ["call", "user_request", "history"]


def labels(frame: pd.DataFrame) -> np.ndarray:
    """Map documented labels to integers."""
    mapped = frame["label"].map({"deny": 0, "approve": 1})
    if mapped.isna().any():
        raise ValueError("Unexpected label")
    return mapped.to_numpy(dtype=np.int8)


def render(frame: pd.DataFrame) -> pd.Series:
    """Apply the frozen renderer."""
    parts = frame[RUNTIME_FIELDS].fillna("").astype(str)
    return (
        "PROPOSED ACTION:\n"
        + parts["call"]
        + "\nUSER REQUEST:\n"
        + parts["user_request"]
        + "\nHISTORY:\n"
        + parts["history"]
    )


def score_summary(y: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """Describe score scale and calibration without fitting a calibrator."""
    output: dict[str, Any] = {
        "prevalence": float(y.mean()),
        "mean_prediction": float(scores.mean()),
        "calibration_in_the_large": float(scores.mean() - y.mean()),
        "brier": float(brier_score_loss(y, scores)),
    }
    for label, value in (("deny", 0), ("approve", 1)):
        selected = scores[y == value]
        output[f"{label}_score_quantiles"] = {
            "p05": float(np.quantile(selected, 0.05)),
            "p50": float(np.quantile(selected, 0.50)),
            "p95": float(np.quantile(selected, 0.95)),
        }
    return output


def subgroup_comparison(
    frame: pd.DataFrame,
    activation: np.ndarray,
    surface: np.ndarray,
) -> dict[str, list[dict[str, Any]]]:
    """Compare both methods within sufficiently large two-class groups."""
    y_all = labels(frame)
    output: dict[str, list[dict[str, Any]]] = {}
    for column in ("difficulty", "lang"):
        groups: list[dict[str, Any]] = []
        for value, positions in frame.groupby(column, dropna=False).indices.items():
            index = np.asarray(positions, dtype=np.int64)
            y = y_all[index]
            if len(index) < 30 or len(np.unique(y)) < 2:
                continue
            activation_auprc = float(average_precision_score(y, activation[index]))
            surface_auprc = float(average_precision_score(y, surface[index]))
            activation_auroc = float(roc_auc_score(y, activation[index]))
            surface_auroc = float(roc_auc_score(y, surface[index]))
            groups.append(
                {
                    "value": str(value),
                    "n": int(len(index)),
                    "approve_n": int(y.sum()),
                    "activation_auprc": activation_auprc,
                    "surface_auprc": surface_auprc,
                    "auprc_delta": activation_auprc - surface_auprc,
                    "activation_auroc": activation_auroc,
                    "surface_auroc": surface_auroc,
                    "auroc_delta": activation_auroc - surface_auroc,
                }
            )
        output[column] = sorted(groups, key=lambda row: row["value"])
    return output


def main() -> None:
    """Reconstruct frozen scores and quantify distribution/calibration drift."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    train = pd.read_parquet(SCREEN / "train.parquet", columns=["rendered", "label"])
    validation = pd.read_parquet(SCREEN / "validation.parquet", columns=["rendered", "label"])
    test = pd.read_parquet(TEST, columns=RUNTIME_FIELDS + ["label", "difficulty", "lang"])
    test["rendered"] = render(test)
    train_y = labels(train)
    validation_y = labels(validation)
    test_y = labels(test)

    artifact = np.load(ARTIFACT)
    start_layer = int(artifact["start_layer"][0])
    end_layer = int(artifact["end_layer"][0])
    means = artifact["mean"].astype(np.float32)
    scales = artifact["scale"].astype(np.float32)
    coefficient = artifact["coefficient"].astype(np.float64).reshape(-1)
    intercept = float(artifact["intercept"][0])

    validation_raw = np.load(VALIDATION_FEATURES, mmap_mode="r")[
        :, start_layer : end_layer + 1
    ].astype(np.float32)
    test_raw = np.load(TEST_FEATURES, mmap_mode="r").astype(np.float32)
    validation_x = ((validation_raw - means) / scales).mean(axis=1)
    test_x = ((test_raw - means) / scales).mean(axis=1)
    validation_activation = expit(validation_x @ coefficient + intercept)
    test_activation = expit(test_x @ coefficient + intercept)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    train_surface = vectorizer.fit_transform(train["rendered"])
    validation_surface = vectorizer.transform(validation["rendered"])
    test_surface = vectorizer.transform(test["rendered"])
    surface_head = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=20261016,
    )
    surface_head.fit(train_surface, train_y)
    validation_surface_scores = surface_head.predict_proba(validation_surface)[:, 1]
    test_surface_scores = surface_head.predict_proba(test_surface)[:, 1]

    heldout = json.loads((HERE / "heldout_result.json").read_text(encoding="utf-8"))
    if not np.isclose(
        average_precision_score(test_y, test_activation),
        heldout["activation"]["metrics"]["auprc"],
        atol=1e-10,
    ):
        raise ValueError("Reconstructed activation score does not match held-out result")

    diagnostics = {
        "schema_version": 1,
        "descriptive_only_no_test_tuning": True,
        "activation": {
            "validation": score_summary(validation_y, validation_activation),
            "heldout": score_summary(test_y, test_activation),
        },
        "surface": {
            "validation": score_summary(validation_y, validation_surface_scores),
            "heldout": score_summary(test_y, test_surface_scores),
        },
        "heldout_score_spearman": float(
            pd.Series(test_activation).corr(pd.Series(test_surface_scores), method="spearman")
        ),
        "heldout_subgroups": subgroup_comparison(test, test_activation, test_surface_scores),
        "interpretation": [
            "A threshold-free ranking gain and a transferable probability threshold are different claims.",
            "These diagnostics consume the already-opened test only descriptively and must not select a revised artifact.",
        ],
    }
    (HERE / "heldout_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
