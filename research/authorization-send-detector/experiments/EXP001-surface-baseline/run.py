#!/usr/bin/env python3
"""Run EXP001: a frozen scalable TF-IDF logistic surface baseline."""

from __future__ import annotations

import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import bootstrap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
    roc_curve,
)


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RAW = PROJECT / "data" / "raw" / "huggingface"
CONFIG_PATH = HERE / "config.json"


def load_split(path: Path) -> tuple[pd.Series, np.ndarray]:
    frame = pd.read_parquet(path, columns=["text", "label"])
    if frame[["text", "label"]].isna().any().any():
        raise ValueError(f"Null text/label in {path}")
    labels = frame["label"].map({"deny": 0, "approve": 1})
    if labels.isna().any():
        raise ValueError(f"Unexpected label in {path}: {sorted(frame['label'].unique())}")
    return frame["text"].astype(str), labels.to_numpy(dtype=np.int8)


def normalized_fingerprint(series: pd.Series) -> set[int]:
    normalized = series.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    return set(pd.util.hash_pandas_object(normalized, index=False).astype("uint64").tolist())


def ece(y_true: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    ids = np.minimum((scores * bins).astype(int), bins - 1)
    table = pd.DataFrame({"bin": ids, "y": y_true, "score": scores}).groupby("bin", observed=True)
    aggregate = table.agg(count=("y", "size"), accuracy=("y", "mean"), confidence=("score", "mean"))
    return float(((aggregate["count"] / len(y_true)) * (aggregate["accuracy"] - aggregate["confidence"]).abs()).sum())


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [float(centre - margin), float(centre + margin)]


def bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    resamples: int,
    seed: int,
) -> list[float]:
    result = bootstrap(
        (y_true, scores),
        statistic,
        paired=True,
        vectorized=False,
        n_resamples=resamples,
        batch=100,
        method="percentile",
        rng=np.random.default_rng(seed),
    )
    return [float(result.confidence_interval.low), float(result.confidence_interval.high)]


def metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float, config: dict[str, Any]) -> dict[str, Any]:
    predicted = (scores >= threshold).astype(np.int8)
    negatives = y_true == 0
    positives = y_true == 1
    false_approve = int(np.sum((predicted == 1) & negatives))
    false_deny = int(np.sum((predicted == 0) & positives))
    auroc = float(roc_auc_score(y_true, scores))
    auprc = float(average_precision_score(y_true, scores))
    return {
        "n": int(len(y_true)),
        "positive_n": int(positives.sum()),
        "negative_n": int(negatives.sum()),
        "threshold": float(threshold),
        "auroc": auroc,
        "auroc_bootstrap_95ci": bootstrap_ci(
            y_true, scores, lambda y, s: float(roc_auc_score(y, s)), config["bootstrap_resamples"], config["bootstrap_seed"]
        ),
        "auprc": auprc,
        "auprc_bootstrap_95ci": bootstrap_ci(
            y_true,
            scores,
            lambda y, s: float(average_precision_score(y, s)),
            config["bootstrap_resamples"],
            config["bootstrap_seed"] + 1,
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "f1": float(f1_score(y_true, predicted)),
        "brier": float(brier_score_loss(y_true, scores)),
        "ece_10_equal_width": ece(y_true, scores),
        "false_approve": false_approve,
        "false_approve_rate": float(false_approve / negatives.sum()),
        "false_approve_rate_wilson_95ci": wilson(false_approve, int(negatives.sum())),
        "false_deny": false_deny,
        "false_deny_rate": float(false_deny / positives.sum()),
        "false_deny_rate_wilson_95ci": wilson(false_deny, int(positives.sum())),
    }


def main() -> None:
    started = time.perf_counter()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    train_text, train_y = load_split(RAW / "auto-1b-data" / "train.parquet")
    validation_text, validation_y = load_split(RAW / "auto-1b-data" / "validation.parquet")
    test_text, test_y = load_split(RAW / "approve-or-deny" / "test.parquet")

    train_fp = normalized_fingerprint(train_text)
    validation_fp = normalized_fingerprint(validation_text)
    test_fp = normalized_fingerprint(test_text)
    overlaps = {
        "train_validation": len(train_fp & validation_fp),
        "train_test": len(train_fp & test_fp),
        "validation_test": len(validation_fp & test_fp),
        "method": "pandas normalized 64-bit fingerprint; source QA separately records within-source SHA-256",
    }
    if any(overlaps[key] for key in ("train_validation", "train_test", "validation_test")):
        raise RuntimeError(f"Cross-split normalized-text overlap: {overlaps}")

    vectorizer = TfidfVectorizer(
        ngram_range=tuple(config["vectorizer"]["ngram_range"]),
        min_df=config["vectorizer"]["min_df"],
        max_df=config["vectorizer"]["max_df"],
        max_features=config["vectorizer"]["max_features"],
        sublinear_tf=config["vectorizer"]["sublinear_tf"],
        dtype=np.float32,
        strip_accents="unicode",
    )
    train_x = vectorizer.fit_transform(train_text)
    validation_x = vectorizer.transform(validation_text)

    candidates: list[dict[str, Any]] = []
    fitted: dict[float, SGDClassifier] = {}
    for alpha in config["classifier"]["alpha_grid"]:
        model = SGDClassifier(
            loss="log_loss",
            alpha=alpha,
            class_weight="balanced",
            average=config["classifier"]["average"],
            max_iter=config["classifier"]["max_iter"],
            tol=config["classifier"]["tol"],
            random_state=config["classifier"]["random_state"],
            n_jobs=-1,
        )
        model.fit(train_x, train_y)
        validation_scores = model.predict_proba(validation_x)[:, 1]
        candidate = {
            "alpha": alpha,
            "validation_auprc": float(average_precision_score(validation_y, validation_scores)),
            "validation_auroc": float(roc_auc_score(validation_y, validation_scores)),
            "epochs": int(model.n_iter_),
        }
        candidates.append(candidate)
        fitted[alpha] = model

    winner = max(candidates, key=lambda item: (item["validation_auprc"], item["alpha"]))
    model = fitted[winner["alpha"]]
    validation_scores = model.predict_proba(validation_x)[:, 1]
    fpr, tpr, thresholds = roc_curve(validation_y, validation_scores)
    eligible = np.flatnonzero(fpr <= 0.01)
    threshold_index = eligible[np.argmax(tpr[eligible])]
    operational_threshold = float(thresholds[threshold_index])

    del train_x, validation_x, train_text, validation_text
    test_x = vectorizer.transform(test_text)
    test_scores = model.predict_proba(test_x)[:, 1]

    results = {
        "experiment_id": config["experiment_id"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "data": {
            "train_rows": int(len(train_y)),
            "validation_rows": int(len(validation_y)),
            "test_rows": int(len(test_y)),
            "cross_split_overlap": overlaps,
        },
        "features": {
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
            "test_matrix_shape": list(test_x.shape),
        },
        "selection_candidates": candidates,
        "winner": winner,
        "validation_operating_point": {
            "target_max_fpr": 0.01,
            "selected_threshold": operational_threshold,
            "achieved_fpr": float(fpr[threshold_index]),
            "achieved_tpr": float(tpr[threshold_index]),
        },
        "test_at_0_5": metrics(test_y, test_scores, 0.5, config),
        "test_at_validation_operating_threshold": metrics(test_y, test_scores, operational_threshold, config),
        "runtime": {
            "elapsed_seconds": float(time.perf_counter() - started),
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
    }
    (HERE / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (HERE / "environment.txt").write_text(
        "\n".join(
            [
                f"python={sys.version.split()[0]}",
                f"platform={platform.platform()}",
                f"numpy={np.__version__}",
                f"pandas={pd.__version__}",
                f"scipy={scipy.__version__}",
                f"scikit-learn={sklearn.__version__}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
