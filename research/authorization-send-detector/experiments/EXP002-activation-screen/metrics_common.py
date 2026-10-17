"""Shared EXP002 binary metrics with reproducible uncertainty."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.stats import bootstrap
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, max_fpr: float) -> dict[str, float]:
    """Return maximum recall under a fixed false-approve budget."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    eligible = np.flatnonzero(fpr <= max_fpr)
    index = eligible[np.argmax(tpr[eligible])]
    return {
        "max_fpr": max_fpr,
        "recall": float(tpr[index]),
        "achieved_fpr": float(fpr[index]),
        "threshold": float(thresholds[index]),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    seed: int,
    resamples: int = 2000,
) -> list[float]:
    """Return a paired row-bootstrap percentile interval."""
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


def binary_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    seed: int,
    include_ci: bool = True,
) -> dict[str, Any]:
    """Compute authorization metrics where approve is the positive class."""
    predicted = (scores >= 0.5).astype(np.int8)
    result: dict[str, Any] = {
        "n": int(len(y_true)),
        "approve_n": int(y_true.sum()),
        "deny_n": int((y_true == 0).sum()),
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y_true, predicted)),
        "f1_at_0_5": float(f1_score(y_true, predicted)),
        "brier": float(brier_score_loss(y_true, scores)),
        "recall_at_5pct_false_approve": recall_at_fpr(y_true, scores, 0.05),
        "recall_at_1pct_false_approve": recall_at_fpr(y_true, scores, 0.01),
    }
    if include_ci:
        result["auroc_row_bootstrap_95ci"] = bootstrap_ci(
            y_true, scores, lambda y, s: float(roc_auc_score(y, s)), seed
        )
        result["auprc_row_bootstrap_95ci"] = bootstrap_ci(
            y_true,
            scores,
            lambda y, s: float(average_precision_score(y, s)),
            seed + 1,
        )
    return result
