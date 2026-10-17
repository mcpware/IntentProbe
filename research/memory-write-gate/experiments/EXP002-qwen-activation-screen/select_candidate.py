#!/usr/bin/env python3
"""Freeze the MEM-EXP002 winner and compare it with the surface baseline."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = HERE / "config.json"
DATA = ROOT / "data" / "processed" / "exp002"
CACHE = ROOT / ".cache" / "exp002" / "Qwen__Qwen2.5-0.5B"
EXP001 = ROOT / "experiments" / "EXP001-surface-baselines"
POOLING_RESULTS = HERE / "pooling_results"
POOLINGS = ("masked_mean", "last_nonpadding", "masked_max")
SEED = 20261017
BOOTSTRAPS = 2_000


def sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_safe_threshold(y: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    """Choose the lowest threshold meeting the false-save target."""
    negative_scores = scores[y == 0]
    unique, counts = np.unique(negative_scores, return_counts=True)
    descending = np.argsort(unique)[::-1]
    unique_desc = unique[descending]
    cumulative = np.cumsum(counts[descending])
    allowed = int(np.floor(max_fpr * negative_scores.size))
    eligible = np.flatnonzero(cumulative <= allowed)
    if eligible.size == 0:
        return float(np.nextafter(unique_desc[0], np.inf))
    return float(unique_desc[eligible[-1]])


def expected_calibration_error(y: np.ndarray, scores: np.ndarray) -> float:
    """Calculate equal-width 10-bin expected calibration error."""
    bins = np.minimum((scores * 10).astype(int), 9)
    counts = np.bincount(bins, minlength=10)
    positives = np.bincount(bins, weights=y, minlength=10)
    score_sums = np.bincount(bins, weights=scores, minlength=10)
    occupied = counts > 0
    accuracy = np.divide(positives, counts, out=np.zeros(10), where=occupied)
    confidence = np.divide(score_sums, counts, out=np.zeros(10), where=occupied)
    return float(
        np.sum(counts[occupied] * np.abs(accuracy[occupied] - confidence[occupied]))
        / len(y)
    )


def point_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Calculate discrimination, decision, and calibration metrics."""
    predicted = scores >= threshold
    positive = y == 1
    negative = ~positive
    return {
        "auprc": float(average_precision_score(y, scores)),
        "auroc": float(roc_auc_score(y, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro")),
        "false_save_rate": float(np.mean(predicted[negative])),
        "false_skip_rate": float(np.mean(~predicted[positive])),
        "save_recall": float(np.mean(predicted[positive])),
        "brier_score": float(brier_score_loss(y, scores)),
        "expected_calibration_error_10bin": expected_calibration_error(y, scores),
        "threshold": float(threshold),
    }


def bootstrap_weights(groups: np.ndarray) -> np.ndarray:
    """Create vectorized seed-cluster bootstrap row weights."""
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(SEED)
    group_counts = rng.multinomial(
        unique.size, np.full(unique.size, 1.0 / unique.size), size=BOOTSTRAPS
    )
    return group_counts[:, inverse].astype(np.float64, copy=False)


def weighted_rank_metrics(
    y: np.ndarray, scores: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate weighted AUROC and average precision for all bootstraps."""
    ascending = np.argsort(scores, kind="stable")
    y_ascending = y[ascending]
    scores_ascending = scores[ascending]
    starts_ascending = np.r_[0, np.flatnonzero(np.diff(scores_ascending)) + 1]
    weights_ascending = weights[:, ascending]
    positive_by_score = np.add.reduceat(
        weights_ascending * y_ascending, starts_ascending, axis=1
    )
    negative_by_score = np.add.reduceat(
        weights_ascending * (1 - y_ascending), starts_ascending, axis=1
    )
    negative_before = np.cumsum(negative_by_score, axis=1) - negative_by_score
    positive_total = positive_by_score.sum(axis=1)
    negative_total = negative_by_score.sum(axis=1)
    auroc = np.divide(
        np.sum(
            positive_by_score * (negative_before + 0.5 * negative_by_score), axis=1
        ),
        positive_total * negative_total,
        out=np.full(weights.shape[0], np.nan),
        where=(positive_total * negative_total) > 0,
    )

    descending = ascending[::-1]
    y_descending = y[descending]
    scores_descending = scores[descending]
    starts_descending = np.r_[0, np.flatnonzero(np.diff(scores_descending)) + 1]
    weights_descending = weights[:, descending]
    positive_desc = np.add.reduceat(
        weights_descending * y_descending, starts_descending, axis=1
    )
    negative_desc = np.add.reduceat(
        weights_descending * (1 - y_descending), starts_descending, axis=1
    )
    cumulative_positive = np.cumsum(positive_desc, axis=1)
    cumulative_total = cumulative_positive + np.cumsum(negative_desc, axis=1)
    precision = np.divide(
        cumulative_positive,
        cumulative_total,
        out=np.zeros_like(cumulative_positive),
        where=cumulative_total > 0,
    )
    positive_sum = positive_desc.sum(axis=1, keepdims=True)
    auprc = np.sum(
        np.divide(
            positive_desc,
            positive_sum,
            out=np.zeros_like(positive_desc),
            where=positive_sum > 0,
        )
        * precision,
        axis=1,
    )
    auprc[positive_sum[:, 0] == 0] = np.nan
    return auroc, auprc


def interval(values: np.ndarray) -> list[float]:
    """Return a finite percentile 95% interval."""
    finite = values[np.isfinite(values)]
    return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]


def safe_rate_intervals(
    y: np.ndarray, scores: np.ndarray, threshold: float, weights: np.ndarray
) -> dict[str, list[float]]:
    """Return seed-clustered intervals for the safety operating point."""
    positive = y == 1
    negative = ~positive
    predicted = scores >= threshold
    negative_denominator = weights @ negative
    positive_denominator = weights @ positive
    false_save = np.divide(
        weights @ (predicted & negative),
        negative_denominator,
        out=np.full(weights.shape[0], np.nan),
        where=negative_denominator > 0,
    )
    save_recall = np.divide(
        weights @ (predicted & positive),
        positive_denominator,
        out=np.full(weights.shape[0], np.nan),
        where=positive_denominator > 0,
    )
    return {
        "false_save_rate_95ci": interval(false_save),
        "save_recall_95ci": interval(save_recall),
    }


def fit_selected(
    winner: dict[str, Any], config: dict[str, Any]
) -> tuple[LogisticRegression, np.ndarray, np.ndarray, np.ndarray]:
    """Refit the frozen winner and return its preprocessing and validation scores."""
    pooling = winner["pooling"]
    start = winner["start_layer"]
    end = winner["end_layer"]
    train = np.load(CACHE / f"train_{pooling}.npy", mmap_mode="r")[:, start : end + 1]
    validation = np.load(CACHE / f"validation_{pooling}.npy", mmap_mode="r")[
        :, start : end + 1
    ]
    train_float = train.astype(np.float32)
    validation_float = validation.astype(np.float32)
    means = train_float.mean(axis=0)
    scales = train_float.std(axis=0)
    scales = np.where(scales < 1e-6, 1.0, scales)
    train_x = ((train_float - means) / scales).mean(axis=1)
    validation_x = ((validation_float - means) / scales).mean(axis=1)
    train_y = pd.read_parquet(DATA / "train.parquet", columns=["target"])[
        "target"
    ].to_numpy(dtype=np.int8)
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
    return model, means, scales, scores


def development_slices(
    frame: pd.DataFrame, activation: np.ndarray, tfidf: np.ndarray
) -> dict[str, Any]:
    """Report exploratory mixed-label and category decision slices."""
    result: dict[str, Any] = {}
    y_all = frame["target"].to_numpy(dtype=np.int8)
    for column in ("language", "difficulty"):
        result[column] = {}
        for value, indices in frame.groupby(column, sort=True).indices.items():
            index = np.asarray(indices)
            y = y_all[index]
            if np.unique(y).size != 2:
                continue
            result[column][str(value)] = {
                "rows": int(index.size),
                "seed_families": int(frame.iloc[index]["seed_id"].nunique()),
                "positive_rate": float(y.mean()),
                "activation_auprc": float(average_precision_score(y, activation[index])),
                "activation_auroc": float(roc_auc_score(y, activation[index])),
                "tfidf_auprc": float(average_precision_score(y, tfidf[index])),
                "tfidf_auroc": float(roc_auc_score(y, tfidf[index])),
            }
    result["category"] = {}
    for value, indices in frame.groupby("category", sort=True).indices.items():
        index = np.asarray(indices)
        y = y_all[index]
        activation_positive = activation[index] >= 0.5
        tfidf_positive = tfidf[index] >= 0.5
        category: dict[str, Any] = {
            "rows": int(index.size),
            "seed_families": int(frame.iloc[index]["seed_id"].nunique()),
            "target": "SAVE" if np.all(y == 1) else "SKIP" if np.all(y == 0) else "MIXED",
            "activation_mean_score": float(activation[index].mean()),
            "tfidf_mean_score": float(tfidf[index].mean()),
        }
        if np.all(y == 1):
            category.update(
                decision_metric="save_recall_at_0_5",
                activation_decision_rate=float(activation_positive.mean()),
                tfidf_decision_rate=float(tfidf_positive.mean()),
            )
        elif np.all(y == 0):
            category.update(
                decision_metric="false_save_rate_at_0_5",
                activation_decision_rate=float(activation_positive.mean()),
                tfidf_decision_rate=float(tfidf_positive.mean()),
            )
        else:
            category.update(
                decision_metric="balanced_accuracy_at_0_5",
                activation_decision_rate=float(
                    balanced_accuracy_score(y, activation_positive)
                ),
                tfidf_decision_rate=float(balanced_accuracy_score(y, tfidf_positive)),
            )
        result["category"][str(value)] = category
    return result


def main() -> None:
    """Select, refit, compare, and serialize the development-only winner."""
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    pooling_files: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    for pooling in POOLINGS:
        path = POOLING_RESULTS / f"{pooling}.json"
        pooling_result = json.loads(path.read_text(encoding="utf-8"))
        assert pooling_result["status"] == "COMPLETED"
        assert pooling_result["completed_candidates"] == 72
        assert len(pooling_result["candidates"]) == 72
        pooling_files[pooling] = pooling_result
        candidates.extend(
            {**candidate, "pooling": pooling}
            for candidate in pooling_result["candidates"]
        )
    assert len(candidates) == 216
    winner = max(candidates, key=lambda item: item["metrics"]["auprc"])
    per_pooling = {
        pooling: {
            **max(
                pooling_files[pooling]["candidates"],
                key=lambda item: item["metrics"]["auprc"],
            ),
            "pooling": pooling,
        }
        for pooling in POOLINGS
    }
    candidate_table = HERE / "candidate_table.csv"
    pd.DataFrame(
        [
            {
                "pooling": candidate["pooling"],
                "key": candidate["key"],
                "kind": candidate["kind"],
                "start_layer": candidate["start_layer"],
                "end_layer": candidate["end_layer"],
                "width": candidate["width"],
                "normalized_depth": candidate["normalized_depth"],
                **candidate["metrics"],
            }
            for candidate in candidates
        ]
    ).to_csv(candidate_table, index=False, float_format="%.17g")

    model, means, scales, activation = fit_selected(winner, config)
    validation = pd.read_parquet(DATA / "validation.parquet")
    y = validation["target"].to_numpy(dtype=np.int8)
    groups = validation["seed_id"].astype(str).to_numpy()
    with np.load(EXP001 / "scores.npz", allow_pickle=True) as baseline:
        assert np.array_equal(y, baseline["validation_y"])
        assert np.array_equal(groups, baseline["validation_seed"].astype(str))
        tfidf = baseline["validation_tfidf_word_char"].astype(np.float64)

    assert abs(average_precision_score(y, activation) - winner["metrics"]["auprc"]) < 1e-12
    safe_threshold = choose_safe_threshold(y, activation, 0.01)
    weights = bootstrap_weights(groups)
    activation_auroc, activation_auprc = weighted_rank_metrics(y, activation, weights)
    tfidf_auroc, tfidf_auprc = weighted_rank_metrics(y, tfidf, weights)
    paired_auprc = activation_auprc - tfidf_auprc
    paired_auroc = activation_auroc - tfidf_auroc

    artifact = HERE / "selected_probe.npz"
    np.savez_compressed(
        artifact,
        coefficient=model.coef_.astype(np.float64),
        intercept=model.intercept_.astype(np.float64),
        train_means=means.astype(np.float32),
        train_scales=scales.astype(np.float32),
        classes=model.classes_,
        start_layer=np.asarray(winner["start_layer"], dtype=np.int64),
        end_layer=np.asarray(winner["end_layer"], dtype=np.int64),
        pooling=np.asarray(winner["pooling"]),
        safe_threshold=np.asarray(safe_threshold, dtype=np.float64),
    )
    scores_path = HERE / "validation_scores.npz"
    np.savez_compressed(
        scores_path,
        y=y,
        seed_id=groups,
        activation=activation,
        tfidf_word_char=tfidf,
    )

    result = {
        "experiment_id": "MEM-EXP002",
        "status": "COMPLETE_DEVELOPMENT_SELECTION_TEST_NOT_TOUCHED",
        "candidate_search": {
            "poolings": list(POOLINGS),
            "candidates_per_pooling": 72,
            "total_candidates": 216,
            "per_pooling_winners": per_pooling,
            "global_winner": winner,
        },
        "development_evaluation": {
            "split": "validation",
            "rows": int(len(validation)),
            "seed_families": int(validation["seed_id"].nunique()),
            "selected_activation": {
                "threshold_0_5": point_metrics(y, activation, 0.5),
                "safe_threshold": point_metrics(y, activation, safe_threshold),
                "clustered_intervals": {
                    "auprc_95ci": interval(activation_auprc),
                    "auroc_95ci": interval(activation_auroc),
                    **safe_rate_intervals(y, activation, safe_threshold, weights),
                    "resamples": BOOTSTRAPS,
                    "unit": "seed_id",
                    "seed": SEED,
                },
            },
            "tfidf_word_char": {
                "threshold_0_5": point_metrics(y, tfidf, 0.5),
            },
            "paired_activation_minus_tfidf": {
                "auprc_difference": float(
                    average_precision_score(y, activation)
                    - average_precision_score(y, tfidf)
                ),
                "auprc_difference_95ci": interval(paired_auprc),
                "auroc_difference": float(
                    roc_auc_score(y, activation) - roc_auc_score(y, tfidf)
                ),
                "auroc_difference_95ci": interval(paired_auroc),
                "resamples": BOOTSTRAPS,
                "unit": "seed_id",
                "seed": SEED,
            },
            "exploratory_slices": development_slices(validation, activation, tfidf),
        },
        "artifacts": {
            "candidate_table": {
                "path": candidate_table.name,
                "sha256": sha256(candidate_table),
            },
            "selected_probe": {
                "path": artifact.name,
                "sha256": sha256(artifact),
            },
            "validation_scores": {
                "path": scores_path.name,
                "sha256": sha256(scores_path),
            },
        },
        "input_hashes": {
            "config": sha256(CONFIG),
            "train": sha256(DATA / "train.parquet"),
            "validation": sha256(DATA / "validation.parquet"),
            "exp001_scores": sha256(EXP001 / "scores.npz"),
            "pooling_results": {
                pooling: sha256(POOLING_RESULTS / f"{pooling}.json")
                for pooling in POOLINGS
            },
        },
        "selection_boundary": (
            "All 216 candidates were selected and evaluated on the same 32-seed validation split. "
            "Intervals and paired differences are development-only, post-selection descriptions, "
            "not confirmatory estimates. The previously consumed official test was not loaded."
        ),
        "claim_boundary": (
            "The source is synthetic silver data. These results establish an engineering candidate "
            "for a future independent human-written holdout; they do not establish deployment-grade "
            "memory authorization."
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
    }
    (HERE / "selection_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
