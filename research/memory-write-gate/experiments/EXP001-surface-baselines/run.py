#!/usr/bin/env python3
"""Run preregistered Memory Write Gate surface baselines."""

from __future__ import annotations

import hashlib
import json
import platform
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = (
    ROOT
    / "data"
    / "raw"
    / "huggingface"
    / "memory-write-gate-20k"
    / "data"
    / "memory_write_gate_20000.jsonl"
)
CONFIG = HERE / "config.json"
SEED = 20261017
BOOTSTRAPS = 2000
EXPECTED_ROWS = {"train": 14_200, "validation": 3_200, "test": 2_600}
EXPECTED_SEEDS = {"train": 142, "validation": 32, "test": 26}
KEYWORD_PATTERN = re.compile(
    r"\b(?:remember|memorize|save this|keep in mind|from now on|always use|don't forget)\b"
    r"|(?:記住|记住|請記低|请记下|保存這|保存这|以後都|以后都|從今以後|从今以后)",
    flags=re.IGNORECASE,
)


def sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_calibration_error(y: np.ndarray, scores: np.ndarray) -> float:
    """Calculate equal-width 10-bin expected calibration error."""
    bins = np.minimum((scores * 10).astype(int), 9)
    counts = np.bincount(bins, minlength=10)
    positives = np.bincount(bins, weights=y, minlength=10)
    score_sums = np.bincount(bins, weights=scores, minlength=10)
    occupied = counts > 0
    accuracy = np.divide(positives, counts, out=np.zeros(10), where=occupied)
    confidence = np.divide(score_sums, counts, out=np.zeros(10), where=occupied)
    return float(np.sum(counts[occupied] * np.abs(accuracy[occupied] - confidence[occupied])) / len(y))


def choose_safe_threshold(y: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    """Choose the lowest validation threshold meeting the false-save target."""
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


def point_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
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
    """Calculate vectorized weighted AUROC and average precision for all bootstraps."""
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
    auroc = np.sum(
        positive_by_score * (negative_before + 0.5 * negative_by_score), axis=1
    ) / (positive_total * negative_total)

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
    auprc = np.sum(
        np.divide(
            positive_desc,
            positive_desc.sum(axis=1, keepdims=True),
            out=np.zeros_like(positive_desc),
            where=positive_desc.sum(axis=1, keepdims=True) > 0,
        )
        * precision,
        axis=1,
    )
    return auroc, auprc


def interval(values: np.ndarray) -> list[float]:
    """Return a percentile 95% interval."""
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def clustered_intervals(
    y: np.ndarray,
    scores: np.ndarray,
    safe_threshold: float,
    weights: np.ndarray,
) -> dict[str, Any]:
    """Calculate vectorized seed-clustered confidence intervals."""
    auroc, auprc = weighted_rank_metrics(y, scores, weights)
    positive = y == 1
    negative = ~positive
    predicted = scores >= safe_threshold
    false_save = np.divide(
        weights @ (predicted & negative),
        weights @ negative,
        out=np.full(weights.shape[0], np.nan),
        where=(weights @ negative) > 0,
    )
    save_recall = np.divide(
        weights @ (predicted & positive),
        weights @ positive,
        out=np.full(weights.shape[0], np.nan),
        where=(weights @ positive) > 0,
    )
    return {
        "auprc_95ci": interval(auprc[np.isfinite(auprc)]),
        "auroc_95ci": interval(auroc[np.isfinite(auroc)]),
        "safe_false_save_rate_95ci": interval(false_save[np.isfinite(false_save)]),
        "safe_save_recall_95ci": interval(save_recall[np.isfinite(save_recall)]),
        "resamples": BOOTSTRAPS,
        "unit": "seed_id",
        "seed": SEED,
    }


def fit_tfidf(
    train_text: pd.Series,
    train_y: np.ndarray,
    validation_text: pd.Series,
    test_text: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the frozen multilingual word-plus-character TF-IDF baseline."""
    representation = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    min_df=2,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    x_train = representation.fit_transform(train_text)
    x_validation = representation.transform(validation_text)
    x_test = representation.transform(test_text)
    model = LogisticRegression(
        class_weight="balanced", max_iter=2_000, random_state=SEED, solver="liblinear"
    )
    model.fit(x_train, train_y)
    return model.predict_proba(x_validation)[:, 1], model.predict_proba(x_test)[:, 1]


def fit_length(
    train_text: pd.Series,
    train_y: np.ndarray,
    validation_text: pd.Series,
    test_text: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the character-count-only control."""
    x_train = train_text.str.len().to_numpy(dtype=float).reshape(-1, 1)
    x_validation = validation_text.str.len().to_numpy(dtype=float).reshape(-1, 1)
    x_test = test_text.str.len().to_numpy(dtype=float).reshape(-1, 1)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_validation_scaled = scaler.transform(x_validation)
    x_test_scaled = scaler.transform(x_test)
    model = LogisticRegression(
        class_weight="balanced", max_iter=2_000, random_state=SEED, solver="lbfgs"
    )
    model.fit(x_train_scaled, train_y)
    return (
        model.predict_proba(x_validation_scaled)[:, 1],
        model.predict_proba(x_test_scaled)[:, 1],
    )


def keyword_scores(text: pd.Series) -> np.ndarray:
    """Apply the preregistered explicit-memory-cue rule vectorially."""
    return text.str.contains(KEYWORD_PATTERN, na=False).to_numpy(dtype=float)


def normalize_text(text: pd.Series) -> pd.Series:
    """Normalize whitespace and case for exact-overlap checks."""
    return text.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def main() -> None:
    """Execute the frozen baseline protocol and consume the test split once."""
    assert sha256(RAW) == "a5e583d8aaa2a0eef049424baf70faf00a4a427bf45052dfe5a214c52c5b29a8"
    frame = pd.read_json(RAW, lines=True)
    assert frame.shape[0] == 20_000
    assert frame["text"].notna().all()
    assert set(frame["label"]) == {"SAVE", "SKIP"}
    assert (frame.groupby("seed_id")["split"].nunique() == 1).all()

    split_counts = frame.groupby("split").size().to_dict()
    seed_counts = frame.groupby("split")["seed_id"].nunique().to_dict()
    assert split_counts == EXPECTED_ROWS
    assert seed_counts == EXPECTED_SEEDS

    train = frame.loc[frame["split"].eq("train")].copy()
    validation = frame.loc[frame["split"].eq("validation")].copy()
    test = frame.loc[frame["split"].eq("test")].copy()
    train_y = train["label"].eq("SAVE").to_numpy(dtype=int)
    validation_y = validation["label"].eq("SAVE").to_numpy(dtype=int)
    test_y = test["label"].eq("SAVE").to_numpy(dtype=int)

    normalized_train = set(normalize_text(train["text"]))
    normalized_validation = set(normalize_text(validation["text"]))
    normalized_test = set(normalize_text(test["text"]))
    overlap = {
        "train_validation": len(normalized_train & normalized_validation),
        "train_test": len(normalized_train & normalized_test),
        "validation_test": len(normalized_validation & normalized_test),
    }
    assert not any(overlap.values())

    prior = float(np.mean(train_y))
    validation_length, test_length = fit_length(
        train["text"], train_y, validation["text"], test["text"]
    )
    validation_tfidf, test_tfidf = fit_tfidf(
        train["text"], train_y, validation["text"], test["text"]
    )
    validation_scores = {
        "label_prior": np.full(validation_y.shape, prior),
        "explicit_keyword": keyword_scores(validation["text"]),
        "length_only": validation_length,
        "tfidf_word_char": validation_tfidf,
    }
    test_scores = {
        "label_prior": np.full(test_y.shape, prior),
        "explicit_keyword": keyword_scores(test["text"]),
        "length_only": test_length,
        "tfidf_word_char": test_tfidf,
    }

    safe_thresholds = {
        name: choose_safe_threshold(validation_y, scores, 0.01)
        for name, scores in validation_scores.items()
    }
    weights = bootstrap_weights(test["seed_id"].astype(str).to_numpy())
    models: dict[str, Any] = {}
    for name in ("label_prior", "explicit_keyword", "length_only", "tfidf_word_char"):
        models[name] = {
            "validation": {
                "threshold_0_5": point_metrics(validation_y, validation_scores[name], 0.5),
                "safe_threshold": point_metrics(
                    validation_y, validation_scores[name], safe_thresholds[name]
                ),
            },
            "test": {
                "threshold_0_5": point_metrics(test_y, test_scores[name], 0.5),
                "safe_threshold": point_metrics(test_y, test_scores[name], safe_thresholds[name]),
                "clustered_intervals": clustered_intervals(
                    test_y, test_scores[name], safe_thresholds[name], weights
                ),
            },
        }

    result = {
        "experiment_id": "MEM-EXP001",
        "status": "COMPLETE_TEST_CONSUMED",
        "input_hashes": {"config": sha256(CONFIG), "raw": sha256(RAW)},
        "rows": split_counts,
        "seed_families": seed_counts,
        "label_counts": {
            split: frame.loc[frame["split"].eq(split), "label"].value_counts().sort_index().to_dict()
            for split in ("train", "validation", "test")
        },
        "normalized_exact_text_overlap": overlap,
        "models": models,
        "expect_then_check": {
            "expected": "surface text will be materially predictive under seed-family holdout",
            "observed_test_tfidf_auprc": models["tfidf_word_char"]["test"][
                "threshold_0_5"
            ]["auprc"],
            "matched": models["tfidf_word_char"]["test"]["threshold_0_5"][
                "auprc"
            ]
            > 0.5,
        },
        "claim_boundary": (
            "This is a synthetic silver source with 200 independent seed families. Test estimates "
            "describe this publisher's generator and do not establish real-user generalization."
        ),
    }
    (HERE / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        HERE / "scores.npz",
        validation_y=validation_y,
        test_y=test_y,
        validation_seed=validation["seed_id"].astype(str).to_numpy(),
        test_seed=test["seed_id"].astype(str).to_numpy(),
        **{f"validation_{name}": value for name, value in validation_scores.items()},
        **{f"test_{name}": value for name, value in test_scores.items()},
    )
    (HERE / "environment.txt").write_text(
        "\n".join(
            [
                f"python={platform.python_version()}",
                f"numpy={np.__version__}",
                f"pandas={pd.__version__}",
                f"scikit-learn={sklearn.__version__}",
                f"platform={platform.platform()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (HERE / "run.log").write_text(
        "MEM-EXP001 completed successfully. Official test is consumed.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
