#!/usr/bin/env python3
"""Select a human-memory readout on development users and open one held-out test."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import FeatureUnion


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = HERE / "config.json"
DATA = ROOT / "data" / "processed" / "exp003" / "lufy_majority.parquet"
CACHE = HERE / "cache"
EXP003_ANALYSIS = HERE.parent / "EXP003-lufy-human-transfer" / "analyze.py"
TABLE = HERE / "candidate_table.csv"
RESULT = HERE / "result.json"
SCORES = HERE / "test_scores.npz"
PROBE = HERE / "human_trained_probe.npz"
SEED = 20261017


def sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def anonymized_user_id(name: str) -> str:
    """Return the frozen non-reversible grouping identifier."""
    return hashlib.sha256(f"intentprobe-lufy-v1:{name}".encode()).hexdigest()[:12]


def load_bootstrap_module():
    """Load the tested vectorized user-bootstrap implementation."""
    spec = importlib.util.spec_from_file_location("memory_exp003_analysis", EXP003_ANALYSIS)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CachedActivationProbe(ClassifierMixin, BaseEstimator):
    """Fit one standardized adjacent-layer readout from memmapped caches."""

    def __init__(
        self,
        cache_dir: str,
        pooling: str = "masked_mean",
        start_layer: int = 0,
        end_layer: int = 0,
        C: float = 1.0,
        max_iter: int = 2000,
        random_state: int = SEED,
    ) -> None:
        self.cache_dir = cache_dir
        self.pooling = pooling
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

    def _raw(self, row_indices: np.ndarray) -> np.ndarray:
        rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
        cache = np.load(Path(self.cache_dir) / f"{self.pooling}.npy", mmap_mode="r")
        return cache[rows, self.start_layer : self.end_layer + 1].astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray):
        cube = self._raw(X)
        self.train_means_ = cube.mean(axis=0)
        self.train_scales_ = cube.std(axis=0)
        self.train_scales_ = np.where(self.train_scales_ < 1e-6, 1.0, self.train_scales_)
        features = ((cube - self.train_means_) / self.train_scales_).mean(axis=1)
        self.head_ = LogisticRegression(
            C=self.C,
            class_weight="balanced",
            solver="liblinear",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.head_.fit(features, y)
        self.classes_ = self.head_.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        cube = self._raw(X)
        features = ((cube - self.train_means_) / self.train_scales_).mean(axis=1)
        return self.head_.predict_proba(features)


def auprc_scorer(estimator: CachedActivationProbe, X: np.ndarray, y: np.ndarray) -> float:
    """Score one group-held-out validation fold."""
    return float(average_precision_score(y, estimator.predict_proba(X)[:, 1]))


def parameter_grid() -> list[dict[str, list[Any]]]:
    """Enumerate the frozen 216-candidate readout family."""
    candidates: list[dict[str, list[Any]]] = []
    for pooling in ("masked_mean", "last_nonpadding", "masked_max"):
        for width in (1, 2, 3):
            for start in range(25 - width + 1):
                candidates.append(
                    {
                        "pooling": [pooling],
                        "start_layer": [start],
                        "end_layer": [start + width - 1],
                    }
                )
    assert len(candidates) == 216
    return candidates


def fit_tfidf(
    train_text: pd.Series, train_y: np.ndarray, test_text: pd.Series
) -> np.ndarray:
    """Fit the matched surface baseline on development users only."""
    representation = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True
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
    train_features = representation.fit_transform(train_text)
    test_features = representation.transform(test_text)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=SEED,
    )
    model.fit(train_features, train_y)
    return model.predict_proba(test_features)[:, 1]


def point_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """Compute the frozen test metrics."""
    return {
        "auprc": float(average_precision_score(y, score)),
        "auroc": float(roc_auc_score(y, score)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, score >= 0.5)),
    }


def main() -> None:
    """Screen on 12 users, then evaluate the winner once on five held-out users."""
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frame = pd.read_parquet(DATA)
    y = frame["label"].to_numpy(dtype=np.int8)
    user_ids = frame["user_name"].astype(str).map(anonymized_user_id).to_numpy()
    test_ids = np.asarray(config["split"]["heldout_test_user_ids"])
    test_mask = np.isin(user_ids, test_ids)
    development_indices = np.flatnonzero(~test_mask)
    test_indices = np.flatnonzero(test_mask)
    development_y = y[development_indices]
    test_y = y[test_indices]
    development_groups = user_ids[development_indices]
    test_groups = user_ids[test_indices]
    assert len(np.unique(development_groups)) == 12
    assert len(np.unique(test_groups)) == 5
    assert set(np.unique(development_groups)).isdisjoint(np.unique(test_groups))

    started = time.perf_counter()
    estimator = CachedActivationProbe(cache_dir=str(CACHE))
    search = GridSearchCV(
        estimator=estimator,
        param_grid=parameter_grid(),
        scoring=auprc_scorer,
        cv=GroupKFold(n_splits=4),
        refit=True,
        n_jobs=4,
        pre_dispatch=4,
        error_score="raise",
        return_train_score=False,
        verbose=1,
    )
    search.fit(
        development_indices[:, None],
        development_y,
        groups=development_groups,
    )
    activation_score = search.best_estimator_.predict_proba(test_indices[:, None])[:, 1]
    tfidf_score = fit_tfidf(
        frame.loc[development_indices, "text"].astype(str),
        development_y,
        frame.loc[test_indices, "text"].astype(str),
    )

    table = pd.DataFrame(
        {
            "pooling": search.cv_results_["param_pooling"].astype(str),
            "start_layer": search.cv_results_["param_start_layer"].astype(int),
            "end_layer": search.cv_results_["param_end_layer"].astype(int),
            "mean_development_auprc": search.cv_results_["mean_test_score"],
            "std_development_auprc": search.cv_results_["std_test_score"],
            "rank_development_auprc": search.cv_results_["rank_test_score"],
        }
    ).sort_values(["rank_development_auprc", "pooling", "start_layer"])
    table.to_csv(TABLE, index=False)

    bootstrap = load_bootstrap_module()
    row_weights = bootstrap.group_bootstrap_row_weights(test_groups)
    activation_auprc, activation_auroc = bootstrap.weighted_rank_bootstrap(
        test_y, activation_score, row_weights
    )
    tfidf_auprc, tfidf_auroc = bootstrap.weighted_rank_bootstrap(
        test_y, tfidf_score, row_weights
    )
    activation_result = point_metrics(test_y, activation_score)
    activation_result.update(
        auprc_user_bootstrap_95ci=bootstrap.percentile(activation_auprc),
        auroc_user_bootstrap_95ci=bootstrap.percentile(activation_auroc),
    )
    tfidf_result = point_metrics(test_y, tfidf_score)
    tfidf_result.update(
        auprc_user_bootstrap_95ci=bootstrap.percentile(tfidf_auprc),
        auroc_user_bootstrap_95ci=bootstrap.percentile(tfidf_auroc),
    )

    best = search.best_params_
    final_estimator = CachedActivationProbe(cache_dir=str(CACHE), **best).fit(
        np.arange(len(frame))[:, None], y
    )
    np.savez_compressed(
        PROBE,
        coefficient=final_estimator.head_.coef_.astype(np.float64),
        intercept=final_estimator.head_.intercept_.astype(np.float64),
        train_means=final_estimator.train_means_.astype(np.float64),
        train_scales=final_estimator.train_scales_.astype(np.float64),
        classes=final_estimator.classes_.astype(np.int8),
        pooling=np.array(best["pooling"]),
        start_layer=np.array(best["start_layer"], dtype=np.int64),
        end_layer=np.array(best["end_layer"], dtype=np.int64),
    )
    np.savez_compressed(
        SCORES,
        label=test_y,
        activation=activation_score,
        tfidf_word_char=tfidf_score,
        user_id=np.asarray(test_groups, dtype="U12"),
    )

    result = {
        "experiment_id": "MEM-EXP005",
        "status": "COMPLETE_DEVELOPMENT_SCREEN_AND_ONE_SHOT_HUMAN_TEST",
        "candidate_count": len(table),
        "development": {
            "rows": len(development_indices),
            "users": int(len(np.unique(development_groups))),
            "positive_rows": int(development_y.sum()),
            "cv": "four-fold GroupKFold by anonymized user",
            "selection_metric": "AUPRC",
            "winner": {
                **best,
                "mean_development_auprc": float(search.best_score_),
            },
        },
        "heldout_test": {
            "rows": len(test_indices),
            "users": int(len(np.unique(test_groups))),
            "positive_rows": int(test_y.sum()),
            "positive_prevalence": float(test_y.mean()),
            "models": {
                "activation": activation_result,
                "tfidf_word_char": tfidf_result,
            },
            "paired_activation_minus_tfidf": {
                "auprc_difference": activation_result["auprc"] - tfidf_result["auprc"],
                "auprc_difference_user_bootstrap_95ci": bootstrap.percentile(
                    activation_auprc - tfidf_auprc
                ),
                "auroc_difference": activation_result["auroc"] - tfidf_result["auroc"],
                "auroc_difference_user_bootstrap_95ci": bootstrap.percentile(
                    activation_auroc - tfidf_auroc
                ),
                "resamples": 2000,
                "unit": "held-out user",
                "seed": SEED,
            },
        },
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": {
            "candidate_table_sha256": sha256(TABLE),
            "test_scores_sha256": sha256(SCORES),
            "human_trained_probe_sha256": sha256(PROBE),
        },
        "release_boundary": config["release_boundary"],
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
