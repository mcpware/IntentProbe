#!/usr/bin/env python3
"""Evaluate a human-trained activation head with leave-one-user-out predictions."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data" / "processed" / "exp003" / "lufy_majority.parquet"
FEATURES = HERE / "masked_mean_output11.npy"
CONFIG = HERE / "config.json"
RESULT = HERE / "result.json"
SCORES = HERE / "oof_scores.npz"
PROBE = HERE / "human_trained_probe.npz"
EXP003_ANALYSIS = HERE.parent / "EXP003-lufy-human-transfer" / "analyze.py"
SEED = 20261017


def sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_exp003_analysis():
    """Load the already-tested vectorized cluster-bootstrap helpers."""
    spec = importlib.util.spec_from_file_location("memory_exp003_analysis", EXP003_ANALYSIS)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def activation_pipeline() -> Pipeline:
    """Build the frozen activation-head recipe."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "head",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=2000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def tfidf_pipeline() -> Pipeline:
    """Build the matched word-plus-character surface baseline."""
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
    return Pipeline(
        [
            ("tfidf", representation),
            (
                "head",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=2000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def point_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """Compute the preregistered aggregate out-of-fold metrics."""
    return {
        "auprc": float(average_precision_score(y, score)),
        "auroc": float(roc_auc_score(y, score)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, score >= 0.5)),
    }


def main() -> None:
    """Generate grouped out-of-fold predictions, intervals, and final local head."""
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frame = pd.read_parquet(DATA)
    x_activation = np.load(FEATURES, mmap_mode="r")
    y = frame["label"].to_numpy(dtype=np.int8)
    groups = frame["user_name"].astype(str).to_numpy()
    assert x_activation.shape == (6470, 896)
    assert len(np.unique(groups)) == 17
    assert int(y.sum()) == 332

    splitter = LeaveOneGroupOut()
    activation_score = cross_val_predict(
        activation_pipeline(),
        x_activation,
        y,
        groups=groups,
        cv=splitter,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    tfidf_score = cross_val_predict(
        tfidf_pipeline(),
        frame["text"].astype(str).to_numpy(),
        y,
        groups=groups,
        cv=splitter,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    bootstrap = load_exp003_analysis()
    row_weights = bootstrap.group_bootstrap_row_weights(groups)
    activation_auprc, activation_auroc = bootstrap.weighted_rank_bootstrap(
        y, activation_score, row_weights
    )
    tfidf_auprc, tfidf_auroc = bootstrap.weighted_rank_bootstrap(y, tfidf_score, row_weights)
    paired_auprc = activation_auprc - tfidf_auprc
    paired_auroc = activation_auroc - tfidf_auroc

    activation_result = point_metrics(y, activation_score)
    activation_result.update(
        auprc_user_bootstrap_95ci=bootstrap.percentile(activation_auprc),
        auroc_user_bootstrap_95ci=bootstrap.percentile(activation_auroc),
    )
    tfidf_result = point_metrics(y, tfidf_score)
    tfidf_result.update(
        auprc_user_bootstrap_95ci=bootstrap.percentile(tfidf_auprc),
        auroc_user_bootstrap_95ci=bootstrap.percentile(tfidf_auroc),
    )

    final_model = activation_pipeline().fit(x_activation, y)
    scaler = final_model.named_steps["scale"]
    head = final_model.named_steps["head"]
    np.savez_compressed(
        PROBE,
        coefficient=head.coef_.astype(np.float64),
        intercept=head.intercept_.astype(np.float64),
        train_means=scaler.mean_.astype(np.float64),
        train_scales=scaler.scale_.astype(np.float64),
        classes=head.classes_.astype(np.int8),
        hidden_state_output=np.array(11, dtype=np.int64),
        pooling=np.array("masked_mean"),
    )
    np.savez_compressed(
        SCORES,
        label=y,
        activation=activation_score,
        tfidf_word_char=tfidf_score,
        user_name=groups,
        system=frame["system"].astype(str).to_numpy(),
    )

    result = {
        "experiment_id": "MEM-EXP004",
        "status": "COMPLETE_HUMAN_TRAINED_GROUPED_EVALUATION",
        "design": config["evaluation"]["design"],
        "rows": len(frame),
        "users": int(frame["user_name"].nunique()),
        "positive_rows": int(y.sum()),
        "positive_prevalence": float(y.mean()),
        "representation": config["backbone"],
        "models": {
            "activation": activation_result,
            "tfidf_word_char": tfidf_result,
        },
        "paired_activation_minus_tfidf": {
            "auprc_difference": activation_result["auprc"] - tfidf_result["auprc"],
            "auprc_difference_user_bootstrap_95ci": bootstrap.percentile(paired_auprc),
            "auroc_difference": activation_result["auroc"] - tfidf_result["auroc"],
            "auroc_difference_user_bootstrap_95ci": bootstrap.percentile(paired_auroc),
            "resamples": 2000,
            "unit": "user_name",
            "seed": SEED,
        },
        "artifacts": {
            "features_sha256": sha256(FEATURES),
            "oof_scores_sha256": sha256(SCORES),
            "human_trained_probe_sha256": sha256(PROBE),
        },
        "selection_boundary": config["selection_boundary"],
        "release_boundary": config["release_boundary"],
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
