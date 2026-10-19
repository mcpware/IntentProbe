#!/usr/bin/env python3
"""Select and test task-specific ToxicChat activation readouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import FeatureUnion


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parents[1]
CONFIG = HERE / "config.json"
DATA = BENCHMARK / "data" / "processed" / "toxicchat0124" / "human_unique.parquet"
CACHE = HERE / "cache"
SEED = 20261017
TASKS = ("toxicity", "jailbreaking")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidates() -> list[tuple[str, int, int]]:
    values = []
    for pooling in ("masked_mean", "last_nonpadding", "masked_max"):
        for width in (1, 2, 3):
            for start in range(25 - width + 1):
                values.append((pooling, start, start + width - 1))
    assert len(values) == 216
    return values


def features(
    cache: np.ndarray,
    rows: np.ndarray,
    start: int,
    end: int,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    cube = cache[rows, start : end + 1].astype(np.float32)
    return ((cube - means) / scales).mean(axis=1)


def score_candidate(
    pooling: str,
    start: int,
    end: int,
    train_rows: np.ndarray,
    train_y: np.ndarray,
    validation_rows: np.ndarray,
    validation_y: np.ndarray,
) -> dict[str, object]:
    cache = np.load(CACHE / f"{pooling}.npy", mmap_mode="r")
    train_cube = cache[train_rows, start : end + 1].astype(np.float32)
    means = train_cube.mean(axis=0)
    scales = np.where(train_cube.std(axis=0) < 1e-6, 1.0, train_cube.std(axis=0))
    train_x = ((train_cube - means) / scales).mean(axis=1)
    validation_x = features(cache, validation_rows, start, end, means, scales)
    head = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=SEED,
    )
    head.fit(train_x, train_y)
    score = head.predict_proba(validation_x)[:, 1]
    return {
        "pooling": pooling,
        "start_layer": start,
        "end_layer": end,
        "width": end - start + 1,
        "normalized_depth": ((start + end) / 2) / 24,
        "validation_auprc": float(average_precision_score(validation_y, score)),
        "validation_auroc": float(roc_auc_score(validation_y, score)),
        "validation_balanced_accuracy_at_0_5": float(
            balanced_accuracy_score(validation_y, score >= 0.5)
        ),
    }


def fit_selected(
    pooling: str,
    start: int,
    end: int,
    fit_rows: np.ndarray,
    fit_y: np.ndarray,
    test_rows: np.ndarray,
) -> tuple[np.ndarray, LogisticRegression, np.ndarray, np.ndarray]:
    cache = np.load(CACHE / f"{pooling}.npy", mmap_mode="r")
    fit_cube = cache[fit_rows, start : end + 1].astype(np.float32)
    means = fit_cube.mean(axis=0)
    raw_scales = fit_cube.std(axis=0)
    scales = np.where(raw_scales < 1e-6, 1.0, raw_scales)
    fit_x = ((fit_cube - means) / scales).mean(axis=1)
    test_x = features(cache, test_rows, start, end, means, scales)
    head = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=SEED,
    )
    head.fit(fit_x, fit_y)
    return head.predict_proba(test_x)[:, 1], head, means, scales


def fit_tfidf(train_text: pd.Series, train_y: np.ndarray, test_text: pd.Series) -> np.ndarray:
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
                    ngram_range=(2, 5), min_df=2, max_features=100_000, sublinear_tf=True
                ),
            ),
        ]
    )
    train_x = representation.fit_transform(train_text)
    test_x = representation.transform(test_text)
    head = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=SEED,
    )
    head.fit(train_x, train_y)
    return head.predict_proba(test_x)[:, 1]


def expected_calibration_error(y: np.ndarray, score: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        upper = edges[index + 1] if index < bins - 1 else edges[index + 1] + 1e-12
        mask = (score >= edges[index]) & (score < upper)
        if mask.any():
            result += float(mask.mean() * abs(y[mask].mean() - score[mask].mean()))
    return result


def point_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "auprc": float(average_precision_score(y, score)),
        "auroc": float(roc_auc_score(y, score)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, score >= 0.5)),
        "macro_f1_at_0_5": float(f1_score(y, score >= 0.5, average="macro")),
        "brier_score": float(brier_score_loss(y, score)),
        "log_loss": float(log_loss(y, score)),
        "ece_10_equal_width": expected_calibration_error(y, score),
    }


def paired_bootstrap(y: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    rng = np.random.default_rng(SEED)
    auroc_diff = []
    auprc_diff = []
    attempts = 0
    while len(auroc_diff) < 2000:
        attempts += 1
        idx = rng.integers(0, len(y), size=len(y))
        if np.unique(y[idx]).size < 2:
            continue
        auroc_diff.append(roc_auc_score(y[idx], left[idx]) - roc_auc_score(y[idx], right[idx]))
        auprc_diff.append(
            average_precision_score(y[idx], left[idx])
            - average_precision_score(y[idx], right[idx])
        )
    return {
        "auroc_difference_95ci": np.percentile(auroc_diff, [2.5, 97.5]).tolist(),
        "auprc_difference_95ci": np.percentile(auprc_diff, [2.5, 97.5]).tolist(),
        "attempts": attempts,
    }


def run_task(task: str, frame: pd.DataFrame, config: dict[str, object]) -> dict[str, object]:
    train_rows = np.flatnonzero(frame["split"].to_numpy() == "train")
    validation_rows = np.flatnonzero(frame["split"].to_numpy() == "validation")
    test_rows = np.flatnonzero(frame["split"].to_numpy() == "test")
    y = frame[task].to_numpy(dtype=np.int8)
    started = time.perf_counter()

    table = pd.DataFrame(
        Parallel(n_jobs=4, verbose=10)(
            delayed(score_candidate)(
                pooling,
                start,
                end,
                train_rows,
                y[train_rows],
                validation_rows,
                y[validation_rows],
            )
            for pooling, start, end in candidates()
        )
    ).sort_values(
        ["validation_auprc", "validation_auroc", "width", "pooling", "start_layer"],
        ascending=[False, False, True, True, True],
    )
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table_path = HERE / f"{task}_candidate_table.csv"
    table.to_csv(table_path, index=False)
    winner = table.iloc[0].to_dict()

    fit_rows = np.concatenate([train_rows, validation_rows])
    activation_score, head, means, scales = fit_selected(
        str(winner["pooling"]),
        int(winner["start_layer"]),
        int(winner["end_layer"]),
        fit_rows,
        y[fit_rows],
        test_rows,
    )
    tfidf_score = fit_tfidf(
        frame.loc[fit_rows, "text"].astype(str),
        y[fit_rows],
        frame.loc[test_rows, "text"].astype(str),
    )
    activation_metrics = point_metrics(y[test_rows], activation_score)
    tfidf_metrics = point_metrics(y[test_rows], tfidf_score)
    paired = paired_bootstrap(y[test_rows], activation_score, tfidf_score)
    paired.update(
        {
            "auprc_difference": activation_metrics["auprc"] - tfidf_metrics["auprc"],
            "auroc_difference": activation_metrics["auroc"] - tfidf_metrics["auroc"],
            "resamples": 2000,
            "unit": "official test prompt",
            "seed": SEED,
        }
    )

    probe_path = HERE / f"{task}_probe.npz"
    scores_path = HERE / f"{task}_test_scores.npz"
    np.savez_compressed(
        probe_path,
        coefficient=head.coef_.astype(np.float64),
        intercept=head.intercept_.astype(np.float64),
        train_means=means.astype(np.float64),
        train_scales=scales.astype(np.float64),
        classes=head.classes_.astype(np.int8),
        pooling=np.array(str(winner["pooling"])),
        start_layer=np.array(int(winner["start_layer"]), dtype=np.int64),
        end_layer=np.array(int(winner["end_layer"]), dtype=np.int64),
    )
    np.savez_compressed(
        scores_path,
        label=y[test_rows],
        activation=activation_score,
        tfidf_word_char=tfidf_score,
        conv_id=frame.loc[test_rows, "conv_id"].to_numpy(dtype=str),
    )
    result = {
        "experiment_id": config["experiment_id"],
        "task": task,
        "status": "COMPLETE_VALIDATION_SCREEN_AND_ONE_SHOT_OFFICIAL_TEST",
        "candidate_count": len(table),
        "selection": {
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "validation_positive_rows": int(y[validation_rows].sum()),
            "metric": "AUPRC",
            "winner": winner,
        },
        "heldout_test": {
            "rows": len(test_rows),
            "positive_rows": int(y[test_rows].sum()),
            "positive_prevalence": float(y[test_rows].mean()),
            "models": {
                "activation": activation_metrics,
                "tfidf_word_char": tfidf_metrics,
            },
            "paired_activation_minus_tfidf": paired,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": {
            "candidate_table_sha256": sha256(table_path),
            "test_scores_sha256": sha256(scores_path),
            "probe_sha256": sha256(probe_path),
        },
        "uncertainty_limitation": config["uncertainty"]["limitation"],
        "research_scope": config["research_scope"],
    }
    result_path = HERE / f"{task}_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frame = pd.read_parquet(DATA)
    selected = TASKS if args.task == "all" else (args.task,)
    results = [run_task(task, frame, config) for task in selected]
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
