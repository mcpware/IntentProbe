#!/usr/bin/env python3
"""Checkpointed 216-readout multiclass screen with a matched TF-IDF baseline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import label_binarize

from run_qwen_binary_experiment import (
    candidates,
    load_experiment,
    selected_features,
    sha256,
)


def make_head(config: dict[str, object]) -> SGDClassifier:
    head = config["head"]
    return SGDClassifier(
        loss=str(head["loss"]),
        alpha=float(head["alpha"]),
        class_weight=str(head["class_weight"]),
        average=bool(head["average"]),
        max_iter=int(head["max_iter"]),
        tol=float(head["tol"]),
        random_state=int(head["random_state"]),
    )


def safe_ovr_probability(head: SGDClassifier, features: np.ndarray) -> np.ndarray:
    """Return finite normalized OVR probabilities without sigmoid underflow."""
    decision = head.decision_function(features)
    probability = expit(np.clip(decision, -50.0, 50.0))
    denominator = probability.sum(axis=1, keepdims=True)
    assert bool(np.isfinite(probability).all())
    assert bool((denominator > 0).all())
    probability = probability / denominator
    assert bool(np.isfinite(probability).all())
    assert bool(np.allclose(probability.sum(axis=1), 1.0, atol=1e-12))
    return probability


def score_candidate(
    cache_dir: Path,
    config: dict[str, object],
    pooling: str,
    start: int,
    end: int,
    train_rows: np.ndarray,
    train_y: np.ndarray,
    validation_rows: np.ndarray,
    validation_y: np.ndarray,
    final_output: int,
    classes: np.ndarray,
) -> dict[str, object]:
    cache = np.load(cache_dir / f"{pooling}.npy", mmap_mode="r")
    train_cube = cache[train_rows, start : end + 1].astype(np.float32)
    means = train_cube.mean(axis=0)
    raw_scales = train_cube.std(axis=0)
    scales = np.where(raw_scales < 1e-6, 1.0, raw_scales)
    train_x = ((train_cube - means) / scales).mean(axis=1)
    validation_x = selected_features(cache, validation_rows, start, end, means, scales)
    head = make_head(config)
    head.fit(train_x, train_y)
    prediction = head.predict(validation_x)
    assert np.array_equal(head.classes_, classes)
    return {
        "pooling": pooling,
        "start_layer": start,
        "end_layer": end,
        "width": end - start + 1,
        "normalized_depth": ((start + end) / 2) / final_output,
        "validation_macro_f1": float(
            f1_score(validation_y, prediction, labels=classes, average="macro", zero_division=0)
        ),
        "validation_accuracy": float(accuracy_score(validation_y, prediction)),
        "validation_balanced_accuracy": float(
            balanced_accuracy_score(validation_y, prediction)
        ),
    }


def fit_activation(
    cache_dir: Path,
    config: dict[str, object],
    winner: dict[str, object],
    fit_rows: np.ndarray,
    fit_y: np.ndarray,
    test_rows: np.ndarray,
) -> tuple[np.ndarray, SGDClassifier, np.ndarray, np.ndarray]:
    pooling = str(winner["pooling"])
    start = int(winner["start_layer"])
    end = int(winner["end_layer"])
    cache = np.load(cache_dir / f"{pooling}.npy", mmap_mode="r")
    fit_cube = cache[fit_rows, start : end + 1].astype(np.float32)
    means = fit_cube.mean(axis=0)
    raw_scales = fit_cube.std(axis=0)
    scales = np.where(raw_scales < 1e-6, 1.0, raw_scales)
    fit_x = ((fit_cube - means) / scales).mean(axis=1)
    test_x = selected_features(cache, test_rows, start, end, means, scales)
    head = make_head(config)
    head.fit(fit_x, fit_y)
    return safe_ovr_probability(head, test_x), head, means, scales


def fit_tfidf(
    config: dict[str, object], train_text: pd.Series, train_y: np.ndarray, test_text: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
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
    head = make_head(config)
    head.fit(train_x, train_y)
    return safe_ovr_probability(head, test_x), head.classes_


def confidence_ece(y: np.ndarray, probability: np.ndarray, classes: np.ndarray, bins: int = 10) -> float:
    prediction_index = probability.argmax(axis=1)
    prediction = classes[prediction_index]
    confidence = probability.max(axis=1)
    correct = prediction == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        upper = edges[index + 1] if index < bins - 1 else edges[index + 1] + 1e-12
        mask = (confidence >= edges[index]) & (confidence < upper)
        if mask.any():
            result += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return result


def metrics(y: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    prediction = classes[probability.argmax(axis=1)]
    onehot = label_binarize(y, classes=classes)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(
            f1_score(y, prediction, labels=classes, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "top3_accuracy": float(top_k_accuracy_score(y, probability, k=3, labels=classes)),
        "macro_ovr_auroc": float(
            roc_auc_score(onehot, probability, average="macro", multi_class="ovr")
        ),
        "macro_ovr_auprc": float(average_precision_score(onehot, probability, average="macro")),
        "log_loss": float(log_loss(y, probability, labels=classes)),
        "multiclass_brier": float(np.mean(np.sum((onehot - probability) ** 2, axis=1))),
        "confidence_ece_10_equal_width": confidence_ece(y, probability, classes),
    }


def paired_bootstrap(
    y: np.ndarray,
    left_probability: np.ndarray,
    right_probability: np.ndarray,
    classes: np.ndarray,
    seed: int,
    resamples: int,
) -> dict[str, object]:
    left_prediction = classes[left_probability.argmax(axis=1)]
    right_prediction = classes[right_probability.argmax(axis=1)]
    rng = np.random.default_rng(seed)
    accuracy_difference = []
    macro_f1_difference = []
    for _ in range(resamples):
        index = rng.integers(0, len(y), size=len(y))
        accuracy_difference.append(
            accuracy_score(y[index], left_prediction[index])
            - accuracy_score(y[index], right_prediction[index])
        )
        macro_f1_difference.append(
            f1_score(
                y[index], left_prediction[index], labels=classes, average="macro", zero_division=0
            )
            - f1_score(
                y[index], right_prediction[index], labels=classes, average="macro", zero_division=0
            )
        )
    return {
        "accuracy_difference_95ci": np.percentile(accuracy_difference, [2.5, 97.5]).tolist(),
        "macro_f1_difference_95ci": np.percentile(macro_f1_difference, [2.5, 97.5]).tolist(),
        "resamples": resamples,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--screen-batch-size", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    experiment, config, _, frame = load_experiment(args.experiment_dir)
    cache_dir = experiment / "cache"
    task = config["task"]
    label_column = str(task["label_column"])
    text_column = str(config["dataset"]["text_column"])
    id_column = str(config["dataset"]["id_column"])
    seed = int(config["head"]["random_state"])
    outputs = int(config["backbone"]["hidden_state_outputs"])
    train_rows = np.flatnonzero(frame["split"].to_numpy() == "train")
    validation_rows = np.flatnonzero(frame["split"].to_numpy() == "validation")
    test_rows = np.flatnonzero(frame["split"].to_numpy() == "test")
    y = frame[label_column].to_numpy(dtype=np.int16)
    classes = np.unique(y)
    assert len(classes) == int(task["classes"])
    selection_rows, _ = train_test_split(
        train_rows,
        train_size=int(task["selection_train_row_budget"]),
        random_state=seed,
        stratify=y[train_rows],
    )
    selection_rows = np.sort(selection_rows)
    started = time.perf_counter()

    partial_path = experiment / "candidate_table_partial.csv"
    all_candidates = candidates(outputs)
    completed_rows: list[dict[str, object]] = []
    completed_keys: set[tuple[str, int, int]] = set()
    if args.resume and partial_path.exists():
        partial = pd.read_csv(partial_path)
        completed_rows = partial.to_dict("records")
        completed_keys = {
            (str(row["pooling"]), int(row["start_layer"]), int(row["end_layer"]))
            for row in completed_rows
        }
        print(f"resuming with {len(completed_keys)}/{len(all_candidates)} complete", flush=True)
    remaining = [item for item in all_candidates if item not in completed_keys]
    for offset in range(0, len(remaining), args.screen_batch_size):
        batch = remaining[offset : offset + args.screen_batch_size]
        completed_rows.extend(
            Parallel(n_jobs=args.jobs, verbose=10)(
                delayed(score_candidate)(
                    cache_dir,
                    config,
                    pooling,
                    start,
                    end,
                    selection_rows,
                    y[selection_rows],
                    validation_rows,
                    y[validation_rows],
                    outputs - 1,
                    classes,
                )
                for pooling, start, end in batch
            )
        )
        pd.DataFrame(completed_rows).to_csv(partial_path, index=False)
        print(f"screened {len(completed_rows)}/{len(all_candidates)} candidates", flush=True)
    assert len(completed_rows) == len(all_candidates)
    table = pd.DataFrame(completed_rows).sort_values(
        ["validation_macro_f1", "validation_accuracy", "width", "pooling", "start_layer"],
        ascending=[False, False, True, True, True],
    )
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table_path = experiment / "candidate_table.csv"
    table.to_csv(table_path, index=False)
    winner = table.iloc[0].to_dict()

    fit_rows = np.concatenate([train_rows, validation_rows])
    activation_probability, head, means, scales = fit_activation(
        cache_dir, config, winner, fit_rows, y[fit_rows], test_rows
    )
    tfidf_probability, tfidf_classes = fit_tfidf(
        config,
        frame.loc[fit_rows, text_column].astype(str),
        y[fit_rows],
        frame.loc[test_rows, text_column].astype(str),
    )
    assert np.array_equal(head.classes_, classes)
    assert np.array_equal(tfidf_classes, classes)
    activation_metrics = metrics(y[test_rows], activation_probability, classes)
    tfidf_metrics = metrics(y[test_rows], tfidf_probability, classes)
    paired = paired_bootstrap(
        y[test_rows],
        activation_probability,
        tfidf_probability,
        classes,
        seed,
        int(config["uncertainty"]["resamples"]),
    )
    paired.update(
        {
            "accuracy_difference": activation_metrics["accuracy"] - tfidf_metrics["accuracy"],
            "macro_f1_difference": activation_metrics["macro_f1"] - tfidf_metrics["macro_f1"],
            "unit": config["uncertainty"]["unit"],
        }
    )
    probe_path = experiment / "probe.npz"
    scores_path = experiment / "test_scores.npz"
    np.savez_compressed(
        probe_path,
        coefficient=head.coef_.astype(np.float64),
        intercept=head.intercept_.astype(np.float64),
        train_means=means.astype(np.float64),
        train_scales=scales.astype(np.float64),
        classes=classes,
        pooling=np.array(str(winner["pooling"])),
        start_layer=np.array(int(winner["start_layer"]), dtype=np.int64),
        end_layer=np.array(int(winner["end_layer"]), dtype=np.int64),
    )
    np.savez_compressed(
        scores_path,
        label=y[test_rows],
        activation_probability=activation_probability,
        tfidf_probability=tfidf_probability,
        classes=classes,
        example_id=frame.loc[test_rows, id_column].to_numpy(dtype=str),
    )
    result = {
        "experiment_id": config["experiment_id"],
        "task": task["id"],
        "status": "COMPLETE_VALIDATION_SCREEN_AND_ONE_SHOT_MULTICLASS_TEST",
        "classes": len(classes),
        "candidate_count": len(table),
        "selection": {
            "available_train_rows": len(train_rows),
            "screening_train_rows": len(selection_rows),
            "validation_rows": len(validation_rows),
            "metric": "macro_F1",
            "winner": winner,
        },
        "heldout_test": {
            "rows": len(test_rows),
            "models": {"activation": activation_metrics, "tfidf_word_char": tfidf_metrics},
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
    (experiment / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
