#!/usr/bin/env python3
"""Compare frozen activation and synthetic-trained TF-IDF on LUFY."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data" / "processed" / "exp003" / "lufy_majority.parquet"
SCORES = HERE / "activation_scores.npz"
SILVER = (
    ROOT
    / "data"
    / "raw"
    / "huggingface"
    / "memory-write-gate-20k"
    / "data"
    / "memory_write_gate_20000.jsonl"
)
BASELINE_SCRIPT = HERE.parent / "EXP001-surface-baselines" / "run.py"
PROBE = HERE.parent / "EXP002-qwen-activation-screen" / "selected_probe.npz"
RESULT = HERE / "result.json"
PAIRED_RESAMPLES = 2000
SEED = 20261017


def load_baseline_module():
    spec = importlib.util.spec_from_file_location("mem_exp001", BASELINE_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def group_bootstrap_row_weights(groups: np.ndarray) -> np.ndarray:
    """Return vectorized cluster-bootstrap multiplicities for every row."""
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(SEED)
    cluster_counts = rng.multinomial(
        len(unique),
        np.full(len(unique), 1.0 / len(unique)),
        size=PAIRED_RESAMPLES,
    )
    return cluster_counts[:, inverse]


def weighted_rank_bootstrap(
    y: np.ndarray, score: np.ndarray, row_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute weighted AP/AUROC for all bootstrap draws without row loops."""
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    weights = np.asarray(row_weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != len(y):
        raise ValueError("Bootstrap weights are not row-aligned")

    # Weighted AUROC as the positive-negative concordance probability. Tied scores
    # receive half credit, matching sklearn's binary roc_auc_score convention.
    ascending = np.argsort(score, kind="mergesort")
    asc_score = score[ascending]
    asc_y = y[ascending]
    asc_weight = weights[:, ascending]
    asc_positive = asc_weight * asc_y[None, :]
    asc_negative = asc_weight * (1 - asc_y)[None, :]
    asc_new_group = np.r_[True, asc_score[1:] != asc_score[:-1]]
    asc_group = np.cumsum(asc_new_group) - 1
    asc_starts = np.flatnonzero(asc_new_group)
    asc_ends = np.r_[asc_starts[1:] - 1, len(y) - 1]
    negative_prefix = np.concatenate(
        [np.zeros((len(weights), 1)), np.cumsum(asc_negative, axis=1)], axis=1
    )
    negative_before_group = negative_prefix[:, asc_starts]
    negative_in_group = (
        negative_prefix[:, asc_ends + 1] - negative_before_group
    )
    concordant = np.sum(
        asc_positive
        * (
            negative_before_group[:, asc_group]
            + 0.5 * negative_in_group[:, asc_group]
        ),
        axis=1,
    )
    positive_total = asc_positive.sum(axis=1)
    negative_total = asc_negative.sum(axis=1)
    auroc = np.divide(
        concordant,
        positive_total * negative_total,
        out=np.full(len(weights), np.nan),
        where=(positive_total > 0) & (negative_total > 0),
    )

    # Average precision uses precision after each complete tied-score group and
    # weights it by that group's share of all positives.
    descending = ascending[::-1]
    desc_score = score[descending]
    desc_y = y[descending]
    desc_weight = weights[:, descending]
    desc_positive = desc_weight * desc_y[None, :]
    desc_new_group = np.r_[True, desc_score[1:] != desc_score[:-1]]
    desc_group = np.cumsum(desc_new_group) - 1
    desc_starts = np.flatnonzero(desc_new_group)
    desc_ends = np.r_[desc_starts[1:] - 1, len(y) - 1]
    cumulative_positive = np.cumsum(desc_positive, axis=1)
    cumulative_total = np.cumsum(desc_weight, axis=1)
    precision_at_group_end = np.divide(
        cumulative_positive[:, desc_ends],
        cumulative_total[:, desc_ends],
        out=np.zeros((len(weights), len(desc_ends))),
        where=cumulative_total[:, desc_ends] > 0,
    )
    auprc = np.divide(
        np.sum(desc_positive * precision_at_group_end[:, desc_group], axis=1),
        positive_total,
        out=np.full(len(weights), np.nan),
        where=positive_total > 0,
    )
    return auprc, auroc


def rank_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "auprc": float(average_precision_score(y, score)),
        "auroc": float(roc_auc_score(y, score)),
    }


def percentile(values: np.ndarray) -> list[float]:
    array = np.asarray(values, dtype=float)
    return [float(np.nanquantile(array, 0.025)), float(np.nanquantile(array, 0.975))]


def evaluate_view(
    frame: pd.DataFrame,
    activation: np.ndarray,
    tfidf: np.ndarray,
    activation_safe_threshold: float,
    tfidf_safe_threshold: float,
) -> dict[str, Any]:
    y = frame["label"].to_numpy(dtype=int)
    groups = frame["user_name"].astype(str).to_numpy()
    bootstrap_weights = group_bootstrap_row_weights(groups)
    models: dict[str, Any] = {}
    bootstrap_metrics: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, score, safe_threshold in (
        ("activation", activation, activation_safe_threshold),
        ("tfidf_word_char", tfidf, tfidf_safe_threshold),
    ):
        point = rank_metrics(y, score)
        predicted = score >= 0.5
        safe_predicted = score >= safe_threshold
        point.update(
            balanced_accuracy_at_0_5=float(
                0.5
                * (
                    np.mean(predicted[y == 1])
                    + np.mean(~predicted[y == 0])
                )
            ),
            false_save_rate_at_0_5=float(np.mean(predicted[y == 0])),
            save_recall_at_0_5=float(np.mean(predicted[y == 1])),
            frozen_safe_threshold=float(safe_threshold),
            false_save_rate_at_frozen_safe_threshold=float(np.mean(safe_predicted[y == 0])),
            save_recall_at_frozen_safe_threshold=float(np.mean(safe_predicted[y == 1])),
        )
        boot_auprc, boot_auroc = weighted_rank_bootstrap(y, score, bootstrap_weights)
        point["auprc_user_bootstrap_95ci"] = percentile(boot_auprc)
        point["auroc_user_bootstrap_95ci"] = percentile(boot_auroc)
        models[name] = point
        bootstrap_metrics[name] = (boot_auprc, boot_auroc)

    activation_auprc, activation_auroc = bootstrap_metrics["activation"]
    tfidf_auprc, tfidf_auroc = bootstrap_metrics["tfidf_word_char"]
    auprc_differences = activation_auprc - tfidf_auprc
    auroc_differences = activation_auroc - tfidf_auroc

    return {
        "rows": len(frame),
        "users": int(frame["user_name"].nunique()),
        "positive_rows": int(y.sum()),
        "positive_prevalence": float(y.mean()),
        "models": models,
        "paired_activation_minus_tfidf": {
            "auprc_difference": models["activation"]["auprc"] - models["tfidf_word_char"]["auprc"],
            "auprc_difference_user_bootstrap_95ci": percentile(auprc_differences),
            "auroc_difference": models["activation"]["auroc"] - models["tfidf_word_char"]["auroc"],
            "auroc_difference_user_bootstrap_95ci": percentile(auroc_differences),
            "resamples": PAIRED_RESAMPLES,
            "unit": "user_name",
            "seed": SEED,
        },
    }


def main() -> None:
    baseline = load_baseline_module()
    frame = pd.read_parquet(DATA)
    with np.load(SCORES, allow_pickle=False) as archive:
        activation = archive["score"].astype(float)
        if not np.array_equal(archive["label"], frame["label"].to_numpy(dtype=np.int8)):
            raise ValueError("Activation scores are not row-aligned")

    silver = pd.read_json(SILVER, lines=True)
    train = silver[silver["split"].eq("train")]
    validation = silver[silver["split"].eq("validation")]
    train_y = train["label"].eq("SAVE").to_numpy(dtype=int)
    validation_y = validation["label"].eq("SAVE").to_numpy(dtype=int)
    validation_tfidf, lufy_tfidf = baseline.fit_tfidf(
        train["text"], train_y, validation["text"], frame["text"]
    )
    tfidf_safe_threshold = baseline.choose_safe_threshold(validation_y, validation_tfidf, 0.01)
    with np.load(PROBE, allow_pickle=False) as probe:
        activation_safe_threshold = float(probe["safe_threshold"])

    views: dict[str, Any] = {
        "all_majority_vote": evaluate_view(
            frame,
            activation,
            lufy_tfidf,
            activation_safe_threshold,
            tfidf_safe_threshold,
        )
    }
    unanimous_mask = frame["unanimous"].to_numpy(dtype=bool)
    views["unanimous_only"] = evaluate_view(
        frame.loc[unanimous_mask].reset_index(drop=True),
        activation[unanimous_mask],
        lufy_tfidf[unanimous_mask],
        activation_safe_threshold,
        tfidf_safe_threshold,
    )

    per_system: dict[str, Any] = {}
    for system in sorted(frame["system"].unique()):
        mask = frame["system"].eq(system).to_numpy()
        y = frame.loc[mask, "label"].to_numpy(dtype=int)
        per_system[system] = {
            "rows": int(mask.sum()),
            "positive_rows": int(y.sum()),
            "activation": rank_metrics(y, activation[mask]),
            "tfidf_word_char": rank_metrics(y, lufy_tfidf[mask]),
        }

    result = {
        "experiment_id": "MEM-EXP003",
        "status": "COMPLETE_EXTERNAL_HUMAN_TRANSFER",
        "selection_use": "none; zero-shot scoring with frozen representation, heads, renderers, and thresholds",
        "views": views,
        "per_system": per_system,
        "claim_boundary": (
            "LUFY measures whether information would be useful in future conversation. It does not "
            "measure storage consent, privacy policy, deletion intent, or retrieval authorization."
        ),
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        HERE / "comparison_scores.npz",
        label=frame["label"].to_numpy(dtype=np.int8),
        activation=activation,
        tfidf_word_char=lufy_tfidf,
        user_name=frame["user_name"].astype(str).to_numpy(),
        system=frame["system"].astype(str).to_numpy(),
        unanimous=frame["unanimous"].to_numpy(dtype=bool),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
