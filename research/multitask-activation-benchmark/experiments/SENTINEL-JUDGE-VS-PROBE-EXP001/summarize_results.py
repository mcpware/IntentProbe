#!/usr/bin/env python3
"""Summarize completed sentinel arms with paired vectorized bootstrap uncertainty."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "evaluation_manifest.json"
PREDICTIONS = HERE / "predictions"
OUTPUT_JSON = HERE / "result.json"
OUTPUT_CSV = HERE / "SENTINEL-SUMMARY.csv"
OUTPUT_MD = HERE / "RESULT_SUMMARY.md"
SEED = 20261018
RESAMPLES = 2000

SCORE_ARCHIVES = {
    ("cost_anchor", "sentiment"): HERE.parent / "SENTIMENT-SST2-LFM25-230M-EXP001/test_scores.npz",
    ("quality_anchor", "sentiment"): HERE.parent / "SENTIMENT-SST2-QWEN35-08B-EXP001/test_scores.npz",
    ("cost_anchor", "toxicity"): HERE.parent / "TOXICCHAT0124-LFM25-230M-EXP001/toxicity_test_scores.npz",
    ("quality_anchor", "toxicity"): HERE.parent / "TOXICCHAT0124-QWEN35-08B-EXP001/toxicity_test_scores.npz",
}
AUTHORIZATION_ARCHIVES = HERE / "activation_archives"


def ece(y: np.ndarray, score: np.ndarray, bins: int = 10) -> float:
    """Return fixed-width expected calibration error."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.clip(np.digitize(score, edges[1:-1], right=False), 0, bins - 1)
    counts = np.bincount(assignments, minlength=bins).astype(np.float64)
    positives = np.bincount(assignments, weights=y, minlength=bins)
    confidence = np.bincount(assignments, weights=score, minlength=bins)
    nonempty = counts > 0
    accuracy = np.divide(positives, counts, out=np.zeros_like(positives), where=nonempty)
    mean_confidence = np.divide(
        confidence, counts, out=np.zeros_like(confidence), where=nonempty
    )
    return float(np.sum(counts * np.abs(accuracy - mean_confidence)) / len(y))


def point_metrics(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Return the declared binary point metrics."""
    prediction = (score >= threshold).astype(np.int8)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "auprc": float(average_precision_score(y, score)),
        "auroc": float(roc_auc_score(y, score)),
        "brier": float(brier_score_loss(y, score)),
        "ece_10_bin": ece(y, score),
    }


def bootstrap_indices(n: int, resamples: int, seed: int) -> np.ndarray:
    """Draw one paired row-bootstrap index matrix."""
    return np.random.default_rng(seed).integers(0, n, size=(resamples, n), dtype=np.int32)


def vectorized_rank_metrics(
    y: np.ndarray, score: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return bootstrap AUPRC and AUROC for continuous scores without row loops."""
    sampled_y = y[indices].astype(np.float64)
    sampled_score = score[indices]
    positive = sampled_y.sum(axis=1)
    negative = sampled_y.shape[1] - positive
    if bool(((positive == 0) | (negative == 0)).any()):
        raise RuntimeError("Bootstrap draw lost a class")

    descending = np.argsort(-sampled_score, axis=1, kind="stable")
    ranked_y = np.take_along_axis(sampled_y, descending, axis=1)
    cumulative_positive = np.cumsum(ranked_y, axis=1)
    precision = cumulative_positive / np.arange(1, sampled_y.shape[1] + 1)[None, :]
    auprc = (precision * ranked_y).sum(axis=1) / positive

    ascending = np.argsort(sampled_score, axis=1, kind="stable")
    ascending_y = np.take_along_axis(sampled_y, ascending, axis=1)
    ranks = np.arange(1, sampled_y.shape[1] + 1, dtype=np.float64)[None, :]
    positive_rank_sum = (ascending_y * ranks).sum(axis=1)
    auroc = (positive_rank_sum - positive * (positive + 1.0) / 2.0) / (positive * negative)
    return auprc, auroc


def group_bootstrap_weights(
    group_ids: np.ndarray, resamples: int, seed: int
) -> np.ndarray:
    """Draw whole groups with replacement and return one weight per original row."""
    _, group_codes = np.unique(np.asarray(group_ids, dtype=str), return_inverse=True)
    group_count = int(group_codes.max()) + 1
    sampled_groups = np.random.default_rng(seed).integers(
        0, group_count, size=(resamples, group_count), dtype=np.int16
    )
    group_weights = np.eye(group_count, dtype=np.int16)[sampled_groups].sum(axis=1)
    return group_weights[:, group_codes]


def vectorized_weighted_rank_metrics(
    y: np.ndarray, score: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted AUPRC/AUROC for group-bootstrap replicate weights."""
    row_weights = np.asarray(weights, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64)
    positive = (row_weights * labels[None, :]).sum(axis=1)
    negative = (row_weights * (1.0 - labels)[None, :]).sum(axis=1)
    if bool(((positive == 0) | (negative == 0)).any()):
        raise RuntimeError("Group-bootstrap draw lost a class")

    descending = np.argsort(-score, kind="stable")
    descending_y = labels[descending]
    descending_weight = row_weights[:, descending]
    cumulative_positive = np.cumsum(
        descending_weight * descending_y[None, :], axis=1
    )
    cumulative_weight = np.cumsum(descending_weight, axis=1)
    precision = np.divide(
        cumulative_positive,
        cumulative_weight,
        out=np.zeros_like(cumulative_positive),
        where=cumulative_weight > 0,
    )
    auprc = (
        precision * descending_weight * descending_y[None, :]
    ).sum(axis=1) / positive

    ascending = np.argsort(score, kind="stable")
    ascending_y = labels[ascending]
    ascending_weight = row_weights[:, ascending]
    negative_weight = ascending_weight * (1.0 - ascending_y)[None, :]
    negative_before = np.cumsum(negative_weight, axis=1) - negative_weight
    positive_weight = ascending_weight * ascending_y[None, :]
    auroc = (positive_weight * negative_before).sum(axis=1) / (positive * negative)
    return auprc, auroc


def interval(values: np.ndarray) -> list[float]:
    """Return a percentile 95% interval."""
    return [float(value) for value in np.quantile(values, (0.025, 0.975))]


def two_sided_sign_p(values: np.ndarray) -> float:
    """Return a finite-bootstrap two-sided sign probability for a paired difference."""
    count = len(values)
    lower = (int((values <= 0).sum()) + 1) / (count + 1)
    upper = (int((values >= 0).sum()) + 1) / (count + 1)
    return float(min(1.0, 2.0 * min(lower, upper)))


def add_bh_fdr(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    """Attach BH q-values to the 12 frozen activation-versus-judge AUPRC tests."""
    indices = [
        index
        for index, row in enumerate(comparisons)
        if "minus-judge_" in row["comparison"]
    ]
    if not indices:
        return {"family_size": 0, "alpha": 0.05, "rejections": 0}
    p_values = np.asarray(
        [comparisons[index]["auprc_two_sided_sign_p"] for index in indices],
        dtype=np.float64,
    )
    order = np.argsort(p_values, kind="stable")
    ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    adjusted_ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    for local_index, comparison_index in enumerate(indices):
        comparisons[comparison_index]["auprc_bh_q"] = float(adjusted[local_index])
        comparisons[comparison_index]["auprc_bh_reject_0_05"] = bool(
            adjusted[local_index] <= 0.05
        )
    return {
        "family_size": len(indices),
        "alpha": 0.05,
        "rejections": int((adjusted <= 0.05).sum()),
        "method": "Benjamini-Hochberg",
        "outcome": "activation-versus-zero-shot/rubric choice-logprob AUPRC",
    }


def load_archive(model: str, task: str, selected_ids: np.ndarray) -> pd.DataFrame | None:
    """Load matched frozen activation and TF-IDF scores when available."""
    if task == "authorization_to_execute":
        path = AUTHORIZATION_ARCHIVES / f"{model}__authorization_to_execute.npz"
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as archive:
            frame = pd.DataFrame(
                {
                    "id": np.asarray(archive["id"]).astype(str),
                    "y": np.asarray(archive["y"], dtype=np.int8),
                    "group_id": np.asarray(archive["group_id"]).astype(str),
                    "activation": np.asarray(archive["activation"], dtype=np.float64),
                    "tfidf": np.asarray(archive["tfidf"], dtype=np.float64),
                }
            )
        frame = frame.loc[frame["id"].isin(selected_ids)].sort_values("id").reset_index(drop=True)
        if not np.array_equal(frame["id"].to_numpy(dtype=str), np.sort(selected_ids)):
            raise RuntimeError(f"Frozen score IDs do not match {model} {task}")
        return frame
    path = SCORE_ARCHIVES.get((model, task))
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as archive:
        frame = pd.DataFrame(
            {
                "id": np.asarray(archive["example_id"]).astype(str),
                "y": np.asarray(archive["label"], dtype=np.int8),
                "activation": np.asarray(archive["activation"], dtype=np.float64),
                "tfidf": np.asarray(archive["tfidf_word_char"], dtype=np.float64),
            }
        )
    frame = frame.loc[frame["id"].isin(selected_ids)].sort_values("id").reset_index(drop=True)
    if not np.array_equal(frame["id"].to_numpy(dtype=str), np.sort(selected_ids)):
        raise RuntimeError(f"Frozen score IDs do not match {model} {task}")
    return frame


def load_judge(path: Path, expected_ids: np.ndarray) -> pd.DataFrame:
    """Load and align one direct-judge prediction artifact."""
    frame = pd.read_json(path, lines=True)
    frame["id"] = frame["id"].astype(str)
    frame = frame.sort_values("id").reset_index(drop=True)
    if not np.array_equal(frame["id"].to_numpy(dtype=str), np.sort(expected_ids)):
        raise RuntimeError(f"Judge IDs do not match frozen manifest: {path}")
    return frame


def summarize() -> dict[str, Any]:
    """Summarize every completed arm without inventing missing cells."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    completed_cells = 0
    expected_cells = 6

    for model in ("cost_anchor", "quality_anchor"):
        for task in ("authorization_to_execute", "toxicity", "sentiment"):
            selected_ids = np.asarray(manifest["tasks"][task]["ids"], dtype=str)
            archive = load_archive(model, task, selected_ids)
            methods: dict[str, tuple[np.ndarray, np.ndarray, float, int]] = {}
            if archive is not None:
                y = archive["y"].to_numpy(dtype=np.int8)
                methods["activation_head"] = (
                    y,
                    archive["activation"].to_numpy(dtype=np.float64),
                    float("nan"),
                    0,
                )
                methods["tfidf_word_char"] = (
                    y,
                    archive["tfidf"].to_numpy(dtype=np.float64),
                    float("nan"),
                    0,
                )

            cell_judges = 0
            for prompt in ("zero_shot", "rubric"):
                for method in ("choice_logprob", "generate"):
                    path = PREDICTIONS / f"{model}__{task}__{prompt}__{method}.jsonl"
                    if not path.exists():
                        continue
                    judge = load_judge(path, selected_ids)
                    judge_y = judge["y"].to_numpy(dtype=np.int8)
                    score = judge["score"].to_numpy(dtype=np.float64)
                    invalid_mask = judge["invalid"].to_numpy(dtype=bool)
                    if method == "generate":
                        score = np.where(invalid_mask, 1 - judge_y, score).astype(np.float64)
                    if archive is not None and not np.array_equal(judge_y, archive["y"].to_numpy()):
                        raise RuntimeError(f"Label mismatch for {path}")
                    metadata = json.loads(path.with_suffix(".metadata.json").read_text())
                    key = f"judge_{prompt}_{method}"
                    methods[key] = (
                        judge_y,
                        score,
                        float(metadata["seconds_per_example"]),
                        int(invalid_mask.sum()),
                    )
                    cell_judges += 1

            if cell_judges == 4:
                completed_cells += 1
            if not methods:
                continue
            first_y = next(iter(methods.values()))[0]
            if task == "authorization_to_execute":
                if archive is None or "group_id" not in archive:
                    raise RuntimeError("Authorization summary requires frozen family IDs")
                bootstrap_input = group_bootstrap_weights(
                    archive["group_id"].to_numpy(dtype=str), RESAMPLES, SEED
                )
                bootstrap_unit = "group_id"
            else:
                bootstrap_input = bootstrap_indices(len(first_y), RESAMPLES, SEED)
                bootstrap_unit = "row"
            bootstrap_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for method_name, (y, score, seconds_per_example, invalid_outputs) in methods.items():
                metrics = point_metrics(y, score)
                is_continuous = method_name not in {
                    "judge_zero_shot_generate",
                    "judge_rubric_generate",
                }
                if is_continuous:
                    if bootstrap_unit == "group_id":
                        ap_boot, auc_boot = vectorized_weighted_rank_metrics(
                            y, score, bootstrap_input
                        )
                    else:
                        ap_boot, auc_boot = vectorized_rank_metrics(
                            y, score, bootstrap_input
                        )
                    bootstrap_cache[method_name] = (ap_boot, auc_boot)
                    ap_ci = interval(ap_boot)
                    auc_ci = interval(auc_boot)
                else:
                    ap_ci = [float("nan"), float("nan")]
                    auc_ci = [float("nan"), float("nan")]
                rows.append(
                    {
                        "model": model,
                        "task": task,
                        "method": method_name,
                        "n": int(len(y)),
                        "positive_n": int(y.sum()),
                        "invalid_outputs": invalid_outputs,
                        "bootstrap_unit": bootstrap_unit,
                        "seconds_per_example_batch": seconds_per_example,
                        **metrics,
                        "auprc_ci_low": ap_ci[0],
                        "auprc_ci_high": ap_ci[1],
                        "auroc_ci_low": auc_ci[0],
                        "auroc_ci_high": auc_ci[1],
                    }
                )

            if "activation_head" in bootstrap_cache:
                activation_ap, activation_auc = bootstrap_cache["activation_head"]
                for comparator in (
                    "tfidf_word_char",
                    "judge_zero_shot_choice_logprob",
                    "judge_rubric_choice_logprob",
                ):
                    if comparator not in bootstrap_cache:
                        continue
                    comparator_ap, comparator_auc = bootstrap_cache[comparator]
                    ap_difference_bootstrap = activation_ap - comparator_ap
                    auc_difference_bootstrap = activation_auc - comparator_auc
                    comparisons.append(
                        {
                            "model": model,
                            "task": task,
                            "comparison": f"activation_head-minus-{comparator}",
                            "auprc_difference": float(
                                next(row["auprc"] for row in rows if row["model"] == model and row["task"] == task and row["method"] == "activation_head")
                                - next(row["auprc"] for row in rows if row["model"] == model and row["task"] == task and row["method"] == comparator)
                            ),
                            "auprc_difference_ci": interval(ap_difference_bootstrap),
                            "auprc_two_sided_sign_p": two_sided_sign_p(
                                ap_difference_bootstrap
                            ),
                            "auroc_difference": float(
                                next(row["auroc"] for row in rows if row["model"] == model and row["task"] == task and row["method"] == "activation_head")
                                - next(row["auroc"] for row in rows if row["model"] == model and row["task"] == task and row["method"] == comparator)
                            ),
                            "auroc_difference_ci": interval(auc_difference_bootstrap),
                            "auroc_two_sided_sign_p": two_sided_sign_p(
                                auc_difference_bootstrap
                            ),
                        }
                    )

    fdr = add_bh_fdr(comparisons)

    return {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "status": "COMPLETE" if completed_cells == expected_cells else "PARTIAL",
        "completed_cells": completed_cells,
        "expected_cells": expected_cells,
        "bootstrap": {
            "resamples": RESAMPLES,
            "seed": SEED,
            "unit": "family_id for authorization; row for public sentiment/toxicity controls",
            "multiple_comparison_boundary": "The 12 activation-versus-zero-shot/rubric choice-logprob AUPRC comparisons use Benjamini-Hochberg FDR 0.05.",
        },
        "fdr": fdr,
        "rows": rows,
        "paired_comparisons": comparisons,
    }


def write_outputs(result: dict[str, Any]) -> None:
    """Write machine-readable and human-readable partial or complete summaries."""
    OUTPUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    frame = pd.DataFrame(result["rows"])
    frame.to_csv(OUTPUT_CSV, index=False)
    lines = [
        "# Sentinel judge-versus-probe result summary",
        "",
        f"Status: **{result['status']}** — {result['completed_cells']}/{result['expected_cells']} six-cell panels have all four judge arms.",
        "",
        "Point estimates and 2,000 paired row-bootstrap intervals are computed from saved per-example scores. Generated-label arms use strict YES/NO parsing; their hard scores are shown without rank-metric bootstrap intervals.",
        "",
    ]
    if frame.empty:
        lines.append("No completed artifacts yet.")
    else:
        display = frame.loc[
            :,
            [
                "model",
                "task",
                "method",
                "n",
                "invalid_outputs",
                "auprc",
                "auroc",
                "balanced_accuracy",
                "brier",
                "seconds_per_example_batch",
            ],
        ].copy()
        for column in ("auprc", "auroc", "balanced_accuracy", "brier", "seconds_per_example_batch"):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        lines.extend([display.to_markdown(index=False), ""])
    if result["paired_comparisons"]:
        lines.extend(["## Paired activation differences", ""])
        for row in result["paired_comparisons"]:
            line = (
                f"- {row['model']} / {row['task']} / {row['comparison']}: "
                f"ΔAUPRC {row['auprc_difference']:+.4f} "
                f"[{row['auprc_difference_ci'][0]:+.4f}, {row['auprc_difference_ci'][1]:+.4f}]; "
                f"ΔAUROC {row['auroc_difference']:+.4f} "
                f"[{row['auroc_difference_ci'][0]:+.4f}, {row['auroc_difference_ci'][1]:+.4f}]."
            )
            if "auprc_bh_q" in row:
                line += (
                    f" AUPRC sign p={row['auprc_two_sided_sign_p']:.4f}, "
                    f"BH q={row['auprc_bh_q']:.4f}, "
                    f"reject={row['auprc_bh_reject_0_05']}."
                )
            lines.append(line)
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Summarize the currently completed immutable artifacts."""
    result = summarize()
    write_outputs(result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "completed_cells": result["completed_cells"],
                "expected_cells": result["expected_cells"],
                "metric_rows": len(result["rows"]),
                "paired_comparisons": len(result["paired_comparisons"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
