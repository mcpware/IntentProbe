#!/usr/bin/env python3
"""Validate and summarize the crossed ToxicChat task-by-backbone matrix."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


BENCHMARK = Path(__file__).resolve().parents[1]
EXPERIMENTS = BENCHMARK / "experiments"
TASKS = ("toxicity", "jailbreaking")


def load_cells() -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    rows = []
    tables = {}
    references = {}
    for experiment in sorted(EXPERIMENTS.glob("TOXICCHAT0124-*-EXP001")):
        config = json.loads((experiment / "config.json").read_text(encoding="utf-8"))
        model = str(config["backbone"]["id"])
        for task in TASKS:
            result = json.loads((experiment / f"{task}_result.json").read_text(encoding="utf-8"))
            scores = np.load(experiment / f"{task}_test_scores.npz")
            table = pd.read_csv(experiment / f"{task}_candidate_table.csv")
            winner = result["selection"]["winner"]
            id_key = "example_id" if "example_id" in scores.files else "conv_id"
            assert np.array_equal(scores["label"], references.setdefault((task, "label"), scores["label"]))
            assert np.array_equal(
                scores[id_key], references.setdefault((task, "example_id"), scores[id_key])
            )
            assert np.array_equal(
                scores["tfidf_word_char"],
                references.setdefault((task, "tfidf_word_char"), scores["tfidf_word_char"]),
            )
            assert abs(average_precision_score(scores["label"], scores["activation"]) - result["heldout_test"]["models"]["activation"]["auprc"]) < 1e-12
            assert abs(roc_auc_score(scores["label"], scores["activation"]) - result["heldout_test"]["models"]["activation"]["auroc"]) < 1e-12
            rows.append(
                {
                    "model": model,
                    "task": task,
                    "pooling": str(winner["pooling"]),
                    "start_layer": int(winner["start_layer"]),
                    "end_layer": int(winner["end_layer"]),
                    "width": int(winner["width"]),
                    "normalized_depth": float(winner["normalized_depth"]),
                    "validation_auprc": float(winner["validation_auprc"]),
                    "test_auprc": float(result["heldout_test"]["models"]["activation"]["auprc"]),
                    "test_auroc": float(result["heldout_test"]["models"]["activation"]["auroc"]),
                    "tfidf_auprc": float(result["heldout_test"]["models"]["tfidf_word_char"]["auprc"]),
                    "tfidf_auroc": float(result["heldout_test"]["models"]["tfidf_word_char"]["auroc"]),
                    "auprc_difference": float(result["heldout_test"]["paired_activation_minus_tfidf"]["auprc_difference"]),
                    "auprc_ci_low": float(result["heldout_test"]["paired_activation_minus_tfidf"]["auprc_difference_95ci"][0]),
                    "auprc_ci_high": float(result["heldout_test"]["paired_activation_minus_tfidf"]["auprc_difference_95ci"][1]),
                    "auroc_difference": float(result["heldout_test"]["paired_activation_minus_tfidf"]["auroc_difference"]),
                    "auroc_ci_low": float(result["heldout_test"]["paired_activation_minus_tfidf"]["auroc_difference_95ci"][0]),
                    "auroc_ci_high": float(result["heldout_test"]["paired_activation_minus_tfidf"]["auroc_difference_95ci"][1]),
                }
            )
            tables[(model, task)] = table
    frame = pd.DataFrame(rows)
    assert len(frame) == 12 and frame["model"].nunique() == 6
    return frame, tables


def choose_policy(tables: dict[tuple[str, str], pd.DataFrame], keys: list[tuple[str, str]]) -> dict[str, object]:
    candidates = []
    for pooling in ("masked_mean", "last_nonpadding", "masked_max"):
        for width in (1, 2, 3):
            for target_depth in np.linspace(0.0, 1.0, 201):
                scores = []
                for key in keys:
                    subset = tables[key]
                    subset = subset[(subset["pooling"] == pooling) & (subset["width"] == width)].copy()
                    distance = (subset["normalized_depth"] - target_depth).abs()
                    selected = subset.loc[distance.idxmin()]
                    scores.append(float(selected["validation_auprc"]))
                candidates.append((float(np.mean(scores)), pooling, width, float(target_depth)))
    mean_score, pooling, width, target_depth = max(candidates)
    return {"pooling": pooling, "width": width, "target_depth": target_depth, "mean_validation_auprc": mean_score}


def policy_scores(policy: dict[str, object], tables: dict[tuple[str, str], pd.DataFrame], keys: list[tuple[str, str]]) -> np.ndarray:
    scores = []
    for key in keys:
        subset = tables[key]
        subset = subset[(subset["pooling"] == policy["pooling"]) & (subset["width"] == policy["width"])]
        selected = subset.loc[(subset["normalized_depth"] - policy["target_depth"]).abs().idxmin()]
        scores.append(float(selected["validation_auprc"]))
    return np.asarray(scores)


def variance_decomposition(frame: pd.DataFrame) -> dict[str, float]:
    pivot = frame.pivot(index="model", columns="task", values="normalized_depth")
    values = pivot.to_numpy()
    grand = values.mean()
    task_means = values.mean(axis=0)
    model_means = values.mean(axis=1)
    total = float(np.square(values - grand).sum())
    task = float(len(pivot.index) * np.square(task_means - grand).sum())
    model = float(len(pivot.columns) * np.square(model_means - grand).sum())
    interaction = total - task - model
    return {
        "total_ss": total,
        "task_share": task / total,
        "model_share": model / total,
        "interaction_share": interaction / total,
    }


def render_markdown(frame: pd.DataFrame, summary: dict[str, object]) -> str:
    policy = summary["validation_policy_comparison"]
    variance = summary["depth"]["descriptive_variance_decomposition"]
    lines = [
        "# ToxicChat: six-backbone crossed activation-readout matrix",
        "",
        "The matrix uses the same 5,441 human-annotated prompts and frozen train/validation/test rows for",
        "both toxicity and jailbreak. All six score artifacts have identical held-out IDs, labels, and",
        "matched TF-IDF probabilities within each task.",
        "",
        "| Task | Backbone | Validation-selected readout | Depth | Test AUPRC | TF-IDF AUPRC | Difference (95% CI) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in frame.sort_values(["task", "model"]).to_dict("records"):
        readout = f"{row['pooling']} {row['start_layer']}--{row['end_layer']}"
        difference = f"{row['auprc_difference']:+.4f} [{row['auprc_ci_low']:+.4f}, {row['auprc_ci_high']:+.4f}]"
        lines.append(
            f"| {row['task']} | {row['model']} | {readout} | {row['normalized_depth']:.3f} | "
            f"{row['test_auprc']:.4f} | {row['tfidf_auprc']:.4f} | {difference} |"
        )
    means = policy["mean_auprc"]
    regret = policy["mean_regret_vs_cell_specific"]
    lines.extend(
        [
            "",
            "## Readout pattern",
            "",
            "Toxicity selected masked mean in all six backbones. Jailbreak split evenly between masked mean",
            "and masked max. No task selected last-nonpadding pooling. Selected normalized depths span",
            f"{summary['depth']['minimum']:.3f} to {summary['depth']['maximum']:.3f}; the median is "
            f"{summary['depth']['median']:.3f}.",
            "",
            "The best single validation policy is masked mean, width 2, centered at normalized depth 0.50.",
            f"Its mean validation AUPRC is {means['universal']:.4f}, compared with {means['cell_specific']:.4f} "
            "for per-task-per-backbone selection. Model-specific policies reduce mean regret from "
            f"{regret['universal']:.4f} to {regret['model_specific']:.4f}; task-specific policies alone leave "
            f"{regret['task_specific']:.4f} regret.",
            "",
            "## Evidence boundary",
            "",
            f"The descriptive normalized-depth decomposition assigns {variance['model_share']:.1%} of variation "
            f"to backbone, {variance['task_share']:.1%} to the two-task main effect, and "
            f"{variance['interaction_share']:.1%} to task-by-backbone interaction. With only two tasks and no "
            "replicated interaction error term, these shares are descriptive rather than population inference.",
            "The policy comparison uses the same validation cells to select and evaluate policies, so its regret",
            "estimates are also descriptive. Held-out activation-versus-TF-IDF claims use the prospectively",
            "specified paired row bootstrap; two of twelve cells have intervals above zero on both AUPRC and",
            "AUROC, both on toxicity (Granite-4.0-350M and MiniCPM5-1B).",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    frame, tables = load_cells()
    keys = list(tables)
    universal = choose_policy(tables, keys)
    universal_scores = policy_scores(universal, tables, keys)
    task_policies = {task: choose_policy(tables, [key for key in keys if key[1] == task]) for task in TASKS}
    model_policies = {model: choose_policy(tables, [key for key in keys if key[0] == model]) for model in sorted(frame["model"].unique())}
    task_scores = np.concatenate([policy_scores(task_policies[task], tables, [key for key in keys if key[1] == task]) for task in TASKS])
    model_scores = np.concatenate([policy_scores(model_policies[model], tables, [key for key in keys if key[0] == model]) for model in sorted(frame["model"].unique())])
    oracle_scores = frame["validation_auprc"].to_numpy()
    depth_pivot = frame.pivot(index="model", columns="task", values="normalized_depth")
    rho, pvalue = spearmanr(depth_pivot["toxicity"], depth_pivot["jailbreaking"])
    summary = {
        "status": "COMPLETE_SIX_BACKBONE_TWO_TASK_CROSSED_MATRIX",
        "cells": len(frame),
        "pooling_counts": {
            task: dict(Counter(frame.loc[frame["task"] == task, "pooling"])) for task in TASKS
        },
        "depth": {
            "median": float(frame["normalized_depth"].median()),
            "minimum": float(frame["normalized_depth"].min()),
            "maximum": float(frame["normalized_depth"].max()),
            "within_model_absolute_task_difference_median": float((depth_pivot["toxicity"] - depth_pivot["jailbreaking"]).abs().median()),
            "cross_task_spearman_rho": float(rho),
            "cross_task_spearman_pvalue": float(pvalue),
            "cross_task_n_models": 6,
            "descriptive_variance_decomposition": variance_decomposition(frame),
        },
        "validation_policy_comparison": {
            "universal": universal,
            "task_specific": task_policies,
            "model_specific": model_policies,
            "mean_auprc": {
                "universal": float(universal_scores.mean()),
                "task_specific": float(task_scores.mean()),
                "model_specific": float(model_scores.mean()),
                "cell_specific": float(oracle_scores.mean()),
            },
            "mean_regret_vs_cell_specific": {
                "universal": float((oracle_scores.mean() - universal_scores.mean())),
                "task_specific": float((oracle_scores.mean() - task_scores.mean())),
                "model_specific": float((oracle_scores.mean() - model_scores.mean())),
            },
            "limitation": "Policies and regret are descriptive on the same validation cells used to choose them.",
        },
        "test": {
            "activation_advantage_both_metrics_count": int(((frame["auprc_ci_low"] > 0) & (frame["auroc_ci_low"] > 0)).sum()),
            "toxicity_activation_advantage_both_metrics_count": int(((frame["task"] == "toxicity") & (frame["auprc_ci_low"] > 0) & (frame["auroc_ci_low"] > 0)).sum()),
            "jailbreak_activation_advantage_both_metrics_count": int(((frame["task"] == "jailbreaking") & (frame["auprc_ci_low"] > 0) & (frame["auroc_ci_low"] > 0)).sum()),
        },
        "inference_boundary": "Task/model shares are descriptive: two tasks provide no replicated interaction error term.",
    }
    frame.sort_values(["task", "model"]).to_csv(EXPERIMENTS / "TOXICCHAT0124-MODEL-TASK-MATRIX.csv", index=False)
    (EXPERIMENTS / "TOXICCHAT0124-MODEL-TASK-MATRIX.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (EXPERIMENTS / "TOXICCHAT0124-MODEL-TASK-MATRIX.md").write_text(
        render_markdown(frame, summary), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
