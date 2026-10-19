#!/usr/bin/env python3
"""Validate and summarize one completed multi-label activation experiment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def load_task(experiment: Path, task_name: str) -> dict[str, object]:
    result_path = experiment / f"{task_name}_result.json"
    scores_path = experiment / f"{task_name}_test_scores.npz"
    table_path = experiment / f"{task_name}_candidate_table.csv"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    arrays = np.load(scores_path)
    table = pd.read_csv(table_path)
    winner = result["selection"]["winner"]
    first = table.iloc[0]
    assert len(table) == int(result["candidate_count"])
    for key in ("pooling", "start_layer", "end_layer", "width"):
        assert str(first[key]) == str(winner[key])
    y = arrays["label"]
    activation = arrays["activation"]
    tfidf = arrays["tfidf_word_char"]
    reported = result["heldout_test"]["models"]
    assert abs(average_precision_score(y, activation) - reported["activation"]["auprc"]) < 1e-12
    assert abs(roc_auc_score(y, activation) - reported["activation"]["auroc"]) < 1e-12
    assert abs(average_precision_score(y, tfidf) - reported["tfidf_word_char"]["auprc"]) < 1e-12
    assert abs(roc_auc_score(y, tfidf) - reported["tfidf_word_char"]["auroc"]) < 1e-12
    paired = result["heldout_test"]["paired_activation_minus_tfidf"]
    return {
        "task": task_name,
        "task_id": result["task"],
        "pooling": winner["pooling"],
        "start_layer": int(winner["start_layer"]),
        "end_layer": int(winner["end_layer"]),
        "width": int(winner["width"]),
        "normalized_depth": float(winner["normalized_depth"]),
        "validation_auprc": float(winner["validation_auprc"]),
        "test_rows": int(result["heldout_test"]["rows"]),
        "test_positive_rows": int(result["heldout_test"]["positive_rows"]),
        "activation_auprc": float(reported["activation"]["auprc"]),
        "activation_auroc": float(reported["activation"]["auroc"]),
        "tfidf_auprc": float(reported["tfidf_word_char"]["auprc"]),
        "tfidf_auroc": float(reported["tfidf_word_char"]["auroc"]),
        "auprc_difference": float(paired["auprc_difference"]),
        "auprc_ci_low": float(paired["auprc_difference_95ci"][0]),
        "auprc_ci_high": float(paired["auprc_difference_95ci"][1]),
        "auroc_difference": float(paired["auroc_difference"]),
        "auroc_ci_low": float(paired["auroc_difference_95ci"][0]),
        "auroc_ci_high": float(paired["auroc_difference_95ci"][1]),
    }


def render(title: str, rows: list[dict[str, object]]) -> str:
    lines = [
        f"# {title}",
        "",
        "All targets use identical frozen text rows, splits, backbone activations, candidate family,",
        "and matched word-plus-character TF-IDF baseline protocol. Each readout was selected on",
        "validation AUPRC before its one-shot test evaluation.",
        "",
        "| Target | Selected readout | Depth | Test positives | Activation AUPRC | TF-IDF AUPRC | Difference (95% CI) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        readout = f"{row['pooling']} {row['start_layer']}--{row['end_layer']}"
        difference = (
            f"{row['auprc_difference']:+.4f} "
            f"[{row['auprc_ci_low']:+.4f}, {row['auprc_ci_high']:+.4f}]"
        )
        lines.append(
            f"| {row['task']} | {readout} | {row['normalized_depth']:.3f} | "
            f"{row['test_positive_rows']} | {row['activation_auprc']:.4f} | "
            f"{row['tfidf_auprc']:.4f} | {difference} |"
        )
    poolings = Counter(str(row["pooling"]) for row in rows)
    depths = [float(row["normalized_depth"]) for row in rows]
    lines.extend(
        [
            "",
            "## Descriptive readout pattern",
            "",
            f"Pooling winners: {dict(sorted(poolings.items()))}. Selected normalized depth ranges from "
            f"{min(depths):.3f} to {max(depths):.3f}. These within-backbone task differences are "
            "descriptive; the six-backbone crossed matrix is required before attributing the pattern to "
            "task rather than one architecture.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir")
    parser.add_argument("--output-stem", required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    experiment = Path(args.experiment_dir).resolve()
    config = json.loads((experiment / "config.json").read_text(encoding="utf-8"))
    tasks = list(config["tasks"])
    rows = [load_task(experiment, task) for task in tasks]
    assert len({int(row["test_rows"]) for row in rows}) == 1
    summary = {
        "experiment_id": config["experiment_id"],
        "status": "COMPLETE_VALIDATED_MULTILABEL_MATRIX",
        "tasks": len(rows),
        "pooling_counts": dict(Counter(str(row["pooling"]) for row in rows)),
        "minimum_normalized_depth": min(float(row["normalized_depth"]) for row in rows),
        "maximum_normalized_depth": max(float(row["normalized_depth"]) for row in rows),
        "activation_advantage_both_metrics": int(
            sum(float(row["auprc_ci_low"]) > 0 and float(row["auroc_ci_low"]) > 0 for row in rows)
        ),
        "tfidf_advantage_both_metrics": int(
            sum(float(row["auprc_ci_high"]) < 0 and float(row["auroc_ci_high"]) < 0 for row in rows)
        ),
        "rows": rows,
    }
    stem = experiment / args.output_stem
    pd.DataFrame(rows).to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    stem.with_suffix(".md").write_text(render(args.title, rows), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
