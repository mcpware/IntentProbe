#!/usr/bin/env python3
"""Validate and summarize a completed six-backbone binary-task matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


BENCHMARK = Path(__file__).resolve().parents[1]
EXPECTED_MODELS = {
    "LiquidAI/LFM2.5-230M": "LFM2.5-230M",
    "ibm-granite/granite-4.0-350m-base": "Granite-4.0-350M",
    "Qwen/Qwen2.5-0.5B": "Qwen2.5-0.5B",
    "Qwen/Qwen3-0.6B-Base": "Qwen3-0.6B",
    "Qwen/Qwen3.5-0.8B": "Qwen3.5-0.8B",
    "openbmb/MiniCPM5-1B-Base": "MiniCPM5-1B",
}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def status_from_interval(result: dict[str, object]) -> str:
    """Classify only the prospectively planned activation-versus-baseline comparison."""
    paired = result["heldout_test"]["paired_activation_minus_tfidf"]
    auprc_low = float(paired["auprc_difference_95ci"][0])
    auroc_low = float(paired["auroc_difference_95ci"][0])
    return "activation_advantage_both_metrics" if min(auprc_low, auroc_low) > 0 else "unresolved"


def load_row(experiment: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Load and independently verify one completed experiment."""
    config_path = experiment / "config.json"
    result_path = experiment / "result.json"
    scores_path = experiment / "test_scores.npz"
    candidates_path = experiment / "candidate_table.csv"
    manifest_path = experiment / "extraction_manifest.json"
    required = (config_path, result_path, scores_path, candidates_path, manifest_path)
    if not all(path.exists() for path in required):
        missing = [str(path) for path in required if not path.exists()]
        raise FileNotFoundError(f"Incomplete experiment {experiment.name}: {missing}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = pd.read_csv(candidates_path)
    arrays_file = np.load(scores_path)
    arrays = {key: arrays_file[key].copy() for key in arrays_file.files}

    model_id = str(config["backbone"]["id"])
    if model_id not in EXPECTED_MODELS:
        raise ValueError(f"Unexpected backbone in matrix: {model_id}")
    if result["status"] != "COMPLETE_VALIDATION_SCREEN_AND_ONE_SHOT_TEST":
        raise ValueError(f"Incomplete result status for {experiment.name}")
    if manifest["status"] != "ALL_POOLING_FEATURES_COMPLETE":
        raise ValueError(f"Incomplete extraction status for {experiment.name}")
    expected_dataset_rows = int(config.get("dataset", {}).get("rows", manifest["rows"]))
    if int(manifest["rows"]) != expected_dataset_rows:
        raise ValueError(f"Extraction row mismatch for {experiment.name}")
    if len(candidates) != int(config["candidate_family"]["candidate_count"]):
        raise ValueError(f"Candidate-count mismatch for {experiment.name}")

    winner = result["selection"]["winner"]
    first = candidates.iloc[0]
    for key in ("pooling", "start_layer", "end_layer", "width"):
        if str(first[key]) != str(winner[key]):
            raise ValueError(f"Winner mismatch for {experiment.name}: {key}")
    labels = arrays["label"]
    activation = arrays["activation"]
    tfidf = arrays["tfidf_word_char"]
    expected_rows = int(result["heldout_test"]["rows"])
    if not (len(labels) == len(activation) == len(tfidf) == len(arrays["example_id"]) == expected_rows):
        raise ValueError(f"Held-out score row mismatch for {experiment.name}")

    reported = result["heldout_test"]["models"]
    recomputed = {
        "activation_auprc": float(average_precision_score(labels, activation)),
        "activation_auroc": float(roc_auc_score(labels, activation)),
        "tfidf_auprc": float(average_precision_score(labels, tfidf)),
        "tfidf_auroc": float(roc_auc_score(labels, tfidf)),
    }
    checks = {
        "activation_auprc": float(reported["activation"]["auprc"]),
        "activation_auroc": float(reported["activation"]["auroc"]),
        "tfidf_auprc": float(reported["tfidf_word_char"]["auprc"]),
        "tfidf_auroc": float(reported["tfidf_word_char"]["auroc"]),
    }
    if any(abs(recomputed[key] - checks[key]) > 1e-12 for key in recomputed):
        raise ValueError(f"Metric recomputation mismatch for {experiment.name}")

    paired = result["heldout_test"]["paired_activation_minus_tfidf"]
    selection_metric = str(result["selection"]["metric"]).lower()
    if selection_metric not in {"auprc", "auroc"}:
        raise ValueError(f"Unsupported selection metric for {experiment.name}: {selection_metric}")
    task_id = str(result.get("task", config["task"]["id"]))
    train_rows = int(
        result["selection"].get(
            "available_train_rows", result["selection"].get("train_rows")
        )
    )
    validation_rows = int(result["selection"]["validation_rows"])
    test_rows = int(result["heldout_test"]["rows"])
    row = {
        "task_id": task_id,
        "selection_metric": selection_metric,
        "dataset_rows": train_rows + validation_rows + test_rows,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "test_rows": test_rows,
        "test_positive_rows": int(result["heldout_test"]["positive_rows"]),
        "uncertainty_limitation": str(result["uncertainty_limitation"]),
        "model": EXPECTED_MODELS[model_id],
        "model_id": model_id,
        "revision": str(config["backbone"]["revision"]),
        "hidden_state_outputs": int(config["backbone"]["hidden_state_outputs"]),
        "hidden_size": int(config["backbone"]["hidden_size"]),
        "candidate_count": len(candidates),
        "pooling": str(winner["pooling"]),
        "start_layer": int(winner["start_layer"]),
        "end_layer": int(winner["end_layer"]),
        "width": int(winner["width"]),
        "normalized_depth": float(winner["normalized_depth"]),
        "validation_auprc": float(winner["validation_auprc"]),
        "validation_auroc": float(winner["validation_auroc"]),
        **recomputed,
        "auprc_difference": float(paired["auprc_difference"]),
        "auprc_difference_95ci_low": float(paired["auprc_difference_95ci"][0]),
        "auprc_difference_95ci_high": float(paired["auprc_difference_95ci"][1]),
        "auroc_difference": float(paired["auroc_difference"]),
        "auroc_difference_95ci_low": float(paired["auroc_difference_95ci"][0]),
        "auroc_difference_95ci_high": float(paired["auroc_difference_95ci"][1]),
        "planned_activation_vs_tfidf_status": status_from_interval(result),
        "result_sha256": sha256(result_path),
        "test_scores_sha256": sha256(scores_path),
    }
    return row, arrays


def markdown(summary: dict[str, object]) -> str:
    """Render the validated matrix as a paper-facing Markdown summary."""
    rows = summary["models"]
    metric = str(summary["selection_metric"])
    metric_label = metric.upper()
    confirmed = [
        row["model"]
        for row in rows
        if row["planned_activation_vs_tfidf_status"] == "activation_advantage_both_metrics"
    ]
    confirmed_text = ", ".join(confirmed) if confirmed else "none"
    confirmed_count = int(summary["planned_comparison_counts"]["activation_advantage_both_metrics"])
    unresolved_count = int(summary["planned_comparison_counts"]["unresolved"])
    count_words = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}
    comparison_sentence = (
        "All six planned within-backbone comparisons have activation-minus-TF-IDF 95% paired "
        f"row-bootstrap intervals above zero on both AUPRC and AUROC: {confirmed_text}."
        if confirmed_count == 6
        else (
            f"{count_words[confirmed_count]} of six planned within-backbone comparisons have "
            "activation-minus-TF-IDF "
            "95% paired row-bootstrap intervals above zero on both AUPRC and AUROC: "
            f"{confirmed_text}. The other {count_words[unresolved_count].lower()} remain unresolved."
        )
    )
    depth_values = [float(row["normalized_depth"]) for row in rows]
    width_values = [int(row["width"]) for row in rows]
    observed_best = min(rows, key=lambda row: int(row["test_primary_metric_rank_exploratory"]))
    validation_best = min(rows, key=lambda row: int(row["validation_primary_metric_rank"]))
    lines = [
        f"# {summary['display_name']}: six-backbone activation-readout matrix",
        "",
        "## Contract and integrity checks",
        "",
        f"All six frozen backbones use the same {summary['dataset_rows']:,} processed examples, the same",
        f"{summary['train_rows']:,}/{summary['validation_rows']:,}/{summary['test_rows']:,} train/validation/test",
        f"split, and the same untouched test rows ({summary['test_positive_rows']:,} positive). Example IDs, labels,",
        "and TF-IDF test probabilities are identical across all six score artifacts. Every reported AUPRC",
        "and AUROC below was independently recomputed from the saved test probabilities.",
        "",
        f"Each backbone selected its readout by validation {metric_label} only. Its one-shot test comparison against",
        "the matched word-plus-character TF-IDF logistic baseline was prospectively planned. Cross-backbone",
        "test ranking is descriptive and exploratory: the test results were already observed before this",
        "matrix summary, so it is not used for a new significance claim.",
        "",
        "## Result",
        "",
        f"| Backbone | Validation-selected readout | Depth | Val {metric_label} | Test AUPRC | Test AUROC | Δ AUPRC vs TF-IDF (95% CI) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        readout = f"{row['pooling']} {row['start_layer']}--{row['end_layer']}"
        ci = (
            f"{row['auprc_difference']:+.4f} "
            f"[{row['auprc_difference_95ci_low']:+.4f}, {row['auprc_difference_95ci_high']:+.4f}]"
        )
        lines.append(
            f"| {row['model']} | {readout} | {row['normalized_depth']:.3f} | "
            f"{row[f'validation_{metric}']:.4f} | {row['activation_auprc']:.4f} | "
            f"{row['activation_auroc']:.4f} | {ci} |"
        )
    lines.extend(
        [
            "",
            f"The common TF-IDF baseline is {summary['baseline']['auprc']:.4f} AUPRC and "
            f"{summary['baseline']['auroc']:.4f} AUROC. {comparison_sentence}",
            "",
            "The main engineering observation is heterogeneity, not a universal winning cut. The selected pooling",
            f"counts are masked mean {summary['readout_counts']['masked_mean']}, last non-padding "
            f"{summary['readout_counts']['last_nonpadding']}, and masked max {summary['readout_counts']['masked_max']}. "
            f"Selected normalized depth spans {min(depth_values):.3f} to {max(depth_values):.3f}, while selected "
            f"windows span {min(width_values)} to {max(width_values)} adjacent outputs. {observed_best['model']} has "
            f"the highest observed test {metric_label}, while {validation_best['model']} has the highest validation "
            f"{metric_label}. This one task cannot establish a globally best backbone.",
            "",
            "## Evidence boundary",
            "",
            "This is one binary task and one fixed split. The available metadata does not support a stronger",
            "clustered bootstrap, so uncertainty is row-level. The matrix supports",
            "the study design—task × backbone × layer × pooling—but the paper claim requires the same six-backbone",
            "protocol across the remaining real-data tasks. No cross-model superiority claim is made here.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Validate matching held-out rows and write JSON, CSV, and Markdown summaries."""
    parser = argparse.ArgumentParser()
    parser.add_argument("task_prefix", help="Experiment-directory prefix, for example SPAM-SMS-UCI-")
    parser.add_argument("--output-stem", required=True)
    parser.add_argument("--display-name", required=True)
    args = parser.parse_args()

    experiments = sorted((BENCHMARK / "experiments").glob(f"{args.task_prefix}*EXP001"))
    loaded = [load_row(experiment) for experiment in experiments]
    rows = [item[0] for item in loaded]
    arrays = [item[1] for item in loaded]
    observed_models = {str(row["model_id"]) for row in rows}
    if observed_models != set(EXPECTED_MODELS):
        raise ValueError(f"Expected six-backbone panel, got {sorted(observed_models)}")

    task_ids = {str(row["task_id"]) for row in rows}
    selection_metrics = {str(row["selection_metric"]) for row in rows}
    split_shapes = {
        (
            int(row["dataset_rows"]),
            int(row["train_rows"]),
            int(row["validation_rows"]),
            int(row["test_rows"]),
            int(row["test_positive_rows"]),
        )
        for row in rows
    }
    if len(task_ids) != 1 or len(selection_metrics) != 1 or len(split_shapes) != 1:
        raise ValueError("Cross-model task contract mismatch")
    task_id = next(iter(task_ids))
    selection_metric = next(iter(selection_metrics))
    dataset_rows, train_rows, validation_rows, test_rows, positive_rows = next(iter(split_shapes))

    reference = arrays[0]
    for arrays_item in arrays[1:]:
        for key in ("label", "example_id", "tfidf_word_char"):
            if not np.array_equal(reference[key], arrays_item[key]):
                raise ValueError(f"Cross-model held-out mismatch in {key}")

    validation_order = sorted(
        rows, key=lambda row: float(row[f"validation_{selection_metric}"]), reverse=True
    )
    validation_rank = {str(row["model_id"]): rank for rank, row in enumerate(validation_order, start=1)}
    test_order = sorted(
        rows, key=lambda row: float(row[f"activation_{selection_metric}"]), reverse=True
    )
    test_rank = {str(row["model_id"]): rank for rank, row in enumerate(test_order, start=1)}
    for row in rows:
        row["validation_primary_metric_rank"] = validation_rank[str(row["model_id"])]
        row["test_primary_metric_rank_exploratory"] = test_rank[str(row["model_id"])]
    rows.sort(key=lambda row: int(row["test_primary_metric_rank_exploratory"]))

    baseline = {
        "auprc": float(rows[0]["tfidf_auprc"]),
        "auroc": float(rows[0]["tfidf_auroc"]),
    }
    summary = {
        "task": task_id,
        "display_name": args.display_name,
        "selection_metric": selection_metric,
        "dataset_rows": dataset_rows,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "test_rows": test_rows,
        "test_positive_rows": positive_rows,
        "status": "COMPLETE_SIX_BACKBONE_VALIDATION_SCREEN_AND_ONE_SHOT_TEST",
        "models_completed": len(rows),
        "heldout_identity_checks": {
            "rows": len(reference["label"]),
            "positive_rows": int(reference["label"].sum()),
            "identical_example_ids": True,
            "identical_labels": True,
            "identical_tfidf_probabilities": True,
        },
        "baseline": baseline,
        "planned_comparison_counts": {
            "activation_advantage_both_metrics": sum(
                row["planned_activation_vs_tfidf_status"] == "activation_advantage_both_metrics"
                for row in rows
            ),
            "unresolved": sum(
                row["planned_activation_vs_tfidf_status"] == "unresolved" for row in rows
            ),
        },
        "readout_counts": {
            pooling: sum(row["pooling"] == pooling for row in rows)
            for pooling in ("masked_mean", "last_nonpadding", "masked_max")
        },
        "cross_model_inference": "exploratory_descriptive_only_no_posthoc_significance_claim",
        "models": rows,
    }

    output_stem = BENCHMARK / args.output_stem
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_stem.with_suffix(".json")
    csv_path = output_stem.with_suffix(".csv")
    markdown_path = output_stem.with_suffix(".md")
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "markdown": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
