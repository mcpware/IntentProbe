#!/usr/bin/env python3
"""Frozen-weight benchmark on a pinned, previously unused agent-skill corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

try:
    from .activation_scanner_core import (
        DECISION_POLICY_NAME,
        choose_runtime_decision,
        extract_runtime_matrix_for_texts,
        load_artifact,
        positive_probabilities,
        runtime_args,
    )
    from .benchmark_hf_release import (
        auc_difference,
        auc_statistic,
        cluster_bootstrap,
        label_array,
        score_metrics,
        sha256_file,
        stratified_bootstrap,
        threshold_metrics,
    )
    from .benchmarks.datasets import Example, normalize_text
    from .scanner_static import static_scan, static_score
    from .train_probe_artifact import load_training_examples
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from research.activation_scanner_core import (  # type: ignore
        DECISION_POLICY_NAME,
        choose_runtime_decision,
        extract_runtime_matrix_for_texts,
        load_artifact,
        positive_probabilities,
        runtime_args,
    )
    from research.benchmark_hf_release import (  # type: ignore
        auc_difference,
        auc_statistic,
        cluster_bootstrap,
        label_array,
        score_metrics,
        sha256_file,
        stratified_bootstrap,
        threshold_metrics,
    )
    from research.benchmarks.datasets import Example, normalize_text  # type: ignore
    from research.scanner_static import static_scan, static_score  # type: ignore
    from research.train_probe_artifact import load_training_examples  # type: ignore


DATASET_ID = "yoonholee/agent-skill-malware"
DATASET_REVISION = "5cff435ddab2cc34261875885de9ac50d40393ec"
DATASET_FILE = "skills.jsonl"
DATASET_FILE_SHA256 = "c7ae101d3d37f53af7dc7b2afc5ea15fe7c51794ef879bdbf2a7f5a342d4e3db"

DEFAULT_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "intentprobe"
    / "scanner"
    / "artifacts"
    / "qwen-pooled-curated-core-l13-15-v2"
)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "datasets"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "_results_published" / "hf_blind_agent_skill_malware.json"

# This is the dataset author's intentionally dataset-specific reference rule,
# reproduced to provide the strongest transparent lexical baseline. It must not
# be interpreted as an out-of-distribution detector.
DATASET_SIGNATURE = re.compile(
    "|".join(
        [
            r"openclawcli",
            r"openclaw-core",
            r"glot\.io/snippets",
            r"rentry\.co/openclaw",
            r"Ddoy233",
            r"denboss99",
            r"base64\s+-[dD]",
            r"91\.92\.242\.30",
            r"openclawcli\.vercel",
            r"webhook\.site",
            r"setup-service\.com",
            r"AuthTool",
            r"pass(?:word)?[:\s]+.{0,10}openclaw",
        ]
    ),
    re.IGNORECASE,
)


def load_pinned_rows() -> tuple[list[dict[str, Any]], Path]:
    path = Path(
        hf_hub_download(
            repo_id=DATASET_ID,
            filename=DATASET_FILE,
            repo_type="dataset",
            revision=DATASET_REVISION,
        )
    )
    actual_sha = sha256_file(path)
    if actual_sha != DATASET_FILE_SHA256:
        raise SystemExit(f"Pinned dataset checksum mismatch: expected {DATASET_FILE_SHA256}, got {actual_sha}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows, path


def as_examples(rows: list[dict[str, Any]]) -> list[Example]:
    label_map = {"benign": 0, "malicious": 1}
    examples: list[Example] = []
    for index, row in enumerate(rows):
        raw_label = str(row.get("label", "")).lower()
        if raw_label not in label_map:
            raise SystemExit(f"Unexpected dataset label at row {index}: {raw_label!r}")
        content = str(row.get("content", ""))
        if not content.strip():
            raise SystemExit(f"Empty content at row {index}")
        skill_name = str(row.get("skill_name") or f"row-{index}")
        examples.append(
            Example(
                text=content,
                label=label_map[raw_label],
                style="full_skill_md",
                source=f"{DATASET_ID}@{DATASET_REVISION}",
                index=index,
                record_id=str(row.get("id") or f"row-{index}"),
                family="real_campaign_skill_malware",
                source_type="public_dataset",
                split_group=skill_name,
                notes="Pinned Hugging Face agent-skill-malware row",
            )
        )
    return examples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    artifact_dir, metadata, weights = load_artifact(args.artifact)
    raw_rows, dataset_path = load_pinned_rows()
    all_rows = as_examples(raw_rows)
    train_rows = load_training_examples(str(metadata["dataset"]["train_source"]), args.data_dir)
    train_normalized = {normalize_text(row.text) for row in train_rows}
    overlap_mask = np.asarray([normalize_text(row.text) in train_normalized for row in all_rows], dtype=bool)
    release_indices = np.flatnonzero(~overlap_mask)
    rows = [all_rows[int(index)] for index in release_indices]

    runtime_ns = runtime_args(
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    matrix, activation_details = extract_runtime_matrix_for_texts(
        texts=[row.text for row in all_rows],
        metadata=metadata,
        args=runtime_ns,
    )
    activation_all = positive_probabilities(matrix, weights)
    static_all = np.asarray([static_score(static_scan(row.text)) for row in all_rows], dtype=np.float64)
    risk_all = np.maximum(activation_all, static_all)
    signature_all = np.asarray([float(bool(DATASET_SIGNATURE.search(row.text))) for row in all_rows])

    tfidf = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2),
        LogisticRegression(max_iter=5000, random_state=args.seed),
    )
    tfidf.fit([row.text for row in train_rows], label_array(train_rows))
    tfidf_all = tfidf.predict_proba([row.text for row in all_rows])[:, 1]

    y = label_array(rows)
    activation = activation_all[release_indices]
    static = static_all[release_indices]
    risk = risk_all[release_indices]
    signature = signature_all[release_indices]
    tfidf_scores = tfidf_all[release_indices]
    groups = np.asarray([row.split_group or row.record_id for row in rows], dtype=str)
    warn_threshold = float(metadata["thresholds"]["warn"])
    block_threshold = float(metadata["thresholds"]["block"])

    decisions: list[str] = []
    for row, activation_value, static_value, risk_value in zip(rows, activation, static, risk, strict=True):
        findings = static_scan(row.text)
        decision, _ = choose_runtime_decision(
            activation_score=float(activation_value),
            static_risk_score=float(static_value),
            risk_score=float(risk_value),
            static_findings=findings,
            warn_threshold=warn_threshold,
            block_threshold=block_threshold,
        )
        decisions.append(decision)

    score_sets = {
        "activation": activation,
        "static": static,
        "risk": risk,
        "tfidf": tfidf_scores,
        "dataset_signature": signature,
    }
    repeats = int(args.bootstrap_repeats)
    score_report: dict[str, Any] = {}
    for offset, (name, scores) in enumerate(score_sets.items()):
        score_report[name] = {
            **score_metrics(y, scores),
            "auroc_ci95_stratified_example_bootstrap": stratified_bootstrap(
                y,
                [scores],
                auc_statistic,
                repeats=repeats,
                seed=args.seed + offset,
            ),
            "auroc_cluster_bootstrap": cluster_bootstrap(
                y,
                [scores],
                groups,
                auc_statistic,
                repeats=repeats,
                seed=args.seed + 100 + offset,
            ),
        }

    comparison = {
        "activation_minus_tfidf_auroc": float(score_report["activation"]["auroc"] - score_report["tfidf"]["auroc"]),
        "ci95_stratified_example_bootstrap": stratified_bootstrap(
            y,
            [activation, tfidf_scores],
            auc_difference,
            repeats=repeats,
            seed=args.seed + 1000,
        ),
        "cluster_bootstrap": cluster_bootstrap(
            y,
            [activation, tfidf_scores],
            groups,
            auc_difference,
            repeats=repeats,
            seed=args.seed + 2000,
        ),
    }

    warn_pred = np.asarray([decision in {"warn", "block", "quarantine"} for decision in decisions], dtype=np.int64)
    block_pred = np.asarray([decision == "block" for decision in decisions], dtype=np.int64)
    report = {
        "schema": "intentprobe-hf-blind-agent-skills-v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": {
            "frozen_weight": True,
            "weight_layer_or_threshold_changed_by_this_run": False,
            "dataset_selected_before_intentprobe_scores_were_computed": True,
            "exact_normalized_training_overlaps_excluded": True,
            "interpretation": "first frozen-weight evaluation on this pinned corpus",
            "decision_policy": DECISION_POLICY_NAME,
            "warn_threshold": warn_threshold,
            "block_threshold": block_threshold,
            "bootstrap_repeats": repeats,
            "seed": args.seed,
        },
        "artifact": {
            "artifact_id": metadata.get("artifact_id"),
            "weights_sha256": sha256_file(artifact_dir / "probe_weights.npz"),
            "metadata_sha256": sha256_file(artifact_dir / "metadata.json"),
        },
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "file": DATASET_FILE,
            "file_sha256": sha256_file(dataset_path),
            "license_metadata": "mit",
            "rows_before_exclusion": len(all_rows),
            "labels_before_exclusion": dict(Counter(row.label for row in all_rows)),
            "exact_normalized_training_overlaps": int(np.sum(overlap_mask)),
            "excluded_ids": [all_rows[int(index)].record_id for index in np.flatnonzero(overlap_mask)],
            "rows_after_exclusion": len(rows),
            "clean_after_exclusion": int(np.sum(y == 0)),
            "malicious_after_exclusion": int(np.sum(y == 1)),
            "skill_name_groups": int(len(np.unique(groups))),
        },
        "score_metrics": score_report,
        "paired_comparison": comparison,
        "policy_metrics": {
            "warn_or_block": threshold_metrics(y, warn_pred),
            "block_only": threshold_metrics(y, block_pred),
            "dataset_signature": threshold_metrics(y, signature.astype(np.int64)),
            "decision_counts": dict(Counter(decisions)),
        },
        "runtime": {
            "elapsed_seconds": float(time.perf_counter() - started),
            "feature_elapsed_seconds": float(activation_details["feature_elapsed_seconds"]),
            "activation_details": activation_details,
        },
        "scored_rows": [
            {
                "id": row.record_id,
                "skill_name_group": row.split_group,
                "label": int(row.label),
                "activation_score": float(activation[index]),
                "static_score": float(static[index]),
                "risk_score": float(risk[index]),
                "tfidf_score": float(tfidf_scores[index]),
                "dataset_signature": int(signature[index]),
                "policy_decision": decisions[index],
            }
            for index, row in enumerate(rows)
        ],
        "limitations": [
            "The dataset has one split; this run treats the full pinned file as evaluation-only and excludes exact training overlaps.",
            "Malicious rows are concentrated in a small number of campaigns and recurring lexical indicators.",
            "The dataset-authored signature baseline is fitted to known indicators in this same corpus and is not an out-of-distribution baseline.",
            "Cluster bootstrap uses skill_name as the dependency group; related campaigns may still span multiple names.",
            "The dataset card says 350 rows (127 malicious, 223 benign), while the pinned file contains 347 rows (124 malicious, 223 benign).",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2 if args.pretty else None, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "dataset": report["dataset"],
                "score_metrics": report["score_metrics"],
                "paired_comparison": report["paired_comparison"],
                "policy_metrics": report["policy_metrics"],
                "runtime": report["runtime"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
