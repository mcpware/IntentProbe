#!/usr/bin/env python3
"""Build the multi-task paper's compact result ledger from frozen artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
BENCHMARK = PAPER.parent
EXPERIMENTS = BENCHMARK / "experiments"
OUTPUT = PAPER / "DERIVED_RESULTS.json"


def load(path: Path) -> dict[str, Any]:
    """Load one JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Derive paper-level counts and ranges without copying values by hand."""
    sentiment = load(EXPERIMENTS / "SENTIMENT-SST2-MODEL-MATRIX.json")
    spam = load(EXPERIMENTS / "SPAM-SMS-UCI-MODEL-MATRIX.json")
    toxicchat = load(EXPERIMENTS / "TOXICCHAT0124-MODEL-TASK-MATRIX.json")
    sentinel = load(EXPERIMENTS / "SENTINEL-JUDGE-VS-PROBE-EXP001" / "result.json")
    scaling = load(EXPERIMENTS / "ONE-FORWARD-MULTIHEAD-EXP001" / "result.json")
    registry = load(EXPERIMENTS / "ONE-FORWARD-MULTIHEAD-EXP001" / "HEAD_REGISTRY.json")

    quality = registry["heads"]
    auprc = np.asarray([row["native_heldout"]["auprc"] for row in quality])
    auroc = np.asarray([row["native_heldout"]["auroc"] for row in quality])
    prevalence = np.asarray(
        [row["native_heldout"]["positive_prevalence"] for row in quality]
    )
    normalized_depth = np.asarray([row["normalized_depth"] for row in quality])
    k1 = next(row for row in scaling["scaling"] if row["k"] == 1)
    k16 = next(row for row in scaling["scaling"] if row["k"] == 16)

    result = {
        "schema_version": 1,
        "paper": "one-forward-sixteen-decisions",
        "status": "DERIVED_FROM_COMPLETE_FROZEN_ARTIFACTS",
        "cross_backbone": {
            "binary_cells": 24,
            "backbones": 6,
            "tasks": 4,
            "sentiment": {
                "cells": sentiment["models_completed"],
                "readout_counts": sentiment["readout_counts"],
                "normalized_depth_range": [
                    min(row["normalized_depth"] for row in sentiment["models"]),
                    max(row["normalized_depth"] for row in sentiment["models"]),
                ],
                "planned_comparison_counts": sentiment["planned_comparison_counts"],
            },
            "spam": {
                "cells": spam["models_completed"],
                "readout_counts": spam["readout_counts"],
                "normalized_depth_range": [
                    min(row["normalized_depth"] for row in spam["models"]),
                    max(row["normalized_depth"] for row in spam["models"]),
                ],
                "planned_comparison_counts": spam["planned_comparison_counts"],
            },
            "toxicchat": toxicchat,
        },
        "shared_qwen_suite": {
            "heads": registry["head_count"],
            "sources": sorted({row["native_source"] for row in quality}),
            "pooling_counts": {
                method: sum(row["pooling"] == method for row in quality)
                for method in ("masked_mean", "masked_max")
            },
            "normalized_depth_range": [
                float(normalized_depth.min()),
                float(normalized_depth.max()),
            ],
            "native_auprc_range": [float(auprc.min()), float(auprc.max())],
            "native_auroc_range": [float(auroc.min()), float(auroc.max())],
            "native_prevalence_range": [
                float(prevalence.min()),
                float(prevalence.max()),
            ],
        },
        "sentinel": {
            "expected_cells": sentinel["expected_cells"],
            "completed_cells": sentinel["completed_cells"],
            "fdr": sentinel["fdr"],
            "summary_artifact": "../experiments/SENTINEL-JUDGE-VS-PROBE-EXP001/RESULT_SUMMARY.md",
        },
        "systems": {
            "k_values": scaling["k_values"],
            "latency_rows": scaling["latency_rows"],
            "ood_rows": scaling["ood_rows"],
            "k1": k1,
            "k16": k16,
            "runtime_replay_worst_mean_absolute_difference": max(
                row["mean_absolute_difference"] for row in scaling["runtime_replay"]
            ),
            "runtime_replay_minimum_pearson": min(
                row["pearson_correlation"] for row in scaling["runtime_replay"]
            ),
        },
        "claim_boundaries": {
            "repeated_forward": scaling["interpretation"]["repeated_forward"],
            "ood": scaling["interpretation"]["ood"],
            "missing_system_arms": [
                "one prompted multi-label judge call",
                "measured fleet of compact task-specific classifiers",
            ],
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
