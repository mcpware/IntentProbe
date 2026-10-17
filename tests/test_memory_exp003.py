"""Integrity tests for the frozen LUFY human-transfer receipt."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "research" / "memory-write-gate" / "experiments" / "EXP003-lufy-human-transfer"


def load_analysis_module():
    """Load the experiment analysis helpers without modifying package paths."""
    spec = importlib.util.spec_from_file_location("memory_exp003_analyze", EXP / "analyze.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_memory_exp003_receipt_locks_external_transfer_design() -> None:
    """Require the frozen human-written design and aggregate sample counts."""
    config = json.loads((EXP / "config.json").read_text(encoding="utf-8"))
    result = json.loads((EXP / "result.json").read_text(encoding="utf-8"))
    view = result["views"]["all_majority_vote"]
    assert config["status"] == "FROZEN_BEFORE_SCORING"
    assert config["cluster_unit"] == "user_name"
    assert result["status"] == "COMPLETE_EXTERNAL_HUMAN_TRANSFER"
    assert view["rows"] == 6470
    assert view["users"] == 17
    assert view["positive_rows"] == 332
    assert "zero-shot" in result["selection_use"]


def test_memory_exp003_paired_intervals_do_not_support_superiority() -> None:
    """Prevent a positive point estimate from becoming an unsupported win claim."""
    result = json.loads((EXP / "result.json").read_text(encoding="utf-8"))
    paired = result["views"]["all_majority_vote"]["paired_activation_minus_tfidf"]
    assert paired["auprc_difference"] > 0
    assert paired["auroc_difference"] > 0
    assert paired["auprc_difference_user_bootstrap_95ci"][0] < 0
    assert paired["auprc_difference_user_bootstrap_95ci"][1] > 0
    assert paired["auroc_difference_user_bootstrap_95ci"][0] < 0
    assert paired["auroc_difference_user_bootstrap_95ci"][1] > 0
    assert paired["resamples"] == 2000
    assert paired["unit"] == "user_name"


def test_memory_exp003_vectorized_weighted_metrics_match_sklearn() -> None:
    """Check weighted AP/AUROC, including tied scores, against row expansion."""
    module = load_analysis_module()
    y = np.array([0, 1, 0, 1, 1, 0], dtype=int)
    score = np.array([0.1, 0.8, 0.4, 0.8, 0.3, 0.4])
    weights = np.array([[1, 2, 1, 1, 0, 3], [2, 1, 0, 3, 2, 1]], dtype=int)
    auprc, auroc = module.weighted_rank_bootstrap(y, score, weights)
    first = np.repeat(np.arange(len(y)), weights[0])
    second = np.repeat(np.arange(len(y)), weights[1])
    expected = np.array(
        [
            [average_precision_score(y[first], score[first]), roc_auc_score(y[first], score[first])],
            [
                average_precision_score(y[second], score[second]),
                roc_auc_score(y[second], score[second]),
            ],
        ]
    )
    np.testing.assert_allclose(np.column_stack([auprc, auroc]), expected, atol=1e-12, rtol=0)
