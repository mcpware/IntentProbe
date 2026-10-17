"""Integrity tests for Memory Write Gate MEM-EXP001."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "research" / "memory-write-gate" / "experiments" / "EXP001-surface-baselines"
RAW = (
    REPO
    / "research"
    / "memory-write-gate"
    / "data"
    / "raw"
    / "huggingface"
    / "memory-write-gate-20k"
    / "data"
    / "memory_write_gate_20000.jsonl"
)


def sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(not RAW.exists(), reason="run the pinned source fetch before raw-hash verification")
def test_memory_exp001_frozen_inputs_and_group_counts() -> None:
    """Lock the consumed experiment to the registered source and group grain."""
    result = json.loads((EXP / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "COMPLETE_TEST_CONSUMED"
    assert result["input_hashes"]["config"] == sha256(EXP / "config.json")
    assert result["input_hashes"]["raw"] == sha256(RAW)
    assert result["rows"] == {"test": 2600, "train": 14200, "validation": 3200}
    assert result["seed_families"] == {"test": 26, "train": 142, "validation": 32}
    assert not any(result["normalized_exact_text_overlap"].values())


def test_memory_exp001_reports_seed_clustered_intervals() -> None:
    """Require every baseline to report test uncertainty at the seed level."""
    result = json.loads((EXP / "result.json").read_text(encoding="utf-8"))
    for model in result["models"].values():
        intervals = model["test"]["clustered_intervals"]
        assert intervals["unit"] == "seed_id"
        assert intervals["resamples"] == 2000
        assert len(intervals["auprc_95ci"]) == 2
        assert len(intervals["auroc_95ci"]) == 2


def test_memory_exp001_safe_threshold_is_validation_selected() -> None:
    """Check that the same frozen threshold is used on validation and test."""
    result = json.loads((EXP / "result.json").read_text(encoding="utf-8"))
    for model in result["models"].values():
        validation_threshold = model["validation"]["safe_threshold"]["threshold"]
        test_threshold = model["test"]["safe_threshold"]["threshold"]
        assert validation_threshold == test_threshold
        assert model["validation"]["safe_threshold"]["false_save_rate"] <= 0.01


def test_vectorized_weighted_rank_metrics_match_sklearn() -> None:
    """Validate the vectorized bootstrap metric implementation at unit weights."""
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    spec = importlib.util.spec_from_file_location("memory_exp001_run", EXP / "run.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scores = np.load(EXP / "scores.npz")
    y = scores["test_y"]
    probabilities = scores["test_tfidf_word_char"]
    weights = np.ones((1, y.size), dtype=float)
    auroc, auprc = module.weighted_rank_metrics(y, probabilities, weights)
    assert abs(float(auroc[0]) - roc_auc_score(y, probabilities)) < 1e-12
    assert abs(float(auprc[0]) - average_precision_score(y, probabilities)) < 1e-12
