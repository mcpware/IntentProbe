"""Integrity tests for the frozen MEM-EXP002 development build."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "research" / "memory-write-gate" / "experiments" / "EXP002-qwen-activation-screen"
PROCESSED = REPO / "research" / "memory-write-gate" / "data" / "processed" / "exp002"


def sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(not PROCESSED.exists(), reason="run build_development.py before processed-data verification")
def test_memory_exp002_build_is_group_isolated_and_hashed() -> None:
    """Lock both development files to the frozen manifest."""
    manifest = json.loads((EXP / "development_manifest.json").read_text(encoding="utf-8"))
    seen: set[str] = set()
    for split, expected_rows, expected_seeds in (
        ("train", 14200, 142),
        ("validation", 3200, 32),
    ):
        path = PROCESSED / f"{split}.parquet"
        frame = pd.read_parquet(path)
        assert len(frame) == expected_rows
        assert frame["seed_id"].nunique() == expected_seeds
        assert sha256(path) == manifest["splits"][split]["sha256"]
        current = set(frame["seed_id"].astype(str))
        assert seen.isdisjoint(current)
        seen |= current


@pytest.mark.skipif(not PROCESSED.exists(), reason="run build_development.py before renderer verification")
def test_memory_exp002_renderer_excludes_labels_and_metadata() -> None:
    """Ensure model input contains only utterance text and constant template text."""
    for split in ("train", "validation"):
        frame = pd.read_parquet(PROCESSED / f"{split}.parquet")
        assert frame["rendered"].str.startswith("USER_UTTERANCE\n").all()
        assert frame["rendered"].str.endswith("\n\nMEMORY_WRITE_DECISION").all()
        assert set(frame["target"]) == {0, 1}


def test_memory_exp002_token_audit_has_complete_coverage() -> None:
    """Prevent extraction if the frozen renderer silently truncates development rows."""
    audit = json.loads((EXP / "TOKEN_COVERAGE_AUDIT.json").read_text(encoding="utf-8"))
    assert audit["status"] == "PASS_COMPLETE_DEVELOPMENT_COVERAGE"
    assert audit["max_tokens"] == 256
    assert audit["splits"]["train"]["maximum_rendered_tokens"] == 205
    assert audit["splits"]["validation"]["maximum_rendered_tokens"] == 180
    assert audit["splits"]["train"]["rows_over_cap"] == 0
    assert audit["splits"]["validation"]["rows_over_cap"] == 0


def test_memory_exp002_complete_candidate_search_and_global_winner() -> None:
    """Require all preregistered candidates and recompute the global winner."""
    result = json.loads((EXP / "selection_result.json").read_text(encoding="utf-8"))
    search = result["candidate_search"]
    assert result["status"] == "COMPLETE_DEVELOPMENT_SELECTION_TEST_NOT_TOUCHED"
    assert search["total_candidates"] == 216
    candidates = []
    keys: set[tuple[str, str]] = set()
    for pooling in ("masked_mean", "last_nonpadding", "masked_max"):
        pooling_result = json.loads(
            (EXP / "pooling_results" / f"{pooling}.json").read_text(encoding="utf-8")
        )
        assert pooling_result["status"] == "COMPLETED"
        assert pooling_result["completed_candidates"] == 72
        assert len(pooling_result["candidates"]) == 72
        for candidate in pooling_result["candidates"]:
            identity = (pooling, candidate["key"])
            assert identity not in keys
            keys.add(identity)
            candidates.append({**candidate, "pooling": pooling})
    expected = max(candidates, key=lambda item: item["metrics"]["auprc"])
    assert search["global_winner"] == expected
    assert expected["pooling"] == "masked_mean"
    assert expected["key"] == "individual:11:11"


def test_memory_exp002_selected_artifacts_and_metrics_are_consistent() -> None:
    """Recompute reported development metrics and lock artifact hashes."""
    result = json.loads((EXP / "selection_result.json").read_text(encoding="utf-8"))
    artifacts = result["artifacts"]
    for record in artifacts.values():
        assert sha256(EXP / record["path"]) == record["sha256"]
    candidate_table = pd.read_csv(EXP / "candidate_table.csv")
    assert candidate_table.shape[0] == 216
    assert candidate_table[["pooling", "key"]].duplicated().sum() == 0
    with np.load(EXP / "validation_scores.npz") as scores:
        y = scores["y"]
        activation = scores["activation"]
        tfidf = scores["tfidf_word_char"]
        assert y.shape == activation.shape == tfidf.shape == (3200,)
        reported = result["development_evaluation"]
        assert abs(
            average_precision_score(y, activation)
            - reported["selected_activation"]["threshold_0_5"]["auprc"]
        ) < 1e-12
        assert abs(
            roc_auc_score(y, activation)
            - reported["selected_activation"]["threshold_0_5"]["auroc"]
        ) < 1e-12
        assert abs(
            average_precision_score(y, activation)
            - average_precision_score(y, tfidf)
            - reported["paired_activation_minus_tfidf"]["auprc_difference"]
        ) < 1e-12
    with np.load(EXP / "selected_probe.npz") as probe:
        assert probe["coefficient"].shape == (1, 896)
        assert probe["train_means"].shape == (1, 896)
        assert probe["train_scales"].shape == (1, 896)
        assert int(probe["start_layer"]) == int(probe["end_layer"]) == 11
        assert str(probe["pooling"]) == "masked_mean"


def test_memory_exp002_uncertainty_is_seed_clustered_and_development_only() -> None:
    """Lock uncertainty grain and prevent accidental confirmatory test claims."""
    result = json.loads((EXP / "selection_result.json").read_text(encoding="utf-8"))
    evaluation = result["development_evaluation"]
    assert evaluation["split"] == "validation"
    assert evaluation["rows"] == 3200
    assert evaluation["seed_families"] == 32
    intervals = evaluation["selected_activation"]["clustered_intervals"]
    paired = evaluation["paired_activation_minus_tfidf"]
    for record in (intervals, paired):
        assert record["unit"] == "seed_id"
        assert record["resamples"] == 2000
        assert record["seed"] == 20261017
    assert "post-selection" in result["selection_boundary"]
    assert "official test was not loaded" in result["selection_boundary"]


def test_memory_exp002_weighted_rank_metrics_match_sklearn() -> None:
    """Validate the vectorized paired-bootstrap metric implementation."""
    spec = importlib.util.spec_from_file_location(
        "memory_exp002_select", EXP / "select_candidate.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with np.load(EXP / "validation_scores.npz") as scores:
        y = scores["y"]
        probabilities = scores["activation"]
    weights = np.ones((1, y.size), dtype=float)
    auroc, auprc = module.weighted_rank_metrics(y, probabilities, weights)
    assert abs(float(auroc[0]) - roc_auc_score(y, probabilities)) < 1e-12
    assert abs(float(auprc[0]) - average_precision_score(y, probabilities)) < 1e-12
