"""Contract checks for the frozen one-forward/many-head experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP = (
    ROOT
    / "research"
    / "multitask-activation-benchmark"
    / "experiments"
    / "ONE-FORWARD-MULTIHEAD-EXP001"
)


def read_json(name: str) -> dict:
    """Load one experiment JSON file."""
    return json.loads((EXP / name).read_text(encoding="utf-8"))


def test_registry_contains_sixteen_independent_compatible_heads() -> None:
    """Every scaling head must be unique and use the pinned shared backbone."""
    config = read_json("config.json")
    registry = read_json("HEAD_REGISTRY.json")
    heads = registry["heads"]
    assert registry["head_count"] == 16 == len(heads)
    assert [head["order"] for head in heads] == list(range(1, 17))
    assert len({head["name"] for head in heads}) == 16
    assert len({head["probe_sha256"] for head in heads}) == 16
    assert {head["model_id"] for head in heads} == {config["backbone"]["id"]}
    assert {head["model_revision"] for head in heads} == {
        config["backbone"]["revision"]
    }
    assert {head["pooling"] for head in heads} <= {"masked_mean", "masked_max"}


def test_workload_is_frozen_real_heldout_panel() -> None:
    """Latency must be a subset of the fixed 64-row-per-source score panel."""
    workload = read_json("WORKLOAD_MANIFEST.json")
    assert workload["latency_rows"] == 80
    assert workload["ood_rows"] == 320
    assert set(workload["sources"]) == {
        "sst2",
        "toxicchat",
        "sms_spam",
        "goemotions",
        "ucc",
    }
    for source in workload["sources"].values():
        assert len(source["ood_ids"]) == 64
        assert len(source["latency_ids"]) == 16
        assert source["latency_ids"] == source["ood_ids"][:16]
        assert len(set(source["ood_ids"])) == 64


def test_complete_timing_cells_are_paired_and_bounded() -> None:
    """All K cells must use identical rows and yield finite positive timings."""
    config = read_json("config.json")
    panels = {
        k: pd.read_json(EXP / "outputs" / f"latency_k{k}.jsonl", lines=True)
        for k in config["k_values"]
    }
    reference = panels[1][["row_key", "source", "example_id", "token_count"]]
    for k, panel in panels.items():
        assert len(panel) == 80
        assert "text" not in panel.columns
        assert panel[["row_key", "source", "example_id", "token_count"]].equals(
            reference
        )
        values = panel[
            ["tokenization_ms", "forward_ms", "heads_ms", "total_ms"]
        ].to_numpy(dtype=float)
        assert np.isfinite(values).all()
        assert (values > 0).all()
        assert np.allclose(
            panel["total_ms"],
            panel["tokenization_ms"] + panel["forward_ms"] + panel["heads_ms"],
            rtol=0,
            atol=1e-6,
        ), k


def test_quality_calibration_and_ood_outputs_are_finite() -> None:
    """Native labels support quality claims; cross-domain scores remain descriptive."""
    result = read_json("result.json")
    assert result["status"] == "COMPLETE"
    assert len(result["native_quality"]) == 16
    assert len(result["ood_score_shifts"]) == 80
    for row in result["native_quality"]:
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "f1",
            "auprc",
            "auroc",
            "brier",
            "ece_10_bin",
        ):
            assert 0 <= row[metric] <= 1
    scores = pd.read_json(EXP / "outputs" / "ood_scores_k16.jsonl", lines=True)
    head_names = [row["head"] for row in result["native_quality"]]
    assert len(scores) == 320
    assert "text" not in scores.columns
    assert np.isfinite(scores[head_names].to_numpy(dtype=float)).all()
    assert ((scores[head_names] >= 0) & (scores[head_names] <= 1)).all().all()


def test_runtime_replay_matches_archived_probe_behavior() -> None:
    """Live full-precision scoring must stay close to cached float16 evaluation."""
    replay = pd.read_csv(EXP / "RUNTIME-REPLAY-CHECK.csv")
    assert len(replay) == 16
    assert replay["rows"].eq(64).all()
    assert replay["mean_absolute_difference"].max() < 0.01
    assert replay["pearson_correlation"].min() > 0.95


def test_scaling_result_preserves_declared_comparator_boundary() -> None:
    """The machine-readable result must not relabel repeated forwards as full judges."""
    result = read_json("result.json")
    interpretation = result["interpretation"]
    assert "lower bound" in interpretation["repeated_forward"]
    assert "Unlabelled" in interpretation["ood"]
    scaling = pd.DataFrame(result["scaling"])
    assert scaling["k"].tolist() == [1, 2, 4, 8, 16]
    assert (scaling["mean_total_ms"] > 0).all()
    assert (scaling["peak_rss_mib"] > 0).all()
    assert scaling.loc[scaling["k"].eq(16), "repeated_over_one_forward_ratio"].iat[0] > 10
