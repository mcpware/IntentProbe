"""Bind the multi-task paper's headline claims to frozen result artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "research" / "multitask-activation-benchmark"
PAPER = BENCH / "paper" / "DRAFT.md"
DERIVED = BENCH / "paper" / "DERIVED_RESULTS.json"
SCALING = BENCH / "experiments" / "ONE-FORWARD-MULTIHEAD-EXP001" / "result.json"
SENTINEL = BENCH / "experiments" / "SENTINEL-JUDGE-VS-PROBE-EXP001" / "result.json"


def load(path: Path) -> dict:
    """Load one JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_paper() -> str:
    """Collapse Markdown wrapping while retaining displayed values."""
    return " ".join(PAPER.read_text(encoding="utf-8").split())


def test_paper_identity_and_scope_are_explicit() -> None:
    """Keep the public author handle and contribution boundary visible."""
    paper = PAPER.read_text(encoding="utf-8")
    assert "**Author:** [Wan Sheng Leung](https://github.com/ithiria894)" in paper
    assert "MCPware, a product and research brand of GEASS LABS INC." in paper
    assert "We do not claim to invent activation probing" in paper
    assert "first ever" not in paper.lower()


def test_derived_ledger_matches_complete_system_artifact() -> None:
    """Paper-level ranges and K endpoints must come from the frozen result."""
    derived = load(DERIVED)
    scaling = load(SCALING)
    registry = load(
        BENCH
        / "experiments"
        / "ONE-FORWARD-MULTIHEAD-EXP001"
        / "HEAD_REGISTRY.json"
    )
    assert derived["status"] == "DERIVED_FROM_COMPLETE_FROZEN_ARTIFACTS"
    assert derived["shared_qwen_suite"]["heads"] == registry["head_count"] == 16
    assert derived["systems"]["k_values"] == scaling["k_values"] == [1, 2, 4, 8, 16]
    assert derived["systems"]["k16"] == scaling["scaling"][-1]
    matched = derived["systems"]["matched_k16_comparator"]
    assert matched["rows"] == 10
    assert round(matched["shared_mean_ms"], 2) == 62.70
    assert round(matched["repeated_mean_ms"], 2) == 956.71
    assert round(matched["repeated_over_shared_ratio"], 2) == 15.26


def test_systems_headline_numbers_match_result() -> None:
    """Bind latency, uncertainty, RSS, and artifact-size claims exactly."""
    paper = normalized_paper()
    result = load(SCALING)
    k16 = next(row for row in result["scaling"] if row["k"] == 16)
    assert f"{k16['mean_total_ms']:.2f} ms" in paper
    assert f"{k16['one_forward_mean_ratio_vs_k1']:.3f}" in paper
    assert (
        f"{k16['one_forward_mean_ratio_ci_low']:.3f}--"
        f"{k16['one_forward_mean_ratio_ci_high']:.3f}"
    ) in paper
    assert f"{k16['repeated_forward_mean_ms']:.2f} ms" in paper
    matched = load(DERIVED)["systems"]["matched_k16_comparator"]
    assert f"{matched['shared_mean_ms']:.2f} ms" in paper
    assert f"ratio of {k16['repeated_over_one_forward_ratio']:.2f}" in paper
    assert f"{k16['peak_rss_delta_vs_k1_mib']:+.2f} MiB" in paper
    assert f"{int(k16['cumulative_probe_bytes']):,} bytes" in paper


def test_sentinel_adverse_authorization_claim_matches_result() -> None:
    """Bind the corrected Qwen3.5 authorization comparison to saved scores."""
    paper = normalized_paper()
    result = load(SENTINEL)
    comparison = next(
        row
        for row in result["paired_comparisons"]
        if row["model"] == "quality_anchor"
        and row["task"] == "authorization_to_execute"
        and row["comparison"]
        == "activation_head-minus-judge_zero_shot_choice_logprob"
    )
    low, high = comparison["auprc_difference_ci"]
    assert f"{comparison['auprc_difference']:+.4f} AUPRC" in paper
    assert f"[{low:.4f}, {high:.4f}]" in paper
    assert f"q={comparison['auprc_bh_q']:.4f}" in paper
    assert result["fdr"]["family_size"] == 12
    assert result["fdr"]["rejections"] == 6


def test_missing_system_arms_remain_visible() -> None:
    """The partial comparator must not be promoted to the complete frontier."""
    paper = normalized_paper()
    derived = load(DERIVED)
    for arm in derived["claim_boundaries"]["missing_system_arms"]:
        assert arm in paper
    assert "repeated-forward lower bound, not a full LLM-judge benchmark" in paper
