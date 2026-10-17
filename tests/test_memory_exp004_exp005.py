"""Integrity tests for the human-trained LUFY experiments."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP004 = ROOT / "research" / "memory-write-gate" / "experiments" / "EXP004-lufy-human-trained"
EXP005 = ROOT / "research" / "memory-write-gate" / "experiments" / "EXP005-lufy-human-readout-screen"
HF = ROOT / "hf" / "intentprobe-qwen2.5-0.5b-memory-write"


def test_exp004_human_training_improves_over_zero_shot_but_not_tfidf() -> None:
    result = json.loads((EXP004 / "result.json").read_text(encoding="utf-8"))
    activation = result["models"]["activation"]
    tfidf = result["models"]["tfidf_word_char"]
    paired = result["paired_activation_minus_tfidf"]
    assert result["rows"] == 6470
    assert result["users"] == 17
    assert result["positive_rows"] == 332
    assert activation["auroc"] > 0.70
    assert activation["auprc"] > result["positive_prevalence"]
    assert activation["auroc"] < tfidf["auroc"]
    assert paired["auroc_difference_user_bootstrap_95ci"][1] < 0
    assert paired["auprc_difference_user_bootstrap_95ci"][1] < 0


def test_exp005_complete_candidate_table_and_development_winner() -> None:
    result = json.loads((EXP005 / "result.json").read_text(encoding="utf-8"))
    table = pd.read_csv(EXP005 / "candidate_table.csv")
    assert result["candidate_count"] == len(table) == 216
    assert table[["pooling", "start_layer", "end_layer"]].duplicated().sum() == 0
    winner = table.sort_values("rank_development_auprc").iloc[0]
    reported = result["development"]["winner"]
    assert winner["pooling"] == reported["pooling"] == "masked_mean"
    assert int(winner["start_layer"]) == reported["start_layer"] == 7
    assert int(winner["end_layer"]) == reported["end_layer"] == 8
    assert int((table.head(20)["pooling"] == "masked_mean").sum()) == 20


def test_exp005_one_shot_test_does_not_claim_activation_superiority() -> None:
    result = json.loads((EXP005 / "result.json").read_text(encoding="utf-8"))
    test = result["heldout_test"]
    paired = test["paired_activation_minus_tfidf"]
    assert test["rows"] == 1748
    assert test["users"] == 5
    assert test["positive_rows"] == 91
    assert paired["auprc_difference_user_bootstrap_95ci"][0] < 0
    assert paired["auprc_difference_user_bootstrap_95ci"][1] > 0
    assert paired["auroc_difference_user_bootstrap_95ci"][0] < 0
    assert paired["auroc_difference_user_bootstrap_95ci"][1] > 0
    assert test["models"]["activation"]["balanced_accuracy_at_0_5"] > test["models"][
        "tfidf_word_char"
    ]["balanced_accuracy_at_0_5"]


def test_exp005_post_test_anchor_is_labelled_and_matches_hf_summary() -> None:
    diagnostic = json.loads(
        (EXP005 / "posthoc_output11_anchor.json").read_text(encoding="utf-8")
    )
    summary = json.loads((HF / "human_readout_research_summary.json").read_text(encoding="utf-8"))
    difference = diagnostic["selected_minus_output11"]
    assert diagnostic["status"] == "POST_TEST_DIAGNOSTIC_NOT_CONFIRMATORY"
    assert difference["auprc_difference_user_bootstrap_95ci"][0] > 0
    assert difference["auroc_difference_user_bootstrap_95ci"][0] > 0
    assert summary["post_test_output11_diagnostic"]["selected_minus_output11_auprc"] == difference[
        "auprc_difference"
    ]
    assert summary["post_test_output11_diagnostic"]["selected_minus_output11_auroc"] == difference[
        "auroc_difference"
    ]
