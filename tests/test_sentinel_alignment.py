from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EXP = (
    ROOT
    / "research/multitask-activation-benchmark/experiments/SENTINEL-JUDGE-VS-PROBE-EXP001"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("sentinel_run_judge", EXP / "run_judge.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_summarizer():
    spec = importlib.util.spec_from_file_location(
        "sentinel_summarize", EXP / "summarize_results.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_manifest_matches_config_and_expected_cells() -> None:
    runner = load_runner()
    config, manifest = runner.load_contract()
    assert set(config["models"]) == {"quality_anchor", "cost_anchor"}
    assert set(config["tasks"]) == {
        "authorization_to_execute",
        "toxicity",
        "sentiment",
    }
    assert manifest["tasks"]["authorization_to_execute"]["summary"]["rows"] == 156
    assert manifest["tasks"]["toxicity"]["summary"]["rows"] == 512
    assert manifest["tasks"]["sentiment"]["summary"]["rows"] == 512


def test_frozen_samples_load_with_exact_ids_and_labels() -> None:
    runner = load_runner()
    required_sources = (runner.SST2, runner.TOXICCHAT, runner.DELEGATION_RUN)
    missing = [path for path in required_sources if not path.exists()]
    if missing:
        pytest.skip("Raw third-party evaluation sources are not redistributed")
    _, manifest = runner.load_contract()
    for task in ("authorization_to_execute", "toxicity", "sentiment"):
        frame = runner.load_task_frame(task, manifest)
        assert frame["id"].is_unique
        assert set(frame["y"].unique()) == {0, 1}
        assert len(frame) == manifest["tasks"][task]["summary"]["rows"]


def test_vectorized_prompt_renderers_have_frozen_boundaries() -> None:
    runner = load_runner()
    texts = np.asarray(["first", "second"])
    cheap = runner.render_prompts(texts, system_prompt="judge", model_alias="cost_anchor")
    quality = runner.render_prompts(texts, system_prompt="judge", model_alias="quality_anchor")
    assert cheap.shape == (2,)
    assert cheap[0].startswith("<|startoftext|><|im_start|>system\njudge")
    assert cheap[1].endswith("second<|im_end|>\n<|im_start|>assistant\n")
    assert quality[0].startswith("<|im_start|>system\njudge")
    assert quality[1].endswith(
        "second<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def test_strict_generated_label_parser_counts_nonlabels_invalid() -> None:
    runner = load_runner()
    predictions, invalid = runner.parse_generated_labels(
        np.asarray(["YES", " no \n", "YES because", "MAYBE", "YES.", ""])
    )
    assert predictions.tolist() == [1, 0, 1, 0, 0, 0]
    assert invalid.tolist() == [False, False, False, True, True, True]


def test_binary_sequence_score_normalization_is_stable() -> None:
    runner = load_runner()
    probabilities = runner.normalized_yes_probability(
        np.asarray([1000.0, -1000.0, 4.0]),
        np.asarray([999.0, -999.0, 4.0]),
    )
    assert np.allclose(probabilities, [0.73105858, 0.26894142, 0.5])


def test_content_cap_is_applied_before_prompt_rendering() -> None:
    runner = load_runner()

    class TinyTokenizer:
        def __call__(self, texts, **kwargs):
            cap = kwargs["max_length"]
            return {"input_ids": [list(range(min(len(text), cap))) for text in texts]}

        def batch_decode(self, rows, **kwargs):
            return ["x" * len(row) for row in rows]

    bounded = runner.truncate_contents(
        TinyTokenizer(), np.asarray(["a" * 20, "b" * 3]), max_content_tokens=5
    )
    assert bounded.tolist() == ["xxxxx", "xxx"]
    prompts = runner.render_prompts(
        bounded, system_prompt="judge", model_alias="cost_anchor"
    )
    assert prompts[0].endswith("xxxxx<|im_end|>\n<|im_start|>assistant\n")


def test_prompts_are_present_for_every_frozen_cell() -> None:
    config = json.loads((EXP / "config.json").read_text(encoding="utf-8"))
    for task in config["tasks"].values():
        assert set(task["prompts"]) == {"zero_shot", "rubric"}
        assert all("YES" in prompt and "NO" in prompt for prompt in task["prompts"].values())


def test_vectorized_bootstrap_rank_metrics_match_sklearn_without_ties() -> None:
    summarizer = load_summarizer()
    y = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.int8)
    score = np.asarray([0.1, 0.8, 0.2, 0.9, 0.7, 0.3], dtype=np.float64)
    indices = np.arange(len(y), dtype=np.int32)[None, :]
    ap, auc = summarizer.vectorized_rank_metrics(y, score, indices)
    metrics = summarizer.point_metrics(y, score)
    assert np.allclose(ap, [metrics["auprc"]])
    assert np.allclose(auc, [metrics["auroc"]])


def test_group_weighted_rank_metrics_match_unweighted_without_ties() -> None:
    summarizer = load_summarizer()
    y = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.int8)
    score = np.asarray([0.1, 0.8, 0.2, 0.9, 0.7, 0.3], dtype=np.float64)
    weights = np.ones((1, len(y)), dtype=np.int16)
    ap, auc = summarizer.vectorized_weighted_rank_metrics(y, score, weights)
    metrics = summarizer.point_metrics(y, score)
    assert np.allclose(ap, [metrics["auprc"]])
    assert np.allclose(auc, [metrics["auroc"]])


def test_group_bootstrap_resamples_whole_groups() -> None:
    summarizer = load_summarizer()
    groups = np.asarray(["a", "a", "b", "b", "c"])
    weights = summarizer.group_bootstrap_weights(groups, resamples=20, seed=7)
    assert weights.shape == (20, 5)
    assert np.array_equal(weights[:, 0], weights[:, 1])
    assert np.array_equal(weights[:, 2], weights[:, 3])
    assert np.all(weights[:, [0, 2, 4]].sum(axis=1) == 3)


def test_bh_fdr_is_monotone_in_p_value_order() -> None:
    summarizer = load_summarizer()
    comparisons = [
        {
            "comparison": f"activation_head-minus-judge_{index}",
            "auprc_two_sided_sign_p": value,
        }
        for index, value in enumerate((0.001, 0.02, 0.03, 0.2))
    ]
    summary = summarizer.add_bh_fdr(comparisons)
    q_values = np.asarray([row["auprc_bh_q"] for row in comparisons])
    assert summary["family_size"] == 4
    assert np.all(np.diff(q_values) >= 0)
    assert comparisons[0]["auprc_bh_reject_0_05"] is True
    assert comparisons[-1]["auprc_bh_reject_0_05"] is False


def test_summary_reads_only_completed_cells() -> None:
    summarizer = load_summarizer()
    required_archive = EXP / "activation_archives" / "cost_anchor__authorization_to_execute.npz"
    if not required_archive.exists():
        pytest.skip("Redistribution snapshot omits frozen score archives")
    result = summarizer.summarize()
    assert result["status"] in {"PARTIAL", "COMPLETE"}
    assert result["expected_cells"] == 6
    assert all(row["n"] > 0 for row in result["rows"])
