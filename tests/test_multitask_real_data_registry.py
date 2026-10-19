import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "research" / "multitask-activation-benchmark"


def test_registry_enforces_real_confirmatory_data():
    registry = json.loads((BENCH / "TASK_REGISTRY.json").read_text(encoding="utf-8"))
    policy = registry["synthetic_policy"]
    assert policy["confirmatory_test_max_fraction"] == 0.0
    assert policy["headline_train_development_max_fraction"] <= 0.1
    assert policy["synthetic_removed_sensitivity_required"] is True
    assert len(registry["tasks"]) >= 15
    assert {task["topic"] for task in registry["tasks"]} >= {
        "binary sentiment",
        "toxicity in real user-to-LLM prompts",
        "jailbreak intent",
        "fine-grained user intent",
        "SMS spam",
    }


def test_sst2_snapshot_is_leakage_clean_if_built():
    data = BENCH / "data" / "processed" / "sst2" / "sst2_root_sentences.parquet"
    if not data.exists():
        return
    frame = pd.read_parquet(data)
    assert frame.groupby("split").size().to_dict() == {
        "test": 1821,
        "train": 6909,
        "validation": 872,
    }
    assert frame["group_id"].nunique() == len(frame)
    groups = {split: set(part["group_id"]) for split, part in frame.groupby("split")}
    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])


def test_toxicchat_human_snapshot_is_leakage_clean_if_built():
    data = BENCH / "data" / "processed" / "toxicchat0124" / "human_unique.parquet"
    if not data.exists():
        return
    frame = pd.read_parquet(data)
    assert len(frame) == 5441
    assert int(frame["toxicity"].sum()) == 695
    assert int(frame["jailbreaking"].sum()) == 182
    assert bool((frame["jailbreaking"] <= frame["toxicity"]).all())
    assert frame["group_id"].nunique() == len(frame)
    groups = {split: set(part["group_id"]) for split, part in frame.groupby("split")}
    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])


def test_sms_spam_snapshot_is_leakage_clean_if_built():
    data = BENCH / "data" / "processed" / "sms_spam_uci" / "sms_spam_unique.parquet"
    if not data.exists():
        return
    frame = pd.read_parquet(data)
    assert len(frame) == 5159
    assert int(frame["label"].sum()) == 642
    assert frame.groupby("split").size().to_dict() == {
        "test": 774,
        "train": 3611,
        "validation": 774,
    }
    assert frame["group_id"].nunique() == len(frame)
    groups = {split: set(part["group_id"]) for split, part in frame.groupby("split")}
    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])


def test_goemotions_neutral_snapshot_is_leakage_clean_if_built():
    data = BENCH / "data" / "processed" / "goemotions" / "neutral_label_unique.parquet"
    if not data.exists():
        return
    frame = pd.read_parquet(data)
    assert len(frame) == 53905
    assert int(frame["neutral_label"].sum()) == 17686
    assert frame.groupby("split").size().to_dict() == {
        "test": 5409,
        "train": 43095,
        "validation": 5401,
    }
    assert frame["comment_id"].nunique() == len(frame)
    assert frame["group_id"].nunique() == len(frame)
    groups = {split: set(part["group_id"]) for split, part in frame.groupby("split")}
    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])


def test_goemotions_emotion_panel_reuses_exact_frozen_rows_if_built():
    neutral_path = BENCH / "data" / "processed" / "goemotions" / "neutral_label_unique.parquet"
    panel_path = BENCH / "data" / "processed" / "goemotions" / "emotion_panel_unique.parquet"
    if not neutral_path.exists() or not panel_path.exists():
        return
    neutral = pd.read_parquet(neutral_path, columns=["comment_id", "text", "split", "group_id"])
    panel = pd.read_parquet(
        panel_path,
        columns=[
            "comment_id",
            "text",
            "split",
            "group_id",
            "admiration",
            "anger",
            "curiosity",
            "sadness",
        ],
    )
    assert neutral[["comment_id", "text", "split", "group_id"]].equals(
        panel[["comment_id", "text", "split", "group_id"]]
    )
    expected = {
        "admiration": {"test": 504, "train": 4106, "validation": 487},
        "anger": {"test": 198, "train": 1561, "validation": 194},
        "curiosity": {"test": 283, "train": 2183, "validation": 246},
        "sadness": {"test": 156, "train": 1318, "validation": 143},
    }
    observed = {
        label: {
            split: int(part[label].sum())
            for split, part in panel.groupby("split", sort=True)
        }
        for label in expected
    }
    assert observed == expected


def test_unhealthy_comments_snapshot_is_leakage_clean_if_built():
    data = BENCH / "data" / "processed" / "unhealthy_comments" / "ucc_unique.parquet"
    if not data.exists():
        return
    frame = pd.read_parquet(data)
    assert len(frame) == 44323
    assert frame.groupby("split").size().to_dict() == {
        "test": 4424,
        "train": 35472,
        "validation": 4427,
    }
    assert frame["comment_id"].nunique() == len(frame)
    assert frame["group_id"].nunique() == len(frame)
    groups = {split: set(part["group_id"]) for split, part in frame.groupby("split")}
    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])
    assert {
        label: int(frame[label].sum())
        for label in ("condescending", "dismissive", "generalisation_unfair", "sarcastic")
    } == {
        "condescending": 2434,
        "dismissive": 1364,
        "generalisation_unfair": 890,
        "sarcastic": 1897,
    }


def test_banking77_snapshot_is_leakage_clean_if_built():
    data = BENCH / "data" / "processed" / "banking77" / "banking77_unique.parquet"
    if not data.exists():
        return
    frame = pd.read_parquet(data)
    assert len(frame) == 13071
    assert frame["label_id"].nunique() == 77
    assert frame.groupby("split").size().to_dict() == {
        "test": 3079,
        "train": 8493,
        "validation": 1499,
    }
    assert frame["group_id"].nunique() == len(frame)
    groups = {split: set(part["group_id"]) for split, part in frame.groupby("split")}
    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])


def test_sentiment_experiment_is_frozen_before_test():
    here = BENCH / "experiments" / "SENTIMENT-SST2-QWEN-EXP001"
    config = json.loads((here / "config.json").read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_BEFORE_FEATURE_EXTRACTION"
    assert config["candidate_family"]["candidate_count"] == 216
    assert config["task"]["test_use"].startswith("one evaluation")


def test_completed_sentiment_result_recomputes_if_scores_exist():
    here = BENCH / "experiments" / "SENTIMENT-SST2-QWEN-EXP001"
    scores_path = here / "test_scores.npz"
    result_path = here / "result.json"
    if not scores_path.exists() or not result_path.exists():
        return
    scores = np.load(scores_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))["heldout_test"]["models"]
    y = scores["label"]
    for name in ("activation", "tfidf_word_char"):
        score = scores[name]
        assert abs(roc_auc_score(y, score) - result[name]["auroc"]) < 1e-12
        assert abs(average_precision_score(y, score) - result[name]["auprc"]) < 1e-12
        assert abs(brier_score_loss(y, score) - result[name]["brier_score"]) < 1e-12


def test_toxicchat_experiment_contract_is_frozen_before_test():
    here = BENCH / "experiments" / "TOXICCHAT0124-QWEN-EXP001"
    config = json.loads((here / "config.json").read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_AFTER_LABEL_BLIND_TOKEN_AUDIT_BEFORE_FEATURE_EXTRACTION"
    assert config["dataset"]["human_annotation_only"] is True
    assert config["candidate_family"]["candidate_count_per_task"] == 216
    assert set(config["tasks"]) == {"toxicity", "jailbreaking"}
    assert config["backbone"]["max_tokens"] == 512
    assert config["label_blind_token_audit"]["rows_over_512"] == 9


def test_sms_spam_experiment_contract_is_frozen_before_test():
    here = BENCH / "experiments" / "SPAM-SMS-UCI-QWEN-EXP001"
    config = json.loads((here / "config.json").read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_AFTER_LABEL_BLIND_TOKEN_AUDIT_BEFORE_FEATURE_EXTRACTION"
    assert config["candidate_family"]["candidate_count"] == 216
    assert config["task"]["primary_selection_metric"] == "AUPRC"
    assert config["backbone"]["max_tokens"] == 256
    assert config["label_blind_token_audit"]["rows_over_cap"] == 1


def test_goemotions_experiment_contract_is_frozen_before_test():
    here = BENCH / "experiments" / "EMOTION-GOEMOTIONS-NEUTRAL-QWEN-EXP001"
    config = json.loads((here / "config.json").read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_AFTER_COMPUTE_PILOT_BEFORE_TEST"
    assert config["candidate_family"]["candidate_count"] == 216
    assert config["task"]["id"] == "goemotions_neutral_label_present"
    assert config["task"]["selection_train_row_budget"] == 5000
    assert config["backbone"]["max_tokens"] == 256
    assert config["label_blind_token_audit"]["rows_over_cap"] == 3


def test_goemotions_panel_contract_is_frozen_before_test():
    here = BENCH / "experiments" / "EMOTION-GOEMOTIONS-PANEL-QWEN-EXP001"
    config = json.loads((here / "config.json").read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_BEFORE_FIRST_EMOTION_PANEL_SCREEN"
    assert config["candidate_family"]["candidate_count_per_task"] == 216
    assert set(config["tasks"]) == {"admiration", "anger", "curiosity", "sadness"}
    assert all(task["selection_train_row_budget"] == 5000 for task in config["tasks"].values())
    assert config["activation_cache_reuse"]["source_experiment"] == (
        "EMOTION-GOEMOTIONS-NEUTRAL-QWEN-EXP001"
    )


def test_ucc_experiment_contract_is_frozen_before_test():
    here = BENCH / "experiments" / "PRAGMATICS-UCC-QWEN-EXP001"
    config = json.loads((here / "config.json").read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_AFTER_LABEL_BLIND_TOKEN_AUDIT_BEFORE_FEATURE_EXTRACTION"
    assert config["candidate_family"]["candidate_count_per_task"] == 216
    assert set(config["tasks"]) == {
        "condescending",
        "dismissive",
        "generalisation_unfair",
        "sarcastic",
    }
    assert config["backbone"]["max_tokens"] == 128
    assert config["label_blind_token_audit"]["rows_over_cap"] == 0


def test_ucc_activation_cache_manifest_is_complete_if_built():
    here = BENCH / "experiments" / "PRAGMATICS-UCC-QWEN-EXP001"
    manifest_path = here / "extraction_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "ALL_POOLING_FEATURES_COMPLETE"
    assert manifest["rows"] == 44323
    assert manifest["shape_per_pooling"] == [44323, 25, 896]
    assert manifest["rows_over_cap"] == 0
    assert manifest["truncation_rate"] == 0.0
    assert manifest["input_sha256"] == (
        "617083fa3662d6371b98cc37cd74ab4fa10c79cc71d6f99cbcd31388c1251d9f"
    )
    assert set(manifest["files"]) == {"masked_mean", "last_nonpadding", "masked_max"}


def test_banking77_experiment_contract_is_frozen_before_test():
    here = BENCH / "experiments" / "INTENT-BANKING77-QWEN-EXP001"
    config = json.loads((here / "config.json").read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_AFTER_MULTICLASS_COMPUTE_PILOT_BEFORE_TEST"
    assert config["candidate_family"]["candidate_count"] == 216
    assert config["task"]["classes"] == 77
    assert config["task"]["primary_selection_metric"] == "macro_F1"
    assert config["task"]["selection_train_row_budget"] == 3000
    assert config["head"]["max_iter"] == 40
    assert config["label_blind_token_audit"]["rows_over_cap"] == 0


def test_completed_goemotions_result_recomputes_if_scores_exist():
    here = BENCH / "experiments" / "EMOTION-GOEMOTIONS-NEUTRAL-QWEN-EXP001"
    scores_path = here / "test_scores.npz"
    result_path = here / "result.json"
    if not scores_path.exists() or not result_path.exists():
        return
    scores = np.load(scores_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))["heldout_test"]["models"]
    y = scores["label"]
    for name in ("activation", "tfidf_word_char"):
        score = scores[name]
        assert abs(roc_auc_score(y, score) - result[name]["auroc"]) < 1e-12
        assert abs(average_precision_score(y, score) - result[name]["auprc"]) < 1e-12
        assert abs(brier_score_loss(y, score) - result[name]["brier_score"]) < 1e-12


def test_completed_goemotions_panel_recomputes_if_built():
    here = BENCH / "experiments" / "EMOTION-GOEMOTIONS-PANEL-QWEN-EXP001"
    summary_path = here / "EMOTION-PANEL-SUMMARY.json"
    if not summary_path.exists():
        return
    score_paths = [
        here / f"{task}_test_scores.npz"
        for task in ("admiration", "anger", "curiosity", "sadness")
    ]
    if not all(path.exists() for path in score_paths):
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "COMPLETE_VALIDATED_MULTILABEL_MATRIX"
    assert summary["tasks"] == 4
    assert summary["pooling_counts"] == {"masked_mean": 4}
    assert summary["minimum_normalized_depth"] == 0.125
    assert summary["maximum_normalized_depth"] == 2 / 3
    for task in ("admiration", "anger", "curiosity", "sadness"):
        scores = np.load(here / f"{task}_test_scores.npz")
        result = json.loads((here / f"{task}_result.json").read_text(encoding="utf-8"))
        y = scores["label"]
        for name in ("activation", "tfidf_word_char"):
            score = scores[name]
            reported = result["heldout_test"]["models"][name]
            assert abs(average_precision_score(y, score) - reported["auprc"]) < 1e-12
            assert abs(roc_auc_score(y, score) - reported["auroc"]) < 1e-12


def test_completed_banking77_result_recomputes_if_scores_exist():
    here = BENCH / "experiments" / "INTENT-BANKING77-QWEN-EXP001"
    scores_path = here / "test_scores.npz"
    result_path = here / "result.json"
    if not scores_path.exists() or not result_path.exists():
        return
    scores = np.load(scores_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))["heldout_test"]["models"]
    y = scores["label"]
    classes = scores["classes"]
    for name, key in (("activation", "activation_probability"), ("tfidf_word_char", "tfidf_probability")):
        probability = scores[key]
        prediction = classes[probability.argmax(axis=1)]
        assert abs(accuracy_score(y, prediction) - result[name]["accuracy"]) < 1e-12
        assert abs(
            f1_score(y, prediction, labels=classes, average="macro", zero_division=0)
            - result[name]["macro_f1"]
        ) < 1e-12


def test_completed_sms_spam_result_recomputes_if_scores_exist():
    here = BENCH / "experiments" / "SPAM-SMS-UCI-QWEN-EXP001"
    scores_path = here / "test_scores.npz"
    result_path = here / "result.json"
    if not scores_path.exists() or not result_path.exists():
        return
    scores = np.load(scores_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))["heldout_test"]["models"]
    y = scores["label"]
    for name in ("activation", "tfidf_word_char"):
        score = scores[name]
        assert abs(roc_auc_score(y, score) - result[name]["auroc"]) < 1e-12
        assert abs(average_precision_score(y, score) - result[name]["auprc"]) < 1e-12
        assert abs(brier_score_loss(y, score) - result[name]["brier_score"]) < 1e-12


def test_completed_sms_six_backbone_matrix_is_aligned_if_built():
    summary_path = BENCH / "experiments" / "SPAM-SMS-UCI-MODEL-MATRIX.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "COMPLETE_SIX_BACKBONE_VALIDATION_SCREEN_AND_ONE_SHOT_TEST"
    assert summary["models_completed"] == 6
    assert summary["heldout_identity_checks"] == {
        "identical_example_ids": True,
        "identical_labels": True,
        "identical_tfidf_probabilities": True,
        "positive_rows": 96,
        "rows": 774,
    }
    assert summary["readout_counts"] == {
        "last_nonpadding": 0,
        "masked_max": 3,
        "masked_mean": 3,
    }
    assert summary["planned_comparison_counts"] == {
        "activation_advantage_both_metrics": 2,
        "unresolved": 4,
    }
    assert summary["cross_model_inference"].startswith("exploratory_descriptive_only")

    score_paths = sorted((BENCH / "experiments").glob("SPAM-SMS-UCI-*/test_scores.npz"))
    if not score_paths:
        return
    assert len(score_paths) == 6
    score_sets = [np.load(path) for path in score_paths]
    for key in ("label", "example_id", "tfidf_word_char"):
        assert all(np.array_equal(score_sets[0][key], scores[key]) for scores in score_sets[1:])


def test_completed_sentiment_six_backbone_matrix_is_aligned_if_built():
    summary_path = BENCH / "experiments" / "SENTIMENT-SST2-MODEL-MATRIX.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "COMPLETE_SIX_BACKBONE_VALIDATION_SCREEN_AND_ONE_SHOT_TEST"
    assert summary["models_completed"] == 6
    assert summary["selection_metric"] == "auroc"
    assert summary["heldout_identity_checks"] == {
        "identical_example_ids": True,
        "identical_labels": True,
        "identical_tfidf_probabilities": True,
        "positive_rows": 909,
        "rows": 1821,
    }
    assert summary["readout_counts"] == {
        "last_nonpadding": 1,
        "masked_max": 0,
        "masked_mean": 5,
    }
    assert summary["planned_comparison_counts"] == {
        "activation_advantage_both_metrics": 6,
        "unresolved": 0,
    }

    score_paths = sorted((BENCH / "experiments").glob("SENTIMENT-SST2-*/test_scores.npz"))
    if not score_paths:
        return
    assert len(score_paths) == 6
    score_sets = [np.load(path) for path in score_paths]
    for key in ("label", "example_id", "tfidf_word_char"):
        assert all(np.array_equal(score_sets[0][key], scores[key]) for scores in score_sets[1:])


def test_sentiment_matrix_configs_preserve_the_anchor_contract():
    experiment_dirs = sorted((BENCH / "experiments").glob("SENTIMENT-SST2-*-EXP001"))
    if len(experiment_dirs) < 6:
        return
    assert len(experiment_dirs) == 6
    for experiment in experiment_dirs:
        config = json.loads((experiment / "config.json").read_text(encoding="utf-8"))
        assert config["task"]["primary_selection_metric"] == "AUROC"
        if experiment.name != "SENTIMENT-SST2-QWEN-EXP001":
            assert config["baseline_class_weight"] is None
            assert config["dataset"]["splits"] == {
                "test": 1821,
                "train": 6909,
                "validation": 872,
            }


def test_completed_toxicchat_results_recompute_if_scores_exist():
    here = BENCH / "experiments" / "TOXICCHAT0124-QWEN-EXP001"
    for task in ("toxicity", "jailbreaking"):
        scores_path = here / f"{task}_test_scores.npz"
        result_path = here / f"{task}_result.json"
        if not scores_path.exists() or not result_path.exists():
            continue
        scores = np.load(scores_path)
        result = json.loads(result_path.read_text(encoding="utf-8"))["heldout_test"]["models"]
        y = scores["label"]
        for name in ("activation", "tfidf_word_char"):
            score = scores[name]
            assert abs(roc_auc_score(y, score) - result[name]["auroc"]) < 1e-12
            assert abs(average_precision_score(y, score) - result[name]["auprc"]) < 1e-12
            assert abs(brier_score_loss(y, score) - result[name]["brier_score"]) < 1e-12


def test_toxicchat_six_backbone_configs_share_one_multilabel_contract_if_built():
    experiment_dirs = sorted((BENCH / "experiments").glob("TOXICCHAT0124-*-EXP001"))
    if len(experiment_dirs) < 6:
        return
    assert len(experiment_dirs) == 6
    observed_models = set()
    for experiment in experiment_dirs:
        config = json.loads((experiment / "config.json").read_text(encoding="utf-8"))
        observed_models.add(config["backbone"]["id"])
        assert config["dataset"]["input"] == "data/processed/toxicchat0124/human_unique.parquet"
        assert config["dataset"]["rows"] == 5441
        assert config["dataset"]["text_column"] == "text"
        assert config["dataset"]["id_column"] == "conv_id"
        assert set(config["tasks"]) == {"toxicity", "jailbreaking"}
        assert config["tasks"]["toxicity"]["positive_rows"]["test"] == 354
        assert config["tasks"]["jailbreaking"]["positive_rows"]["test"] == 89
        assert config["backbone"]["max_tokens"] == 512
        assert config["label_blind_token_audit"]["rows_over_cap"] <= 13
    assert len(observed_models) == 6


def test_completed_toxicchat_crossed_matrix_if_built():
    summary_path = BENCH / "experiments" / "TOXICCHAT0124-MODEL-TASK-MATRIX.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "COMPLETE_SIX_BACKBONE_TWO_TASK_CROSSED_MATRIX"
    assert summary["cells"] == 12
    assert summary["pooling_counts"]["toxicity"] == {"masked_mean": 6}
    assert sum(summary["pooling_counts"]["jailbreaking"].values()) == 6
    assert summary["test"]["toxicity_activation_advantage_both_metrics_count"] == 2
    assert summary["test"]["jailbreak_activation_advantage_both_metrics_count"] == 0
    policy = summary["validation_policy_comparison"]
    assert policy["universal"] == {
        "mean_validation_auprc": 0.7871935590683958,
        "pooling": "masked_mean",
        "target_depth": 0.5,
        "width": 2,
    }
    assert policy["mean_regret_vs_cell_specific"]["model_specific"] < policy["mean_regret_vs_cell_specific"]["task_specific"]
