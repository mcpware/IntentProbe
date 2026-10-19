from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "research" / "schemas"
BENCHMARK_DIR = ROOT / "research" / "multitask-activation-benchmark"
SYSTEMS_RESULT = (
    BENCHMARK_DIR / "experiments" / "ONE-FORWARD-MULTIHEAD-EXP001" / "result.json"
)


def load_schema(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def test_architecture_contract_schemas_are_valid() -> None:
    for name in (
        "intentprobe_readout_manifest.schema.json",
        "intentprobe_multihead_receipt.schema.json",
    ):
        Draft202012Validator.check_schema(load_schema(name))


def test_readout_manifest_contract_accepts_complete_example() -> None:
    schema = load_schema("intentprobe_readout_manifest.schema.json")
    instance = {
        "schema_version": "1.0.0",
        "probe_id": "example.authorization",
        "probe_version": "0.1.0",
        "task_contract": {
            "task_id": "authorization",
            "task_type": "binary",
            "labels": ["not_authorized", "authorized"],
            "positive_label": "authorized",
            "intended_decision": "Score whether the context supports executing the proposed action now.",
            "excluded_interpretations": ["This score is not a replacement for a mandatory send gate."],
        },
        "backbone": {
            "model_id": "example/model",
            "model_revision": "example-revision",
            "tokenizer_id": "example/model",
            "tokenizer_revision": "example-revision",
            "hidden_state_indexing": "embedding_is_output_0",
            "dtype": "float32",
            "attention_implementation": None,
        },
        "renderer": {
            "renderer_id": "authorization-event",
            "renderer_version": "1.0.0",
            "max_length": 256,
            "truncation_side": "left",
            "field_order": ["history", "user_request", "proposed_action"],
            "special_token_policy": "tokenizer-default",
            "chat_template": None,
        },
        "readout": {
            "token_region": "all_valid_tokens",
            "pooling": "masked_mean",
            "pooling_parameters": {},
            "hidden_outputs": [13, 14],
            "normalized_depths": [0.5417, 0.5833],
            "layer_combination": "concatenate",
            "normalization": {
                "method": "standardize",
                "fit_split": "train",
                "artifact": "normalizer.npz",
            },
        },
        "head": {
            "family": "logistic_regression",
            "artifact": "probe.npz",
            "output_kind": "logit",
        },
        "calibration": {
            "method": "platt",
            "fit_split": "validation",
            "artifact": "calibration.json",
        },
        "operating_points": [
            {
                "name": "review",
                "threshold": 0.5,
                "comparison": ">=",
                "selected_on": "validation",
            }
        ],
        "artifacts": [
            {
                "path": "probe.npz",
                "sha256": "a" * 64,
            }
        ],
        "compatibility": {
            "runtime_schema": "intentprobe-multihead-receipt-v1",
            "minimum_runtime_version": "0.2.0",
        },
    }

    Draft202012Validator(schema).validate(instance)


def test_multihead_receipt_accepts_two_decisions_sharing_one_backbone_run() -> None:
    schema = load_schema("intentprobe_multihead_receipt.schema.json")
    instance = {
        "receipt_version": "1.0.0",
        "runtime_version": "0.2.0",
        "event_id": "event-example-1",
        "created_at": "2026-10-18T12:00:00Z",
        "input_fingerprint": {
            "algorithm": "sha256",
            "digest": "b" * 64,
            "payload_retained": False,
        },
        "backbone_runs": [
            {
                "run_id": "backbone-run-1",
                "model_id": "example/model",
                "model_revision": "example-revision",
                "renderer_id": "shared-event",
                "renderer_version": "1.0.0",
                "hidden_outputs": [11, 13, 14],
                "device": "cpu",
                "dtype": "float32",
                "elapsed_seconds": 0.2,
            }
        ],
        "decisions": [
            {
                "probe_id": "example.authorization",
                "probe_version": "0.1.0",
                "manifest_sha256": "c" * 64,
                "backbone_run_id": "backbone-run-1",
                "readout": {
                    "token_region": "all_valid_tokens",
                    "pooling": "masked_mean",
                    "hidden_outputs": [13, 14],
                    "layer_combination": "concatenate",
                    "normalization": "standardize",
                },
                "raw_output": 0.9,
                "calibrated_output": 0.8,
                "predicted_label": "authorized",
                "abstained": False,
                "operating_point": "review",
                "policy": {
                    "policy_id": "example-send-policy",
                    "policy_version": "1.0.0",
                    "decision": "review",
                    "reasons": ["Final send confirmation is required."],
                },
                "head_elapsed_seconds": 0.001,
            },
            {
                "probe_id": "example.memory-write",
                "probe_version": "0.1.0",
                "manifest_sha256": "d" * 64,
                "backbone_run_id": "backbone-run-1",
                "readout": {
                    "token_region": "all_valid_tokens",
                    "pooling": "masked_mean",
                    "hidden_outputs": [11],
                    "layer_combination": "single",
                    "normalization": "none",
                },
                "raw_output": 0.2,
                "calibrated_output": 0.2,
                "predicted_label": "skip",
                "abstained": False,
                "operating_point": "default",
                "policy": {
                    "policy_id": "example-memory-policy",
                    "policy_version": "1.0.0",
                    "decision": "skip",
                    "reasons": ["Score is below the write threshold."],
                },
                "head_elapsed_seconds": 0.001,
            },
        ],
        "timing": {
            "render_seconds": 0.01,
            "backbone_seconds": 0.2,
            "heads_seconds": 0.002,
            "policy_seconds": 0.001,
            "total_seconds": 0.213,
        },
    }

    Draft202012Validator(schema).validate(instance)


def test_public_architecture_release_matches_frozen_systems_result() -> None:
    result = json.loads(SYSTEMS_RESULT.read_text(encoding="utf-8"))
    rows = {row["k"]: row for row in result["scaling"]}
    assert result["status"] == "COMPLETE"
    assert result["head_count"] == 16
    assert result["k_values"] == [1, 2, 4, 8, 16]
    assert result["latency_rows"] == 80
    assert round(rows[1]["mean_total_ms"], 2) == 68.63
    assert round(rows[16]["mean_total_ms"], 2) == 73.34
    assert round(rows[16]["repeated_forward_mean_ms"], 2) == 956.71
    assert round(rows[16]["repeated_over_one_forward_ratio"], 2) == 15.26


def test_public_release_identity_numbers_and_figures_are_present() -> None:
    paths = (
        ROOT / "docs" / "ONE_FORWARD_MANY_DECISIONS.md",
        BENCHMARK_DIR / "ARCHITECTURE_DISCLOSURE_V0_1.md",
        BENCHMARK_DIR / "paper" / "DRAFT.md",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "Nicole BeWater" in text
        assert "68.63" in text
        assert "73.34" in text
        assert "62.70" in text
        assert "956.71" in text
        assert "15.26" in text

    for name in (
        "one-forward-many-decisions.svg",
        "one-forward-many-decisions.png",
        "one-forward-many-decisions-latency.svg",
        "one-forward-many-decisions-latency.png",
    ):
        path = ROOT / "docs" / name
        assert path.is_file()
        assert path.stat().st_size > 1_000
