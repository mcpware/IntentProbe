#!/usr/bin/env python3
"""Create frozen per-backbone configs from one completed binary experiment."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


BENCHMARK = Path(__file__).resolve().parents[1]
PROJECT = BENCHMARK.parents[1]
AUTH_MANIFESTS = (
    PROJECT
    / "research"
    / "authorization-send-detector"
    / "experiments"
    / "EXP002-activation-screen"
    / "feature_manifests"
)

ALIASES = {
    "LiquidAI/LFM2.5-230M": "LFM25-230M",
    "ibm-granite/granite-4.0-350m-base": "GRANITE4-350M",
    "Qwen/Qwen2.5-0.5B": "QWEN25-05B",
    "Qwen/Qwen3-0.6B-Base": "QWEN3-06B",
    "Qwen/Qwen3.5-0.8B": "QWEN35-08B",
    "openbmb/MiniCPM5-1B-Base": "MINICPM5-1B",
}


def normalize_source_config(
    source: dict[str, object], source_dir: Path
) -> tuple[dict[str, object], Path, pd.DataFrame]:
    """Map early binary experiments onto the shared matrix contract."""
    normalized = copy.deepcopy(source)
    if "dataset" in normalized:
        data_path = BENCHMARK / str(normalized["dataset"]["input"])
        return normalized, data_path, pd.read_parquet(data_path)

    task = normalized["task"]
    data_path = BENCHMARK / str(task["input"])
    frame = pd.read_parquet(data_path)
    required = {"example_id", "text", "label", "split"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Legacy binary config lacks shared columns: {sorted(required - set(frame.columns))}")
    split_counts = frame.groupby("split").size().to_dict()
    if set(split_counts) != {"train", "validation", "test"}:
        raise ValueError(f"Unexpected legacy splits: {split_counts}")
    normalized["dataset"] = {
        "id": str(task["id"]),
        "input": str(task["input"]),
        "rows": len(frame),
        "splits": {key: int(value) for key, value in split_counts.items()},
        "text_column": "text",
        "id_column": "example_id",
        "exact_duplicate_policy": "use the leakage-clean processed snapshot frozen by the source experiment",
    }
    normalized["task"] = {
        **task,
        "label_column": "label",
        "positive_label": 1,
        "positive_rows": {
            split: int(part["label"].sum()) for split, part in frame.groupby("split")
        },
    }
    normalized["backbone"]["truncation_side"] = "right"
    normalized["backbone"]["trust_remote_code"] = True
    normalized["baseline_class_weight"] = None
    normalized["research_scope"] = normalized.get(
        "release_boundary",
        "Internal research benchmark. Any later public release is a separate project-owner decision.",
    )
    return normalized, data_path, frame


def read_model_records() -> list[dict[str, object]]:
    """Read the completed authorization model manifests as the panel lock."""
    manifests = sorted(AUTH_MANIFESTS.glob("*.json"))
    records: list[dict[str, object]] = []
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        train = payload["splits"]["train"]
        records.append(
            {
                "id": payload["model"]["id"],
                "revision": payload["model"]["revision"],
                "hidden_state_outputs": int(train["hidden_state_count"]),
                "hidden_size": int(train["hidden_size"]),
                "model_class": payload["model_class"],
            }
        )
    if set(ALIASES) != {str(record["id"]) for record in records}:
        raise ValueError("Authorization manifests do not match the frozen six-backbone panel")
    return records


def token_audit(texts: list[str], backbone: dict[str, object]) -> dict[str, object]:
    """Compute a label-blind tokenizer coverage audit for one frozen backbone."""
    tokenizer = AutoTokenizer.from_pretrained(
        str(backbone["id"]),
        revision=str(backbone["revision"]),
        local_files_only=True,
        trust_remote_code=True,
    )
    lengths = np.asarray(
        tokenizer(texts, add_special_tokens=True, truncation=False, return_length=True)["length"],
        dtype=np.int32,
    )
    cap = int(backbone["max_tokens"])
    return {
        "maximum_tokens": int(lengths.max()),
        "p50_tokens": float(np.percentile(lengths, 50)),
        "p95_tokens": float(np.percentile(lengths, 95)),
        "p99_tokens": float(np.percentile(lengths, 99)),
        "rows_over_cap": int(np.sum(lengths > cap)),
    }


def main() -> None:
    """Freeze one config per model without touching experiment labels or test scores."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source_experiment")
    parser.add_argument("--exclude-anchor", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source_experiment).resolve()
    raw_source = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    source, data_path, frame = normalize_source_config(raw_source, source_dir)
    frame = frame[[str(source["dataset"]["text_column"])]].copy()
    expected = int(source["dataset"]["rows"])
    if len(frame) != expected or frame.iloc[:, 0].isna().any():
        raise ValueError(f"Expected {expected} non-null texts, got {len(frame)}")
    texts = frame.iloc[:, 0].astype(str).tolist()

    created: list[dict[str, object]] = []
    for record in read_model_records():
        if args.exclude_anchor and record["id"] == source["backbone"]["id"]:
            continue
        alias = ALIASES[str(record["id"])]
        config = copy.deepcopy(source)
        source_token = str(source["experiment_id"]).split("-QWEN-")[0]
        config["experiment_id"] = f"{source_token}-{alias}-EXP001"
        config["status"] = "FROZEN_AFTER_LABEL_BLIND_TOKEN_AUDIT_BEFORE_FEATURE_EXTRACTION"
        config["backbone"] = {
            **record,
            "max_tokens": int(source["backbone"]["max_tokens"]),
            "truncation_side": str(source["backbone"].get("truncation_side", "right")),
            "trust_remote_code": True,
        }
        outputs = int(record["hidden_state_outputs"])
        config["candidate_family"]["candidate_count"] = 9 * outputs - 9
        config["label_blind_token_audit"] = token_audit(texts, config["backbone"])
        target = source_dir.parent / str(config["experiment_id"])
        target.mkdir(parents=False, exist_ok=False)
        (target / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        created.append(
            {
                "experiment_id": config["experiment_id"],
                "model": record["id"],
                "hidden_state_outputs": outputs,
                "hidden_size": int(record["hidden_size"]),
                "candidate_count": int(config["candidate_family"]["candidate_count"]),
                "token_audit": config["label_blind_token_audit"],
            }
        )
    print(json.dumps({"created": created}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
