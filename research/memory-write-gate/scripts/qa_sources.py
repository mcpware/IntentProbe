#!/usr/bin/env python3
"""Run structural, leakage, and label QA on acquired Memory Write Gate data."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "huggingface"
OUTPUT = ROOT / "data" / "manifests" / "qa_summary.json"


def normalized_hash(series: pd.Series) -> pd.Series:
    normalized = series.fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    return normalized.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())


def counts(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    available = [column for column in columns if column in frame]
    if not available:
        return {}
    grouped = frame.groupby(available, dropna=False).size()
    return {
        " | ".join(map(str, key if isinstance(key, tuple) else (key,))): int(value)
        for key, value in grouped.items()
    }


def summarize_write_gate() -> dict[str, Any]:
    path = RAW / "memory-write-gate-20k" / "data" / "memory_write_gate_20000.jsonl"
    frame = pd.read_json(path, lines=True)
    fingerprints = normalized_hash(frame["text"])
    result: dict[str, Any] = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "nulls": {key: int(value) for key, value in frame.isna().sum().items() if value},
        "unique_normalized_texts": int(fingerprints.nunique()),
        "duplicate_rows_by_text": int(fingerprints.duplicated(keep=False).sum()),
        "label_counts": counts(frame, ["label"]),
        "split_counts": counts(frame, ["split"]),
        "split_label_counts": counts(frame, ["split", "label"]),
        "language_label_counts": counts(frame, ["language", "label"]),
        "category_label_counts": counts(frame, ["category", "label"]),
        "difficulty_label_counts": counts(frame, ["difficulty", "label"]),
        "seed_families": int(frame["seed_id"].nunique()),
        "seed_families_crossing_splits": int((frame.groupby("seed_id")["split"].nunique() > 1).sum()),
    }
    contradiction_counts = frame.assign(_fingerprint=fingerprints).groupby("_fingerprint")["label"].nunique()
    result["contradictory_text_groups"] = int((contradiction_counts > 1).sum())
    return result


def summarize_sp_mem() -> dict[str, Any]:
    root = RAW / "sp-mem"
    files = pd.Series(sorted(root.glob("*/*.json")), dtype="object")
    domains = files.map(lambda path: path.parent.name)
    return {
        "history_files": int(len(files)),
        "domain_file_counts": {str(key): int(value) for key, value in domains.value_counts().sort_index().items()},
        "unique_filenames": int(files.map(lambda path: path.name).nunique()),
    }


def summarize_memoryctrl() -> dict[str, Any]:
    root = RAW / "memoryctrl"
    paths = {
        "conversations": root / "conversations" / "data.parquet",
        "whole_recall_mcq": root / "whole_recall_mcq" / "test.parquet",
        "slot_recall_mcq": root / "slot_recall_mcq" / "test.parquet",
    }
    result: dict[str, Any] = {}
    for key, path in paths.items():
        frame = pd.read_parquet(path)
        entry: dict[str, Any] = {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "nulls": {name: int(value) for name, value in frame.isna().sum().items() if value},
            "unique_sample_ids": int(frame["sample_id"].nunique()) if "sample_id" in frame else None,
        }
        if "turn_role" in frame:
            entry["turn_role_counts"] = counts(frame, ["turn_role"])
        if "qa_family" in frame:
            entry["qa_family_counts"] = counts(frame, ["qa_family"])
        result[key] = entry
    return result


def main() -> None:
    results = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {
            "memory-write-gate-20k": summarize_write_gate(),
            "sp-mem": summarize_sp_mem(),
            "memoryctrl": summarize_memoryctrl(),
        },
        "excluded": {
            "memory-traces": "Acquisition failed because the Hugging Face dataset is gated; no terms were inspected or accepted."
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
