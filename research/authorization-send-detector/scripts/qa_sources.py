#!/usr/bin/env python3
"""Run structural QA on downloaded authorization datasets."""

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


def load_parquet_tree(path: Path) -> pd.DataFrame:
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files under {path}")
    return pd.concat((pd.read_parquet(file) for file in files), ignore_index=True)


def summarize_frame(frame: pd.DataFrame, text_column: str | None, label_column: str | None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "nulls": {key: int(value) for key, value in frame.isna().sum().items() if value},
    }
    if label_column and label_column in frame:
        summary["label_counts"] = {
            str(key): int(value) for key, value in frame[label_column].value_counts(dropna=False).items()
        }
    if text_column and text_column in frame:
        fingerprints = normalized_hash(frame[text_column])
        summary["unique_normalized_texts"] = int(fingerprints.nunique())
        summary["duplicate_rows_by_text"] = int(fingerprints.duplicated(keep=False).sum())
        if label_column and label_column in frame:
            contradiction_counts = frame.assign(_fingerprint=fingerprints).groupby("_fingerprint")[label_column].nunique()
            summary["contradictory_text_groups"] = int((contradiction_counts > 1).sum())
    return summary


def cross_tab(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    """Return a stable multi-column count table without iterating over rows."""
    available = [column for column in columns if column in frame]
    if not available:
        return {}
    counts = frame.groupby(available, dropna=False).size()
    return {" | ".join(map(str, key if isinstance(key, tuple) else (key,))): int(value) for key, value in counts.items()}


def group_overlap(frame: pd.DataFrame, group_column: str, split_column: str) -> int | None:
    """Count groups that appear in more than one split."""
    if group_column not in frame or split_column not in frame:
        return None
    return int((frame.groupby(group_column)[split_column].nunique() > 1).sum())


def main() -> None:
    results: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {},
    }

    approve = load_parquet_tree(RAW / "approve-or-deny")
    approve_text = "text" if "text" in approve else "user_request"
    approve_label = "label" if "label" in approve else None
    results["datasets"]["approve-or-deny"] = summarize_frame(approve, approve_text, approve_label)

    auto = load_parquet_tree(RAW / "auto-1b-data")
    auto_text = "text" if "text" in auto else "user_request"
    auto_label = "label" if "label" in auto else None
    auto_summary = summarize_frame(auto, auto_text, auto_label)
    auto_fingerprints = normalized_hash(auto[auto_text])
    auto_with_fingerprint = auto.assign(_fingerprint=auto_fingerprints)
    auto_summary["subset_label_counts"] = cross_tab(auto, ["subset", "label"])
    auto_summary["duplicate_fingerprints_crossing_subsets"] = int(
        (auto_with_fingerprint.groupby("_fingerprint")["subset"].nunique() > 1).sum()
    )
    auto_summary["duplicate_fingerprint_hashes"] = sorted(
        auto_with_fingerprint.loc[auto_fingerprints.duplicated(keep=False), "_fingerprint"].unique().tolist()
    )
    results["datasets"]["auto-1b-data"] = auto_summary

    auth_files = sorted((RAW / "authorizationpairs").rglob("*.jsonl"))
    auth = pd.concat(
        (pd.read_json(file, lines=True).assign(_source_file=file.name) for file in auth_files),
        ignore_index=True,
    )
    auth_text = next((name for name in ("prompt", "request", "user_request", "input") if name in auth), None)
    auth_label = next((name for name in ("label", "reference", "decision") if name in auth), None)
    auth_summary = summarize_frame(auth, auth_text, auth_label)
    auth_summary["file_track_label_counts"] = cross_tab(auth, ["_source_file", "track", "label"])
    auth_summary["groups_crossing_files"] = group_overlap(auth, "group_id", "_source_file")
    if "ambiguous" in auth:
        auth_summary["ambiguous_counts"] = {
            str(key): int(value) for key, value in auth["ambiguous"].value_counts(dropna=False).items()
        }
    results["datasets"]["authorizationpairs"] = auth_summary

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
