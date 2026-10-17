#!/usr/bin/env python3
"""Create the immutable deterministic EXP002 train/validation screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RAW = PROJECT / "data" / "raw" / "huggingface" / "auto-1b-data"
PROCESSED = PROJECT / "data" / "processed" / "exp002"
CONFIG = HERE / "config.json"
MANIFEST = HERE / "screen_manifest.json"
RUNTIME_FIELDS = ["call", "user_request", "history"]
STRATIFY_FIELDS = ["label", "category", "difficulty"]
AUDIT_FIELDS = ["lang", "length_bucket", "framework"]


def sha256(path: Path) -> str:
    """Return SHA-256 for a fixed input or output file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render(frame: pd.DataFrame) -> pd.Series:
    """Render runtime-available fields without label or rationale leakage."""
    parts = frame[RUNTIME_FIELDS].fillna("").astype(str)
    return (
        "PROPOSED ACTION:\n"
        + parts["call"]
        + "\nUSER REQUEST:\n"
        + parts["user_request"]
        + "\nHISTORY:\n"
        + parts["history"]
    )


def deterministic_sample(frame: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    """Take a reproducible label-balanced sample with broad metadata coverage."""
    if rows > len(frame):
        raise ValueError(f"Requested {rows} rows from only {len(frame)}")
    parts = frame[STRATIFY_FIELDS].fillna("<NA>").astype(str)
    composite = parts[STRATIFY_FIELDS[0]]
    for column in STRATIFY_FIELDS[1:]:
        composite = composite.str.cat(parts[column], sep="|")
    frequencies = composite.map(composite.value_counts())
    safe_strata = composite.where(frequencies >= 2, frame["label"].astype(str) + "|RARE")
    selected, _ = train_test_split(
        frame,
        train_size=rows,
        random_state=seed,
        shuffle=True,
        stratify=safe_strata,
    )
    return selected.sort_index().reset_index(drop=True)


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize the selected screen without exposing raw text."""
    return {
        "rows": int(len(frame)),
        "label_counts": {str(key): int(value) for key, value in frame["label"].value_counts().items()},
        "category_count": int(frame["category"].nunique()),
        "framework_count": int(frame["framework"].nunique()),
        "language_counts": {str(key): int(value) for key, value in frame["lang"].value_counts().items()},
        "rendered_char_quantiles": {
            str(key): float(value)
            for key, value in frame["rendered"].str.len().quantile([0, 0.5, 0.9, 0.99, 1]).items()
        },
        "unique_fingerprints": int(frame["fingerprint"].nunique()),
    }


def main() -> None:
    """Build and verify the screen files and provenance manifest."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    outputs: dict[str, Path] = {}
    summaries: dict[str, Any] = {}
    fingerprints: dict[str, set[str]] = {}
    PROCESSED.mkdir(parents=True, exist_ok=True)

    for offset, split in enumerate(("train", "validation")):
        input_path = RAW / f"{split}.parquet"
        frame = pd.read_parquet(input_path)
        required = set(RUNTIME_FIELDS + STRATIFY_FIELDS + AUDIT_FIELDS)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{split}: missing columns {sorted(missing)}")
        selected = deterministic_sample(
            frame,
            rows=config["screen_rows"][split],
            seed=config["sampling_seed"] + offset,
        )
        selected = selected[RUNTIME_FIELDS + STRATIFY_FIELDS + AUDIT_FIELDS].copy()
        selected["rendered"] = render(selected)
        normalized = selected["rendered"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        selected["fingerprint"] = pd.util.hash_pandas_object(normalized, index=False).astype("uint64").astype(str)
        if selected[["rendered", "label", "fingerprint"]].isna().any().any():
            raise ValueError(f"{split}: null after rendering")
        if selected["fingerprint"].duplicated().any():
            raise ValueError(f"{split}: duplicate rendered context")
        output_path = PROCESSED / f"{split}.parquet"
        selected.to_parquet(output_path, index=False)
        outputs[split] = output_path
        summaries[split] = summarize(selected)
        fingerprints[split] = set(selected["fingerprint"])

    overlap = fingerprints["train"] & fingerprints["validation"]
    if overlap:
        raise ValueError(f"Cross-split rendered overlap: {len(overlap)}")

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256(CONFIG),
        "inputs": {
            split: {
                "path": str((RAW / f"{split}.parquet").relative_to(PROJECT)),
                "sha256": sha256(RAW / f"{split}.parquet"),
            }
            for split in outputs
        },
        "outputs": {
            split: {
                "path": str(path.relative_to(PROJECT)),
                "sha256": sha256(path),
                "summary": summaries[split],
            }
            for split, path in outputs.items()
        },
        "cross_split_overlap": 0,
        "leakage_assertion": "Only call, user_request, and history feed rendered text; label/rationale/text are excluded.",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
