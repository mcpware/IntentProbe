#!/usr/bin/env python3
"""Build a preregistered four-emotion panel from the frozen GoEmotions snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
SOURCE = BENCHMARK / "data" / "processed" / "goemotions" / "neutral_label_unique.parquet"
OUTPUT = BENCHMARK / "data" / "processed" / "goemotions" / "emotion_panel_unique.parquet"
MANIFEST = BENCHMARK / "data" / "manifests" / "goemotions_emotion_panel.json"
EXPECTED_SOURCE_SHA256 = "a6e25ab3a7fdd3b2d768a578011996ba8c55921dd2a5742e272bf8e9597b4550"

# Locked before looking at any activation result. These four labels span a
# positive social emotion, an activating negative emotion, an epistemic
# emotion, and a low-arousal negative emotion. Each has at least 150 positives
# in the untouched official test split after the existing leakage repair.
TARGETS = {
    "admiration": 0,
    "anger": 2,
    "curiosity": 7,
    "sadness": 25,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    assert sha256(SOURCE) == EXPECTED_SOURCE_SHA256
    frame = pd.read_parquet(SOURCE)
    assert len(frame) == 53905
    assert frame["comment_id"].nunique() == len(frame)
    assert frame["group_id"].nunique() == len(frame)

    label_sets = frame["label_ids"].map(lambda value: {int(item) for item in value.split(",")})
    for name, label_id in TARGETS.items():
        frame[name] = label_sets.map(lambda labels, target=label_id: int(target in labels))

    expected_counts = {
        "admiration": {"test": 504, "train": 4106, "validation": 487},
        "anger": {"test": 198, "train": 1561, "validation": 194},
        "curiosity": {"test": 283, "train": 2183, "validation": 246},
        "sadness": {"test": 156, "train": 1318, "validation": 143},
    }
    observed_counts = {
        target: {
            split: int(part[target].sum())
            for split, part in frame.groupby("split", sort=True)
        }
        for target in TARGETS
    }
    assert observed_counts == expected_counts

    output_columns = [
        "comment_id",
        "text",
        *TARGETS,
        "split",
        "group_id",
        "source_tier",
        "label_ids",
        "emotion_label_count",
    ]
    output = frame[output_columns].copy()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT, index=False)
    manifest = {
        "dataset_id": "goemotions_preregistered_four_emotion_panel",
        "status": "READY_FOR_FROZEN_BENCHMARK",
        "source_snapshot": str(SOURCE.relative_to(BENCHMARK)),
        "source_snapshot_sha256": sha256(SOURCE),
        "source_revision": "3ef52111c1fabc301880818f5a547570e926a76b",
        "selection_lock": (
            "Targets were fixed before activation screening to cover distinct emotion families "
            "while requiring at least 150 positive examples in the untouched test split."
        ),
        "targets": TARGETS,
        "positive_rows": observed_counts,
        "rows": len(output),
        "splits": {
            split: len(part) for split, part in output.groupby("split", sort=True)
        },
        "row_order_identity": "identical comment_id order to the frozen neutral-label snapshot",
        "model_input_columns": ["text"],
        "excluded_from_model_input": [
            *TARGETS,
            "split",
            "group_id",
            "comment_id",
            "label_ids",
            "emotion_label_count",
        ],
        "output_sha256": sha256(OUTPUT),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
