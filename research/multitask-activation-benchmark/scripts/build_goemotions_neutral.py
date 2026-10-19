#!/usr/bin/env python3
"""Build an exact-text-clean GoEmotions neutral-label benchmark."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
RAW = BENCHMARK / "data" / "raw" / "goemotions"
OUTPUT = BENCHMARK / "data" / "processed" / "goemotions" / "neutral_label_unique.parquet"
MANIFEST = BENCHMARK / "data" / "manifests" / "goemotions_neutral.json"
REVISION = "3ef52111c1fabc301880818f5a547570e926a76b"
FILES = {
    "train.tsv": {"sha256": "1c254a142be5c00e80d819b9ae1bbd36d94b2eeb8f4b1271846508d57e57d9c5", "rows": 43410},
    "dev.tsv": {"sha256": "575489c079c9de1097062a01738f998590d6b7ead66dd1c9fd1d2ba01fd8bc62", "rows": 5426},
    "test.tsv": {"sha256": "0587b2dd8b27b97352adbfc3fb083d46005c8946657fdc2b1ca8b1cc7f1f8be4", "rows": 5427},
    "emotions.txt": {"sha256": "45c3ef86782d2a4d7fedcd6d8c111aa0d0e94720689bd164fac94fefb4495a89"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def main() -> None:
    for name, expected in FILES.items():
        assert sha256(RAW / name) == expected["sha256"]
    emotions = (RAW / "emotions.txt").read_text(encoding="utf-8").splitlines()
    assert len(emotions) == 28
    assert emotions[27] == "neutral"

    frames = []
    for split, name in (("train", "train.tsv"), ("validation", "dev.tsv"), ("test", "test.tsv")):
        frame = pd.read_csv(
            RAW / name,
            sep="\t",
            header=None,
            names=["text", "label_ids", "comment_id"],
            dtype=str,
            keep_default_na=False,
        )
        assert len(frame) == FILES[name]["rows"]
        frame["split"] = split
        frame["neutral_label"] = frame["label_ids"].map(
            lambda value: int("27" in value.split(","))
        )
        frame["emotion_label_count"] = frame["label_ids"].map(lambda value: len(value.split(",")))
        frame["normalized_text"] = frame["text"].map(normalize)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    assert len(raw) == 54263
    assert raw["comment_id"].nunique() == len(raw)
    assert int(raw["neutral_label"].sum()) == 17772

    grouped = raw.groupby("normalized_text", sort=False)
    label_cardinality = grouped["neutral_label"].nunique()
    conflict_groups = set(label_cardinality[label_cardinality > 1].index)
    assert len(conflict_groups) == 38
    conflict_rows = int(raw["normalized_text"].isin(conflict_groups).sum())
    assert conflict_rows == 116
    clean = raw[~raw["normalized_text"].isin(conflict_groups)].copy()
    clean["priority"] = clean["split"].map({"test": 0, "validation": 1, "train": 2})
    clean = clean.sort_values(["priority"], kind="stable").drop_duplicates(
        "normalized_text", keep="first"
    )
    clean["group_id"] = clean["normalized_text"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    clean["source_tier"] = "A_natural_reddit_comment_human_labels"
    clean = clean[
        [
            "comment_id",
            "text",
            "neutral_label",
            "split",
            "group_id",
            "source_tier",
            "label_ids",
            "emotion_label_count",
        ]
    ].reset_index(drop=True)
    observed_counts = {
        split: {"rows": len(part), "neutral_positive": int(part["neutral_label"].sum())}
        for split, part in clean.groupby("split", sort=True)
    }
    assert observed_counts == {
        "test": {"rows": 5409, "neutral_positive": 1778},
        "train": {"rows": 43095, "neutral_positive": 14146},
        "validation": {"rows": 5401, "neutral_positive": 1762},
    }
    assert clean["group_id"].nunique() == len(clean)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(OUTPUT, index=False)
    manifest = {
        "dataset_id": "goemotions_neutral_label_unique",
        "status": "READY_FOR_FROZEN_BENCHMARK",
        "official_source": "https://github.com/google-research/google-research/tree/master/goemotions",
        "source_revision": REVISION,
        "raw_files": {
            name: {"bytes": (RAW / name).stat().st_size, "sha256": sha256(RAW / name)}
            for name in FILES
        },
        "source_rows": len(raw),
        "source_neutral_positive": int(raw["neutral_label"].sum()),
        "source_neutral_with_other_labels": int(
            ((raw["neutral_label"] == 1) & (raw["emotion_label_count"] > 1)).sum()
        ),
        "normalized_exact_text_groups": int(raw["normalized_text"].nunique()),
        "rows_in_duplicate_groups": int(raw["normalized_text"].duplicated(keep=False).sum()),
        "cross_split_exact_text_groups": int(
            sum(group["split"].nunique() > 1 for _, group in grouped)
        ),
        "neutral_label_conflict_groups_removed": len(conflict_groups),
        "neutral_label_conflict_rows_removed": conflict_rows,
        "rows_after_conflict_removal_and_exact_dedup": len(clean),
        "splits": observed_counts,
        "dedup_priority": ["test", "validation", "train"],
        "independent_unit_limitation": "Filtered files retain unique comment IDs but omit author and thread IDs.",
        "model_input_columns": ["text"],
        "excluded_from_model_input": [
            "neutral_label",
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
