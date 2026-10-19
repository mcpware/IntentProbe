#!/usr/bin/env python3
"""Build a leakage-repaired multi-attribute Unhealthy Comments snapshot."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
RAW = BENCHMARK / "data" / "raw" / "unhealthy_comments"
OUTPUT = BENCHMARK / "data" / "processed" / "unhealthy_comments" / "ucc_unique.parquet"
MANIFEST = BENCHMARK / "data" / "manifests" / "unhealthy_comments.json"
REVISION = "55934d0079073095af8b69807d235e40c8aceff4"
FILES = {
    "train.csv": "8ed2018ae06b9f3b44e3642a8fe05aab45475a3dc07726a8d5af097e5fac7513",
    "val.csv": "4968adb126904bb8778ff4da5c91db5e98f065a8ce3d11026017f227bf47ee69",
    "test.csv": "e657c55978bd67290c1867b04dc64a0ee8c868073979953656349d54cacb1758",
}
LABELS = (
    "antagonize",
    "condescending",
    "dismissive",
    "generalisation",
    "generalisation_unfair",
    "healthy",
    "hostile",
    "sarcastic",
)


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
        assert sha256(RAW / name) == expected

    frames = []
    for split, name in (("train", "train.csv"), ("validation", "val.csv"), ("test", "test.csv")):
        frame = pd.read_csv(RAW / name)
        frame["split"] = split
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    assert len(raw) == 44355
    assert raw["_unit_id"].nunique() == len(raw)
    missing_text_rows = int(raw["comment"].isna().sum())
    assert missing_text_rows == 1
    raw = raw[raw["comment"].notna()].copy()
    assert len(raw) == 44354
    assert all(set(raw[label].dropna().unique()) <= {0, 1} for label in LABELS)
    raw["normalized_text"] = raw["comment"].astype(str).map(normalize)

    grouped = raw.groupby("normalized_text", sort=False)
    conflict_groups: set[str] = set()
    for label in LABELS:
        cardinality = grouped[label].nunique()
        conflict_groups.update(cardinality[cardinality > 1].index)
    assert len(conflict_groups) == 2
    assert int(raw["normalized_text"].isin(conflict_groups).sum()) == 4

    clean = raw[~raw["normalized_text"].isin(conflict_groups)].copy()
    clean["priority"] = clean["split"].map({"test": 0, "validation": 1, "train": 2})
    clean = clean.sort_values("priority", kind="stable").drop_duplicates(
        "normalized_text", keep="first"
    )
    clean["group_id"] = clean["normalized_text"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    clean["source_tier"] = "A_natural_news_comment_human_labels"
    clean["comment_id"] = clean["_unit_id"].map(lambda value: f"ucc-{int(value)}")
    clean = clean[
        [
            "comment_id",
            "comment",
            *LABELS,
            "split",
            "group_id",
            "source_tier",
            "_trusted_judgments",
            *[f"{label}:confidence" for label in LABELS],
        ]
    ].rename(columns={"comment": "text"}).reset_index(drop=True)

    assert len(clean) == 44323
    assert clean.groupby("split").size().to_dict() == {
        "test": 4424,
        "train": 35472,
        "validation": 4427,
    }
    assert clean["group_id"].nunique() == len(clean)
    positive_rows = {
        label: {
            split: int(part[label].sum())
            for split, part in clean.groupby("split", sort=True)
        }
        for label in LABELS
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(OUTPUT, index=False)
    manifest = {
        "dataset_id": "unhealthy_comments_ucc_unique",
        "status": "READY_FOR_FROZEN_BENCHMARK",
        "official_source": "https://github.com/conversationai/unhealthy-conversations",
        "source_revision": REVISION,
        "licence": "CC BY-NC-SA 4.0",
        "raw_files": {
            name: {"bytes": (RAW / name).stat().st_size, "sha256": sha256(RAW / name)}
            for name in FILES
        },
        "source_rows": 44355,
        "rows_with_missing_text_excluded": missing_text_rows,
        "normalized_exact_text_groups": int(raw["normalized_text"].nunique()),
        "rows_in_duplicate_groups": int(raw["normalized_text"].duplicated(keep=False).sum()),
        "cross_split_exact_text_groups": int(
            sum(group["split"].nunique() > 1 for _, group in grouped)
        ),
        "label_conflict_groups_removed": len(conflict_groups),
        "rows_after_conflict_removal_and_exact_dedup": len(clean),
        "splits": {
            split: len(part) for split, part in clean.groupby("split", sort=True)
        },
        "positive_rows": positive_rows,
        "dedup_priority": ["test", "validation", "train"],
        "independent_unit_limitation": (
            "The released benchmark exposes comment IDs but no author or article/thread grouping."
        ),
        "model_input_columns": ["text"],
        "excluded_from_model_input": [
            *LABELS,
            "split",
            "group_id",
            "comment_id",
            "_trusted_judgments",
            *[f"{label}:confidence" for label in LABELS],
        ],
        "output_sha256": sha256(OUTPUT),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
