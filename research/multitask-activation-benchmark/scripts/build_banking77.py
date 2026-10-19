#!/usr/bin/env python3
"""Build an exact-text-clean BANKING77 multiclass benchmark."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
RAW = BENCHMARK / "data" / "raw" / "banking77"
OUTPUT = BENCHMARK / "data" / "processed" / "banking77" / "banking77_unique.parquet"
MANIFEST = BENCHMARK / "data" / "manifests" / "banking77.json"
REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SEED = 20261017
FILES = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    "categories.json": "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
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
    for name, expected_hash in FILES.items():
        assert sha256(RAW / name) == expected_hash
    categories = json.loads((RAW / "categories.json").read_text(encoding="utf-8"))
    assert len(categories) == 77
    category_to_id = {category: index for index, category in enumerate(categories)}

    frames = []
    for split, name, expected_rows in (
        ("train", "train.csv", 10003),
        ("test", "test.csv", 3080),
    ):
        frame = pd.read_csv(RAW / name, dtype=str, keep_default_na=False)
        assert len(frame) == expected_rows
        assert set(frame["category"]) == set(categories)
        frame["split"] = split
        frame["normalized_text"] = frame["text"].map(normalize)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    groups = raw.groupby("normalized_text", sort=False)
    assert int((groups["category"].nunique() > 1).sum()) == 0
    duplicate_sizes = groups.size()
    cross_split_groups = int(sum(group["split"].nunique() > 1 for _, group in groups))
    assert cross_split_groups == 7

    raw["priority"] = raw["split"].map({"test": 0, "train": 1})
    clean = raw.sort_values(["priority"], kind="stable").drop_duplicates(
        "normalized_text", keep="first"
    ).copy()
    train_pool = clean.index[clean["split"] == "train"]
    _, validation_index = train_test_split(
        train_pool,
        test_size=0.15,
        random_state=SEED,
        stratify=clean.loc[train_pool, "category"],
    )
    clean.loc[validation_index, "split"] = "validation"
    clean["label_id"] = clean["category"].map(category_to_id).astype(int)
    clean["group_id"] = clean["normalized_text"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    clean["example_id"] = clean["group_id"].map(lambda value: f"banking77-{value[:16]}")
    clean["source_tier"] = "A_human_banking_query_intent"
    clean = clean[
        ["example_id", "text", "category", "label_id", "split", "group_id", "source_tier"]
    ].reset_index(drop=True)

    split_counts = clean.groupby("split").size().to_dict()
    assert split_counts == {"test": 3079, "train": 8493, "validation": 1499}
    assert clean["group_id"].nunique() == len(clean)
    assert clean["label_id"].nunique() == 77
    class_ranges = {
        split: {
            "minimum_rows_per_class": int(part["label_id"].value_counts().min()),
            "maximum_rows_per_class": int(part["label_id"].value_counts().max()),
        }
        for split, part in clean.groupby("split", sort=True)
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(OUTPUT, index=False)
    manifest = {
        "dataset_id": "banking77_unique",
        "status": "READY_FOR_FROZEN_MULTICLASS_BENCHMARK",
        "official_source": "https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data",
        "source_revision": REVISION,
        "raw_files": {
            name: {"bytes": (RAW / name).stat().st_size, "sha256": sha256(RAW / name)}
            for name in FILES
        },
        "source_rows": len(raw),
        "classes": len(categories),
        "duplicate_groups": int((duplicate_sizes > 1).sum()),
        "rows_in_duplicate_groups": int(duplicate_sizes[duplicate_sizes > 1].sum()),
        "cross_split_exact_text_groups": cross_split_groups,
        "label_conflict_groups": 0,
        "rows_after_exact_dedup": len(clean),
        "splits": split_counts,
        "class_count_ranges": class_ranges,
        "validation_protocol": "15% label-stratified split of leakage-clean official train",
        "split_seed": SEED,
        "model_input_columns": ["text"],
        "excluded_from_model_input": ["category", "label_id", "split", "group_id", "example_id"],
        "output_sha256": sha256(OUTPUT),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
