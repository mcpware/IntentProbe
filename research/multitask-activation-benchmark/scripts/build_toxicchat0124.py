#!/usr/bin/env python3
"""Build leakage-clean human-only ToxicChat toxicity and jailbreak tasks."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "data" / "raw" / "toxicchat0124"
OUT = ROOT / "data" / "processed" / "toxicchat0124" / "human_unique.parquet"
MANIFEST = ROOT / "data" / "processed" / "toxicchat0124" / "manifest.json"
REVISION = "29df8e4dba60e1f4af4b4075c0705c5b313548a8"
EXPECTED = {
    "train.csv": "702eb9b7cac96c3c35e28b9b95855a71f26f21afba8666f9d243f1fa469e81ed",
    "test.csv": "3c2e49889626f7738dca0a29bface0ba0a0595b2ffdd17f0e02f19df7c3c4c9b",
}
SEED = 20261017


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        split: {
            "rows": len(part),
            "toxicity_positive": int(part["toxicity"].sum()),
            "jailbreak_positive": int(part["jailbreaking"].sum()),
        }
        for split, part in frame.groupby("split", sort=True)
    }


def main() -> None:
    pieces = []
    for split in ("train", "test"):
        path = RAW / f"{split}.csv"
        assert sha256(path) == EXPECTED[path.name]
        part = pd.read_csv(path)
        assert len(part) == (5082 if split == "train" else 5083)
        part["original_split"] = split
        pieces.append(part)
    full = pd.concat(pieces, ignore_index=True)
    assert len(full) == 10165
    assert full["conv_id"].nunique() == len(full)

    human = full.loc[
        full["human_annotation"],
        ["conv_id", "user_input", "toxicity", "jailbreaking", "original_split"],
    ].copy()
    assert len(human) == 5654
    assert int(human["toxicity"].sum()) == 746
    assert int(human["jailbreaking"].sum()) == 204
    assert bool((human["jailbreaking"] <= human["toxicity"]).all())
    human["normalized_text"] = human["user_input"].map(normalize)
    duplicate_rows = int(human.duplicated("normalized_text", keep=False).sum())
    cross_overlap = len(
        set(human.loc[human.original_split == "train", "normalized_text"])
        & set(human.loc[human.original_split == "test", "normalized_text"])
    )
    assert duplicate_rows == 413
    assert cross_overlap == 99
    conflicts = human.groupby("normalized_text")[["toxicity", "jailbreaking"]].nunique().max(axis=1)
    assert int((conflicts > 1).sum()) == 0

    # Preserve the official test whenever a prompt repeats. Within a split, retain the first conv_id.
    human["_priority"] = human["original_split"].map({"test": 0, "train": 1})
    human = (
        human.sort_values(["_priority", "conv_id"])
        .drop_duplicates("normalized_text", keep="first")
        .drop(columns="_priority")
        .reset_index(drop=True)
    )
    assert len(human) == 5441

    original_train = human[human.original_split == "train"].copy()
    official_test = human[human.original_split == "test"].copy()
    joint_label = (
        original_train["toxicity"].astype(str) + ":" + original_train["jailbreaking"].astype(str)
    )
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    fit_index, validation_index = next(splitter.split(original_train, joint_label))
    original_train["split"] = "train"
    original_train.iloc[validation_index, original_train.columns.get_loc("split")] = "validation"
    official_test["split"] = "test"
    frame = pd.concat([original_train, official_test], ignore_index=True)
    frame["group_id"] = frame["normalized_text"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    )
    frame = frame.rename(columns={"user_input": "text"}).drop(columns="normalized_text")
    frame["source_tier"] = "A"
    assert frame["group_id"].nunique() == len(frame)
    split_groups = {split: set(part.group_id) for split, part in frame.groupby("split")}
    assert split_groups["train"].isdisjoint(split_groups["validation"])
    assert split_groups["train"].isdisjoint(split_groups["test"])
    assert split_groups["validation"].isdisjoint(split_groups["test"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT, index=False)
    manifest = {
        "dataset_id": "toxicchat0124_human_unique",
        "status": "READY_FOR_FROZEN_BENCHMARK",
        "source_repository": "lmsys/toxic-chat",
        "source_revision": REVISION,
        "source_license": "CC BY-NC 4.0",
        "raw_files": {
            name: {"sha256": digest, "bytes": (RAW / name).stat().st_size}
            for name, digest in EXPECTED.items()
        },
        "source_rows": len(full),
        "observed_human_annotation_rows": 5654,
        "observed_human_toxicity_positive": 746,
        "observed_human_jailbreak_positive": 204,
        "card_discrepancy": "Pinned files contain 5,654 human_annotation=true rows; the dataset card reports 5,634.",
        "rows_in_duplicate_groups_before_repair": duplicate_rows,
        "original_cross_split_exact_overlap_groups": cross_overlap,
        "label_conflict_groups": 0,
        "rows_after_exact_dedup": len(frame),
        "counts": counts(frame),
        "validation_split": "20% stratified split of leakage-clean original train by joint toxicity:jailbreaking label",
        "split_seed": SEED,
        "model_input_columns": ["text"],
        "excluded_from_model_input": [
            "model_output",
            "human_annotation",
            "openai_moderation",
            "toxicity",
            "jailbreaking",
            "conv_id",
            "original_split",
        ],
        "output_sha256": sha256(OUT),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
