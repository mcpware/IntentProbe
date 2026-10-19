#!/usr/bin/env python3
"""Build an exact-deduplicated, stratified SMS Spam benchmark snapshot."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
RAW = BENCHMARK / "data" / "raw" / "sms_spam_uci" / "sms_spam_collection.zip"
OUTPUT = BENCHMARK / "data" / "processed" / "sms_spam_uci" / "sms_spam_unique.parquet"
MANIFEST = BENCHMARK / "data" / "manifests" / "sms_spam_uci.json"
EXPECTED_ARCHIVE_SHA256 = "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3"
SEED = 20261017


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        split: {"rows": len(part), "spam_positive": int(part["label"].sum())}
        for split, part in frame.groupby("split", sort=True)
    }


def main() -> None:
    assert sha256(RAW) == EXPECTED_ARCHIVE_SHA256
    with zipfile.ZipFile(RAW) as archive:
        lines = archive.read("SMSSpamCollection").decode("utf-8").splitlines()
    records = []
    for source_row, line in enumerate(lines, start=1):
        label, text = line.split("\t", 1)
        assert label in {"ham", "spam"}
        records.append(
            {
                "source_row": source_row,
                "text": text,
                "label": int(label == "spam"),
                "normalized_text": normalize(text),
            }
        )
    raw = pd.DataFrame(records)
    assert len(raw) == 5574
    assert raw["label"].value_counts().to_dict() == {0: 4827, 1: 747}

    groups = raw.groupby("normalized_text", sort=False)
    conflicts = groups["label"].nunique()
    assert int((conflicts > 1).sum()) == 0
    duplicate_sizes = groups.size()
    unique = raw.drop_duplicates("normalized_text", keep="first").copy()
    assert len(unique) == 5159
    assert unique["label"].value_counts().to_dict() == {0: 4517, 1: 642}

    train_index, remainder_index = train_test_split(
        unique.index,
        test_size=0.30,
        random_state=SEED,
        stratify=unique["label"],
    )
    validation_index, test_index = train_test_split(
        remainder_index,
        test_size=0.50,
        random_state=SEED,
        stratify=unique.loc[remainder_index, "label"],
    )
    unique["split"] = "train"
    unique.loc[validation_index, "split"] = "validation"
    unique.loc[test_index, "split"] = "test"
    unique["group_id"] = unique["normalized_text"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    unique["message_id"] = unique["group_id"].map(lambda value: f"sms-{value[:16]}")
    unique["source_tier"] = "A_natural_human_message_corpus"
    unique = unique[
        ["message_id", "text", "label", "split", "group_id", "source_tier", "source_row"]
    ].reset_index(drop=True)

    split_counts = counts(unique)
    assert split_counts == {
        "test": {"rows": 774, "spam_positive": 96},
        "train": {"rows": 3611, "spam_positive": 449},
        "validation": {"rows": 774, "spam_positive": 97},
    }
    assert unique["group_id"].nunique() == len(unique)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    unique.to_parquet(OUTPUT, index=False)
    manifest = {
        "dataset_id": "sms_spam_uci_unique",
        "status": "READY_FOR_FROZEN_BENCHMARK",
        "official_source": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
        "uci_id": 228,
        "dataset_doi": "10.24432/C5CC84",
        "archive_bytes": RAW.stat().st_size,
        "archive_sha256": sha256(RAW),
        "source_rows": len(raw),
        "source_ham": int((raw["label"] == 0).sum()),
        "source_spam": int((raw["label"] == 1).sum()),
        "duplicate_groups": int((duplicate_sizes > 1).sum()),
        "rows_in_duplicate_groups": int(duplicate_sizes[duplicate_sizes > 1].sum()),
        "label_conflict_groups": int((conflicts > 1).sum()),
        "rows_after_exact_dedup": len(unique),
        "splits": split_counts,
        "split_seed": SEED,
        "split_protocol": "70/15/15 stratified split after normalized exact-text de-duplication",
        "independent_unit_limitation": "The merged collection does not retain per-message source corpus, author, or sender IDs.",
        "model_input_columns": ["text"],
        "excluded_from_model_input": ["label", "split", "group_id", "source_row", "message_id"],
        "output_sha256": sha256(OUTPUT),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
