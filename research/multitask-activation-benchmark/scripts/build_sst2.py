#!/usr/bin/env python3
"""Build an immutable, audited binary SST root-sentence task from the official ZIP."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "data" / "raw" / "sst" / "stanfordSentimentTreebank.zip"
OUT = ROOT / "data" / "processed" / "sst2" / "sst2_root_sentences.parquet"
MANIFEST = ROOT / "data" / "processed" / "sst2" / "manifest.json"
EXPECTED_ZIP_SHA256 = "3f5209483b46bbf129cacbbbe6ae02fe780407034f61cf6342b7833257c3f1db"
PREFIX = "stanfordSentimentTreebank/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def text_file(archive: zipfile.ZipFile, name: str):
    return io.TextIOWrapper(archive.open(PREFIX + name), encoding="utf-8")


def main() -> None:
    assert RAW.is_file(), f"missing official SST ZIP: {RAW}"
    assert sha256(RAW) == EXPECTED_ZIP_SHA256
    with zipfile.ZipFile(RAW) as archive:
        dictionary: dict[str, int] = {}
        for line in text_file(archive, "dictionary.txt"):
            phrase, phrase_id = line.rstrip("\n").rsplit("|", 1)
            dictionary[phrase] = int(phrase_id)

        labels: dict[int, float] = {}
        lines = text_file(archive, "sentiment_labels.txt")
        next(lines)
        for line in lines:
            phrase_id, score = line.rstrip("\n").split("|")
            labels[int(phrase_id)] = float(score)

        split_by_sentence: dict[int, str] = {}
        lines = text_file(archive, "datasetSplit.txt")
        next(lines)
        split_names = {1: "train", 2: "test", 3: "validation"}
        for line in lines:
            sentence_id, split_id = line.rstrip("\n").split(",")
            split_by_sentence[int(sentence_id)] = split_names[int(split_id)]

        root_texts = [
            line.rstrip("\n").replace("|", " ") for line in text_file(archive, "SOStr.txt")
        ]

    rows = []
    neutral = 0
    for sentence_id, text in enumerate(root_texts, start=1):
        score = labels[dictionary[text]]
        if score <= 0.4:
            label = 0
        elif score > 0.6:
            label = 1
        else:
            neutral += 1
            continue
        norm = normalized_text(text)
        rows.append(
            {
                "example_id": f"sst-{sentence_id:05d}",
                "sentence_id": sentence_id,
                "text": text,
                "label": label,
                "sentiment_score": score,
                "split": split_by_sentence[sentence_id],
                "group_id": hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16],
                "source_tier": "A",
            }
        )

    frame = pd.DataFrame(rows).sort_values("sentence_id").reset_index(drop=True)
    raw_counts = frame.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict("index")
    assert raw_counts == {
        "test": {0: 912, 1: 909},
        "train": {0: 3310, 1: 3610},
        "validation": {0: 428, 1: 444},
    }

    # SST contains repeated root strings, including exact train--test overlaps. Preserve the
    # official test and validation rows, then remove later duplicate training rows.
    split_priority = {"test": 0, "validation": 1, "train": 2}
    frame["_split_priority"] = frame["split"].map(split_priority)
    duplicated_rows = int(frame.duplicated("group_id", keep=False).sum())
    before_dedup = len(frame)
    frame = (
        frame.sort_values(["_split_priority", "sentence_id"])
        .drop_duplicates("group_id", keep="first")
        .drop(columns="_split_priority")
        .sort_values("sentence_id")
        .reset_index(drop=True)
    )
    removed_duplicate_rows = before_dedup - len(frame)
    counts = frame.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict("index")
    assert counts == {
        "test": {0: 912, 1: 909},
        "train": {0: 3304, 1: 3605},
        "validation": {0: 428, 1: 444},
    }
    overlap = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        left_groups = set(frame.loc[frame.split == left, "group_id"])
        right_groups = set(frame.loc[frame.split == right, "group_id"])
        overlap[f"{left}__{right}"] = len(left_groups & right_groups)
    assert not any(overlap.values())
    assert frame["group_id"].nunique() == len(frame)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT, index=False)
    manifest = {
        "dataset_id": "sentiment_sst2",
        "status": "READY_FOR_FROZEN_BENCHMARK",
        "source_url": "https://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip",
        "source_zip_bytes": RAW.stat().st_size,
        "source_zip_sha256": EXPECTED_ZIP_SHA256,
        "rows": len(frame),
        "raw_binary_rows_before_exact_dedup": before_dedup,
        "dropped_neutral_rows": neutral,
        "rows_in_duplicate_groups_before_repair": duplicated_rows,
        "removed_duplicate_training_rows": removed_duplicate_rows,
        "raw_counts_by_split_and_label": raw_counts,
        "counts_by_split_and_label": counts,
        "unique_normalized_text_groups": int(frame["group_id"].nunique()),
        "cross_split_exact_normalized_overlap": overlap,
        "label_rule": "negative if score <= 0.4; positive if score > 0.6; neutral scores in (0.4, 0.6] dropped",
        "source_tier": "A: human movie-review snippets with human sentiment judgments",
        "licence_boundary": "internal research; official README contains citation instructions but no explicit redistribution licence",
        "output_path": str(OUT.relative_to(ROOT)),
        "output_sha256": sha256(OUT),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
