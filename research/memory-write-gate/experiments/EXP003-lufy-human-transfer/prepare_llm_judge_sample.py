#!/usr/bin/env python3
"""Prepare a deterministic balanced LUFY sample for prompted LLM baselines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data" / "processed" / "exp003" / "lufy_majority.parquet"
OUTPUT = ROOT / "data" / "processed" / "exp003" / "llm_judge_sample.jsonl"
UNANIMOUS_OUTPUT = (
    ROOT / "data" / "processed" / "exp003" / "llm_judge_unanimous_sample.jsonl"
)
MANIFEST = HERE / "llm_judge_sample_manifest.json"
SEED = 20261017


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matched_sample(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Return all positives plus one user/system-matched negative per positive."""
    positive = frame[frame["label"].eq(1)].copy()
    negative = frame[frame["label"].eq(0)].copy()
    rng = np.random.default_rng(seed)
    selected_negative_indices: list[int] = []
    available = set(negative.index.tolist())

    # Match every positive to one negative from the same user and chatbot
    # system when possible.  Fall back to the same user, then the full pool.
    for row in positive.itertuples():
        candidates = [
            index
            for index in available
            if negative.at[index, "user_name"] == row.user_name
            and negative.at[index, "system"] == row.system
        ]
        if not candidates:
            candidates = [
                index
                for index in available
                if negative.at[index, "user_name"] == row.user_name
            ]
        if not candidates:
            candidates = list(available)
        chosen = int(rng.choice(candidates))
        selected_negative_indices.append(chosen)
        available.remove(chosen)

    sample = pd.concat([positive, negative.loc[selected_negative_indices]], ignore_index=True)
    return sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def write_sample(sample: pd.DataFrame, output: Path) -> None:
    """Write quarantined judge inputs without publishing them in Git."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, row in sample.iterrows():
            handle.write(
                json.dumps(
                    {
                        "sample_id": f"J{index:04d}",
                        "row_id": row["row_id"],
                        "text": row["text"],
                        "label": int(row["label"]),
                        "user_name": row["user_name"],
                        "system": row["system"],
                        "unanimous": bool(row["unanimous"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    frame = pd.read_parquet(DATA)
    sample = matched_sample(frame, SEED)
    if len(sample) != 664 or int(sample["label"].sum()) != 332:
        raise ValueError("Balanced LUFY judge sample contract changed")
    unanimous = matched_sample(frame[frame["unanimous"]].copy(), SEED + 1)
    if len(unanimous) != 104 or int(unanimous["label"].sum()) != 52:
        raise ValueError("Unanimous balanced LUFY judge sample contract changed")
    write_sample(sample, OUTPUT)
    write_sample(unanimous, UNANIMOUS_OUTPUT)
    manifest = {
        "experiment_id": "MEM-EXP003",
        "status": "PREPARED",
        "seed": SEED,
        "selection": "all 332 majority-positive rows plus one same-user/system matched negative per positive",
        "rows": len(sample),
        "positive_rows": int(sample["label"].sum()),
        "negative_rows": int((sample["label"] == 0).sum()),
        "users": int(sample["user_name"].nunique()),
        "source_sha256": sha256(DATA),
        "output_sha256": sha256(OUTPUT),
        "unanimous_sample": {
            "rows": len(unanimous),
            "positive_rows": int(unanimous["label"].sum()),
            "negative_rows": int((unanimous["label"] == 0).sum()),
            "users": int(unanimous["user_name"].nunique()),
            "output_sha256": sha256(UNANIMOUS_OUTPUT),
        },
        "publication_boundary": "The JSONL contains quarantined historical LUFY text and is not committed or redistributed.",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
