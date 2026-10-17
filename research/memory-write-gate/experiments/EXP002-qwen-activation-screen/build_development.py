#!/usr/bin/env python3
"""Build immutable MEM-EXP002 train and validation renderer outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = (
    ROOT
    / "data"
    / "raw"
    / "huggingface"
    / "memory-write-gate-20k"
    / "data"
    / "memory_write_gate_20000.jsonl"
)
OUTPUT = ROOT / "data" / "processed" / "exp002"
CONFIG = HERE / "config.json"


def sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Render development splits without exposing excluded metadata to the model."""
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert sha256(RAW) == config["source"]["raw_sha256"]
    frame = pd.read_json(RAW, lines=True)
    development = frame.loc[frame["split"].isin(["train", "validation"])].copy()
    development["rendered"] = (
        "USER_UTTERANCE\n"
        + development["text"].astype(str)
        + "\n\nMEMORY_WRITE_DECISION"
    )
    development["target"] = development["label"].map({"SKIP": 0, "SAVE": 1})
    assert development["target"].notna().all()
    assert development["rendered"].notna().all()
    assert (development.groupby("seed_id")["split"].nunique() == 1).all()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "experiment_id": "MEM-EXP002",
        "config_sha256": sha256(CONFIG),
        "raw_sha256": sha256(RAW),
        "renderer": config["renderer"],
        "splits": {},
    }
    for split in ("train", "validation"):
        split_frame = development.loc[
            development["split"].eq(split),
            ["id", "seed_id", "rendered", "target", "language", "category", "difficulty"],
        ].reset_index(drop=True)
        assert len(split_frame) == config["development_rows"][split]
        assert split_frame["seed_id"].nunique() == config["development_seed_families"][split]
        path = OUTPUT / f"{split}.parquet"
        split_frame.to_parquet(path, index=False)
        manifest["splits"][split] = {
            "rows": len(split_frame),
            "seed_families": int(split_frame["seed_id"].nunique()),
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
        }
    (HERE / "development_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
