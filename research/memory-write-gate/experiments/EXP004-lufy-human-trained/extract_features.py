#!/usr/bin/env python3
"""Extract frozen Qwen output-11 masked-mean features for LUFY."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = HERE / "config.json"
DATA = ROOT / "data" / "processed" / "exp003" / "lufy_majority.parquet"
FEATURES = HERE / "masked_mean_output11.npy"
MANIFEST = HERE / "feature_manifest.json"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Run bounded batched inference and write a row-aligned feature matrix."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    backbone = config["backbone"]
    frame = pd.read_parquet(DATA)
    assert len(frame) == 6470
    assert frame["user_name"].nunique() == 17
    assert int(frame["label"].sum()) == 332

    tokenizer = AutoTokenizer.from_pretrained(
        backbone["id"], revision=backbone["revision"], local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    lengths = np.asarray(
        tokenizer(
            frame["rendered"].tolist(),
            add_special_tokens=True,
            truncation=False,
            return_length=True,
        )["length"],
        dtype=int,
    )
    assert int(lengths.max()) <= backbone["max_tokens"]

    model = AutoModel.from_pretrained(
        backbone["id"],
        revision=backbone["revision"],
        local_files_only=True,
        dtype="auto",
    )
    model.eval()
    feature_matrix = np.lib.format.open_memmap(
        FEATURES, mode="w+", dtype=np.float32, shape=(len(frame), model.config.hidden_size)
    )
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(frame), args.batch_size):
            stop = min(start + args.batch_size, len(frame))
            tokens = tokenizer(
                frame["rendered"].iloc[start:stop].tolist(),
                padding=True,
                truncation=True,
                max_length=backbone["max_tokens"],
                return_tensors="pt",
            )
            output = model(**tokens, output_hidden_states=True, return_dict=True)
            mask = tokens["attention_mask"].to(dtype=torch.float32)[:, :, None]
            hidden = output.hidden_states[backbone["hidden_state_output"]].float()
            feature_matrix[start:stop] = (
                (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            ).cpu().numpy()
            if start and start % (args.batch_size * 50) == 0:
                print(f"extracted {start}/{len(frame)}", flush=True)
    feature_matrix.flush()
    completed_at = datetime.now(timezone.utc)
    observed = np.load(FEATURES, mmap_mode="r")
    assert observed.shape == (6470, 896)
    assert bool(np.isfinite(observed).all())
    manifest = {
        "experiment_id": "MEM-EXP004",
        "status": "FEATURES_COMPLETE",
        "rows": len(frame),
        "users": int(frame["user_name"].nunique()),
        "positive_rows": int(frame["label"].sum()),
        "shape": list(observed.shape),
        "dtype": str(observed.dtype),
        "batch_size": args.batch_size,
        "maximum_rendered_tokens": int(lengths.max()),
        "rows_over_cap": int(np.sum(lengths > backbone["max_tokens"])),
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "input_sha256": sha256(DATA),
        "features_sha256": sha256(FEATURES),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
