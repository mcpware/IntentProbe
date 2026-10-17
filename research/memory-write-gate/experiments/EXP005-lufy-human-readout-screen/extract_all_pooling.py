#!/usr/bin/env python3
"""Cache every Qwen hidden output under three token-pooling rules."""

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
CACHE = HERE / "cache"
MANIFEST = HERE / "extraction_manifest.json"
POOLING = ("masked_mean", "last_nonpadding", "masked_max")


def sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Extract one row-aligned float16 cache per pooling rule."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
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
    assert model.config.hidden_size == backbone["hidden_size"]
    CACHE.mkdir(parents=True, exist_ok=True)
    shape = (len(frame), backbone["hidden_state_outputs"], backbone["hidden_size"])
    caches = {
        name: np.lib.format.open_memmap(CACHE / f"{name}.npy", mode="w+", dtype=np.float16, shape=shape)
        for name in POOLING
    }

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
            hidden = torch.stack(output.hidden_states, dim=1).float()
            assert hidden.shape[1] == backbone["hidden_state_outputs"]
            mask = tokens["attention_mask"].to(dtype=torch.bool)
            float_mask = mask[:, None, :, None].to(dtype=torch.float32)
            mean = (hidden * float_mask).sum(dim=2) / float_mask.sum(dim=2).clamp_min(1.0)
            last_index = mask.sum(dim=1) - 1
            last = hidden[torch.arange(len(hidden)), :, last_index, :]
            maximum = hidden.masked_fill(~mask[:, None, :, None], -torch.inf).amax(dim=2)
            caches["masked_mean"][start:stop] = mean.cpu().numpy().astype(np.float16)
            caches["last_nonpadding"][start:stop] = last.cpu().numpy().astype(np.float16)
            caches["masked_max"][start:stop] = maximum.cpu().numpy().astype(np.float16)
            if start and start % (args.batch_size * 100) == 0:
                print(f"extracted {start}/{len(frame)}", flush=True)

    for cache in caches.values():
        cache.flush()
    completed_at = datetime.now(timezone.utc)
    files = {}
    for name in POOLING:
        path = CACHE / f"{name}.npy"
        observed = np.load(path, mmap_mode="r")
        assert observed.shape == shape
        assert bool(np.isfinite(observed).all())
        files[name] = {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    manifest = {
        "experiment_id": "MEM-EXP005",
        "status": "ALL_POOLING_FEATURES_COMPLETE",
        "rows": len(frame),
        "users": int(frame["user_name"].nunique()),
        "positive_rows": int(frame["label"].sum()),
        "shape_per_pooling": list(shape),
        "dtype": "float16",
        "batch_size": args.batch_size,
        "maximum_rendered_tokens": int(lengths.max()),
        "rows_over_cap": int(np.sum(lengths > backbone["max_tokens"])),
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "input_sha256": sha256(DATA),
        "files": files,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
