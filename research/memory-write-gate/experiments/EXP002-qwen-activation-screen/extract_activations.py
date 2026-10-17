#!/usr/bin/env python3
"""Extract every Qwen hidden output for MEM-EXP002 development splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, disable_caching
from transformers import AutoModel, AutoTokenizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = HERE / "config.json"
PROCESSED = ROOT / "data" / "processed" / "exp002"
CACHE = ROOT / ".cache" / "exp002" / "Qwen__Qwen2.5-0.5B"
MANIFEST = HERE / "feature_manifest.json"


def sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Extract vectorized batches into preallocated memory-mapped arrays."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resume-train-from", type=int, default=0)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    model_config: dict[str, Any] = config["model"]
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config["head"]["random_state"])
    np.random.seed(config["head"]["random_state"])

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = config["renderer"]["truncation_side"]

    load_started = time.perf_counter()
    model = AutoModel.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        local_files_only=True,
        dtype="auto",
    )
    model.eval()
    manifest: dict[str, Any] = {
        "experiment_id": "MEM-EXP002",
        "status": "RUNNING",
        "config_sha256": sha256(CONFIG),
        "model": model_config,
        "model_class": type(model).__name__,
        "batch_size": args.batch_size,
        "max_tokens": config["renderer"]["max_tokens"],
        "load_seconds": time.perf_counter() - load_started,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "splits": {},
    }
    disable_caching()

    for split in ("train", "validation"):
        split_started = time.perf_counter()
        input_path = PROCESSED / f"{split}.parquet"
        frame = pd.read_parquet(input_path, columns=["rendered"])
        expected_rows = config["development_rows"][split]
        assert len(frame) == expected_rows
        assert frame["rendered"].notna().all()

        probe_tokens = tokenizer(
            frame["rendered"].iloc[:1].tolist(),
            padding=True,
            truncation=True,
            max_length=config["renderer"]["max_tokens"],
            return_tensors="pt",
        )
        with torch.inference_mode():
            probe_output = model(**probe_tokens, output_hidden_states=True, return_dict=True)
        hidden_count = len(probe_output.hidden_states)
        hidden_size = int(probe_output.hidden_states[-1].shape[-1])
        del probe_output, probe_tokens

        paths = {
            pooling: CACHE / f"{split}_{pooling}.npy" for pooling in config["pooling"]
        }
        resume_from = args.resume_train_from if split == "train" else 0
        if resume_from:
            stores = {pooling: np.load(path, mmap_mode="r+") for pooling, path in paths.items()}
            for store in stores.values():
                assert store.shape == (expected_rows, hidden_count, hidden_size)
                written = np.any(store[:, 0, :] != 0, axis=1)
                assert written[:resume_from].all()
                assert not written[resume_from:].any()
        else:
            stores = {
                pooling: np.lib.format.open_memmap(
                    path,
                    mode="w+",
                    dtype=np.float16,
                    shape=(expected_rows, hidden_count, hidden_size),
                )
                for pooling, path in paths.items()
            }
        dataset = Dataset.from_dict({"rendered": frame["rendered"].iloc[resume_from:].tolist()})

        def extract_batch(batch: dict[str, list[str]], indices: list[int]) -> dict[str, list[int]]:
            """Pool one framework-managed batch across all hidden outputs."""
            tokens = tokenizer(
                batch["rendered"],
                padding=True,
                truncation=True,
                max_length=config["renderer"]["max_tokens"],
                return_tensors="pt",
            )
            with torch.inference_mode():
                output = model(**tokens, output_hidden_states=True, return_dict=True)
            hidden = torch.stack(output.hidden_states, dim=1).float()
            attention = tokens["attention_mask"].to(dtype=torch.bool)
            mask = attention[:, None, :, None]
            mean_pooled = (hidden * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1)
            max_pooled = hidden.masked_fill(~mask, float("-inf")).amax(dim=2)
            positions = torch.arange(attention.shape[1])[None, :]
            last_indices = (positions * attention.to(dtype=positions.dtype)).max(dim=1).values
            gather_indices = last_indices[:, None, None, None].expand(
                -1, hidden.shape[1], 1, hidden.shape[3]
            )
            last_pooled = hidden.gather(dim=2, index=gather_indices).squeeze(2)
            row_indices = np.asarray(indices, dtype=np.int64) + resume_from
            stores["masked_mean"][row_indices] = mean_pooled.cpu().numpy().astype(np.float16)
            stores["last_nonpadding"][row_indices] = last_pooled.cpu().numpy().astype(np.float16)
            stores["masked_max"][row_indices] = max_pooled.cpu().numpy().astype(np.float16)
            return {"done": [1] * len(indices)}

        completed = dataset.map(
            extract_batch,
            batched=True,
            batch_size=args.batch_size,
            with_indices=True,
            remove_columns=["rendered"],
            load_from_cache_file=False,
            keep_in_memory=True,
            desc=f"MEM-EXP002 {split}",
        )
        assert len(completed) == expected_rows - resume_from
        assert sum(completed["done"]) == expected_rows - resume_from
        for store in stores.values():
            store.flush()
        manifest["splits"][split] = {
            "input_path": str(input_path.relative_to(ROOT)),
            "input_sha256": sha256(input_path),
            "rows": expected_rows,
            "resumed_from_row": resume_from,
            "hidden_state_count": hidden_count,
            "hidden_size": hidden_size,
            "feature_dtype": "float16",
            "elapsed_seconds": time.perf_counter() - split_started,
            "features": {
                pooling: {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for pooling, path in paths.items()
            },
        }
        MANIFEST.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    manifest.update(
        status="COMPLETED",
        finished_at=datetime.now(timezone.utc).isoformat(),
        peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    )
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": manifest["status"], "splits": manifest["splits"]}, indent=2))


if __name__ == "__main__":
    main()
