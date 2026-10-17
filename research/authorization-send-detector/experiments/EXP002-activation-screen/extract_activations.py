#!/usr/bin/env python3
"""Extract every hidden layer with mean and last-token pooling for one model."""

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
PROJECT = HERE.parents[1]
REGISTRY = PROJECT / "MODEL_REGISTRY.json"
CONFIG = HERE / "config.json"
PROCESSED = PROJECT / "data" / "processed" / "exp002"
CACHE = PROJECT / ".cache" / "exp002"
MANIFESTS = HERE / "feature_manifests"


def sha256(path: Path) -> str:
    """Return SHA-256 for one input or generated feature file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slug(model_id: str) -> str:
    """Return a filesystem-safe stable model key."""
    return model_id.replace("/", "__")


def resolve_model(model_id: str) -> dict[str, Any]:
    """Resolve one exact activation-model registry entry."""
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if model_id not in config["activation_models"]:
        raise ValueError(f"{model_id} is not an EXP002 activation model")
    matches = [model for model in registry["models"] if model["id"] == model_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one registry record for {model_id}, got {len(matches)}")
    return matches[0]


def main() -> None:
    """Extract both official screen splits for one frozen model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    record = resolve_model(args.model)
    model_slug = slug(record["id"])
    model_cache = CACHE / model_slug
    model_cache.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config["sampling_seed"])
    np.random.seed(config["sampling_seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        record["id"], revision=record["revision"], local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = config["renderer"]["truncation_side"]

    load_started = time.perf_counter()
    model = AutoModel.from_pretrained(
        record["id"],
        revision=record["revision"],
        local_files_only=True,
        trust_remote_code=True,
        dtype="auto",
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started

    model_manifest: dict[str, Any] = {
        "schema_version": 1,
        "model": record,
        "model_class": type(model).__name__,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "batch_size": args.batch_size,
        "max_tokens": config["renderer"]["max_tokens"],
        "load_seconds": load_seconds,
        "splits": {},
    }
    disable_caching()

    for split in ("train", "validation"):
        split_started = time.perf_counter()
        input_path = PROCESSED / f"{split}.parquet"
        frame = pd.read_parquet(input_path, columns=["rendered"])
        expected_rows = config["screen_rows"][split]
        if len(frame) != expected_rows or frame["rendered"].isna().any():
            raise ValueError(f"{split}: expected {expected_rows} non-null rows, got {len(frame)}")

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
            pooling: model_cache / f"{split}_{pooling}.npy"
            for pooling in config["pooling"]
        }
        stores = {
            pooling: np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=np.float16,
                shape=(expected_rows, hidden_count, hidden_size),
            )
            for pooling, path in paths.items()
        }

        dataset = Dataset.from_dict({"rendered": frame["rendered"].tolist()})

        def extract_batch(batch: dict[str, list[str]], indices: list[int]) -> dict[str, list[int]]:
            """Extract one vectorized framework batch into preallocated feature arrays."""
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
            attention = tokens["attention_mask"].to(dtype=torch.float32)
            mask = attention[:, None, :, None]
            mean_pooled = (hidden * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
            positions = torch.arange(attention.shape[1])[None, :]
            last_indices = (positions * attention.to(dtype=positions.dtype)).max(dim=1).values
            gather_indices = last_indices[:, None, None, None].expand(
                -1, hidden.shape[1], 1, hidden.shape[3]
            )
            last_pooled = hidden.gather(dim=2, index=gather_indices).squeeze(2)
            row_indices = np.asarray(indices, dtype=np.int64)
            stores["masked_mean"][row_indices] = mean_pooled.cpu().numpy().astype(np.float16)
            stores["last_nonpadding"][row_indices] = last_pooled.cpu().numpy().astype(np.float16)
            return {"done": [1] * len(indices)}

        completed = dataset.map(
            extract_batch,
            batched=True,
            batch_size=args.batch_size,
            with_indices=True,
            remove_columns=["rendered"],
            load_from_cache_file=False,
            keep_in_memory=True,
            desc=f"{record['id']} {split}",
        )
        if len(completed) != expected_rows or sum(completed["done"]) != expected_rows:
            raise RuntimeError(f"{split}: incomplete extraction")
        for store in stores.values():
            store.flush()
        split_manifest = {
            "input_path": str(input_path.relative_to(PROJECT)),
            "input_sha256": sha256(input_path),
            "rows": expected_rows,
            "hidden_state_count": hidden_count,
            "hidden_size": hidden_size,
            "feature_dtype": "float16",
            "features": {
                pooling: {
                    "path": str(path.relative_to(PROJECT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for pooling, path in paths.items()
            },
            "elapsed_seconds": time.perf_counter() - split_started,
        }
        model_manifest["splits"][split] = split_manifest

    model_manifest.update(
        status="COMPLETED",
        finished_at=datetime.now(timezone.utc).isoformat(),
        peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    )
    output_path = MANIFESTS / f"{model_slug}.json"
    output_path.write_text(json.dumps(model_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(model_manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
