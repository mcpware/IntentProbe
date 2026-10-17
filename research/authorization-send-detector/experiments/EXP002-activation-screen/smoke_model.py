#!/usr/bin/env python3
"""Verify one pinned model can emit text hidden states on local CPU."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
REGISTRY = PROJECT / "MODEL_REGISTRY.json"
DATA = PROJECT / "data" / "processed" / "exp002" / "train.parquet"
OUTPUT = HERE / "smoke"


def model_record(model_id: str) -> dict[str, Any]:
    """Resolve one exact model record."""
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    matches = [model for model in registry["models"] if model["id"] == model_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one registry record for {model_id}, got {len(matches)}")
    return matches[0]


def slug(model_id: str) -> str:
    """Return a filesystem-safe stable model key."""
    return model_id.replace("/", "__")


def main() -> None:
    """Load one model, run four rows, and record hidden-state compatibility."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    record = model_record(args.model)
    result: dict[str, Any] = {
        "model": record,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    started = time.perf_counter()
    try:
        texts = pd.read_parquet(DATA, columns=["rendered"])["rendered"].iloc[:4].tolist()
        tokenizer = AutoTokenizer.from_pretrained(
            record["id"], revision=record["revision"], local_files_only=True, trust_remote_code=True
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has neither pad_token nor eos_token")
            tokenizer.pad_token = tokenizer.eos_token
        tokens = tokenizer(texts, padding=True, truncation=True, max_length=32, return_tensors="pt")
        config = AutoConfig.from_pretrained(
            record["id"], revision=record["revision"], local_files_only=True, trust_remote_code=True
        )
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
        forward_started = time.perf_counter()
        with torch.inference_mode():
            output = model(**tokens, output_hidden_states=True, return_dict=True)
        forward_seconds = time.perf_counter() - forward_started
        hidden_states = output.hidden_states
        if not hidden_states:
            raise ValueError("Model returned no hidden_states")
        shapes = [list(hidden_states[0].shape), list(hidden_states[-1].shape)]
        if shapes[0] != shapes[1] or shapes[0][0] != 4:
            raise ValueError(f"Unexpected hidden-state shapes: {shapes}")
        result.update(
            status="PASS",
            model_class=type(model).__name__,
            model_type=config.model_type,
            hidden_state_count=len(hidden_states),
            first_last_shapes=shapes,
            hidden_dtype=str(hidden_states[-1].dtype),
            load_seconds=load_seconds,
            forward_seconds=forward_seconds,
        )
    except Exception as exc:
        result.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
    result.update(
        finished_at=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter() - started,
        peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"{slug(record['id'])}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
