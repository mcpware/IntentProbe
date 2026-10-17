#!/usr/bin/env python3
"""Download every pinned benchmark model and record a file inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "MODEL_REGISTRY.json"
OUTPUT = ROOT / "data" / "manifests" / "model_retrieval_manifest.json"
ALLOW_PATTERNS = [
    "*.safetensors",
    "*.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template*",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "*.py",
]


def sha256(path: Path) -> str:
    """Return the SHA-256 of one local model file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Fetch selected or all pinned model snapshots."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", help="Exact Hugging Face model ID; repeatable.")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    selected = set(args.model or [])
    known = {model["id"] for model in registry["models"]}
    unknown = selected - known
    if unknown:
        raise SystemExit(f"Unknown model IDs: {sorted(unknown)}")
    models = [model for model in registry["models"] if not selected or model["id"] in selected]

    records: list[dict[str, Any]] = []
    for model in models:
        record: dict[str, Any] = {
            "model": model,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            snapshot = Path(
                snapshot_download(
                    model["id"],
                    revision=model["revision"],
                    allow_patterns=ALLOW_PATTERNS,
                    max_workers=args.max_workers,
                )
            )
            files = sorted(path for path in snapshot.rglob("*") if path.is_file())
            inventory = [
                {
                    "path": str(path.relative_to(snapshot)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in files
            ]
            record.update(
                status="GOT_MODEL",
                # Keep the public manifest portable. The absolute Hugging Face
                # cache path is machine-specific and can expose a local username.
                snapshot_revision=snapshot.name,
                file_count=len(inventory),
                total_bytes=sum(item["bytes"] for item in inventory),
                files=inventory,
            )
        except Exception as exc:
            record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        records.append(record)
        print(f"{model['id']}: {record['status']}", flush=True)

    if selected and OUTPUT.exists():
        previous = json.loads(OUTPUT.read_text(encoding="utf-8")).get("records", [])
        by_id = {record["model"]["id"]: record for record in previous}
        by_id.update({record["model"]["id"]: record for record in records})
        records = [
            by_id[model["id"]]
            for model in registry["models"]
            if model["id"] in by_id
        ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "registry_sha256": sha256(REGISTRY),
        "allow_patterns": ALLOW_PATTERNS,
        "records": records,
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    failures = [record["model"]["id"] for record in records if record["status"] != "GOT_MODEL"]
    if failures:
        raise SystemExit(f"Failed models: {failures}")


if __name__ == "__main__":
    main()
