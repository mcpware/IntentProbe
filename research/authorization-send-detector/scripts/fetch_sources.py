#!/usr/bin/env python3
"""Fetch pinned authorization source snapshots and write a provenance manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "SOURCE_REGISTRY.json"
RAW = ROOT / "data" / "raw"
MANIFESTS = ROOT / "data" / "manifests"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", help="Fetch only the named registry key.")
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_huggingface(source: dict[str, Any], workers: int) -> Path:
    target = RAW / "huggingface" / source["key"]
    snapshot_download(
        source["id"],
        repo_type="dataset",
        revision=source["revision"],
        local_dir=target,
        max_workers=workers,
    )
    return target


def fetch_github(source: dict[str, Any]) -> Path:
    target = RAW / "github" / source["key"]
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(target)], check=True)
        subprocess.run(
            ["git", "-C", str(target), "remote", "add", "origin", f"https://github.com/{source['id']}.git"], check=True
        )
    subprocess.run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", source["revision"]], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD"], check=True)
    return target


def fetch_arxiv(source: dict[str, Any]) -> Path:
    target = RAW / "papers" / f"{source['key']}_{source['id']}{source['revision']}.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://arxiv.org/pdf/{source['id']}{source['revision']}"
    with urllib.request.urlopen(url, timeout=120) as response:
        content = response.read()
    if not content.startswith(b"%PDF"):
        raise ValueError(f"arXiv response is not a PDF: {url}")
    target.write_bytes(content)
    return target


def inventory(target: Path) -> list[dict[str, Any]]:
    files = sorted(path for path in target.rglob("*") if path.is_file() and "/.git/" not in path.as_posix()) if target.is_dir() else [target]
    return [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]


def main() -> None:
    args = parse_args()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    selected = set(args.source or [])
    sources = [
        source
        for source in registry["sources"]
        if source.get("download") and (not selected or source["key"] in selected)
    ]
    unknown = selected - {source["key"] for source in registry["sources"]}
    if unknown:
        raise SystemExit(f"Unknown source keys: {sorted(unknown)}")

    MANIFESTS.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for source in sources:
        started = datetime.now(timezone.utc).isoformat()
        record: dict[str, Any] = {"source": source, "started_at": started}
        try:
            if source["kind"] == "huggingface_dataset":
                target = fetch_huggingface(source, args.max_workers)
            elif source["kind"] == "github_repo":
                target = fetch_github(source)
            elif source["kind"] == "arxiv_pdf":
                target = fetch_arxiv(source)
            else:
                raise ValueError(f"Unsupported source kind: {source['kind']}")
            files = inventory(target)
            record.update(
                status="GOT_DATA",
                target=str(target.relative_to(ROOT)),
                file_count=len(files),
                total_bytes=sum(item["bytes"] for item in files),
                files=files,
            )
        except Exception as exc:  # manifest must preserve failed retrievals
            record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        records.append(record)
        print(f"{source['key']}: {record['status']}", flush=True)

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "registry_sha256": sha256(REGISTRY),
        "records": records,
    }
    output = MANIFESTS / "retrieval_manifest.json"
    if selected and output.exists():
        previous = json.loads(output.read_text(encoding="utf-8")).get("records", [])
        by_key = {record["source"]["key"]: record for record in previous}
        by_key.update({record["source"]["key"]: record for record in records})
        manifest["records"] = [
            by_key[source["key"]]
            for source in registry["sources"]
            if source.get("download") and source["key"] in by_key
        ]
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    failed = [record["source"]["key"] for record in records if record["status"] != "GOT_DATA"]
    if failed:
        raise SystemExit(f"Failed sources: {failed}")


if __name__ == "__main__":
    main()
