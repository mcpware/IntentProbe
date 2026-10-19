#!/usr/bin/env python3
"""Fetch the pinned Unhealthy Comments Corpus files without extra packages."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
RAW = BENCHMARK / "data" / "raw" / "unhealthy_comments"
REVISION = "55934d0079073095af8b69807d235e40c8aceff4"
BASE = f"https://raw.githubusercontent.com/conversationai/unhealthy-conversations/{REVISION}"
FILES = {
    "train.csv": {
        "url": f"{BASE}/corpus/train.csv",
        "sha256": "8ed2018ae06b9f3b44e3642a8fe05aab45475a3dc07726a8d5af097e5fac7513",
    },
    "val.csv": {
        "url": f"{BASE}/corpus/val.csv",
        "sha256": "4968adb126904bb8778ff4da5c91db5e98f065a8ce3d11026017f227bf47ee69",
    },
    "test.csv": {
        "url": f"{BASE}/corpus/test.csv",
        "sha256": "e657c55978bd67290c1867b04dc64a0ee8c868073979953656349d54cacb1758",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, spec in FILES.items():
        target = RAW / name
        if not target.exists() or sha256(target) != spec["sha256"]:
            temporary = target.with_suffix(target.suffix + ".part")
            urllib.request.urlretrieve(spec["url"], temporary)
            if sha256(temporary) != spec["sha256"]:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"Hash mismatch for {name}")
            temporary.replace(target)
        records[name] = {"bytes": target.stat().st_size, "sha256": sha256(target)}
    print(json.dumps({"revision": REVISION, "files": records}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
