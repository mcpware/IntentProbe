from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_source_keys_and_revisions_are_unique() -> None:
    registry = json.loads((ROOT / "SOURCE_REGISTRY.json").read_text(encoding="utf-8"))
    keys = [source["key"] for source in registry["sources"]]
    assert len(keys) == len(set(keys))
    assert all(source.get("revision") for source in registry["sources"])


def test_model_ids_are_unique() -> None:
    registry = json.loads((ROOT / "MODEL_REGISTRY.json").read_text(encoding="utf-8"))
    model_ids = [model["id"] for model in registry["models"]]
    assert len(model_ids) == len(set(model_ids))


def test_release_training_never_includes_unclear_license() -> None:
    registry = json.loads((ROOT / "SOURCE_REGISTRY.json").read_text(encoding="utf-8"))
    unclear = [source for source in registry["sources"] if source["license"] == "unclear"]
    assert unclear
    assert all(source["role"].startswith("quarantine") for source in unclear)
