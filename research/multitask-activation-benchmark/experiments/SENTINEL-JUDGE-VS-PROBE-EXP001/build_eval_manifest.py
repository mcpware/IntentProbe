#!/usr/bin/env python3
"""Freeze sentinel evaluation IDs before model scoring."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parents[1]
PROJECT = BENCHMARK.parents[1]
CONFIG = HERE / "config.json"
OUTPUT = HERE / "evaluation_manifest.json"
SST2 = BENCHMARK / "data/processed/sst2/sst2_root_sentences.parquet"
TOXICCHAT = BENCHMARK / "data/processed/toxicchat0124/human_unique.parquet"
DELEGATION_RUN = (
    PROJECT
    / "research/authorization-send-detector/experiments/EXP004-delegationcheck-transfer/run.py"
)


def sha256(path: Path) -> str:
    """Return a file SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path) -> Any:
    """Load a local experiment module without installing the project."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stratified_ids(
    frame: pd.DataFrame, *, id_column: str, label_column: str, rows: int, seed: int
) -> list[str]:
    """Select one deterministic prevalence-preserving stratified subset."""
    assert frame[id_column].is_unique
    assert frame[label_column].nunique() == 2
    selected, _ = train_test_split(
        frame,
        train_size=rows,
        random_state=seed,
        shuffle=True,
        stratify=frame[label_column],
    )
    return sorted(selected[id_column].astype(str).tolist())


def summarize(frame: pd.DataFrame, id_column: str, label_column: str) -> dict[str, object]:
    """Return non-sensitive sample integrity counts."""
    return {
        "rows": int(len(frame)),
        "unique_ids": int(frame[id_column].nunique()),
        "label_counts": {
            str(key): int(value)
            for key, value in frame[label_column].value_counts().sort_index().items()
        },
    }


def main() -> None:
    """Write the immutable evaluation-ID manifest."""
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite frozen manifest: {OUTPUT}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    seed = int(config["seed"])

    sst = pd.read_parquet(SST2)
    sst_test = sst.loc[sst["split"].eq("test")].copy()
    toxic = pd.read_parquet(TOXICCHAT)
    toxic_test = toxic.loc[toxic["split"].eq("test")].copy()
    delegation = load_module("sentinel_delegation_source", DELEGATION_RUN).build_external_frame()

    sentiment_ids = stratified_ids(
        sst_test, id_column="example_id", label_column="label", rows=512, seed=seed
    )
    toxicity_ids = stratified_ids(
        toxic_test, id_column="conv_id", label_column="toxicity", rows=512, seed=seed
    )
    authorization_ids = sorted(delegation["id"].astype(str).tolist())

    selected_sst = sst_test.loc[sst_test["example_id"].astype(str).isin(sentiment_ids)]
    selected_toxic = toxic_test.loc[toxic_test["conv_id"].astype(str).isin(toxicity_ids)]
    selected_delegation = delegation.loc[delegation["id"].astype(str).isin(authorization_ids)]
    assert len(selected_sst) == 512
    assert len(selected_toxic) == 512
    assert len(selected_delegation) == 156

    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "seed": seed,
        "config_sha256": sha256(CONFIG),
        "tasks": {
            "authorization_to_execute": {
                "source_sha256": json.loads(
                    (HERE.parents[2] / "authorization-send-detector/experiments/EXP005-delegationcheck-full-context/config.json").read_text(
                        encoding="utf-8"
                    )
                )["source"]["dataset_sha256"],
                "ids": authorization_ids,
                "summary": summarize(selected_delegation, "id", "y"),
                "group_count": int(selected_delegation["group_id"].nunique()),
            },
            "toxicity": {
                "source_path": str(TOXICCHAT.relative_to(PROJECT)),
                "source_sha256": sha256(TOXICCHAT),
                "ids": toxicity_ids,
                "summary": summarize(selected_toxic, "conv_id", "toxicity"),
            },
            "sentiment": {
                "source_path": str(SST2.relative_to(PROJECT)),
                "source_sha256": sha256(SST2),
                "ids": sentiment_ids,
                "summary": summarize(selected_sst, "example_id", "label"),
            },
        },
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value["summary"] for key, value in manifest["tasks"].items()}, indent=2))


if __name__ == "__main__":
    main()
