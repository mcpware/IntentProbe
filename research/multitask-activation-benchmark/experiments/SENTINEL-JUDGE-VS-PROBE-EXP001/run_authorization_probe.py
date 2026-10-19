#!/usr/bin/env python3
"""Fit one frozen EXP002 authorization head and score DelegationCheck."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, disable_caching
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parents[1]
PROJECT = BENCHMARK.parents[1]
AUTH = PROJECT / "research/authorization-send-detector"
AUTH_EXP002 = AUTH / "experiments/EXP002-activation-screen"
AUTH_DATA = AUTH / "data/processed/exp002"
AUTH_CACHE = AUTH / ".cache/exp002"
SURFACE_ARCHIVE = AUTH / "experiments/EXP005-delegationcheck-full-context/scores.npz"
ARCHIVES = HERE / "activation_archives"
ARTIFACTS = HERE / "probe_artifacts"


def load_judge_helpers() -> Any:
    """Load the adjacent runner by path so module imports are working-directory agnostic."""
    path = HERE / "run_judge.py"
    spec = importlib.util.spec_from_file_location("sentinel_run_judge_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load shared sentinel runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


JUDGE = load_judge_helpers()


def model_slug(model_id: str) -> str:
    """Return the authorization-screen filesystem key."""
    return model_id.replace("/", "__")


def selected_candidate(model_id: str) -> dict[str, Any]:
    """Load the validation-selected EXP002 readout without reselecting it."""
    path = AUTH_EXP002 / "probe_results" / f"{model_slug(model_id)}.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["model"] != model_id or result["selection_metric"] != "auprc":
        raise RuntimeError(f"Unexpected authorization-screen result: {path}")
    best = result["best"]
    if best["pooling"] != "masked_mean":
        raise RuntimeError("Sentinel authorization extraction requires selected masked mean")
    return best


def labels(split: str) -> np.ndarray:
    """Load the unchanged silver authorization labels."""
    frame = pd.read_parquet(AUTH_DATA / f"{split}.parquet", columns=["label"])
    values = frame["label"].map({"deny": 0, "approve": 1})
    if values.isna().any():
        raise RuntimeError(f"Unexpected {split} label")
    return values.to_numpy(dtype=np.int8)


def fit_frozen_head(
    model_id: str, best: dict[str, Any], seed: int
) -> tuple[LogisticRegression, np.ndarray, np.ndarray, dict[str, float]]:
    """Refit exactly the selected head and reproduce its validation AUPRC."""
    slug = model_slug(model_id)
    selected = np.arange(best["start_layer"], best["end_layer"] + 1, dtype=np.int64)
    train_source = np.load(AUTH_CACHE / slug / "train_masked_mean.npy", mmap_mode="r")
    validation_source = np.load(
        AUTH_CACHE / slug / "validation_masked_mean.npy", mmap_mode="r"
    )
    train_raw = np.asarray(train_source[:, selected, :], dtype=np.float32)
    validation_raw = np.asarray(validation_source[:, selected, :], dtype=np.float32)
    mean = train_raw.mean(axis=0, keepdims=True)
    scale = train_raw.std(axis=0, keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)
    train_x = ((train_raw - mean) / scale).mean(axis=1)
    validation_x = ((validation_raw - mean) / scale).mean(axis=1)
    train_y = labels("train")
    validation_y = labels("validation")
    head = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=20261016,
    )
    head.fit(train_x, train_y)
    validation_score = head.predict_proba(validation_x)[:, 1]
    reproduced = float(average_precision_score(validation_y, validation_score))
    recorded = float(best["metrics"]["auprc"])
    delta = reproduced - recorded
    if abs(delta) > 1e-12:
        raise RuntimeError(
            f"Frozen validation reproduction failed: {reproduced} vs {recorded}"
        )
    return head, mean[0], scale[0], {
        "recorded_auprc": recorded,
        "reproduced_auprc": reproduced,
        "absolute_difference": abs(delta),
        "seed": seed,
    }


def extract_selected_layers(
    *,
    frame: pd.DataFrame,
    model_config: dict[str, Any],
    selected_layers: np.ndarray,
    max_length: int,
    batch_size: int,
    hidden_size: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Extract only the frozen masked-mean layers for the human evaluation rows."""
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    model = AutoModel.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        local_files_only=True,
        trust_remote_code=True,
        dtype="auto",
    ).eval()
    load_seconds = time.perf_counter() - load_started

    unbounded = tokenizer(
        frame["text"].astype(str).tolist(), add_special_tokens=True, truncation=False
    )
    token_lengths = (
        pd.Series(unbounded["input_ids"], dtype="object").str.len().to_numpy(dtype=np.int64)
    )
    features = np.empty(
        (len(frame), len(selected_layers), hidden_size), dtype=np.float32
    )
    dataset = Dataset.from_pandas(frame.loc[:, ["text"]], preserve_index=False)
    scoring_started = time.perf_counter()

    def extract_batch(
        batch: dict[str, list[str]], indices: list[int]
    ) -> dict[str, list[int]]:
        tokens = tokenizer(
            batch["text"],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            output = model(**tokens, output_hidden_states=True, return_dict=True)
        mask = tokens["attention_mask"].to(dtype=torch.float32)[:, :, None]
        pooled = torch.stack(
            [
                (output.hidden_states[int(layer)].float() * mask).sum(dim=1)
                / mask.sum(dim=1).clamp_min(1.0)
                for layer in selected_layers
            ],
            dim=1,
        )
        row_indices = np.asarray(indices, dtype=np.int64)
        features[row_indices] = pooled.cpu().numpy()
        return {"done": np.ones(len(indices), dtype=np.int8).tolist()}

    completed = dataset.map(
        extract_batch,
        batched=True,
        batch_size=batch_size,
        with_indices=True,
        remove_columns=["text"],
        load_from_cache_file=False,
        keep_in_memory=True,
        desc=f"{model_config['id']} authorization selected layers",
    )
    if len(completed) != len(frame) or int(np.asarray(completed["done"]).sum()) != len(frame):
        raise RuntimeError("Incomplete authorization activation extraction")
    return features, token_lengths, load_seconds, time.perf_counter() - scoring_started


def build_parser() -> argparse.ArgumentParser:
    """Build the runner CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("quality_anchor", "cost_anchor"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def main() -> None:
    """Create one reproducible human-transfer authorization score archive."""
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    config, manifest = JUDGE.load_contract()
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    disable_caching()
    frame = JUDGE.load_task_frame("authorization_to_execute", manifest)
    model_config = config["models"][args.model]
    best = selected_candidate(model_config["id"])
    selected_layers = np.arange(
        best["start_layer"], best["end_layer"] + 1, dtype=np.int64
    )
    head, mean, scale, validation_check = fit_frozen_head(
        model_config["id"], best, seed
    )

    ARCHIVES.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    archive_path = ARCHIVES / f"{args.model}__authorization_to_execute.npz"
    metadata_path = archive_path.with_suffix(".metadata.json")
    artifact_path = ARTIFACTS / f"{args.model}__authorization_to_execute.npz"
    if archive_path.exists() or metadata_path.exists() or artifact_path.exists():
        raise SystemExit(f"Refusing to overwrite an authorization artifact for {args.model}")

    features, lengths, load_seconds, extraction_seconds = extract_selected_layers(
        frame=frame,
        model_config=model_config,
        selected_layers=selected_layers,
        max_length=int(config["tasks"]["authorization_to_execute"]["max_length"]),
        batch_size=args.batch_size,
        hidden_size=int(mean.shape[-1]),
    )
    activation_x = ((features - mean[None, :, :]) / scale[None, :, :]).mean(axis=1)
    activation_score = head.predict_proba(activation_x)[:, 1]
    with np.load(SURFACE_ARCHIVE, allow_pickle=False) as surface:
        surface_ids = np.asarray(surface["id"]).astype(str)
        surface_score = np.asarray(surface["surface"], dtype=np.float64)
    expected_ids = frame["id"].to_numpy(dtype=str)
    if not np.array_equal(surface_ids, expected_ids):
        raise RuntimeError("Frozen TF-IDF IDs do not match the sentinel authorization frame")
    if not np.isfinite(activation_score).all() or not np.isfinite(surface_score).all():
        raise RuntimeError("Non-finite authorization score")

    np.savez_compressed(
        artifact_path,
        selected_layers=selected_layers,
        layer_mean=mean,
        layer_scale=scale,
        coefficient=head.coef_.astype(np.float64),
        intercept=head.intercept_.astype(np.float64),
    )
    np.savez_compressed(
        archive_path,
        id=expected_ids,
        y=frame["y"].to_numpy(dtype=np.int8),
        group_id=frame["group_id"].to_numpy(dtype=str),
        activation=activation_score.astype(np.float64),
        tfidf=surface_score,
    )
    max_length = int(config["tasks"]["authorization_to_execute"]["max_length"])
    metadata = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_alias": args.model,
        "model": model_config,
        "task": "authorization_to_execute",
        "selection_source": "authorization EXP002 validation only",
        "training_source": "authorization EXP002 silver train only",
        "human_evaluation_labels_used_for_training": False,
        "selected_candidate": best,
        "validation_reproduction": validation_check,
        "renderer": "DelegationCheck visible fields v1",
        "max_length": max_length,
        "full_token_length": {
            "min": int(lengths.min()),
            "median": float(np.median(lengths)),
            "p90": float(np.quantile(lengths, 0.9)),
            "max": int(lengths.max()),
            "rows_exceeding_cap": int((lengths > max_length).sum()),
        },
        "rows": int(len(frame)),
        "positive_rows": int(frame["y"].sum()),
        "group_count": int(frame["group_id"].nunique()),
        "batch_size": args.batch_size,
        "load_seconds": load_seconds,
        "extraction_seconds": extraction_seconds,
        "seconds_per_example": extraction_seconds / len(frame),
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "archive_sha256": JUDGE.sha256(archive_path),
        "probe_artifact_sha256": JUDGE.sha256(artifact_path),
        "surface_archive_sha256": JUDGE.sha256(SURFACE_ARCHIVE),
        "config_sha256": JUDGE.sha256(HERE / "config.json"),
        "evaluation_manifest_sha256": JUDGE.sha256(HERE / "evaluation_manifest.json"),
        "python": platform.python_version(),
        "packages": JUDGE.package_versions(),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
