#!/usr/bin/env python3
"""Run one-forward multi-head latency or cross-domain score diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset, disable_caching
from transformers import AutoModel, AutoTokenizer


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
CONFIG = HERE / "config.json"
REGISTRY = HERE / "HEAD_REGISTRY.json"
WORKLOAD = HERE / "WORKLOAD_MANIFEST.json"
OUTPUTS = HERE / "outputs"


def sha256(path: Path) -> str:
    """Return a file SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def validate_frozen_inputs(
    config: dict[str, Any], registry: dict[str, Any], workload: dict[str, Any]
) -> None:
    """Refuse mismatched manifests or changed probe artifacts."""
    expected_config_hash = sha256(CONFIG)
    if registry["config_sha256"] != expected_config_hash:
        raise RuntimeError("HEAD_REGISTRY.json no longer matches config.json")
    if workload["config_sha256"] != expected_config_hash:
        raise RuntimeError("WORKLOAD_MANIFEST.json no longer matches config.json")
    if registry["head_count"] != len(config["heads"]):
        raise RuntimeError("Head count changed after registry freeze")
    for head in registry["heads"]:
        probe_path = PROJECT / head["probe_path"]
        if sha256(probe_path) != head["probe_sha256"]:
            raise RuntimeError(f"Probe changed after freeze: {head['name']}")


def load_workload(workload: dict[str, Any], mode: str) -> pd.DataFrame:
    """Load and align only the frozen real heldout workload rows."""
    frames: list[pd.DataFrame] = []
    id_key = "latency_ids" if mode == "latency" else "ood_ids"
    for source_name, source in workload["sources"].items():
        path = PROJECT / source["path"]
        if sha256(path) != source["file_sha256"]:
            raise RuntimeError(f"Workload source changed after freeze: {source_name}")
        id_column = source["id_column"]
        text_column = source["text_column"]
        split_column = source["split_column"]
        frame = pd.read_parquet(
            path, columns=[id_column, text_column, split_column]
        )
        frame = frame.loc[
            frame[split_column].eq(source["split"]), [id_column, text_column]
        ].copy()
        frame[id_column] = frame[id_column].astype(str)
        requested = pd.DataFrame(
            {
                "example_id": np.asarray(source[id_key], dtype=str),
                "source_order": np.arange(len(source[id_key]), dtype=np.int64),
            }
        )
        selected = requested.merge(
            frame.rename(columns={id_column: "example_id", text_column: "text"}),
            on="example_id",
            how="left",
            validate="one_to_one",
        )
        if selected["text"].isna().any():
            raise RuntimeError(f"Frozen workload IDs missing from {source_name}")
        selected.insert(0, "source", source_name)
        frames.append(selected)
    combined = pd.concat(frames, ignore_index=True)
    combined["row_key"] = combined["source"] + "::" + combined["example_id"]
    if combined["row_key"].duplicated().any():
        raise RuntimeError("Workload row keys are not unique")
    return combined[["row_key", "source", "example_id", "text"]]


def load_heads(registry: dict[str, Any], k: int) -> list[dict[str, Any]]:
    """Load the first K frozen logistic readouts."""
    heads: list[dict[str, Any]] = []
    for record in registry["heads"][:k]:
        with np.load(PROJECT / record["probe_path"], allow_pickle=False) as probe:
            heads.append(
                {
                    "name": record["name"],
                    "pooling": record["pooling"],
                    "start_layer": record["start_layer"],
                    "end_layer": record["end_layer"],
                    "coefficient": torch.from_numpy(
                        np.asarray(probe["coefficient"], dtype=np.float32)[0]
                    ),
                    "intercept": torch.tensor(
                        float(np.asarray(probe["intercept"], dtype=np.float32)[0]),
                        dtype=torch.float32,
                    ),
                    "means": torch.from_numpy(
                        np.asarray(probe["train_means"], dtype=np.float32)
                    ),
                    "scales": torch.from_numpy(
                        np.asarray(probe["train_scales"], dtype=np.float32)
                    ),
                }
            )
    return heads


def pooled_hidden(hidden: torch.Tensor, mask: torch.Tensor, pooling: str) -> torch.Tensor:
    """Pool a batch of token-level hidden states with the frozen rule."""
    expanded = mask[:, :, None]
    if pooling == "masked_mean":
        weights = expanded.to(dtype=hidden.dtype)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    if pooling == "masked_max":
        return hidden.masked_fill(~expanded, -torch.inf).amax(dim=1)
    raise RuntimeError(f"Unsupported pooling: {pooling}")


def apply_heads(
    hidden_states: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    heads: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Apply K independent normalized logistic readouts to one model forward."""
    mask = attention_mask.to(dtype=torch.bool)
    score_columns: dict[str, np.ndarray] = {}
    for head in heads:
        layer_vectors = torch.stack(
            [
                pooled_hidden(hidden_states[layer].float(), mask, head["pooling"])
                for layer in range(head["start_layer"], head["end_layer"] + 1)
            ],
            dim=1,
        )
        features = ((layer_vectors - head["means"]) / head["scales"]).mean(dim=1)
        logits = features @ head["coefficient"] + head["intercept"]
        score_columns[head["name"]] = torch.sigmoid(logits).cpu().numpy()
    return score_columns


def load_backbone(config: dict[str, Any]) -> tuple[Any, Any, float]:
    """Load the pinned tokenizer and base model from the local cache."""
    backbone = config["backbone"]
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        backbone["id"],
        revision=backbone["revision"],
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = backbone["truncation_side"]
    model = AutoModel.from_pretrained(
        backbone["id"],
        revision=backbone["revision"],
        local_files_only=True,
        trust_remote_code=True,
        dtype="auto",
    )
    model.eval()
    return tokenizer, model, time.perf_counter() - started


def tokenize(
    tokenizer: Any, texts: list[str], config: dict[str, Any]
) -> dict[str, torch.Tensor]:
    """Tokenize with the frozen cap and truncation rule."""
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=int(config["backbone"]["max_tokens"]),
        return_tensors="pt",
    )


def warmup(
    frame: pd.DataFrame,
    tokenizer: Any,
    model: Any,
    heads: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    """Warm tokenizer, model, pooling, and all loaded heads before timing."""
    count = int(config["latency"]["warmup_rows"])
    warmup_texts = frame["text"].iloc[:count].astype(str).tolist()
    for text_value in warmup_texts:
        tokens = tokenize(tokenizer, [text_value], config)
        with torch.inference_mode():
            output = model(**tokens, output_hidden_states=True, return_dict=True)
            apply_heads(output.hidden_states, tokens["attention_mask"], heads)


def run_latency(
    frame: pd.DataFrame,
    tokenizer: Any,
    model: Any,
    heads: list[dict[str, Any]],
    k: int,
    config: dict[str, Any],
) -> tuple[Path, Path]:
    """Measure one-forward K-head inference and a K-forward lower bound."""
    output_path = OUTPUTS / f"latency_k{k}.jsonl"
    lower_path = OUTPUTS / f"repeated_forward_lower_bound_k{k}.jsonl"
    if output_path.exists() or lower_path.exists():
        raise RuntimeError(f"Refusing to overwrite completed K={k} timing artifacts")
    warmup(frame, tokenizer, model, heads, config)
    disable_caching()
    dataset = Dataset.from_pandas(frame, preserve_index=False)

    def timed_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if len(batch["text"]) != 1:
            raise RuntimeError("Latency benchmark requires batch size one")
        total_start = time.perf_counter_ns()
        tokens = tokenize(tokenizer, [str(batch["text"][0])], config)
        tokenized_at = time.perf_counter_ns()
        with torch.inference_mode():
            output = model(**tokens, output_hidden_states=True, return_dict=True)
        forwarded_at = time.perf_counter_ns()
        scores = apply_heads(output.hidden_states, tokens["attention_mask"], heads)
        completed_at = time.perf_counter_ns()
        score_vector = np.asarray(
            [float(scores[head["name"]][0]) for head in heads], dtype=np.float64
        )
        return {
            "row_key": [batch["row_key"][0]],
            "source": [batch["source"][0]],
            "example_id": [str(batch["example_id"][0])],
            "k": [k],
            "token_count": [int(tokens["attention_mask"].sum().item())],
            "tokenization_ms": [(tokenized_at - total_start) / 1e6],
            "forward_ms": [(forwarded_at - tokenized_at) / 1e6],
            "heads_ms": [(completed_at - forwarded_at) / 1e6],
            "total_ms": [(completed_at - total_start) / 1e6],
            "head_score_sum": [float(score_vector.sum())],
            "head_score_l2": [float(np.linalg.norm(score_vector))],
        }

    measured = dataset.map(
        timed_batch,
        batched=True,
        batch_size=1,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        keep_in_memory=True,
        desc=f"one forward + {k} heads",
    )
    measured.to_pandas().to_json(output_path, orient="records", lines=True)

    lower_count = int(config["repeated_forward_lower_bound_rows"])
    lower_dataset = Dataset.from_pandas(frame.iloc[:lower_count], preserve_index=False)

    def repeated_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if len(batch["text"]) != 1:
            raise RuntimeError("Repeated-forward benchmark requires batch size one")
        total_start = time.perf_counter_ns()
        tokens = tokenize(tokenizer, [str(batch["text"][0])], config)
        tokenized_at = time.perf_counter_ns()
        checksum = 0.0
        with torch.inference_mode():
            for _ in range(k):
                output = model(**tokens, output_hidden_states=False, return_dict=True)
                checksum += float(output.last_hidden_state[0, -1, 0].item())
        completed_at = time.perf_counter_ns()
        return {
            "row_key": [batch["row_key"][0]],
            "source": [batch["source"][0]],
            "example_id": [str(batch["example_id"][0])],
            "k": [k],
            "token_count": [int(tokens["attention_mask"].sum().item())],
            "tokenization_ms": [(tokenized_at - total_start) / 1e6],
            "forward_ms": [(completed_at - tokenized_at) / 1e6],
            "total_ms": [(completed_at - total_start) / 1e6],
            "hidden_checksum": [checksum],
        }

    lower = lower_dataset.map(
        repeated_batch,
        batched=True,
        batch_size=1,
        remove_columns=lower_dataset.column_names,
        load_from_cache_file=False,
        keep_in_memory=True,
        desc=f"{k} repeated base forwards",
    )
    lower.to_pandas().to_json(lower_path, orient="records", lines=True)
    return output_path, lower_path


def run_ood(
    frame: pd.DataFrame,
    tokenizer: Any,
    model: Any,
    heads: list[dict[str, Any]],
    k: int,
    config: dict[str, Any],
) -> tuple[Path, None]:
    """Score all frozen cross-domain texts after one forward per batch."""
    if k != len(config["heads"]):
        raise RuntimeError("Cross-domain diagnostics run once with all frozen heads")
    output_path = OUTPUTS / f"ood_scores_k{k}.jsonl"
    if output_path.exists():
        raise RuntimeError("Refusing to overwrite completed OOD score artifact")
    disable_caching()
    dataset = Dataset.from_pandas(frame, preserve_index=False)

    def score_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        tokens = tokenize(tokenizer, [str(value) for value in batch["text"]], config)
        with torch.inference_mode():
            output = model(**tokens, output_hidden_states=True, return_dict=True)
            scores = apply_heads(output.hidden_states, tokens["attention_mask"], heads)
        result: dict[str, list[Any]] = {
            "row_key": list(batch["row_key"]),
            "source": list(batch["source"]),
            "example_id": [str(value) for value in batch["example_id"]],
            "token_count": tokens["attention_mask"].sum(dim=1).cpu().tolist(),
        }
        result.update({name: values.astype(float).tolist() for name, values in scores.items()})
        return result

    scored = dataset.map(
        score_batch,
        batched=True,
        batch_size=int(config["ood"]["batch_size"]),
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        keep_in_memory=True,
        desc=f"one forward + {k} heads: score shifts",
    )
    scored.to_pandas().to_json(output_path, orient="records", lines=True)
    return output_path, None


def main() -> None:
    """Run one frozen scaling cell and write auditable metadata."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("latency", "ood"), required=True)
    parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args()
    config = load_json(CONFIG)
    registry = load_json(REGISTRY)
    workload = load_json(WORKLOAD)
    validate_frozen_inputs(config, registry, workload)
    if args.k not in config["k_values"]:
        raise RuntimeError(f"K must be one of {config['k_values']}")
    metadata_path = OUTPUTS / f"{args.mode}_k{args.k}.metadata.json"
    if metadata_path.exists():
        raise RuntimeError(f"Refusing to overwrite {metadata_path}")
    OUTPUTS.mkdir(exist_ok=True)
    started_at = datetime.now(timezone.utc)
    process_started = time.perf_counter()
    frame = load_workload(workload, args.mode)
    heads = load_heads(registry, args.k)
    tokenizer, model, load_seconds = load_backbone(config)
    observed_outputs = int(model.config.num_hidden_layers) + 1
    observed_hidden_size = int(model.config.hidden_size)
    if observed_outputs != int(config["backbone"]["hidden_state_outputs"]):
        raise RuntimeError("Loaded model hidden-state count does not match freeze")
    if observed_hidden_size != int(config["backbone"]["hidden_size"]):
        raise RuntimeError("Loaded model hidden size does not match freeze")
    runner = run_latency if args.mode == "latency" else run_ood
    primary, secondary = runner(frame, tokenizer, model, heads, args.k, config)
    artifacts = [primary] + ([] if secondary is None else [secondary])
    metadata = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "mode": args.mode,
        "k": args.k,
        "head_names": [head["name"] for head in heads],
        "rows": len(frame),
        "model_id": config["backbone"]["id"],
        "model_revision": config["backbone"]["revision"],
        "model_load_seconds": load_seconds,
        "elapsed_seconds": time.perf_counter() - process_started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "torch_num_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "package_versions": {
            "datasets": datasets.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "config_sha256": sha256(CONFIG),
        "registry_sha256": sha256(REGISTRY),
        "workload_sha256": sha256(WORKLOAD),
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in artifacts
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
