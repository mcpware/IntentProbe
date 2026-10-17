#!/usr/bin/env python3
"""Evaluate the frozen EXP002 activation probe once on the held-out test."""

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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModel, AutoTokenizer

from metrics_common import binary_metrics


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
CONFIG = HERE / "config.json"
FROZEN = HERE / "FROZEN_SELECTION.json"
REGISTRY = PROJECT / "MODEL_REGISTRY.json"
SCREEN = PROJECT / "data" / "processed" / "exp002"
TEST = PROJECT / "data" / "raw" / "huggingface" / "approve-or-deny" / "test.parquet"
CACHE = PROJECT / ".cache" / "exp002" / "heldout"
RESULT = HERE / "heldout_result.json"
RUNTIME_FIELDS = ["call", "user_request", "history"]


def load(path: Path) -> dict[str, Any]:
    """Load one UTF-8 JSON document."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    """Return SHA-256 for one evidence file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render(frame: pd.DataFrame) -> pd.Series:
    """Apply the frozen runtime renderer without label/rationale leakage."""
    parts = frame[RUNTIME_FIELDS].fillna("").astype(str)
    return (
        "PROPOSED ACTION:\n"
        + parts["call"]
        + "\nUSER REQUEST:\n"
        + parts["user_request"]
        + "\nHISTORY:\n"
        + parts["history"]
    )


def labels(frame: pd.DataFrame) -> np.ndarray:
    """Map the two documented labels to integers."""
    mapped = frame["label"].map({"deny": 0, "approve": 1})
    if mapped.isna().any():
        raise ValueError(f"Unexpected labels: {sorted(frame['label'].dropna().unique())}")
    return mapped.to_numpy(dtype=np.int8)


def resolve_model(model_id: str) -> dict[str, Any]:
    """Resolve the exact pinned model registry record."""
    records = [record for record in load(REGISTRY)["models"] if record["id"] == model_id]
    if len(records) != 1:
        raise ValueError(f"Expected one registry record for {model_id}, got {len(records)}")
    return records[0]


def make_head(head: dict[str, Any]) -> LogisticRegression:
    """Build the frozen linear head."""
    return LogisticRegression(
        C=head["C"],
        class_weight="balanced",
        solver=head["solver"],
        max_iter=head["max_iter"],
        random_state=head["random_state"],
    )


def fixed_operating_point(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    """Evaluate one validation-frozen threshold without test-set tuning."""
    predicted = scores >= threshold
    positive = y == 1
    negative = ~positive
    true_positive = int(np.sum(predicted & positive))
    false_positive = int(np.sum(predicted & negative))
    return {
        "threshold_from_validation": float(threshold),
        "approve_recall": true_positive / int(positive.sum()),
        "false_approve_rate": false_positive / int(negative.sum()),
        "approved_rows": int(predicted.sum()),
        "true_approves": true_positive,
        "false_approves": false_positive,
    }


def paired_bootstrap(
    y: np.ndarray,
    activation: np.ndarray,
    surface: np.ndarray,
    seed: int,
    resamples: int = 2000,
) -> dict[str, Any]:
    """Return paired row-bootstrap differences for threshold-free metrics."""
    rng = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = {"auprc": [], "auroc": []}
    n = len(y)
    for _ in range(resamples):
        indices = rng.integers(0, n, size=n)
        sample_y = y[indices]
        if len(np.unique(sample_y)) < 2:
            continue
        deltas["auprc"].append(
            float(
                average_precision_score(sample_y, activation[indices])
                - average_precision_score(sample_y, surface[indices])
            )
        )
        deltas["auroc"].append(
            float(
                roc_auc_score(sample_y, activation[indices])
                - roc_auc_score(sample_y, surface[indices])
            )
        )
    result: dict[str, Any] = {}
    for metric, values in deltas.items():
        array = np.asarray(values, dtype=np.float64)
        negative_or_zero = (np.sum(array <= 0) + 1) / (len(array) + 1)
        positive_or_zero = (np.sum(array >= 0) + 1) / (len(array) + 1)
        result[metric] = {
            "activation_minus_surface": float(
                (average_precision_score(y, activation) - average_precision_score(y, surface))
                if metric == "auprc"
                else (roc_auc_score(y, activation) - roc_auc_score(y, surface))
            ),
            "paired_row_bootstrap_95ci": [
                float(np.quantile(array, 0.025)),
                float(np.quantile(array, 0.975)),
            ],
            "two_sided_bootstrap_p": float(min(1.0, 2 * min(negative_or_zero, positive_or_zero))),
            "resamples": int(len(array)),
        }
    return result


def subgroup_metrics(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, list[dict[str, Any]]]:
    """Report descriptive performance for sufficiently large two-class groups."""
    output: dict[str, list[dict[str, Any]]] = {}
    y_all = labels(frame)
    for column in ("category", "difficulty", "lang"):
        groups: list[dict[str, Any]] = []
        for value, indices in frame.groupby(column, dropna=False).indices.items():
            index = np.asarray(indices, dtype=np.int64)
            y = y_all[index]
            if len(index) < 30 or len(np.unique(y)) < 2:
                continue
            groups.append(
                {
                    "value": str(value),
                    "n": int(len(index)),
                    "approve_n": int(y.sum()),
                    "auprc": float(average_precision_score(y, scores[index])),
                    "auroc": float(roc_auc_score(y, scores[index])),
                }
            )
        output[column] = sorted(groups, key=lambda row: row["value"])
    return output


def main() -> None:
    """Run the one-shot held-out comparison after verifying the freeze record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    started = time.perf_counter()
    if RESULT.exists():
        raise FileExistsError(f"Held-out result already exists: {RESULT}")
    frozen = load(FROZEN)
    if not frozen.get("frozen_before_heldout_read"):
        raise ValueError("Selection was not certified as frozen")
    config = load(CONFIG)
    record = resolve_model(frozen["model"])

    train = pd.read_parquet(SCREEN / "train.parquet", columns=["rendered", "label"])
    validation = pd.read_parquet(SCREEN / "validation.parquet", columns=["rendered", "label"])
    test = pd.read_parquet(
        TEST,
        columns=RUNTIME_FIELDS + ["label", "category", "difficulty", "lang"],
    )
    test["rendered"] = render(test)
    if len(test) != 3000 or test[["rendered", "label"]].isna().any().any():
        raise ValueError("Held-out test must contain 3,000 non-null rendered rows and labels")
    train_y = labels(train)
    test_y = labels(test)

    normalize = lambda values: values.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    train_keys = set(normalize(train["rendered"]))
    validation_keys = set(normalize(validation["rendered"]))
    test_keys = set(normalize(test["rendered"]))
    overlap = {
        "train_test_exact_rendered": len(train_keys & test_keys),
        "validation_test_exact_rendered": len(validation_keys & test_keys),
        "test_unique_rendered": len(test_keys),
    }
    if overlap["train_test_exact_rendered"] or overlap["validation_test_exact_rendered"]:
        raise ValueError(f"Exact rendered leakage into held-out test: {overlap}")

    surface_started = time.perf_counter()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    train_surface = vectorizer.fit_transform(train["rendered"])
    test_surface = vectorizer.transform(test["rendered"])
    surface_head = make_head(config["head"])
    surface_head.fit(train_surface, train_y)
    surface_scores = surface_head.predict_proba(test_surface)[:, 1]
    surface_seconds = time.perf_counter() - surface_started

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
    tokenizer.truncation_side = frozen["renderer"]["truncation_side"]
    model = AutoModel.from_pretrained(
        record["id"],
        revision=record["revision"],
        local_files_only=True,
        trust_remote_code=True,
        dtype="auto",
    )
    model.eval()

    start_layer = int(frozen["start_layer"])
    end_layer = int(frozen["end_layer"])
    width = end_layer - start_layer + 1
    probe = tokenizer(
        test["rendered"].iloc[:1].tolist(),
        padding=True,
        truncation=True,
        max_length=frozen["renderer"]["max_tokens"],
        return_tensors="pt",
    )
    with torch.inference_mode():
        probe_output = model(**probe, output_hidden_states=True, return_dict=True)
    if end_layer >= len(probe_output.hidden_states):
        raise ValueError("Frozen layer range exceeds model hidden states")
    hidden_size = int(probe_output.hidden_states[end_layer].shape[-1])
    del probe, probe_output

    CACHE.mkdir(parents=True, exist_ok=True)
    raw_path = CACHE / "frozen_test_selected_layers.npy"
    raw_test = np.lib.format.open_memmap(
        raw_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(test), width, hidden_size),
    )
    dataset = Dataset.from_dict({"rendered": test["rendered"].tolist()})
    disable_caching()
    extraction_started = time.perf_counter()

    def extract_batch(batch: dict[str, list[str]], indices: list[int]) -> dict[str, list[int]]:
        """Extract only the two frozen hidden outputs with masked-mean pooling."""
        tokens = tokenizer(
            batch["rendered"],
            padding=True,
            truncation=True,
            max_length=frozen["renderer"]["max_tokens"],
            return_tensors="pt",
        )
        with torch.inference_mode():
            output = model(**tokens, output_hidden_states=True, return_dict=True)
        hidden = torch.stack(output.hidden_states[start_layer : end_layer + 1], dim=1).float()
        mask = tokens["attention_mask"].to(dtype=torch.float32)[:, None, :, None]
        pooled = (hidden * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        raw_test[np.asarray(indices, dtype=np.int64)] = pooled.cpu().numpy().astype(np.float16)
        return {"done": [1] * len(indices)}

    completed = dataset.map(
        extract_batch,
        batched=True,
        batch_size=args.batch_size,
        with_indices=True,
        remove_columns=["rendered"],
        load_from_cache_file=False,
        keep_in_memory=True,
        desc=f"heldout {record['id']} layers {start_layer}-{end_layer}",
    )
    if len(completed) != len(test) or sum(completed["done"]) != len(test):
        raise RuntimeError("Incomplete held-out activation extraction")
    raw_test.flush()
    extraction_seconds = time.perf_counter() - extraction_started

    model_slug = record["id"].replace("/", "__")
    train_path = PROJECT / ".cache" / "exp002" / model_slug / f"train_{frozen['pooling']}.npy"
    raw_train_all = np.load(train_path, mmap_mode="r")
    raw_train = raw_train_all[:, start_layer : end_layer + 1].astype(np.float32)
    train_means = raw_train.mean(axis=0, keepdims=True)
    train_scales = raw_train.std(axis=0, keepdims=True)
    train_scales = np.where(train_scales < 1e-6, 1.0, train_scales)
    train_activation = ((raw_train - train_means) / train_scales).mean(axis=1)
    test_activation = ((np.asarray(raw_test, dtype=np.float32) - train_means) / train_scales).mean(axis=1)
    activation_head = make_head(frozen["head"])
    activation_head.fit(train_activation, train_y)
    activation_scores = activation_head.predict_proba(test_activation)[:, 1]

    surface_validation = load(HERE / "surface_baseline.json")["metrics"]
    activation_validation = frozen["validation_metrics"]
    surface_metrics = binary_metrics(test_y, surface_scores, config["sampling_seed"])
    activation_metrics = binary_metrics(test_y, activation_scores, config["sampling_seed"] + 10)
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "one_shot_heldout": True,
        "frozen_selection_sha256": sha256(FROZEN),
        "test": {
            "source": str(TEST.relative_to(PROJECT)),
            "sha256": sha256(TEST),
            "rows": int(len(test)),
            "approve_n": int(test_y.sum()),
            "deny_n": int((test_y == 0).sum()),
            "exact_rendered_overlap": overlap,
        },
        "activation": {
            "model": frozen["model"],
            "pooling": frozen["pooling"],
            "start_layer": start_layer,
            "end_layer": end_layer,
            "metrics": activation_metrics,
            "validation_frozen_operating_points": {
                "5pct": fixed_operating_point(
                    test_y,
                    activation_scores,
                    activation_validation["recall_at_5pct_false_approve"]["threshold"],
                ),
                "1pct": fixed_operating_point(
                    test_y,
                    activation_scores,
                    activation_validation["recall_at_1pct_false_approve"]["threshold"],
                ),
            },
            "subgroups": subgroup_metrics(test, activation_scores),
            "selected_feature_path": str(raw_path.relative_to(PROJECT)),
            "selected_feature_sha256": sha256(raw_path),
            "extraction_seconds": extraction_seconds,
            "examples_per_second": len(test) / extraction_seconds,
        },
        "matched_surface": {
            "model": "word_1_2_tfidf_logistic",
            "metrics": surface_metrics,
            "validation_frozen_operating_points": {
                "5pct": fixed_operating_point(
                    test_y,
                    surface_scores,
                    surface_validation["recall_at_5pct_false_approve"]["threshold"],
                ),
                "1pct": fixed_operating_point(
                    test_y,
                    surface_scores,
                    surface_validation["recall_at_1pct_false_approve"]["threshold"],
                ),
            },
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
            "elapsed_seconds": surface_seconds,
        },
        "paired_comparison": paired_bootstrap(
            test_y, activation_scores, surface_scores, config["sampling_seed"] + 20
        ),
        "artifacts": {
            "activation_head_coefficients_shape": list(activation_head.coef_.shape),
            "normalization_shape": list(train_means.shape),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    np.savez_compressed(
        CACHE / "frozen_probe_artifact.npz",
        coefficient=activation_head.coef_.astype(np.float32),
        intercept=activation_head.intercept_.astype(np.float32),
        mean=train_means.astype(np.float32),
        scale=train_scales.astype(np.float32),
        start_layer=np.asarray([start_layer], dtype=np.int16),
        end_layer=np.asarray([end_layer], dtype=np.int16),
    )
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "activation": result["activation"]["metrics"],
                "matched_surface": result["matched_surface"]["metrics"],
                "paired_comparison": result["paired_comparison"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
