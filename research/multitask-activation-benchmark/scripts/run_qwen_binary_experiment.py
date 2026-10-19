#!/usr/bin/env python3
"""Shared hidden-state extraction and binary readout screen for frozen backbones."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, disable_caching
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer


BENCHMARK = Path(__file__).resolve().parents[1]
POOLING = ("masked_mean", "last_nonpadding", "masked_max")


def cache_directory(experiment: Path, config: dict[str, object]) -> Path:
    """Resolve an optional immutable shared activation cache."""
    cache_source = config.get("cache_source")
    if cache_source is None:
        return experiment / "cache"
    path = (BENCHMARK / str(cache_source)).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Shared activation cache does not exist: {path}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_experiment(path: str) -> tuple[Path, dict[str, object], Path, pd.DataFrame]:
    experiment = Path(path).resolve()
    config = json.loads((experiment / "config.json").read_text(encoding="utf-8"))
    data_path = BENCHMARK / str(config["dataset"]["input"])
    frame = pd.read_parquet(data_path).reset_index(drop=True)
    expected_splits = config["dataset"]["splits"]
    assert frame.groupby("split").size().to_dict() == {
        key: int(value) for key, value in sorted(expected_splits.items())
    }
    assert len(frame) == int(config["dataset"]["rows"])
    return experiment, config, data_path, frame


def extract(experiment_path: str, batch_size: int, resume: bool = False) -> None:
    experiment, config, data_path, frame = load_experiment(experiment_path)
    backbone = config["backbone"]
    cache_dir = cache_directory(experiment, config)
    tokenizer = AutoTokenizer.from_pretrained(
        backbone["id"],
        revision=backbone["revision"],
        local_files_only=True,
        trust_remote_code=bool(backbone.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = backbone["truncation_side"]
    texts = frame[str(config["dataset"]["text_column"])].astype(str).tolist()
    lengths = np.asarray(
        tokenizer(texts, add_special_tokens=True, truncation=False, return_length=True)["length"],
        dtype=np.int32,
    )
    audit = config["label_blind_token_audit"]
    assert int(lengths.max()) == int(audit["maximum_tokens"])
    assert int(np.sum(lengths > backbone["max_tokens"])) == int(audit["rows_over_cap"])

    model = AutoModel.from_pretrained(
        backbone["id"],
        revision=backbone["revision"],
        local_files_only=True,
        trust_remote_code=bool(backbone.get("trust_remote_code", True)),
        dtype="auto",
    )
    model.eval()
    probe_tokens = tokenizer(
        texts[:1],
        padding=True,
        truncation=True,
        max_length=int(backbone["max_tokens"]),
        return_tensors="pt",
    )
    with torch.inference_mode():
        probe_output = model(**probe_tokens, output_hidden_states=True, return_dict=True)
    observed_outputs = len(probe_output.hidden_states)
    observed_hidden_size = int(probe_output.hidden_states[-1].shape[-1])
    assert observed_outputs == int(backbone["hidden_state_outputs"])
    assert observed_hidden_size == int(backbone["hidden_size"])
    del probe_output, probe_tokens
    cache_dir.mkdir(parents=True, exist_ok=True)
    shape = (
        len(frame),
        int(backbone["hidden_state_outputs"]),
        int(backbone["hidden_size"]),
    )
    cache_paths = {name: cache_dir / f"{name}.npy" for name in POOLING}
    if resume:
        assert all(path.exists() for path in cache_paths.values())
        caches = {
            name: np.lib.format.open_memmap(path, mode="r+")
            for name, path in cache_paths.items()
        }
        assert all(cache.shape == shape and cache.dtype == np.float16 for cache in caches.values())
        written_masks = {
            name: np.any(cache[:, 0, :] != 0, axis=1) for name, cache in caches.items()
        }
        first_mask = written_masks[POOLING[0]]
        assert all(np.array_equal(first_mask, mask) for mask in written_masks.values())
        completed_mask = first_mask
        print(f"resuming with {int(completed_mask.sum())}/{len(frame)} rows complete", flush=True)
    else:
        caches = {
            name: np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float16, shape=shape
            )
            for name, path in cache_paths.items()
        }
        completed_mask = np.zeros(len(frame), dtype=bool)

    order = np.argsort(lengths, kind="stable")
    order = order[~completed_mask[order]]
    rows_already_complete = int(completed_mask.sum())
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    disable_caching()
    pending = frame.iloc[order][[str(config["dataset"]["text_column"])]].copy()
    pending.insert(0, "row_index", order.astype(np.int64))
    pending[str(config["dataset"]["text_column"])] = pending[
        str(config["dataset"]["text_column"])
    ].astype(str)
    dataset = Dataset.from_pandas(pending, preserve_index=False)

    def extract_batch(batch: dict[str, list[object]]) -> dict[str, list[int]]:
        """Extract one framework-managed vectorized batch."""
        batch_rows = np.asarray(batch["row_index"], dtype=np.int64)
        tokens = tokenizer(
            batch[str(config["dataset"]["text_column"])],
            padding=True,
            truncation=True,
            max_length=int(backbone["max_tokens"]),
            return_tensors="pt",
        )
        with torch.inference_mode():
            output = model(**tokens, output_hidden_states=True, return_dict=True)
        hidden = torch.stack(output.hidden_states, dim=1).float()
        assert hidden.shape[1] == int(backbone["hidden_state_outputs"])
        mask = tokens["attention_mask"].to(dtype=torch.bool)
        float_mask = mask[:, None, :, None].to(dtype=torch.float32)
        mean = (hidden * float_mask).sum(dim=2) / float_mask.sum(dim=2).clamp_min(1.0)
        last_index = mask.sum(dim=1) - 1
        last = hidden[torch.arange(len(hidden)), :, last_index, :]
        maximum = hidden.masked_fill(~mask[:, None, :, None], -torch.inf).amax(dim=2)
        caches["masked_mean"][batch_rows] = mean.cpu().numpy().astype(np.float16)
        caches["last_nonpadding"][batch_rows] = last.cpu().numpy().astype(np.float16)
        caches["masked_max"][batch_rows] = maximum.cpu().numpy().astype(np.float16)
        return {"done": np.ones(len(batch_rows), dtype=np.int8).tolist()}

    completed = dataset.map(
        extract_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        keep_in_memory=True,
        desc=f"{config['experiment_id']} hidden states",
    )
    assert len(completed) == len(order)
    assert int(np.asarray(completed["done"], dtype=np.int64).sum()) == len(order)

    for cache in caches.values():
        cache.flush()
    files = {}
    for name in POOLING:
        path = cache_dir / f"{name}.npy"
        observed = np.load(path, mmap_mode="r")
        assert observed.shape == shape
        assert bool(np.isfinite(observed).all())
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "experiment_id": config["experiment_id"],
        "status": "ALL_POOLING_FEATURES_COMPLETE",
        "rows": len(frame),
        "shape_per_pooling": list(shape),
        "dtype": "float16",
        "batch_size": batch_size,
        "resumed": resume,
        "rows_preexisting_at_start": rows_already_complete,
        "maximum_tokens_before_truncation": int(lengths.max()),
        "p50_tokens": float(np.percentile(lengths, 50)),
        "p95_tokens": float(np.percentile(lengths, 95)),
        "p99_tokens": float(np.percentile(lengths, 99)),
        "rows_over_cap": int(np.sum(lengths > int(backbone["max_tokens"]))),
        "truncation_rate": float(np.mean(lengths > int(backbone["max_tokens"]))),
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "input_sha256": sha256(data_path),
        "files": files,
    }
    (experiment / "extraction_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def candidates(outputs: int) -> list[tuple[str, int, int]]:
    values = []
    for pooling in POOLING:
        for width in (1, 2, 3):
            for start in range(outputs - width + 1):
                values.append((pooling, start, start + width - 1))
    assert len(values) == 9 * outputs - 9
    return values


def selected_features(
    cache: np.ndarray,
    rows: np.ndarray,
    start: int,
    end: int,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    cube = cache[rows, start : end + 1].astype(np.float32)
    return ((cube - means) / scales).mean(axis=1)


def score_candidate(
    cache_dir: Path,
    seed: int,
    pooling: str,
    start: int,
    end: int,
    train_rows: np.ndarray,
    train_y: np.ndarray,
    validation_rows: np.ndarray,
    validation_y: np.ndarray,
    final_output: int,
) -> dict[str, object]:
    cache = np.load(cache_dir / f"{pooling}.npy", mmap_mode="r")
    train_cube = cache[train_rows, start : end + 1].astype(np.float32)
    means = train_cube.mean(axis=0)
    raw_scales = train_cube.std(axis=0)
    scales = np.where(raw_scales < 1e-6, 1.0, raw_scales)
    train_x = ((train_cube - means) / scales).mean(axis=1)
    validation_x = selected_features(cache, validation_rows, start, end, means, scales)
    head = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=seed,
    )
    head.fit(train_x, train_y)
    score = head.predict_proba(validation_x)[:, 1]
    return {
        "pooling": pooling,
        "start_layer": start,
        "end_layer": end,
        "width": end - start + 1,
        "normalized_depth": ((start + end) / 2) / final_output,
        "validation_auprc": float(average_precision_score(validation_y, score)),
        "validation_auroc": float(roc_auc_score(validation_y, score)),
        "validation_balanced_accuracy_at_0_5": float(
            balanced_accuracy_score(validation_y, score >= 0.5)
        ),
    }


def fit_activation(
    cache_dir: Path,
    seed: int,
    winner: dict[str, object],
    fit_rows: np.ndarray,
    fit_y: np.ndarray,
    test_rows: np.ndarray,
) -> tuple[np.ndarray, LogisticRegression, np.ndarray, np.ndarray]:
    pooling = str(winner["pooling"])
    start = int(winner["start_layer"])
    end = int(winner["end_layer"])
    cache = np.load(cache_dir / f"{pooling}.npy", mmap_mode="r")
    fit_cube = cache[fit_rows, start : end + 1].astype(np.float32)
    means = fit_cube.mean(axis=0)
    raw_scales = fit_cube.std(axis=0)
    scales = np.where(raw_scales < 1e-6, 1.0, raw_scales)
    fit_x = ((fit_cube - means) / scales).mean(axis=1)
    test_x = selected_features(cache, test_rows, start, end, means, scales)
    head = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=seed,
    )
    head.fit(fit_x, fit_y)
    return head.predict_proba(test_x)[:, 1], head, means, scales


def fit_tfidf(
    seed: int,
    train_text: pd.Series,
    train_y: np.ndarray,
    test_text: pd.Series,
    class_weight: str | None,
) -> np.ndarray:
    representation = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5), min_df=2, max_features=100_000, sublinear_tf=True
                ),
            ),
        ]
    )
    train_x = representation.fit_transform(train_text)
    test_x = representation.transform(test_text)
    head = LogisticRegression(
        C=1.0,
        class_weight=class_weight,
        solver="liblinear",
        max_iter=2000,
        random_state=seed,
    )
    head.fit(train_x, train_y)
    return head.predict_proba(test_x)[:, 1]


def expected_calibration_error(y: np.ndarray, score: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        upper = edges[index + 1] if index < bins - 1 else edges[index + 1] + 1e-12
        mask = (score >= edges[index]) & (score < upper)
        if mask.any():
            result += float(mask.mean() * abs(y[mask].mean() - score[mask].mean()))
    return result


def point_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "auprc": float(average_precision_score(y, score)),
        "auroc": float(roc_auc_score(y, score)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, score >= 0.5)),
        "macro_f1_at_0_5": float(f1_score(y, score >= 0.5, average="macro")),
        "brier_score": float(brier_score_loss(y, score)),
        "log_loss": float(log_loss(y, score)),
        "ece_10_equal_width": expected_calibration_error(y, score),
    }


def paired_bootstrap(
    seed: int, resamples: int, y: np.ndarray, left: np.ndarray, right: np.ndarray
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    auroc_diff = []
    auprc_diff = []
    attempts = 0
    while len(auroc_diff) < resamples:
        attempts += 1
        idx = rng.integers(0, len(y), size=len(y))
        if np.unique(y[idx]).size < 2:
            continue
        auroc_diff.append(roc_auc_score(y[idx], left[idx]) - roc_auc_score(y[idx], right[idx]))
        auprc_diff.append(
            average_precision_score(y[idx], left[idx])
            - average_precision_score(y[idx], right[idx])
        )
    return {
        "auroc_difference_95ci": np.percentile(auroc_diff, [2.5, 97.5]).tolist(),
        "auprc_difference_95ci": np.percentile(auprc_diff, [2.5, 97.5]).tolist(),
        "attempts": attempts,
    }


def screen_task(
    experiment: Path,
    config: dict[str, object],
    frame: pd.DataFrame,
    task: dict[str, object],
    artifact_prefix: str,
    jobs: int,
    resume: bool = False,
    screen_batch_size: int = 24,
) -> dict[str, object]:
    """Screen one frozen binary target against a shared activation cache."""
    cache_dir = cache_directory(experiment, config)
    label_column = str(task["label_column"])
    text_column = str(config["dataset"]["text_column"])
    id_column = str(config["dataset"]["id_column"])
    seed = int(config["head"]["random_state"])
    outputs = int(config["backbone"]["hidden_state_outputs"])
    train_rows = np.flatnonzero(frame["split"].to_numpy() == "train")
    validation_rows = np.flatnonzero(frame["split"].to_numpy() == "validation")
    test_rows = np.flatnonzero(frame["split"].to_numpy() == "test")
    y = frame[label_column].to_numpy(dtype=np.int8)
    selection_train_rows = train_rows
    selection_budget = int(task.get("selection_train_row_budget", len(train_rows)))
    if selection_budget < len(train_rows):
        selection_train_rows, _ = train_test_split(
            train_rows,
            train_size=selection_budget,
            random_state=seed,
            stratify=y[train_rows],
        )
        selection_train_rows = np.sort(selection_train_rows)
    assert len(selection_train_rows) == min(selection_budget, len(train_rows))
    started = time.perf_counter()

    partial_path = experiment / f"{artifact_prefix}candidate_table_partial.csv"
    all_candidates = candidates(outputs)
    completed_rows: list[dict[str, object]] = []
    completed_keys: set[tuple[str, int, int]] = set()
    if resume and partial_path.exists():
        partial = pd.read_csv(partial_path)
        completed_rows = partial.to_dict("records")
        completed_keys = {
            (str(row["pooling"]), int(row["start_layer"]), int(row["end_layer"]))
            for row in completed_rows
        }
        print(f"resuming with {len(completed_keys)}/{len(all_candidates)} candidates complete", flush=True)
    remaining = [item for item in all_candidates if item not in completed_keys]
    for offset in range(0, len(remaining), screen_batch_size):
        batch = remaining[offset : offset + screen_batch_size]
        completed_rows.extend(
            Parallel(n_jobs=jobs, verbose=10)(
                delayed(score_candidate)(
                    cache_dir,
                    seed,
                    pooling,
                    start,
                    end,
                    selection_train_rows,
                    y[selection_train_rows],
                    validation_rows,
                    y[validation_rows],
                    outputs - 1,
                )
                for pooling, start, end in batch
            )
        )
        pd.DataFrame(completed_rows).to_csv(partial_path, index=False)
        print(f"screened {len(completed_rows)}/{len(all_candidates)} candidates", flush=True)
    assert len(completed_rows) == len(all_candidates)
    primary_metric = str(task["primary_selection_metric"]).lower()
    if primary_metric not in {"auprc", "auroc"}:
        raise ValueError(f"Unsupported binary selection metric: {primary_metric}")
    secondary_metric = "auroc" if primary_metric == "auprc" else "auprc"
    table = pd.DataFrame(completed_rows).sort_values(
        [
            f"validation_{primary_metric}",
            f"validation_{secondary_metric}",
            "width",
            "pooling",
            "start_layer",
        ],
        ascending=[False, False, True, True, True],
    )
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table_path = experiment / f"{artifact_prefix}candidate_table.csv"
    table.to_csv(table_path, index=False)
    winner = table.iloc[0].to_dict()

    fit_rows = np.concatenate([train_rows, validation_rows])
    activation_score, head, means, scales = fit_activation(
        cache_dir, seed, winner, fit_rows, y[fit_rows], test_rows
    )
    tfidf_score = fit_tfidf(
        seed,
        frame.loc[fit_rows, text_column].astype(str),
        y[fit_rows],
        frame.loc[test_rows, text_column].astype(str),
        config.get("baseline_class_weight", "balanced"),
    )
    activation_metrics = point_metrics(y[test_rows], activation_score)
    tfidf_metrics = point_metrics(y[test_rows], tfidf_score)
    paired = paired_bootstrap(
        seed,
        int(config["uncertainty"]["resamples"]),
        y[test_rows],
        activation_score,
        tfidf_score,
    )
    paired.update(
        {
            "auprc_difference": activation_metrics["auprc"] - tfidf_metrics["auprc"],
            "auroc_difference": activation_metrics["auroc"] - tfidf_metrics["auroc"],
            "resamples": int(config["uncertainty"]["resamples"]),
            "unit": config["uncertainty"]["unit"],
            "seed": seed,
        }
    )
    probe_path = experiment / f"{artifact_prefix}probe.npz"
    scores_path = experiment / f"{artifact_prefix}test_scores.npz"
    np.savez_compressed(
        probe_path,
        coefficient=head.coef_.astype(np.float64),
        intercept=head.intercept_.astype(np.float64),
        train_means=means.astype(np.float64),
        train_scales=scales.astype(np.float64),
        classes=head.classes_.astype(np.int8),
        pooling=np.array(str(winner["pooling"])),
        start_layer=np.array(int(winner["start_layer"]), dtype=np.int64),
        end_layer=np.array(int(winner["end_layer"]), dtype=np.int64),
    )
    np.savez_compressed(
        scores_path,
        label=y[test_rows],
        activation=activation_score,
        tfidf_word_char=tfidf_score,
        example_id=frame.loc[test_rows, id_column].to_numpy(dtype=str),
    )
    result = {
        "experiment_id": config["experiment_id"],
        "task": task["id"],
        "status": "COMPLETE_VALIDATION_SCREEN_AND_ONE_SHOT_TEST",
        "candidate_count": len(table),
        "selection": {
            "available_train_rows": len(train_rows),
            "screening_train_rows": len(selection_train_rows),
            "screening_train_positive_rows": int(y[selection_train_rows].sum()),
            "validation_rows": len(validation_rows),
            "validation_positive_rows": int(y[validation_rows].sum()),
            "metric": primary_metric.upper(),
            "winner": winner,
        },
        "heldout_test": {
            "rows": len(test_rows),
            "positive_rows": int(y[test_rows].sum()),
            "positive_prevalence": float(y[test_rows].mean()),
            "models": {"activation": activation_metrics, "tfidf_word_char": tfidf_metrics},
            "paired_activation_minus_tfidf": paired,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": {
            "candidate_table_sha256": sha256(table_path),
            "test_scores_sha256": sha256(scores_path),
            "probe_sha256": sha256(probe_path),
        },
        "uncertainty_limitation": config["uncertainty"]["limitation"],
        "research_scope": config["research_scope"],
    }
    (experiment / f"{artifact_prefix}result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def screen(experiment_path: str, jobs: int, resume: bool = False, screen_batch_size: int = 24) -> None:
    experiment, config, _, frame = load_experiment(experiment_path)
    if "tasks" not in config:
        screen_task(
            experiment,
            config,
            frame,
            config["task"],
            "",
            jobs,
            resume=resume,
            screen_batch_size=screen_batch_size,
        )
        return

    task_results = {}
    for task_name, task_config in config["tasks"].items():
        task = dict(task_config)
        task.setdefault("id", task_name)
        task.setdefault("label_column", task_name)
        task_results[task_name] = screen_task(
            experiment,
            config,
            frame,
            task,
            f"{task_name}_",
            jobs,
            resume=resume,
            screen_batch_size=screen_batch_size,
        )
    summary = {
        "experiment_id": config["experiment_id"],
        "status": "COMPLETE_MULTILABEL_VALIDATION_SCREEN_AND_ONE_SHOT_TEST",
        "tasks": {
            name: {
                "winner": result["selection"]["winner"],
                "heldout_test": result["heldout_test"],
            }
            for name, result in task_results.items()
        },
    }
    (experiment / "result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("extract", "screen"))
    parser.add_argument("experiment_dir")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--screen-batch-size", type=int, default=24)
    args = parser.parse_args()
    if args.action == "extract":
        extract(args.experiment_dir, args.batch_size, resume=args.resume)
    else:
        screen(
            args.experiment_dir,
            args.jobs,
            resume=args.resume,
            screen_batch_size=args.screen_batch_size,
        )


if __name__ == "__main__":
    main()
