#!/usr/bin/env python3
"""Run one frozen local-judge arm for the sentinel alignment panel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, disable_caching
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parents[1]
PROJECT = BENCHMARK.parents[1]
CONFIG = HERE / "config.json"
MANIFEST = HERE / "evaluation_manifest.json"
PREDICTIONS = HERE / "predictions"
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
    """Load one local module without installing the project."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and verify the frozen contract and evaluation manifest."""
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["config_sha256"] != sha256(CONFIG):
        raise RuntimeError("Frozen evaluation manifest does not match config.json")
    if manifest["experiment_id"] != config["experiment_id"]:
        raise RuntimeError("Experiment ID mismatch")
    return config, manifest


def load_task_frame(task: str, manifest: dict[str, Any]) -> pd.DataFrame:
    """Load one selected evaluation frame without changing source data."""
    selected_ids = np.asarray(manifest["tasks"][task]["ids"], dtype=str)
    if task == "sentiment":
        source = pd.read_parquet(SST2)
        source = source.loc[source["split"].eq("test"), ["example_id", "text", "label", "group_id"]]
        frame = source.rename(columns={"example_id": "id", "label": "y"})
    elif task == "toxicity":
        source = pd.read_parquet(TOXICCHAT)
        source = source.loc[source["split"].eq("test"), ["conv_id", "text", "toxicity", "group_id"]]
        frame = source.rename(columns={"conv_id": "id", "toxicity": "y"})
    elif task == "authorization_to_execute":
        source = load_module("sentinel_delegation_eval", DELEGATION_RUN).build_external_frame()
        frame = source.loc[:, ["id", "rendered", "y", "group_id"]].rename(
            columns={"rendered": "text"}
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    frame["id"] = frame["id"].astype(str)
    frame = frame.loc[frame["id"].isin(selected_ids)].sort_values("id").reset_index(drop=True)
    if not np.array_equal(frame["id"].to_numpy(dtype=str), np.sort(selected_ids)):
        raise RuntimeError(f"Selected IDs do not match source frame for {task}")
    if frame["id"].duplicated().any() or frame["y"].nunique() != 2:
        raise RuntimeError(f"Invalid binary evaluation frame for {task}")
    expected = manifest["tasks"][task]["summary"]
    observed_counts = {
        str(key): int(value) for key, value in frame["y"].value_counts().sort_index().items()
    }
    if len(frame) != int(expected["rows"]) or observed_counts != expected["label_counts"]:
        raise RuntimeError(f"Frozen sample integrity mismatch for {task}")
    return frame


def render_prompts(
    texts: np.ndarray, *, system_prompt: str, model_alias: str
) -> np.ndarray:
    """Render the two pinned local chat templates with vectorized string operations."""
    values = np.asarray(texts, dtype=str)
    if model_alias == "cost_anchor":
        prefix = "<|startoftext|><|im_start|>system\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n"
    elif model_alias == "quality_anchor":
        prefix = "<|im_start|>system\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    else:
        raise ValueError(f"Unknown model alias: {model_alias}")
    user_prefix = "<|im_end|>\n<|im_start|>user\n"
    rendered = np.char.add(prefix, system_prompt)
    rendered = np.char.add(rendered, user_prefix)
    rendered = np.char.add(rendered, values)
    return np.char.add(rendered, suffix)


def truncate_contents(tokenizer: Any, texts: np.ndarray, max_content_tokens: int) -> np.ndarray:
    """Apply the frozen cap to content without truncating the judge instruction or answer."""
    encoded = tokenizer(
        np.asarray(texts, dtype=str).tolist(),
        add_special_tokens=False,
        truncation=True,
        max_length=max_content_tokens,
    )
    return np.asarray(
        tokenizer.batch_decode(
            encoded["input_ids"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ),
        dtype=str,
    )


def label_sequence_scores(
    *, model: Any, tokenizer: Any, prompts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score the exact YES and NO continuations in two vectorized forwards."""
    tokenizer.padding_side = "right"
    prompt_tokens = tokenizer(
        prompts.tolist(),
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    prompt_lengths = prompt_tokens["attention_mask"].sum(dim=1).to(dtype=torch.long)
    label_scores: list[np.ndarray] = []
    forward_times: list[float] = []
    for label in ("YES", "NO"):
        full_texts = np.char.add(prompts, label)
        tokens = tokenizer(
            full_texts.tolist(),
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        full_lengths = tokens["attention_mask"].sum(dim=1).to(dtype=torch.long)
        started = time.perf_counter()
        with torch.inference_mode():
            logits = model(**tokens).logits
        forward_times.append(time.perf_counter() - started)
        target_ids = tokens["input_ids"][:, 1:]
        token_log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        selected = token_log_probs.gather(dim=-1, index=target_ids[..., None]).squeeze(-1)
        original_positions = torch.arange(1, tokens["input_ids"].shape[1])[None, :]
        label_mask = (original_positions >= prompt_lengths[:, None]) & (
            original_positions < full_lengths[:, None]
        )
        if not bool(label_mask.any(dim=1).all()):
            raise RuntimeError("A YES/NO continuation was fully truncated")
        sequence_mean = (selected * label_mask).sum(dim=1) / label_mask.sum(dim=1)
        label_scores.append(sequence_mean.cpu().numpy().astype(np.float64))
    return label_scores[0], label_scores[1], np.asarray(forward_times, dtype=np.float64)


def normalized_yes_probability(yes_score: np.ndarray, no_score: np.ndarray) -> np.ndarray:
    """Normalize two sequence scores into a stable binary probability."""
    matrix = np.column_stack((yes_score, no_score))
    shifted = matrix - matrix.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities[:, 0]


def parse_generated_labels(outputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the frozen strict first-token YES/NO parser."""
    series = pd.Series(np.asarray(outputs, dtype=str), dtype="string")
    first = series.str.strip().str.upper().str.split().str[0].fillna("")
    valid = first.isin(["YES", "NO"]).to_numpy(dtype=bool)
    predictions = np.where(first.eq("YES").to_numpy(dtype=bool), 1, 0).astype(np.int8)
    return predictions, ~valid


def load_model(model_config: dict[str, Any]) -> tuple[Any, Any, float]:
    """Load one pinned model and tokenizer from the local Hub cache."""
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has no padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    loader = model_config["loader"]
    model_class = (
        AutoModelForImageTextToText if loader == "image_text_to_text" else AutoModelForCausalLM
    )
    model = model_class.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        local_files_only=True,
        trust_remote_code=True,
        dtype="auto",
    ).eval()
    return tokenizer, model, time.perf_counter() - started


def run_choice_logprob(
    *, dataset: Dataset, model: Any, tokenizer: Any, system_prompt: str, model_alias: str,
    max_length: int, batch_size: int
) -> Dataset:
    """Run batched exact-label sequence scoring."""
    def score_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        bounded_texts = truncate_contents(
            tokenizer, np.asarray(batch["text"], dtype=str), max_length
        )
        prompts = render_prompts(
            bounded_texts,
            system_prompt=system_prompt,
            model_alias=model_alias,
        )
        started = time.perf_counter()
        yes_score, no_score, forward_times = label_sequence_scores(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
        )
        elapsed = time.perf_counter() - started
        probability = normalized_yes_probability(yes_score, no_score)
        count = len(probability)
        return {
            "score": probability.tolist(),
            "prediction": (probability >= 0.5).astype(np.int8).tolist(),
            "invalid": np.zeros(count, dtype=bool).tolist(),
            "raw_output": np.full(count, "", dtype=str).tolist(),
            "yes_sequence_logprob": yes_score.tolist(),
            "no_sequence_logprob": no_score.tolist(),
            "elapsed_seconds_per_example": np.full(count, elapsed / count).tolist(),
            "model_forward_seconds_per_example": np.full(
                count, forward_times.sum() / count
            ).tolist(),
        }

    return dataset.map(
        score_batch,
        batched=True,
        batch_size=batch_size,
        load_from_cache_file=False,
        keep_in_memory=True,
        desc="exact YES/NO sequence scoring",
    )


def run_generate(
    *, dataset: Dataset, model: Any, tokenizer: Any, system_prompt: str, model_alias: str,
    max_length: int, max_new_tokens: int, batch_size: int
) -> Dataset:
    """Run deterministic generation and the strict parser."""
    def generate_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        bounded_texts = truncate_contents(
            tokenizer, np.asarray(batch["text"], dtype=str), max_length
        )
        prompts = render_prompts(
            bounded_texts,
            system_prompt=system_prompt,
            model_alias=model_alias,
        )
        tokenizer.padding_side = "left"
        tokens = tokenizer(
            prompts.tolist(),
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.perf_counter() - started
        generated = np.asarray(
            tokenizer.batch_decode(
                output[:, tokens["input_ids"].shape[1] :], skip_special_tokens=True
            ),
            dtype=str,
        )
        predictions, invalid = parse_generated_labels(generated)
        count = len(generated)
        return {
            "score": predictions.astype(np.float64).tolist(),
            "prediction": predictions.tolist(),
            "invalid": invalid.tolist(),
            "raw_output": generated.tolist(),
            "yes_sequence_logprob": np.full(count, np.nan).tolist(),
            "no_sequence_logprob": np.full(count, np.nan).tolist(),
            "elapsed_seconds_per_example": np.full(count, elapsed / count).tolist(),
            "model_forward_seconds_per_example": np.full(count, elapsed / count).tolist(),
        }

    return dataset.map(
        generate_batch,
        batched=True,
        batch_size=batch_size,
        load_from_cache_file=False,
        keep_in_memory=True,
        desc="deterministic YES/NO generation",
    )


def package_versions() -> dict[str, str]:
    """Return the packages that affect this benchmark."""
    names = ("datasets", "numpy", "pandas", "scikit-learn", "torch", "transformers")
    return {name: importlib.metadata.version(name) for name in names}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("quality_anchor", "cost_anchor"), required=True)
    parser.add_argument(
        "--task", choices=("authorization_to_execute", "toxicity", "sentiment"), required=True
    )
    parser.add_argument("--prompt", choices=("zero_shot", "rubric"), required=True)
    parser.add_argument("--method", choices=("choice_logprob", "generate"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def main() -> None:
    """Execute one immutable model-task-prompt-method arm."""
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    config, manifest = load_contract()
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.set_num_threads(max(1, torch.get_num_threads()))
    frame = load_task_frame(args.task, manifest)
    model_config = config["models"][args.model]
    task_config = config["tasks"][args.task]
    system_prompt = task_config["prompts"][args.prompt]

    output_path = PREDICTIONS / f"{args.model}__{args.task}__{args.prompt}__{args.method}.jsonl"
    metadata_path = output_path.with_suffix(".metadata.json")
    if output_path.exists() or metadata_path.exists():
        raise SystemExit(f"Refusing to overwrite existing result: {output_path}")

    disable_caching()
    dataset = Dataset.from_pandas(frame, preserve_index=False)
    tokenizer, model, load_seconds = load_model(model_config)
    started = time.perf_counter()
    if args.method == "choice_logprob":
        scored = run_choice_logprob(
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            model_alias=args.model,
            max_length=int(task_config["max_length"]),
            batch_size=args.batch_size,
        )
    else:
        scored = run_generate(
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            model_alias=args.model,
            max_length=int(task_config["max_length"]),
            max_new_tokens=int(config["judge_scoring"]["max_new_tokens"]),
            batch_size=args.batch_size,
        )
    scoring_seconds = time.perf_counter() - started
    result = scored.to_pandas()
    assert len(result) == len(frame)
    assert result["id"].astype(str).is_unique
    assert np.isfinite(result["score"].to_numpy(dtype=np.float64)).all()
    assert result["prediction"].isin([0, 1]).all()
    result = result.drop(columns=["text"])

    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    result.to_json(output_path, orient="records", lines=True, force_ascii=False)
    metadata = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "model_alias": args.model,
        "model": model_config,
        "task": args.task,
        "prompt": args.prompt,
        "prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "method": args.method,
        "rows": int(len(result)),
        "positive_rows": int(result["y"].sum()),
        "invalid_outputs": int(result["invalid"].sum()),
        "batch_size": args.batch_size,
        "load_seconds": load_seconds,
        "scoring_seconds": scoring_seconds,
        "seconds_per_example": scoring_seconds / len(result),
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "config_sha256": sha256(CONFIG),
        "evaluation_manifest_sha256": sha256(MANIFEST),
        "predictions_sha256": sha256(output_path),
        "python": platform.python_version(),
        "packages": package_versions(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
