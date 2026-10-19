#!/usr/bin/env python3
"""Summarize scaling, native quality, calibration, and score-shift diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.json"
REGISTRY = HERE / "HEAD_REGISTRY.json"
WORKLOAD = HERE / "WORKLOAD_MANIFEST.json"
OUTPUTS = HERE / "outputs"
RESULT = HERE / "result.json"
SCALING_CSV = HERE / "SCALING-SUMMARY.csv"
QUALITY_CSV = HERE / "NATIVE-HEAD-QUALITY.csv"
OOD_CSV = HERE / "OOD-SCORE-SHIFTS.csv"
REPLAY_CSV = HERE / "RUNTIME-REPLAY-CHECK.csv"
REPORT = HERE / "RESULT_SUMMARY.md"
PROJECT = HERE.parents[3]


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def interval(values: np.ndarray) -> list[float]:
    """Return a percentile 95% interval."""
    return [float(value) for value in np.quantile(values, (0.025, 0.975))]


def paired_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    statistic: Callable[..., np.ndarray],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Return a paired point ratio and bootstrap interval."""
    if len(numerator) != len(denominator):
        raise RuntimeError("Paired ratio inputs have different lengths")
    indices = np.random.default_rng(seed).integers(
        0, len(numerator), size=(resamples, len(numerator)), dtype=np.int32
    )
    sampled_numerator = statistic(numerator[indices], axis=1)
    sampled_denominator = statistic(denominator[indices], axis=1)
    ratios = sampled_numerator / sampled_denominator
    return {
        "ratio": float(statistic(numerator) / statistic(denominator)),
        "ci_95": interval(ratios),
    }


def align_frames(frames: dict[int, pd.DataFrame], expected_rows: int) -> None:
    """Validate identical paired rows, token counts, and ordering across K."""
    reference = frames[min(frames)][["row_key", "source", "example_id", "token_count"]]
    if len(reference) != expected_rows or reference["row_key"].duplicated().any():
        raise RuntimeError("Invalid K=1 timing panel")
    for k, frame in frames.items():
        observed = frame[["row_key", "source", "example_id", "token_count"]]
        if len(frame) != expected_rows or not observed.equals(reference):
            raise RuntimeError(f"Timing panel changed at K={k}")


def load_timing(
    config: dict[str, Any], workload: dict[str, Any]
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame], dict[int, dict[str, Any]]]:
    """Load all required complete timing cells and their metadata."""
    timings: dict[int, pd.DataFrame] = {}
    repeated: dict[int, pd.DataFrame] = {}
    metadata: dict[int, dict[str, Any]] = {}
    for k in config["k_values"]:
        timings[k] = pd.read_json(OUTPUTS / f"latency_k{k}.jsonl", lines=True)
        repeated[k] = pd.read_json(
            OUTPUTS / f"repeated_forward_lower_bound_k{k}.jsonl", lines=True
        )
        metadata[k] = load_json(OUTPUTS / f"latency_k{k}.metadata.json")
    align_frames(timings, int(workload["latency_rows"]))
    align_frames(repeated, int(config["repeated_forward_lower_bound_rows"]))
    return timings, repeated, metadata


def summarize_scaling(
    config: dict[str, Any], workload: dict[str, Any]
) -> list[dict[str, Any]]:
    """Summarize paired timing and process-memory scaling."""
    timings, repeated, metadata = load_timing(config, workload)
    resamples = int(config["uncertainty"]["resamples"])
    seed = int(config["uncertainty"]["seed"])
    baseline = timings[1]["total_ms"].to_numpy(dtype=np.float64)
    repeated_baseline = repeated[1]["total_ms"].to_numpy(dtype=np.float64)
    baseline_rss = float(metadata[1]["peak_rss_kib"]) / 1024.0
    rows: list[dict[str, Any]] = []
    for k in config["k_values"]:
        frame = timings[k]
        lower = repeated[k]
        total = frame["total_ms"].to_numpy(dtype=np.float64)
        repeated_total = lower["total_ms"].to_numpy(dtype=np.float64)
        matched = frame.iloc[: len(lower)]
        one_forward_matched = matched["total_ms"].to_numpy(dtype=np.float64)
        mean_ratio = paired_ratio(total, baseline, np.mean, resamples, seed + k)
        median_ratio = paired_ratio(total, baseline, np.median, resamples, seed + 100 + k)
        repeated_ratio = paired_ratio(
            repeated_total,
            repeated_baseline,
            np.mean,
            resamples,
            seed + 200 + k,
        )
        advantage = paired_ratio(
            repeated_total,
            one_forward_matched,
            np.mean,
            resamples,
            seed + 300 + k,
        )
        peak_rss_mib = float(metadata[k]["peak_rss_kib"]) / 1024.0
        rows.append(
            {
                "k": k,
                "rows": len(frame),
                "mean_tokenization_ms": float(frame["tokenization_ms"].mean()),
                "mean_forward_ms": float(frame["forward_ms"].mean()),
                "mean_heads_ms": float(frame["heads_ms"].mean()),
                "mean_total_ms": float(frame["total_ms"].mean()),
                "median_total_ms": float(frame["total_ms"].median()),
                "p95_total_ms": float(frame["total_ms"].quantile(0.95)),
                "one_forward_mean_ratio_vs_k1": mean_ratio["ratio"],
                "one_forward_mean_ratio_ci_low": mean_ratio["ci_95"][0],
                "one_forward_mean_ratio_ci_high": mean_ratio["ci_95"][1],
                "one_forward_median_ratio_vs_k1": median_ratio["ratio"],
                "one_forward_median_ratio_ci_low": median_ratio["ci_95"][0],
                "one_forward_median_ratio_ci_high": median_ratio["ci_95"][1],
                "repeated_forward_mean_ms": float(lower["total_ms"].mean()),
                "repeated_forward_ratio_vs_k1": repeated_ratio["ratio"],
                "repeated_forward_ratio_ci_low": repeated_ratio["ci_95"][0],
                "repeated_forward_ratio_ci_high": repeated_ratio["ci_95"][1],
                "repeated_over_one_forward_ratio": advantage["ratio"],
                "repeated_over_one_forward_ci_low": advantage["ci_95"][0],
                "repeated_over_one_forward_ci_high": advantage["ci_95"][1],
                "peak_rss_mib": peak_rss_mib,
                "peak_rss_delta_vs_k1_mib": peak_rss_mib - baseline_rss,
                "cumulative_probe_bytes": int(
                    sum(
                        head["probe_bytes"]
                        for head in load_json(REGISTRY)["heads"][:k]
                    )
                ),
            }
        )
    return rows


def native_quality(registry: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten frozen heldout discrimination and calibration metrics."""
    rows: list[dict[str, Any]] = []
    for head in registry["heads"]:
        rows.append(
            {
                "order": head["order"],
                "head": head["name"],
                "native_source": head["native_source"],
                "pooling": head["pooling"],
                "start_layer": head["start_layer"],
                "end_layer": head["end_layer"],
                **head["native_heldout"],
            }
        )
    return rows


def native_archive(head: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load IDs, labels, and scores from one frozen native heldout archive."""
    with np.load(PROJECT / head["scores_path"], allow_pickle=False) as archive:
        id_key = "conv_id" if "conv_id" in archive.files else "example_id"
        return (
            np.asarray(archive[id_key]).astype(str),
            np.asarray(archive["label"], dtype=np.int8),
            np.asarray(archive["activation"], dtype=np.float64),
        )


def summarize_score_shifts(
    registry: dict[str, Any], workload: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Describe cross-domain score shifts and replay consistency without fake labels."""
    frame = pd.read_json(OUTPUTS / "ood_scores_k16.jsonl", lines=True)
    expected_rows = int(workload["ood_rows"])
    if len(frame) != expected_rows or frame["row_key"].duplicated().any():
        raise RuntimeError("Invalid cross-domain score artifact")
    shifts: list[dict[str, Any]] = []
    replay: list[dict[str, Any]] = []
    for head in registry["heads"]:
        head_name = head["name"]
        native_ids, _, native_scores = native_archive(head)
        for source_name in workload["sources"]:
            source_scores = frame.loc[
                frame["source"].eq(source_name), head_name
            ].to_numpy(dtype=np.float64)
            if len(source_scores) != int(workload["sources"][source_name]["ood_ids"].__len__()):
                raise RuntimeError(f"Incomplete OOD source {source_name} for {head_name}")
            ks = ks_2samp(source_scores, native_scores, method="auto")
            shifts.append(
                {
                    "head": head_name,
                    "native_source": head["native_source"],
                    "scored_source": source_name,
                    "rows": len(source_scores),
                    "mean_score": float(source_scores.mean()),
                    "median_score": float(np.median(source_scores)),
                    "p95_score": float(np.quantile(source_scores, 0.95)),
                    "rate_at_or_above_0_5": float(np.mean(source_scores >= 0.5)),
                    "native_full_mean_score": float(native_scores.mean()),
                    "mean_shift_vs_native_full": float(
                        source_scores.mean() - native_scores.mean()
                    ),
                    "ks_distance_vs_native_full": float(ks.statistic),
                }
            )

        native_runtime = frame.loc[
            frame["source"].eq(head["native_source"]),
            ["example_id", head_name],
        ].copy()
        archive_frame = pd.DataFrame(
            {"example_id": native_ids, "archived_score": native_scores}
        )
        matched = native_runtime.merge(
            archive_frame, on="example_id", how="inner", validate="one_to_one"
        )
        if len(matched) != len(native_runtime):
            raise RuntimeError(f"Could not replay all native IDs for {head_name}")
        difference = (
            matched[head_name].to_numpy(dtype=np.float64)
            - matched["archived_score"].to_numpy(dtype=np.float64)
        )
        replay.append(
            {
                "head": head_name,
                "native_source": head["native_source"],
                "rows": len(matched),
                "mean_absolute_difference": float(np.mean(np.abs(difference))),
                "max_absolute_difference": float(np.max(np.abs(difference))),
                "pearson_correlation": float(
                    np.corrcoef(
                        matched[head_name].to_numpy(dtype=np.float64),
                        matched["archived_score"].to_numpy(dtype=np.float64),
                    )[0, 1]
                ),
            }
        )
    return shifts, replay


def markdown_table(frame: pd.DataFrame, columns: list[str], decimals: int = 3) -> str:
    """Render a compact Markdown table without optional tabulate dependency."""
    selected = frame[columns].copy()
    for column in selected.select_dtypes(include=["number"]).columns:
        selected[column] = selected[column].map(lambda value: f"{value:.{decimals}f}")
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = "\n".join(
        "| " + " | ".join(map(str, values)) + " |"
        for values in selected.itertuples(index=False, name=None)
    )
    return "\n".join((header, rule, body))


def write_report(
    scaling: pd.DataFrame,
    quality: pd.DataFrame,
    shifts: pd.DataFrame,
    replay: pd.DataFrame,
    result: dict[str, Any],
) -> None:
    """Write a concise evidence-backed result narrative."""
    k16 = scaling.loc[scaling["k"].eq(16)].iloc[0]
    best_quality = quality.sort_values("auprc", ascending=False).iloc[0]
    worst_quality = quality.sort_values("auprc", ascending=True).iloc[0]
    largest_shift = shifts.iloc[shifts["mean_shift_vs_native_full"].abs().argmax()]
    content = f"""# One-forward multi-head result

Status: **COMPLETE**. All five frozen K cells, the 16-head score-shift panel, and all 16 native heldout quality/calibration records passed alignment checks.

## Main result

On this CPU run, one Qwen2.5-0.5B forward plus 16 independently trained activation heads took {k16['mean_total_ms']:.2f} ms per example on average. That was {k16['one_forward_mean_ratio_vs_k1']:.3f}× the one-head time (paired-bootstrap 95% CI {k16['one_forward_mean_ratio_ci_low']:.3f}–{k16['one_forward_mean_ratio_ci_high']:.3f}). Sixteen repeated base-model forwards took {k16['repeated_forward_mean_ms']:.2f} ms and were {k16['repeated_over_one_forward_ratio']:.2f}× slower than the one-forward/16-head path on the matched ten prompts (95% CI {k16['repeated_over_one_forward_ci_low']:.2f}–{k16['repeated_over_one_forward_ci_high']:.2f}). This repeated-forward comparator omits task prompts, LM-head scoring, and generation, so it is a lower bound for 16 separate LLM-judge calls.

Peak process RSS was {k16['peak_rss_mib']:.1f} MiB at K=16, a {k16['peak_rss_delta_vs_k1_mib']:+.1f} MiB change from K=1. The saved probe artifacts total {int(k16['cumulative_probe_bytes']):,} bytes at K=16.

{markdown_table(scaling, ['k', 'mean_total_ms', 'mean_heads_ms', 'one_forward_mean_ratio_vs_k1', 'repeated_forward_mean_ms', 'repeated_over_one_forward_ratio', 'peak_rss_mib'])}

## Quality is task-specific

The 16 heads retain their own native heldout labels. AUPRC ranges from {worst_quality['auprc']:.3f} ({worst_quality['head']}, prevalence {worst_quality['positive_prevalence']:.3f}) to {best_quality['auprc']:.3f} ({best_quality['head']}, prevalence {best_quality['positive_prevalence']:.3f}). The systems result therefore does not imply that every head is equally useful; it shows that many validated readouts can share one backbone pass.

{markdown_table(quality, ['head', 'native_source', 'pooling', 'start_layer', 'end_layer', 'rows', 'positive_prevalence', 'auprc', 'auroc', 'balanced_accuracy', 'brier', 'ece_10_bin'])}

## Cross-domain diagnostic

The 320-row panel is deliberately unlabelled outside each head's native task. It measures score movement, not cross-domain error. The largest absolute mean shift was {largest_shift['mean_shift_vs_native_full']:+.3f} for `{largest_shift['head']}` on `{largest_shift['scored_source']}` relative to that head's full native heldout score distribution (KS distance {largest_shift['ks_distance_vs_native_full']:.3f}). Treat these shifts as monitoring/calibration evidence, not accuracy claims.

Runtime replay against the saved float16-cache evaluation scores had a worst mean absolute difference of {replay['mean_absolute_difference'].max():.6f}, a worst maximum absolute difference of {replay['max_absolute_difference'].max():.6f}, and a minimum Pearson correlation of {replay['pearson_correlation'].min():.6f} across the 16 heads. This quantifies the sensitivity introduced by using live full-precision hidden states rather than cached float16 activations.

## Interpretation boundary

- Supported: one base-model pass can serve 16 real, independently trained classification heads with nearly flat latency and memory in this fixed CPU environment.
- Supported: direct same-weight judging is not uniformly equivalent to activation readouts; the separate six-cell sentinel found strong task/model interactions.
- Not supported: universal probe superiority, production calibration on every deployment domain, or accuracy for the unlabelled cross-domain panel.
- Comparator boundary: repeated base forwards are a conservative systems lower bound, not a full implementation of 16 prompted LLM judges.

Machine-readable detail is in `result.json`; row-level artifacts contain IDs, timings, and scores but no source text.
"""
    REPORT.write_text(content, encoding="utf-8")


def main() -> None:
    """Validate all artifacts and write final summaries."""
    if RESULT.exists():
        raise SystemExit("Refusing to overwrite completed result.json")
    config = load_json(CONFIG)
    registry = load_json(REGISTRY)
    workload = load_json(WORKLOAD)
    scaling_rows = summarize_scaling(config, workload)
    quality_rows = native_quality(registry)
    shift_rows, replay_rows = summarize_score_shifts(registry, workload)
    scaling = pd.DataFrame(scaling_rows)
    quality = pd.DataFrame(quality_rows)
    shifts = pd.DataFrame(shift_rows)
    replay = pd.DataFrame(replay_rows)
    scaling.to_csv(SCALING_CSV, index=False)
    quality.to_csv(QUALITY_CSV, index=False)
    shifts.to_csv(OOD_CSV, index=False)
    replay.to_csv(REPLAY_CSV, index=False)
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "COMPLETE",
        "k_values": config["k_values"],
        "head_count": registry["head_count"],
        "latency_rows": workload["latency_rows"],
        "ood_rows": workload["ood_rows"],
        "bootstrap_resamples": config["uncertainty"]["resamples"],
        "scaling": scaling_rows,
        "native_quality": quality_rows,
        "ood_score_shifts": shift_rows,
        "runtime_replay": replay_rows,
        "interpretation": {
            "ood": config["ood"]["interpretation"],
            "repeated_forward": config["latency"]["repeated_forward_baseline"],
        },
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(scaling, quality, shifts, replay, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "k_values": result["k_values"],
                "head_count": result["head_count"],
                "latency_rows": result["latency_rows"],
                "ood_rows": result["ood_rows"],
                "k16": scaling_rows[-1],
                "runtime_replay_worst_mae": float(
                    replay["mean_absolute_difference"].max()
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
