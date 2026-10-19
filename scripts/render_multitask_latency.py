#!/usr/bin/env python3
"""Render separate scaling and matched-cost panels from frozen timing artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from render_one_forward_many_decisions import (
    BLUE,
    NAVY,
    ORANGE,
    RESULT_DIR,
    ROOT,
    bootstrap_mean_ci,
)


OUTPUT_DIR = ROOT / "research/multitask-activation-benchmark/paper/outputs/figures"
STEM = "latency-panels"
K_VALUES = np.array([1, 2, 4, 8, 16])


def load_panels(result_dir: Path = RESULT_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate panel alignment and derive display values without changing estimates."""
    shared_frames = [
        pd.read_json(result_dir / "outputs" / f"latency_k{k}.jsonl", lines=True)
        for k in K_VALUES
    ]
    repeated_frames = [
        pd.read_json(result_dir / "outputs" / f"repeated_forward_lower_bound_k{k}.jsonl", lines=True)
        for k in K_VALUES
    ]
    assert all(len(frame) == 80 and frame.row_key.is_unique for frame in shared_frames)
    assert all(len(frame) == 10 and frame.row_key.is_unique for frame in repeated_frames)
    assert all(frame.row_key.equals(shared_frames[0].row_key) for frame in shared_frames)
    assert all(
        shared.iloc[:10].row_key.reset_index(drop=True).equals(repeated.row_key)
        for shared, repeated in zip(shared_frames, repeated_frames, strict=True)
    )
    summary = pd.read_csv(result_dir / "SCALING-SUMMARY.csv").sort_values("k").reset_index(drop=True)
    assert np.array_equal(summary.k, K_VALUES) and summary.rows.eq(80).all()
    full_means = np.stack([frame.total_ms.to_numpy(dtype=float) for frame in shared_frames]).mean(axis=1)
    assert np.isfinite(full_means).all() and (full_means > 0).all()
    assert np.allclose(summary.mean_total_ms, full_means)
    assert np.allclose(summary.one_forward_mean_ratio_vs_k1, full_means / full_means[0])
    scaling = pd.DataFrame({
        "k": K_VALUES,
        "mean_ms": full_means,
        "increase_pct": 100 * (summary.one_forward_mean_ratio_vs_k1 - 1),
        "ci_low_pct": 100 * (summary.one_forward_mean_ratio_ci_low - 1),
        "ci_high_pct": 100 * (summary.one_forward_mean_ratio_ci_high - 1),
    })
    assert scaling.ci_low_pct.le(scaling.increase_pct).all()
    assert scaling.increase_pct.le(scaling.ci_high_pct).all()
    matched_shared = [frame.iloc[:10].total_ms.to_numpy(dtype=float) for frame in shared_frames]
    matched_repeated = [frame.total_ms.to_numpy(dtype=float) for frame in repeated_frames]
    shared_ci = np.array([
        bootstrap_mean_ci(values, seed=20261018 + int(k))
        for k, values in zip(K_VALUES, matched_shared, strict=True)
    ])
    repeated_ci = np.array([
        bootstrap_mean_ci(values, seed=20261118 + int(k))
        for k, values in zip(K_VALUES, matched_repeated, strict=True)
    ])
    matched = pd.DataFrame({
        "k": K_VALUES,
        "shared_ms": np.stack(matched_shared).mean(axis=1),
        "shared_ci_low_ms": shared_ci[:, 0],
        "shared_ci_high_ms": shared_ci[:, 1],
        "repeated_ms": np.stack(matched_repeated).mean(axis=1),
        "repeated_ci_low_ms": repeated_ci[:, 0],
        "repeated_ci_high_ms": repeated_ci[:, 1],
    })
    assert np.allclose(matched.repeated_ms, summary.repeated_forward_mean_ms)
    assert np.isclose(matched.repeated_ms.iloc[-1] / matched.shared_ms.iloc[-1], 15.258962751202098)
    return scaling, matched


def make_figure(scaling: pd.DataFrame, matched: pd.DataFrame) -> plt.Figure:
    """Give each input panel its own equally sized, explicitly labelled linear axes."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.4, 4.8))
    left.errorbar(
        scaling.k, scaling.increase_pct,
        yerr=np.stack([scaling.increase_pct - scaling.ci_low_pct, scaling.ci_high_pct - scaling.increase_pct]),
        color=BLUE, marker="o", linewidth=2, markersize=5, capsize=3,
    )
    left.set_title("(a) Shared-path overhead · n=80", loc="left", fontsize=12, pad=12, color=NAVY)
    left.set_ylabel("Mean latency increase relative to K=1 (%)", fontsize=10)
    left.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    left.set_ylim(-0.4, 10)
    left.set_yticks([0, 2, 4, 6, 8, 10])
    left.annotate(
        "+6.9% at K=16\n68.63 → 73.34 ms",
        xy=(16, scaling.increase_pct.iloc[-1]), xytext=(6.0, 8.6),
        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1},
        fontsize=10, color=BLUE,
    )
    right.errorbar(
        matched.k, matched.shared_ms,
        yerr=np.stack([matched.shared_ms - matched.shared_ci_low_ms, matched.shared_ci_high_ms - matched.shared_ms]),
        color=BLUE, marker="o", linewidth=2, markersize=5, capsize=3, label="One forward + K heads",
    )
    right.errorbar(
        matched.k, matched.repeated_ms,
        yerr=np.stack([matched.repeated_ms - matched.repeated_ci_low_ms, matched.repeated_ci_high_ms - matched.repeated_ms]),
        color=ORANGE, marker="s", linestyle="--", linewidth=2, markersize=5, capsize=3,
        label="K repeated forwards (lower bound)",
    )
    right.set_title("(b) Matched inference cost · n=10", loc="left", fontsize=12, pad=12, color=NAVY)
    right.set_ylabel("Mean latency per input (ms)", fontsize=10)
    right.set_ylim(0, 1120)
    right.set_yticks([0, 250, 500, 750, 1000])
    right.legend(loc="upper left", frameon=False, fontsize=9)
    right.annotate(
        "956.71 ms\n15.26× shared",
        xy=(16, matched.repeated_ms.iloc[-1]), xytext=(7.1, 800),
        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1},
        fontsize=10, color=ORANGE,
    )
    right.annotate(
        "62.70 ms",
        xy=(16, matched.shared_ms.iloc[-1]), xytext=(10.2, 200),
        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1},
        fontsize=10, color=BLUE,
    )
    for axis in (left, right):
        axis.set_xlim(0, 17)
        axis.set_xticks(K_VALUES)
        axis.set_xlabel("Number of heads (K)", fontsize=10)
        axis.tick_params(labelsize=9)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.86, wspace=0.38)
    fig.text(
        0.02, 0.025,
        "Qwen2.5-0.5B · batch 1 · six CPU threads · 95% row-bootstrap intervals\n"
        "(a) Paired ratios on 80 inputs. (b) Mean times on the same 10 inputs; repeated forwards omit prompting, output scoring and generation.",
        fontsize=8, color="#66717E",
    )
    return fig


def render(output_dir: Path = OUTPUT_DIR) -> None:
    """Write deterministic SVG and 300-DPI PNG without overwriting companion figures."""
    scaling, matched = load_panels()
    figure = make_figure(scaling, matched)
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"svg.hashsalt": "intentprobe-dual-latency-2026-10-19"}):
        figure.savefig(output_dir / f"{STEM}.svg", metadata={"Date": "2026-10-19"})
        figure.savefig(output_dir / f"{STEM}.png", dpi=300, metadata={"Software": "MCPware IntentProbe"})
    plt.close(figure)


def main() -> None:
    """Render to the paper's figure directory or an explicit QA directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    render(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
