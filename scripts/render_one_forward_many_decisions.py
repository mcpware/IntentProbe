#!/usr/bin/env python3
"""Render the public IntentProbe architecture and latency figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = (
    ROOT
    / "research"
    / "multitask-activation-benchmark"
    / "experiments"
    / "ONE-FORWARD-MULTIHEAD-EXP001"
)
OUTPUTS = RESULT_DIR / "outputs"
OUTPUT_DIR = ROOT / "docs"

NAVY = "#17324D"
BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
LIGHT_BLUE = "#EAF2F8"
LIGHT_ORANGE = "#FFF1E6"
LIGHT_GREEN = "#EAF5EC"
LIGHT_GREY = "#F3F5F7"
MID_GREY = "#66717E"
WHITE = "#FFFFFF"


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
        "font.size": 11,
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.grid": True,
        "grid.alpha": 0.20,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.18,
        "svg.fonttype": "none",
        "svg.hashsalt": "intentprobe-architecture-presentation-v0.1.2",
    }
)


def save_public_figure(fig: plt.Figure, stem: str) -> None:
    """Save deterministic public PNG/SVG artifacts."""
    svg_path = OUTPUT_DIR / f"{stem}.svg"
    png_path = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(svg_path, metadata={"Date": "2026-10-18"})
    fig.savefig(png_path, metadata={"Software": "MCPware IntentProbe"})
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )


def add_box(
    ax: Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str,
    title_size: float = 11.5,
    body_size: float = 8.8,
) -> None:
    """Add one rounded architecture box."""
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.010,rounding_size=0.015",
        linewidth=1.6,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.64,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=NAVY,
    )
    ax.text(
        x + width / 2,
        y + height * 0.27,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color="#313B45",
        linespacing=1.18,
    )


def add_arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MID_GREY,
    style: str = "-",
    width: float = 1.8,
) -> None:
    """Add one flow arrow."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=width,
            linestyle=style,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def render_architecture() -> None:
    """Render a literal one-pass activation bus with task-specific fan-out."""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.957,
        "One local model pass. Many task-specific probes.",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        color=NAVY,
    )
    ax.text(
        0.5,
        0.913,
        "A reusable local semantic sensor — every task head can improve independently",
        ha="center",
        va="center",
        fontsize=13.5,
        color=MID_GREY,
    )

    ax.text(
        0.035,
        0.825,
        "OFFLINE  ·  TRAIN AND ADMIT EACH HEAD ON ITS OWN EVIDENCE",
        fontsize=10.5,
        fontweight="bold",
        color=ORANGE,
    )
    add_box(
        ax,
        x=0.035,
        y=0.700,
        width=0.205,
        height=0.095,
        title="Labelled examples",
        body="One declared task boundary",
        facecolor=LIGHT_ORANGE,
        edgecolor=ORANGE,
    )
    add_box(
        ax,
        x=0.300,
        y=0.700,
        width=0.240,
        height=0.095,
        title="Readout search",
        body="Pooling · layers · model · head",
        facecolor=LIGHT_ORANGE,
        edgecolor=ORANGE,
    )
    add_box(
        ax,
        x=0.600,
        y=0.700,
        width=0.330,
        height=0.095,
        title="Versioned head manifest",
        body="Weights · calibration · threshold · hashes",
        facecolor=LIGHT_ORANGE,
        edgecolor=ORANGE,
    )
    add_arrow(ax, (0.240, 0.747), (0.300, 0.747), color=ORANGE)
    add_arrow(ax, (0.540, 0.747), (0.600, 0.747), color=ORANGE)

    ax.text(
        0.035,
        0.635,
        "RUNTIME  ·  PAY FOR THE MODEL ONCE",
        fontsize=10.5,
        fontweight="bold",
        color=BLUE,
    )
    add_box(
        ax,
        x=0.030,
        y=0.325,
        width=0.105,
        height=0.145,
        title="Input",
        body="Request\n+ context",
        facecolor=LIGHT_BLUE,
        edgecolor=BLUE,
        title_size=12,
    )
    add_arrow(ax, (0.135, 0.397), (0.175, 0.397), color=BLUE)

    model = FancyBboxPatch(
        (0.175, 0.165),
        0.185,
        0.455,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=2.0,
        facecolor=LIGHT_BLUE,
        edgecolor=BLUE,
    )
    ax.add_patch(model)
    ax.text(0.2675, 0.575, "Frozen local model", ha="center", va="center", fontsize=13, fontweight="bold", color=NAVY)
    ax.text(0.2675, 0.542, "ONE forward pass", ha="center", va="center", fontsize=10.5, fontweight="bold", color=BLUE)
    layer_y = np.linspace(0.215, 0.505, 13)
    for index, y_value in enumerate(layer_y):
        shade = 0.50 + 0.35 * index / (len(layer_y) - 1)
        ax.plot([0.205, 0.330], [y_value, y_value], color=BLUE, alpha=shade, linewidth=2.0)
    ax.text(0.2675, 0.190, "hidden states", ha="center", va="center", fontsize=9, color=MID_GREY)

    add_arrow(ax, (0.360, 0.397), (0.397, 0.397), color=BLUE)
    bus = FancyBboxPatch(
        (0.397, 0.155),
        0.042,
        0.475,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=2.0,
        facecolor="#DDEBF5",
        edgecolor=BLUE,
    )
    ax.add_patch(bus)
    ax.text(0.418, 0.646, "HIDDEN-STATE BUS", ha="center", va="bottom", fontsize=9.4, fontweight="bold", color=BLUE)

    head_specs = [
        (0.536, "Toxicity", "mean · outputs 18–20", "L18–20"),
        (0.458, "Spam", "max · outputs 15–17", "L15–17"),
        (0.380, "Sentiment", "mean · output 12", "L12"),
        (0.302, "Jailbreak", "max · outputs 5–7", "L5–7"),
        (0.224, "Anger", "mean · output 5", "L5"),
        (0.146, "+ 11 more heads", "independent manifests", "other"),
    ]
    for y_value, title, body, layer_label in head_specs:
        tick_y = y_value + 0.036
        ax.plot([0.405, 0.431], [tick_y, tick_y], color=BLUE, linewidth=2.3)
        ax.text(0.390, tick_y, layer_label, ha="right", va="center", fontsize=7.7, color=MID_GREY)
        ax.plot([0.439, 0.477], [tick_y, tick_y], color=GREEN, linewidth=1.5)
        add_box(
            ax,
            x=0.477,
            y=y_value,
            width=0.188,
            height=0.072,
            title=title,
            body=body,
            facecolor=LIGHT_GREEN,
            edgecolor=GREEN,
            title_size=9.4,
            body_size=7.2,
        )
        ax.plot([0.665, 0.710], [tick_y, tick_y], color=GREEN, linewidth=1.25)

    ax.plot([0.710, 0.710], [0.182, 0.572], color=GREEN, linewidth=2.0)
    add_arrow(ax, (0.710, 0.397), (0.750, 0.397), color=GREEN, width=2.0)
    ax.axvline(0.735, ymin=0.145, ymax=0.645, color="#9AA4AF", linewidth=1.2, linestyle="--")
    ax.text(0.748, 0.616, "HOST CONTROL BOUNDARY", ha="left", va="center", fontsize=8.8, fontweight="bold", color=MID_GREY)

    policy = FancyBboxPatch(
        (0.750, 0.220),
        0.210,
        0.330,
        boxstyle="round,pad=0.014,rounding_size=0.020",
        linewidth=1.8,
        facecolor=LIGHT_GREY,
        edgecolor=NAVY,
    )
    ax.add_patch(policy)
    ax.text(0.855, 0.500, "Calibrated scores", ha="center", va="center", fontsize=11.5, fontweight="bold", color=NAVY)
    ax.text(0.855, 0.455, "↓", ha="center", va="center", fontsize=15, color=MID_GREY)
    ax.text(0.855, 0.415, "Host policy", ha="center", va="center", fontsize=12, fontweight="bold", color=NAVY)
    ax.text(0.855, 0.365, "allow · warn · review · block", ha="center", va="center", fontsize=9.2, color="#313B45")
    ax.text(0.855, 0.315, "receipt + irreversible action", ha="center", va="center", fontsize=9.2, color="#313B45")
    ax.text(0.855, 0.270, "outside the sensor", ha="center", va="center", fontsize=9.2, fontweight="bold", color=ORANGE)

    ax.annotate(
        "load one tested head\nwithout retraining the others",
        xy=(0.570, 0.608),
        xytext=(0.655, 0.675),
        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.3, "linestyle": "--"},
        ha="center",
        va="center",
        fontsize=8.7,
        color=ORANGE,
    )

    result_box = FancyBboxPatch(
        (0.035, 0.035),
        0.925,
        0.072,
        boxstyle="round,pad=0.010,rounding_size=0.014",
        linewidth=1.2,
        facecolor=LIGHT_GREY,
        edgecolor="#C8D0D8",
    )
    ax.add_patch(result_box)
    ax.text(0.060, 0.071, "MEASURED", ha="left", va="center", fontsize=9.5, fontweight="bold", color=MID_GREY)
    ax.text(0.145, 0.071, "1 → 16 heads", ha="left", va="center", fontsize=10.5, fontweight="bold", color=NAVY)
    ax.text(0.270, 0.071, "68.63 → 73.34 ms", ha="left", va="center", fontsize=14.2, fontweight="bold", color=GREEN)
    ax.text(0.455, 0.071, "+6.9% total latency", ha="left", va="center", fontsize=10.5, fontweight="bold", color=NAVY)
    ax.text(0.635, 0.071, "n=80 · paired ratio 1.069 · 95% CI [1.056, 1.078]", ha="left", va="center", fontsize=8.5, color=MID_GREY)

    save_public_figure(fig, "one-local-model-pass-many-task-specific-probes")
    plt.close(fig)


def bootstrap_mean_ci(values: np.ndarray, *, seed: int) -> tuple[float, float]:
    """Return a fixed-seed 2,000-resample percentile interval for the mean."""
    assert values.ndim == 1 and len(values) == 10
    indices = np.random.default_rng(seed).integers(0, len(values), size=(2000, len(values)))
    sampled_means = values[indices].mean(axis=1)
    low, high = np.quantile(sampled_means, (0.025, 0.975))
    return float(low), float(high)


def render_latency() -> None:
    """Render matched-prompt shared-path and repeated-forward latency curves."""
    k_values = np.asarray([1, 2, 4, 8, 16], dtype=np.int64)
    shared_frames = [
        pd.read_json(OUTPUTS / f"latency_k{k}.jsonl", lines=True) for k in k_values
    ]
    repeated_frames = [
        pd.read_json(
            OUTPUTS / f"repeated_forward_lower_bound_k{k}.jsonl", lines=True
        )
        for k in k_values
    ]
    assert all(len(frame) == 80 for frame in shared_frames)
    assert all(len(frame) == 10 for frame in repeated_frames)
    assert all(
        shared.iloc[:10]["row_key"].reset_index(drop=True).equals(repeated["row_key"])
        for shared, repeated in zip(shared_frames, repeated_frames, strict=True)
    )

    shared_values = [
        frame.iloc[:10]["total_ms"].to_numpy(dtype=np.float64)
        for frame in shared_frames
    ]
    repeated_values = [
        frame["total_ms"].to_numpy(dtype=np.float64) for frame in repeated_frames
    ]
    shared_mean = np.asarray([values.mean() for values in shared_values])
    repeated_mean = np.asarray([values.mean() for values in repeated_values])
    shared_ci = np.asarray(
        [bootstrap_mean_ci(values, seed=20261018 + int(k)) for k, values in zip(k_values, shared_values, strict=True)]
    )
    repeated_ci = np.asarray(
        [bootstrap_mean_ci(values, seed=20261118 + int(k)) for k, values in zip(k_values, repeated_values, strict=True)]
    )
    shared_error = np.vstack((shared_mean - shared_ci[:, 0], shared_ci[:, 1] - shared_mean))
    repeated_error = np.vstack((repeated_mean - repeated_ci[:, 0], repeated_ci[:, 1] - repeated_mean))
    ratio_k16 = repeated_mean[-1] / shared_mean[-1]
    assert np.isclose(ratio_k16, 15.258962751202098)

    x = np.arange(len(k_values))
    fig, ax = plt.subplots(figsize=(12.0, 7.3))
    ax.errorbar(
        x,
        shared_mean,
        yerr=shared_error,
        marker="o",
        markersize=6.5,
        capsize=3,
        linewidth=2.6,
        color=BLUE,
        label="One model forward + K heads",
    )
    ax.errorbar(
        x,
        repeated_mean,
        yerr=repeated_error,
        marker="o",
        markersize=6.5,
        capsize=3,
        linewidth=2.6,
        color=ORANGE,
        label="K repeated base-model forwards (lower bound)",
    )
    ax.set_title("One pass stays flat. Repeating the model does not.", loc="left")
    ax.set_xlabel("Number of probes (K)")
    ax.set_ylabel("Mean latency per input (ms)")
    ax.set_xticks(x, labels=[str(value) for value in k_values])
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", frameon=False)
    ax.annotate(
        f"{repeated_mean[-1]:.2f} ms\n{ratio_k16:.2f}× the shared path",
        xy=(x[-1], repeated_mean[-1]),
        xytext=(2.70, 760),
        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.5},
        fontsize=11.2,
        fontweight="bold",
        color=ORANGE,
    )
    ax.annotate(
        "Break-even by K=2\nin this fixed run",
        xy=(x[1], repeated_mean[1]),
        xytext=(0.48, 225),
        arrowprops={"arrowstyle": "->", "color": NAVY, "lw": 1.2},
        fontsize=9.2,
        fontweight="bold",
        color=NAVY,
    )

    inset = inset_axes(ax, width="43%", height="36%", loc="center left", borderpad=1.75)
    inset.errorbar(
        x,
        shared_mean,
        yerr=shared_error,
        marker="o",
        capsize=2.5,
        linewidth=2.0,
        color=BLUE,
    )
    inset.set_title("Shared path, same 10 prompts", fontsize=10.2, loc="left")
    inset.set_xticks(x, labels=[str(value) for value in k_values])
    inset.set_ylim(50, 75)
    inset.set_ylabel("ms", fontsize=8.5)
    inset.tick_params(labelsize=8.3)
    inset.grid(True, alpha=0.18)

    fig.text(
        0.01,
        0.010,
        "Qwen2.5-0.5B · batch 1 · matched 10-prompt panel · error bars are 2,000-resample row-bootstrap 95% CIs. "
        "The repeated-forward arm omits task prompts, LM-head scoring and generation.",
        fontsize=8.5,
        color=MID_GREY,
    )
    save_public_figure(fig, "one-local-model-pass-many-task-specific-probes-latency")
    plt.close(fig)


def main() -> None:
    """Render all public figures after checking their frozen inputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    render_architecture()
    render_latency()


if __name__ == "__main__":
    main()
