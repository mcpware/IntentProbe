"""Render the public IntentProbe architecture and latency figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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
OUTPUT_DIR = ROOT / "docs"

NAVY = "#17324D"
BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
TEAL = "#72B7B2"
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
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.18,
        "svg.hashsalt": "intentprobe-architecture-v0.1.0",
    }
)


def save_public_figure(fig: plt.Figure, stem: str) -> None:
    """Save deterministic public PNG/SVG artifacts without trailing SVG whitespace."""
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
    title_color: str = NAVY,
) -> None:
    """Add one rounded architecture box."""
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
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
        fontsize=12.5,
        fontweight="bold",
        color=title_color,
    )
    ax.text(
        x + width / 2,
        y + height * 0.30,
        body,
        ha="center",
        va="center",
        fontsize=9.5,
        color="#313B45",
        linespacing=1.25,
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
    """Add one labelled-flow arrow."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=width,
        linestyle=style,
        color=color,
        shrinkA=3,
        shrinkB=3,
    )
    ax.add_patch(arrow)


def render_architecture() -> None:
    """Render the offline-training and shared-runtime architecture."""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.955,
        "One model pass. Many learned decisions.",
        ha="center",
        va="center",
        fontsize=27,
        fontweight="bold",
        color=NAVY,
    )
    ax.text(
        0.5,
        0.912,
        "IntentProbe — One Forward, Many Decisions",
        ha="center",
        va="center",
        fontsize=14,
        color=MID_GREY,
    )

    ax.text(0.035, 0.785, "OFFLINE: TEACH EACH SMALL HEAD", fontsize=10.5, fontweight="bold", color=ORANGE)
    add_box(
        ax,
        x=0.035,
        y=0.635,
        width=0.20,
        height=0.115,
        title="Labelled examples",
        body="Human-defined task boundary\nand held-out evaluation",
        facecolor=LIGHT_ORANGE,
        edgecolor=ORANGE,
    )
    add_box(
        ax,
        x=0.300,
        y=0.635,
        width=0.24,
        height=0.115,
        title="Readout search",
        body="Choose token pooling, layers,\nnormalization and head",
        facecolor=LIGHT_ORANGE,
        edgecolor=ORANGE,
    )
    add_box(
        ax,
        x=0.605,
        y=0.635,
        width=0.325,
        height=0.115,
        title="Versioned head registry",
        body="Small learned weights + task contract\n+ calibration + threshold + hashes",
        facecolor=LIGHT_ORANGE,
        edgecolor=ORANGE,
    )
    add_arrow(ax, (0.235, 0.692), (0.300, 0.692), color=ORANGE)
    add_arrow(ax, (0.540, 0.692), (0.605, 0.692), color=ORANGE)

    ax.text(0.035, 0.535, "RUNTIME: READ THE MODEL ONCE", fontsize=10.5, fontweight="bold", color=BLUE)
    runtime_y = 0.335
    box_h = 0.135
    add_box(
        ax,
        x=0.035,
        y=runtime_y,
        width=0.155,
        height=box_h,
        title="Input",
        body="Request, context\nand proposed action",
        facecolor=LIGHT_BLUE,
        edgecolor=BLUE,
    )
    add_box(
        ax,
        x=0.230,
        y=runtime_y,
        width=0.185,
        height=box_h,
        title="Frozen local model",
        body="One forward pass\nHidden states exposed",
        facecolor=LIGHT_BLUE,
        edgecolor=BLUE,
    )
    add_box(
        ax,
        x=0.455,
        y=runtime_y,
        width=0.165,
        height=box_h,
        title="Activation bus",
        body="Reuse selected layers\nand valid-token masks",
        facecolor=LIGHT_BLUE,
        edgecolor=BLUE,
    )
    add_box(
        ax,
        x=0.660,
        y=runtime_y,
        width=0.155,
        height=box_h,
        title="Many heads",
        body="Toxicity · spam · mood\nauthorization · routing · …",
        facecolor=LIGHT_GREEN,
        edgecolor=GREEN,
    )
    add_box(
        ax,
        x=0.855,
        y=runtime_y,
        width=0.110,
        height=box_h,
        title="Decisions",
        body="Scores, policy\nand receipt",
        facecolor=LIGHT_GREEN,
        edgecolor=GREEN,
    )
    add_arrow(ax, (0.190, 0.402), (0.230, 0.402), color=BLUE)
    add_arrow(ax, (0.415, 0.402), (0.455, 0.402), color=BLUE)
    add_arrow(ax, (0.620, 0.402), (0.660, 0.402), color=GREEN)
    add_arrow(ax, (0.815, 0.402), (0.855, 0.402), color=GREEN)
    add_arrow(ax, (0.765, 0.635), (0.738, 0.470), color=ORANGE, style="--", width=1.5)
    ax.text(
        0.790,
        0.545,
        "load tested heads",
        ha="left",
        va="center",
        fontsize=9.5,
        color=ORANGE,
    )

    callout = FancyBboxPatch(
        (0.035, 0.105),
        0.930,
        0.135,
        boxstyle="round,pad=0.014,rounding_size=0.018",
        linewidth=1.2,
        facecolor=LIGHT_GREY,
        edgecolor="#C8D0D8",
    )
    ax.add_patch(callout)
    ax.text(0.065, 0.190, "Measured on one six-core CPU", fontsize=11.5, fontweight="bold", color=NAVY)
    ax.text(0.065, 0.145, "1 head", fontsize=10, color=MID_GREY)
    ax.text(0.135, 0.145, "68.63 ms", fontsize=15, fontweight="bold", color=BLUE)
    ax.text(0.295, 0.145, "16 heads", fontsize=10, color=MID_GREY)
    ax.text(0.390, 0.145, "73.34 ms", fontsize=15, fontweight="bold", color=GREEN)
    ax.text(0.555, 0.145, "16 repeated forwards", fontsize=10, color=MID_GREY)
    ax.text(0.750, 0.145, "956.71 ms", fontsize=15, fontweight="bold", color=ORANGE)
    ax.text(
        0.965,
        0.047,
        "Qwen2.5-0.5B · batch 1 · repeated-forward result is a conservative lower bound for separate judges",
        ha="right",
        va="center",
        fontsize=8.8,
        color=MID_GREY,
    )

    save_public_figure(fig, "one-forward-many-decisions")
    plt.close(fig)


def render_latency() -> None:
    """Render the measured shared-path and repeated-forward latency curves."""
    frame = pd.read_csv(RESULT_DIR / "SCALING-SUMMARY.csv")
    expected_k = [1, 2, 4, 8, 16]
    assert frame["k"].tolist() == expected_k
    assert frame["rows"].eq(80).all()
    assert frame.loc[frame["k"].eq(16), "repeated_over_one_forward_ratio"].iat[0] > 15

    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    ax.plot(
        frame["k"],
        frame["mean_total_ms"],
        marker="o",
        linewidth=2.6,
        color=BLUE,
        label="One model forward + K heads",
    )
    ax.plot(
        frame["k"],
        frame["repeated_forward_mean_ms"],
        marker="o",
        linewidth=2.6,
        color=ORANGE,
        label="K repeated base-model forwards (lower bound)",
    )
    ax.set_title("Adding heads is cheap; repeating the model is not", loc="left")
    ax.set_xlabel("Number of decisions (K)")
    ax.set_ylabel("Mean latency per input (ms)")
    ax.set_xticks(expected_k)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", frameon=False)

    k16 = frame.loc[frame["k"].eq(16)].iloc[0]
    ax.annotate(
        f"{k16['repeated_over_one_forward_ratio']:.2f}×\nrepeated / shared",
        xy=(16, k16["repeated_forward_mean_ms"]),
        xytext=(11.3, 760),
        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.5},
        fontsize=11,
        fontweight="bold",
        color=ORANGE,
    )

    inset = inset_axes(ax, width="43%", height="38%", loc="center left", borderpad=1.7)
    inset.plot(frame["k"], frame["mean_total_ms"], marker="o", linewidth=2.1, color=BLUE)
    inset.set_title("Shared path, zoomed", fontsize=10.5, loc="left")
    inset.set_xticks(expected_k)
    inset.set_ylim(65, 76)
    inset.set_ylabel("ms", fontsize=9)
    inset.tick_params(labelsize=8.5)
    inset.grid(True, alpha=0.22)

    fig.text(
        0.01,
        0.012,
        "Qwen2.5-0.5B, batch 1, 80 real held-out texts per K. Two warm-ups excluded. "
        "The repeated-forward arm omits task prompts, LM-head scoring and generation.",
        fontsize=8.7,
        color=MID_GREY,
    )
    save_public_figure(fig, "one-forward-many-decisions-latency")
    plt.close(fig)


def main() -> None:
    """Render all public figures after checking their frozen inputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    render_architecture()
    render_latency()


if __name__ == "__main__":
    main()
