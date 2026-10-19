#!/usr/bin/env python3
"""Plot the frozen 16-head readout census without rerunning model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from render_one_forward_many_decisions import BLUE, NAVY, ORANGE, ROOT


REGISTRY = ROOT / "research/multitask-activation-benchmark/experiments/ONE-FORWARD-MULTIHEAD-EXP001/HEAD_REGISTRY.json"
OUTPUT_DIR = ROOT / "research/multitask-activation-benchmark/paper/outputs/figures"
STEM = "selected-readout-windows"


def readout_frame(registry: dict) -> pd.DataFrame:
    """Validate the frozen head manifest and return its ordered selection census."""
    heads = pd.json_normalize(registry["heads"]).sort_values("order").reset_index(drop=True)
    output_count = registry["backbone"]["hidden_state_outputs"]
    assert registry["backbone"]["id"] == "Qwen/Qwen2.5-0.5B"
    assert registry["head_count"] == len(heads) == 16
    assert output_count == 25
    assert heads.name.is_unique
    assert np.array_equal(heads.order.to_numpy(), np.arange(1, 17))
    assert heads.pooling.isin(["masked_mean", "masked_max"]).all()
    assert heads.start_layer.ge(0).all() and heads.end_layer.lt(output_count).all()
    assert heads.width.between(1, 3).all()
    assert (heads.end_layer - heads.start_layer + 1).eq(heads.width).all()
    assert heads.pooling.value_counts().to_dict() == {"masked_mean": 11, "masked_max": 5}
    assert heads.start_layer.min() == 2 and heads.end_layer.max() == 20
    return heads


def selection_matrix(heads: pd.DataFrame, output_count: int = 25) -> np.ndarray:
    """Encode selected outputs as mean=1 or max=2, leaving other outputs zero."""
    outputs = np.arange(output_count)[None, :]
    selected = (outputs >= heads.start_layer.to_numpy()[:, None]) & (
        outputs <= heads.end_layer.to_numpy()[:, None]
    )
    pooling = np.where(heads.pooling.eq("masked_mean"), 1, 2)[:, None]
    matrix = np.where(selected, pooling, 0)
    assert matrix.shape == (16, output_count)
    assert np.array_equal(np.count_nonzero(matrix, axis=1), heads.width.to_numpy())
    return matrix


def make_figure(heads: pd.DataFrame) -> plt.Figure:
    """Draw exact selected windows with text pooling labels for greyscale access."""
    matrix = selection_matrix(heads)
    labels = (
        heads.name.str.replace("sentiment_positive", "Positive sentiment", regex=False)
        .str.replace("emotion_", "Emotion: ", regex=False)
        .str.replace("pragmatics_", "", regex=False)
        .str.replace("_", " ", regex=False)
        .str.capitalize()
    )
    pooling = heads.pooling.str.replace("masked_", "", regex=False)
    with plt.rc_context({"svg.hashsalt": "intentprobe-readout-census-2026-10-19"}):
        fig, ax = plt.subplots(figsize=(8.5, 5.8))
        ax.pcolormesh(
            np.arange(26) - 0.5,
            np.arange(17) - 0.5,
            matrix,
            cmap=ListedColormap(["#FAFBFC", BLUE, ORANGE]),
            vmin=0,
            vmax=2,
            edgecolors="white",
            linewidth=1.1,
            rasterized=False,
        )
        ax.axvspan(23.5, 24.5, facecolor="#E9EDF2", hatch="///", edgecolor="#C4CCD5", lw=0)
        ax.set_ylim(15.5, -0.5)
        ax.set_xlim(-0.5, 24.5)
        ax.set_yticks(np.arange(16), labels=labels, fontsize=10)
        ax.set_xticks(np.arange(0, 25, 2))
        ax.set_xlabel("Hidden-state output index (0 = embeddings; 24 = final)", fontsize=10)
        ax.tick_params(axis="both", length=0, pad=5)
        ax.grid(False)
        ax.spines[["left", "bottom"]].set_visible(False)
        right = ax.secondary_yaxis("right")
        right.set_yticks(np.arange(16), labels=pooling, fontsize=9)
        right.tick_params(length=0, pad=5)
        right.spines["right"].set_visible(False)
        ax.set_title("Selected readouts span outputs 2–20", loc="left", fontsize=15, pad=36, color=NAVY)
        ax.legend(
            handles=[Patch(facecolor=BLUE, label="Masked mean (11)"), Patch(facecolor=ORANGE, label="Masked max (5)")],
            loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=9,
            borderaxespad=0,
        )
        fig.text(
            0.02, 0.015,
            "Qwen2.5-0.5B · 16 frozen heads · HEAD_REGISTRY.json · 19 October 2026\n"
            "Filled cells show selected windows, not accuracy. No selected window includes output 24.",
            fontsize=8, color="#66717E",
        )
        fig.subplots_adjust(left=0.28, right=0.92, top=0.82, bottom=0.16)
    return fig


def render(output_dir: Path = OUTPUT_DIR) -> None:
    """Save vector and 300-DPI figures from the pinned manifest only."""
    heads = readout_frame(json.loads(REGISTRY.read_text(encoding="utf-8")))
    fig = make_figure(heads)
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"svg.hashsalt": "intentprobe-readout-census-2026-10-19"}):
        fig.savefig(output_dir / f"{STEM}.svg", metadata={"Date": "2026-10-19"})
        fig.savefig(output_dir / f"{STEM}.png", dpi=300, metadata={"Software": "MCPware IntentProbe"})
    plt.close(fig)


def main() -> None:
    """Render to the paper's figure directory or an explicit QA directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    render(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
