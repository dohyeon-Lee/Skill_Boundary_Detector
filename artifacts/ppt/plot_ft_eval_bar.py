#!/usr/bin/env python3
"""Create a PPT-ready grouped bar chart for LIBERO-10 FT evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent

CHECKPOINTS = ["5k", "10k"]
SERIES = {
    "PI05": [42, 44],
    "GR00T N1.7": [48, 37],
}

# Exact colors sampled from the supplied reference image.
COLORS = {
    "PI05": "#FFC000",
    "GR00T N1.7": "#8ED973",
}


def build_figure() -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.titlesize": 27,
            "axes.labelsize": 19,
            "xtick.labelsize": 17,
            "ytick.labelsize": 15,
            "legend.fontsize": 16,
        }
    )

    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(CHECKPOINTS), dtype=float)
    width = 0.26
    offsets = (-width / 2, width / 2)

    for (label, values), offset in zip(SERIES.items(), offsets, strict=True):
        bars = ax.bar(
            x + offset,
            values,
            width=width * 0.88,
            color=COLORS[label],
            edgecolor="none",
            label=label,
            zorder=3,
        )
        ax.bar_label(
            bars,
            labels=[f"{value}%" for value in values],
            padding=5,
            fontsize=17,
            fontweight="semibold",
            color="#202020",
        )

    fig.suptitle(
        "LIBERO-10 FT Success Rate",
        fontsize=27,
        y=0.965,
        fontweight="semibold",
    )
    ax.set_xlabel("Fine-tuning checkpoint", labelpad=16)
    ax.set_ylabel("Success rate (%)", labelpad=14)
    ax.set_xticks(x, CHECKPOINTS)
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))

    ax.grid(axis="y", color="#D9D9D9", linewidth=1.0, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#7A7A7A")
    ax.spines["bottom"].set_color("#7A7A7A")
    ax.tick_params(axis="both", colors="#202020")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=2.6,
    )
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.13, top=0.79)
    return fig


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    stem = OUTPUT_DIR / "ft-eval-5k-10k-bar-chart"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(
        OUTPUT_DIR / "ft-eval-5k-10k-bar-chart-transparent.png",
        dpi=300,
        transparent=True,
    )
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
