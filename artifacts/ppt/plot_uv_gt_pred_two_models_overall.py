#!/usr/bin/env python3
"""Create a PPT-ready overall GT-vs-pred SR chart for three UV VSA models."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
MODEL_LABELS = ["A", "B", "C"]
GT = [80, 80, 86]
PRED = [68, 76, 84]
COLORS = {"GT": "#4472C4", "Pred": "#FFC000"}


def build_figure() -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 14,
            "legend.fontsize": 16,
        }
    )

    fig, ax = plt.subplots(figsize=(8.5, 10.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(MODEL_LABELS), dtype=float)
    width = 0.30
    offset = 0.18
    gt_bars = ax.bar(
        x - offset,
        GT,
        width=width,
        color=COLORS["GT"],
        edgecolor="none",
        label="GT",
        zorder=3,
    )
    pred_bars = ax.bar(
        x + offset,
        PRED,
        width=width,
        color=COLORS["Pred"],
        edgecolor="none",
        label="Pred",
        zorder=3,
    )

    for bars, values in ((gt_bars, GT), (pred_bars, PRED)):
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.8,
                str(value),
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="semibold",
                color="#202020",
            )

    fig.suptitle(
        "GT vs. Predicted Skill + UV",
        fontsize=25,
        y=0.965,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.895,
        "Overall success rate · 100k VSA · 10 tasks × 10 episodes",
        ha="center",
        va="center",
        fontsize=13,
        color="#666666",
    )

    ax.set_ylabel("Success rate (%)", labelpad=12)
    ax.set_xticks(x, MODEL_LABELS)
    ax.set_xlim(-0.65, 2.65)
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
        bbox_to_anchor=(0.5, 0.85),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=2.5,
    )
    fig.subplots_adjust(left=0.14, right=0.975, bottom=0.12, top=0.75)
    return fig


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    stem = OUTPUT_DIR / "uv-vsa-gt-vs-pred-three-models-overall-success-rate"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(
        OUTPUT_DIR / "uv-vsa-gt-vs-pred-three-models-overall-success-rate-transparent.png",
        dpi=300,
        transparent=True,
    )
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
