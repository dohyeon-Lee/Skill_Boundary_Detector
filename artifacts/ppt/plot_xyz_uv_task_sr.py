#!/usr/bin/env python3
"""Create a PPT-ready task-wise SR comparison for XYZ and UV VSA."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent

LABELS = [f"Task {index}" for index in range(10)] + ["Overall"]
XYZ = [100, 90, 100, 100, 80, 100, 90, 90, 100, 90, 94]
UV = [100, 60, 90, 60, 100, 90, 70, 70, 70, 90, 80]

COLORS = {
    "VSA-XYZ": "#4472C4",
    "VSA-UV": "#FFC000",
}


def build_figure() -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.labelsize": 18,
            "xtick.labelsize": 13,
            "ytick.labelsize": 14,
            "legend.fontsize": 16,
        }
    )

    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Leave a small visual gap before the aggregate column.
    x = np.array([*range(10), 10.8], dtype=float)
    width = 0.30
    offset = 0.18

    xyz_bars = ax.bar(
        x - offset,
        XYZ,
        width=width,
        color=COLORS["VSA-XYZ"],
        edgecolor="none",
        label="VSA-XYZ",
        zorder=3,
    )
    uv_bars = ax.bar(
        x + offset,
        UV,
        width=width,
        color=COLORS["VSA-UV"],
        edgecolor="none",
        label="VSA-UV",
        zorder=3,
    )

    for xyz_bar, uv_bar, xyz_value, uv_value in zip(
        xyz_bars, uv_bars, XYZ, UV, strict=True
    ):
        same_height = xyz_value == uv_value
        xyz_center = xyz_bar.get_x() + xyz_bar.get_width() / 2
        uv_center = uv_bar.get_x() + uv_bar.get_width() / 2
        # Equal values should sit on the same visual baseline. Nudge the text
        # outward horizontally instead of staggering it vertically.
        label_nudge = 0.025 if same_height else 0.0
        ax.text(
            xyz_center - label_nudge,
            xyz_value + 1.8,
            str(xyz_value),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
            color="#202020",
        )
        ax.text(
            uv_center + label_nudge,
            uv_value + 1.8,
            str(uv_value),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
            color="#202020",
        )

    fig.suptitle(
        "VSA-XYZ vs. VSA-UV — Task-wise Success Rate",
        fontsize=27,
        y=0.97,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.905,
        "100k checkpoint · Term_Token_3 · GT skill / focus / goal",
        ha="center",
        va="center",
        fontsize=14,
        color="#666666",
    )

    ax.set_ylabel("Success rate (%)", labelpad=13)
    ax.set_xticks(x, LABELS)
    ax.set_ylim(0, 112)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.axvline(9.9, color="#BFBFBF", linewidth=1.0, zorder=1)

    ax.grid(axis="y", color="#D9D9D9", linewidth=1.0, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#7A7A7A")
    ax.spines["bottom"].set_color("#7A7A7A")
    ax.tick_params(axis="both", colors="#202020")

    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.865),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=2.6,
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.12, top=0.75)
    return fig


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    stem = OUTPUT_DIR / "xyz-vs-uv-vsa-task-success-rate"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(
        OUTPUT_DIR / "xyz-vs-uv-vsa-task-success-rate-transparent.png",
        dpi=300,
        transparent=True,
    )
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
