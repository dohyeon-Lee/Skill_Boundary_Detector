#!/usr/bin/env python3
"""Create a PPT-ready SR comparison for the largest UV auxiliary gaps."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent

LABELS = ["Task 1\n(3 skills)", "Task 3\n(3 skills)", "Task 6\n(2 skills)"]
PATCHES = [60, 60, 70]
XYZ = [90, 90, 30]

COLORS = {"patches": "#4472C4", "xyz": "#70AD47"}


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

    fig, ax = plt.subplots(figsize=(8.5, 10.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(LABELS), dtype=float)
    width = 0.30
    offset = 0.17

    patches_bars = ax.bar(
        x - offset,
        PATCHES,
        width=width,
        color=COLORS["patches"],
        edgecolor="none",
        label="patches",
        zorder=3,
    )
    xyz_bars = ax.bar(
        x + offset,
        XYZ,
        width=width,
        color=COLORS["xyz"],
        edgecolor="none",
        label="xyz",
        zorder=3,
    )

    for patches_bar, xyz_bar, patches_value, xyz_value in zip(
        patches_bars,
        xyz_bars,
        PATCHES,
        XYZ,
        strict=True,
    ):
        patches_center = patches_bar.get_x() + patches_bar.get_width() / 2
        xyz_center = xyz_bar.get_x() + xyz_bar.get_width() / 2
        ax.text(
            patches_center,
            patches_value + 1.8,
            str(patches_value),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
            color="#202020",
        )
        ax.text(
            xyz_center,
            xyz_value + 1.8,
            str(xyz_value),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
            color="#202020",
        )

    fig.suptitle(
        "Patches vs. XYZ Auxiliary\nLargest Task-wise Gaps",
        fontsize=24,
        y=0.965,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.865,
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
        bbox_to_anchor=(0.5, 0.825),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=2.6,
    )
    fig.subplots_adjust(left=0.13, right=0.975, bottom=0.12, top=0.72)
    return fig


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    stem = OUTPUT_DIR / "uv-patches-vs-xyz-key-tasks-success-rate"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(
        OUTPUT_DIR / "uv-patches-vs-xyz-key-tasks-success-rate-transparent.png",
        dpi=300,
        transparent=True,
    )
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
