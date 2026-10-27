#!/usr/bin/env python3
"""Create a PPT-ready GT-vs-pred UV evaluation chart for two VSA models."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent

LABELS = [f"Task {index}" for index in range(10)] + ["Overall"]

# PT 100k GT evaluation recorded for the Notion model
# UV_Fullskill_Bspline_3_lastlayer_lastvis_diffaux.
FULLSKILL_DIFFAUX_GT = [100, 90, 100, 90, 80, 70, 30, 70, 70, 100, 80]
# metrics/eval_info_merged.json from the completed UV predictor evaluation.
FULLSKILL_DIFFAUX_PRED = [100, 60, 90, 70, 80, 90, 50, 80, 60, 80, 76]

# metrics/eval_info_merged.json from the corresponding GT and predictor runs.
BSPLINE_DIFFAUX3_GT = [100, 70, 100, 80, 100, 100, 70, 80, 100, 60, 86]
BSPLINE_DIFFAUX3_PRED = [100, 80, 100, 80, 80, 100, 50, 80, 90, 80, 84]

COLORS = {"GT": "#4472C4", "Pred": "#FFC000"}


def _draw_panel(
    ax: plt.Axes,
    title: str,
    gt_values: list[int],
    pred_values: list[int],
) -> None:
    x = np.array([*range(10), 10.8], dtype=float)
    width = 0.31
    offset = 0.18

    gt_bars = ax.bar(
        x - offset,
        gt_values,
        width=width,
        color=COLORS["GT"],
        edgecolor="none",
        label="GT",
        zorder=3,
    )
    pred_bars = ax.bar(
        x + offset,
        pred_values,
        width=width,
        color=COLORS["Pred"],
        edgecolor="none",
        label="Pred",
        zorder=3,
    )

    for gt_bar, pred_bar, gt_value, pred_value in zip(
        gt_bars, pred_bars, gt_values, pred_values, strict=True
    ):
        same_height = gt_value == pred_value
        nudge = 0.025 if same_height else 0.0
        gt_center = gt_bar.get_x() + gt_bar.get_width() / 2
        pred_center = pred_bar.get_x() + pred_bar.get_width() / 2
        ax.text(
            gt_center - nudge,
            gt_value + 1.6,
            str(gt_value),
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="semibold",
            color="#202020",
        )
        ax.text(
            pred_center + nudge,
            pred_value + 1.6,
            str(pred_value),
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="semibold",
            color="#202020",
        )

    ax.set_title(title, fontsize=16, fontweight="semibold", pad=9)
    ax.set_ylabel("Success rate (%)", labelpad=9)
    ax.set_xticks(x, LABELS)
    ax.set_ylim(0, 113)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.axvline(9.9, color="#BFBFBF", linewidth=1.0, zorder=1)
    ax.grid(axis="y", color="#D9D9D9", linewidth=1.0, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#7A7A7A")
    ax.spines["bottom"].set_color("#7A7A7A")
    ax.tick_params(axis="both", colors="#202020")


def build_figure() -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
            "legend.fontsize": 14,
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(13.333, 8.5), sharex=False, sharey=True)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")

    _draw_panel(
        axes[0],
        "Fullskill lastvis DiffAux   (Overall: GT 80 → Pred 76)",
        FULLSKILL_DIFFAUX_GT,
        FULLSKILL_DIFFAUX_PRED,
    )
    _draw_panel(
        axes[1],
        "Bspline lastlayer DiffAux3   (Overall: GT 86 → Pred 84)",
        BSPLINE_DIFFAUX3_GT,
        BSPLINE_DIFFAUX3_PRED,
    )

    fig.suptitle(
        "GT vs. Predicted Skill + UV — Task-wise Success Rate",
        fontsize=24,
        y=0.985,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.943,
        "100k VSA · Term_Token_3 · 10 episodes/task · UV Predictor mode3 wristX",
        ha="center",
        va="center",
        fontsize=12.5,
        color="#666666",
    )

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.916),
        ncol=2,
        frameon=False,
        handlelength=1.4,
        columnspacing=2.4,
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.075, top=0.83, hspace=0.43)
    return fig


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    stem = OUTPUT_DIR / "uv-vsa-gt-vs-pred-two-models-task-success-rate"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(
        OUTPUT_DIR / "uv-vsa-gt-vs-pred-two-models-task-success-rate-transparent.png",
        dpi=300,
        transparent=True,
    )
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
