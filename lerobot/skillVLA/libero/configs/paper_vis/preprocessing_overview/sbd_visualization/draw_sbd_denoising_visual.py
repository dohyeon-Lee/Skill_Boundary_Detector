#!/usr/bin/env python3
"""Draw a compact paper figure for the SBD denoising-probe mechanism."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path(__file__).resolve().parent

PEACH = "#F3A77F"
ORANGE = "#C95618"
DARK = "#263B45"
MID_GRAY = "#9DA8AE"
LIGHT_GRAY = "#D9E0E3"
LINE_GRAY = "#B7C0C4"


def rounded_box(ax, x: float, y: float, w: float, h: float, **kwargs) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=14",
        **kwargs,
    )
    ax.add_patch(patch)
    return patch


def draw_dp_and_steps(ax) -> float:
    rounded_box(ax, 300, 686, 400, 70, facecolor=PEACH, edgecolor="none")
    ax.text(
        500,
        721,
        "Proprio-only DP",
        ha="center",
        va="center",
        color="white",
        fontsize=20,
        fontweight="bold",
    )

    ax.add_patch(
        FancyArrowPatch(
            (500, 685),
            (500, 645),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.8,
            color=DARK,
        )
    )

    xs = np.linspace(105, 895, 10)
    step_labels = list(range(10, 0, -1))
    selected_label = 7

    ax.plot((xs[0], xs[-1]), (608, 608), color=LINE_GRAY, linewidth=2.0, zorder=1)
    for x, label in zip(xs, step_labels):
        selected = label == selected_label
        ax.scatter(
            [x],
            [608],
            s=820 if selected else 500,
            facecolor=PEACH if selected else LIGHT_GRAY,
            edgecolor=ORANGE if selected else MID_GRAY,
            linewidth=2.4 if selected else 1.2,
            zorder=3,
        )
        ax.text(
            x,
            608,
            str(label),
            ha="center",
            va="center",
            fontsize=14.5 if selected else 13.5,
            fontweight="bold" if selected else "normal",
            color=DARK,
            zorder=4,
        )

    selected_x = float(xs[step_labels.index(selected_label)])
    return selected_x


def draw_card(ax, x: float, y: float, w: float, h: float) -> None:
    rounded_box(
        ax,
        x,
        y,
        w,
        h,
        facecolor="#FFFFFF",
        edgecolor=LINE_GRAY,
        linewidth=1.35,
        alpha=0.93,
    )
    cx, cy = x + w / 2, y + (h - 40) / 2
    ax.plot((x + 30, x + w - 25), (cy, cy), color="#E5E9EB", linewidth=1.0, zorder=0)
    ax.plot((cx, cx), (y + 24, y + h - 54), color="#E5E9EB", linewidth=1.0, zorder=0)


def draw_probe_cloud(ax, rng: np.random.Generator) -> None:
    x, y, w, h = 44, 105, 392, 380
    draw_card(ax, x, y, w, h)
    center = np.array([x + w / 2, y + 174])

    covariance = np.array([[73.0**2, 0.0], [0.0, 65.0**2]])
    points = rng.multivariate_normal(center, covariance, size=42)
    points[:, 0] = np.clip(points[:, 0], x + 35, x + w - 30)
    points[:, 1] = np.clip(points[:, 1], y + 35, y + h - 65)
    ax.scatter(points[:, 0], points[:, 1], s=58, color="#98A3A8", alpha=0.70, edgecolor="none", zorder=2)

    ax.scatter(
        [center[0]],
        [center[1]],
        marker="*",
        s=185,
        facecolor=PEACH,
        edgecolor=ORANGE,
        linewidth=1.4,
        zorder=5,
    )


def covariance_matrix(width: float, height: float, angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    scales = np.diag([(width / 4.6) ** 2, (height / 4.6) ** 2])
    return rotation @ scales @ rotation.T


def draw_cluster_cloud(ax, rng: np.random.Generator) -> None:
    x, y, w, h = 564, 105, 392, 380
    draw_card(ax, x, y, w, h)

    centers = [
        (x + 125, y + 245),
        (x + 278, y + 126),
    ]
    shapes = [
        (122, 44, 28),
        (118, 44, -52),
    ]
    palette = ["#D46C35", "#E99166"]
    counts = [21, 21]

    for center, shape, color, count in zip(centers, shapes, palette, counts):
        width, height, angle = shape
        ax.add_patch(
            Ellipse(
                center,
                width,
                height,
                angle=angle,
                facecolor=color,
                edgecolor=color,
                linewidth=1.4,
                alpha=0.13,
                zorder=1,
            )
        )
        points = rng.multivariate_normal(center, covariance_matrix(width, height, angle), size=count)
        ax.scatter(points[:, 0], points[:, 1], s=66, color=color, alpha=0.76, edgecolor="none", zorder=3)

    reference = (x + w / 2, y + 174)
    ax.scatter(
        [reference[0]],
        [reference[1]],
        marker="*",
        s=185,
        facecolor=PEACH,
        edgecolor=ORANGE,
        linewidth=1.4,
        zorder=5,
    )


def render() -> None:
    rng = np.random.default_rng(17)
    fig, ax = plt.subplots(figsize=(10.0, 7.8), dpi=300)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 780)
    ax.set_aspect("equal")
    ax.axis("off")

    draw_dp_and_steps(ax)
    draw_probe_cloud(ax, rng)
    draw_cluster_cloud(ax, rng)

    ax.add_patch(
        FancyArrowPatch(
            (454, 287),
            (548, 287),
            arrowstyle="-|>",
            mutation_scale=22,
            linewidth=2.4,
            color=ORANGE,
            zorder=6,
        )
    )

    stem = "sbd_denoising_step7_overview"
    for suffix in ("png", "svg"):
        fig.savefig(
            OUTPUT_DIR / f"{stem}.{suffix}",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.04,
            dpi=300,
        )
    plt.close(fig)


if __name__ == "__main__":
    render()
