#!/usr/bin/env python3
"""Draw a paper-style 3x3x3 FSQ cube viewed toward the upper-right."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


OUTPUT_DIR = Path(__file__).resolve().parent

COLORS = {
    "cyan": "#61CBF4",
    "purple": "#D86ECC",
    "orange": "#F2AA84",
}

# Visible annotations are distributed across the top, front, and right faces.
TOP_SELECTED = {(0, 1): COLORS["cyan"]}
FRONT_SELECTED = {(1, 1): COLORS["purple"]}
RIGHT_SELECTED = {(1, 2): COLORS["orange"]}

EDGE_COLOR = "#34464F"
EDGE_WIDTH = 1.0


def project(x: float, y: float, z: float) -> tuple[float, float]:
    """Face-forward dimetric projection whose depth recedes upper-right."""
    return (x + 0.52 * y, z + 0.30 * y)


def mix(hex_color: str, target: str, amount: float) -> str:
    c = tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    t = tuple(int(target[i : i + 2], 16) for i in (1, 3, 5))
    out = tuple(round(a * (1 - amount) + b * amount) for a, b in zip(c, t))
    return "#" + "".join(f"{v:02X}" for v in out)


def add_tile(ax, points, fill: str) -> None:
    ax.add_patch(
        Polygon(
            points,
            closed=True,
            facecolor=fill,
            edgecolor=EDGE_COLOR,
            linewidth=EDGE_WIDTH,
            joinstyle="round",
            antialiased=True,
        )
    )


def draw_front_face(ax) -> None:
    for z in range(3):
        for x in range(3):
            selected = FRONT_SELECTED.get((x, z))
            fill = selected if selected else "#E1E6E8"
            add_tile(
                ax,
                [
                    project(x, 0, z),
                    project(x + 1, 0, z),
                    project(x + 1, 0, z + 1),
                    project(x, 0, z + 1),
                ],
                fill,
            )


def draw_right_face(ax) -> None:
    for z in range(3):
        for y in range(3):
            selected = RIGHT_SELECTED.get((y, z))
            fill = mix(selected, "#000000", 0.06) if selected else "#C8D1D5"
            add_tile(
                ax,
                [
                    project(3, y, z),
                    project(3, y + 1, z),
                    project(3, y + 1, z + 1),
                    project(3, y, z + 1),
                ],
                fill,
            )


def draw_top_face(ax) -> None:
    for y in range(3):
        for x in range(3):
            selected = TOP_SELECTED.get((x, y))
            fill = mix(selected, "#FFFFFF", 0.12) if selected else "#F0F2F3"
            add_tile(
                ax,
                [
                    project(x, y, 3),
                    project(x + 1, y, 3),
                    project(x + 1, y + 1, 3),
                    project(x, y + 1, 3),
                ],
                fill,
            )


def render() -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=300)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.set_aspect("equal")
    ax.axis("off")

    # Flat pastel surfaces and uniform strokes match the surrounding diagram.
    draw_front_face(ax)
    draw_right_face(ax)
    draw_top_face(ax)

    ax.autoscale_view()
    ax.margins(0.055)

    stem = "fsq_space_3x3x3_right_view"
    for suffix in ("png", "svg"):
        fig.savefig(
            OUTPUT_DIR / f"{stem}.{suffix}",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.03,
            dpi=300,
        )
    plt.close(fig)


if __name__ == "__main__":
    render()
