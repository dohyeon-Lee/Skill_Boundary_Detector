#!/usr/bin/env python3
"""Draw the 3x3x3 FSQ space as a lightweight point lattice."""

from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt


OUTPUT_DIR = Path(__file__).resolve().parent

COLORS = {
    "cyan": "#61CBF4",
    "purple": "#D86ECC",
    "orange": "#F2AA84",
}

# Three visible, spatially separated skill codes.
HIGHLIGHTS = {
    (0, 0, 2): COLORS["cyan"],
    (2, 1, 1): COLORS["purple"],
    (0, 2, 0): COLORS["orange"],
}


def project(i: int, j: int, k: int) -> tuple[float, float, float]:
    """Use the exact camera transform from the fsq_codebook HTML."""
    vx, vy, vz = i - 1, j - 1, k - 1
    yaw, pitch = -0.63, 0.46
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    xr = cy * vx - sy * vy
    yr = sy * vx + cy * vy
    x = 300 + xr * 145
    y = 265 + yr * 145 * sp - vz * 145 * cp
    depth = yr * cp + vz * sp
    return x, y, depth


def draw_edges(ax) -> None:
    for i in range(3):
        for j in range(3):
            for k in range(3):
                p0 = project(i, j, k)
                for di, dj, dk in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    q = (i + di, j + dj, k + dk)
                    if max(q) <= 2:
                        p1 = project(*q)
                        ax.plot(
                            (p0[0], p1[0]),
                            (p0[1], p1[1]),
                            color="#646464",
                            linewidth=1.2,
                            alpha=0.42,
                            solid_capstyle="round",
                            zorder=1,
                        )


def draw_points(ax) -> None:
    gray_points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                coord = (i, j, k)
                if coord not in HIGHLIGHTS:
                    x, y, depth = project(*coord)
                    gray_points.append((depth, x, y))

    gray_points.sort()

    ax.scatter(
        [point[1] for point in gray_points],
        [point[2] for point in gray_points],
        s=495,
        facecolor="#9AA3A8",
        edgecolor="none",
        linewidth=0,
        zorder=3,
    )

    for coord, color in HIGHLIGHTS.items():
        x, y, _ = project(*coord)
        ax.scatter(
            [x],
            [y],
            s=1116,
            facecolor=color,
            edgecolor="none",
            linewidth=0,
            zorder=4,
        )


def render() -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.2), dpi=300)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.set_aspect("equal")
    ax.axis("off")

    draw_edges(ax)
    draw_points(ax)

    # Match the source SVG's fixed 600x520 viewBox exactly.
    ax.set_xlim(0, 600)
    ax.set_ylim(520, 0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    stem = "fsq_space_3x3x3_point_lattice"
    for suffix in ("png", "svg"):
        fig.savefig(
            OUTPUT_DIR / f"{stem}.{suffix}",
            transparent=True,
            pad_inches=0,
            dpi=300,
        )
    plt.close(fig)


if __name__ == "__main__":
    render()
