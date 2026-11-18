#!/usr/bin/env python3
"""Draw a minimal 3x3x3 FSQ lattice with three highlighted cells."""

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

# Three sample codes scattered across different lateral and depth locations.
HIGHLIGHTS = {
    (0, 0, 2): COLORS["cyan"],      # (-1, -1,  1)
    (2, 1, 1): COLORS["purple"],    # ( 1,  0,  0)
    (0, 2, 0): COLORS["orange"],    # (-1,  1, -1)
}

# The x-z plane is nearly front-facing; y recedes toward the upper-left.
EX = (1.00, 0.00)
EY = (-0.42, 0.30)
EZ = (0.00, 1.00)
GAP = 1.00
SIDE = 1.00


def project(x: float, y: float, z: float) -> tuple[float, float]:
    return (
        x * EX[0] + y * EY[0] + z * EZ[0],
        x * EX[1] + y * EY[1] + z * EZ[1],
    )


def mix(hex_color: str, target: str, amount: float) -> str:
    c = tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    t = tuple(int(target[i : i + 2], 16) for i in (1, 3, 5))
    out = tuple(round(a * (1 - amount) + b * amount) for a, b in zip(c, t))
    return "#" + "".join(f"{v:02X}" for v in out)


def cube_faces(i: int, j: int, k: int) -> list[list[tuple[float, float]]]:
    x, y, z = i * GAP, j * GAP, k * GAP
    p000 = project(x, y, z)
    p100 = project(x + SIDE, y, z)
    p010 = project(x, y + SIDE, z)
    p001 = project(x, y, z + SIDE)
    p101 = project(x + SIDE, y, z + SIDE)
    p011 = project(x, y + SIDE, z + SIDE)
    p111 = project(x + SIDE, y + SIDE, z + SIDE)
    return [
        [p000, p010, p011, p001],  # receding side
        [p000, p100, p101, p001],  # front face
        [p001, p101, p111, p011],  # top face
    ]


def draw_cube(ax, coord: tuple[int, int, int], color: str | None) -> None:
    if color is None:
        return

    fills = [
        mix(color, "#000000", 0.12),
        color,
        mix(color, "#FFFFFF", 0.22),
    ]

    for face, fill in zip(cube_faces(*coord), fills):
        ax.add_patch(
            Polygon(
                face,
                closed=True,
                facecolor=fill,
                edgecolor="none",
                linewidth=0,
                alpha=0.98,
                antialiased=True,
                zorder=3,
            )
        )


def draw_lattice(ax) -> None:
    """Draw the 3x3x3 code space as one continuous wire lattice."""
    line_color = "#84939A"
    line_alpha = 0.56
    line_width = 0.72

    # Lines parallel to each latent dimension. Every family has 4x4 lines,
    # creating 3 cells along all three dimensions without separate gray cubes.
    for y in range(4):
        for z in range(4):
            p0, p1 = project(0, y, z), project(3, y, z)
            ax.plot(*zip(p0, p1), color=line_color, alpha=line_alpha, lw=line_width, solid_capstyle="round")
    for x in range(4):
        for z in range(4):
            p0, p1 = project(x, 0, z), project(x, 3, z)
            ax.plot(*zip(p0, p1), color=line_color, alpha=line_alpha, lw=line_width, solid_capstyle="round")
    for x in range(4):
        for y in range(4):
            p0, p1 = project(x, y, 0), project(x, y, 3)
            ax.plot(*zip(p0, p1), color=line_color, alpha=line_alpha, lw=line_width, solid_capstyle="round")


def render() -> None:
    fig, ax = plt.subplots(figsize=(5.8, 6.2), dpi=300)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.set_aspect("equal")
    ax.axis("off")

    # A single continuous lattice replaces the 27 individual gray voxels.
    draw_lattice(ax)

    # Fill only the three annotated cells. Drawing these after the lattice
    # suppresses distracting lines through the colored faces while preserving
    # a hint of the underlying grid through transparency.
    for coord, color in sorted(HIGHLIGHTS.items(), key=lambda item: item[0][1], reverse=True):
        draw_cube(ax, coord, color)

    ax.autoscale_view()
    ax.margins(0.055)

    stem = "fsq_space_3x3x3_borderless_face"
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
