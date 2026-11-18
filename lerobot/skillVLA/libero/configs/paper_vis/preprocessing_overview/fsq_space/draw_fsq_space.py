#!/usr/bin/env python3
"""Draw a publication-ready 3x3x3 FSQ latent space.

The three highlighted voxels use the annotation colors from the overview figure.
Both transparent PNG and editable SVG variants are exported.
"""

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

# Coordinate values map directly to the three FSQ levels {-1, 0, 1}.
HIGHLIGHTS = {
    (0, 0, 2): COLORS["cyan"],      # (-1, -1,  1)
    (1, 2, 1): COLORS["purple"],    # ( 0,  1,  0)
    (2, 0, 0): COLORS["orange"],    # ( 1, -1, -1)
}

EX = (1.00, 0.48)
EY = (-1.00, 0.48)
EZ = (0.00, 1.10)
GAP = 1.22
SIDE = 0.76


def project(x: float, y: float, z: float) -> tuple[float, float]:
    """Project an FSQ coordinate into a clean isometric view."""
    return (
        x * EX[0] + y * EY[0] + z * EZ[0],
        x * EX[1] + y * EY[1] + z * EZ[1],
    )


def mix(hex_color: str, target: str, amount: float) -> str:
    """Mix a hex color toward another hex color."""
    c = tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    t = tuple(int(target[i : i + 2], 16) for i in (1, 3, 5))
    out = tuple(round(a * (1 - amount) + b * amount) for a, b in zip(c, t))
    return "#" + "".join(f"{v:02X}" for v in out)


def cube_faces(i: int, j: int, k: int) -> list[list[tuple[float, float]]]:
    """Return left, right, and top faces for one discrete FSQ voxel."""
    x, y, z = i * GAP, j * GAP, k * GAP
    p000 = project(x, y, z)
    p100 = project(x + SIDE, y, z)
    p010 = project(x, y + SIDE, z)
    p001 = project(x, y, z + SIDE)
    p101 = project(x + SIDE, y, z + SIDE)
    p011 = project(x, y + SIDE, z + SIDE)
    p111 = project(x + SIDE, y + SIDE, z + SIDE)
    return [
        [p000, p010, p011, p001],  # left face
        [p000, p100, p101, p001],  # right face
        [p001, p101, p111, p011],  # top face
    ]


def draw_cube(ax, coord: tuple[int, int, int], color: str | None) -> None:
    faces = cube_faces(*coord)
    if color is None:
        fills = ["#DDE5E9", "#E8EEF1", "#F6F8F9"]
        edge = "#60717A"
        face_alpha = 0.34
        line_width = 1.15
    else:
        fills = [mix(color, "#000000", 0.12), color, mix(color, "#FFFFFF", 0.20)]
        edge = mix(color, "#000000", 0.35)
        face_alpha = 0.98
        line_width = 2.0

    for face, fill in zip(faces, fills):
        ax.add_patch(
            Polygon(
                face,
                closed=True,
                facecolor=fill,
                edgecolor=edge,
                linewidth=line_width,
                alpha=face_alpha,
                joinstyle="round",
            )
        )


def draw_outer_frame(ax) -> None:
    """Add a subtle wireframe showing the overall 3x3x3 cube volume."""
    lo = -0.18
    hi = 2 * GAP + SIDE + 0.18
    corners = {
        (x, y, z): project(x, y, z)
        for x in (lo, hi)
        for y in (lo, hi)
        for z in (lo, hi)
    }
    for x in (lo, hi):
        for y in (lo, hi):
            ax.plot(*zip(corners[(x, y, lo)], corners[(x, y, hi)]), color="#253A45", lw=1.25, alpha=0.34)
    for y in (lo, hi):
        for z in (lo, hi):
            ax.plot(*zip(corners[(lo, y, z)], corners[(hi, y, z)]), color="#253A45", lw=1.25, alpha=0.34)
    for x in (lo, hi):
        for z in (lo, hi):
            ax.plot(*zip(corners[(x, lo, z)], corners[(x, hi, z)]), color="#253A45", lw=1.25, alpha=0.34)


def draw_axes(ax) -> None:
    """Draw three compact latent-dimension axes with {-1,0,1} ticks."""
    start = project(-0.40, -0.40, -0.35)
    specs = [
        ((3.55, 0.0, 0.0), r"$z_1$", EX),
        ((0.0, 3.55, 0.0), r"$z_2$", EY),
        ((0.0, 0.0, 3.55), r"$z_3$", EZ),
    ]
    for endpoint, label, vec in specs:
        end = project(endpoint[0] - 0.40, endpoint[1] - 0.40, endpoint[2] - 0.35)
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="-|>", color="#24343C", lw=1.7, mutation_scale=12),
            zorder=20,
        )
        ax.text(end[0] + 0.18 * vec[0], end[1] + 0.18 * vec[1], label, fontsize=15, color="#24343C", ha="center", va="center")

    # Compact value key; avoids cluttering all three perspective axes.
    ax.text(0, -1.22, r"FSQ levels: $\{-1,\,0,\,1\}$", ha="center", va="center", fontsize=13.5, color="#354750")


def render(with_axes: bool) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 6.2), dpi=300)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.set_aspect("equal")
    ax.axis("off")

    draw_outer_frame(ax)

    # Painter's order: distant/lower voxels first. Highlights are redrawn last
    # so even an internal code remains legible in the transparent lattice.
    coords = [(i, j, k) for k in range(3) for i in range(3) for j in range(3)]
    coords.sort(key=lambda q: (q[0] + q[1], q[2]), reverse=True)
    for coord in coords:
        draw_cube(ax, coord, None)
    for coord, color in HIGHLIGHTS.items():
        draw_cube(ax, coord, color)

    if with_axes:
        draw_axes(ax)
        stem = "fsq_space_3x3x3_axes"
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-1.55, 7.4)
    else:
        stem = "fsq_space_3x3x3"
        ax.set_xlim(-4.15, 4.15)
        ax.set_ylim(-0.55, 7.1)

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
    render(with_axes=False)
    render(with_axes=True)
