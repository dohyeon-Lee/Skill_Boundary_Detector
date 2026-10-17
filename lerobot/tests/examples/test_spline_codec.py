"""
Spline codec test: variable-length action trajectory → N control points + length → reconstructed trajectory.

Pipeline
--------
  encode : (T, action_dim) → control_points (N, action_dim)  +  length scalar
  decode : control_points  +  length  → (T, action_dim)

Each action dimension is handled independently via a 1-D cubic spline
parameterised on [0, 1].  The T original samples are mapped to N evenly-spaced
parameter values for the control points, and reconstruction re-samples back to T
points.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb
from scipy.interpolate import make_interp_spline


N_CONTROL = 30
SPLINE_DEGREE = 3
SKILL_DATASET_DIR = "/data2/dohyeon/SBD/outputs/skill_dataset_25/skills"


# ── Codec ──────────────────────────────────────────────────────────────────────

GRIPPER_DIM = -1  # last dimension is always interpolated with degree 0 (nearest)


def encode(trajectory: np.ndarray, n_control: int = N_CONTROL, degree: int = SPLINE_DEGREE) -> tuple[np.ndarray, int]:
    T, D = trajectory.shape
    if T < degree + 1:
        raise ValueError(f"Trajectory too short: T={T} for degree={degree}")
    t_orig = np.linspace(0.0, 1.0, T)
    t_ctrl = np.linspace(0.0, 1.0, n_control)
    ctrl_pts = np.zeros((n_control, D), dtype=np.float32)
    for d in range(D):
        k = 0 if d == (D + GRIPPER_DIM) % D else degree
        spl = make_interp_spline(t_orig, trajectory[:, d], k=k)
        ctrl_pts[:, d] = spl(t_ctrl)
    return ctrl_pts, T


def decode(ctrl_pts: np.ndarray, length: int, degree: int = SPLINE_DEGREE) -> np.ndarray:
    n_control, D = ctrl_pts.shape
    t_ctrl = np.linspace(0.0, 1.0, n_control)
    t_out  = np.linspace(0.0, 1.0, length)
    recon = np.zeros((length, D), dtype=np.float32)
    for d in range(D):
        k = 0 if d == (D + GRIPPER_DIM) % D else degree
        spl = make_interp_spline(t_ctrl, ctrl_pts[:, d], k=k)
        recon[:, d] = spl(t_out)
    return recon


# ── Metrics ────────────────────────────────────────────────────────────────────

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def max_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


# ── Plot ───────────────────────────────────────────────────────────────────────

def make_recon_figure(
    original: np.ndarray,
    reconstructed: np.ndarray,
    ctrl_pts: np.ndarray,
    title: str = "",
    dims_to_plot: int = 7,
) -> plt.Figure:
    T = len(original)
    n_ctrl = len(ctrl_pts)
    D = min(dims_to_plot, original.shape[1])

    t_orig = np.arange(T)
    t_ctrl = np.linspace(0, T - 1, n_ctrl)

    fig, axes = plt.subplots(D, 1, figsize=(12, 2.5 * D))
    if D == 1:
        axes = [axes]

    for d, ax in enumerate(axes):
        ax.plot(t_orig, original[:, d],      label="original",      lw=1.5, alpha=0.8)
        ax.plot(t_orig, reconstructed[:, d], label="reconstructed", lw=1.5, linestyle="--", alpha=0.9)
        ax.scatter(t_ctrl, ctrl_pts[:, d],   label="control pts",   s=25, zorder=5, color="red")
        ax.set_ylabel(f"dim {d}", fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("timestep")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    return fig


# ── Data loading ───────────────────────────────────────────────────────────────

def load_skills(dataset_dir: str, n_per_task: int = 10) -> list[dict]:
    """Load up to n_per_task skills from each task folder."""
    skills = []
    for task_dir in sorted(Path(dataset_dir).iterdir()):
        if not task_dir.is_dir():
            continue
        files = sorted(task_dir.glob("*.npz"))[:n_per_task]
        for f in files:
            d = np.load(f)
            skills.append({
                "actions":     d["actions"],        # (T, 7)
                "states":      d["states"],          # (T, 8)
                "task":        task_dir.name,
                "file":        f.name,
                "episode_id":  int(d["episode_id"]),
                "skill_index": int(d["skill_index"]),
            })
    return skills


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_control",     type=int, default=N_CONTROL)
    parser.add_argument("--degree",        type=int, default=SPLINE_DEGREE, help="spline degree (1=linear, 2=quadratic, 3=cubic, ...)")
    parser.add_argument("--n_per_task",    type=int, default=10)
    parser.add_argument("--dataset_dir",   type=str, default=SKILL_DATASET_DIR)
    parser.add_argument("--wandb_project", type=str, default="spline_codec_test")
    args = parser.parse_args()

    run = wandb.init(
        project=args.wandb_project,
        config={"n_control": args.n_control, "degree": args.degree, "n_per_task": args.n_per_task},
    )

    skills = load_skills(args.dataset_dir, n_per_task=args.n_per_task)
    print(f"Loaded {len(skills)} skills from {args.dataset_dir}")

    all_rmse, all_maxerr = [], []

    for i, skill in enumerate(skills):
        traj = skill["actions"]          # (T, 7)
        ctrl_pts, length = encode(traj, n_control=args.n_control, degree=args.degree)
        recon = decode(ctrl_pts, length, degree=args.degree)

        r = rmse(traj, recon)
        m = max_err(traj, recon)
        all_rmse.append(r)
        all_maxerr.append(m)

        label = f"{skill['task']}/{skill['file']}"
        print(f"[{i:03d}] {label:<40}  T={length:>4}  RMSE={r:.5f}  MaxErr={m:.5f}")

        fig = make_recon_figure(
            traj, recon, ctrl_pts,
            title=f"{label}  T={length}  RMSE={r:.5f}  MaxErr={m:.5f}",
        )
        run.log({
            f"recon/{skill['task']}_ep{skill['episode_id']:04d}_sk{skill['skill_index']}": wandb.Image(fig),
            "rmse":    r,
            "max_err": m,
            "T":       length,
            "skill_idx": i,
        })
        plt.close(fig)

    run.log({
        "summary/rmse_mean":    float(np.mean(all_rmse)),
        "summary/rmse_std":     float(np.std(all_rmse)),
        "summary/maxerr_mean":  float(np.mean(all_maxerr)),
        "summary/maxerr_max":   float(np.max(all_maxerr)),
    })

    print(f"\nRMSE  mean={np.mean(all_rmse):.5f}  std={np.std(all_rmse):.5f}")
    print(f"MaxErr mean={np.mean(all_maxerr):.5f}  max={np.max(all_maxerr):.5f}")
    run.finish()


if __name__ == "__main__":
    main()
