"""
Vector Field Error Analysis on GT demo actions.

Feeds GT demo action chunks directly into the diffusion UNet
at a specified late denoising timestep, and plots the predicted
error vector field (xyz of the last action in the chunk) over the episode.

Usage:
    python examples/libero/replay_demo_vf_error.py \
      --dataset_dir /scratch/mdorazi/Skill_Boundary_Detector/libero_dataset/libero_90 \
      --task_id 0 --n_episodes 3 \
      --policy_path /scratch/mdorazi/Skill_Boundary_Detector/outputs/dp_libero_90/checkpoints/last/pretrained_model \
      --output_dir /scratch/mdorazi/Skill_Boundary_Detector/outputs/vf_error \
      --num_inference_steps 10 \
      --eval_at_step 8
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tyro

sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class Args:
    dataset_dir: str = "/scratch/mdorazi/Skill_Boundary_Detector/libero_dataset/libero_90"
    task_id: int | None = None
    episode_ids: str | None = None
    """JSON list of episode indices, e.g. '[0,1,2]'."""
    n_episodes: int = 3
    output_dir: str = "/scratch/mdorazi/Skill_Boundary_Detector/outputs/vf_error"
    policy_path: str = ""
    device: str = "cuda"
    seed: int = 42

    num_inference_steps: int = 10
    """Total number of inference denoising steps."""
    eval_at_step: int = 8
    """Which denoising step index (0-based) to evaluate at."""
    noise_scheduler_type: str | None = None
    """Override the policy's noise scheduler type (e.g. 'DDIM'). None = use policy config."""

    replan_interval: int = 0
    """0 = use policy default (n_action_steps)."""

    wandb_project: str | None = None
    wandb_run_name: str = "vf_error"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(dataset_dir: Path) -> pd.DataFrame:
    files = sorted((dataset_dir / "data").rglob("file-*.parquet"))
    if not files:
        files = sorted((dataset_dir / "data").rglob("episode_*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_episodes_meta(dataset_dir: Path) -> pd.DataFrame:
    files = sorted((dataset_dir / "meta" / "episodes").rglob("file-*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def get_video_path(dataset_dir, episode_index, camera_key, episodes_meta):
    row = episodes_meta[episodes_meta["episode_index"] == episode_index].iloc[0]
    chunk_idx = row[f"videos/{camera_key}/chunk_index"]
    file_idx  = row[f"videos/{camera_key}/file_index"]
    return dataset_dir / "videos" / camera_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"


def get_episode_timestamps(dataset_dir, episode_index, episodes_meta, camera_key):
    row = episodes_meta[episodes_meta["episode_index"] == episode_index].iloc[0]
    return float(row[f"videos/{camera_key}/from_timestamp"]), float(row[f"videos/{camera_key}/to_timestamp"])


# ── Policy loading ─────────────────────────────────────────────────────────────

def load_policy(policy_path: str, device: str, num_inference_steps: int, noise_scheduler_type: str | None = None):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy, _make_noise_scheduler
    from lerobot.policies.factory import make_pre_post_processors

    policy = DiffusionPolicy.from_pretrained(policy_path)
    policy = policy.to(device).eval()

    cfg = policy.config
    scheduler_type = noise_scheduler_type if noise_scheduler_type is not None else cfg.noise_scheduler_type
    policy.diffusion.noise_scheduler = _make_noise_scheduler(
        scheduler_type,
        num_train_timesteps=cfg.num_train_timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        beta_schedule=cfg.beta_schedule,
        clip_sample=cfg.clip_sample,
        clip_sample_range=cfg.clip_sample_range,
        prediction_type=cfg.prediction_type,
    )
    policy.diffusion.num_inference_steps = num_inference_steps
    policy.diffusion.noise_scheduler.set_timesteps(num_inference_steps)

    timesteps = policy.diffusion.noise_scheduler.timesteps.tolist()
    print(f"  [policy] scheduler={scheduler_type}, prediction_type={cfg.prediction_type}, "
          f"num_train_timesteps={cfg.num_train_timesteps}, "
          f"inference_steps={num_inference_steps}")
    print(f"  [policy] timesteps: {timesteps}")

    preprocessor, _ = make_pre_post_processors(cfg, pretrained_path=policy_path)
    return policy, preprocessor


# ── Vector field query ─────────────────────────────────────────────────────────

@torch.no_grad()
def query_vf(
    policy,
    global_cond: torch.Tensor,       # (1, global_cond_dim)
    gt_action_chunk: torch.Tensor,   # (1, horizon, action_dim)
    eval_at_step: int,
) -> np.ndarray:
    """Add noise to GT action chunk at the given denoising step's timestep,
    feed into UNet, and return the prediction error at the last action position.

    Error = UNet_pred - eps  (for epsilon prediction_type)
          = UNet_pred - gt   (for sample prediction_type)

    Returns: (action_dim,) error vector for the last action position.
    """
    scheduler = policy.diffusion.noise_scheduler
    t = scheduler.timesteps[eval_at_step]
    t_batch = torch.full((1,), t, dtype=torch.long, device=gt_action_chunk.device)

    # Sample random noise and corrupt GT action chunk at timestep t
    eps = torch.randn_like(gt_action_chunk)
    noisy_chunk = scheduler.add_noise(gt_action_chunk, eps, t_batch)

    # Query UNet with noisy GT action
    pred = policy.diffusion.unet(noisy_chunk, t_batch, global_cond=global_cond)

    # Compute error based on prediction type
    pred_type = policy.config.prediction_type
    if pred_type == "epsilon":
        error = pred - eps          # predicted noise minus actual noise
    elif pred_type == "sample":
        error = pred - gt_action_chunk  # predicted clean action minus GT
    else:
        error = pred                # v-prediction or unknown: return raw pred

    # Return error vector at the last action position: (action_dim,)
    return error[0, -1, :].cpu().numpy()


# ── Episode-level analysis ─────────────────────────────────────────────────────

def run_episode(
    policy,
    preprocessor,
    ep_df: pd.DataFrame,
    cam_clips: list[Path],
    camera_keys: list[str],
    eval_at_step: int,
    replan_interval: int,
) -> dict:
    """Returns replan_ts and vf values (action_dim,) per chunk."""
    from lerobot.utils.constants import OBS_STATE

    n_action_steps = policy.config.n_action_steps
    horizon        = policy.config.horizon
    eff_interval   = replan_interval if replan_interval > 0 else n_action_steps
    device         = next(policy.parameters()).device

    states = np.stack(ep_df["observation.state"].values)
    cam_frames: dict[str, np.ndarray] = {}
    for cam_key, clip_path in zip(camera_keys, cam_clips):
        reader = imageio.get_reader(str(clip_path))
        cam_frames[cam_key] = np.stack([f for f in reader])
        reader.close()

    T = min(len(ep_df), *(len(v) for v in cam_frames.values()))
    states     = states[:T]
    gt_actions = np.stack(ep_df["action"].values[:T])
    for k in cam_frames:
        cam_frames[k] = cam_frames[k][:T]

    replan_ts = []
    vf_values = []   # list of (action_dim,) arrays

    policy.reset()

    for t in range(0, T, eff_interval):
        obs = {k: torch.from_numpy(cam_frames[k][t]).float().div(255.0).permute(2, 0, 1)
               for k in camera_keys}
        obs[OBS_STATE] = torch.from_numpy(states[t]).float()
        obs = preprocessor(obs)

        batch = {k: v.unsqueeze(0).unsqueeze(0).to(device) for k, v in obs.items()}

        global_cond = policy.diffusion._prepare_global_conditioning(batch)

        # Build GT action chunk at t
        end   = min(t + horizon, T)
        chunk = gt_actions[t:end]
        if len(chunk) < horizon:
            chunk = np.concatenate([chunk, np.tile(chunk[-1:], (horizon - len(chunk), 1))], axis=0)
        gt_chunk = torch.from_numpy(chunk).float().unsqueeze(0).to(device)  # (1, H, D)

        vf = query_vf(policy, global_cond, gt_chunk, eval_at_step)

        replan_ts.append(t)
        vf_values.append(vf)

    return {
        "replan_ts": replan_ts,
        "vf_values": np.stack(vf_values),  # (N, action_dim)
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_vf_xyz(replan_ts, vf_values: np.ndarray, ep_id: int, ep_tag: str,
                eval_at_step: int, output_dir: Path) -> Path:
    """Plot xyz (first 3 dims) of the predicted error vector over time."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    labels = ["x", "y", "z"]
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(replan_ts, vf_values[:, i], marker="o", markersize=3)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylabel(f"vf_{label}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frame")
    axes[0].set_title(f"ep{ep_id} | step={eval_at_step} | {ep_tag[:60]}")
    fig.tight_layout()

    path = output_dir / f"ep{ep_id:05d}_vf_step{eval_at_step}.png"
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    return path


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: Args) -> None:
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    print("Loading metadata...")
    episodes_meta = load_episodes_meta(dataset_dir)
    tasks_meta    = pd.read_parquet(dataset_dir / "meta" / "tasks.parquet").reset_index()

    if args.episode_ids is not None:
        episode_ids = json.loads(args.episode_ids)
    elif args.task_id is not None:
        target_lang = tasks_meta[tasks_meta["task_index"] == args.task_id].iloc[0]["task"]
        ep_of_task  = episodes_meta[episodes_meta["tasks"].apply(
            lambda t: target_lang in ([str(x) for x in t] if isinstance(t, (list, np.ndarray)) else [str(t)])
        )]
        episode_ids = ep_of_task["episode_index"].tolist()[:args.n_episodes]
        print(f"Task {args.task_id}: '{target_lang}' — {len(episode_ids)} episodes")
    else:
        raise ValueError("Provide --task_id or --episode_ids")

    video_cols  = [c for c in episodes_meta.columns if c.startswith("videos/") and c.endswith("/chunk_index")]
    camera_keys = [c.split("/")[1] for c in video_cols]

    all_data = load_data(dataset_dir)

    print(f"Loading policy from {args.policy_path} ...")
    policy, preprocessor = load_policy(args.policy_path, args.device, args.num_inference_steps, args.noise_scheduler_type)

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project, name=args.wandb_run_name,
            config=vars(args), reinit=True,
        )

    all_results = []

    for ep_id in episode_ids:
        row = episodes_meta[episodes_meta["episode_index"] == ep_id]
        if row.empty:
            print(f"  Episode {ep_id} not found, skipping.")
            continue
        tasks_list = row.iloc[0]["tasks"]
        lang   = str(tasks_list[0] if isinstance(tasks_list, (list, np.ndarray)) else tasks_list)
        ep_tag = f"ep{ep_id}: {lang[:80]}"
        print(f"\nEpisode {ep_id}: '{lang}'")

        # Extract clips
        cam_clips = []
        for cam_key in camera_keys:
            src   = get_video_path(dataset_dir, ep_id, cam_key, episodes_meta).resolve()
            start, end = get_episode_timestamps(dataset_dir, ep_id, episodes_meta, cam_key)
            clip_path = clips_dir / f"ep{ep_id:05d}_{cam_key.replace('.','_')}.mp4"
            import subprocess
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(start), "-to", str(end),
                 "-i", str(src), "-c", "copy", str(clip_path)],
                check=True, capture_output=True,
            )
            cam_clips.append(clip_path)

        ep_df = all_data[all_data["episode_index"] == ep_id].reset_index(drop=True)

        print(f"  Running VF query at step {args.eval_at_step}/{args.num_inference_steps} ...")
        result = run_episode(
            policy, preprocessor, ep_df, cam_clips, camera_keys,
            args.eval_at_step, args.replan_interval,
        )

        plot_path = plot_vf_xyz(
            result["replan_ts"], result["vf_values"],
            ep_id, ep_tag, args.eval_at_step, output_dir,
        )
        print(f"  Saved: {plot_path.name}")

        if wandb_run is not None:
            import wandb
            for t, vf in zip(result["replan_ts"], result["vf_values"]):
                wandb_run.log({"frame": t, "vf_x": float(vf[0]), "vf_y": float(vf[1]), "vf_z": float(vf[2])})
            wandb_run.log({f"ep{ep_id}/plot": wandb.Image(str(plot_path))})

        all_results.append({
            "episode_id": ep_id,
            "language": lang,
            "replan_ts": result["replan_ts"],
            "vf_values": result["vf_values"].tolist(),
        })

    results_path = output_dir / "vf_error_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDone. Results saved to {results_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
