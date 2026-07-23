#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluate a SkillVLA policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
lerobot-eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
lerobot-eval \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

```
PYTHONPATH=/data2/dohyeon/SBD/lerobot/src \
  /data2/dohyeon/SBD/.venv/bin/lerobot-eval \
  --policy.path=/data2/dohyeon/SBD/outputs/dp_libero90_yonsei_pretrain/checkpoints/060000/pretrained_model \
  --env.type=libero \
  --env.task=libero_90 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --policy.device=cuda \
  --env.task_ids="[0,1,2]" \
  --wandb_project=lerobot_libero \
  --rename_map='{"observation.images.image2": "observation.images.wrist_image"}'

```

```
PYTHONPATH=/scratch/mdorazi/Skill_Boundary_Detector/lerobot/src \
  /scratch/mdorazi/Skill_Boundary_Detector/.venv/bin/lerobot-eval \
  --policy.path=/scratch/mdorazi/Skill_Boundary_Detector/outputs/pi05_libero_spatial_object/checkpoints/013000/pretrained_model \
  --env.type=libero \
  --env.task=libero_object \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --policy.device=cuda \
  --env.task_ids="[0,1,2]" \
  --wandb_project=libero_pi05_SO \
  --rename_map='{"observation.images.image2": "observation.images.wrist_image"}'

```


Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
"""

import concurrent.futures as cf
import html
import json
import logging
import os
import subprocess
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import Any, TypedDict

import einops
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import (
    add_envs_task,
    check_env_attributes_and_types,
    close_envs,
    preprocess_observation,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.types import PolicyAction
from lerobot.utils.constants import ACTION, DONE, OBS_STR, REWARD
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    init_logging,
    inside_slurm,
)


def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.
        render_callback: Optional rendering callback to be used after the environments are reset, and after
            every step.
    Returns:
        The dictionary described above.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    # Reset the policy and environments.
    policy.reset()
    observation, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    step = 0
    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
        leave=False,
    )
    check_env_attributes_and_types(env)
    while not np.all(done) and step < max_steps:
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation(observation)
        if return_observations:
            all_observations.append(deepcopy(observation))

        # Infer "task" from attributes of environments.
        # TODO: works with SyncVectorEnv but not AsyncVectorEnv
        observation = add_envs_task(env, observation)

        # Apply environment-specific preprocessing (e.g., LiberoProcessorStep for LIBERO)
        observation = env_preprocessor(observation)

        observation = preprocessor(observation)
        with torch.inference_mode():
            action = policy.select_action(observation)
        action = postprocessor(action)

        action_transition = {ACTION: action}
        action_transition = env_postprocessor(action_transition)
        action = action_transition[ACTION]
        record_executed_action = getattr(policy, "record_executed_action", None)
        if record_executed_action is not None:
            record_executed_action(action)

        # Convert to CPU / numpy.
        action_numpy: np.ndarray = action.to("cpu").numpy()
        assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

        # Apply the next action.
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if render_callback is not None:
            render_callback(env)

        # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
        # available if none of the envs finished.
        if "final_info" in info:
            final_info = info["final_info"]
            if not isinstance(final_info, dict):
                raise RuntimeError(
                    "Unsupported `final_info` format: expected dict (Gymnasium >= 1.0). "
                    "You're likely using an older version of gymnasium (< 1.0). Please upgrade."
                )
            successes = final_info["is_success"].tolist()
        else:
            successes = [False] * env.num_envs

        # Keep track of which environments are done so far.
        # Mark the episode as done if we reach the maximum step limit.
        # This ensures that the rollout always terminates cleanly at `max_steps`,
        # and allows logging/saving (e.g., videos) to be triggered consistently.
        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)

        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        step += 1
        running_success_rate = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
        )
        progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()

    # Track the final observation.
    if return_observations:
        observation = preprocess_observation(observation)
        all_observations.append(deepcopy(observation))

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    if return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret[OBS_STR] = stacked_observations

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return ret


def _skill_fsq_levels(policy: PreTrainedPolicy) -> list[int]:
    config = getattr(getattr(policy, "model", None), "config", getattr(policy, "config", None))
    levels = getattr(config, "skill_fsq_levels", None)
    if levels is None:
        raise ValueError(f"Policy config {type(config).__name__} has no skill_fsq_levels.")
    if isinstance(levels, str):
        cleaned = levels.replace("[", " ").replace("]", " ").replace(",", " ")
        levels = [int(v) for v in cleaned.split()]
    return [int(v) for v in levels]


def _token_to_fsq_coord(token: int, levels: list[int]) -> list[int]:
    token = int(max(0, token))
    coords = []
    base = 1
    for level in levels:
        coords.append((token // base) % int(level))
        base *= int(level)
    return coords


def _save_frame_png(frame: np.ndarray, path: Path, size: int = 144) -> str:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(frame, dtype=np.uint8)[..., :3]
    Image.fromarray(arr).resize((size, size), Image.BILINEAR).save(path)
    return path.name


def _plot_skill_progress(path: Path, end_probs: list[dict], *, end_threshold: float = 0.5) -> str:
    """Per-skill FSQ terminator curve: predicted progress (0→1) + end probability over
    the skill's steps. (Replaces the old 7-dim action graph — the decoder action is no
    longer used directly by skillVLA.)"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(end_probs or [], key=lambda r: int(r.get("skill_step", 0)))
    prog_steps = [int(r["skill_step"]) for r in items if r.get("progress") is not None]
    progress = [float(r["progress"]) for r in items if r.get("progress") is not None]
    end_steps = [int(r["skill_step"]) for r in items]
    end_p = [float(r.get("prob", 0.0)) for r in items]

    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    if prog_steps:
        ax.plot(prog_steps, progress, color="#1f77b4", linewidth=1.6, marker="o", markersize=2.5, label="progress")
    if end_steps:
        ax.plot(end_steps, end_p, color="#d62728", linewidth=1.3, linestyle="--", label="end prob")
    ax.axhline(float(end_threshold), color="#888888", linewidth=0.8, linestyle=":", label=f"end thr {end_threshold:g}")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("skill step", fontsize=8)
    ax.set_ylabel("FSQ terminator", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def _plot_skill_timeline_comparison(
    path: Path,
    gt_timeline: list[dict],
    actual_skills: list[dict],
) -> str | None:
    """Wide horizontal Gantt comparing GT skill-transition timing (top row) against the runtime
    FSQ-terminator timing (bottom row). Both follow the same ordered skill codes, so each segment
    is colored by its FSQ token and faint guide lines mark the GT transition boundaries — the
    terminator firing early/late shows as its colored bands shifting left/right of those lines.

    gt_timeline: [{"token", "length"}, ...] (GT demo frame count per skill).
    actual_skills: episode "skills" payload [{"token", "start_t", "end_t"}, ...] (env steps).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if not gt_timeline and not actual_skills:
        return None

    def tok_color(tok: int):
        return cm.tab20((int(tok) % 20) / 20.0)

    # GT segments laid out cumulatively from t=0.
    gt_segs, t0 = [], 0
    for s in gt_timeline:
        length = max(int(s.get("length", 0) or 0), 0)
        gt_segs.append((t0, length, int(s.get("token", -1))))
        t0 += length
    gt_total = t0

    # Actual (terminator) segments from recorded start/end timesteps.
    act_segs = []
    for s in actual_skills:
        st, en = int(s.get("start_t", 0)), int(s.get("end_t", 0))
        act_segs.append((st, max(en - st, 0), int(s.get("token", -1))))
    act_total = max((st + length for st, length, _ in act_segs), default=0)

    xmax = max(gt_total, act_total, 1)
    fig, ax = plt.subplots(figsize=(13, 1.9))
    rows = [("GT (demo)", gt_segs, 1.1), ("Terminator", act_segs, 0.1)]
    for _label, segs, y in rows:
        for st, length, tok in segs:
            if length <= 0:
                continue
            ax.add_patch(Rectangle((st, y), length, 0.7, facecolor=tok_color(tok),
                                   edgecolor="white", linewidth=1.0))
            if length >= xmax * 0.03:
                ax.text(st + length / 2, y + 0.35, f"#{tok}", ha="center", va="center",
                        fontsize=7, color="#17202a")

    # Faint guide lines at GT transition boundaries (span both rows for visual comparison).
    for boundary, *_ in gt_segs[1:]:
        ax.axvline(boundary, color="#888888", linewidth=0.7, linestyle="--", alpha=0.55)
    if gt_total:
        ax.axvline(gt_total, color="#888888", linewidth=0.7, linestyle="--", alpha=0.55)

    ax.set_xlim(0, xmax * 1.01)
    ax.set_ylim(-0.05, 2.0)
    ax.set_yticks([0.45, 1.45])
    ax.set_yticklabels(["terminator", "GT (demo)"], fontsize=9)
    ax.set_xlabel("timestep", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title("skill transition timing — GT vs terminator (dashed = GT boundaries)", fontsize=9)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def _load_raw_dataset_meta(dataset_dir: Path):
    import pandas as pd

    files = sorted((dataset_dir / "meta" / "episodes").rglob("file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet files under {dataset_dir / 'meta' / 'episodes'}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def _annotate_eval_video(frames: np.ndarray, success: bool, task_description: str | None) -> np.ndarray:
    """Eval-video annotation: a TOP bar colored by the episode outcome (green SUCCESS / red FAIL) and,
    when available, a BOTTOM bar with the task language prompt (word-wrapped). Both bars are static, so
    they are rendered ONCE and broadcast over time. frames (t, H, W, 3) uint8 → taller frames."""
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415

    t, h, w = frames.shape[:3]

    def _font(size: int):
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
        except Exception:  # noqa: BLE001  (bitmap fallback — size is then fixed/small)
            return ImageFont.load_default()

    # top outcome bar
    top_h = max(18, h // 10)
    top = Image.new("RGB", (w, top_h), (34, 139, 34) if success else (178, 34, 34))
    draw = ImageDraw.Draw(top)
    label = "SUCCESS" if success else "FAIL"
    font = _font(max(10, int(top_h * 0.62)))
    draw.text(((w - draw.textlength(label, font=font)) / 2, top_h * 0.14), label,
              fill=(255, 255, 255), font=font)
    parts = [np.broadcast_to(np.asarray(top, dtype=frames.dtype), (t, top_h, w, 3)), frames]

    # bottom language-prompt bar (wrapped to the frame width)
    if task_description:
        fs = max(10, int(h * 0.055))
        pfont = _font(fs)
        lines, cur = [], ""
        for word in str(task_description).split():
            trial = f"{cur} {word}".strip()
            if draw.textlength(trial, font=pfont) <= w - 8:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        line_h = fs + 4
        bot_h = 6 + line_h * len(lines)
        bot = Image.new("RGB", (w, bot_h), (20, 20, 20))
        bdraw = ImageDraw.Draw(bot)
        for i, ln in enumerate(lines):
            bdraw.text(((w - bdraw.textlength(ln, font=pfont)) / 2, 3 + i * line_h), ln,
                       fill=(240, 240, 240), font=pfont)
        parts.append(np.broadcast_to(np.asarray(bot, dtype=frames.dtype), (t, bot_h, w, 3)))
    return np.concatenate(parts, axis=1)


def _raw_video_path(dataset_dir: Path, episodes_meta, episode_id: int, image_key: str) -> Path:
    row = episodes_meta[episodes_meta["episode_index"] == int(episode_id)].iloc[0]
    chunk_idx = int(row[f"videos/{image_key}/chunk_index"])
    file_idx = int(row[f"videos/{image_key}/file_index"])
    return dataset_dir / "videos" / image_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"


def _read_video_frame(path: Path, frame_index: int) -> np.ndarray:
    try:
        import imageio.v2 as imageio

        reader = imageio.get_reader(str(path))
        try:
            return np.asarray(reader.get_data(max(0, int(frame_index))))[..., :3].astype(np.uint8)
        finally:
            reader.close()
    except Exception:
        from torchvision.io import read_video

        frames, _, _ = read_video(str(path), output_format="THWC", pts_unit="sec")
        if frames.numel() == 0:
            raise ValueError(f"No frames decoded from {path}")
        idx = max(0, min(int(frame_index), int(frames.shape[0]) - 1))
        return frames[idx].numpy().astype(np.uint8)[..., :3]


def _libero_task_descriptions(suite_name: str) -> dict[int, str]:
    try:
        from libero.libero import benchmark

        suite = benchmark.get_benchmark_dict()[suite_name]()
        return {int(i): str(task.language) for i, task in enumerate(suite.tasks)}
    except Exception as exc:
        logging.warning("Could not load LIBERO task descriptions for %s: %s", suite_name, exc)
        return {}


def _save_training_skill_examples(
    *,
    tokens: set[int],
    assets_dir: Path,
    skill_latents_path: str | None,
    raw_dataset_dir: str | None,
    image_key: str | None,
    n_samples: int,
    image_size: int = 112,
) -> dict[int, list[dict[str, str]]]:
    if not tokens or not skill_latents_path or not raw_dataset_dir or not image_key:
        return {}
    latents_path = Path(skill_latents_path)
    dataset_dir = Path(raw_dataset_dir)
    if not latents_path.exists() or not dataset_dir.exists():
        logging.warning(
            "Skipping FSQ training examples: missing skill_latents_path=%s or raw_dataset_dir=%s",
            latents_path,
            dataset_dir,
        )
        return {}

    data = np.load(str(latents_path), mmap_mode="r")
    required = {"tokens", "episode_id", "frame_start", "frame_end"}
    if not required.issubset(set(data.files)):
        logging.warning("Skipping FSQ examples because %s does not contain %s", latents_path, sorted(required))
        return {}

    meta = _load_raw_dataset_meta(dataset_dir)
    token_arr = np.asarray(data["tokens"])
    rng = np.random.default_rng(0)  # reproducible RANDOM sampling of examples per token
    out: dict[int, list[dict[str, str]]] = {}
    for token in sorted(tokens):
        matches = np.flatnonzero(token_arr == int(token))
        k = max(0, int(n_samples))
        if k and len(matches) > k:
            rows = np.sort(rng.choice(matches, size=k, replace=False))  # random subset, shown in order
        else:
            rows = matches[:k]
        examples = []
        for local_i, row in enumerate(rows):
            episode_id = int(np.asarray(data["episode_id"])[row])
            frame_start = int(np.asarray(data["frame_start"])[row])
            frame_end = max(frame_start, int(np.asarray(data["frame_end"])[row]) - 1)
            try:
                video = _raw_video_path(dataset_dir, meta, episode_id, image_key)
                start_img = _read_video_frame(video, frame_start)
                end_img = _read_video_frame(video, frame_end)
            except Exception as exc:
                logging.warning("Could not load FSQ example token=%s row=%s: %s", token, row, exc)
                continue
            start_name = _save_frame_png(start_img, assets_dir / f"token{token:04d}_ex{local_i:02d}_start.png", image_size)
            end_name = _save_frame_png(end_img, assets_dir / f"token{token:04d}_ex{local_i:02d}_end.png", image_size)
            examples.append(
                {
                    "start": start_name,
                    "end": end_name,
                    "episode_id": episode_id,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                }
            )
        out[int(token)] = examples
    return out


def _index_success_badge(valid_pcts: list) -> str:
    if not valid_pcts:
        return ""
    overall = sum(valid_pcts) / len(valid_pcts)
    n_ok = sum(1 for p in valid_pcts if p >= 50)
    pct_str = f"{overall:.0f}%"
    color = "#22c55e" if overall >= 50 else "#ef4444"
    return f'<div class="badge" style="background:{color}">{pct_str} Success Rate &nbsp;({n_ok} / {len(valid_pcts)} tasks)</div>'

def _index_card_badge(pc_success) -> str:
    if pc_success is None or (isinstance(pc_success, float) and pc_success != pc_success):
        return ""
    if pc_success >= 50:
        return '<span class="success">Success</span>'
    return '<span class="success" style="background:#fee2e2;color:#991b1b;">Fail</span>'

def _write_index_html(
    *,
    output_dir: Path,
    per_task_infos: list[dict],
    task_descriptions: dict[int, str],
    job_name: str = "",
) -> str:
    """Generate output_dir/index.html — matches the reference template, adds language prompt."""
    output_dir = Path(output_dir)
    sorted_tasks = sorted(per_task_infos, key=lambda x: (x.get("task_group", ""), x.get("task_id", 0)))

    def _task_pc(metrics: dict):
        pc = metrics.get("pc_success")
        if pc is None:
            s = metrics.get("successes", [])
            pc = sum(1 for x in s if x) / len(s) * 100 if s else None
        return float(pc) if pc is not None and not (isinstance(pc, float) and pc != pc) else None

    valid_pcts = [p for t in sorted_tasks if (p := _task_pc(t.get("metrics", {}))) is not None]
    badge_html = _index_success_badge(valid_pcts)

    suite_name = "LIBERO-90"
    if sorted_tasks:
        tg = str(sorted_tasks[0].get("task_group", ""))
        if tg:
            suite_name = tg.upper().replace("_", "-")

    cards_html = ""
    for task_info in sorted_tasks:
        task_id = int(task_info.get("task_id", 0))
        metrics = task_info.get("metrics", {})
        desc = task_descriptions.get(task_id, "")
        desc_esc = html.escape(desc)
        card_badge = _index_card_badge(_task_pc(metrics))

        video_tag = ""
        for vp in metrics.get("video_paths", [])[:1]:
            vp = Path(vp)
            try:
                rel = vp.relative_to(output_dir)
            except ValueError:
                rel = vp
            video_tag = f'<video controls playsinline preload="metadata" src="{rel}"></video>'

        skill_href = ""
        for hp in metrics.get("skill_html_paths", [])[:1]:
            hp = Path(hp)
            try:
                rel_h = hp.relative_to(output_dir)
            except ValueError:
                rel_h = hp
            skill_href = str(rel_h)

        prompt_html = f'<div class="card-prompt">{desc_esc}</div>' if desc_esc else ""
        footer_html = f'<div class="card-footer"><a href="{skill_href}">View Skill Trace</a></div>' if skill_href else ""

        cards_html += f"""    <div class="card">
      <div class="card-header">
        <span class="card-title">Task {task_id:02d}</span>
        {card_badge}
      </div>
      {video_tag}
      {prompt_html}
      {footer_html}
    </div>
"""

    index_path = output_dir / "index.html"
    index_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SkillVLA FSQ Eval — {suite_name}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f0f2f5; color: #1a1d23; }}
    header {{ background: #1a1d23; color: #fff; padding: 24px 32px; }}
    header h1 {{ font-size: 20px; font-weight: 700; letter-spacing: -0.3px; }}
    header p {{ margin-top: 6px; font-size: 13px; color: #9aa3b0; font-family: monospace; word-break: break-all; }}
    .badge {{ display: inline-block; margin-top: 14px; color: #fff; font-size: 15px; font-weight: 700; padding: 5px 14px; border-radius: 20px; }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 28px 20px; }}
    h2 {{ font-size: 16px; font-weight: 600; color: #4b5563; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
    .card {{ background: #fff; border: 1px solid #dde2ea; border-radius: 12px; overflow: hidden; }}
    .card-header {{ padding: 12px 16px; border-bottom: 1px solid #eef0f4; display: flex; justify-content: space-between; align-items: center; }}
    .card-title {{ font-weight: 700; font-size: 14px; }}
    .success {{ background: #dcfce7; color: #15803d; font-size: 12px; font-weight: 600; padding: 3px 9px; border-radius: 12px; }}
    video {{ width: 100%; display: block; background: #000; }}
    .card-prompt {{ padding: 10px 16px; font-size: 12px; color: #4b5563; line-height: 1.45; border-top: 1px solid #eef0f4; min-height: 52px; }}
    .card-footer {{ padding: 10px 16px; }}
    .card-footer a {{ display: block; text-align: center; background: #3b82f6; color: #fff; text-decoration: none; font-size: 13px; font-weight: 600; padding: 8px; border-radius: 8px; }}
    .card-footer a:hover {{ background: #2563eb; }}
    @media (max-width: 480px) {{ .grid {{ grid-template-columns: 1fr; }} header {{ padding: 18px 16px; }} }}
  </style>
</head>
<body>
<header>
  <h1>SkillVLA FSQ Eval — {suite_name}</h1>
  <p>{html.escape(job_name)}</p>
  {badge_html}
</header>
<main>
  <h2>Tasks</h2>
  <div class="grid">
{cards_html}  </div>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return str(index_path)


def _write_task_skill_html(
    *,
    task_group: str,
    task_id: int,
    task_description: str | None,
    records: list[dict],
    policy: PreTrainedPolicy,
    output_dir: Path,
    train_samples: int,
    skill_latents_path: str | None,
    raw_dataset_dir: str | None,
    image_key: str | None,
) -> str | None:
    if not records:
        return None

    task_dir = output_dir / f"{task_group}_task{task_id:02d}"
    assets_dir = task_dir / "assets"
    task_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    levels = _skill_fsq_levels(policy)
    num_tokens = int(np.prod(levels))
    used_tokens: set[int] = set()
    episode_payloads = []

    for episode_record in records:
        episode_idx = int(episode_record.get("episode_index", 0))
        frames = np.asarray(episode_record.get("frames", []), dtype=np.uint8)
        trace = episode_record.get("trace", [])
        gt_trace = episode_record.get("gt_trace", [])  # [{token, length}] GT demo timing (oracle eval)
        skill_payloads = []
        for skill in trace:
            token = int(skill.get("codebook_token", -1))
            if token < 0 or token >= num_tokens:
                continue
            used_tokens.add(token)
            skill_idx = int(skill.get("skill_index", len(skill_payloads)))
            start_t = max(0, int(skill.get("episode_timestep", 0)))
            length = max(
                int(skill.get("length", 0) or 0),
                len(skill.get("expert_actions", [])),
                len(skill.get("decoder_actions", [])),
                1,
            )
            end_t = max(start_t, start_t + length)
            if len(frames) > 0:
                start_frame = frames[min(start_t, len(frames) - 1)]
                end_frame = frames[min(end_t, len(frames) - 1)]
            else:
                start_frame = np.zeros((144, 144, 3), dtype=np.uint8)
                end_frame = start_frame
            stem = f"ep{episode_idx:03d}_skill{skill_idx:03d}_token{token:04d}"
            start_name = _save_frame_png(start_frame, assets_dir / f"{stem}_start.png")
            end_name = _save_frame_png(end_frame, assets_dir / f"{stem}_end.png")
            graph_name = _plot_skill_progress(
                assets_dir / f"{stem}_progress.png",
                skill.get("end_probs", []),
                end_threshold=float(getattr(policy.config, "skill_end_threshold",
                                            getattr(policy.config, "skill_decoder_end_threshold", 0.5))),
            )
            skill_payloads.append(
                {
                    "skill_index": skill_idx,
                    "token": token,
                    "coord": _token_to_fsq_coord(token, levels),
                    "start": start_name,
                    "end": end_name,
                    "graph": graph_name,
                    "start_t": start_t,
                    "end_t": end_t,
                    "skill_length": length,
                    "source": str(skill.get("skill_source", "pred")),
                    "patch_flags_start": skill.get("patch_flags_start"),  # [is_red, is_green] per patch at skill start
                    "patch_flags_end": skill.get("patch_flags_end"),    # [is_red, is_green] per patch at skill end
                }
            )
        # GT-vs-terminator skill-transition timing comparison (oracle eval only; gt_trace present).
        timeline_graph = None
        if gt_trace:
            timeline_graph = _plot_skill_timeline_comparison(
                assets_dir / f"ep{episode_idx:03d}_timeline.png", gt_trace, skill_payloads
            )
        episode_payloads.append(
            {"episode_index": episode_idx, "skills": skill_payloads, "timeline_graph": timeline_graph}
        )

    examples_by_token = _save_training_skill_examples(
        tokens=used_tokens,
        assets_dir=assets_dir,
        skill_latents_path=skill_latents_path,
        raw_dataset_dir=raw_dataset_dir,
        image_key=image_key,
        n_samples=train_samples,
    )

    payload = {
        "task_group": task_group,
        "task_id": int(task_id),
        "task_description": task_description or "",
        "levels": levels,
        "chunk_size": int(getattr(policy.config, "chunk_size", 0)),
        "episodes": episode_payloads,
        "examples": examples_by_token,
    }
    data_json = json.dumps(payload)
    html_path = task_dir / "skill_trace.html"
    task_description_html = html.escape(task_description or "unknown")
    html_path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>SkillVLA FSQ Trace task {task_id}</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f5f6f8; color: #17202a; }}
    header {{ padding: 14px 18px; background: #ffffff; border-bottom: 1px solid #d8dee8; }}
    h1 {{ font-size: 18px; margin: 0; }}
    .prompt {{ margin-top: 8px; font-size: 14px; color: #344054; max-width: 1200px; line-height: 1.4; }}
    .prompt b {{ color: #17202a; }}
    .episode {{ margin: 16px; padding: 14px; background: white; border: 1px solid #d8dee8; border-radius: 8px; }}
    .episode-title {{ font-weight: 700; margin-bottom: 10px; }}
    .timeline {{ width: 100%; display: block; margin: 2px 0 14px; border: 1px solid #d8dee8; border-radius: 6px; }}
    .row {{ display: grid; grid-template-columns: 600px 1fr; gap: 16px; align-items: start; }}
    .cube-wrap {{ position: sticky; top: 12px; border: 1px solid #cfd7e5; border-radius: 8px; padding: 10px; background: #fbfcff; }}
    svg {{ width: 100%; height: 540px; display: block; }}
    .skill-card--short {{ background: #f0f2f5; border-color: #c8cdd6; opacity: 0.6; }}
    .skills {{ overflow-x: auto; display: flex; gap: 18px; padding-bottom: 10px; }}
    .skill-card {{ flex: 0 0 330px; border: 1px solid #ccd5e3; border-radius: 8px; padding: 8px; cursor: pointer; background: #ffffff; }}
    .skill-card.active {{ border-color: #d62728; box-shadow: 0 0 0 2px rgba(214,39,40,0.18); }}
    .skill-title {{ font-weight: 700; font-size: 13px; margin-bottom: 7px; }}
    .pair {{ display: flex; gap: 8px; }}
    .pair img {{ width: 144px; height: 144px; object-fit: cover; border: 1px solid #ccd5e3; }}
    .graph {{ margin-top: 8px; width: 100%; border: 1px solid #ccd5e3; }}
    .examples {{ margin-top: 14px; padding: 12px; background: #eef1f5; border: 1px solid #ccd5e3; border-radius: 8px; min-height: 138px; }}
    .examples h3 {{ margin: 0 0 10px; font-size: 14px; }}
    .example-strip {{ display: flex; gap: 18px; overflow-x: auto; }}
    .example {{ display: flex; gap: 6px; align-items: start; }}
    .example img {{ width: 112px; height: 112px; object-fit: cover; border: 1px solid #b8c2d1; }}
    .muted {{ color: #687386; font-size: 12px; }}
    .patch-grid-wrap {{ margin-top: 8px; }}
    .patch-grid-label {{ font-size: 11px; color: #687386; margin-bottom: 4px; }}
    .patch-grid {{ display: inline-grid; gap: 1px; border: 1px solid #ccd5e3; }}
    .patch-grid .p {{ width: 16px; height: 16px; }}
  </style>
</head>
<body>
<header>
  <h1>SkillVLA FSQ Trace: {html.escape(task_group)} / task {int(task_id)}</h1>
  <div class="prompt"><b>Language prompt:</b> {task_description_html}</div>
</header>
<div id="app"></div>
<script>
const DATA = {data_json};
const ASSET = "assets/";

function coordToToken(coord, levels) {{
  let token = 0, base = 1;
  for (let i = 0; i < coord.length; i++) {{ token += coord[i] * base; base *= levels[i]; }}
  return token;
}}

function project(c, levels) {{
  const lx=Math.max(1,levels[0]-1), ly=Math.max(1,levels[1]-1);
  const lz=levels.length>2 ? Math.max(1,levels[2]-1) : 1;
  const xn=(c[0]/lx-0.5)*2, yn=(c[1]/ly-0.5)*2;
  const zn=levels.length>2 ? (c[2]/lz-0.5)*2 : 0;
  const yaw=-0.63, pitch=0.46;
  const cyaw=Math.cos(yaw), syaw=Math.sin(yaw), cp=Math.cos(pitch), sp=Math.sin(pitch);
  const xr=cyaw*xn-syaw*yn, yr=syaw*xn+cyaw*yn;
  const scale=140;
  return [300+xr*scale, 275+yr*scale*sp-zn*scale*cp, yr*cp+zn*sp];
}}

function renderCube(svg, selectedToken) {{
  const levels = DATA.levels;
  const maxToken = levels.reduce((a,b)=>a*b, 1);
  svg.innerHTML = "";
  const NS = "http://www.w3.org/2000/svg";
  const make = (name, attrs) => {{
    const el = document.createElementNS(NS, name);
    Object.entries(attrs).forEach(([k,v]) => el.setAttribute(k, v));
    svg.appendChild(el);
    return el;
  }};
  // inner grid lines
  for (let t = 0; t < maxToken; t++) {{
    const c = []; let base = 1;
    for (const L of levels) {{ c.push(Math.floor(t / base) % L); base *= L; }}
    for (let d = 0; d < Math.min(3, levels.length); d++) {{
      const n = c.slice(); n[d] += 1;
      if (n[d] < levels[d]) {{
        const [x1,y1] = project(c, levels), [x2,y2] = project(n, levels);
        make("line", {{x1, y1, x2, y2, stroke: "rgba(100,100,100,0.5)", "stroke-width": 1.5}});
      }}
    }}
  }}
  // outer box edges
  const BL = levels.map(l => l-1);
  const corners = [0,1,2,3,4,5,6,7].map(b => [BL[0]*(b&1), BL[1]*((b>>1)&1), (BL[2]||0)*((b>>2)&1)]);
  [[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]].forEach(([a,b]) => {{
    const [x1,y1] = project(corners[a], levels), [x2,y2] = project(corners[b], levels);
    make("line", {{x1, y1, x2, y2, stroke: "rgba(50,50,50,0.75)", "stroke-width": 2.2}});
  }});
  // depth-sorted circles
  const pts = [];
  for (let t = 0; t < maxToken; t++) {{
    const c = []; let base = 1;
    for (const L of levels) {{ c.push(Math.floor(t / base) % L); base *= L; }}
    const p = project(c, levels);
    const used = DATA.episodes.some(ep => ep.skills.some(s => s.token === t));
    pts.push({{t, p, used}});
  }}
  pts.sort((a,b) => a.p[2] - b.p[2]);
  pts.forEach(({{t, p, used}}) => {{
    make("circle", {{
      cx: p[0], cy: p[1], r: t === selectedToken ? 9 : (used ? 5.5 : 3.8),
      fill: t === selectedToken ? "#d62728" : (used ? "#2f6f9f" : "#d7dde8"),
      stroke: t === selectedToken ? "#8b0000" : "#26384d",
      "stroke-width": t === selectedToken ? 2.0 : 0.9,
      "data-token": t
    }}).addEventListener("click", () => selectToken(t));
  }});
}}

function buildPatchGrid(flags, cols) {{
  return flags.map(([r, g]) => {{
    let color;
    if (r >= 0.5 && g >= 0.5) color = "#2196f3";
    else if (r >= 0.5)         color = "#e53935";
    else if (g >= 0.5)         color = "#43a047";
    else                       color = "#f0f2f5";
    return `<div class="p" style="background:${{color}}"></div>`;
  }}).join("");
}}

function renderPatchGrids(flagsStart, flagsEnd) {{
  if (!flagsStart && !flagsEnd) return "";
  const flags = flagsStart || flagsEnd;
  const cols = Math.round(Math.sqrt(flags.length));
  let html = `<div class="patch-grid-wrap"><div class="patch-grid-label">flag predictor patches (red=is_red · green=is_green · blue=both)</div><div style="display:flex;gap:8px;align-items:flex-start">`;
  if (flagsStart) html += `<div><div class="muted" style="margin-bottom:2px">start</div><div class="patch-grid" style="grid-template-columns:repeat(${{cols}},16px)">${{buildPatchGrid(flagsStart,cols)}}</div></div>`;
  if (flagsEnd)   html += `<div><div class="muted" style="margin-bottom:2px">end</div><div class="patch-grid" style="grid-template-columns:repeat(${{cols}},16px)">${{buildPatchGrid(flagsEnd,cols)}}</div></div>`;
  html += `</div></div>`;
  return html;
}}

function renderExamples(root, token) {{
  const examples = DATA.examples[String(token)] || DATA.examples[token] || [];
  root.innerHTML = `<h3>FSQ training samples for token #${{token}}</h3>`;
  if (!examples.length) {{
    root.innerHTML += `<div class="muted">No saved training examples for this token.</div>`;
    return;
  }}
  const strip = document.createElement("div");
  strip.className = "example-strip";
  for (const ex of examples) {{
    const box = document.createElement("div");
    box.className = "example";
    box.innerHTML = `<div><img src="${{ASSET + ex.start}}"><div class="muted">ep${{ex.episode_id}} f${{ex.frame_start}}</div></div>` +
                    `<div><img src="${{ASSET + ex.end}}"><div class="muted">f${{ex.frame_end}}</div></div>`;
    strip.appendChild(box);
  }}
  root.appendChild(strip);
}}

function selectToken(token) {{
  document.querySelectorAll(".skill-card").forEach(el => el.classList.toggle("active", Number(el.dataset.token) === token));
  document.querySelectorAll(".cube").forEach(svg => renderCube(svg, token));
  document.querySelectorAll(".examples").forEach(root => renderExamples(root, token));
}}

const app = document.getElementById("app");
for (const ep of DATA.episodes) {{
  const section = document.createElement("section");
  section.className = "episode";
  section.innerHTML = `<div class="episode-title">Episode ${{ep.episode_index}}</div>`;
  if (ep.timeline_graph) {{
    section.innerHTML += `<img class="timeline" src="${{ASSET + ep.timeline_graph}}">`;
  }}
  const row = document.createElement("div");
  row.className = "row";
  const cubeWrap = document.createElement("div");
  cubeWrap.className = "cube-wrap";
  cubeWrap.innerHTML = `<svg class="cube" viewBox="0 0 600 540"></svg>`;
  const skills = document.createElement("div");
  skills.className = "skills";
  for (const s of ep.skills) {{
    const card = document.createElement("div");
    card.className = "skill-card" + (DATA.chunk_size > 0 && s.skill_length < DATA.chunk_size ? " skill-card--short" : "");
    card.dataset.token = s.token;
    card.innerHTML = `<div class="skill-title">skill ${{s.skill_index + 1}} | token #${{s.token}} | [${{s.coord.join(", ")}}]</div>` +
      `<div class="muted">${{s.source}} · t=${{s.start_t}}→${{s.end_t}}</div>` +
      `<div class="pair"><img src="${{ASSET + s.start}}"><img src="${{ASSET + s.end}}"></div>` +
      renderPatchGrids(s.patch_flags_start, s.patch_flags_end) +
      `<img class="graph" src="${{ASSET + s.graph}}">`;
    card.addEventListener("click", () => selectToken(s.token));
    skills.appendChild(card);
  }}
  row.appendChild(cubeWrap);
  row.appendChild(skills);
  const examples = document.createElement("div");
  examples.className = "examples";
  section.appendChild(row);
  section.appendChild(examples);
  app.appendChild(section);
  renderCube(cubeWrap.querySelector("svg"), ep.skills[0]?.token ?? -1);
}}
selectToken(DATA.episodes[0]?.skills[0]?.token ?? 0);
</script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return str(html_path)


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int = 0,
    video_frame_stride: int = 1,
    video_fps: int | None = None,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    forced_skill_token_sequences: list[list[int]] | None = None,
    reference_skill_token_sequences: list[list[int]] | None = None,
    collect_skill_html: bool = False,
    task_description: str | None = None,
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
        task_description: The task's language prompt — rendered into a bottom bar of every episode
            video (the top bar is colored green/red by that episode's success).
        max_episodes_rendered: Maximum number of episodes to render into videos.
        videos_dir: Where to save rendered videos.
        return_episode_data: Whether to return episode data for online training. Incorporates the data into
            the "episodes" key of the returned dictionary.
        start_seed: The first seed to use for the first individual rollout. For all subsequent rollouts the
            seed is incremented by 1. If not provided, the environments are not manually seeded.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    if not isinstance(policy, PreTrainedPolicy):
        exc = ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        )
        try:
            from peft import PeftModel

            if not isinstance(policy, PeftModel):
                raise exc
        except ImportError:
            raise exc from None

    start = time.time()
    policy.eval()

    # Determine how many batched rollouts we need to get n_episodes. Note that if n_episodes is not evenly
    # divisible by env.num_envs we end up discarding some data in the last batch.
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    # Keep track of some metrics.
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    skill_plot_paths = []
    skill_timeline_paths = []
    skill_token_records: list[dict] = []
    skill_html_records: list[dict] = []
    threads = []  # for video saving threads
    n_episodes_rendered = 0  # for saving the correct number of videos
    render_frame_index = 0
    video_frame_stride = max(1, int(video_frame_stride))

    # Callback for visualization.
    def render_frame(env: gym.vector.VectorEnv):
        # noqa: B023
        nonlocal render_frame_index
        all_frames = None
        if collect_skill_html:
            if isinstance(env, gym.vector.SyncVectorEnv):
                all_frames = np.stack([sub_env.render() for sub_env in env.envs])
            elif isinstance(env, gym.vector.AsyncVectorEnv):
                all_frames = np.stack(env.call("render"))
            ep_html_frames.append(all_frames)  # noqa: B023

        if n_episodes_rendered >= max_episodes_rendered:
            render_frame_index += 1
            return
        if render_frame_index % video_frame_stride != 0:
            render_frame_index += 1
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if all_frames is not None:
            ep_frames.append(all_frames[:n_to_render_now])  # noqa: B023
        elif isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render_now)]))  # noqa: B023
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            # Here we must render all frames and discard any we don't need.
            ep_frames.append(np.stack(env.call("render")[:n_to_render_now]))
        render_frame_index += 1

    if max_episodes_rendered > 0:
        video_paths: list[str] = []

    if return_episode_data:
        episode_data: dict | None = None

    # we dont want progress bar when we use slurm, since it clutters the logs
    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        # Cache frames for rendering videos. Each item will be (b, h, w, c), and the list indexes the rollout
        # step.
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []
        if collect_skill_html:
            ep_html_frames: list[np.ndarray] = []
        render_frame_index = 0

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)
            )
        if forced_skill_token_sequences is not None:
            start = batch_ix * env.num_envs
            end = start + env.num_envs
            set_forced = getattr(policy, "set_forced_skill_token_sequences", None)
            if set_forced is None:
                raise ValueError("Policy does not support forced skill token sequences.")
            set_forced(forced_skill_token_sequences[start:end])
        if reference_skill_token_sequences is not None:
            start = batch_ix * env.num_envs
            end = start + env.num_envs
            set_reference = getattr(policy, "set_reference_skill_token_sequences", None)
            if set_reference is None:
                raise ValueError("Policy does not support reference skill token sequences.")
            set_reference(reference_skill_token_sequences[start:end])
        rollout_data = rollout(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seeds=list(seeds) if seeds else None,
            return_observations=return_episode_data,
            render_callback=render_frame if (max_episodes_rendered > 0 or collect_skill_html) else None,
        )

        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        n_steps = rollout_data["done"].shape[1]
        # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)
        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        # Extend metrics.
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())
        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.append(None)

        # FIXME: episode_data is either None or it doesn't exist
        if return_episode_data:
            this_episode_data = _compile_episode_data(
                rollout_data,
                done_indices,
                start_episode_index=batch_ix * env.num_envs,
                start_data_index=(0 if episode_data is None else (episode_data["index"][-1].item() + 1)),
                fps=env.unwrapped.metadata["render_fps"],
            )
            if episode_data is None:
                episode_data = this_episode_data
            else:
                # Some sanity checks to make sure we are correctly compiling the data.
                assert episode_data["episode_index"][-1] + 1 == this_episode_data["episode_index"][0]
                assert episode_data["index"][-1] + 1 == this_episode_data["index"][0]
                # Concatenate the episode data.
                episode_data = {k: torch.cat([episode_data[k], this_episode_data[k]]) for k in episode_data}

        # Maybe render video for visualization.
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)  # (b, t, *)
            for stacked_frames, done_index, ep_success in zip(
                batch_stacked_frames, done_indices.flatten().tolist(),
                batch_successes.flatten().tolist(), strict=False,
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                clip = _annotate_eval_video(   # top bar: green/red outcome; bottom bar: language prompt
                    stacked_frames[: done_index // video_frame_stride + 1],  # exclude auto-reset frame
                    bool(ep_success), task_description)
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        clip,
                        int(video_fps or max(1, env.unwrapped.metadata["render_fps"] // video_frame_stride)),
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1

        if collect_skill_html and len(ep_html_frames) > 0:
            batch_html_frames = np.stack(ep_html_frames, axis=1)  # (b, t, h, w, c)
            trace = []
            get_skill_trace = getattr(policy, "get_skill_trace", None)
            if get_skill_trace is not None:
                trace = get_skill_trace()
            gt_timeline: dict[int, list[dict]] = {}
            get_gt_timeline = getattr(policy, "get_gt_timeline", None)
            if get_gt_timeline is not None:
                gt_timeline = get_gt_timeline() or {}
            by_batch: dict[int, list[dict]] = defaultdict(list)
            for record in trace:
                by_batch[int(record.get("batch_index", 0))].append(record)
            for local_i, done_index in enumerate(done_indices.flatten().tolist()):
                episode_index = batch_ix * env.num_envs + local_i
                if episode_index >= n_episodes or local_i >= batch_html_frames.shape[0]:
                    continue
                max_frame = min(int(done_index) + 1, batch_html_frames.shape[1])
                skill_html_records.append(
                    {
                        "episode_index": episode_index,
                        "frames": batch_html_frames[local_i, :max_frame].copy(),
                        "trace": by_batch.get(local_i, []),
                        "gt_trace": gt_timeline.get(local_i, []),
                    }
                )

        progbar.set_postfix(
            {"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"}
        )

    # Wait till all video rendering threads are done.
    for thread in threads:
        thread.join()

    # Compile eval info.
    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
            }
            for i, (sum_reward, max_reward, success, seed) in enumerate(
                zip(
                    sum_rewards[:n_episodes],
                    max_rewards[:n_episodes],
                    all_successes[:n_episodes],
                    all_seeds[:n_episodes],
                    strict=True,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
    }

    if return_episode_data:
        info["episodes"] = episode_data

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths
    info["skill_plot_paths"] = skill_plot_paths
    info["skill_timeline_paths"] = skill_timeline_paths
    info["skill_token_records"] = skill_token_records
    info["skill_html_records"] = skill_html_records

    return info


def _compile_episode_data(
    rollout_data: dict, done_indices: Tensor, start_episode_index: int, start_data_index: int, fps: float
) -> dict:
    """Convenience function for `eval_policy(return_episode_data=True)`

    Compiles all the rollout data into a Hugging Face dataset.

    Similar logic is implemented when datasets are pushed to hub (see: `push_to_hub`).
    """
    ep_dicts = []
    total_frames = 0
    for ep_ix in range(rollout_data[ACTION].shape[0]):
        # + 2 to include the first done frame and the last observation frame.
        num_frames = done_indices[ep_ix].item() + 2
        total_frames += num_frames

        # Here we do `num_frames - 1` as we don't want to include the last observation frame just yet.
        ep_dict = {
            ACTION: rollout_data[ACTION][ep_ix, : num_frames - 1],
            "episode_index": torch.tensor([start_episode_index + ep_ix] * (num_frames - 1)),
            "frame_index": torch.arange(0, num_frames - 1, 1),
            "timestamp": torch.arange(0, num_frames - 1, 1) / fps,
            DONE: rollout_data["done"][ep_ix, : num_frames - 1],
            "next.success": rollout_data["success"][ep_ix, : num_frames - 1],
            REWARD: rollout_data["reward"][ep_ix, : num_frames - 1].type(torch.float32),
        }

        # For the last observation frame, all other keys will just be copy padded.
        for k in ep_dict:
            ep_dict[k] = torch.cat([ep_dict[k], ep_dict[k][-1:]])

        for key in rollout_data[OBS_STR]:
            ep_dict[key] = rollout_data[OBS_STR][key][ep_ix, :num_frames]

        ep_dicts.append(ep_dict)

    data_dict = {}
    for key in ep_dicts[0]:
        data_dict[key] = torch.cat([x[key] for x in ep_dicts])

    data_dict["index"] = torch.arange(start_data_index, start_data_index + total_frames, 1)

    return data_dict


def _skill_token_table(policy: PreTrainedPolicy) -> torch.Tensor | None:
    """FSQ codebook table for the similarity metrics: every flat code's grid coordinate z_q
    (little-endian strides, z_q = level_id - half — the codebook's own geometry, the same
    representation the policies' skill_proj conditioning consumes)."""
    levels = _skill_fsq_levels(policy)
    n_tokens = int(np.prod(levels))
    if n_tokens <= 1:
        return None
    half = torch.tensor([(lv - 1) / 2.0 for lv in levels], dtype=torch.float32)
    coords = torch.tensor(
        [_token_to_fsq_coord(t, levels) for t in range(n_tokens)], dtype=torch.float32
    )
    return coords - half[None, :]


def _save_skill_trace_plots(
    policy: PreTrainedPolicy, output_dir: Path, episode_index: int, episode_steps: int | None = None
) -> tuple[list[str], list[str], list[dict]]:
    get_skill_trace = getattr(policy, "get_skill_trace", None)
    if get_skill_trace is None:
        return [], [], []

    trace = get_skill_trace()
    if not trace:
        return [], [], []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logging.warning(f"Could not save SkillVLA skill plots because matplotlib failed to import: {exc}")
        return [], [], []

    def _merge_action_chunks(chunks: list[dict], length: int, dim: int) -> np.ndarray | None:
        if not chunks:
            return None

        merged = np.full((length, dim), np.nan, dtype=np.float32)
        for chunk in chunks:
            start = int(chunk.get("start", 0))
            actions = np.asarray(chunk.get("actions", []), dtype=np.float32)
            if actions.ndim != 2 or start >= length:
                continue
            end = min(start + actions.shape[0], length)
            if end <= start:
                continue
            merged[start:end, : min(dim, actions.shape[1])] = actions[: end - start, :dim]

        if np.isnan(merged).all():
            return None
        return merged

    def _codebook_similarity(pred_token, label_token) -> dict[str, float | int]:
        default = {
            "codebook_cosine": np.nan,
            "codebook_l2": np.nan,
            "codebook_label_neighbor_rank": -1,
        }
        if pred_token is None or label_token is None:
            return default
        codebook = _skill_token_table(policy)
        if codebook is None or codebook.numel() == 0:
            return default
        pred = int(pred_token)
        label = int(label_token)
        if pred < 0 or label < 0 or pred >= codebook.shape[0] or label >= codebook.shape[0]:
            return default
        cb = codebook
        pred_vec = cb[pred]
        label_vec = cb[label]
        dists = torch.linalg.vector_norm(cb - label_vec[None, :], dim=-1)
        rank = int((dists < dists[pred]).sum().item() + 1)
        return {
            "codebook_cosine": float(F.cosine_similarity(pred_vec[None], label_vec[None]).item()),
            "codebook_l2": float(torch.linalg.vector_norm(pred_vec - label_vec).item()),
            "codebook_label_neighbor_rank": rank,
        }

    action_names = ["x", "y", "z", "r", "p", "yaw", "gripper"]
    skill_dir = output_dir / "skill_plots" / f"episode_{episode_index:04d}"
    timeline_dir = output_dir / "skill_timelines" / f"episode_{episode_index:04d}"
    skill_dir.mkdir(parents=True, exist_ok=True)
    timeline_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    timeline_paths = []
    skill_start_timesteps = []
    token_records = []
    length_records = []
    for record in trace:
        skill_index = int(record.get("skill_index", len(saved_paths)))
        batch_index = int(record.get("batch_index", 0))
        episode_timestep = int(record.get("episode_timestep", 0))
        codebook_token = record.get("codebook_token")
        raw_actions_value = record.get("raw_actions")
        raw_actions = None if raw_actions_value is None else np.asarray(raw_actions_value)
        skill_length = int(record.get("length", 0 if raw_actions is None else raw_actions.shape[0]))
        dataset_skill_length = record.get("dataset_skill_length")
        dataset_prior_length = record.get("dataset_prior_length")
        label_prior_length = record.get("label_prior_length")
        label_codebook_token = record.get("label_codebook_token")
        token_match = record.get("token_match")
        similarity = _codebook_similarity(codebook_token, label_codebook_token)
        skill_source = record.get("skill_source", "unknown")
        has_label_records = bool(record.get("has_label_records", False))
        has_label_prior = bool(record.get("has_label_prior", False))
        end_signal_timestep = record.get("end_signal_timestep")
        end_signal_prob = record.get("end_signal_prob")
        skill_start_timesteps.append(
            (
                episode_timestep,
                skill_index,
                codebook_token,
                label_codebook_token,
                token_match,
                end_signal_timestep,
                end_signal_prob,
                skill_length,
                similarity["codebook_cosine"],
            )
        )
        token_records.append({
            "episode": episode_index + batch_index,
            "skill_idx": skill_index + 1,
            "episode_timestep": episode_timestep,
            "skill_source": skill_source,
            "codebook_token": codebook_token if codebook_token is not None else -1,
            "label_codebook_token": label_codebook_token if label_codebook_token is not None else -1,
            "token_match": token_match if token_match is not None else False,
            "has_label_records": has_label_records,
            "has_label_prior": has_label_prior,
            "skill_length": skill_length,
            "dataset_skill_length": dataset_skill_length if dataset_skill_length is not None else -1,
            "dataset_prior_length": dataset_prior_length if dataset_prior_length is not None else -1,
            "label_prior_length": label_prior_length if label_prior_length is not None else -1,
            "end_signal_timestep": end_signal_timestep if end_signal_timestep is not None else -1,
            "end_signal_prob": end_signal_prob if end_signal_prob is not None else -1.0,
            "codebook_cosine": similarity["codebook_cosine"],
            "codebook_l2": similarity["codebook_l2"],
            "codebook_label_neighbor_rank": similarity["codebook_label_neighbor_rank"],
            "eval_minus_dataset_skill_length": (
                skill_length - int(dataset_skill_length) if dataset_skill_length is not None else None
            ),
            "eval_minus_dataset_prior_length": (
                skill_length - int(dataset_prior_length) if dataset_prior_length is not None else None
            ),
            "pred_minus_label_prior_length": (
                skill_length - int(label_prior_length) if label_prior_length is not None else None
            ),
        })
        length_records.append(
            {
                "skill_idx": skill_index + 1,
                "pred_codebook_token": codebook_token,
                "label_codebook_token": label_codebook_token,
                "eval_prior_length": skill_length,
                "dataset_skill_length": dataset_skill_length,
                "dataset_prior_length": dataset_prior_length,
                "label_prior_length": label_prior_length,
                "end_signal_timestep": end_signal_timestep,
                "end_signal_prob": end_signal_prob,
                **similarity,
            }
        )
        if raw_actions is None:
            continue
        timesteps = np.arange(raw_actions.shape[0])
        dim = raw_actions.shape[1]
        labels = action_names[:dim] + [f"a{i}" for i in range(len(action_names), dim)]
        dataset_prior = record.get("dataset_prior_raw_actions")
        dataset_prior = None if dataset_prior is None else np.asarray(dataset_prior)
        label_prior = record.get("label_prior_raw_actions")
        label_prior = None if label_prior is None else np.asarray(label_prior)
        expert_actions = _merge_action_chunks(
            record.get("expert_raw_actions", []),
            length=raw_actions.shape[0],
            dim=dim,
        )

        fig, axes = plt.subplots(dim, 1, figsize=(12, max(2.0 * dim, 7.0)), sharex=True)
        axes = np.atleast_1d(axes)
        for action_index in range(dim):
            axes[action_index].plot(
                timesteps,
                raw_actions[:, action_index],
                linewidth=1.5,
                label="pred-token sim-start VQAE prior",
            )
            if label_prior is not None and label_prior.ndim == 2:
                axes[action_index].plot(
                    np.arange(label_prior.shape[0]),
                    label_prior[:, action_index],
                    linewidth=1.2,
                    linestyle="-.",
                    alpha=0.9,
                    label="label-token sim-start VQAE prior",
                )
            if dataset_prior is not None and dataset_prior.ndim == 2:
                axes[action_index].plot(
                    np.arange(dataset_prior.shape[0]),
                    dataset_prior[:, action_index],
                    linewidth=1.2,
                    linestyle=":",
                    alpha=0.9,
                    label="dataset-start VQAE prior",
                )
            if expert_actions is not None:
                axes[action_index].plot(
                    timesteps,
                    expert_actions[:, action_index],
                    linewidth=1.3,
                    linestyle="--",
                    alpha=0.9,
                    label="action expert",
                )
            axes[action_index].set_ylabel(labels[action_index])
            axes[action_index].grid(True, alpha=0.25)
            if action_index == 0:
                axes[action_index].legend(loc="best")

        token_str = f" [token #{codebook_token}]" if codebook_token is not None else ""
        label_token = record.get("label_codebook_token")
        if label_token is not None:
            token_str += f" / label #{label_token}"
        fig.suptitle(f"Skill {skill_index + 1}{token_str} raw actions: VQAE prior vs action expert")
        axes[-1].set_xlabel("skill timestep")
        fig.tight_layout()

        plot_path = skill_dir / f"skill_{skill_index + 1:03d}_batch_{batch_index:02d}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        saved_paths.append(str(plot_path))

    if skill_start_timesteps:
        skill_start_timesteps = sorted(skill_start_timesteps)
        x_values = [step for step, *_rest in skill_start_timesteps]
        y_values = [skill_index + 1 for _, skill_index, *_rest in skill_start_timesteps]
        if episode_steps is not None and episode_steps > 0:
            x_values = [*x_values, episode_steps]
            y_values = [*y_values, y_values[-1]]

        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.step(x_values, y_values, where="post", linewidth=2.0)
        for (
            timestep,
            skill_index,
            codebook_token,
            label_token,
            token_match,
            end_timestep,
            end_prob,
            skill_length,
            codebook_cosine,
        ) in skill_start_timesteps:
            ax.axvline(timestep, color="tab:red", alpha=0.3, linewidth=1.0)
            if end_timestep is not None:
                ax.axvline(int(end_timestep), color="tab:green", alpha=0.35, linewidth=1.0, linestyle="--")
            if label_token is not None:
                sim_label = "" if np.isnan(codebook_cosine) else f"\ncos={codebook_cosine:.2f}"
                token_label = f"\np#{codebook_token}/l#{label_token}{sim_label}"
            else:
                token_label = f"\np#{codebook_token}" if codebook_token is not None else ""
            if end_timestep is not None:
                token_label += f"\nend@{int(end_timestep)}"
            ax.text(
                timestep,
                skill_index + 1,
                f"S{skill_index + 1}{token_label}",
                rotation=90,
                va="bottom",
                ha="right",
                fontsize="small",
            )
        ax.set_title("Skill calls over task timesteps")
        ax.set_xlabel("task timestep")
        ax.set_ylabel("active skill")
        ax.set_yticks([skill_index + 1 for _, skill_index, *_rest in skill_start_timesteps])
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()

        timeline_path = timeline_dir / "skill_timeline.png"
        fig.savefig(timeline_path, dpi=150)
        plt.close(fig)
        timeline_paths.append(str(timeline_path))

        pred_tokens = [
            -1 if pred_token is None else int(pred_token)
            for _step, _skill_index, pred_token, _label_token, _match, *_rest in skill_start_timesteps
        ]
        label_tokens = [
            np.nan if label_token is None else int(label_token)
            for _step, _skill_index, _pred_token, label_token, _match, *_rest in skill_start_timesteps
        ]
        fig, ax = plt.subplots(figsize=(max(8, len(skill_start_timesteps) * 1.1), 4.0))
        skill_x = np.arange(1, len(skill_start_timesteps) + 1)
        ax.plot(skill_x, pred_tokens, marker="o", linewidth=1.8, label="pred token")
        ax.plot(skill_x, label_tokens, marker="x", linewidth=1.8, linestyle="--", label="label token")
        for i, (_step, _skill_index, pred_token, label_token, match, *_rest) in enumerate(skill_start_timesteps, start=1):
            label = f"p{pred_token}/l{label_token}" if label_token is not None else f"p{pred_token}/l?"
            ax.text(i, pred_tokens[i - 1], label, fontsize="small", ha="center", va="bottom")
            if label_token is not None and not bool(match):
                ax.axvspan(i - 0.35, i + 0.35, color="tab:red", alpha=0.08)
        ax.set_title("Predicted vs label skill tokens")
        ax.set_xlabel("skill")
        ax.set_ylabel("codebook token")
        ax.set_xticks(skill_x)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()

        token_compare_path = timeline_dir / "pred_vs_label_tokens.png"
        fig.savefig(token_compare_path, dpi=150)
        plt.close(fig)
        timeline_paths.append(str(token_compare_path))

        cosine_values = [
            float(cosine)
            for *_prefix, cosine in skill_start_timesteps
            if not np.isnan(cosine)
        ]
        if cosine_values:
            similarity_x = [
                i
                for i, (*_prefix, cosine) in enumerate(skill_start_timesteps, start=1)
                if not np.isnan(cosine)
            ]
            fig, ax = plt.subplots(figsize=(max(8, len(skill_start_timesteps) * 1.1), 4.0))
            ax.plot(similarity_x, cosine_values, marker="o", linewidth=1.8)
            ax.set_ylim(-1.05, 1.05)
            ax.set_title("Predicted vs label skill embedding cosine")
            ax.set_xlabel("skill order")
            ax.set_ylabel("codebook cosine similarity")
            ax.set_xticks(np.arange(1, len(skill_start_timesteps) + 1))
            ax.grid(True, alpha=0.25)
            fig.tight_layout()

            similarity_path = timeline_dir / "pred_vs_label_codebook_similarity.png"
            fig.savefig(similarity_path, dpi=150)
            plt.close(fig)
            timeline_paths.append(str(similarity_path))

    return saved_paths, timeline_paths, token_records


def _as_scalar_int(value: Any) -> int:
    arr = np.asarray(value)
    return int(arr.reshape(-1)[0])


def _as_int_list(value: Any) -> list[int]:
    arr = np.asarray(value)
    return [int(x) for x in arr.reshape(-1).tolist()]


def _load_label_skill_token_sequences(
    dataset_dir: str | Path,
    *,
    episode_offset: int = 0,
    n_episodes: int | None = None,
    task_ids: set[int] | None = None,
) -> dict[int, list[list[dict]]]:
    """Load real FSQ skill-token records grouped by LIBERO task_index."""
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("pandas is required to load label skill tokens from a LeRobot dataset.") from exc

    dataset_dir = Path(dataset_dir)
    data_files = sorted((dataset_dir / "data").glob("**/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet data files found under {dataset_dir / 'data'}")

    columns = [
        "episode_index",
        "frame_index",
        "task_index",
        "skill_sequence",
        "skill_length_sequence",
        "skill_sequence_len",
    ]
    frames = [pd.read_parquet(path, columns=columns) for path in data_files]
    df = pd.concat(frames, ignore_index=True).sort_values(["episode_index", "frame_index"])
    if task_ids is not None:
        df = df[df["task_index"].map(_as_scalar_int).isin(task_ids)]

    episode_skills: dict[int, tuple[int, list[dict]]] = {}
    for episode_index, ep_df in df.groupby("episode_index", sort=True):
        task_index = _as_scalar_int(ep_df["task_index"].iloc[0])
        row = ep_df.iloc[0]
        seq = _as_int_list(row["skill_sequence"])
        lengths = _as_int_list(row["skill_length_sequence"])
        seq_len = _as_scalar_int(row["skill_sequence_len"])
        first_frame = _as_scalar_int(ep_df["frame_index"].iloc[0])
        real_tokens = seq[1 : max(1, seq_len - 1)]  # drop BOS and EOS/PAD
        real_lengths = lengths[1 : 1 + len(real_tokens)]

        records: list[dict] = []
        cursor = first_frame
        for token, skill_length in zip(real_tokens, real_lengths, strict=False):
            skill_length = int(skill_length)
            frame_start = cursor
            frame_end = cursor + max(1, skill_length)
            records.append(
                {
                    "token": int(token),
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "skill_length": int(frame_end - frame_start),
                }
            )
            cursor = frame_end
        episode_skills[int(episode_index)] = (task_index, records)

    by_task: dict[int, list[tuple[int, list[dict]]]] = defaultdict(list)
    for episode_index, (task_index, records) in episode_skills.items():
        by_task[task_index].append((episode_index, records))

    out: dict[int, list[list[dict]]] = {}
    for task_index, episodes in by_task.items():
        selected = [records for _episode_index, records in sorted(episodes)]
        if episode_offset:
            selected = selected[episode_offset:]
        if n_episodes is not None:
            selected = selected[:n_episodes]
        out[task_index] = selected
    return out


def _load_gt_skill_sequences(
    dataset_dir: str | Path,
    *,
    n_episodes: int | None = None,
    task_ids: set[int] | None = None,
) -> dict[int, list[list[dict]]]:
    """GT skill sequences for Stage-2 oracle eval, grouped by LIBERO task_index.

    Reads the (no-BOS) ``skill_sequence`` = [skill0 .. skill_{N-1}, EOS, PAD...] and the matching
    ``skill_length_sequence`` (GT demo frames per skill). Returns task_index -> per-episode list of
    ``[{"token", "gt_length"}]`` (consumed by ``policy.set_forced_skill_token_sequences``).
    """
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("pandas is required to load GT skill sequences.") from exc

    dataset_dir = Path(dataset_dir)
    data_files = sorted((dataset_dir / "data").glob("**/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet data files found under {dataset_dir / 'data'}")
    columns = ["episode_index", "frame_index", "task_index", "skill_sequence",
               "skill_length_sequence", "skill_sequence_len"]
    df = pd.concat([pd.read_parquet(p, columns=columns) for p in data_files], ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"])
    if task_ids is not None:
        df = df[df["task_index"].map(_as_scalar_int).isin(task_ids)]

    # Real FSQ codes are < skill_num_embeddings; EOS/PAD are >= it. Filter specials by VALUE
    # (scheme-agnostic — works with or without a leading BOS), matching stage1's eval_oracle loader.
    info_path = dataset_dir / "meta" / "info.json"
    num_emb = json.loads(info_path.read_text()).get("skill_num_embeddings") if info_path.is_file() else None

    by_task: dict[int, list[tuple[int, list[dict]]]] = defaultdict(list)
    for episode_index, ep_df in df.groupby("episode_index", sort=True):
        row = ep_df.iloc[0]
        task_index = _as_scalar_int(row["task_index"])
        seq = _as_int_list(row["skill_sequence"])
        lengths = _as_int_list(row["skill_length_sequence"])
        if num_emb is not None:
            idxs = [i for i in range(len(seq)) if seq[i] < int(num_emb)]
        else:  # fallback (no info.json): no-BOS schema → real skills are the first (seq_len - 1)
            idxs = list(range(max(0, _as_scalar_int(row["skill_sequence_len"]) - 1)))
        records = [
            {"token": int(seq[i]), "gt_length": int(lengths[i] if i < len(lengths) else 0)}
            for i in idxs
        ]
        if records:  # skip skill-less episodes (no real skills)
            by_task[task_index].append((int(episode_index), records))

    out: dict[int, list[list[dict]]] = {}
    for task_index, episodes in by_task.items():
        selected = [records for _ep, records in sorted(episodes)]
        out[task_index] = selected if n_episodes is None else selected[:n_episodes]
    return out


def _episode_exact_override(
    envs: dict,
    init_states_path: str | Path,
    suite_name: str,
    *,
    gt_skill_dataset_dir: str | Path | None = None,
    n_episodes: int | None = None,
) -> dict[tuple[str, int], list[list[dict]]] | None:
    """EPISODE-EXACT eval (ported from stage1_eval): replace each task env's LIBERO built-in init
    states with the matched DATASET episodes' MuJoCo init states (``eval_init_states.npz``, built by
    stage1_eval/oracle_matching; ordered by episode_index), so every rollout reproduces a specific
    dataset episode's scene. Tasks with no matched episode are dropped (closed). Needs SyncVectorEnv
    (eval.use_async_envs=false).

    With ``gt_skill_dataset_dir`` (use_gt_skill): joins the skillvla dataset's GT skill sequences and
    returns {(task_group, task_id): [skills_ep0, ...]} aligned to the SAME episode order as the
    injected init states (index i pairs with init state i) — i.e. each rollout gets THAT scene's own
    GT sequence, not just a same-task one. Returns None when no GT dir is given (scene override only).
    """
    npz = np.load(str(init_states_path), allow_pickle=True)
    # scene_file (minus '_demo.hdf5') == the LIBERO task .name → a unique task_id.
    from libero.libero import benchmark  # noqa: PLC0415

    suite = benchmark.get_benchmark_dict()[suite_name]()
    name_to_id = {str(t.name): i for i, t in enumerate(suite.tasks)}
    per_task: dict[int, list[dict]] = defaultdict(list)
    for ep, st, sf in zip(npz["episode_index"], npz["init_states"], npz["scene_file"], strict=True):
        sf = str(sf)
        task_name = sf[: -len("_demo.hdf5")] if sf.endswith("_demo.hdf5") else sf
        tid = name_to_id.get(task_name)
        if tid is not None:
            per_task[tid].append({"episode_index": int(ep), "init_state": np.asarray(st, np.float64)})
    for tid in per_task:
        per_task[tid].sort(key=lambda r: r["episode_index"])

    # Per-EPISODE GT skills (value-filtered specials, mirroring stage1's eval_oracle loader).
    skills_by_ep: dict[int, list[dict]] | None = None
    if gt_skill_dataset_dir:
        import pandas as pd  # noqa: PLC0415

        gt_dir = Path(gt_skill_dataset_dir)
        info_path = gt_dir / "meta" / "info.json"
        num_emb = json.loads(info_path.read_text()).get("skill_num_embeddings") if info_path.is_file() else None
        data_files = sorted((gt_dir / "data").glob("**/*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"No parquet data files found under {gt_dir / 'data'}")
        cols = ["episode_index", "frame_index", "skill_sequence", "skill_length_sequence"]
        df = pd.concat([pd.read_parquet(p, columns=cols) for p in data_files], ignore_index=True)
        skills_by_ep = {}
        for episode_index, ep_df in df.groupby("episode_index", sort=True):
            row = ep_df.sort_values("frame_index").iloc[0]
            seq, lens = _as_int_list(row["skill_sequence"]), _as_int_list(row["skill_length_sequence"])
            skills = [{"token": int(seq[i]), "gt_length": int(lens[i] if i < len(lens) else 0)}
                      for i in range(len(seq)) if num_emb is None or seq[i] < int(num_emb)]
            if skills:
                skills_by_ep[int(episode_index)] = skills

    forced: dict[tuple[str, int], list[list[dict]]] | None = {} if skills_by_ep is not None else None
    matched = 0
    for task_group, group in envs.items():
        for task_id in list(group.keys()):
            records = per_task.get(int(task_id), [])
            if skills_by_ep is not None:  # keep only episodes that ALSO have a GT sequence (aligned pairing)
                records = [r for r in records if r["episode_index"] in skills_by_ep]
            if not records:
                # No matched dataset episode for this task (e.g. the npz is for a different suite —
                # libero_goal / libero_10 have none): KEEP the task and let it run the ORIGINAL
                # seed-based eval (default LIBERO init states) instead of dropping it.
                logging.warning("episode-exact: no matched episodes for task_id=%s — "
                                "falling back to seed-based reset for it.", task_id)
                continue
            if n_episodes is not None and len(records) < n_episodes:
                logging.warning("episode-exact: task_id=%s has only %d matched episodes (< n_episodes=%d).",
                                task_id, len(records), n_episodes)
            subs = getattr(group[task_id], "envs", None)
            if subs is None:
                raise RuntimeError("Episode-exact eval needs SyncVectorEnv (set eval.use_async_envs=false).")
            init_arr = np.stack([r["init_state"] for r in records]).astype(np.float64)
            for sub in subs:
                base = sub.unwrapped
                base.init_states = True           # ensure reset() takes the set_init_state path
                base._init_states = init_arr      # env indexes by init_state_id (= global episode index)
            if forced is not None:
                forced[(task_group, int(task_id))] = [skills_by_ep[r["episode_index"]] for r in records]
            matched += 1
    total = sum(len(g) for g in envs.values())
    if matched == 0:
        # No task matched (e.g. the npz is for another suite) → episode-exact is a no-op: no scene was
        # overridden. Return None so the caller falls through to the ORIGINAL eval paths (seed-based
        # reset; non-aligned GT loading under use_gt_skill), exactly as if init_states_path were unset.
        logging.warning("episode-exact: 0/%d tasks matched %s — running the ORIGINAL eval.",
                        total, str(init_states_path))
        return None
    logging.info("episode-exact: %d/%d tasks matched (unmatched → seed-based reset).", matched, total)
    return forced


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    supported = {"skill_vla"}
    if cfg.policy is None or cfg.policy.type not in supported:
        policy_type = None if cfg.policy is None else cfg.policy.type
        raise ValueError(
            "lerobot_skillvla_eval is reserved for SkillVLA checkpoints "
            f"(expected one of {sorted(supported)}, got {policy_type!r})."
        )

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making environment.")
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )

    # EPISODE-EXACT eval (opt-in): reset every rollout to a matched DATASET episode's MuJoCo init
    # state; with use_gt_skill the GT sequences returned here are ALIGNED to those same episodes.
    ep_exact_forced = None
    episode_exact = bool(getattr(cfg.eval, "init_states_path", None))
    if episode_exact:
        # Unmatched tasks (e.g. a libero_goal/libero_10 suite the npz doesn't cover) are NOT dropped —
        # they fall back to the original seed-based eval. _episode_exact_override logs the match count.
        ep_exact_forced = _episode_exact_override(
            envs, cfg.eval.init_states_path, cfg.env.task,
            gt_skill_dataset_dir=(getattr(cfg.policy, "gt_skill_dataset_dir", None)
                                  if getattr(cfg.policy, "use_gt_skill", False) else None),
            n_episodes=cfg.eval.n_episodes)

    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )

    policy.eval()

    forced_skill_token_sequences_by_task = None
    reference_skill_token_sequences_by_task = None
    # Old skill-predictor "label skill tokens" eval (forcing/comparing GT token sequences) — removed in
    # the Stage-2 redesign (the VLM predicts skills). getattr keeps the new config (which lacks these
    # fields) from crashing; the whole block is skipped for Stage-2.
    needs_label_sequences = getattr(cfg.policy, "use_label_skill_tokens_eval", False) or getattr(
        cfg.policy, "compare_label_skill_tokens_eval", False
    )
    if needs_label_sequences:
        if not cfg.policy.label_skill_dataset_dir:
            raise ValueError(
                "--policy.label_skill_dataset_dir is required when "
                "--policy.use_label_skill_tokens_eval=true or "
                "--policy.compare_label_skill_tokens_eval=true"
            )
        label_sequences = _load_label_skill_token_sequences(
            cfg.policy.label_skill_dataset_dir,
            episode_offset=cfg.policy.label_skill_episode_offset,
            n_episodes=cfg.eval.n_episodes,
            task_ids={task_id for task_map in envs.values() for task_id in task_map},
        )
        target = {} if cfg.policy.use_label_skill_tokens_eval else None
        reference_target = {} if cfg.policy.compare_label_skill_tokens_eval else None
        for task_group, task_map in envs.items():
            for task_id in task_map:
                sequences = label_sequences.get(task_id)
                if sequences is None or len(sequences) < cfg.eval.n_episodes:
                    raise ValueError(
                        f"Not enough label skill-token episodes for task_id={task_id}: "
                        f"found {0 if sequences is None else len(sequences)}, need {cfg.eval.n_episodes}."
                    )
                if target is not None:
                    target[(task_group, task_id)] = sequences[: cfg.eval.n_episodes]
                if reference_target is not None:
                    reference_target[(task_group, task_id)] = sequences[: cfg.eval.n_episodes]
        forced_skill_token_sequences_by_task = target
        reference_skill_token_sequences_by_task = reference_target
        logging.info(
            "Using training-set label skill tokens for eval: "
            f"dataset={cfg.policy.label_skill_dataset_dir}, "
            f"episode_offset={cfg.policy.label_skill_episode_offset}, "
            f"force_tokens={cfg.policy.use_label_skill_tokens_eval}, "
            f"compare_tokens={cfg.policy.compare_label_skill_tokens_eval}"
        )

    # Stage-2 oracle eval: teacher-force the dataset's GT skill sequence (per task) into the policy's
    # cond-encoder (set_forced_skill_token_sequences); skill_advance_mode picks GT vs terminator timing.
    if getattr(cfg.policy, "use_gt_skill", False):
        gt_dir = getattr(cfg.policy, "gt_skill_dataset_dir", None)
        if not gt_dir:
            raise ValueError("--policy.gt_skill_dataset_dir is required when --policy.use_gt_skill=true")
        if ep_exact_forced is not None:
            # EPISODE-EXACT: sequences already aligned to the injected init states (index i ↔ scene i).
            forced_skill_token_sequences_by_task = {}
            for task_group, task_map in envs.items():
                for task_id in task_map:
                    seqs = ep_exact_forced.get((task_group, task_id))
                    if not seqs or len(seqs) < cfg.eval.n_episodes:
                        raise ValueError(
                            f"Not enough EPISODE-EXACT GT episodes for task_id={task_id}: "
                            f"found {0 if not seqs else len(seqs)}, need {cfg.eval.n_episodes} "
                            f"(npz={cfg.eval.init_states_path}, dir={gt_dir})."
                        )
                    forced_skill_token_sequences_by_task[(task_group, task_id)] = seqs[: cfg.eval.n_episodes]
            logging.info(
                "Oracle eval: EPISODE-EXACT GT skills (scene-aligned) | advance_mode=%s | tasks=%d",
                getattr(cfg.policy, "skill_advance_mode", "terminator"),
                len(forced_skill_token_sequences_by_task),
            )
        else:
            eval_task_ids = {task_id for task_map in envs.values() for task_id in task_map}
            gt_by_task = _load_gt_skill_sequences(gt_dir, n_episodes=cfg.eval.n_episodes, task_ids=eval_task_ids)
            forced_skill_token_sequences_by_task = {}
            for task_group, task_map in envs.items():
                for task_id in task_map:
                    seqs = gt_by_task.get(task_id)
                    if not seqs or len(seqs) < cfg.eval.n_episodes:
                        raise ValueError(
                            f"Not enough GT skill episodes for task_id={task_id}: "
                            f"found {0 if not seqs else len(seqs)}, need {cfg.eval.n_episodes} (dir={gt_dir})."
                        )
                    forced_skill_token_sequences_by_task[(task_group, task_id)] = seqs[: cfg.eval.n_episodes]
            logging.info(
                "Oracle eval: GT skills from %s | advance_mode=%s | tasks=%d",
                gt_dir, getattr(cfg.policy, "skill_advance_mode", "terminator"),
                len(forced_skill_token_sequences_by_task),
            )

    # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    # Create environment-specific preprocessor and postprocessor (e.g., for LIBERO environments)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
    task_id_to_desc = _libero_task_descriptions(cfg.env.task)

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=cfg.eval.max_videos_per_task,
            video_frame_stride=cfg.eval.video_frame_stride,
            video_fps=cfg.eval.video_fps,
            videos_dir=Path(cfg.output_dir) / "videos",
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
            forced_skill_token_sequences_by_task=forced_skill_token_sequences_by_task,
            reference_skill_token_sequences_by_task=reference_skill_token_sequences_by_task,
            skill_html_dir=(Path(cfg.output_dir) / "skill_html") if cfg.eval.skill_html else None,
            skill_html_train_samples=cfg.eval.skill_html_train_samples,
            skill_html_skill_latents_path=cfg.eval.skill_html_skill_latents_path,
            skill_html_raw_dataset_dir=cfg.eval.skill_html_raw_dataset_dir,
            skill_html_image_key=cfg.eval.skill_html_image_key,
            task_descriptions=task_id_to_desc,
            on_task_done_cmd=getattr(cfg.eval, "on_task_done_cmd", None),
        )
        print("Overall Aggregated Metrics:")
        print(info["overall"])

        # Print per-suite stats
        for task_group, task_group_info in info.items():
            print(f"\nAggregated Metrics for {task_group}:")
            print(task_group_info)
    # Close all vec envs
    close_envs(envs)

    # Save info. Task-split submissions (submit_eval.sh eval_num_gpus>1) share ONE output_dir — suffix
    # the summary per chunk (TASK_TAG, e.g. "t0-4") so concurrent jobs don't clobber each other's json.
    _tag = os.environ.get("TASK_TAG", "").strip()
    with open(Path(cfg.output_dir) / (f"eval_info_{_tag}.json" if _tag else "eval_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # Generate index.html for Netlify / static hosting
    if cfg.eval.skill_html:
        try:
            index_path = _write_index_html(
                output_dir=Path(cfg.output_dir),
                per_task_infos=info.get("per_task", []),
                task_descriptions=task_id_to_desc,
                job_name=getattr(cfg, "job_name", ""),
            )
            logging.info("Generated index.html: %s", index_path)
        except Exception as exc:
            logging.warning("Failed to write index.html: %s", exc)

    # Log to wandb if enabled
    wandb_project = getattr(cfg, "wandb_project", None)
    if wandb_project:
        try:
            import wandb

            policy_path = str(cfg.policy.pretrained_path) if cfg.policy and cfg.policy.pretrained_path else "unknown"
            wandb.init(
                project=wandb_project,
                name=cfg.job_name,
                config={
                    "policy_path": policy_path,
                    "env": cfg.env.type,
                    "n_episodes": cfg.eval.n_episodes,
                },
            )

            def _success_to_float(value) -> float:
                if hasattr(value, "item"):
                    value = value.item()
                return float(bool(value))

            task_labels: list[str] = []
            task_success_rates: list[float] = []
            for task_info in sorted(info.get("per_task", []), key=lambda t: t.get("task_id", 0)):
                task_id = task_info.get("task_id", 0)
                successes = task_info.get("metrics", {}).get("successes", [])
                success_values = [_success_to_float(s) for s in successes]
                success_rate = float(np.mean(success_values)) if success_values else float("nan")
                task_labels.append(f"task{int(task_id):02d}")
                task_success_rates.append(success_rate)
            if task_labels:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                height = max(2.8, 0.38 * len(task_labels) + 0.9)
                fig, ax = plt.subplots(figsize=(8.0, height))
                y = np.arange(len(task_labels))
                values = np.nan_to_num(np.asarray(task_success_rates, dtype=np.float32), nan=0.0)
                ax.barh(y, values, color="#4C78A8")
                ax.set_yticks(y)
                ax.set_yticklabels(task_labels)
                ax.invert_yaxis()
                ax.set_xlim(0.0, 1.0)
                ax.set_xlabel("success rate")
                ax.grid(axis="x", alpha=0.25)
                for yi, value in zip(y, values, strict=False):
                    ax.text(min(value + 0.02, 0.98), yi, f"{value:.2f}", va="center", fontsize=8)
                fig.tight_layout()
                # task-split chunks suffix their partial chart (the merge step regenerates the full one)
                chart_path = Path(cfg.output_dir) / (f"task_success_rates_{_tag}.png" if _tag else "task_success_rates.png")
                fig.savefig(chart_path, dpi=160)
                plt.close(fig)
                wandb.log({"charts/task_success_rate": wandb.Image(str(chart_path))})

            # Log videos separately (wandb.Video objects can't be batched with bar charts cleanly)
            fps = cfg.env.fps if hasattr(cfg.env, "fps") else 20
            video_log: dict = {}
            for task_info in info.get("per_task", []):
                task_id = task_info.get("task_id", 0)
                task_group = task_info.get("task_group", "unknown")
                desc = task_id_to_desc.get(task_id, f"task_{task_id}")
                label = f"task{task_id:02d}: {desc}"
                for ep_idx, video_path in enumerate(task_info.get("metrics", {}).get("video_paths", [])):
                    video_path = Path(video_path)
                    if video_path.exists():
                        video_log[f"videos/{task_group}/{label}/ep{ep_idx:02d}"] = wandb.Video(str(video_path), fps=fps)
            if video_log:
                wandb.log(video_log)

            wandb.finish()
            logging.info("Logged eval results to wandb.")
        except Exception as e:
            logging.warning(f"wandb logging failed: {e}")

    logging.info("End of eval")


# ---- typed payload returned by one task eval ----
class TaskMetrics(TypedDict):
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]
    video_paths: list[str]
    skill_plot_paths: list[str]
    skill_timeline_paths: list[str]
    skill_token_records: list[dict]
    skill_html_paths: list[str]
    skill_html_records: list[dict]


ACC_KEYS = (
    "sum_rewards",
    "max_rewards",
    "successes",
    "video_paths",
    "skill_plot_paths",
    "skill_timeline_paths",
    "skill_token_records",
    "skill_html_paths",
)


def eval_one(
    env: gym.vector.VectorEnv,
    *,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int,
    video_frame_stride: int,
    video_fps: int | None,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
    forced_skill_token_sequences: list[list[int]] | None = None,
    reference_skill_token_sequences: list[list[int]] | None = None,
    collect_skill_html: bool = False,
    task_description: str | None = None,
) -> TaskMetrics:
    """Evaluates one task_id of one suite using the provided vec env."""

    task_videos_dir = videos_dir

    task_result = eval_policy(
        env=env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        video_frame_stride=video_frame_stride,
        video_fps=video_fps,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        forced_skill_token_sequences=forced_skill_token_sequences,
        reference_skill_token_sequences=reference_skill_token_sequences,
        collect_skill_html=collect_skill_html,
        task_description=task_description,
    )

    per_episode = task_result["per_episode"]
    return TaskMetrics(
        sum_rewards=[ep["sum_reward"] for ep in per_episode],
        max_rewards=[ep["max_reward"] for ep in per_episode],
        successes=[ep["success"] for ep in per_episode],
        video_paths=task_result.get("video_paths", []),
        skill_plot_paths=task_result.get("skill_plot_paths", []),
        skill_timeline_paths=task_result.get("skill_timeline_paths", []),
        skill_token_records=task_result.get("skill_token_records", []),
        skill_html_paths=[],
        skill_html_records=task_result.get("skill_html_records", []),
    )


def run_one(
    task_group: str,
    task_id: int,
    env,
    *,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    max_episodes_rendered: int,
    video_frame_stride: int,
    video_fps: int | None,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
    forced_skill_token_sequences_by_task: dict[tuple[str, int], list[list[int]]] | None = None,
    reference_skill_token_sequences_by_task: dict[tuple[str, int], list[list[int]]] | None = None,
    skill_html_dir: Path | None = None,
    skill_html_train_samples: int = 6,
    skill_html_skill_latents_path: str | None = None,
    skill_html_raw_dataset_dir: str | None = None,
    skill_html_image_key: str | None = None,
    task_descriptions: dict[int, str] | None = None,
):
    """
    Run eval_one for a single (task_group, task_id, env).
    Returns (task_group, task_id, task_metrics_dict).
    This function is intentionally module-level to make it easy to test.
    """
    task_videos_dir = None
    if videos_dir is not None:
        task_videos_dir = videos_dir / f"{task_group}_{task_id}"
        task_videos_dir.mkdir(parents=True, exist_ok=True)

    # Call the existing eval_one (assumed to return TaskMetrics-like dict)
    metrics = eval_one(
        env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        video_frame_stride=video_frame_stride,
        video_fps=video_fps,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        forced_skill_token_sequences=(
            None
            if forced_skill_token_sequences_by_task is None
            else forced_skill_token_sequences_by_task.get((task_group, task_id))
        ),
        reference_skill_token_sequences=(
            None
            if reference_skill_token_sequences_by_task is None
            else reference_skill_token_sequences_by_task.get((task_group, task_id))
        ),
        collect_skill_html=skill_html_dir is not None,
        task_description=None if task_descriptions is None else task_descriptions.get(int(task_id)),
    )
    if skill_html_dir is not None:
        html_path = _write_task_skill_html(
            task_group=task_group,
            task_id=task_id,
            task_description=None if task_descriptions is None else task_descriptions.get(int(task_id)),
            records=metrics.get("skill_html_records", []),
            policy=policy,
            output_dir=skill_html_dir,
            train_samples=skill_html_train_samples,
            skill_latents_path=skill_html_skill_latents_path,
            raw_dataset_dir=skill_html_raw_dataset_dir,
            image_key=skill_html_image_key,
        )
        metrics["skill_html_paths"] = [] if html_path is None else [html_path]
    metrics.pop("skill_html_records", None)
    # ensure we always provide video_paths key to simplify accumulation
    if max_episodes_rendered > 0:
        metrics.setdefault("video_paths", [])
    metrics.setdefault("skill_plot_paths", [])
    metrics.setdefault("skill_timeline_paths", [])
    metrics.setdefault("skill_html_paths", [])
    # compute per-task pc_success so index.html can show the badge
    successes = metrics.get("successes", [])
    if successes:
        metrics["pc_success"] = sum(1 for s in successes if s) / len(successes) * 100
    return task_group, task_id, metrics


def eval_policy_all(
    envs: dict[str, dict[int, gym.vector.VectorEnv]],
    policy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    *,
    max_episodes_rendered: int = 0,
    video_frame_stride: int = 1,
    video_fps: int | None = None,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    max_parallel_tasks: int = 1,
    forced_skill_token_sequences_by_task: dict[tuple[str, int], list[list[int]]] | None = None,
    reference_skill_token_sequences_by_task: dict[tuple[str, int], list[list[int]]] | None = None,
    skill_html_dir: Path | None = None,
    skill_html_train_samples: int = 6,
    skill_html_skill_latents_path: str | None = None,
    skill_html_raw_dataset_dir: str | None = None,
    skill_html_image_key: str | None = None,
    task_descriptions: dict[int, str] | None = None,
    on_task_done_cmd: str | None = None,
) -> dict:
    """
    Evaluate a nested `envs` dict: {task_group: {task_id: vec_env}}.
    on_task_done_cmd: shell command run (best-effort) AFTER EACH task's videos are written — used to
    stitch the multi-model side-by-side grid task-by-task (progressive, like stage1) instead of once
    at job end. "{task_id}"/"{task_group}" placeholders are substituted. Sequential path only.
    This implementation flattens tasks, runs them sequentially or via ThreadPoolExecutor,
    accumulates per-group and overall statistics, and returns the same aggregate metrics
    schema as the single-env evaluator (avg_sum_reward / avg_max_reward / pc_success / timings)
    plus per-task infos.
    """
    start_t = time.time()

    # Flatten envs into list of (task_group, task_id, env)
    tasks = [(tg, tid, vec) for tg, group in envs.items() for tid, vec in group.items()]

    # accumulators: track metrics at both per-group level and across all groups
    group_acc: dict[str, dict[str, list]] = defaultdict(lambda: {k: [] for k in ACC_KEYS})
    overall: dict[str, list] = {k: [] for k in ACC_KEYS}
    per_task_infos: list[dict] = []

    # small inline helper to accumulate one task's metrics into accumulators
    def _accumulate_to(group: str, metrics: dict):
        # metrics expected to contain 'sum_rewards', 'max_rewards', 'successes', optionally 'video_paths'
        # but eval_one may store per-episode lists; we assume metrics uses scalars averaged per task as before.
        # To be robust, accept scalars or lists.
        def _append(key, value):
            if value is None:
                return
            if isinstance(value, list):
                group_acc[group][key].extend(value)
                overall[key].extend(value)
            else:
                group_acc[group][key].append(value)
                overall[key].append(value)

        _append("sum_rewards", metrics.get("sum_rewards"))
        _append("max_rewards", metrics.get("max_rewards"))
        _append("successes", metrics.get("successes"))
        # video_paths is list-like
        paths = metrics.get("video_paths", [])
        if paths:
            group_acc[group]["video_paths"].extend(paths)
            overall["video_paths"].extend(paths)
        skill_plot_paths = metrics.get("skill_plot_paths", [])
        if skill_plot_paths:
            group_acc[group]["skill_plot_paths"].extend(skill_plot_paths)
            overall["skill_plot_paths"].extend(skill_plot_paths)
        skill_timeline_paths = metrics.get("skill_timeline_paths", [])
        if skill_timeline_paths:
            group_acc[group]["skill_timeline_paths"].extend(skill_timeline_paths)
            overall["skill_timeline_paths"].extend(skill_timeline_paths)
        skill_token_records = metrics.get("skill_token_records", [])
        if skill_token_records:
            group_acc[group]["skill_token_records"].extend(skill_token_records)
            overall["skill_token_records"].extend(skill_token_records)
        skill_html_paths = metrics.get("skill_html_paths", [])
        if skill_html_paths:
            group_acc[group]["skill_html_paths"].extend(skill_html_paths)
            overall["skill_html_paths"].extend(skill_html_paths)

    # Choose runner (sequential vs threaded)
    task_runner = partial(
        run_one,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        video_frame_stride=video_frame_stride,
        video_fps=video_fps,
        videos_dir=videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        forced_skill_token_sequences_by_task=forced_skill_token_sequences_by_task,
        reference_skill_token_sequences_by_task=reference_skill_token_sequences_by_task,
        skill_html_dir=skill_html_dir,
        skill_html_train_samples=skill_html_train_samples,
        skill_html_skill_latents_path=skill_html_skill_latents_path,
        skill_html_raw_dataset_dir=skill_html_raw_dataset_dir,
        skill_html_image_key=skill_html_image_key,
        task_descriptions=task_descriptions,
    )

    if max_parallel_tasks <= 1:
        # sequential path (single accumulator path on the main thread)
        # NOTE: keeping a single-threaded accumulator avoids concurrent list appends or locks
        for task_group, task_id, env in tasks:
            tg, tid, metrics = task_runner(task_group, task_id, env)
            _accumulate_to(tg, metrics)
            per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
            if on_task_done_cmd:   # progressive per-task stitch (best-effort; never fail the eval)
                try:
                    cmd = on_task_done_cmd.format(task_id=tid, task_group=tg)
                    subprocess.run(cmd, shell=True, check=False)
                except Exception as exc:  # noqa: BLE001
                    logging.warning("on_task_done_cmd failed for task %s: %s", tid, exc)
    else:
        # threaded path: submit all tasks, consume completions on main thread and accumulate there
        with cf.ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
            fut2meta = {}
            for task_group, task_id, env in tasks:
                fut = executor.submit(task_runner, task_group, task_id, env)
                fut2meta[fut] = (task_group, task_id)
            for fut in cf.as_completed(fut2meta):
                tg, tid, metrics = fut.result()
                _accumulate_to(tg, metrics)
                per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})

    # compute aggregated metrics helper (robust to lists/scalars)
    def _agg_from_list(xs):
        if not xs:
            return float("nan")
        arr = np.array(xs, dtype=float)
        return float(np.nanmean(arr))

    # compute per-group aggregates
    groups_aggregated = {}
    for group, acc in group_acc.items():
        groups_aggregated[group] = {
            "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),
            "avg_max_reward": _agg_from_list(acc["max_rewards"]),
            "pc_success": _agg_from_list(acc["successes"]) * 100 if acc["successes"] else float("nan"),
            "n_episodes": len(acc["sum_rewards"]),
            "video_paths": list(acc["video_paths"]),
            "skill_plot_paths": list(acc["skill_plot_paths"]),
            "skill_timeline_paths": list(acc["skill_timeline_paths"]),
            "skill_token_records": list(acc["skill_token_records"]),
            "skill_html_paths": list(acc["skill_html_paths"]),
        }

    # overall aggregates
    overall_agg = {
        "avg_sum_reward": _agg_from_list(overall["sum_rewards"]),
        "avg_max_reward": _agg_from_list(overall["max_rewards"]),
        "pc_success": _agg_from_list(overall["successes"]) * 100 if overall["successes"] else float("nan"),
        "n_episodes": len(overall["sum_rewards"]),
        "eval_s": time.time() - start_t,
        "eval_ep_s": (time.time() - start_t) / max(1, len(overall["sum_rewards"])),
        "video_paths": list(overall["video_paths"]),
        "skill_plot_paths": list(overall["skill_plot_paths"]),
        "skill_timeline_paths": list(overall["skill_timeline_paths"]),
        "skill_token_records": list(overall["skill_token_records"]),
        "skill_html_paths": list(overall["skill_html_paths"]),
    }

    return {
        "per_task": per_task_infos,
        "per_group": groups_aggregated,
        "overall": overall_agg,
    }


def main():
    init_logging()
    register_third_party_plugins()
    eval_main()


if __name__ == "__main__":
    main()
