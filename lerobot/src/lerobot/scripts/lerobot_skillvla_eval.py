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
import json
import logging
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


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
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
    threads = []  # for video saving threads
    n_episodes_rendered = 0  # for saving the correct number of videos

    # Callback for visualization.
    def render_frame(env: gym.vector.VectorEnv):
        # noqa: B023
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render_now)]))  # noqa: B023
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            # Here we must render all frames and discard any we don't need.
            ep_frames.append(np.stack(env.call("render")[:n_to_render_now]))

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

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)
            )
        rollout_data = rollout(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seeds=list(seeds) if seeds else None,
            return_observations=return_episode_data,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
        )

        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        n_steps = rollout_data["done"].shape[1]
        # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)
        if videos_dir is not None:
            new_skill_plot_paths, new_skill_timeline_paths = _save_skill_trace_plots(
                policy=policy,
                output_dir=videos_dir,
                episode_index=batch_ix * env.num_envs,
                episode_steps=int(done_indices.max().item()) + 1,
            )
            skill_plot_paths.extend(new_skill_plot_paths)
            skill_timeline_paths.extend(new_skill_timeline_paths)

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
            for stacked_frames, done_index in zip(
                batch_stacked_frames, done_indices.flatten().tolist(), strict=False
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        stacked_frames[: done_index + 1],  # + 1 to capture the last observation
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1

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


def _save_skill_trace_plots(
    policy: PreTrainedPolicy, output_dir: Path, episode_index: int, episode_steps: int | None = None
) -> tuple[list[str], list[str]]:
    get_skill_trace = getattr(policy, "get_skill_trace", None)
    if get_skill_trace is None:
        return [], []

    trace = get_skill_trace()
    if not trace:
        return [], []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logging.warning(f"Could not save SkillVLA skill plots because matplotlib failed to import: {exc}")
        return [], []

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

    action_names = ["x", "y", "z", "r", "p", "yaw", "gripper"]
    skill_dir = output_dir / "skill_plots" / f"episode_{episode_index:04d}"
    timeline_dir = output_dir / "skill_timelines" / f"episode_{episode_index:04d}"
    skill_dir.mkdir(parents=True, exist_ok=True)
    timeline_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    timeline_paths = []
    skill_start_timesteps = []
    for record in trace:
        raw_actions = np.asarray(record["raw_actions"])
        skill_index = int(record.get("skill_index", len(saved_paths)))
        batch_index = int(record.get("batch_index", 0))
        episode_timestep = int(record.get("episode_timestep", 0))
        skill_start_timesteps.append((episode_timestep, skill_index))
        timesteps = np.arange(raw_actions.shape[0])
        dim = raw_actions.shape[1]
        labels = action_names[:dim] + [f"a{i}" for i in range(len(action_names), dim)]
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
                label="VAE prior",
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

        fig.suptitle(f"Skill {skill_index + 1} raw actions: VAE prior vs action expert")
        axes[-1].set_xlabel("skill timestep")
        fig.tight_layout()

        plot_path = skill_dir / f"skill_{skill_index + 1:03d}_batch_{batch_index:02d}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        saved_paths.append(str(plot_path))

    if skill_start_timesteps:
        skill_start_timesteps = sorted(skill_start_timesteps)
        x_values = [step for step, _ in skill_start_timesteps]
        y_values = [skill_index + 1 for _, skill_index in skill_start_timesteps]
        if episode_steps is not None and episode_steps > 0:
            x_values = [*x_values, episode_steps]
            y_values = [*y_values, y_values[-1]]

        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.step(x_values, y_values, where="post", linewidth=2.0)
        for timestep, skill_index in skill_start_timesteps:
            ax.axvline(timestep, color="tab:red", alpha=0.3, linewidth=1.0)
            ax.text(
                timestep,
                skill_index + 1,
                f"S{skill_index + 1}",
                rotation=90,
                va="bottom",
                ha="right",
                fontsize="small",
            )
        ax.set_title("Skill calls over task timesteps")
        ax.set_xlabel("task timestep")
        ax.set_ylabel("active skill")
        ax.set_yticks([skill_index + 1 for _, skill_index in skill_start_timesteps])
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()

        timeline_path = timeline_dir / "skill_timeline.png"
        fig.savefig(timeline_path, dpi=150)
        plt.close(fig)
        timeline_paths.append(str(timeline_path))

    return saved_paths, timeline_paths


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    if cfg.policy is None or cfg.policy.type != "skill_vla":
        policy_type = None if cfg.policy is None else cfg.policy.type
        raise ValueError(
            "lerobot_skillvla_eval is reserved for SkillVLA checkpoints "
            f"(expected policy.type='skill_vla', got {policy_type!r})."
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

    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )

    policy.eval()

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

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=10,
            videos_dir=Path(cfg.output_dir) / "videos",
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
        )
        print("Overall Aggregated Metrics:")
        print(info["overall"])

        # Print per-suite stats
        for task_group, task_group_info in info.items():
            print(f"\nAggregated Metrics for {task_group}:")
            print(task_group_info)
    # Close all vec envs
    close_envs(envs)

    # Save info
    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

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

            # Build task_id -> language description mapping from LIBERO suite
            task_id_to_desc: dict[int, str] = {}
            try:
                from libero.libero import benchmark
                suite_name = cfg.env.task
                suite = benchmark.get_benchmark_dict()[suite_name]()
                for i, task in enumerate(suite.tasks):
                    task_id_to_desc[i] = task.language
            except Exception:
                pass

            # summary/ : overall & per-group scalar metrics
            summary_log: dict = {}
            overall = info["overall"]
            summary_log["summary/pc_success"] = overall["pc_success"]
            summary_log["summary/success_rate"] = overall["pc_success"] / 100.0
            summary_log["summary/success_count"] = int(
                round((overall["pc_success"] / 100.0) * overall["n_episodes"])
            )
            summary_log["summary/n_episodes"] = overall["n_episodes"]
            summary_log["summary/avg_sum_reward"] = overall["avg_sum_reward"]
            summary_log["summary/avg_max_reward"] = overall["avg_max_reward"]

            for group_name, group_info in info.get("per_group", {}).items():
                summary_log[f"summary/{group_name}/pc_success"] = group_info["pc_success"]
                summary_log[f"summary/{group_name}/avg_sum_reward"] = group_info["avg_sum_reward"]
            wandb.run.summary.update(summary_log)
            wandb.log(
                {
                    "eval/overall_pc_success": overall["pc_success"],
                    "eval/overall_success_rate": overall["pc_success"] / 100.0,
                    "eval/overall_success_count": summary_log["summary/success_count"],
                    "eval/overall_n_episodes": overall["n_episodes"],
                }
            )

            # charts/ : per-task success bar chart + per-episode success line chart
            bar_data: list[list] = []
            episode_table_data: list[list] = []
            episode_xs: list[list[int]] = []
            episode_ys: list[list[float]] = []
            task_keys: list[str] = []

            def _success_to_float(value) -> float:
                if hasattr(value, "item"):
                    value = value.item()
                return float(bool(value))

            for task_info in sorted(info.get("per_task", []), key=lambda t: t.get("task_id", 0)):
                task_id = task_info.get("task_id", 0)
                successes = task_info.get("metrics", {}).get("successes", [])
                success_values = [_success_to_float(s) for s in successes]
                success_count = int(sum(success_values))
                desc = task_id_to_desc.get(task_id, f"task_{task_id}")
                label = f"task{task_id:02d}: {desc}"

                bar_data.append([label, success_count])
                episode_xs.append(list(range(1, len(success_values) + 1)))
                episode_ys.append(success_values)
                task_keys.append(f"task{task_id:02d}")
                for episode_idx, success in enumerate(success_values, start=1):
                    episode_table_data.append([task_id, label, episode_idx, success])

            charts_log: dict = {}
            if bar_data:
                table = wandb.Table(data=bar_data, columns=["task", "success_count"])
                charts_log["charts/task_success_bar"] = wandb.plot.bar(
                    table, "task", "success_count", title="Success Count per Task"
                )
            if episode_table_data:
                charts_log["tables/per_task_episode_success"] = wandb.Table(
                    data=episode_table_data,
                    columns=["task_id", "task", "episode", "success"],
                )
            if episode_xs:
                charts_log["charts/per_task_episode_success"] = wandb.plot.line_series(
                    xs=episode_xs,
                    ys=episode_ys,
                    keys=task_keys,
                    title="Per-Task Success per Episode",
                    xname="episode",
                )
            if charts_log:
                wandb.log(charts_log)

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

            def _wandb_slot_name(task_group: str, task_id: int) -> str:
                safe_group = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in task_group)
                return f"{safe_group}_task{task_id:02d}"

            skill_plot_log: dict = {}
            for task_info in info.get("per_task", []):
                task_id = task_info.get("task_id", 0)
                task_group = task_info.get("task_group", "unknown")
                slot_name = _wandb_slot_name(task_group, task_id)
                for plot_idx, plot_path in enumerate(task_info.get("metrics", {}).get("skill_plot_paths", [])):
                    plot_path = Path(plot_path)
                    if plot_path.exists():
                        skill_plot_log[f"skill_plots_{slot_name}/skill{plot_idx + 1:03d}"] = (
                            wandb.Image(str(plot_path))
                        )
            if skill_plot_log:
                wandb.log(skill_plot_log)

            skill_timeline_log: dict = {}
            for task_info in info.get("per_task", []):
                task_id = task_info.get("task_id", 0)
                task_group = task_info.get("task_group", "unknown")
                slot_name = _wandb_slot_name(task_group, task_id)
                for plot_idx, plot_path in enumerate(task_info.get("metrics", {}).get("skill_timeline_paths", [])):
                    plot_path = Path(plot_path)
                    if plot_path.exists():
                        skill_timeline_log[f"skill_timelines_{slot_name}/timeline{plot_idx + 1:03d}"] = (
                            wandb.Image(str(plot_path))
                        )
            if skill_timeline_log:
                wandb.log(skill_timeline_log)

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


ACC_KEYS = (
    "sum_rewards",
    "max_rewards",
    "successes",
    "video_paths",
    "skill_plot_paths",
    "skill_timeline_paths",
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
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
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
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )

    per_episode = task_result["per_episode"]
    return TaskMetrics(
        sum_rewards=[ep["sum_reward"] for ep in per_episode],
        max_rewards=[ep["max_reward"] for ep in per_episode],
        successes=[ep["success"] for ep in per_episode],
        video_paths=task_result.get("video_paths", []),
        skill_plot_paths=task_result.get("skill_plot_paths", []),
        skill_timeline_paths=task_result.get("skill_timeline_paths", []),
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
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
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
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )
    # ensure we always provide video_paths key to simplify accumulation
    if max_episodes_rendered > 0:
        metrics.setdefault("video_paths", [])
    metrics.setdefault("skill_plot_paths", [])
    metrics.setdefault("skill_timeline_paths", [])
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
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    max_parallel_tasks: int = 1,
) -> dict:
    """
    Evaluate a nested `envs` dict: {task_group: {task_id: vec_env}}.
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
        videos_dir=videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )

    if max_parallel_tasks <= 1:
        # sequential path (single accumulator path on the main thread)
        # NOTE: keeping a single-threaded accumulator avoids concurrent list appends or locks
        for task_group, task_id, env in tasks:
            tg, tid, metrics = task_runner(task_group, task_id, env)
            _accumulate_to(tg, metrics)
            per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
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
