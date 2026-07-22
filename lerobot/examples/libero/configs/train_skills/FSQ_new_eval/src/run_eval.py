#!/usr/bin/env python3
"""Closed-loop LIBERO eval for one FSQ_new A/B/C panel."""

import json
import logging
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.skill_expert.processor_skill_expert import (
    make_skill_expert_pre_post_processors,
)
from lerobot.processor import RenameObservationsProcessorStep
from lerobot.scripts.lerobot_skillvla_eval import (
    _libero_task_descriptions,
    close_envs,
    eval_policy_all,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.random_utils import set_seed

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from fsq_new_policy import (  # noqa: E402
    FSQNewExpertPolicy,
    FSQNewOraclePolicy,
    FSQNewTerminator,
    override_init_states,
)
from oracle_data import load_fsq_episode_data  # noqa: E402

log = logging.getLogger(__name__)


def _write_success_chart(info: dict, output_dir: Path, tag: str) -> Path | None:
    labels, rates = [], []
    for task in sorted(info.get("per_task", []), key=lambda item: int(item.get("task_id", 0))):
        successes = task.get("metrics", {}).get("successes", [])
        values = [float(bool(value.item() if hasattr(value, "item") else value)) for value in successes]
        labels.append(f"task{int(task.get('task_id', 0)):02d}")
        rates.append(float(np.mean(values)) if values else 0.0)
    if not labels:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, max(2.8, 0.38 * len(labels) + 0.9)))
    y = np.arange(len(labels))
    ax.barh(y, rates, color="#4C78A8")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("success rate")
    ax.grid(axis="x", alpha=0.25)
    for row, value in zip(y, rates, strict=True):
        ax.text(min(value + 0.02, 0.98), row, f"{value:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    path = output_dir / (f"task_success_rates_{tag}.png" if tag else "task_success_rates.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig) -> None:
    spec_raw = os.environ.get("PANEL_SPEC", "")
    if not spec_raw:
        raise ValueError("PANEL_SPEC is required for FSQ_new eval.")
    spec = json.loads(spec_raw)
    device = get_safe_torch_device(cfg.policy.device, log=True)
    set_seed(cfg.seed)

    policy = FSQNewExpertPolicy(
        spec["fsq_path"],
        device,
        mode=spec["mode"],
        n_action_steps=int(cfg.policy.n_action_steps),
        dino_model_path=spec["dino_model_path"],
        raw_dataset_dir=spec["raw_dataset_dir"],
    )
    terminator = FSQNewTerminator(policy)
    oracle = FSQNewOraclePolicy(
        policy,
        terminator,
        end_threshold=float(cfg.policy.skill_end_threshold),
        progress_threshold=float(cfg.policy.skill_end_progress_threshold),
        end_mode=str(cfg.policy.skill_end_mode),
        advance_mode=str(spec["advance_mode"]),
        max_skill_len=int(cfg.policy.inference_skill_max_length),
        n_action_steps=int(cfg.policy.n_action_steps),
    )
    preprocessor, postprocessor = make_skill_expert_pre_post_processors(
        policy.config, dataset_stats=policy.dataset_stats()
    )
    for step in preprocessor.steps:
        if isinstance(step, RenameObservationsProcessorStep):
            step.rename_map = cfg.rename_map
            break

    logging.info("Making FSQ_new episode-exact environment for panel %s", spec["label"])
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=cfg.env, policy_cfg=policy.config
    )
    episode_data = load_fsq_episode_data(
        spec["latents_path"],
        spec["init_states_path"],
        spec["raw_dataset_dir"],
        cfg.env.task,
    )
    forced = override_init_states(envs, episode_data)
    if not forced:
        close_envs(envs)
        raise RuntimeError("No episode-exact FSQ skill sequences matched the requested LIBERO tasks.")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptions = _libero_task_descriptions(cfg.env.task)
    try:
        with torch.no_grad(), (
            torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext()
        ):
            info = eval_policy_all(
                envs=envs,
                policy=oracle,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=cfg.eval.n_episodes,
                max_episodes_rendered=cfg.eval.max_videos_per_task,
                video_frame_stride=cfg.eval.video_frame_stride,
                video_fps=cfg.eval.video_fps,
                videos_dir=output_dir / "videos",
                start_seed=cfg.seed,
                max_parallel_tasks=cfg.env.max_parallel_tasks,
                forced_skill_token_sequences_by_task=forced,
                reference_skill_token_sequences_by_task=None,
                skill_html_dir=(output_dir / "skill_html") if cfg.eval.skill_html else None,
                skill_html_train_samples=cfg.eval.skill_html_train_samples,
                skill_html_skill_latents_path=spec["latents_path"],
                skill_html_raw_dataset_dir=spec["raw_dataset_dir"],
                skill_html_image_key="observation.images.image",
                task_descriptions=descriptions,
                on_task_done_cmd=os.environ.get("ON_TASK_DONE_CMD") or None,
            )
    finally:
        close_envs(envs)

    tag = os.environ.get("TASK_TAG", "").strip()
    info_path = output_dir / (f"eval_info_{tag}.json" if tag else "eval_info.json")
    info_path.write_text(json.dumps(info, indent=2))
    chart = _write_success_chart(info, output_dir, tag)
    print(f"[{spec['label']}] overall={info['overall']} chart={chart}")

    project = getattr(cfg, "wandb_project", None)
    if project:
        try:
            import wandb

            wandb.init(
                project=project,
                name=cfg.job_name,
                config={"fsq_path": spec["fsq_path"], "mode": spec["mode"]},
            )
            payload = {f"overall/{key}": value for key, value in info["overall"].items()
                       if isinstance(value, (int, float))}
            if chart is not None:
                payload["charts/task_success_rate"] = wandb.Image(str(chart))
            wandb.log(payload)
            wandb.finish()
        except Exception as exc:  # noqa: BLE001
            logging.warning("W&B logging failed: %s", exc)


if __name__ == "__main__":
    eval_main()
