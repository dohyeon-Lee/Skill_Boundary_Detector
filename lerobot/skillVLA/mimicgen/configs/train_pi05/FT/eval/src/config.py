"""Resolve the Coffee baseline and exact-reset settings without loading a GPU model."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import common

training = common.module("coffee_pi05_ft", Path(__file__).resolve().parents[2] / "src/config.py")
DEFAULT = Path(__file__).resolve().parents[1] / "pi05_eval_config.yaml"


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def prepare(cfg):
    cfg = dict(cfg)
    if "policy_path" in cfg:
        return cfg
    train_path = DEFAULT.parents[1] / "pi05_FT_config.yaml"
    ft = training.build_settings(common.load_config(train_path))
    step = positive(cfg["checkpoint"], "checkpoint")
    action_steps = positive(cfg["n_action_steps"], "n_action_steps")
    if action_steps > ft["ft_chunk_size"]:
        raise ValueError("n_action_steps exceeds the PT checkpoint's action chunk")
    root = Path(cfg["project_root"])
    policy = Path(ft["ft_output_dir"]) / "checkpoints" / f"{step:06d}" / "pretrained_model"
    label = training.shared.step_label(str(step))
    name = f'{ft["ft_run_name"]}_FT{label}_{cfg["n_episodes"]}eps_r{action_steps}'
    if cfg["episode_exact"]:
        name += "_exact"
    cfg.update(
        policy_path=str(policy),
        dataset_dir=ft["ft_dataset_dir"],
        dataset_source=ft["ft_dataset"],
        tokenizer_path=str(ft["pi05_tokenizer_path"]),
        simulator_python=str(root / ".venv_mimicgen/bin/python"),
        mimicgen_source_root=str(root / "third_party/mimicgen"),
        init_states_path=str(root / cfg["dataset_root"] / "skillvla" / ft["ft_dataset"] / "eval_init_states.npz")
                         if cfg["episode_exact"] else "",
        eval_out_dir=str(DEFAULT.parent / "outputs" / name),
        run_name=name,
    )
    return cfg


def build_settings(cfg):
    cfg = prepare(cfg)
    positive(cfg["n_episodes"], "n_episodes")
    positive(cfg["max_episode_steps"], "max_episode_steps")
    for key in ("episode_exact", "repeat_episodes"):
        if not isinstance(cfg[key], bool):
            raise ValueError(f"{key} must be true or false")
    if cfg["init_states_path"]:
        from lerobot.utils.mimicgen_init_states import coffee_records

        coffee_records(cfg["init_states_path"], cfg["n_episodes"], cfg["repeat_episodes"])
    settings = {key: cfg[key] for key in (
        "project_root", "server", "policy_path", "dataset_dir", "tokenizer_path", "simulator_python",
        "mimicgen_source_root", "init_states_path", "eval_out_dir", "run_name", "n_episodes",
        "n_action_steps", "max_episode_steps", "seed", "repeat_episodes",
    )}
    flat = training.shared.flatten_stage_blocks({**cfg, "stage": "ft"})
    settings.update({key.replace("ft_", "eval_", 1): value for key, value in
                     training.shared.slurm_settings(flat, "ft", cpus=8, mem="64G", time="3:00:00", qos="base_qos").items()})
    settings.update(
        python_bin=str(Path(cfg["project_root"]) / ".venv/bin/python"),
        code_root=str(common.CODE_ROOT),
        video_json=cfg["video"],
        task_ids=[0],
        models_json=[{"label": "pi05", "policy_path": cfg["policy_path"]}],
        checkpoint_ready=(Path(cfg["policy_path"]) / "config.json").is_file(),
        wandb_project=cfg["logging"]["wandb"]["project"] if cfg["logging"]["wandb"]["enable"] else "",
    )
    return settings


if __name__ == "__main__":
    common.command_line(DEFAULT, prepare, build_settings)
