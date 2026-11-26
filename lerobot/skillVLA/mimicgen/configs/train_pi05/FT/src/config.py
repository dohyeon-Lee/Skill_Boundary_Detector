"""Coffee pi05 FT, sharing LIBERO training and SkillVLA's episode selection."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import common

shared = common.module("shared_pi05_train", common.SHARED / "train_pi05/PT/src/train_pi05_config.py")
DEFAULT = Path(__file__).resolve().parents[1] / "pi05_FT_config.yaml"


def prepare(cfg):
    cfg = dict(cfg)
    if "ft_dataset_episodes" in cfg:
        return cfg
    comparison = common.CONFIGS / cfg.pop("comparison_config")
    reference = yaml.safe_load(comparison.read_text())
    data = reference["dataset"]
    source = str(data["source"])
    if Path(source).name != source:
        raise ValueError("dataset.source must be a dataset folder name")
    dataset = Path(cfg["project_root"]) / cfg["dataset_root"] / source
    selection = common.module("coffee_episode_selection", common.SHARED / "train_skills/src/train_skills_config.py")
    episodes = selection.resolve_training_episodes({"dataset": data}, dataset)
    warm = cfg.pop("warm_start")
    run = str(warm["run"])
    if Path(run).name != run or not str(warm["checkpoint"]).isdigit():
        raise ValueError("warm_start requires a run folder name and numeric checkpoint")
    cfg.update(
        ft_dataset=source,
        ft_dataset_episodes=episodes,
        comparison_dataset=dict(data),
        ft_pretrained_outputs_root=warm.get("outputs_root", cfg["outputs_root"]),
        ft_pretrained_run_name=run,
        ft_pretrained_checkpoint=f'{int(warm["checkpoint"]):06d}',
        ft_exp=(selection.training_episode_suffix({"dataset": data}, episodes).lstrip("_")
                if data.get("episode_ids") else "epall" if episodes is None else f"ep{len(episodes)}"),
    )
    return cfg


def build_settings(cfg):
    cfg = prepare(cfg)
    settings = shared.build_settings(cfg)
    # FT loads the PT checkpoint directly; the original base weights are not an input.
    settings.pop("pi_base", None)
    checkpoint = Path(settings["ft_pretrained_model_path"])
    for name in ("config.json", "train_config.json", "policy_preprocessor.json",
                 "policy_postprocessor.json"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    if not list(checkpoint.glob("model*.safetensors")):
        raise FileNotFoundError(f"No model weights in {checkpoint}")
    policy = json.loads((checkpoint / "config.json").read_text())
    info = json.loads((Path(settings["ft_dataset_dir"]) / "meta/info.json").read_text())
    for name, feature in policy["input_features"].items():
        if name not in info["features"] or info["features"][name]["shape"] != feature["shape"]:
            raise ValueError(f"Coffee observation {name} does not match the LIBERO pi05 checkpoint")
    if info["features"]["action"]["shape"] != policy["output_features"]["action"]["shape"]:
        raise ValueError("Coffee action shape does not match the LIBERO pi05 checkpoint")
    settings.update(
        lerobot_root=str(common.CODE_ROOT / "lerobot"),
        code_root=str(common.CODE_ROOT),
        server=cfg["server"],
        ft_dataset_episodes=cfg["ft_dataset_episodes"],
        dataset_dir=str(settings["ft_dataset_dir"]),
        output_dir=str(settings["ft_output_dir"]),
        run_name=settings["ft_run_name"],
        steps=settings["ft_steps"],
        protect_checkpoints=settings["ft_protect_checkpoints"],
    )
    return settings


if __name__ == "__main__":
    common.command_line(DEFAULT, prepare, build_settings)
