#!/usr/bin/env python3
"""Config for SkillVLA stage-3 PRE-TRAINING (PT).

Pick the skillvla dataset to train on by NAME + sub-folder:
    {project_root}/{dataset_root}/{skillvla_dataset_root}/{source_dataset}/{run_tag}/
which holds  skillvla/ + FSQ.pt + dino.npz  (built by configs/train_skillVLA/build_data).
FSQ levels / dim / codebook size are read from FSQ.pt by the model itself, so only
paths + training knobs are needed here. Roots come from build_data's yaml.
Emits shell exports (--shell).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import as_bool, as_list, get_value, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "skillVLA_train_config.yaml"
BUILD_DATA_CONFIG_PATH = _HERE.parent.parent.parent / "build_data" / "train_skillVLA_config.yaml"


def _is_set(v) -> bool:
    if v is None:
        return False
    if isinstance(v, str) and v.strip() == "":
        return False
    if isinstance(v, (list, tuple, dict)) and len(v) == 0:
        return False
    return True


def _roots() -> tuple[Path, Path, Path, Path]:
    """(project_root, dataset_root, skillvla_root, lerobot_root) from build_data yaml."""
    bc = load_config(str(BUILD_DATA_CONFIG_PATH))
    project_root = Path(str(get_value(bc, "project_root"))).expanduser()
    dataset_root = project_root / str(get_value(bc, "dataset_root", "libero_dataset"))
    skillvla_root = dataset_root / str(get_value(bc, "skillvla_dataset_root", "skillvla_dataset"))
    return project_root, dataset_root, skillvla_root, project_root / "lerobot"


def build_settings(cfg: dict) -> dict:
    project_root, dataset_root, skillvla_root, lerobot_root = _roots()

    source_dataset = str(get_value(cfg, "source_dataset"))
    run_tag = str(get_value(cfg, "run_tag"))
    run_dir = skillvla_root / source_dataset / run_tag

    batch_size = int(get_value(cfg, "batch_size", 32))
    num_gpus = int(get_value(cfg, "num_gpus", 1))
    lr_base = float(get_value(cfg, "lr_base", 2.5e-05))
    exp = str(get_value(cfg, "exp", "")).strip()

    # branch suffix: branch2 = terminator + reconstructor chunk suffix; branch1 = terminator only.
    use_recon_suffix = as_bool(get_value(cfg, "use_reconstructor_chunk_suffix", True))
    branch = "branch2" if use_recon_suffix else "branch1"

    run_name = f"{source_dataset}_{run_tag}_batch{batch_size}_{branch}"
    if exp:
        run_name = f"{run_name}_{exp}"
    vla_root = project_root / str(get_value(cfg, "skillvla_outputs_root", "skillVLA_outputs"))
    output_dir = vla_root / f"PT_{run_name}"

    settings: dict = {
        # roots
        "project_root": project_root,
        "lerobot_root": lerobot_root,
        # which skillvla dataset (name + sub-folder)
        "source_dataset": source_dataset,
        "run_tag": run_tag,
        "skillvla_run_dir": run_dir,
        "skillvla_dataset_dir": run_dir / "skillvla",
        "fsq_ckpt": run_dir / "FSQ.pt",
        "dino_tokens_path": run_dir / "dino.npz",
        "dino_cache_path": run_dir / "dino_tokens.cache.npy",
        "skill_latents_path": run_dir / "skill_latents.npz",   # eval skill_html ref (may be cleaned up)
        "raw_dataset_dir": dataset_root / source_dataset,       # raw LeRobot dataset (eval ref)
        "image_key": str(get_value(cfg, "image_key", "observation.images.image")),
        "skill_decoder_state_indices": str(get_value(cfg, "skill_decoder_state_indices", "[0,1,2,3,4,5,6,7]")),
        # model init
        "pi_base": str(get_value(cfg, "pi_base", "lerobot/pi05_base")),
        "image_model_path": str(get_value(cfg, "image_model_path", "/data2/dohyeon/SBD/models/dinov3-vits16")),
        # output
        "skillvla_outputs_root": vla_root,
        "pt_run_name": run_name,
        "pt_output_dir": output_dir,
        "repo_id": f"dohyeon/{source_dataset}",
        # optimization
        "batch_size": batch_size,
        "num_workers": int(get_value(cfg, "num_workers", 4)),
        "num_gpus": num_gpus,
        "lr": lr_base * num_gpus,
        "steps": int(get_value(cfg, "steps", 100000)),
        "save_freq": int(get_value(cfg, "save_freq", 5000)),
        # skillvla knobs (FSQ levels/dim/codebook are auto-read from FSQ.pt by the model)
        "skill_predictor_loss_weight": str(get_value(cfg, "skill_predictor_loss_weight", 1.0)),
        "skill_boundary_random_p": int(get_value(cfg, "skill_boundary_random_p", 10)),
        "skill_decoder_end_threshold": str(get_value(cfg, "skill_decoder_end_threshold", 0.5)),
        "use_reconstructor_chunk_suffix": use_recon_suffix,
        "detach_action_prefix_grad": as_bool(get_value(cfg, "detach_action_prefix_grad", False)),
        "detach_sp_prefix": as_bool(get_value(cfg, "detach_sp_prefix", False)),
        # wandb
        "wandb_enable": as_bool(get_value(cfg, "wandb_enable", True)),
        "wandb_project": str(get_value(cfg, "wandb_project", "VLA_posttrain")),
    }

    part = ",".join(as_list(get_value(cfg, "train_partition", ["debug"]))) or "debug"
    excl = ",".join(as_list(get_value(cfg, "train_exclude_nodes", [])))
    settings.update({
        "train_partition": part,
        "train_qos": str(get_value(cfg, "train_qos", "big_qos")),
        "train_gres": str(get_value(cfg, "train_gres", "gpu:1")),
        "train_cpus_per_task": int(get_value(cfg, "train_cpus_per_task", 16)),
        "train_mem": str(get_value(cfg, "train_mem", "128G")),
        "train_time": str(get_value(cfg, "train_time", "48:00:00")),
        "train_nodelist": str(get_value(cfg, "train_nodelist", "")),
        "train_exclude_nodes": excl,
    })
    return settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--shell", action="store_true")
    args = ap.parse_args()
    settings = build_settings(load_config(args.config))
    if args.shell:
        print_shell(settings)
    else:
        for k, v in settings.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
