#!/usr/bin/env python3
"""Config resolver for configs/train_pi05_cycle (block-cyclic PT mini-experiments).

Emits shell exports for cycle_PT.sbatch. PT-only: FT/eval reuse the regular
configs/train_pi05 pipeline on the produced checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "cycle" / "cycle_config.yaml"


def _find_global(start: Path) -> Path | None:
    for d in [start.resolve(), *start.resolve().parents]:
        candidate = d / "global_config.yaml"
        if candidate.exists():
            return candidate
    return None


def load_config(path: Path) -> dict[str, Any]:
    config_path = Path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    gpath = _find_global(config_path.parent)
    if gpath is not None and gpath.resolve() != config_path.resolve():
        with open(gpath, "r", encoding="utf-8") as f:
            gcfg = yaml.safe_load(f) or {}
        cfg = {**gcfg, **cfg}
    return cfg


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    return [part.strip() for part in text.split(",") if part.strip()] if text else []


def get_value(cfg: dict[str, Any], key: str, default: Any = None, *, env: str | None = None) -> Any:
    if env and env in os.environ:
        return os.environ[env]
    return cfg.get(key, default)


def resolve_path(project_root: "Path | str", value: Any, default: str = "") -> str:
    s = str(value if value not in (None, "", "null") else default).strip()
    if not s:
        return ""
    p = Path(s).expanduser()
    return str(p if p.is_absolute() else (Path(project_root) / p))


def shell_value(value: Any) -> str:
    if isinstance(value, bool):
        value = "true" if value else "false"
    elif isinstance(value, (list, tuple, dict)):
        value = json.dumps(value)
    return shlex.quote(str(value))


def print_shell(settings: dict[str, Any]) -> None:
    for key, value in settings.items():
        print(f"export {key.upper()}={shell_value(value)}")


def fmt_num(x: float) -> str:
    """Compact number for run names: 0.5 → '05', 1.0 → '1', 0.25 → '025'."""
    s = f"{x:g}".replace(".", "")
    return s


def build_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    lerobot_root = project_root / "lerobot"
    pi05_tokenizer_path = Path(resolve_path(
        project_root,
        get_value(cfg, "pi05_tokenizer", "models/paligemma-3b-pt-224-tokenizer"),
    ))
    required_tokenizer_files = ("config.json", "tokenizer_config.json", "tokenizer.json")
    missing_tokenizer_files = [
        name for name in required_tokenizer_files if not (pi05_tokenizer_path / name).is_file()
    ]
    if missing_tokenizer_files:
        raise FileNotFoundError(
            f"Local PaliGemma tokenizer is incomplete at {pi05_tokenizer_path}: "
            f"missing {missing_tokenizer_files}."
        )
    outputs_root = str(get_value(cfg, "outputs_root", "outputs"))
    cycle_pt_root = project_root / outputs_root / str(get_value(cfg, "pi05_cycle_outputs_root", "pi05_cycle_PT"))

    pt_dataset = str(get_value(cfg, "pt_dataset", "libero_90_full_full", env="PT_DATASET"))
    pt_dataset_root = str(get_value(cfg, "pt_dataset_root", get_value(cfg, "dataset_root", "libero_dataset"), env="PT_DATASET_ROOT"))
    pt_batch_size = int(get_value(cfg, "pt_batch_size", 16, env="PT_BATCH_SIZE"))
    pt_exp = str(get_value(cfg, "pt_exp", "", env="PT_EXP")).strip()

    n_groups = int(get_value(cfg, "cycle_n_groups", 8, env="CYCLE_N_GROUPS"))
    phase_steps = int(get_value(cfg, "cycle_phase_steps", 500, env="CYCLE_PHASE_STEPS"))
    n_cycles = int(get_value(cfg, "cycle_n_cycles", 0, env="CYCLE_N_CYCLES"))  # >0 overrides phase_steps
    reptile_beta = float(get_value(cfg, "cycle_reptile_beta", 1.0, env="CYCLE_REPTILE_BETA"))
    _bend = get_value(cfg, "cycle_reptile_beta_end", -1.0, env="CYCLE_REPTILE_BETA_END")
    reptile_beta_end = float(_bend) if _bend not in (None, "") else -1.0  # 빈 값 = 스케줄 off
    iid_baseline = str(get_value(cfg, "cycle_iid_baseline", False, env="CYCLE_IID_BASELINE")).strip().lower() in {"1", "true", "yes", "on"}
    pt_lr = float(get_value(cfg, "pt_lr_base", 2.5e-05, env="PT_LR_BASE"))
    # constant-LR mode: decay_lr := peak lr → scheduler flat after warmup (LR-artifact control)
    constant_lr = str(get_value(cfg, "pt_constant_lr", False, env="PT_CONSTANT_LR")).strip().lower() in {"1", "true", "yes", "on"}
    pt_decay_lr = pt_lr if constant_lr else float(get_value(cfg, "pt_decay_lr", 2.5e-06, env="PT_DECAY_LR"))

    # Condition is encoded in the run name so wandb/output dirs separate ablations cleanly.
    # Encode whichever was specified: c{n_cycles} (phase auto-computed) or p{phase_steps}.
    sched_tag = f"c{n_cycles}" if n_cycles > 0 else f"p{phase_steps}"
    prefix = "PTiid" if iid_baseline else "PTcyc"
    run_name = f"{prefix}_{pt_dataset}_pi05_batch{pt_batch_size}_g{n_groups}{sched_tag}"
    if not iid_baseline and reptile_beta_end >= 0:
        run_name += f"_b{fmt_num(reptile_beta)}to{fmt_num(reptile_beta_end)}"
    elif not iid_baseline and reptile_beta < 1.0:
        run_name += f"_b{fmt_num(reptile_beta)}"
    if constant_lr:
        run_name += "_constlr"
    if abs(pt_lr - 2.5e-05) > 1e-12:  # 기본 LR이 아니면 이름에 명시 (예: _lr5e-05)
        run_name += f"_lr{pt_lr:g}"
    if pt_exp:
        run_name += f"_{pt_exp}"

    settings = {
        "project_root": project_root,
        "lerobot_root": lerobot_root,
        "python_bin": project_root / ".venv" / "bin" / "python",
        "cycle_train_script": Path(__file__).resolve().parent / "lerobot_train_cycle.py",
        "pi05_cycle_outputs_root": cycle_pt_root,
        "pi_base": resolve_path(project_root, get_value(cfg, "pi_base", "models/pi05_base")),
        "pi05_tokenizer_path": pi05_tokenizer_path,
        # PT
        "pt_dataset": pt_dataset,
        "pt_dataset_root": pt_dataset_root,
        "pt_dataset_dir": project_root / pt_dataset_root / pt_dataset,
        "pt_batch_size": pt_batch_size,
        "pt_num_workers": int(get_value(cfg, "pt_num_workers", 4, env="PT_NUM_WORKERS")),
        "pt_exp": pt_exp,
        "pt_lr": pt_lr,
        "pt_decay_lr": pt_decay_lr,  # == pt_lr when pt_constant_lr=true (flat after warmup)
        "pt_constant_lr": constant_lr,
        "pt_steps": int(get_value(cfg, "pt_steps", 20000, env="PT_STEPS")),
        "pt_save_freq": int(get_value(cfg, "pt_save_freq", 5000, env="PT_SAVE_FREQ")),
        "pt_log_freq": int(get_value(cfg, "pt_log_freq", 200, env="PT_LOG_FREQ")),
        "pt_wandb_project": str(get_value(cfg, "pt_wandb_project", "VLA_cycle", env="PT_WANDB_PROJECT")),
        "pt_run_name": run_name,
        "pt_output_dir": cycle_pt_root / run_name,
        # cycle curriculum
        "cycle_n_groups": n_groups,
        "cycle_phase_steps": phase_steps,
        "cycle_n_cycles": n_cycles,
        "cycle_group_seed": int(get_value(cfg, "cycle_group_seed", 0, env="CYCLE_GROUP_SEED")),
        "cycle_reptile_beta": reptile_beta,
        "cycle_reptile_beta_end": reptile_beta_end,
        "cycle_probe_batches": int(get_value(cfg, "cycle_probe_batches", 2, env="CYCLE_PROBE_BATCHES")),
        "cycle_probe_seed": int(get_value(cfg, "cycle_probe_seed", 12345, env="CYCLE_PROBE_SEED")),
        "cycle_probe_grad_group": int(get_value(cfg, "cycle_probe_grad_group", -1, env="CYCLE_PROBE_GRAD_GROUP")),
        "cycle_iid_baseline": iid_baseline,
    }
    # ── eval (cycle_eval): closed-loop LIBERO success on a cycle-PT checkpoint ──
    eval_model = str(get_value(cfg, "eval_model", "", env="EVAL_MODEL")).strip()
    if not eval_model:
        # default: the run described by the cycle/Reptile toggles in this yaml
        eval_model = run_name
    eval_checkpoint = str(get_value(cfg, "eval_checkpoint", "020000", env="CHECKPOINT"))
    eval_target_task = str(get_value(cfg, "eval_target_task", "libero_90", env="TARGET_TASK"))
    eval_exp = str(get_value(cfg, "eval_exp", "", env="EVAL_EXP")).strip()  # 출력 폴더/run name 추가 서픽스
    eval_exp_suffix = f"_{eval_exp}" if eval_exp else ""
    settings.update({
        "eval_bin": project_root / ".venv" / "bin" / "lerobot-eval",
        "eval_model": eval_model,
        "eval_checkpoint": eval_checkpoint,
        "eval_policy_path": cycle_pt_root / eval_model / "checkpoints" / eval_checkpoint / "pretrained_model",
        "eval_groups_json": cycle_pt_root / eval_model / "groups.json",
        "eval_target_task": eval_target_task,
        "eval_task_ids": str(get_value(cfg, "eval_task_ids", json.dumps(list(range(90))), env="TASK_IDS")),
        "eval_n_episodes": int(get_value(cfg, "eval_n_episodes", 5, env="N_EPISODES")),
        "eval_episode_offset": int(get_value(cfg, "eval_episode_offset", 25, env="EPISODE_OFFSET")),
        "eval_n_action_steps": int(get_value(cfg, "eval_n_action_steps", 5, env="N_ACTION_STEPS")),
        "eval_batch_size": int(get_value(cfg, "eval_batch_size", 1, env="EVAL_BATCH_SIZE")),
        "eval_max_parallel_tasks": int(get_value(cfg, "eval_max_parallel_tasks", 1, env="MAX_PARALLEL_TASKS")),
        "eval_max_videos_per_task": int(get_value(cfg, "eval_max_videos_per_task", 1, env="MAX_VIDEOS_PER_TASK")),
        "eval_video_frame_stride": int(get_value(cfg, "eval_video_frame_stride", 2, env="VIDEO_FRAME_STRIDE")),
        "eval_video_fps": int(get_value(cfg, "eval_video_fps", 10, env="VIDEO_FPS")),
        "eval_wandb_project": str(get_value(cfg, "eval_wandb_project", "VLA_eval", env="WANDB_PROJECT")),
        "eval_wandb_run_name": str(os.environ.get("WANDB_RUN_NAME", f"{eval_model}_{eval_checkpoint}_{eval_target_task}{eval_exp_suffix}")),
    })
    # ── FT (cycle_ft): replay-free finetune of a cycle-PT ckpt + live PT-group probes ──
    ft_source = get_value(cfg, "ft_source", {}) or {}
    ft_src_dir = str(ft_source.get("model_dir", "")).strip()
    ft_src_label = str(ft_source.get("label", "src")).strip()
    ft_src_ckpt = str(get_value(cfg, "ft_source_checkpoint", "020000", env="FT_SOURCE_CHECKPOINT"))
    ft_src_base = Path(ft_src_dir) if Path(ft_src_dir).is_absolute() else cycle_pt_root / ft_src_dir
    ft_dataset = str(get_value(cfg, "ft_dataset", "libero_10_full_2", env="FT_DATASET"))
    ft_batch_size = int(get_value(cfg, "ft_batch_size", 16, env="FT_BATCH_SIZE"))
    ft_exp = str(get_value(cfg, "ft_exp", "", env="FT_EXP")).strip()
    ft_run_name = f"FTcyc_{ft_src_label}_{ft_src_ckpt}_{ft_dataset}_batch{ft_batch_size}"
    if ft_exp:
        ft_run_name += f"_{ft_exp}"
    cycle_ft_root = project_root / outputs_root / str(get_value(cfg, "pi05_cycle_ft_outputs_root", "pi05_cycle_FT"))
    settings.update({
        "ft_dataset": ft_dataset,
        "ft_dataset_dir": project_root / pt_dataset_root / ft_dataset,
        "ft_batch_size": ft_batch_size,
        "ft_num_workers": int(get_value(cfg, "ft_num_workers", 4, env="FT_NUM_WORKERS")),
        "ft_lr": float(get_value(cfg, "ft_lr", 2.5e-05, env="FT_LR")),
        "ft_steps": int(get_value(cfg, "ft_steps", 15000, env="FT_STEPS")),
        "ft_save_freq": int(get_value(cfg, "ft_save_freq", 5000, env="FT_SAVE_FREQ")),
        "ft_log_freq": int(get_value(cfg, "ft_log_freq", 200, env="FT_LOG_FREQ")),
        "ft_wandb_project": str(get_value(cfg, "ft_wandb_project", "VLA_cycle_FT", env="FT_WANDB_PROJECT")),
        "ft_run_name": ft_run_name,
        "ft_output_dir": cycle_ft_root / ft_run_name,
        "ft_pretrained_model_path": ft_src_base / "checkpoints" / ft_src_ckpt / "pretrained_model",
        # PT 그룹 probe용. 소스가 groups.json 없는 일반 PT(ref)면 아무 cycle 런 것으로 override
        # (그룹 분할은 dataset+seed 고정이라 모든 런에서 동일).
        "ft_groups_json": str(get_value(cfg, "ft_groups_json", "", env="FT_GROUPS_JSON")) or str(ft_src_base / "groups.json"),
        "ft_probe_every": int(get_value(cfg, "ft_probe_every", 250, env="FT_PROBE_EVERY")),
        "ft_freeze_vision_encoder": str(get_value(cfg, "ft_freeze_vision_encoder", False)).lower() in {"1", "true", "yes", "on"},
        "ft_train_expert_only": str(get_value(cfg, "ft_train_expert_only", False)).lower() in {"1", "true", "yes", "on"},
        "ft_gres": str(get_value(cfg, "ft_gres", "gpu:1")),
        "ft_cpus_per_task": int(get_value(cfg, "ft_cpus_per_task", 16)),
        "ft_mem": str(get_value(cfg, "ft_mem", "128G")),
        "ft_time": str(get_value(cfg, "ft_time", "24:00:00")),
    })

    # ── compare eval: N models side-by-side, task-by-task incremental stitching ──
    compare_models = get_value(cfg, "compare_models", []) or []
    compare_labels = [str(m.get("label", f"m{i}")) for i, m in enumerate(compare_models)]
    compare_checkpoint = str(get_value(cfg, "compare_checkpoint", eval_checkpoint, env="COMPARE_CHECKPOINT"))
    settings.update({
        "compare_models": compare_models,  # shell_value → JSON string
        "compare_checkpoint": compare_checkpoint,
        "compare_run_name": ("_vs_".join(compare_labels) or "compare") + f"_{compare_checkpoint}_{eval_target_task}{eval_exp_suffix}",
        # GPU 병렬 샤드 수 (SLURM array 크기). env N_SHARDS가 yaml보다 우선.
        "compare_n_shards": int(get_value(cfg, "compare_n_shards", 1, env="N_SHARDS")),
    })
    # Slurm (partition/qos/nodelist/exclude canonical in global_config.yaml)
    settings.update({
        "pt_partition": ",".join(as_list(get_value(cfg, "train_partition", ["debug"]))) or "debug",
        "pt_nodelist": str(get_value(cfg, "train_nodelist", "")),
        "pt_exclude_nodes": ",".join(as_list(get_value(cfg, "train_exclude_nodes", []))),
        "pt_qos": str(get_value(cfg, "train_qos", "big_qos")),
        "pt_gres": str(get_value(cfg, "pt_gres", "gpu:1")),
        "pt_cpus_per_task": int(get_value(cfg, "pt_cpus_per_task", 16)),
        "pt_mem": str(get_value(cfg, "pt_mem", "128G")),
        "pt_time": str(get_value(cfg, "pt_time", "24:00:00")),
        "eval_partition": ",".join(as_list(get_value(cfg, "train_partition", ["debug"]))) or "debug",
        "eval_nodelist": str(get_value(cfg, "train_nodelist", "")),
        "eval_exclude_nodes": ",".join(as_list(get_value(cfg, "train_exclude_nodes", []))),
        "eval_qos": str(get_value(cfg, "eval_qos", get_value(cfg, "train_qos", "base_qos"))),
        "eval_gres": str(get_value(cfg, "eval_gres", "gpu:1")),
        "eval_cpus_per_task": int(get_value(cfg, "eval_cpus_per_task", 8)),
        "eval_mem": str(get_value(cfg, "eval_mem", "32G")),
        "eval_time": str(get_value(cfg, "eval_time", "48:00:00")),
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
        for key, value in settings.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
