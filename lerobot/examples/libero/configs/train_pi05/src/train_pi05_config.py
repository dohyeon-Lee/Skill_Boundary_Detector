#!/usr/bin/env python3
"""Config resolver for configs/train_pi05.

Emits shell exports for pi05 PT/FT/eval sbatch files without depending on the
legacy configs/data_generation/pipeline_config.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "pi05" / "pi05_config.yaml"


def _find_global(start: Path) -> Path | None:
    """Walk up from `start` to find the nearest global_config.yaml (stops at first hit)."""
    for d in [start.resolve(), *start.resolve().parents]:
        candidate = d / "global_config.yaml"
        if candidate.exists():
            return candidate
    return None


def load_config(path: Path) -> dict[str, Any]:
    config_path = Path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # Merge the nearest global_config.yaml as a base (module cfg keys win for roots).
    gpath = _find_global(config_path.parent)
    if gpath is not None and gpath.resolve() != config_path.resolve():
        with open(gpath, "r", encoding="utf-8") as f:
            gcfg = yaml.safe_load(f) or {}
        cfg = {**gcfg, **cfg}
    return cfg


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def get_value(cfg: dict[str, Any], key: str, default: Any = None, *, env: str | None = None) -> Any:
    if env and env in os.environ:
        return os.environ[env]
    return cfg.get(key, default)


def resolve_path(project_root: "Path | str", value: Any, default: str = "") -> str:
    """Resolve a config path against project_root: absolute → as-is; relative → under project_root;
    blank → "". Keeps model paths portable across servers (project_root from global_config)."""
    s = str(value if value not in (None, "", "null") else default).strip()
    if not s:
        return ""
    p = Path(s).expanduser()
    return str(p if p.is_absolute() else (Path(project_root) / p))


def get_nonempty(cfg: dict[str, Any], key: str, default: Any = None, *, env: str | None = None) -> Any:
    value = get_value(cfg, key, default, env=env)
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    return value


def shell_value(value: Any) -> str:
    if isinstance(value, bool):
        value = "true" if value else "false"
    elif isinstance(value, (list, tuple, dict)):
        value = json.dumps(value)
    return shlex.quote(str(value))


def print_shell(settings: dict[str, Any]) -> None:
    for key, value in settings.items():
        print(f"export {key.upper()}={shell_value(value)}")


def _ft_probe_settings(cfg: dict[str, Any], project_root: Path, ft_dataset_root: str,
                       ft_pre_dataset: str, pt_ckpt: Path) -> dict[str, Any]:
    """PT-forgetting probe knobs for pi05 FT — the SAME lerobot_train infra as skillVLA FT (fixed
    PT-dataset batches, pinned noise, wandb probe/* + probe_forget/*). ft_probe_pt_forgetting=false
    → empty root = probe off. PT dataset derived from the PT checkpoint's train_config.json;
    falls back to {project_root}/{ft_dataset_root}/{ft_pretrained_dataset}."""
    on = as_bool(get_value(cfg, "ft_probe_pt_forgetting", False, env="FT_PROBE_PT_FORGETTING"))
    root, repo = "", ""
    if on:
        tc = pt_ckpt / "train_config.json"
        if tc.is_file():
            ds = json.loads(tc.read_text()).get("dataset") or {}
            root, repo = str(ds.get("root") or ""), str(ds.get("repo_id") or "")
        # 이식 면역: 다른 서버에서 학습된 PT는 그 서버의 절대경로가 박혀 있음 → 존재하지 않으면
        # 이 서버의 {project_root}/{ft_dataset_root}/{데이터셋명}으로 재앵커 (FSQ/ISS와 같은 패턴).
        if root and not Path(root).is_dir():
            root = str(project_root / ft_dataset_root / Path(root).name)
        if not root:
            root = str(project_root / ft_dataset_root / ft_pre_dataset)
        if not repo:
            repo = f"lerobot/{Path(root).name}"
        if not Path(root).is_dir():
            raise ValueError(f"ft_probe_pt_forgetting=true but the PT dataset dir does not exist: {root}")
    return {
        "ft_probe_dataset_root": root,
        "ft_probe_dataset_repo_id": repo,
        "ft_probe_every": int(get_value(cfg, "ft_probe_every", 250, env="FT_PROBE_EVERY")),
        "ft_probe_batches": int(get_value(cfg, "ft_probe_batches", 4, env="FT_PROBE_BATCHES")),
        "ft_probe_seed": int(get_value(cfg, "ft_probe_seed", 12345, env="FT_PROBE_SEED")),
    }


def run_name(prefix: str, dataset: str, batch_size: int, exp: str, tag: str = "") -> str:
    name = f"{prefix}_{dataset}_pi05_batch{batch_size}"
    if tag:
        name = f"{name}_{tag}"
    if exp:
        name = f"{name}_{exp}"
    return name


def frz_lora_tag(fv: bool, fl: bool, lora_enable: bool, lora_llm: bool, lora_vision: bool) -> str:
    """Run-name tag: '{freeze_vision}{freeze_language}_{lora_llm}{lora_vision}' as t/f, e.g. 'tt_ff'.
    The LoRA half is EFFECTIVE (lora_enable AND the per-part flag) so it reads f when LoRA is off."""
    frz = ("t" if fv else "f") + ("t" if fl else "f")
    lo = ("t" if (lora_enable and lora_llm) else "f") + ("t" if (lora_enable and lora_vision) else "f")
    return f"{frz}_{lo}"


def slurm_settings(cfg: dict[str, Any], prefix: str, *, cpus: int, mem: str, time: str, qos: str) -> dict[str, Any]:
    # partition/qos/nodelist/exclude are canonical (global_config.yaml train_*); output keys keep
    # the per-job prefix so the pt/ft/eval sbatch files read the same $<PREFIX>_* vars.
    return {
        f"{prefix}_partition": ",".join(as_list(get_value(cfg, "train_partition", ["debug"]))) or "debug",
        f"{prefix}_nodelist": str(get_value(cfg, "train_nodelist", "")),
        f"{prefix}_exclude_nodes": ",".join(as_list(get_value(cfg, "train_exclude_nodes", []))),
        f"{prefix}_qos": str(get_value(cfg, "train_qos", qos)),
        f"{prefix}_gres": str(get_value(cfg, f"{prefix}_gres", "gpu:1")),
        f"{prefix}_cpus_per_task": int(get_value(cfg, f"{prefix}_cpus_per_task", cpus)),
        f"{prefix}_mem": str(get_value(cfg, f"{prefix}_mem", mem)),
        f"{prefix}_time": str(get_value(cfg, f"{prefix}_time", time)),
    }


def build_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    lerobot_root = project_root / "lerobot"
    # {project_root}/{outputs_root}/{pi05_PT|pi05_FT}: outputs_root comes from global_config.yaml
    # (switchable to outputs_filtered). PT and FT write to SEPARATE subdirs; FT still reads its
    # pretrained PT checkpoint from the PT subdir.
    outputs_root = str(get_value(cfg, "outputs_root", "outputs"))
    pi05_pt_root = project_root / outputs_root / str(get_value(cfg, "pi05_outputs_root", "pi05_PT"))
    pi05_ft_root = project_root / outputs_root / str(get_value(cfg, "pi05_ft_outputs_root", "pi05_FT"))

    pt_dataset = str(get_value(cfg, "pt_dataset", "libero_90", env="PT_DATASET"))
    pt_dataset_root = str(get_value(cfg, "pt_dataset_root", get_value(cfg, "dataset_root", "libero_dataset"), env="PT_DATASET_ROOT"))
    pt_batch_size = int(get_value(cfg, "pt_batch_size", 32, env="PT_BATCH_SIZE"))
    pt_num_gpus = int(get_value(cfg, "pt_num_gpus", 1, env="PT_NUM_GPUS"))
    pt_exp = str(get_value(cfg, "pt_exp", "exp1", env="PT_EXP")).strip()
    # freeze + LoRA run-name tag: {freeze_vision}{freeze_language}_{lora_llm}{lora_vision}, e.g. tt_ff.
    pt_freeze_vis = as_bool(get_value(cfg, "pt_freeze_vision_encoder", False, env="PI05_PT_FREEZE_VISION_ENCODER"))
    pt_freeze_lang = as_bool(get_value(cfg, "pt_freeze_language_model", False, env="PI05_PT_FREEZE_LANGUAGE_MODEL"))
    pt_lora_enable = as_bool(get_value(cfg, "pt_lora_enable", False, env="PI05_PT_LORA_ENABLE"))
    pt_lora_llm = as_bool(get_value(cfg, "pt_lora_llm", True, env="PI05_PT_LORA_LLM"))
    pt_lora_vision = as_bool(get_value(cfg, "pt_lora_vision", False, env="PI05_PT_LORA_VISION"))
    pt_tag = frz_lora_tag(pt_freeze_vis, pt_freeze_lang, pt_lora_enable, pt_lora_llm, pt_lora_vision)
    pt_run_name = run_name("PT", pt_dataset, pt_batch_size, pt_exp, tag=pt_tag)

    ft_dataset = str(get_value(cfg, "ft_dataset", "libero_10_op1_10", env="FT_DATASET"))
    ft_dataset_root = str(get_value(cfg, "ft_dataset_root", get_value(cfg, "dataset_root", "libero_dataset"), env="FT_DATASET_ROOT"))
    ft_batch_size = int(get_value(cfg, "ft_batch_size", 32, env="FT_BATCH_SIZE"))
    ft_exp = str(get_value(cfg, "ft_exp", "exp2", env="FT_EXP")).strip()
    ft_pre_dataset = str(get_value(cfg, "ft_pretrained_dataset", pt_dataset, env="FT_PRETRAINED_DATASET"))
    ft_pre_batch = int(get_value(cfg, "ft_pretrained_batch_size", pt_batch_size, env="FT_PRETRAINED_BATCH_SIZE"))
    ft_pre_exp = str(get_value(cfg, "ft_pretrained_exp", pt_exp, env="FT_PRETRAINED_EXP")).strip()
    ft_pre_ckpt = str(get_value(cfg, "ft_pretrained_checkpoint", "050000", env="FT_PRETRAINED_CHECKPOINT"))
    # PT warm-start 소스: ft_pretrained_run_name(폴더명 그대로)이 있으면 그걸 쓰고, 없으면 레거시
    # 방식(dataset/batch/exp로 PT_{dataset}_pi05_batch{bs}[_{exp}] 재조립)으로 폴백.
    ft_pre_run_name = (str(get_value(cfg, "ft_pretrained_run_name", "", env="FT_PRETRAINED_RUN_NAME") or "").strip()
                       or run_name("PT", ft_pre_dataset, ft_pre_batch, ft_pre_exp, tag=pt_tag))
    # freeze + LoRA 태그 = {freeze_vision}{freeze_language}_{lora_llm}{lora_vision} (예: tt_ff)
    ft_freeze_vis = as_bool(get_value(cfg, "ft_freeze_vision_encoder", False, env="PI05_FT_FREEZE_VISION_ENCODER"))
    ft_freeze_lang = as_bool(get_value(cfg, "ft_freeze_language_model", False, env="PI05_FT_FREEZE_LANGUAGE_MODEL"))
    ft_lora_enable = as_bool(get_value(cfg, "ft_lora_enable", False, env="PI05_FT_LORA_ENABLE"))
    ft_lora_llm = as_bool(get_value(cfg, "ft_lora_llm", True, env="PI05_FT_LORA_LLM"))
    ft_lora_vision = as_bool(get_value(cfg, "ft_lora_vision", False, env="PI05_FT_LORA_VISION"))
    ft_frz_tag = frz_lora_tag(ft_freeze_vis, ft_freeze_lang, ft_lora_enable, ft_lora_llm, ft_lora_vision)
    ft_run_name = f"FT_{ft_pre_ckpt}PT_{ft_dataset}_pi05_batch{ft_batch_size}_{ft_frz_tag}"
    if ft_exp:
        ft_run_name = f"{ft_run_name}_{ft_exp}"

    eval_stage = str(get_value(cfg, "eval_stage", "FT", env="EVAL_STAGE")).upper()
    eval_checkpoint = str(get_value(cfg, "eval_checkpoint", "001000", env="CHECKPOINT"))
    eval_model = str(get_nonempty(cfg, "eval_model", "", env="MODEL")).strip()
    if not eval_model:
        eval_model = ft_run_name if eval_stage == "FT" else pt_run_name
    eval_target_task = str(get_value(cfg, "eval_target_task", "libero_90", env="TARGET_TASK"))
    eval_wandb_run = str(os.environ.get("WANDB_RUN_NAME", f"{eval_model}_{eval_checkpoint}_{eval_target_task}"))

    settings = {
        "project_root": project_root,
        "lerobot_root": lerobot_root,
        "python_bin": project_root / ".venv" / "bin" / "python",
        "train_bin": project_root / ".venv" / "bin" / "lerobot-train",
        "eval_bin": project_root / ".venv" / "bin" / "lerobot-eval",
        "pi05_pt_outputs_root": pi05_pt_root,
        "pi05_ft_outputs_root": pi05_ft_root,
        "pi_base": resolve_path(project_root, get_value(cfg, "pi_base", "models/pi05_base")),
        # PT
        "pt_dataset": pt_dataset,
        "pt_dataset_root": pt_dataset_root,
        "pt_dataset_dir": project_root / pt_dataset_root / pt_dataset,
        "pt_batch_size": pt_batch_size,
        "pt_num_gpus": pt_num_gpus,
        "pt_num_workers": int(get_value(cfg, "pt_num_workers", 4, env="PT_NUM_WORKERS")),
        "pt_exp": pt_exp,
        "pt_lr": float(get_value(cfg, "pt_lr_base", 2.5e-05, env="PT_LR_BASE")) * pt_num_gpus,
        "pt_steps": int(get_value(cfg, "pt_steps", 100000, env="PT_STEPS")),
        "pt_save_freq": int(get_value(cfg, "pt_save_freq", 5000, env="PT_SAVE_FREQ")),
        "pt_wandb_project": str(get_value(cfg, "pt_wandb_project", "VLA_posttrain", env="PT_WANDB_PROJECT")),
        "pt_run_name": pt_run_name,
        "pt_output_dir": pi05_pt_root / pt_run_name,
        # PT freeze probes (expert always trains): vision tower / Gemma LLM, independently.
        "pt_freeze_vision_encoder": pt_freeze_vis,
        "pt_freeze_language_model": pt_freeze_lang,
        # PT LoRA (trainable low-rank adapters on the FROZEN VLM parts). run-name tag: {frz}_{lora}.
        "pt_lora_enable": pt_lora_enable,
        "pt_lora_rank": int(get_value(cfg, "pt_lora_rank", 8, env="PI05_PT_LORA_RANK")),
        "pt_lora_alpha": float(get_value(cfg, "pt_lora_alpha", 16.0, env="PI05_PT_LORA_ALPHA")),
        "pt_lora_dropout": float(get_value(cfg, "pt_lora_dropout", 0.0, env="PI05_PT_LORA_DROPOUT")),
        "pt_lora_llm": pt_lora_llm,
        "pt_lora_vision": pt_lora_vision,
        "pt_lora_targets": str(get_value(cfg, "pt_lora_targets", "q,k,v,o", env="PI05_PT_LORA_TARGETS")),
        # FT
        "ft_dataset": ft_dataset,
        "ft_dataset_root": ft_dataset_root,
        "ft_dataset_dir": project_root / ft_dataset_root / ft_dataset,
        "ft_batch_size": ft_batch_size,
        "ft_num_workers": int(get_value(cfg, "ft_num_workers", 4, env="FT_NUM_WORKERS")),
        "ft_exp": ft_exp,
        "ft_lr": str(get_value(cfg, "ft_lr", 2.5e-05, env="FT_LR")),
        "ft_steps": int(get_value(cfg, "ft_steps", 5000, env="FT_STEPS")),
        "ft_save_freq": int(get_value(cfg, "ft_save_freq", 500, env="FT_SAVE_FREQ")),
        "ft_wandb_project": str(get_value(cfg, "ft_wandb_project", "VLA_Finetune", env="FT_WANDB_PROJECT")),
        "ft_run_name": ft_run_name,
        "ft_output_dir": pi05_ft_root / ft_run_name,                 # FT → pi05_FT
        "ft_pretrained_run_name": ft_pre_run_name,
        "ft_pretrained_checkpoint": ft_pre_ckpt,
        "ft_pretrained_model_path": pi05_pt_root / ft_pre_run_name / "checkpoints" / ft_pre_ckpt / "pretrained_model",  # PT source ← pi05_PT
        # FT freeze probes (expert always trains; run name carries them as the _{frz}_{lora} tag)
        "ft_freeze_vision_encoder": ft_freeze_vis,
        "ft_freeze_language_model": ft_freeze_lang,
        # FT LoRA
        "ft_lora_enable": ft_lora_enable,
        "ft_lora_rank": int(get_value(cfg, "ft_lora_rank", 8, env="PI05_FT_LORA_RANK")),
        "ft_lora_alpha": float(get_value(cfg, "ft_lora_alpha", 16.0, env="PI05_FT_LORA_ALPHA")),
        "ft_lora_dropout": float(get_value(cfg, "ft_lora_dropout", 0.0, env="PI05_FT_LORA_DROPOUT")),
        "ft_lora_llm": ft_lora_llm,
        "ft_lora_vision": ft_lora_vision,
        "ft_lora_targets": str(get_value(cfg, "ft_lora_targets", "q,k,v,o", env="PI05_FT_LORA_TARGETS")),
        # PT-forgetting probe (skillVLA FT와 동일 인프라/wandb 형식: probe/* + probe_forget/*):
        # FT 중 probe_every 스텝마다 PT 데이터셋 고정 배치를 forward-only 재측정. PT 데이터셋은
        # PT 체크포인트의 train_config.json(dataset.root/repo_id)에서 유도, 없으면 yaml 경로로 폴백.
        **_ft_probe_settings(cfg, project_root, ft_dataset_root, ft_pre_dataset,
                             pi05_pt_root / ft_pre_run_name / "checkpoints" / ft_pre_ckpt / "pretrained_model"),
        # Eval
        "eval_target_task": eval_target_task,
        "eval_task_ids": str(get_value(cfg, "eval_task_ids", "[0,1,2,3,4,5,6,7,8,9]", env="TASK_IDS")),
        "eval_n_episodes": int(get_value(cfg, "eval_n_episodes", 5, env="N_EPISODES")),
        "eval_episode_offset": int(get_value(cfg, "eval_episode_offset", 25, env="EPISODE_OFFSET")),
        "eval_n_action_steps": int(get_value(cfg, "eval_n_action_steps", 5, env="N_ACTION_STEPS")),
        "eval_batch_size": int(get_value(cfg, "eval_batch_size", 1, env="EVAL_BATCH_SIZE")),
        "eval_stage": eval_stage,
        "eval_checkpoint": eval_checkpoint,
        "eval_model": eval_model,
        "eval_policy_path": Path(os.environ.get("POLICY_PATH", str(
            (pi05_ft_root if eval_stage == "FT" else pi05_pt_root) / eval_model / "checkpoints" / eval_checkpoint / "pretrained_model"))),
        "eval_max_parallel_tasks": int(get_value(cfg, "eval_max_parallel_tasks", 1, env="MAX_PARALLEL_TASKS")),
        "eval_max_videos_per_task": int(get_value(cfg, "eval_max_videos_per_task", 1, env="MAX_VIDEOS_PER_TASK")),
        "eval_video_frame_stride": int(get_value(cfg, "eval_video_frame_stride", 2, env="VIDEO_FRAME_STRIDE")),
        "eval_video_fps": int(get_value(cfg, "eval_video_fps", 10, env="VIDEO_FPS")),
        "eval_wandb_project": str(get_value(cfg, "eval_wandb_project", "VLA_eval", env="WANDB_PROJECT")),
        "eval_wandb_run_name": eval_wandb_run,
    }
    settings.update(slurm_settings(cfg, "pt", cpus=16, mem="128G", time="48:00:00", qos="big_qos"))
    settings.update(slurm_settings(cfg, "ft", cpus=16, mem="128G", time="48:00:00", qos="pro6000_qos"))
    settings.update(slurm_settings(cfg, "eval", cpus=8, mem="32G", time="48:00:00", qos="base_qos"))
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
