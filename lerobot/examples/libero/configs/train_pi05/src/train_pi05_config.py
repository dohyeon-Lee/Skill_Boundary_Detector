#!/usr/bin/env python3
"""Config resolver for configs/train_pi05.

Emits shell exports for the pi05 PT/FT sbatch files. Evaluation has its own resolver:
pi05_eval/src/pi05_eval_config.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import runpy
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


def _load_global(path: Path) -> dict[str, Any]:
    """global_config.yaml plus its per-server overlay (configs/src/global_config_loader.py)."""
    loader = next(
        directory / "src" / "global_config_loader.py"
        for directory in Path(__file__).resolve().parents
        if (directory / "src" / "global_config_loader.py").is_file()
    )
    return runpy.run_path(str(loader))["load_global_config"](path)


def load_config(path: Path) -> dict[str, Any]:
    config_path = Path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # Merge the nearest global_config.yaml as a base (module cfg keys win for roots).
    gpath = _find_global(config_path.parent)
    if gpath is not None and gpath.resolve() != config_path.resolve():
        gcfg = _load_global(gpath)
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


# Stage-1-style nested blocks → the flat keys the rest of this resolver reads. ``stage: pt|ft``
# in the yaml says whose training block it is. Flat keys still work (older snapshots, eval yaml);
# a nested value wins when both are present.
_NESTED_KEYS = {
    ("training", "dataloader", "batch_size"): "{s}_batch_size",
    ("training", "dataloader", "workers"): "{s}_num_workers",
    ("training", "dataloader", "gpus"): "{s}_num_gpus",
    ("training", "optimizer", "base_lr"): {"pt": "pt_lr_base", "ft": "ft_lr"},
    ("training", "gradient_checkpointing"): "{s}_gradient_checkpointing",
    ("training", "schedule", "steps"): "{s}_steps",
    ("training", "schedule", "lr_mode"): "{s}_lr_mode",
    ("training", "schedule", "warmup_steps"): "{s}_warmup_steps",
    ("training", "schedule", "lr_decay_steps"): "{s}_decay_steps",
    ("training", "schedule", "decay_lr"): "{s}_decay_lr",
    ("training", "schedule", "log_every"): "{s}_log_freq",
    ("training", "schedule", "save_every"): "{s}_save_freq",
    ("logging", "wandb", "enable"): "{s}_wandb_enable",
    ("logging", "wandb", "project"): "{s}_wandb_project",
    ("slurm", "gres"): "{s}_gres",
    ("slurm", "cpus"): "{s}_cpus_per_task",
    ("slurm", "memory"): "{s}_mem",
    ("slurm", "time"): "{s}_time",
}
_FT_UNSUPPORTED = {"ft_num_gpus"}  # the FT job is launched as a single process


def flatten_stage_blocks(cfg: dict[str, Any]) -> dict[str, Any]:
    """Expand the nested training/logging/slurm blocks of a ``stage: pt|ft`` yaml."""
    nested_present = any(key in cfg for key in ("training", "logging", "slurm"))
    stage = str(cfg.get("stage", "") or "").strip().lower()
    if not nested_present:
        return cfg
    if stage not in {"pt", "ft"}:
        raise ValueError("A yaml with training/logging/slurm blocks must set stage: pt | ft.")
    flat = {k: v for k, v in cfg.items() if k not in {"training", "logging", "slurm"}}

    def walk(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, dict) and path not in _NESTED_KEYS:
            for key, value in node.items():
                walk(value, (*path, str(key)))
            return
        target = _NESTED_KEYS.get(path)
        if target is None:
            raise ValueError(f"Unsupported pi05 config key: {'.'.join(path)}")
        name = target[stage] if isinstance(target, dict) else target.format(s=stage)
        if name in _FT_UNSUPPORTED:
            raise ValueError(f"{'.'.join(path)} is not configurable for pi05 FT.")
        flat[name] = node

    for block in ("training", "logging", "slurm"):
        if block in cfg:
            walk(cfg[block], (block,))
    return flat


def step_label(step: str) -> str:
    """``030000`` -> ``30k`` (same convention as NewTask FT and relabeled_85k); others unchanged."""
    step = str(step).strip()
    if step.isdigit() and int(step) > 0 and int(step) % 1000 == 0:
        return f"{int(step) // 1000}k"
    return step


def run_name(dataset: str, batch_size: int, exp: str) -> str:
    """``bs{batch}_{dataset}[_{exp}]`` — the output group (pi05_PT / pi05_FT) already names the stage.
    Freeze probes and the LR schedule are not encoded; distinguish such runs with ``*_exp``."""
    name = f"bs{batch_size}_{dataset}"
    return f"{name}_{exp}" if exp else name


def grounding_mode(enabled: Any) -> str:
    """yaml true/false toggle → policy.proprio_grounding (none | episode_start_xyz)."""
    return "episode_start_xyz" if as_bool(enabled) else "none"


def _pt_freeze(pt_ckpt: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """FT inherits the freeze probes of the PT checkpoint it loads (train_config.json is the source
    of truth). train_config.json missing (PT not built yet) → the optional ft_freeze_* /
    ft_proprio_grounding yaml values."""
    p: dict = {}
    tc = pt_ckpt / "train_config.json"
    if tc.is_file():
        try:
            p = json.loads(tc.read_text()).get("policy") or {}
        except Exception:
            p = {}
    return {
        "freeze_vis": as_bool(p.get("freeze_vision_encoder", get_value(cfg, "ft_freeze_vision_encoder", False))),
        "freeze_lang": as_bool(p.get("freeze_language_model", get_value(cfg, "ft_freeze_language_model", False))),
        # The FT input must live in the PT checkpoint's state coordinates, so grounding is
        # inherited, never chosen. Checkpoints older than the field were trained ungrounded.
        # The action horizon is part of the loaded model; FT must not silently fall back to a default.
        "chunk_size": int(p.get("chunk_size", get_value(cfg, "ft_chunk_size", 10))),
        "proprio_grounding": (
            str(p.get("proprio_grounding", "none") or "none")
            if p
            else grounding_mode(get_value(cfg, "ft_proprio_grounding", False))
        ),
    }


def _auto_labels(model_dirs: list[str]) -> list[str]:
    """Distinguishing middle token(s) of each model_dir (strip the common leading + trailing _-tokens),
    used as the side-by-side panel label when a model has no explicit label. (Same as stage2_eval.)"""
    if len(model_dirs) <= 1:
        return list(model_dirs)
    toks = [d.split("_") for d in model_dirs]
    short = min(len(t) for t in toks)
    p = 0
    while p < short and all(t[p] == toks[0][p] for t in toks):
        p += 1
    s = 0
    while s < short - p and all(t[-1 - s] == toks[0][-1 - s] for t in toks):
        s += 1
    return ["_".join(t[p: len(t) - s]) or d for t, d in zip(toks, model_dirs)]


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
    cfg = flatten_stage_blocks(cfg)
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    lerobot_root = project_root / "lerobot"
    pi_base = Path(
        resolve_path(project_root, get_value(cfg, "pi_base", "models/pi05_base"))
    )
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
    # {project_root}/{outputs_root}/{pi05_PT|pi05_FT}: outputs_root comes from global_config.yaml
    # (switchable to outputs_filtered). PT and FT write to SEPARATE subdirs; FT still reads its
    # pretrained PT checkpoint from the PT subdir.
    outputs_root = str(get_value(cfg, "outputs_root", "outputs"))
    pi05_pt_root = project_root / outputs_root / str(get_value(cfg, "pi05_outputs_root", "pi05_PT"))
    pi05_ft_root = project_root / outputs_root / str(get_value(cfg, "pi05_ft_outputs_root", "pi05_FT"))

    pt_dataset = str(get_value(cfg, "pt_dataset", "libero_90", env="PT_DATASET"))
    pt_dataset_root = str(get_value(cfg, "pt_dataset_root", get_value(cfg, "dataset_root", "libero_dataset"), env="PT_DATASET_ROOT"))
    pt_dataset_dir = project_root / pt_dataset_root / pt_dataset
    pt_batch_size = int(get_value(cfg, "pt_batch_size", 32, env="PT_BATCH_SIZE"))
    pt_chunk_size = int(get_value(cfg, "pt_chunk_size", 10, env="PT_CHUNK_SIZE"))
    if pt_chunk_size < 1:
        raise ValueError(f"pt_chunk_size must be >= 1, got {pt_chunk_size}")
    pt_num_gpus = int(get_value(cfg, "pt_num_gpus", 1, env="PT_NUM_GPUS"))
    pt_exp = str(get_value(cfg, "pt_exp", "exp1", env="PT_EXP")).strip()
    pt_lr_mode = str(
        get_value(cfg, "pt_lr_mode", "cosine_decay", env="PT_LR_MODE")
    ).strip().lower()
    if pt_lr_mode not in {"cosine_decay", "warmup_constant"}:
        raise ValueError(
            "pt_lr_mode must be 'cosine_decay' or 'warmup_constant', got "
            f"{pt_lr_mode!r}."
        )
    pt_warmup_steps = int(
        get_value(cfg, "pt_warmup_steps", 1000, env="PT_WARMUP_STEPS")
    )
    pt_decay_steps = int(
        get_value(cfg, "pt_decay_steps", 30000, env="PT_DECAY_STEPS")
    )
    pt_decay_lr = float(
        get_value(cfg, "pt_decay_lr", 2.5e-6, env="PT_DECAY_LR")
    )
    if pt_warmup_steps < 0:
        raise ValueError("pt_warmup_steps must be non-negative.")
    if pt_decay_steps <= 0:
        raise ValueError("pt_decay_steps must be positive.")
    if not math.isfinite(pt_decay_lr) or pt_decay_lr <= 0.0:
        raise ValueError("pt_decay_lr must be finite and positive.")
    # Freeze probes (the action expert always trains). They are not part of the run name.
    pt_freeze_vis = as_bool(get_value(cfg, "pt_freeze_vision_encoder", False, env="PI05_PT_FREEZE_VISION_ENCODER"))
    pt_freeze_lang = as_bool(get_value(cfg, "pt_freeze_language_model", False, env="PI05_PT_FREEZE_LANGUAGE_MODEL"))
    pt_run_name = run_name(pt_dataset, pt_batch_size, pt_exp)

    ft_dataset = str(get_value(cfg, "ft_dataset", "libero_10_op1_10", env="FT_DATASET"))
    ft_dataset_root = str(get_value(cfg, "ft_dataset_root", get_value(cfg, "dataset_root", "libero_dataset"), env="FT_DATASET_ROOT"))
    ft_batch_size = int(get_value(cfg, "ft_batch_size", 32, env="FT_BATCH_SIZE"))
    ft_exp = str(get_value(cfg, "ft_exp", "exp2", env="FT_EXP")).strip()
    ft_pre_dataset = str(get_value(cfg, "ft_pretrained_dataset", pt_dataset, env="FT_PRETRAINED_DATASET"))
    ft_pre_batch = int(get_value(cfg, "ft_pretrained_batch_size", pt_batch_size, env="FT_PRETRAINED_BATCH_SIZE"))
    ft_pre_exp = str(get_value(cfg, "ft_pretrained_exp", pt_exp, env="FT_PRETRAINED_EXP")).strip()
    ft_pre_ckpt = str(get_value(cfg, "ft_pretrained_checkpoint", "050000", env="FT_PRETRAINED_CHECKPOINT"))
    # PT warm-start source: ft_pretrained_run_name (folder name as-is); blank → rebuild it from the
    # ft_pretrained_{dataset,batch_size,exp} keys with the current naming rule.
    ft_pre_run_name = (str(get_value(cfg, "ft_pretrained_run_name", "", env="FT_PRETRAINED_RUN_NAME") or "").strip()
                       or run_name(ft_pre_dataset, ft_pre_batch, ft_pre_exp))
    ft_pt_ckpt = pi05_pt_root / ft_pre_run_name / "checkpoints" / ft_pre_ckpt / "pretrained_model"
    _pf = _pt_freeze(ft_pt_ckpt, cfg)
    ft_freeze_vis, ft_freeze_lang = _pf["freeze_vis"], _pf["freeze_lang"]
    ft_run_name = f"{run_name(ft_dataset, ft_batch_size, '')}_PT{step_label(ft_pre_ckpt)}"
    # Same schedule knobs as PT. Resolver default stays cosine_decay (older flat yamls);
    # the shipped FT yaml selects warmup_constant like PT.
    ft_lr_mode = str(get_value(cfg, "ft_lr_mode", "cosine_decay", env="FT_LR_MODE")).strip().lower()
    if ft_lr_mode not in {"cosine_decay", "warmup_constant"}:
        raise ValueError(f"ft lr_mode must be 'cosine_decay' or 'warmup_constant', got {ft_lr_mode!r}.")
    ft_warmup_steps = int(get_value(cfg, "ft_warmup_steps", 1000, env="FT_WARMUP_STEPS"))
    ft_decay_steps = int(get_value(cfg, "ft_decay_steps", 30000, env="FT_DECAY_STEPS"))
    ft_decay_lr = float(get_value(cfg, "ft_decay_lr", 2.5e-6, env="FT_DECAY_LR"))
    if ft_warmup_steps < 0 or ft_decay_steps <= 0 or not math.isfinite(ft_decay_lr) or ft_decay_lr <= 0.0:
        raise ValueError("Invalid FT schedule: warmup_steps >= 0, lr_decay_steps > 0, decay_lr > 0.")
    if ft_exp:
        ft_run_name = f"{ft_run_name}_{ft_exp}"

    settings = {
        "project_root": project_root,
        "lerobot_root": lerobot_root,
        "python_bin": project_root / ".venv" / "bin" / "python",
        "train_bin": project_root / ".venv" / "bin" / "lerobot-train",
        "pi05_pt_outputs_root": pi05_pt_root,
        "pi05_ft_outputs_root": pi05_ft_root,
        "pi_base": pi_base,
        "pi05_tokenizer_path": pi05_tokenizer_path,
        # PT
        "pt_dataset": pt_dataset,
        "pt_dataset_root": pt_dataset_root,
        "pt_dataset_dir": pt_dataset_dir,
        "pt_batch_size": pt_batch_size,
        "pt_chunk_size": pt_chunk_size,
        "pt_num_gpus": pt_num_gpus,
        "pt_num_workers": int(get_value(cfg, "pt_num_workers", 4, env="PT_NUM_WORKERS")),
        "pt_exp": pt_exp,
        "pt_lr": float(get_value(cfg, "pt_lr_base", 2.5e-05, env="PT_LR_BASE")) * pt_num_gpus,
        "pt_lr_mode": pt_lr_mode,
        "pt_warmup_steps": pt_warmup_steps,
        "pt_decay_steps": pt_decay_steps,
        "pt_decay_lr": pt_decay_lr,
        "pt_steps": int(get_value(cfg, "pt_steps", 100000, env="PT_STEPS")),
        "pt_save_freq": int(get_value(cfg, "pt_save_freq", 5000, env="PT_SAVE_FREQ")),
        "pt_log_freq": int(get_value(cfg, "pt_log_freq", 100, env="PT_LOG_FREQ")),
        "pt_wandb_enable": as_bool(get_value(cfg, "pt_wandb_enable", True, env="PT_WANDB_ENABLE")),
        "pt_wandb_project": str(get_value(cfg, "pt_wandb_project", "VLA_posttrain", env="PT_WANDB_PROJECT")),
        "pt_run_name": pt_run_name,
        "pt_output_dir": pi05_pt_root / pt_run_name,
        # PT freeze probes (expert always trains): vision tower / Gemma LLM, independently.
        "pt_freeze_vision_encoder": pt_freeze_vis,
        "pt_freeze_language_model": pt_freeze_lang,
        "pt_gradient_checkpointing": as_bool(
            get_value(cfg, "pt_gradient_checkpointing", True, env="PI05_PT_GRADIENT_CHECKPOINTING")
        ),
        "pt_proprio_grounding": grounding_mode(
            get_value(cfg, "pt_proprio_grounding", False, env="PI05_PT_PROPRIO_GROUNDING")
        ),
        # FT
        "ft_dataset": ft_dataset,
        "ft_dataset_root": ft_dataset_root,
        "ft_dataset_dir": project_root / ft_dataset_root / ft_dataset,
        "ft_batch_size": ft_batch_size,
        "ft_num_workers": int(get_value(cfg, "ft_num_workers", 4, env="FT_NUM_WORKERS")),
        "ft_exp": ft_exp,
        "ft_lr": str(get_value(cfg, "ft_lr", 2.5e-05, env="FT_LR")),
        "ft_lr_mode": ft_lr_mode,
        "ft_warmup_steps": ft_warmup_steps,
        "ft_decay_steps": ft_decay_steps,
        "ft_decay_lr": ft_decay_lr,
        "ft_chunk_size": _pf["chunk_size"],
        "ft_steps": int(get_value(cfg, "ft_steps", 5000, env="FT_STEPS")),
        "ft_save_freq": int(get_value(cfg, "ft_save_freq", 500, env="FT_SAVE_FREQ")),
        "ft_log_freq": int(get_value(cfg, "ft_log_freq", 100, env="FT_LOG_FREQ")),
        "ft_wandb_enable": as_bool(get_value(cfg, "ft_wandb_enable", True, env="FT_WANDB_ENABLE")),
        "ft_wandb_project": str(get_value(cfg, "ft_wandb_project", "VLA_Finetune", env="FT_WANDB_PROJECT")),
        "ft_run_name": ft_run_name,
        "ft_output_dir": pi05_ft_root / ft_run_name,                 # FT → pi05_FT
        "ft_pretrained_run_name": ft_pre_run_name,
        "ft_pretrained_checkpoint": ft_pre_ckpt,
        "ft_pretrained_model_path": pi05_pt_root / ft_pre_run_name / "checkpoints" / ft_pre_ckpt / "pretrained_model",  # PT source ← pi05_PT
        # FT freeze probes, inherited from the PT checkpoint (expert always trains)
        "ft_freeze_vision_encoder": ft_freeze_vis,
        "ft_freeze_language_model": ft_freeze_lang,
        "ft_proprio_grounding": _pf["proprio_grounding"],
        "ft_gradient_checkpointing": as_bool(
            get_value(cfg, "ft_gradient_checkpointing", True, env="PI05_FT_GRADIENT_CHECKPOINTING")
        ),
        # PT-forgetting probe (skillVLA FT와 동일 인프라/wandb 형식: probe/* + probe_forget/*):
        # FT 중 probe_every 스텝마다 PT 데이터셋 고정 배치를 forward-only 재측정. PT 데이터셋은
        # PT 체크포인트의 train_config.json(dataset.root/repo_id)에서 유도, 없으면 yaml 경로로 폴백.
        **_ft_probe_settings(cfg, project_root, ft_dataset_root, ft_pre_dataset,
                             pi05_pt_root / ft_pre_run_name / "checkpoints" / ft_pre_ckpt / "pretrained_model"),
    }
    settings.update(slurm_settings(cfg, "pt", cpus=16, mem="128G", time="48:00:00", qos="big_qos"))
    settings.update(slurm_settings(cfg, "ft", cpus=16, mem="128G", time="48:00:00", qos="pro6000_qos"))
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
