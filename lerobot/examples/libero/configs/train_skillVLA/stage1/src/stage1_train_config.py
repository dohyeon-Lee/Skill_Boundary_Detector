#!/usr/bin/env python3
"""Config for SkillVLA Stage-1 training (standalone action expert, policy.type=skill_expert).

Trains the action expert by flow matching on the build_data skillvla dataset (raw images +
skill_sequence/skill_index + actions). The data run's FSQ.pt supplies the exact action-expert
contract and warm-start weights; no DINO-token / VLM artifacts are needed here. The expert
encodes raw images with its own DINOv3 and reads the GT skill code from the dataset. Emits
shell exports (--shell).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import as_bool, as_levels, as_list, get_value, load_config, print_shell, resolve_path  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "stage1_train_config.yaml"


def _normalize_config(cfg: dict) -> dict:
    """Accept the compact grouped YAML while retaining flat-key config compatibility."""
    out = dict(cfg)
    nested_keys = {
        "skillvla_dataset_root": ("dataset", "skillvla_root"),
        "source_dataset": ("dataset", "source"),
        "run_tag": ("dataset", "run"),
        "fsq_path": ("initialization", "fsq_path"),
        "init_from_pi05": ("initialization", "pi05"),
        "pi_base": ("initialization", "pi_base"),
        "vision_backbone": ("vision", "backbone"),
        "freeze_vision_encoder": ("vision", "freeze"),
        "dino_model_path": ("vision", "dino_model"),
        "dino_lr": ("vision", "dino_lr"),
        "siglip_lr": ("vision", "siglip_lr"),
        "siglip_image_size": ("vision", "siglip_size"),
        "lora_expert": ("expert", "lora", "enabled"),
        "lora_targets": ("expert", "lora", "targets"),
        "lora_rank": ("expert", "lora", "rank"),
        "lora_alpha": ("expert", "lora", "alpha"),
        "lora_dropout": ("expert", "lora", "dropout"),
        "lora_lr_scale": ("expert", "lora", "lr_scale"),
        "image_free_lora_prob": ("regime", "image_free", "probability"),
        "image_free_lora_anchor_weight": ("regime", "image_free", "anchor_weight"),
        "n_action_steps": ("execution", "action_steps"),
        "transition_jitter_enabled": ("transition_randomization", "enabled"),
        "skill_start_loss_weight": ("loss", "skill_start_weight"),
        "skill_end_loss_weight": ("loss", "skill_end_weight"),
        "action_weight": ("loss", "weighted"),
        "train_terminator": ("terminator", "train"),
        "terminator_freeze_vision_encoder": ("terminator", "freeze_vision"),
        "terminator_end_target_sigma": ("terminator", "target_sigma"),
        "terminator_end_pos_weight": ("terminator", "end_weight"),
        "terminator_lr_scale": ("terminator", "lr_scale"),
        "terminator_dino_model_path": ("terminator", "dino_model"),
        "exp": ("run", "suffix"),
        "batch_size": ("training", "dataloader", "batch_size"),
        "num_workers": ("training", "dataloader", "workers"),
        "num_gpus": ("training", "dataloader", "gpus"),
        "lr_base": ("training", "optimizer", "base_lr"),
        "steps": ("training", "schedule", "steps"),
        "log_freq": ("training", "schedule", "log_every"),
        "save_freq": ("training", "schedule", "save_every"),
        "wandb_enable": ("logging", "wandb", "enable"),
        "wandb_project": ("logging", "wandb", "project"),
        "train_partition": ("slurm", "partition"),
        "train_qos": ("slurm", "qos"),
        "train_gres": ("slurm", "gres"),
        "train_cpus_per_task": ("slurm", "cpus"),
        "train_mem": ("slurm", "memory"),
        "train_time": ("slurm", "time"),
        "train_nodelist": ("slurm", "nodes"),
        "train_exclude_nodes": ("slurm", "exclude_nodes"),
    }
    for flat_key, path in nested_keys.items():
        if flat_key in out:
            continue
        value = cfg
        for part in path:
            if not isinstance(value, dict) or part not in value:
                break
            value = value[part]
        else:
            out[flat_key] = value
    if isinstance(out.get("lora_targets"), (list, tuple)):
        out["lora_targets"] = ",".join(str(v) for v in out["lora_targets"])
    if str(out.get("n_action_steps", "")).strip().lower() in ("auto", "fsq"):
        out["n_action_steps"] = None
    return out


def _blank(value: Any) -> bool:
    return value is None or str(value).strip().lower() in ("", "null", "none")


def _lora_targets_run_suffix(spec: str) -> str:
    """Keep attention-only run paths stable while target-expanded runs cannot overwrite them."""
    tokens = [token.strip().lower() for token in spec.split(",") if token.strip()]
    if tokens == ["q", "k", "v", "o"]:
        return ""
    tag = "-".join(token.replace("_", "") for token in tokens)
    return f"_lt{tag}" if tag else ""


def _anchor_weight_run_suffix(weight: float) -> str:
    """Keep the historical weight=1 run path while separating relaxed B-anchor sweeps."""
    if weight == 1.0:
        return ""
    return f"_aw{weight:g}".replace(".", "p")


def _action_weight_run_suffix(start: float, end: float) -> str:
    """Preserve historical 1→3 run names while separating every other weighting range."""
    if start == 1.0 and end == 3.0:
        return ""
    return f"_sw{start:g}_ew{end:g}".replace(".", "p")


def _localize_model_path(project_root: Path, value: Any, default: str) -> Path:
    raw = str(value if not _blank(value) else default).strip()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        return resolve_path(project_root, raw)
    if path.exists():
        return path
    parts = path.parts
    if "models" in parts:
        idx = parts.index("models")
        return project_root.joinpath(*parts[idx:])
    return path


def _load_fsq_config(fsq_path: Path):
    fsq_path = Path(fsq_path).expanduser()
    if not fsq_path.exists():
        raise FileNotFoundError(
            f"FSQ checkpoint not found: {fsq_path}. "
            "Stage-1 derives its action-expert contract from this checkpoint; build/copy the "
            "SkillVLA data run first or set fsq_path explicitly."
        )
    import torch  # noqa: PLC0415

    libero_dir = _HERE.parents[4]
    if str(libero_dir) not in sys.path:
        sys.path.insert(0, str(libero_dir))
    from FSQ import _checkpoint_config  # noqa: PLC0415

    checkpoint = torch.load(str(fsq_path), map_location="cpu", weights_only=False)
    return _checkpoint_config(checkpoint)


def _from_fsq_or_override(cfg: dict, name: str, fsq_value, *, cast=None):
    value = get_value(cfg, name, None)
    if _blank(value):
        return fsq_value
    parsed = cast(value) if cast is not None else value
    if parsed != fsq_value:
        raise ValueError(
            f"{name} is derived from FSQ.pt and must match it. "
            f"YAML={parsed!r}, FSQ={fsq_value!r}."
        )
    return parsed


def build_settings(cfg: dict) -> dict:
    cfg = _normalize_config(cfg)
    # Standalone: every root is declared in this yaml (no build_data dependency).
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    dataset_root = project_root / str(get_value(cfg, "dataset_root", "dataset"))
    skillvla_root = dataset_root / str(get_value(cfg, "skillvla_dataset_root", "skillvla_dataset"))
    lerobot_root = project_root / "lerobot"

    source_dataset = str(get_value(cfg, "source_dataset"))
    run_tag = str(get_value(cfg, "run_tag"))
    run_dir = skillvla_root / source_dataset / run_tag
    fsq_path_raw = str(get_value(cfg, "fsq_path", "")).strip()
    fsq_path = (resolve_path(project_root, fsq_path_raw)
                if fsq_path_raw and fsq_path_raw.lower() not in ("null", "none") else run_dir / "FSQ.pt")
    fsq_cfg = _load_fsq_config(fsq_path)
    transition_jitter_enabled = as_bool(get_value(cfg, "transition_jitter_enabled", True))
    dataset_info_path = run_dir / "skillvla" / "meta" / "info.json"
    if transition_jitter_enabled:
        if not dataset_info_path.is_file():
            raise FileNotFoundError(
                f"Transition randomization needs the built dataset metadata: {dataset_info_path}")
        dataset_info = json.loads(dataset_info_path.read_text())
        transition_jitter_pmax = int(dataset_info.get("skill_pmax", -1))
        if transition_jitter_pmax < 0:
            raise ValueError(f"Dataset metadata has no valid skill_pmax: {dataset_info_path}")
        transition_jitter_distribution = str(
            dataset_info.get("skill_jitter_distribution", "half_normal")
        ).strip().lower().replace("-", "_").replace(" ", "_")
        if transition_jitter_distribution not in {"half_normal", "uniform"}:
            raise ValueError(
                f"Dataset metadata has invalid skill_jitter_distribution: {dataset_info_path} "
                f"({transition_jitter_distribution!r})"
            )
    else:
        transition_jitter_pmax = 0
        transition_jitter_distribution = "half_normal"

    batch_size = int(get_value(cfg, "batch_size", 32))
    num_gpus = int(get_value(cfg, "num_gpus", 1))
    lr_base = float(get_value(cfg, "lr_base", 2.5e-05))
    exp = str(get_value(cfg, "exp", "")).strip()

    # FSQ action reconstructor -> Stage-1 action expert is an exact warm start, so the expert
    # tensor contract is derived from the checkpoint. YAML may only restate the same value.
    action_expert_variant = str(_from_fsq_or_override(
        cfg, "action_expert_variant", fsq_cfg.action_expert_variant))
    state_cond_mode = str(_from_fsq_or_override(
        cfg, "state_cond_mode", fsq_cfg.state_cond_mode)).strip().lower()
    chunk_size = int(_from_fsq_or_override(cfg, "chunk_size", int(fsq_cfg.chunk_size), cast=int))
    max_state_dim = int(_from_fsq_or_override(cfg, "max_state_dim", int(fsq_cfg.max_state_dim), cast=int))
    max_action_dim = int(_from_fsq_or_override(cfg, "max_action_dim", int(fsq_cfg.max_action_dim), cast=int))
    min_period = float(_from_fsq_or_override(cfg, "min_period", float(fsq_cfg.min_period), cast=float))
    max_period = float(_from_fsq_or_override(cfg, "max_period", float(fsq_cfg.max_period), cast=float))
    time_sampling_beta_alpha = float(_from_fsq_or_override(
        cfg, "time_sampling_beta_alpha", float(fsq_cfg.time_sampling_beta_alpha), cast=float))
    time_sampling_beta_beta = float(_from_fsq_or_override(
        cfg, "time_sampling_beta_beta", float(fsq_cfg.time_sampling_beta_beta), cast=float))
    time_sampling_scale = float(_from_fsq_or_override(
        cfg, "time_sampling_scale", float(fsq_cfg.time_sampling_scale), cast=float))
    time_sampling_offset = float(_from_fsq_or_override(
        cfg, "time_sampling_offset", float(fsq_cfg.time_sampling_offset), cast=float))
    n_action_steps = get_value(cfg, "n_action_steps", None)
    n_action_steps = chunk_size if n_action_steps in (None, "", "null") else int(n_action_steps)

    cond_encoder_variant = str(get_value(cfg, "cond_encoder_variant", "")).strip()
    if cond_encoder_variant.lower() in ("none", "null"):  # blank yaml → omit (use action expert's variant)
        cond_encoder_variant = ""
    skill_start_w = float(get_value(cfg, "skill_start_loss_weight", 1.0))  # start weighting (action_weight only)
    skill_end_w = float(get_value(cfg, "skill_end_loss_weight", 1.0))      # end weighting (action_weight only)
    action_weight = as_bool(get_value(cfg, "action_weight", False))        # per-sample sw-weight the action MSE
    if skill_start_w <= 0.0 or skill_end_w <= 0.0:
        raise ValueError(
            "skill_start_loss_weight and skill_end_loss_weight must both be > 0 "
            f"(got {skill_start_w} and {skill_end_w})."
        )
    train_terminator = as_bool(get_value(cfg, "train_terminator", False))
    term_freeze_source = get_value(cfg, "terminator_freeze_vision_encoder", None)
    terminator_freeze_vision_encoder = (
        bool(getattr(fsq_cfg, "freeze_vision_encoder", True))
        if _blank(term_freeze_source)
        else as_bool(term_freeze_source)
    )

    # Stage-1 adapts the FSQ action expert either through LoRA (frozen base) or full fine-tuning. A batches
    # train against actions; B batches have NO image/cond and anchor the image-free expert to frozen FSQ.
    lora_expert = as_bool(get_value(cfg, "lora_expert", False))
    lora_rank = int(get_value(cfg, "lora_rank", 8))
    lora_alpha_raw = get_value(cfg, "lora_alpha", "auto")
    lora_alpha = (2.0 * lora_rank if _blank(lora_alpha_raw) or str(lora_alpha_raw).strip().lower() == "auto"
                  else float(lora_alpha_raw))
    lora_dropout = float(get_value(cfg, "lora_dropout", 0.0))
    lora_targets = str(get_value(cfg, "lora_targets", "q,k,v,o"))
    lora_lr_scale = float(get_value(cfg, "lora_lr_scale", 1.0))
    image_free_lora_prob = float(get_value(cfg, "image_free_lora_prob", 0.0) or 0.0)
    image_free_lora_anchor_weight = float(get_value(cfg, "image_free_lora_anchor_weight", 1.0))
    if not 0.0 <= image_free_lora_prob < 1.0:
        raise ValueError(f"image_free_lora_prob must be in [0, 1), got {image_free_lora_prob}.")
    if image_free_lora_anchor_weight <= 0.0:
        raise ValueError("image_free_lora_anchor_weight must be > 0.")

    init_from_pi05 = as_bool(get_value(cfg, "init_from_pi05", True))
    pi_base_source = get_value(cfg, "pi_base", None)
    if _blank(pi_base_source):
        pi_base_source = getattr(fsq_cfg, "pi_base", "models/pi05_base")
    pi_base = _localize_model_path(project_root, pi_base_source, "models/pi05_base") if init_from_pi05 else ""

    dino_lr = get_value(cfg, "dino_lr", None)
    dino_lr_str = "" if dino_lr in (None, "", "null") else str(dino_lr)
    siglip_lr = get_value(cfg, "siglip_lr", None)
    siglip_lr_str = "" if siglip_lr in (None, "", "null") else str(siglip_lr)

    vision_backbone = (str(get_value(cfg, "vision_backbone", "dino")).strip().lower() or "dino")
    freeze_vision_encoder = as_bool(get_value(cfg, "freeze_vision_encoder", False))  # SELECTED backbone; frozen → "_freeze" suffix

    # run-name: run_tag + backbone[_freeze] + batch + state-cond [+ LoRA regime] [+ weighted] [+ exp]. source_dataset /
    # chunk_size are OMITTED (fixed — never swept). freeze_vision_encoder → "_freeze" ONLY when frozen
    # (trainable = plain backbone → back-compat with existing "..._siglip_batch..." runs). The Stage-2
    # parser already recognizes the "_freeze" suffix. action_weight → "_weighted" ONLY when true.
    backbone_tag = f"{vision_backbone}_freeze" if freeze_vision_encoder else vision_backbone
    run_name = f"{run_tag}_{backbone_tag}_batch{batch_size}_{state_cond_mode}"
    if lora_expert:
        run_name = f"{run_name}_lorae{lora_rank}{_lora_targets_run_suffix(lora_targets)}"
        if image_free_lora_prob > 0.0:
            run_name = f"{run_name}_if{int(round(image_free_lora_prob * 100))}"
            run_name = f"{run_name}{_anchor_weight_run_suffix(image_free_lora_anchor_weight)}"
    else:
        run_name = f"{run_name}_expertft"
        if image_free_lora_prob > 0.0:
            run_name = f"{run_name}_if{int(round(image_free_lora_prob * 100))}"
            run_name = f"{run_name}{_anchor_weight_run_suffix(image_free_lora_anchor_weight)}"
    if action_weight:
        run_name = f"{run_name}_weighted{_action_weight_run_suffix(skill_start_w, skill_end_w)}"
    if train_terminator and not terminator_freeze_vision_encoder:
        run_name = f"{run_name}_termvis_tuned"
    if transition_jitter_pmax > 0:
        run_name = f"{run_name}_tj{transition_jitter_pmax}"
        if transition_jitter_distribution != "half_normal":
            run_name = f"{run_name}_{transition_jitter_distribution}"
    if exp:
        run_name = f"{run_name}_{exp}"
    # Single outputs root from yaml; the per-stage subdir is fixed here (not in yaml).
    outputs_root = project_root / str(get_value(cfg, "outputs_root", "outputs"))
    vla_root = outputs_root / "skillVLA_stage1"
    output_dir = vla_root / run_name   # under skillVLA_stage1/, so no extra stage prefix

    skill_fsq_levels = list(_from_fsq_or_override(
        cfg, "skill_fsq_levels", list(fsq_cfg.fsq_levels), cast=lambda v: list(as_levels(v))))
    vocab_default = 1
    for _lvl in skill_fsq_levels:
        vocab_default *= _lvl
    skill_vocab_size = int(get_value(cfg, "skill_vocab_size", vocab_default))

    # ── Co-trained terminator (ONLINE DINO): use the DINO backbone recorded by THIS FSQ checkpoint.
    # A yaml override is allowed, but defaulting to a different DINO size silently makes checkpoint
    # weights incompatible (and may point at a config-only local model directory).
    term_dino_source = get_value(cfg, "terminator_dino_model_path", None)
    if _blank(term_dino_source):
        term_dino_source = getattr(fsq_cfg, "dino_model_path", "models/dinov3-vits16")
    terminator_dino_model_path = _localize_model_path(
        project_root, term_dino_source, "models/dinov3-vits16")

    settings: dict = {
        # roots
        "project_root": project_root,
        "lerobot_root": lerobot_root,
        # dataset (raw skillvla dataset: images + skill columns + actions)
        "source_dataset": source_dataset,
        "run_tag": run_tag,
        "skillvla_dataset_dir": run_dir / "skillvla",
        "repo_id": f"dohyeon/{source_dataset}",
        # conditioning (joint only; the FSQ checkpoint fixes the skill route)
        "cond_encoder_variant": cond_encoder_variant,  # "" → same as action_expert_variant
        "state_cond_mode": state_cond_mode,       # state (prefix) | state_skill (AdaRMS) | broadcast
        "action_expert_variant": action_expert_variant,
        # model init
        "pi_base": pi_base,                       # "" → train the expert from scratch
        # vision encoder: "dino" or "siglip" (siglip warm-starts from pi_base's vision_tower)
        "vision_backbone": str(get_value(cfg, "vision_backbone", "dino")),
        "dino_model_path": resolve_path(project_root, get_value(cfg, "dino_model_path", "models/dinov3-vits16")),
        "dino_lr": dino_lr_str,                   # "" → same LR as the rest
        "freeze_vision_encoder": freeze_vision_encoder,  # freeze the SELECTED backbone (dino|siglip)
        "siglip_lr": siglip_lr_str,               # "" → same LR as the rest (when unfrozen)
        "siglip_image_size": int(get_value(cfg, "siglip_image_size", 224)),
        "skill_vocab_size": skill_vocab_size,
        "skill_fsq_levels": "[" + ",".join(str(x) for x in skill_fsq_levels) + "]",
        "transition_jitter_pmax": transition_jitter_pmax,
        "transition_jitter_distribution": transition_jitter_distribution,
        "max_state_dim": max_state_dim,
        "max_action_dim": max_action_dim,
        "min_period": min_period,
        "max_period": max_period,
        "time_sampling_beta_alpha": time_sampling_beta_alpha,
        "time_sampling_beta_beta": time_sampling_beta_beta,
        "time_sampling_scale": time_sampling_scale,
        "time_sampling_offset": time_sampling_offset,
        # frozen FSQ expert + image-steering LoRA
        "lora_expert": lora_expert,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_targets": lora_targets,
        "lora_lr_scale": lora_lr_scale,
        "image_free_lora_prob": image_free_lora_prob,
        "image_free_lora_anchor_weight": image_free_lora_anchor_weight,
        # output
        "skillvla_outputs_root": vla_root,
        "pt_run_name": run_name,
        "pt_output_dir": output_dir,
        # action chunk horizon
        "chunk_size": chunk_size,
        "n_action_steps": n_action_steps,
        # loss: A-only linear endpoint weighting from skill start to skill end. B remains uniform.
        "skill_start_loss_weight": skill_start_w,
        "skill_end_loss_weight": skill_end_w,
        "action_weight": action_weight,          # per-sample sw-weight the action MSE → wandb loss_weighted
        # co-trained FSQ terminator (gradient-disjoint; AUTO-derived from run_tag)
        "train_terminator": train_terminator,
        "terminator_freeze_vision_encoder": terminator_freeze_vision_encoder,
        "fsq_path": fsq_path,                     # {run_dir}/FSQ.pt (warm-start the terminator)
        "terminator_end_target_sigma": float(get_value(cfg, "terminator_end_target_sigma", 1.0)),
        "terminator_end_pos_weight": float(get_value(cfg, "terminator_end_pos_weight", 1.0)),
        "terminator_lr_scale": float(get_value(cfg, "terminator_lr_scale", 1.0)),
        "terminator_dino_model_path": terminator_dino_model_path,   # ONLINE DINO의 로컬 모델 경로 (이식성)
        # optimization
        "batch_size": batch_size,
        "num_workers": int(get_value(cfg, "num_workers", 8)),
        "num_gpus": num_gpus,
        "lr": lr_base * num_gpus,
        "steps": int(get_value(cfg, "steps", 100000)),
        "log_freq": int(get_value(cfg, "log_freq", 200)),
        "save_freq": int(get_value(cfg, "save_freq", 5000)),
        # wandb
        "wandb_enable": as_bool(get_value(cfg, "wandb_enable", True)),
        "wandb_project": str(get_value(cfg, "wandb_project", "VLA_stage1")),
    }

    part = ",".join(as_list(get_value(cfg, "train_partition", ["big"]))) or "big"
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
