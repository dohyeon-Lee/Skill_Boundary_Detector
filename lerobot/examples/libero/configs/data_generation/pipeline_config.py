#!/usr/bin/env python3
"""SkillVLA 파이프라인 공유 설정.

설정 방법:
  1. pipeline_config.yaml 의 값을 직접 수정한다  ← 기본 방법
  2. 환경변수로 override 한다 (일반 값만; homedir/projdir는 yaml 우선)

이 파일이 관리하는 것: 어떤 artifact를 쓸지 (경로 파생)
이 파일이 관리하지 않는 것: DP/FSQ 학습 하이퍼파라미터
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

_YAML_PATH = Path(__file__).parent / "pipeline_config.yaml"


def _load_yaml() -> dict:
    if not _YAML_PATH.exists():
        return {}
    with open(_YAML_PATH) as f:
        text = f.read()
    if _YAML_AVAILABLE:
        return yaml.safe_load(text) or {}

    # Minimal flat-YAML fallback for bootstrap interpreters without PyYAML.
    # This config intentionally uses only top-level scalar/list values.
    out: dict = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if value == "":
            out[key] = ""
            continue
        low = value.lower()
        if low in {"true", "false"}:
            out[key] = low == "true"
            continue
        try:
            out[key] = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            out[key] = value.strip("\"'")
    return out


def _get(yaml_cfg: dict, key: str, default) -> str:
    """환경변수 > yaml > default 순으로 값을 가져온다."""
    env_key = key.upper()
    if env_key in os.environ:
        return os.environ[env_key]
    val = yaml_cfg.get(key)
    if val is not None:
        return str(val)
    return str(default)


def _get_yaml_first(yaml_cfg: dict, key: str, default) -> str:
    """yaml > 환경변수 > default.

    Use this for root paths so inherited Slurm environment variables do not
    accidentally point jobs at another checkout.
    """
    val = yaml_cfg.get(key)
    if val is not None:
        return str(val)
    env_key = key.upper()
    if env_key in os.environ:
        return os.environ[env_key]
    return str(default)


def _get_int(yaml_cfg: dict, key: str, default: int) -> int:
    return int(_get(yaml_cfg, key, default))


def _get_bool(yaml_cfg: dict, key: str, default: bool) -> bool:
    env_key = key.upper()
    if env_key in os.environ:
        raw = os.environ[env_key]
        return raw.lower() in {"1", "true", "yes", "y", "on"}
    val = yaml_cfg.get(key)
    if val is not None:
        if isinstance(val, bool):
            return val
        return str(val).lower() in {"1", "true", "yes", "y", "on"}
    return default


def _get_levels(yaml_cfg: dict, default: tuple[int, ...]) -> tuple[int, ...]:
    """FSQ levels: 환경변수 > yaml > default."""
    if "FSQ_LEVELS" in os.environ:
        raw = os.environ["FSQ_LEVELS"]
    elif "fsq_levels" in yaml_cfg:
        val = yaml_cfg["fsq_levels"]
        if isinstance(val, list):
            return tuple(int(v) for v in val)
        raw = str(val)
    else:
        return default
    cleaned = raw.replace("[", " ").replace("]", " ").replace(",", " ")
    values = tuple(int(x) for x in cleaned.split())
    return values if values else default


def _visual_backbone(backbone: str) -> dict[str, str | int]:
    if backbone == "dinov2_small":
        return {
            "image_model_repo": "facebook/dinov2-small",
            "image_model_dir": "dinov2-small",
            "image_feature_tag": "dinov2_small",
            "image_exp_tag": "dino",
            "n_patch_raw": 256,
        }
    if backbone in {"dinov3_vits16", "dinov3_vits14"}:
        return {
            "image_model_repo": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "image_model_dir": "dinov3-vits16",
            "image_feature_tag": "dinov3_vits16",
            "image_exp_tag": "dinov3_vits16",
            "n_patch_raw": 196,
        }
    if backbone == "dinov3_convnext_small":
        return {
            "image_model_repo": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
            "image_model_dir": "dinov3-convnext-small",
            "image_feature_tag": "dinov3_convnext_small",
            "image_exp_tag": "dinov3_convnext_small",
            "n_patch_raw": 196,
        }
    raise ValueError(
        f"Unknown visual_backbone={backbone!r}. "
        "Use dinov2_small, dinov3_vits16, or dinov3_convnext_small."
    )


@dataclass(frozen=True)
class PipelineConfig:
    homedir: str
    projdir: str
    datadir: str
    data: str
    target_task: str
    data_dir: str
    raw_dataset_dir: str

    visual_backbone: str
    image_key: str
    image_model_repo: str
    image_model_dir: str
    image_model_path: str
    image_feature_tag: str
    image_exp_tag: str
    patch_grid: int
    image_size: int
    n_patch_raw: int

    frame_dino_dir: str
    dino_tokens_path: str
    dino_cache_path: str

    skillset: str
    skillset_dir: str
    fsq_precompute_dir: str
    sam2_checkpoint: str
    sam2_masks_dir: str
    sam2_flags_path: str

    fsq_levels: tuple[int, ...]
    fsq_levels_str: str
    fsq_levels_arg: str
    fsq_dim: int
    fsq_num_embeddings: int
    eef_mode: str
    zero_start_eef: bool
    action_mode: str
    decoder_image_mode: str
    fsq_epoch: str
    fsq_model_data: str
    fsq_model_skillset: str
    fsq_train_run: str
    fsq_source_ckpt: str
    fsq_run: str
    skillvla_root: str
    fsq_package_dir: str
    fsq_ckpt: str
    skill_latents_path: str

    skillvla_dataset: str
    skillvla_dataset_dir: str
    max_order: int
    max_length: int
    inference_skill_max_order: int
    inference_skill_max_length: int
    skill_decoder_state_indices: str
    skill_decoder_prior_noise_ratio: float

    dp_checkpoint: str
    dp_policy: str
    dp_policy_path: str
    dino_feature_dir: str
    dp_n_obs_steps: int
    dp_n_action_steps: int
    dp_horizon: int
    dp_batch_size: int
    dp_num_workers: int
    dp_save_freq: int
    dp_dino_feature_dim: int
    dp_dino_visual_feature_dim: int
    dp_dino_transformer_n_layers: int
    dp_dino_transformer_n_heads: int

    block_lang: bool
    detach_action_prefix_grad: bool


def load_config(data: str | None = None) -> PipelineConfig:
    y = _load_yaml()

    homedir = _get_yaml_first(y, "homedir", "/data2/dohyeon")
    projdir = _get_yaml_first(y, "projdir", "/SBD")
    datadir = _get(y, "datadir", "libero_dataset")
    data_name = data or _get(y, "data", "libero_90")
    if data_name.endswith("_skillvla"):
        data_name = data_name.removesuffix("_skillvla")

    visual_backbone = _get(y, "visual_backbone", "dinov3_vits16")
    visual = _visual_backbone(visual_backbone)
    image_feature_tag = str(visual["image_feature_tag"])
    image_exp_tag = str(visual["image_exp_tag"])
    patch_grid = _get_int(y, "patch_grid", 8)
    image_size = _get_int(y, "image_size", 224)

    root = f"{homedir}{projdir}"
    data_dir = f"{root}/{datadir}/{data_name}_data"
    raw_dataset_dir = f"{root}/{datadir}/{data_name}"
    fsq_precompute_dir = f"{data_dir}/{data_name}_for_FSQ"
    skillvla_root = f"{data_dir}/{data_name}_for_skillVLA"

    levels = _get_levels(y, (5, 5, 5))
    fsq_tag = "fsq" + "".join(str(v) for v in levels)
    fsq_num_embeddings = math.prod(levels)

    eef_mode = _get(y, "eef_mode", "abs")
    zero_start_eef = eef_mode == "zero"
    action_mode = _get(y, "action_mode", "single_step")
    decoder_image_mode = _get(y, "decoder_image_mode", "dino_flags")
    fsq_epoch = _get(y, "fsq_epoch", "0500")

    fsq_model_data = _get(y, "fsq_model_data", "libero_90")
    fsq_model_skillset = f"{fsq_model_data}_skillset"
    fsq_train_run = _get(
        y, "fsq_train_run",
        f"{fsq_model_data}_{fsq_tag}_{image_exp_tag}_{eef_mode}_{action_mode}_{decoder_image_mode}",
    )
    fsq_run = _get(y, "fsq_run", f"{fsq_train_run}_FSQ_epoch{fsq_epoch}")
    fsq_package_dir = f"{skillvla_root}/{fsq_run}"

    skillset = _get(y, "skillset", f"{data_name}_skillset")
    dp_checkpoint = _get(y, "dp_checkpoint", "080000")
    dp_n_obs_steps = _get_int(y, "dp_n_obs_steps", 10)
    dp_n_action_steps = _get_int(y, "dp_n_action_steps", 16)
    dp_horizon = _get_int(y, "dp_horizon", 16)
    dp_policy = _get(
        y, "dp_policy",
        f"dp_{data_name}_dino_{image_feature_tag}_pg{patch_grid}_obs{dp_n_obs_steps}_horizon{dp_horizon}",
    )

    sam2_ckpt_default = f"{root}/models/sam2/sam2.1_hiera_large.pt"
    sam2_checkpoint_val = _get(y, "sam2_checkpoint", sam2_ckpt_default)
    if not sam2_checkpoint_val:
        sam2_checkpoint_val = sam2_ckpt_default

    return PipelineConfig(
        homedir=homedir,
        projdir=projdir,
        datadir=datadir,
        data=data_name,
        target_task=_get(y, "target_task", data_name),
        data_dir=data_dir,
        raw_dataset_dir=raw_dataset_dir,
        visual_backbone=visual_backbone,
        image_key=_get(y, "image_key", "observation.images.image"),
        image_model_repo=str(visual["image_model_repo"]),
        image_model_dir=str(visual["image_model_dir"]),
        image_model_path=f"{root}/models/{visual['image_model_dir']}",
        image_feature_tag=image_feature_tag,
        image_exp_tag=image_exp_tag,
        patch_grid=patch_grid,
        image_size=image_size,
        n_patch_raw=int(visual["n_patch_raw"]),
        frame_dino_dir=f"{data_dir}/{data_name}_DINO/{image_feature_tag}_pg{patch_grid}",
        dino_tokens_path=f"{fsq_precompute_dir}/{image_feature_tag}_tokens.npz",
        dino_cache_path=f"{fsq_precompute_dir}/{image_feature_tag}_tokens.features.npy",
        skillset=skillset,
        skillset_dir=f"{fsq_precompute_dir}/{skillset}",
        fsq_precompute_dir=fsq_precompute_dir,
        sam2_checkpoint=sam2_checkpoint_val,
        sam2_masks_dir=f"{fsq_precompute_dir}/sam2_masks",
        sam2_flags_path=f"{fsq_precompute_dir}/patch_flags.npz",
        fsq_levels=levels,
        fsq_levels_str=" ".join(str(v) for v in levels),
        fsq_levels_arg="[" + ",".join(str(v) for v in levels) + "]",
        fsq_dim=len(levels),
        fsq_num_embeddings=fsq_num_embeddings,
        eef_mode=eef_mode,
        zero_start_eef=zero_start_eef,
        action_mode=action_mode,
        decoder_image_mode=decoder_image_mode,
        fsq_epoch=fsq_epoch,
        fsq_model_data=fsq_model_data,
        fsq_model_skillset=fsq_model_skillset,
        fsq_train_run=fsq_train_run,
        fsq_source_ckpt=_get(
            y, "fsq_source_ckpt",
            f"{root}/outputs/{fsq_model_skillset}_FSQ/{fsq_train_run}/FSQ_epoch{fsq_epoch}.pt",
        ),
        fsq_run=fsq_run,
        skillvla_root=skillvla_root,
        fsq_package_dir=fsq_package_dir,
        fsq_ckpt=f"{fsq_package_dir}/FSQ.pt",
        skill_latents_path=f"{fsq_package_dir}/skill_latents.npz",
        skillvla_dataset=f"{data_name}_skillvla",
        skillvla_dataset_dir=f"{skillvla_root}/{data_name}_skillvla",
        max_order=_get_int(y, "max_order", 0),
        max_length=_get_int(y, "max_length", 200),
        inference_skill_max_order=_get_int(y, "inference_skill_max_order", 20),
        inference_skill_max_length=_get_int(y, "inference_skill_max_length", 200),
        skill_decoder_state_indices=_get(y, "skill_decoder_state_indices", "[0,1,2,3,4,5,6]"),
        skill_decoder_prior_noise_ratio=float(_get(y, "skill_decoder_prior_noise_ratio", 0.0)),
        dp_checkpoint=dp_checkpoint,
        dp_policy=dp_policy,
        dp_policy_path=f"{root}/outputs/{dp_policy}/checkpoints/{dp_checkpoint}/pretrained_model",
        dino_feature_dir=f"{data_dir}/{data_name}_DINO/{image_feature_tag}_pg{patch_grid}",
        dp_n_obs_steps=dp_n_obs_steps,
        dp_n_action_steps=dp_n_action_steps,
        dp_horizon=dp_horizon,
        dp_batch_size=_get_int(y, "dp_batch_size", 64),
        dp_num_workers=_get_int(y, "dp_num_workers", 4),
        dp_save_freq=_get_int(y, "dp_save_freq", 5000),
        dp_dino_feature_dim=_get_int(y, "dp_dino_feature_dim", 384),
        dp_dino_visual_feature_dim=_get_int(y, "dp_dino_visual_feature_dim", 256),
        dp_dino_transformer_n_layers=_get_int(y, "dp_dino_transformer_n_layers", 1),
        dp_dino_transformer_n_heads=_get_int(y, "dp_dino_transformer_n_heads", 4),
        block_lang=_get_bool(y, "block_lang", True),
        detach_action_prefix_grad=_get_bool(y, "detach_action_prefix_grad", False),
    )


def _shell_name(name: str) -> str:
    return name.upper()


def print_shell(cfg: PipelineConfig, prefix: str = "") -> None:
    for key, value in asdict(cfg).items():
        if key == "fsq_levels":
            continue
        if isinstance(value, bool):
            value = "true" if value else "false"
        print(f"export {prefix}{_shell_name(key)}={shlex.quote(str(value))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shell", action="store_true", help="Print bash export statements.")
    parser.add_argument("--data", default=None, help="Override data for this invocation.")
    parser.add_argument("--prefix", default="", help="Optional prefix for --shell variable names.")
    args = parser.parse_args()
    cfg = load_config(args.data)
    if args.shell:
        print_shell(cfg, prefix=args.prefix)
        return
    for key, value in asdict(cfg).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
