#!/usr/bin/env python3
"""Resolve paths and settings for the foveated GT-skill image preview."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "src"))

from train_skills_config import as_list, get_value, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent / "foveated_skill_preview_config.yaml"


def _at(config: dict, section: str, key: str, default=None):
    value = config.get(section, {}) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{section} must be a YAML mapping.")
    return value.get(key, default)


def _subsection(config: dict, section: str, subsection: str) -> dict:
    parent = config.get(section, {}) or {}
    if not isinstance(parent, dict):
        raise ValueError(f"{section} must be a YAML mapping.")
    value = parent.get(subsection, {}) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{section}.{subsection} must be a YAML mapping.")
    return value


def _range(
    value: object,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
    integer: bool = False,
) -> list[int] | list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} must be a two-value [min, max] range.")
    cast = int if integer else float
    result = [cast(item) for item in value]
    if result[0] > result[1]:
        raise ValueError(f"{field} minimum cannot exceed its maximum.")
    if minimum is not None and result[0] < minimum:
        raise ValueError(f"{field} values must be >= {minimum}.")
    if maximum is not None and result[1] > maximum:
        raise ValueError(f"{field} values must be <= {maximum}.")
    return result


def _resolve(project_root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _safe_name(value: object) -> str:
    name = str(value or "").strip()
    if not name or re.fullmatch(r"[A-Za-z0-9._-]+", name) is None:
        raise ValueError(
            "output_name must contain only letters, digits, '.', '_', and '-'."
        )
    return name


def _int_list_or_all(value: object, *, field: str) -> str:
    if isinstance(value, str) and value.strip().lower() == "all":
        return "all"
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be 'all' or a non-empty integer list.")
    result = [int(item) for item in value]
    if any(item < 0 for item in result) or len(result) != len(set(result)):
        raise ValueError(f"{field} must contain unique non-negative integers.")
    return json.dumps(result, separators=(",", ":"))


def _optional_int_list(value: object, *, field: str) -> str:
    if value in (None, ""):
        value = []
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an integer list.")
    result = [int(item) for item in value]
    if any(item < 0 for item in result) or len(result) != len(set(result)):
        raise ValueError(f"{field} must contain unique non-negative integers.")
    return json.dumps(result, separators=(",", ":"))


def build_settings(config: dict) -> dict:
    project_root = _resolve(Path.cwd(), get_value(config, "project_root"))
    dataset_root = _resolve(project_root, get_value(config, "dataset_root", "dataset"))

    source_dataset = str(get_value(config, "source_dataset", "")).strip()
    run_name = str(get_value(config, "skillvla_run", "")).strip()
    target_task = str(get_value(config, "target_task", "")).strip()
    if not source_dataset or not run_name or not target_task:
        raise ValueError("source_dataset, skillvla_run, and target_task are required.")
    if Path(source_dataset).name != source_dataset or Path(run_name).name != run_name:
        raise ValueError("source_dataset and skillvla_run must be bare folder names.")

    run_dir = dataset_root / "skillvla_dataset" / source_dataset / run_name
    skill_dataset_dir = run_dir / "skillvla"
    latents_path = run_dir / "skill_latents.npz"
    info_path = skill_dataset_dir / "meta" / "info.json"
    exact_path = dataset_root / "skillvla_dataset" / source_dataset / "eval_init_states.npz"
    original_dataset_dir = _resolve(
        project_root,
        get_value(
            config,
            "original_dataset_dir",
            f"libero_original_dataset/{target_task}",
        ),
    )
    required = {
        "SkillVLA dataset": skill_dataset_dir,
        "skill assignments": latents_path,
        "SkillVLA metadata": info_path,
        "episode-exact map": exact_path,
        "original LIBERO dataset": original_dataset_dir,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing foveated-preview input(s):\n  " + "\n  ".join(missing))

    task_ids = _int_list_or_all(get_value(config, "task_ids", "all"), field="task_ids")
    episode_ids = _optional_int_list(
        get_value(config, "episode_ids", []), field="episode_ids"
    )
    episodes_per_task = int(get_value(config, "episodes_per_task", 2))
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive.")
    episode_selection = str(get_value(config, "episode_selection", "first")).lower()
    if episode_selection not in {"first", "random"}:
        raise ValueError("episode_selection must be first|random.")

    camera = str(_at(config, "foveation", "camera", "agentview")).strip()
    mode = str(_at(config, "foveation", "mode", "partial_fov")).strip().lower()
    crop_size = int(_at(config, "foveation", "crop_size", 96))
    output_size = int(_at(config, "foveation", "output_size", 224))
    inner_box = _subsection(config, "foveation", "inner_box")
    inner_box_enabled = bool(inner_box.get("enabled", True))
    inner_box_mode = str(inner_box.get("mode", "box")).strip().lower()
    inner_box_size = int(inner_box.get("size", 32))
    inner_box_line_width = int(inner_box.get("line_width", 3))
    shape = str(_at(config, "foveation", "shape", "square")).strip().lower()
    sharp_size = int(_at(config, "foveation", "sharp_size", 96))
    feather = int(_at(config, "foveation", "feather", 20))
    blur_radius = float(_at(config, "foveation", "blur_radius", 18.0))
    if not camera:
        raise ValueError("foveation.camera must be non-empty.")
    if mode not in {"partial_fov", "crop"}:
        raise ValueError("foveation.mode must be partial_fov|crop.")
    if crop_size <= 0 or output_size <= 0:
        raise ValueError("foveation.crop_size and output_size must be positive.")
    if inner_box_size <= 0 or inner_box_line_width <= 0:
        raise ValueError("foveation.inner_box size and line_width must be positive.")
    if inner_box_mode not in {"box", "blur"}:
        raise ValueError("foveation.inner_box.mode must be box|blur.")
    if inner_box_size > crop_size:
        raise ValueError("foveation.inner_box.size cannot exceed foveation.crop_size.")
    if shape not in {"square", "circle"}:
        raise ValueError("foveation.shape must be square|circle.")
    if sharp_size <= 0 or feather < 0 or blur_radius <= 0:
        raise ValueError(
            "foveation requires sharp_size>0, feather>=0, and blur_radius>0."
        )

    color = _subsection(config, "randomization", "color")
    crop = _subsection(config, "randomization", "crop")
    blur = _subsection(config, "randomization", "blur")
    color_enabled = bool(color.get("enabled", True))
    crop_enabled = bool(crop.get("enabled", True))
    random_blur_enabled = bool(blur.get("enabled", True))
    color_brightness = _range(
        color.get("brightness", [0.8, 1.2]),
        field="randomization.color.brightness",
        minimum=0.0,
    )
    color_contrast = _range(
        color.get("contrast", [0.8, 1.2]),
        field="randomization.color.contrast",
        minimum=0.0,
    )
    color_saturation = _range(
        color.get("saturation", [0.8, 1.2]),
        field="randomization.color.saturation",
        minimum=0.0,
    )
    color_hue = _range(
        color.get("hue", [-0.05, 0.05]),
        field="randomization.color.hue",
        minimum=-0.5,
        maximum=0.5,
    )
    crop_offset = _range(
        crop.get("offset_px", [-24, 24]),
        field="randomization.crop.offset_px",
        integer=True,
    )
    inner_box_offset = _range(
        crop.get("inner_box_offset_px", [-4, 4]),
        field="randomization.crop.inner_box_offset_px",
        integer=True,
    )
    random_blur_radius = _range(
        blur.get("blur_radius", [0.0, 4.0]),
        field="randomization.blur.blur_radius",
        minimum=0.0,
    )

    output_name = _safe_name(get_value(config, "output_name", "foveated_top_preview"))
    for suffix in ("_crop", "_partial_fov"):
        if output_name.endswith(suffix):
            output_name = output_name[: -len(suffix)]
            break
    output_name = f"{output_name}_{mode}"
    output_dir = _HERE.parent / "outputs" / "foveated_skill_preview" / output_name
    partitions = [str(value).strip() for value in as_list(get_value(config, "train_partition", "debug")) if str(value).strip()]
    excludes = [str(value).strip() for value in as_list(get_value(config, "train_exclude_nodes", [])) if str(value).strip()]

    return {
        "project_root": str(project_root),
        "lerobot_root": str(project_root / "lerobot"),
        "skill_dataset_dir": str(skill_dataset_dir),
        "skill_latents_path": str(latents_path),
        "eval_init_states_path": str(exact_path),
        "original_dataset_dir": str(original_dataset_dir),
        "target_task": target_task,
        "task_ids": task_ids,
        "episode_ids": episode_ids,
        "episodes_per_task": episodes_per_task,
        "episode_selection": episode_selection,
        "eval_seed": int(get_value(config, "seed", 42)),
        "foveation_camera": camera,
        "foveation_mode": mode,
        "foveation_crop_size": crop_size,
        "foveation_output_size": output_size,
        "foveation_inner_box_enabled": int(inner_box_enabled),
        "foveation_inner_box_mode": inner_box_mode,
        "foveation_inner_box_size": inner_box_size,
        "foveation_inner_box_line_width": inner_box_line_width,
        "foveation_shape": shape,
        "foveation_sharp_size": sharp_size,
        "foveation_feather": feather,
        "foveation_blur_radius": blur_radius,
        "random_color_enabled": int(color_enabled),
        "random_color_brightness": json.dumps(color_brightness, separators=(",", ":")),
        "random_color_contrast": json.dumps(color_contrast, separators=(",", ":")),
        "random_color_saturation": json.dumps(color_saturation, separators=(",", ":")),
        "random_color_hue": json.dumps(color_hue, separators=(",", ":")),
        "random_crop_enabled": int(crop_enabled),
        "random_crop_offset_px": json.dumps(crop_offset, separators=(",", ":")),
        "random_crop_inner_box_offset_px": json.dumps(
            inner_box_offset, separators=(",", ":")
        ),
        "random_blur_enabled": int(random_blur_enabled),
        "random_blur_radius": json.dumps(random_blur_radius, separators=(",", ":")),
        "preview_output_dir": str(output_dir),
        "preview_partition": ",".join(partitions) or "debug",
        "preview_qos": str(get_value(config, "train_qos", "base_qos")),
        "preview_nodelist": str(get_value(config, "train_nodelist", "")),
        "preview_exclude_nodes": ",".join(excludes),
        "preview_gres": str(_at(config, "slurm", "gres", "gpu:1")),
        "preview_cpus": int(_at(config, "slurm", "cpus", 4)),
        "preview_memory": str(_at(config, "slurm", "memory", "16G")),
        "preview_time": str(_at(config, "slurm", "time", "02:00:00")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    settings = build_settings(load_config(args.config))
    if args.shell:
        print_shell(settings)
    else:
        for key, value in settings.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
