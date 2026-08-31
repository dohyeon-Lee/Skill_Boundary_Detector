#!/usr/bin/env python3
"""Config resolver for CALVIN raw-dataset download utilities."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "calvin_dataset_config.yaml"
REQUIRED_VARIANT_KEYS = {"archive", "extracted_dir", "url", "sha256"}


def _find_global(start: Path) -> Path | None:
    for directory in [start.resolve(), *start.resolve().parents]:
        candidate = directory / "global_config.yaml"
        if candidate.is_file():
            return candidate
    return None


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        local = yaml.safe_load(stream) or {}
    global_path = _find_global(config_path.parent)
    if global_path is None or global_path == config_path:
        return local
    with global_path.open(encoding="utf-8") as stream:
        global_config = yaml.safe_load(stream) or {}
    return {**global_config, **local}


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).replace(",", " ").split() if part.strip()]


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected a boolean, got {value!r}")


def project_root(config: dict[str, Any]) -> Path:
    value = str(config.get("project_root", "")).strip()
    if not value:
        raise ValueError("project_root is missing; set it in configs/global_config.yaml")
    return Path(value).expanduser().resolve()


def _project_path(config: dict[str, Any], value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (project_root(config) / path).resolve()


def calvin_root(config: dict[str, Any]) -> Path:
    local = str(config.get("calvin_dataset_root", "") or "").strip()
    root = local or str(config.get("dataset_root", "dataset")).strip()
    return _project_path(config, root)


def global_dataset_root(config: dict[str, Any]) -> Path:
    return _project_path(config, str(config.get("dataset_root", "dataset")).strip())


def calvin_raw_root(config: dict[str, Any]) -> Path:
    subdir = str(config.get("calvin_raw_subdir", "_calvin_raw")).strip()
    if not subdir:
        raise ValueError("calvin_raw_subdir must not be blank")
    path = Path(subdir).expanduser()
    return path.resolve() if path.is_absolute() else (calvin_root(config) / path).resolve()


def variants(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = config.get("calvin_variants") or {}
    if not isinstance(raw, dict) or not raw:
        raise ValueError("calvin_variants must contain at least one variant")
    result: dict[str, dict[str, Any]] = {}
    for name, value in raw.items():
        spec = dict(value or {})
        missing = sorted(REQUIRED_VARIANT_KEYS - set(spec))
        if missing:
            raise ValueError(f"calvin_variants[{name!r}] is missing keys: {missing}")
        sha256 = str(spec["sha256"]).strip().lower()
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise ValueError(f"calvin_variants[{name!r}].sha256 is not a SHA-256 digest")
        archive = Path(str(spec["archive"]))
        extracted_dir = Path(str(spec["extracted_dir"]))
        if archive.name != str(archive) or archive.name in {"", ".", ".."}:
            raise ValueError(f"calvin_variants[{name!r}].archive must be a filename")
        if extracted_dir.name != str(extracted_dir) or extracted_dir.name in {"", ".", ".."}:
            raise ValueError(f"calvin_variants[{name!r}].extracted_dir must be one directory name")
        parsed_url = urlparse(str(spec["url"]))
        if parsed_url.scheme not in {"http", "https", "file"}:
            raise ValueError(f"calvin_variants[{name!r}].url must use http, https, or file")
        spec["sha256"] = sha256
        result[str(name)] = spec
    return result


def selected_variants(config: dict[str, Any], only: str = "") -> list[str]:
    available = variants(config)
    selected = as_list(only) if str(only).strip() else as_list(config.get("calvin_download_variants"))
    if not selected:
        raise ValueError("calvin_download_variants must select at least one variant")
    unknown = [name for name in selected if name not in available]
    if unknown:
        raise ValueError(f"unknown CALVIN variant(s) {unknown}; available={list(available)}")
    if len(selected) != len(set(selected)):
        raise ValueError(f"duplicate CALVIN variants selected: {selected}")
    return selected


def optional_positive_int(value: Any, key: str) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    parsed = int(text)
    if parsed < 1:
        raise ValueError(f"{key} must be blank or a positive integer")
    return parsed


def conversion_settings(config: dict[str, Any]) -> dict[str, Any]:
    available = variants(config)
    variant = str(config.get("calvin_convert_variant", "debug")).strip()
    if variant not in available:
        raise ValueError(
            f"unknown calvin_convert_variant={variant!r}; available={list(available)}"
        )
    split = str(config.get("calvin_convert_split", "training")).strip().lower()
    if split not in {"training", "validation"}:
        raise ValueError("calvin_convert_split must be training or validation")

    mode = str(config.get("calvin_convert_mode", "annotated")).strip().lower()
    if mode not in {"annotated", "play"}:
        raise ValueError("calvin_convert_mode must be annotated or play")
    task_split = str(config.get("calvin_task_split", "all")).strip().lower()
    if task_split not in {"all", "pretrain", "heldout"}:
        raise ValueError("calvin_task_split must be all, pretrain, or heldout")
    heldout_tasks = as_list(config.get("calvin_heldout_tasks", []))
    if len(heldout_tasks) != len(set(heldout_tasks)):
        raise ValueError("calvin_heldout_tasks contains duplicate task IDs")
    if task_split != "all" and not heldout_tasks:
        raise ValueError(
            f"calvin_task_split={task_split} requires non-empty calvin_heldout_tasks"
        )
    if mode == "play" and task_split == "heldout":
        raise ValueError(
            "calvin_convert_mode=play does not support calvin_task_split=heldout; "
            "use annotated + heldout"
        )

    action = str(config.get("calvin_policy_action", "relative")).strip().lower()
    if action not in {"relative", "absolute"}:
        raise ValueError("calvin_policy_action must be relative or absolute")
    state = str(config.get("calvin_policy_state", "robot_obs")).strip().lower()
    if state not in {"robot_obs", "tcp_pose_gripper", "joint_gripper"}:
        raise ValueError(
            "calvin_policy_state must be robot_obs, tcp_pose_gripper, or joint_gripper"
        )
    preserve_mode = str(config.get("calvin_preserve_raw_mode", "hardlink")).strip().lower()
    if preserve_mode not in {"hardlink", "copy", "none"}:
        raise ValueError("calvin_preserve_raw_mode must be hardlink, copy, or none")

    raw_image_size = str(config.get("calvin_convert_image_size", 224)).strip().lower()
    if raw_image_size == "native":
        image_size: int | str = "native"
    else:
        image_size = int(raw_image_size)
        if image_size < 1:
            raise ValueError("calvin_convert_image_size must be native or a positive integer")

    output_root = global_dataset_root(config)
    name_parts = ["calvin", variant]
    if split != "training":
        name_parts.append(split)
    if mode == "play":
        name_parts.append("play")
    if task_split != "all":
        name_parts.append(task_split)
    output_name = "_".join(name_parts) + "_full_full"

    extracted_root = calvin_raw_root(config) / str(available[variant]["extracted_dir"])
    partitions = ",".join(as_list(config.get("train_partition", ["debug"]))) or "debug"
    exclude_nodes = ",".join(as_list(config.get("train_exclude_nodes", [])))
    annotation_folder = str(
        config.get("calvin_convert_annotation_folder", "lang_annotations")
    ).strip()
    if Path(annotation_folder).name != annotation_folder or annotation_folder in {"", ".", ".."}:
        raise ValueError("calvin_convert_annotation_folder must be one directory name")

    return {
        "calvin_convert_variant": variant,
        "calvin_convert_split": split,
        "calvin_convert_mode": mode,
        "calvin_task_split": task_split,
        "calvin_heldout_tasks": heldout_tasks,
        "calvin_convert_source_root": extracted_root,
        "calvin_convert_source_dir": extracted_root / split,
        "calvin_convert_output_root": output_root,
        "calvin_convert_output_name": output_name,
        "calvin_convert_output_dir": output_root / output_name,
        "calvin_convert_repo_id": f"dohyeon/{output_name}",
        "calvin_convert_overwrite": as_bool(config.get("calvin_convert_overwrite", False)),
        "calvin_policy_action": action,
        "calvin_policy_state": state,
        "calvin_convert_image_size": image_size,
        "calvin_convert_fps": int(config.get("calvin_convert_fps", 30)),
        "calvin_convert_annotation_folder": annotation_folder,
        "calvin_preserve_raw_mode": preserve_mode,
        "calvin_convert_vcodec": str(config.get("calvin_convert_vcodec", "libsvtav1")),
        "calvin_convert_streaming_encoding": as_bool(
            config.get("calvin_convert_streaming_encoding", False)
        ),
        "calvin_convert_encoder_queue_maxsize": int(
            config.get("calvin_convert_encoder_queue_maxsize", 30)
        ),
        "calvin_convert_encoder_threads": str(
            config.get("calvin_convert_encoder_threads", "") or ""
        ).strip(),
        "calvin_convert_batch_encoding_size": int(
            config.get("calvin_convert_batch_encoding_size", 1)
        ),
        "calvin_convert_image_writer_threads": int(
            config.get("calvin_convert_image_writer_threads", 10)
        ),
        "calvin_convert_image_writer_processes": int(
            config.get("calvin_convert_image_writer_processes", 5)
        ),
        "calvin_convert_max_episodes": optional_positive_int(
            config.get(
                "calvin_convert_max_episodes",
                config.get("calvin_convert_max_annotations", ""),
            ),
            "calvin_convert_max_episodes",
        ),
        "calvin_convert_max_frames_per_episode": optional_positive_int(
            config.get(
                "calvin_convert_max_frames_per_episode",
                config.get("calvin_convert_max_frames_per_annotation", ""),
            ),
            "calvin_convert_max_frames_per_episode",
        ),
        "calvin_convert_partition": partitions,
        "calvin_convert_qos": str(config.get("train_qos", "base_qos")),
        "calvin_convert_gres": str(config.get("calvin_convert_gres", "gpu:1")),
        "calvin_convert_cpus_per_task": int(
            config.get("calvin_convert_cpus_per_task", 16)
        ),
        "calvin_convert_mem": str(config.get("calvin_convert_mem", "64G")),
        "calvin_convert_time": str(config.get("calvin_convert_time", "24:00:00")),
        "calvin_convert_nodelist": str(config.get("train_nodelist", "") or ""),
        "calvin_convert_exclude_nodes": exclude_nodes,
    }


def build_settings(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "project_root": project_root(config),
        "calvin_root": calvin_root(config),
        "calvin_raw_root": calvin_raw_root(config),
        "calvin_download_variants": " ".join(selected_variants(config, os.environ.get("CALVIN_ONLY", ""))),
        **conversion_settings(config),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    try:
        settings = build_settings(load_config(args.config))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if args.shell:
        for key, value in settings.items():
            if isinstance(value, bool):
                shell_value = str(value).lower()
            elif value is None:
                shell_value = ""
            else:
                shell_value = str(value)
            print(f"export {key.upper()}={shlex.quote(shell_value)}")
    else:
        for key, value in settings.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
