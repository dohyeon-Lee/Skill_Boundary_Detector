#!/usr/bin/env python3
"""Resolve wrist_uv_probe YAML into validated shell settings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parent
PROBE_DIR = SRC_DIR.parent
CONFIGS_ROOT = PROBE_DIR.parents[2]
sys.path.insert(0, str(CONFIGS_ROOT / "train_skills" / "src"))

from train_skills_config import as_list, get_value, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = PROBE_DIR / "wrist_uv_probe_config.yaml"
ORIENTATIONS = ("libero", "raw", "flip_y", "flip_xy")


def _path(project_root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else project_root / path


def _existing(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


def settings(config: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(str(config["project_root"])).expanduser()
    source = str(get_value(config, "source_dataset", "")).strip()
    skill_run = str(get_value(config, "skill_run", "")).strip()
    suite = str(get_value(config, "suite", "")).strip()
    if not source or not skill_run or not suite:
        raise ValueError("source_dataset, skill_run and suite are required.")

    dataset_root = _path(project_root, get_value(config, "dataset_root", "dataset"))
    run_dir = _existing(
        dataset_root / "skillvla_dataset" / source / skill_run, "SkillVLA run"
    )
    original_root = _path(
        project_root, get_value(config, "original_dataset_root", "libero_original_dataset")
    )
    counts = {
        name: int(get_value(config, name, default))
        for name, default in (("episodes", 2), ("skills_per_episode", 3), ("frames_per_skill", 3))
    }
    for name, value in counts.items():
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}.")

    selections = {
        name: [int(value) for value in as_list(get_value(config, name, []))]
        for name in ("task_ids", "episode_ids")
    }
    orientations = [str(value).strip() for value in as_list(get_value(config, "orientations", list(ORIENTATIONS)))]
    unknown = [value for value in orientations if value not in ORIENTATIONS]
    if not orientations or unknown:
        raise ValueError(f"orientations must be a non-empty subset of {list(ORIENTATIONS)}; got {orientations}.")

    output_name = str(get_value(config, "output_name", "") or "").strip() or f"{source}_{skill_run}"
    if "/" in output_name:
        raise ValueError(f"output_name must be a folder name, got {output_name!r}.")

    return {
        "probe_skill_dataset_dir": _existing(run_dir / "skillvla", "SkillVLA LeRobot dataset"),
        "probe_skill_latents_path": _existing(run_dir / "skill_latents.npz", "skill_latents.npz"),
        "probe_eval_init_states_path": _existing(
            dataset_root / "skillvla_dataset" / source / "eval_init_states.npz", "eval_init_states.npz"
        ),
        "probe_original_dataset_dir": _existing(original_root / suite, "original LIBERO demos"),
        "probe_suite": suite,
        "probe_camera": str(get_value(config, "camera", "robot0_eye_in_hand")).strip(),
        "probe_eef_site": str(get_value(config, "eef_site", "gripper0_grip_site")).strip(),
        "probe_rotation_frame": str(get_value(config, "rotation_frame", "robot0_right_hand")).strip(),
        "probe_video_key": str(get_value(config, "video_key", "observation.images.wrist_image")).strip(),
        "probe_task_ids": ",".join(str(value) for value in selections["task_ids"]),
        "probe_episode_ids": ",".join(str(value) for value in selections["episode_ids"]),
        "probe_episodes": counts["episodes"],
        "probe_skills_per_episode": counts["skills_per_episode"],
        "probe_frames_per_skill": counts["frames_per_skill"],
        "probe_orientations": ",".join(orientations),
        "probe_patch_grid": int(get_value(config, "patch_grid", 14)),
        "probe_soft_sigma": float(get_value(config, "soft_sigma", 0.7)),
        "probe_out_dir": PROBE_DIR / "outputs" / output_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    resolved = settings(load_config(args.config))
    if args.shell:
        print_shell(resolved)
    else:
        for key, value in resolved.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
