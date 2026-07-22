#!/usr/bin/env python3
"""Resolve original-FSQ broadcast-scale closed-loop eval settings."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "src"))

from train_skills_config import as_bool, as_list, get_value, load_config, print_shell, resolve_path  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "fsq_eval_config.yaml"


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_") or "panel"


def _scales(value: object) -> tuple[float, ...]:
    raw = value
    if isinstance(raw, str):
        raw = [item.strip() for item in raw.split(",") if item.strip()]
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    scales = tuple(float(item) for item in raw)
    if not scales or any(not math.isfinite(item) or item < 0.0 for item in scales):
        raise ValueError(f"broadcast_scales must contain finite values >= 0, got {value!r}.")
    if len(set(scales)) != len(scales):
        raise ValueError(f"broadcast_scales must not contain duplicates, got {value!r}.")
    return scales


def _scale_title(scale: float) -> str:
    percent = scale * 100.0
    text = f"{percent:.4f}".rstrip("0").rstrip(".")
    return f"{text}%"


def _checkpoint(run_dir: Path, value: object) -> tuple[Path, str]:
    raw = str(value or "best").strip().lower()
    if raw in {"best", "fsq", "fsq.pt"}:
        return run_dir / "FSQ.pt", "best"
    if raw == "last":
        periodic = list(run_dir.glob("FSQ_epoch*.pt"))
        if not periodic:
            return run_dir / "FSQ.pt", "best"
        latest = max(
            periodic,
            key=lambda path: int(re.search(r"FSQ_epoch(\d+)", path.stem).group(1)),
        )
        return latest, latest.stem.removeprefix("FSQ_epoch")
    if raw.isdigit():
        tag = f"{int(raw):04d}"
        return run_dir / f"FSQ_epoch{tag}.pt", tag
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = run_dir / path
    return path, path.stem


def _skillset_dir(
    meta: dict,
    entry: dict,
    cfg: dict,
    fsq_dataset_root: Path,
    source: str,
    project_root: Path,
) -> Path:
    explicit = str(get_value(entry, "skills_dir", "") or "").strip()
    if explicit:
        return resolve_path(project_root, explicit)

    mode = str(meta["skillset_mode"])
    threshold = ""
    if str(meta.get("skillset_boundary_threshold_mode", "episode_mean")) == "global_mean":
        threshold = "_globalref" if str(meta.get("skillset_global_threshold_source", "")) else "_globalmean"
    min_skills = int(get_value(
        entry,
        "skillset_min_skills",
        meta.get("skillset_min_skills", get_value(cfg, "skillset_min_skills", 2)),
    ))
    minimum = "" if min_skills == 2 else f"_ms{min_skills}"
    custom = str(get_value(
        entry,
        "skillset_output_suffix",
        meta.get("skillset_output_suffix", get_value(cfg, "skillset_output_suffix", "")),
    ) or "").strip()
    if custom and not custom.startswith("_"):
        custom = "_" + custom
    segment = (
        f"seg_{meta['dp_run_name']}_ck{meta['dp_checkpoint']}_{mode}"
        f"{threshold}{minimum}{custom}"
    )
    return fsq_dataset_root / source / "FSQ_inputs" / segment / "skillset" / "skills"


def build_settings(cfg: dict) -> dict:
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    dataset_root = project_root / str(get_value(cfg, "dataset_root", "dataset_filtered"))
    outputs_root = project_root / str(get_value(cfg, "outputs_root", "outputs_filtered"))
    source = str(get_value(cfg, "source_dataset", "libero_90_full_full"))
    raw_dataset_dir = dataset_root / source
    fsq_dataset_root = dataset_root / str(get_value(cfg, "fsq_dataset_root", "FSQ_dataset"))
    fsq_outputs = outputs_root / "FSQ"
    dino_model_path = resolve_path(
        project_root, get_value(cfg, "dino_model_path", "models/dinov3-vitl16")
    )

    models_yaml = get_value(cfg, "models", None)
    if not isinstance(models_yaml, list) or not models_yaml:
        raise ValueError(
            "Set models to a non-empty list of {run, checkpoint, label} entries."
        )

    panels: list[dict] = []
    used_keys: set[str] = set()
    default_scales = get_value(cfg, "broadcast_scales", [1.0, 0.6, 0.3])
    for model_index, entry in enumerate(models_yaml):
        run = str(get_value(entry, "run", get_value(entry, "model_dir", "")) or "").strip()
        if not run:
            raise ValueError("Each models[] entry needs run: <FSQ output folder name>.")
        run_dir = Path(run).expanduser() if Path(run).expanduser().is_absolute() else fsq_outputs / run
        meta_path = run_dir / "fsq_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"FSQ metadata not found: {meta_path}")
        meta = json.loads(meta_path.read_text())
        model_path, checkpoint_tag = _checkpoint(
            run_dir, get_value(entry, "checkpoint", "best")
        )
        if not model_path.is_file():
            raise FileNotFoundError(f"FSQ checkpoint not found: {model_path}")
        skills_dir = _skillset_dir(
            meta, entry, cfg, fsq_dataset_root, source, project_root
        )
        if not skills_dir.is_dir():
            raise FileNotFoundError(
                f"FSQ skill NPZ directory not found: {skills_dir}. "
                "For historical runs missing skillset_min_skills metadata, set it per model."
            )
        latents_path = run_dir / f"skill_latents_eval_{checkpoint_tag}.npz"
        advance = str(
            get_value(entry, "advance_mode", get_value(cfg, "advance_mode", "terminator"))
        ).lower()
        if advance not in {"gt", "terminator"}:
            raise ValueError(f"advance_mode must be gt|terminator, got {advance!r}.")
        model_label = str(get_value(entry, "label", run)).strip() or run
        scales = _scales(get_value(entry, "broadcast_scales", default_scales))
        state_cond_mode = str(meta.get("state_cond_mode", ""))
        if state_cond_mode and state_cond_mode != "broadcast" and any(scale != 1.0 for scale in scales):
            raise ValueError(
                f"{run} uses state_cond_mode={state_cond_mode!r}; only scale 1.0 is valid."
            )
        for scale in scales:
            scale_title = _scale_title(scale)
            display = f"{model_label} [{scale_title}]"
            key = _safe(display)
            if key in used_keys:
                key = f"{key}-{model_index}"
            used_keys.add(key)
            panels.append({
                "key": key,
                "label": display,
                "run": run_dir.name,
                "checkpoint": checkpoint_tag,
                "broadcast_scale": scale,
                "advance_mode": advance,
                "fsq_path": str(model_path),
                "skills_dir": str(skills_dir),
                "latents_path": str(latents_path),
                "raw_dataset_dir": str(raw_dataset_dir),
                "dino_model_path": str(dino_model_path),
            })

    canonical_inits = fsq_dataset_root / source / "eval_init_states.npz"
    legacy_inits = dataset_root / "skillvla_dataset" / source / "eval_init_states.npz"
    explicit_inits = str(get_value(cfg, "eval_init_states_path", "") or "").strip()
    init_states_path = resolve_path(project_root, explicit_inits) if explicit_inits else canonical_inits
    for panel in panels:
        panel["init_states_path"] = str(init_states_path)

    output_name = str(get_value(cfg, "output_name", "") or "").strip()
    if not output_name:
        output_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = _HERE.parent.parent
    output_dir = eval_dir / "outputs" / _safe(output_name)
    labels = [panel["key"] for panel in panels]

    part = ",".join(as_list(get_value(cfg, "train_partition", ["debug"]))) or "debug"
    return {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        "fsq_eval_dir": eval_dir,
        "models_json": json.dumps(panels, separators=(",", ":")),
        "models_labels": json.dumps(labels, separators=(",", ":")),
        "models_per_row": int(get_value(cfg, "models_per_row", 3) or 0),
        "eval_out_dir": output_dir,
        "source_dataset": source,
        "raw_dataset_dir": raw_dataset_dir,
        "eval_init_states_path": init_states_path,
        "legacy_eval_init_states_path": legacy_inits,
        "original_dataset_dir": project_root / "libero_original_dataset" / str(
            get_value(cfg, "target_task", "libero_90")
        ),
        "target_task": str(get_value(cfg, "target_task", "libero_90")),
        "task_ids": str(get_value(cfg, "task_ids", "[0,1,2,3,4,5,6,7,8,9]")),
        "eval_num_gpus": max(1, int(get_value(cfg, "eval_num_gpus", 1))),
        "n_episodes": int(get_value(cfg, "n_episodes", 5)),
        "eval_batch_size": int(get_value(cfg, "eval_batch_size", 1)),
        "max_parallel_tasks": int(get_value(cfg, "max_parallel_tasks", 1)),
        "n_action_steps": int(get_value(cfg, "n_action_steps", 5)),
        "max_videos_per_task": int(get_value(cfg, "max_videos_per_task", 1)),
        "video_frame_stride": int(get_value(cfg, "video_frame_stride", 2)),
        "video_fps": int(get_value(cfg, "video_fps", 10)),
        "skill_end_mode": str(get_value(cfg, "skill_end_mode", "or")),
        "skill_end_threshold": float(get_value(cfg, "skill_end_threshold", 0.5)),
        "skill_end_progress_threshold": float(
            get_value(cfg, "skill_end_progress_threshold", 0.95)
        ),
        "inference_skill_max_length": int(
            get_value(cfg, "inference_skill_max_length", 150)
        ),
        "skill_html": as_bool(get_value(cfg, "skill_html", True)),
        "skill_html_train_samples": int(get_value(cfg, "skill_html_train_samples", 5)),
        "wandb_enable": as_bool(get_value(cfg, "wandb_enable", True)),
        "wandb_project": str(get_value(cfg, "wandb_project", "FSQ_eval")),
        "wandb_run_name": f"FSQEval_{_safe(output_name)}",
        "eval_partition": part,
        "eval_qos": str(get_value(cfg, "train_qos", "base_qos")),
        "eval_gres": str(get_value(cfg, "eval_gres", "gpu:1")),
        "eval_cpus_per_task": int(get_value(cfg, "eval_cpus_per_task", 8)),
        "eval_mem": str(get_value(cfg, "eval_mem", "64G")),
        "eval_time": str(get_value(cfg, "eval_time", "2:00:00")),
        "eval_nodelist": str(get_value(cfg, "train_nodelist", "")),
        "eval_exclude_nodes": ",".join(as_list(get_value(cfg, "train_exclude_nodes", []))),
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
