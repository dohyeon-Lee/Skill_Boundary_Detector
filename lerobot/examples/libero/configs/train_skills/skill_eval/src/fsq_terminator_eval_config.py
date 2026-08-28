#!/usr/bin/env python3
"""Resolve a GT-skill probe of FSQ co-trained terminators.

Unlike train_skillVLA/terminator_eval this never loads an action policy: the
skills come from the FSQ skillset and their GT end frames are already known, so
there is nothing for a policy to roll out and nothing for a MAIN terminator to
decide. Every listed model is scored the same way, against the same GT skills.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / "src"))

from eval_config import _resolve_fsq_artifact  # noqa: E402
from train_skills_config import (  # noqa: E402
    as_bool,
    as_list,
    get_value,
    load_config,
    print_shell,
)

DEFAULT_CONFIG_PATH = _HERE.parent / "fsq_terminator_eval_config.yaml"


def _at(config: dict, section: str, key: str, default=None):
    value = config.get(section, {}) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{section} must be a YAML mapping.")
    return value.get(key, default)


def _json_int_list(value: object, *, field: str, allow_empty: bool) -> list[int]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or (not allow_empty and not value):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise ValueError(f"{field} must be {qualifier}.")
    result = [int(item) for item in value]
    if any(item < 0 for item in result) or len(result) != len(set(result)):
        raise ValueError(f"{field} must contain unique non-negative integers: {result}.")
    return result


def _safe_relative_output(value: object, *, default: str) -> Path:
    text = str(value or "").strip() or default
    path = Path(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"output_name must be a safe relative path, got {text!r}.")
    if any(not re.fullmatch(r"[A-Za-z0-9._-]+", part) for part in path.parts):
        raise ValueError(
            "output_name components may contain only letters, digits, '.', '_', and '-': "
            f"{text!r}."
        )
    return path


def _safe_label(value: object, *, index: int) -> str:
    text = str(value or "").strip() or f"model{index}"
    if not re.fullmatch(r"[A-Za-z0-9._-]+", text):
        raise ValueError(
            "terminator_models[].label may contain only letters, digits, '.', '_', "
            f"and '-': {text!r}."
        )
    return text


def _model_entries(config: dict) -> list[dict]:
    """The (label, source, run, checkpoint) rows to probe, in listed order.

    Checkpoints are named explicitly rather than swept: each entry is a
    terminator the user picked, mirroring terminator_eval's terminator_models.
    """
    # Read the raw value: as_list() stringifies mappings, which would turn each
    # entry into "{'label': ...}" and lose every field.
    raw = get_value(config, "terminator_models", []) or []
    if not isinstance(raw, list):
        raise ValueError("terminator_models must be a list of mappings.")
    entries = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(
                "terminator_models entries must be mappings with run_name and checkpoint."
            )
        unknown = sorted(set(item) - {"label", "source", "run_name", "checkpoint"})
        if unknown:
            raise ValueError(
                f"terminator_models[{index}] has unsupported fields {unknown}."
            )
        source = str(item.get("source", "fsq")).strip().lower()
        source = {"aux": "auxiliary", "skill_aux": "auxiliary"}.get(source, source)
        if source not in {"fsq", "auxiliary"}:
            raise ValueError(
                f"terminator_models[{index}].source must be fsq|auxiliary, got {source!r}."
            )
        run_name = str(item.get("run_name", "")).strip()
        if not run_name:
            raise ValueError(f"terminator_models[{index}] is missing run_name.")
        checkpoint = str(item.get("checkpoint", "last")).strip().lower()
        if not checkpoint:
            raise ValueError(f"terminator_models[{index}] has an empty checkpoint.")
        entries.append(
            {
                "label": _safe_label(item.get("label"), index=index),
                "source": source,
                "run_name": run_name,
                "checkpoint": checkpoint,
            }
        )
    if not entries:
        raise ValueError("terminator_models must list at least one model.")
    labels = [entry["label"] for entry in entries]
    if len(labels) != len(set(labels)):
        raise ValueError(f"terminator_models labels must be unique: {labels}.")
    return entries


def _resolve_auxiliary_checkpoint(run_dir: Path, checkpoint: str) -> tuple[Path, str]:
    """Resolve one LeRobot ``pretrained_model`` directory by training step."""
    candidates: list[tuple[int, Path]] = []
    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.is_dir():
        for path in checkpoints_dir.iterdir():
            if path.is_dir() and path.name.isdigit():
                model_dir = path / "pretrained_model"
                if (model_dir / "config.json").is_file() and (
                    model_dir / "model.safetensors"
                ).is_file():
                    candidates.append((int(path.name), model_dir))
    if not candidates:
        raise FileNotFoundError(
            f"No complete auxiliary checkpoints found under {checkpoints_dir}."
        )
    if checkpoint == "last":
        step, model_dir = max(candidates, key=lambda item: item[0])
        return model_dir, str(step)
    if not checkpoint.isdigit() or int(checkpoint) <= 0:
        raise ValueError(
            "Auxiliary checkpoint must be 'last' or a positive training step, "
            f"got {checkpoint!r}."
        )
    wanted = int(checkpoint)
    matches = [item for item in candidates if item[0] == wanted]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Auxiliary checkpoint step {wanted} not found under {checkpoints_dir}."
        )
    step, model_dir = matches[0]
    return model_dir, str(step)


def _resolve_auxiliary_artifact(
    entry: dict,
    *,
    project_root: Path,
    dataset_root: Path,
    outputs_root: Path,
) -> dict:
    """Resolve a standalone ``skill_aux`` terminator plus its immutable FSQ space."""
    run_dir = outputs_root / "skillVLA_terminator" / entry["run_name"]
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Auxiliary terminator run folder not found: {run_dir}")
    checkpoint_dir, step = _resolve_auxiliary_checkpoint(run_dir, entry["checkpoint"])
    source = json.loads((checkpoint_dir / "config.json").read_text())
    if source.get("type") != "skill_aux" or not as_bool(
        source.get("train_terminator", False)
    ):
        raise ValueError(
            "Auxiliary terminator eval requires policy.type=skill_aux with "
            f"train_terminator=true: {checkpoint_dir}."
        )
    fsq_path = Path(str(source.get("fsq_path", "") or "")).expanduser()
    if not fsq_path.is_absolute():
        fsq_path = project_root / fsq_path
    if not fsq_path.is_file():
        raise FileNotFoundError(f"Auxiliary checkpoint FSQ.pt not found: {fsq_path}")
    code_space = str(source.get("skill_code_space_id", "") or "").strip()
    if not code_space:
        code_space = fsq_path.parent.name
    levels = source.get("skill_fsq_levels")
    if not isinstance(levels, list) or not levels:
        raise ValueError(
            f"Auxiliary checkpoint is missing skill_fsq_levels: {checkpoint_dir}."
        )
    # SkillVLA data identities wrap the original FSQ run as
    # FSQ<levels>_<fsq-run>_<epoch>_<segmentation...>. Recover that immutable
    # source from existing FSQ run folders instead of guessing at underscores
    # inside the run name.
    fsq_prefix = "FSQ" + "".join(str(int(level)) for level in levels) + "_"
    source_candidates: list[tuple[int, str, str]] = []
    fsq_outputs = outputs_root / "FSQ"
    if fsq_outputs.is_dir():
        for candidate in fsq_outputs.iterdir():
            if not candidate.is_dir() or not (candidate / "fsq_meta.json").is_file():
                continue
            prefix = fsq_prefix + candidate.name + "_"
            if not code_space.startswith(prefix):
                continue
            checkpoint_tag = code_space[len(prefix) :].split("_", 1)[0]
            if checkpoint_tag == "best" or (
                checkpoint_tag.isdigit() and int(checkpoint_tag) > 0
            ):
                source_candidates.append(
                    (len(candidate.name), candidate.name, checkpoint_tag)
                )
    if not source_candidates:
        raise FileNotFoundError(
            "Could not map auxiliary skill_code_space_id back to an FSQ run: "
            f"{code_space!r} under {fsq_outputs}."
        )
    _, fsq_run_name, fsq_checkpoint = max(source_candidates)
    base = _resolve_fsq_artifact(
        {"fsq_eval_run_name": fsq_run_name},
        dataset_root=dataset_root,
        outputs_root=outputs_root,
        checkpoint=fsq_checkpoint,
    )
    return {
        **base,
        "fsq_eval_run_name": entry["run_name"],
        "fsq_eval_selected_checkpoint": entry["checkpoint"],
        "fsq_eval_resolved_checkpoint": step,
        "fsq_eval_model_dir": str(run_dir),
        # The original FSQ model still owns the encoder/code assignment.
        "fsq_eval_model_path": str(fsq_path),
        "fsq_eval_epoch_tag": f"step{int(step):06d}",
        "fsq_eval_terminator_overlay_path": str(checkpoint_dir),
        "fsq_eval_model_source": "auxiliary",
        "fsq_eval_code_space_id": code_space,
    }


def build_settings(config: dict, *, model_override: str | None = None) -> dict:
    entries = _model_entries(config)
    project_root = Path(str(get_value(config, "project_root"))).expanduser().resolve()
    dataset_root = Path(str(get_value(config, "dataset_root", "dataset"))).expanduser()
    outputs_root = Path(str(get_value(config, "outputs_root", "outputs"))).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    if not outputs_root.is_absolute():
        outputs_root = project_root / outputs_root

    resolved = []
    for entry in entries:
        if entry["source"] == "auxiliary":
            artifact = _resolve_auxiliary_artifact(
                entry,
                project_root=project_root,
                dataset_root=dataset_root,
                outputs_root=outputs_root,
            )
        else:
            artifact = _resolve_fsq_artifact(
                {**config, "fsq_eval_run_name": entry["run_name"]},
                dataset_root=dataset_root,
                outputs_root=outputs_root,
                checkpoint=entry["checkpoint"],
            )
            artifact["fsq_eval_terminator_overlay_path"] = ""
            artifact["fsq_eval_model_source"] = "fsq"
            artifact["fsq_eval_code_space_id"] = entry["run_name"]
        resolved.append({**entry, **artifact})

    # FSQ-original checkpoints carry no terminator module at all: their only
    # termination signal lives inside the rnn decoder's open-loop unroll, which
    # cannot be queried against real GT frames. Catch them here rather than
    # letting the v3 loader fail mid-job with a "legacy format_version" message.
    original = [
        item["label"]
        for item in resolved
        if Path(item["fsq_eval_meta_path"]).name == "fsq_original_meta.json"
    ]
    if original:
        raise ValueError(
            f"terminator_models {original} are FSQ-original runs, which have no "
            "co-trained terminator module to probe (FSQ_original.py defines none; "
            "its rnn decoder emits termination inline and oneshot emits none). "
            "List FSQ v3 runs here, or score those runs with fsq_original_eval.py."
        )

    # Per-skill comparison only means anything when every model segmented the
    # dataset the same way: same skillset -> same GT skills and same GT end
    # frames. Different skillsets would silently compare different questions.
    skillsets = {item["fsq_eval_skillset_dir"] for item in resolved}
    if len(skillsets) != 1:
        raise ValueError(
            "Every terminator_models entry must share one FSQ skillset so the GT "
            f"skills and their end frames match; got {sorted(skillsets)}."
        )
    datasets = {item["fsq_eval_dataset_dir"] for item in resolved}
    if len(datasets) != 1:
        raise ValueError(f"Every entry must share one source dataset; got {sorted(datasets)}.")

    selected = resolved[0]
    if model_override is not None:
        matches = [item for item in resolved if item["label"] == model_override]
        if len(matches) != 1:
            raise ValueError(
                f"Model override {model_override!r} is not one of "
                f"{[item['label'] for item in resolved]}."
            )
        selected = matches[0]

    target_task = str(get_value(config, "target_task", "libero_90")).strip()
    if not target_task:
        raise ValueError("target_task must be non-empty.")
    raw_task_ids = get_value(config, "task_ids", [0])
    all_tasks = isinstance(raw_task_ids, str) and raw_task_ids.strip().lower() == "all"
    task_ids = (
        None
        if all_tasks
        else _json_int_list(raw_task_ids, field="task_ids", allow_empty=False)
    )
    episode_ids = _json_int_list(
        get_value(config, "episode_ids", []), field="episode_ids", allow_empty=True
    )
    episodes_per_task = int(get_value(config, "episodes_per_task", 2))
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive.")
    # The skillset numbers tasks by the dataset's own task_index; the
    # episode-exact tools use LIBERO suite ids. Same numbers, different tasks.
    task_id_space = str(get_value(config, "task_id_space", "dataset")).strip().lower()
    if task_id_space not in {"dataset", "suite"}:
        raise ValueError("task_id_space must be dataset|suite.")
    episode_selection = str(
        get_value(config, "episode_selection", "first")
    ).strip().lower()
    if episode_selection not in {"first", "random"}:
        raise ValueError("episode_selection must be first|random.")

    output_relative = _safe_relative_output(
        get_value(config, "output_name", ""), default="fsq_terminator_probe"
    )
    eval_dir = _HERE.parent
    collection_dir = eval_dir / "outputs" / "fsq_terminator_eval" / output_relative
    exclude = as_list(get_value(config, "train_exclude_nodes", []))
    return {
        "project_root": str(project_root),
        "lerobot_root": str(project_root / "lerobot"),
        "fsq_terminator_eval_dir": str(eval_dir),
        "fsq_model_labels": " ".join(item["label"] for item in resolved),
        "fsq_model_count": len(resolved),
        "fsq_model_label": selected["label"],
        "fsq_model_path": selected["fsq_eval_model_path"],
        "fsq_terminator_overlay_path": selected[
            "fsq_eval_terminator_overlay_path"
        ],
        "fsq_model_source": selected["fsq_eval_model_source"],
        "fsq_code_space_id": selected["fsq_eval_code_space_id"],
        "fsq_model_run": selected["run_name"],
        "fsq_model_epoch_tag": selected["fsq_eval_epoch_tag"],
        "fsq_skills_dir": str(Path(selected["fsq_eval_skillset_dir"]) / "skills"),
        "skill_dataset_dir": selected["fsq_eval_dataset_dir"],
        "target_task": target_task,
        "task_ids": "all"
        if task_ids is None
        else json.dumps(task_ids, separators=(",", ":")),
        "episode_ids": json.dumps(episode_ids, separators=(",", ":")),
        "episodes_per_task": episodes_per_task,
        "episode_selection": episode_selection,
        "task_id_space": task_id_space,
        "eval_seed": int(get_value(config, "seed", 42)),
        "eval_resume": str(as_bool(get_value(config, "resume", False))).lower(),
        "eval_collection_dir": str(collection_dir),
        "eval_out_dir": str(collection_dir / "models" / selected["label"]),
        "eval_end_threshold": float(get_value(config, "end_threshold", 0.5)),
        "eval_max_plot_entries": int(get_value(config, "max_plot_entries", 0)),
        "eval_max_plot_samples": int(get_value(config, "max_plot_samples", 3)),
        "eval_batch_size": int(get_value(config, "batch_size", 64)),
        "video_frame_stride": int(_at(config, "video", "frame_stride", 1)),
        "video_fps": int(_at(config, "video", "fps", 10)),
        "eval_num_gpus": int(get_value(config, "eval_num_gpus", 1)),
        "eval_partition": ",".join(
            as_list(get_value(config, "train_partition", ["debug"]))
        ) or "debug",
        "eval_qos": str(get_value(config, "train_qos", "base_qos")),
        "eval_gres": str(_at(config, "slurm", "gres", "gpu:1")),
        "eval_cpus_per_task": int(_at(config, "slurm", "cpus", 8)),
        "eval_mem": str(_at(config, "slurm", "memory", "64G")),
        "eval_time": str(_at(config, "slurm", "time", "01:00:00")),
        "eval_nodelist": str(get_value(config, "train_nodelist", "")),
        "eval_exclude_nodes": ",".join(exclude),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    settings = build_settings(load_config(args.config), model_override=args.model_label)
    if args.shell:
        print_shell(settings)
    else:
        for key, value in settings.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
