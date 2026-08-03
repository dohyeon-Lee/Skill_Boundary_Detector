#!/usr/bin/env python3
"""skill_eval config: emit evaluation-only knobs + slurm settings as shell exports.

Root paths come from train_skills_config.py; this only owns eval-specific knobs.
Shared by dp_eval_config.yaml and fsq_eval_config.yaml — each yaml carries only its
own knobs (missing ones fall back to defaults), and the submit scripts force the
eval_run_dp / eval_run_fsq toggles through the environment (env wins over yaml).
Keys shared by both evals (thumb size, image key, slurm resources) use neutral
`eval_*` names; the historical `fsq_eval_*` spellings still work as fallbacks.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# reuse the train_skills config helpers (yaml load + shell emitter)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from train_skills_config import as_bool, as_list, get_value, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "eval_config.yaml"


def _folder_name(value: object, key: str) -> str:
    name = str(value or "").strip()
    if not name:
        raise ValueError(f"{key} must be a non-empty folder name.")
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"{key} must be a folder name, not a path: {name!r}")
    return name


def _resolve_fsq_artifact(
    cfg: dict,
    *,
    dataset_root: Path,
    outputs_root: Path,
    checkpoint: str,
) -> dict[str, str]:
    """Resolve one FSQ run entirely from its immutable training metadata."""
    run_name = _folder_name(get_value(cfg, "fsq_eval_run_name", ""), "fsq_eval_run_name")
    model_dir = outputs_root / "FSQ" / run_name
    if not model_dir.is_dir():
        raise FileNotFoundError(f"FSQ run folder not found: {model_dir}")

    if checkpoint in {"0", "best"}:
        model_path = model_dir / "FSQ.pt"
        epoch_tag = "best"
        resolved_checkpoint = "best"
        latents_path = model_dir / "skill_latents.npz"
    elif checkpoint == "last":
        candidates: list[tuple[int, Path]] = []
        for path in model_dir.glob("FSQ_epoch*.pt"):
            match = re.fullmatch(r"FSQ_epoch(\d+)\.pt", path.name)
            if match:
                candidates.append((int(match.group(1)), path))
        if not candidates:
            raise FileNotFoundError(f"No FSQ_epoch*.pt checkpoints found in {model_dir}")
        epoch, model_path = max(candidates, key=lambda item: item[0])
        epoch_tag = f"epoch{epoch:04d}"
        resolved_checkpoint = str(epoch)
        latents_path = model_dir / f"skill_latents_{epoch_tag}.npz"
    elif checkpoint.isdigit() and int(checkpoint) > 0:
        epoch = int(checkpoint)
        model_path = model_dir / f"FSQ_epoch{epoch:04d}.pt"
        epoch_tag = f"epoch{epoch:04d}"
        resolved_checkpoint = str(epoch)
        latents_path = model_dir / f"skill_latents_{epoch_tag}.npz"
    else:
        raise ValueError(
            "fsq_eval_checkpoint must be 'last', 'best', or a positive epoch, "
            f"got {checkpoint!r}."
        )
    if not model_path.is_file():
        raise FileNotFoundError(f"FSQ checkpoint not found: {model_path}")

    meta_path = model_dir / "fsq_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"FSQ training metadata not found: {meta_path}")
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid FSQ training metadata: {meta_path}: {error}") from error
    components = {
        key: _folder_name(meta.get(key), f"fsq_meta.{key}")
        for key in (
            "fsq_dataset_root",
            "target_dataset",
            "fsq_inputs_name",
            "skillset_seg_name",
            "skillset_name",
        )
    }
    skillset_dir = (
        dataset_root
        / components["fsq_dataset_root"]
        / components["target_dataset"]
        / components["fsq_inputs_name"]
        / components["skillset_seg_name"]
        / components["skillset_name"]
    )
    dataset_dir = dataset_root / components["target_dataset"]
    if not (skillset_dir / "skills").is_dir():
        raise FileNotFoundError(f"FSQ training skillset not found: {skillset_dir / 'skills'}")
    if not (dataset_dir / "videos").is_dir():
        raise FileNotFoundError(f"FSQ source dataset videos not found: {dataset_dir / 'videos'}")

    return {
        "fsq_eval_run_name": run_name,
        "fsq_eval_selected_checkpoint": checkpoint,
        "fsq_eval_resolved_checkpoint": resolved_checkpoint,
        "fsq_eval_model_dir": str(model_dir),
        "fsq_eval_model_path": str(model_path),
        "fsq_eval_epoch_tag": epoch_tag,
        "fsq_eval_latents_path": str(latents_path),
        "fsq_eval_meta_path": str(meta_path),
        "fsq_eval_skillset_dir": str(skillset_dir),
        "fsq_eval_dataset_dir": str(dataset_dir),
    }


def _fsq_checkpoints(cfg: dict) -> list[str]:
    checkpoints = [
        str(value).strip().lower()
        for value in as_list(get_value(cfg, "fsq_eval_checkpoint", "last"))
        if str(value).strip()
    ]
    if not checkpoints:
        raise ValueError("fsq_eval_checkpoint must contain at least one checkpoint.")
    if len(checkpoints) != len(set(checkpoints)):
        raise ValueError(f"fsq_eval_checkpoint contains duplicates: {checkpoints}.")
    return checkpoints


def build_settings(
    config_path: str | None = None,
    *,
    checkpoint_override: str | None = None,
) -> dict:
    cfg = load_config(config_path or DEFAULT_CONFIG_PATH)
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    dataset_root = Path(str(get_value(cfg, "dataset_root", "dataset"))).expanduser()
    outputs_root = Path(str(get_value(cfg, "outputs_root", "outputs"))).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    if not outputs_root.is_absolute():
        outputs_root = project_root / outputs_root
    eval_run_fsq = as_bool(get_value(cfg, "eval_run_fsq", True))
    eval_run_dp = as_bool(get_value(cfg, "eval_run_dp", True))
    fsq_checkpoints = _fsq_checkpoints(cfg) if eval_run_fsq else []
    fsq_artifacts = [
        _resolve_fsq_artifact(
            cfg,
            dataset_root=dataset_root,
            outputs_root=outputs_root,
            checkpoint=checkpoint,
        )
        for checkpoint in fsq_checkpoints
    ]
    if len({artifact["fsq_eval_model_path"] for artifact in fsq_artifacts}) != len(
        fsq_artifacts
    ):
        raise ValueError(
            "fsq_eval_checkpoint entries must resolve to different checkpoint files."
        )
    if checkpoint_override is not None:
        selected_checkpoint = str(checkpoint_override).strip().lower()
        fsq_artifact = _resolve_fsq_artifact(
            cfg,
            dataset_root=dataset_root,
            outputs_root=outputs_root,
            checkpoint=selected_checkpoint,
        )
    else:
        fsq_artifact = fsq_artifacts[0] if fsq_artifacts else {}
    fsq_eval_num_gpus = int(get_value(cfg, "fsq_eval_num_gpus", 1))
    if fsq_eval_num_gpus <= 0:
        raise ValueError("fsq_eval_num_gpus must be positive.")
    if fsq_checkpoints and fsq_eval_num_gpus > len(fsq_checkpoints):
        raise ValueError(
            "fsq_eval_num_gpus cannot exceed the number of checkpoints: "
            f"gpus={fsq_eval_num_gpus}, checkpoints={len(fsq_checkpoints)}."
        )
    dp_skillset_dir = ""
    if eval_run_dp:
        component_keys = (
            "fsq_dataset_root",
            "target_dataset",
            "fsq_inputs_name",
            "skillset_seg_name",
            "skillset_name",
        )
        components = {
            key: str(get_value(cfg, key, "")).strip()
            for key in component_keys
        }
        missing = [key for key, value in components.items() if not value]
        if missing:
            raise ValueError(f"DP eval artifact path is missing components: {missing}")
        invalid = [
            key
            for key, value in components.items()
            if Path(value).name != value
        ]
        if invalid:
            raise ValueError(
                "DP eval artifact components must be folder names, not paths: "
                f"{invalid}"
            )
        dp_skillset_dir = str(
            dataset_root
            / components["fsq_dataset_root"]
            / components["target_dataset"]
            / components["fsq_inputs_name"]
            / components["skillset_seg_name"]
            / components["skillset_name"]
        )
    # Slurm partition/qos/nodelist/exclude are canonical (global_config.yaml train_*).
    exclude = as_list(get_value(cfg, "train_exclude_nodes", []))
    output_suffix = str(get_value(cfg, "dp_eval_output_suffix", "")).strip()
    if output_suffix and not re.fullmatch(r"[A-Za-z0-9._-]+", output_suffix):
        raise ValueError(
            "dp_eval_output_suffix may contain only letters, digits, '.', '_' and '-', "
            f"got {output_suffix!r}"
        )
    random_far_fraction = float(get_value(cfg, "fsq_eval_random_far_fraction", 0.1))
    if not 0.0 < random_far_fraction <= 1.0:
        raise ValueError(
            f"fsq_eval_random_far_fraction must be in (0,1], got {random_far_fraction}."
        )
    return {
        "project_root":            str(project_root),
        "lerobot_root":            str(project_root / "lerobot"),
        "dataset_root":            str(dataset_root),
        "outputs_root":            str(outputs_root),
        "fsq_outputs_root":        str(outputs_root / "FSQ"),
        # which eval(s) to run
        "eval_run_fsq":            str(eval_run_fsq).lower(),
        "eval_run_dp":             str(eval_run_dp).lower(),
        "fsq_eval_checkpoints":    " ".join(
            artifact["fsq_eval_resolved_checkpoint"] for artifact in fsq_artifacts
        ),
        "fsq_eval_num_gpus":       fsq_eval_num_gpus,
        # DP artifact selection. Its immutable manifest owns all provenance.
        "dp_eval_skillset_dir":    dp_skillset_dir,
        "dp_eval_output_suffix":   output_suffix,
        # DP skill-boundary eval knobs
        "dp_eval_n_episodes":      int(get_value(cfg, "dp_eval_n_episodes", 10)),
        "dp_eval_task_ids":        " ".join(as_list(get_value(cfg, "dp_eval_task_ids", []))),
        "dp_eval_skill_video":     str(as_bool(get_value(cfg, "dp_eval_skill_video", False))).lower(),
        "dp_eval_show_start_end_frames": str(
            as_bool(get_value(cfg, "dp_eval_show_start_end_frames", True))
        ).lower(),
        "dp_eval_show_cos_graph":  str(as_bool(get_value(cfg, "dp_eval_show_cos_graph", True))).lower(),
        "dp_eval_show_gripper_graph": str(
            as_bool(get_value(cfg, "dp_eval_show_gripper_graph", True))
        ).lower(),
        **fsq_artifact,
        "fsq_eval_n_action_steps": int(get_value(cfg, "fsq_eval_n_action_steps", 5)),
        "fsq_eval_n_samples":      int(get_value(cfg, "fsq_eval_n_samples", 10)),
        "fsq_eval_max_entries":    int(get_value(cfg, "fsq_eval_max_entries", 10)),
        "fsq_eval_decoder_scope":  str(get_value(cfg, "fsq_eval_decoder_scope", "samples")),
        "fsq_eval_end_threshold":  str(get_value(cfg, "fsq_eval_end_threshold", 0.5)),
        "fsq_eval_random_far_skill": str(
            as_bool(get_value(cfg, "fsq_eval_random_far_skill", True))
        ).lower(),
        "fsq_eval_random_far_fraction": str(random_far_fraction),
        "fsq_eval_thumb_size":     int(get_value(cfg, "eval_thumb_size", get_value(cfg, "fsq_eval_thumb_size", 160))),
        "fsq_eval_image_key":      str(get_value(cfg, "eval_image_key", get_value(cfg, "fsq_eval_image_key", "observation.images.image"))),
        "fsq_eval_wandb_project":  str(get_value(cfg, "fsq_eval_wandb_project", "VAE_eval")),
        "fsq_eval_partition":      ",".join(as_list(get_value(cfg, "train_partition", ["debug"]))) or "debug",
        "fsq_eval_qos":            str(get_value(cfg, "train_qos", "base_qos")),
        "fsq_eval_gres":           str(get_value(cfg, "eval_gres", get_value(cfg, "fsq_eval_gres", "gpu:1"))),
        "fsq_eval_cpus_per_task":  int(get_value(cfg, "eval_cpus_per_task", get_value(cfg, "fsq_eval_cpus_per_task", 4))),
        "fsq_eval_mem":            str(get_value(cfg, "eval_mem", get_value(cfg, "fsq_eval_mem", "64G" if eval_run_fsq else "32G"))),
        "fsq_eval_time":           str(get_value(cfg, "eval_time", get_value(cfg, "fsq_eval_time", "02:00:00"))),
        "fsq_eval_nodelist":       str(get_value(cfg, "train_nodelist", "")),
        "fsq_eval_exclude_nodes":  ",".join(exclude),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--shell", action="store_true")
    args = ap.parse_args()
    settings = build_settings(args.config, checkpoint_override=args.checkpoint)
    if args.shell:
        print_shell(settings)
    else:
        for k, v in settings.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
