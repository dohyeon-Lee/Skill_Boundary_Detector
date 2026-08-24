#!/usr/bin/env python3
"""Resolve an FSQ-only, episode-exact GT skill replay evaluation."""

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

DEFAULT_CONFIG_PATH = _HERE.parent / "fsq_gt_replay_config.yaml"


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


def _normalize_checkpoints(values: object, *, field: str) -> list[str]:
    checkpoints = [
        str(value).strip().lower()
        for value in as_list(values)
        if str(value).strip()
    ]
    if not checkpoints:
        raise ValueError(f"{field} must contain at least one checkpoint.")
    if len(checkpoints) != len(set(checkpoints)):
        raise ValueError(f"{field} contains duplicates: {checkpoints}.")
    return checkpoints


def _fsq_checkpoints(config: dict) -> list[str]:
    return _normalize_checkpoints(
        get_value(config, "fsq_eval_checkpoint", "last"), field="fsq_eval_checkpoint"
    )


def _run_names(config: dict) -> list[str]:
    names = [
        str(value).strip()
        for value in as_list(get_value(config, "fsq_eval_run_name", ""))
        if str(value).strip()
    ]
    if not names:
        raise ValueError("fsq_eval_run_name must name at least one FSQ run.")
    if len(names) != len(set(names)):
        raise ValueError(f"fsq_eval_run_name contains duplicates: {names}.")
    return names


def _model_names(config: dict, run_names: list[str]) -> list[str]:
    """Resolve the user-facing HTML label paired with each FSQ run."""
    raw_names = get_value(config, "fsq_eval_model_name", None)
    if raw_names is None:
        return list(run_names)
    values = raw_names if isinstance(raw_names, (list, tuple)) else [raw_names]
    names = [str(value).strip() for value in values]
    if len(names) != len(run_names) or any(not name for name in names):
        raise ValueError(
            "fsq_eval_model_name must contain one non-empty display name for each "
            f"fsq_eval_run_name entry ({len(run_names)} expected, got {len(names)})."
        )
    if len(names) != len(set(names)):
        raise ValueError(f"fsq_eval_model_name contains duplicates: {names}.")
    return names


def _model_entries(config: dict) -> list[dict[str, str]]:
    """Resolve paired run-folder and user-facing model names.

    ``fsq_eval_models`` is the preferred form. The two legacy scalar/list keys
    remain accepted so old snapshots and launch commands keep working.
    """
    raw_entries = get_value(config, "fsq_eval_models", None)
    if raw_entries is None:
        run_names = _run_names(config)
        model_names = _model_names(config, run_names)
        return [
            {"run_name": run_name, "model_name": model_name}
            for run_name, model_name in zip(run_names, model_names, strict=True)
        ]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("fsq_eval_models must be a non-empty list of mappings.")
    entries: list[dict[str, str]] = []
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise ValueError(
                f"fsq_eval_models[{index}] must be a mapping with run_name and model_name."
            )
        run_name = str(raw_entry.get("run_name") or "").strip()
        model_name = str(raw_entry.get("model_name") or "").strip()
        if not run_name or not model_name:
            raise ValueError(
                f"fsq_eval_models[{index}] requires non-empty run_name and model_name."
            )
        entries.append({"run_name": run_name, "model_name": model_name})
    run_names = [entry["run_name"] for entry in entries]
    model_names = [entry["model_name"] for entry in entries]
    if len(run_names) != len(set(run_names)):
        raise ValueError(f"fsq_eval_models contains duplicate run_name values: {run_names}.")
    if len(model_names) != len(set(model_names)):
        raise ValueError(
            f"fsq_eval_models contains duplicate model_name values: {model_names}."
        )
    return entries


def build_settings(
    config: dict,
    *,
    checkpoint_override: str | None = None,
    run_override: str | None = None,
    checkpoint_list_override: list[str] | None = None,
) -> dict:
    model_entries = _model_entries(config)
    run_names = [entry["run_name"] for entry in model_entries]
    model_names = [entry["model_name"] for entry in model_entries]
    if run_override is not None and run_override not in run_names:
        raise ValueError(
            f"Run override {run_override!r} is not one of the configured run names {run_names}."
        )
    selected_run = run_override or run_names[0]
    selected_model_name = model_names[run_names.index(selected_run)]
    config = {**config, "fsq_eval_run_name": selected_run}
    project_root = Path(str(get_value(config, "project_root"))).expanduser().resolve()
    dataset_root = Path(str(get_value(config, "dataset_root", "dataset"))).expanduser()
    outputs_root = Path(str(get_value(config, "outputs_root", "outputs"))).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    if not outputs_root.is_absolute():
        outputs_root = project_root / outputs_root

    if checkpoint_list_override is None:
        # Submission time: training may still be running, so checkpoints that do
        # not exist yet are dropped and the run is sized to what is on disk now.
        requested_checkpoints = _fsq_checkpoints(config)
        allow_missing = True
    else:
        # Job time: the submitting shell froze this list, so every entry existed
        # then and must still exist.  Rescanning here instead would let the
        # expected set grow with checkpoints written after submission, and the
        # collection report would then wait forever for jobs that never ran.
        requested_checkpoints = _normalize_checkpoints(
            checkpoint_list_override, field="--expected-checkpoints"
        )
        allow_missing = False
    artifacts = []
    skipped_checkpoints = []
    for checkpoint in requested_checkpoints:
        artifact = _resolve_fsq_artifact(
            config,
            dataset_root=dataset_root,
            outputs_root=outputs_root,
            checkpoint=checkpoint,
            missing_ok=allow_missing,
        )
        if artifact is None:
            skipped_checkpoints.append(checkpoint)
        else:
            artifacts.append(artifact)
    if not artifacts:
        raise FileNotFoundError(
            f"No fsq_eval_checkpoint entry {requested_checkpoints} has a trained "
            "FSQ checkpoint file yet."
        )
    if skipped_checkpoints:
        print(
            "fsq_gt_replay: skipping not-yet-trained checkpoints "
            f"{skipped_checkpoints}; evaluating "
            f"{[a['fsq_eval_resolved_checkpoint'] for a in artifacts]}.",
            file=sys.stderr,
        )
    resolved_checkpoints = [
        artifact["fsq_eval_resolved_checkpoint"] for artifact in artifacts
    ]
    # Encoding a checkpoint's skill latents is the only GPU computation in this
    # pipeline; replay itself decodes dataset video on CPU. Listing missing
    # latents keeps encoding in one prepass even when replay reserves a GPU only
    # to satisfy its inherited QOS.
    missing_latents = [
        artifact["fsq_eval_resolved_checkpoint"]
        for artifact in artifacts
        if not Path(artifact["fsq_eval_latents_path"]).is_file()
    ]
    epoch_tags = [artifact["fsq_eval_epoch_tag"] for artifact in artifacts]
    if len(resolved_checkpoints) != len(set(resolved_checkpoints)):
        raise ValueError(
            "fsq_eval_checkpoint entries resolve to duplicate checkpoints: "
            f"{resolved_checkpoints}."
        )
    if len(epoch_tags) != len(set(epoch_tags)):
        raise ValueError(
            f"fsq_eval_checkpoint entries resolve to duplicate epoch tags: {epoch_tags}."
        )
    selected_checkpoint = str(
        checkpoint_override or resolved_checkpoints[0]
    ).strip().lower()
    selected = [
        artifact
        for artifact in artifacts
        if selected_checkpoint
        in {
            artifact["fsq_eval_selected_checkpoint"],
            artifact["fsq_eval_resolved_checkpoint"],
        }
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Checkpoint override {selected_checkpoint!r} is not one of "
            f"{resolved_checkpoints}."
        )
    artifact = selected[0]
    model_path = Path(artifact["fsq_eval_model_path"])
    latents_path = Path(artifact["fsq_eval_latents_path"])
    skill_dataset_dir = Path(artifact["fsq_eval_dataset_dir"])
    skills_dir = Path(artifact["fsq_eval_skillset_dir"]) / "skills"
    dataset_dirs = {item["fsq_eval_dataset_dir"] for item in artifacts}
    artifact_run_names = {item["fsq_eval_run_name"] for item in artifacts}
    if len(dataset_dirs) != 1 or len(artifact_run_names) != 1:
        raise ValueError("All replay checkpoints must belong to one FSQ run and dataset.")

    target_dataset = skill_dataset_dir.name
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
    episode_selection = str(
        get_value(config, "episode_selection", "first")
    ).strip().lower()
    if episode_selection not in {"first", "random"}:
        raise ValueError("episode_selection must be first|random.")

    episode_source = str(
        get_value(config, "episode_source", "exact")
    ).strip().lower()
    if episode_source not in {"exact", "dataset"}:
        raise ValueError("episode_source must be exact|dataset.")
    required = [
        ("FSQ checkpoint", model_path),
        ("FSQ source dataset", skill_dataset_dir),
    ]
    if episode_source == "exact":
        eval_init_states_path = (
            dataset_root / "skillvla_dataset" / target_dataset / "eval_init_states.npz"
        )
        original_dataset_dir = project_root / "libero_original_dataset" / target_task
        required += [
            ("episode-exact map", eval_init_states_path),
            ("original LIBERO dataset", original_dataset_dir),
        ]
    else:
        # The dataset's own task table numbers every episode, so neither the
        # rendered episode-exact map nor the original HDF5s are consulted;
        # target_task stays on as the report label only.
        eval_init_states_path = None
        original_dataset_dir = None
    for label, path in required:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    # eval_num_gpus has always been the replay concurrency throttle rather than
    # a GRES count. Keep accepting it as a legacy alias for max_concurrent.
    requested_concurrency = int(
        get_value(
            config,
            "eval_max_concurrent",
            get_value(config, "eval_num_gpus", 1),
        )
    )
    workers_per_checkpoint = int(get_value(config, "workers_per_checkpoint", 1))
    if requested_concurrency <= 0 or workers_per_checkpoint <= 0:
        raise ValueError(
            "eval_max_concurrent and workers_per_checkpoint must be positive."
        )
    if episode_ids:
        selected_episode_upper_bound = len(episode_ids)
    elif task_ids is not None:
        selected_episode_upper_bound = len(task_ids) * episodes_per_task
    else:
        # task_ids: all — the dataset decides at run time, so no bound to clamp by.
        selected_episode_upper_bound = None
    worker_count = (
        workers_per_checkpoint
        if selected_episode_upper_bound is None
        else min(workers_per_checkpoint, selected_episode_upper_bound)
    )
    # Clamp to the number of array tasks that will actually exist: one task now
    # replays a chunk of checkpoints, so len(artifacts) overstates it.
    raw_checkpoints_per_job = get_value(config, "checkpoints_per_job", 4)
    if (
        isinstance(raw_checkpoints_per_job, str)
        and raw_checkpoints_per_job.strip().lower() == "all"
    ):
        checkpoints_per_job = len(artifacts)
    else:
        checkpoints_per_job = int(raw_checkpoints_per_job)
        if checkpoints_per_job <= 0:
            raise ValueError("checkpoints_per_job must be positive or 'all'.")
    chunk_count = -(-len(artifacts) // checkpoints_per_job)
    concurrent_jobs = min(requested_concurrency, chunk_count * worker_count)

    run_name = artifact["fsq_eval_run_name"]
    epoch_tag = artifact["fsq_eval_epoch_tag"]
    default_output = (
        f"{run_name}/{epoch_tags[0]}"
        if len(epoch_tags) == 1
        else f"{run_name}/compare_{'_'.join(epoch_tags)}"
    )
    output_value = str(get_value(config, "output_name", "") or "").strip()
    if len(run_names) > 1 and not output_value:
        raise ValueError(
            "output_name is required when fsq_eval_models lists several runs; "
            "it becomes the comparison parent folder."
        )
    output_relative = _safe_relative_output(output_value, default=default_output)
    report_title = output_relative.as_posix()
    eval_dir = _HERE.parent
    replay_outputs = eval_dir / "outputs" / "fsq_gt_replay"
    # With several runs, output_name is the comparison parent: each run keeps its
    # own collection beneath it and the combined page lands in <parent>/compare.
    if len(run_names) > 1:
        compare_dir = replay_outputs / output_relative / "compare"
        compare_collection_dirs = [
            str(replay_outputs / output_relative / name) for name in run_names
        ]
        collection_dir = replay_outputs / output_relative / run_name
    else:
        compare_dir = None
        compare_collection_dirs = []
        collection_dir = replay_outputs / output_relative
    checkpoint_output_dir = collection_dir / "checkpoints" / epoch_tag
    exclude = as_list(get_value(config, "train_exclude_nodes", []))
    # A blank replay override inherits the canonical global train_* Slurm
    # settings. Clusters that reject zero-GPU jobs on their GPU partitions can
    # still set an explicit CPU partition/QOS in this module's slurm section.
    local_replay_partitions = [
        str(value).strip()
        for value in as_list(_at(config, "slurm", "replay_partition", ""))
        if str(value).strip()
    ]
    global_train_partitions = [
        str(value).strip()
        for value in as_list(get_value(config, "train_partition", ["debug"]))
        if str(value).strip()
    ]
    replay_partitions = (
        local_replay_partitions or global_train_partitions or ["debug"]
    )
    local_replay_qos = str(_at(config, "slurm", "replay_qos", "") or "").strip()
    global_train_qos = str(get_value(config, "train_qos", "base_qos") or "").strip()
    replay_qos = local_replay_qos or global_train_qos or "base_qos"
    latent_gres = str(_at(config, "slurm", "gres", "gpu:1") or "gpu:1").strip()
    configured_replay_gres = str(
        _at(config, "slurm", "replay_gres", "") or ""
    ).strip()
    # Leaving replay_qos blank selects the global train placement, whose QOS
    # may require a GPU GRES even though replay computation itself stays on CPU.
    # An explicit replay_qos selects replay placement and permits true CPU-only
    # execution when replay_gres is blank.
    replay_gres = (
        configured_replay_gres
        if local_replay_qos
        else configured_replay_gres or "gpu:1"
    )
    return {
        "project_root": str(project_root),
        "lerobot_root": str(project_root / "lerobot"),
        "fsq_gt_replay_dir": str(eval_dir),
        "fsq_model_path": str(model_path),
        "fsq_latents_path": str(latents_path),
        "fsq_skills_dir": str(skills_dir),
        "skill_dataset_dir": str(skill_dataset_dir),
        "episode_source": episode_source,
        "eval_init_states_path": str(eval_init_states_path or ""),
        "original_dataset_dir": str(original_dataset_dir or ""),
        "fsq_run_name": run_name,
        "fsq_model_name": selected_model_name,
        "fsq_eval_run_names": " ".join(run_names),
        "fsq_run_count": len(run_names),
        "fsq_epoch_tag": epoch_tag,
        "fsq_eval_checkpoints": " ".join(resolved_checkpoints),
        "fsq_missing_latents": " ".join(missing_latents),
        "fsq_missing_latents_count": len(missing_latents),
        "fsq_skipped_checkpoints": " ".join(skipped_checkpoints),
        "fsq_expected_epoch_tags": json.dumps(epoch_tags, separators=(",", ":")),
        "fsq_checkpoint_count": len(artifacts),
        "target_task": target_task,
        "task_ids": "all"
        if task_ids is None
        else json.dumps(task_ids, separators=(",", ":")),
        "episode_ids": json.dumps(episode_ids, separators=(",", ":")),
        "episodes_per_task": episodes_per_task,
        "episode_selection": episode_selection,
        "eval_seed": int(get_value(config, "seed", 42)),
        "eval_max_concurrent": concurrent_jobs,
        "eval_worker_count": worker_count,
        "eval_resume": str(as_bool(get_value(config, "resume", False))).lower(),
        "eval_report_title": report_title,
        "eval_collection_dir": str(collection_dir),
        "eval_out_dir": str(checkpoint_output_dir),
        "eval_compare_dir": str(compare_dir) if compare_dir else "",
        "eval_compare_collection_dirs": json.dumps(
            compare_collection_dirs, separators=(",", ":")
        )
        if compare_collection_dirs
        else "",
        "eval_partition": ",".join(
            as_list(get_value(config, "train_partition", ["debug"]))
        ) or "debug",
        "eval_qos": str(get_value(config, "train_qos", "base_qos")),
        # In inherited train placement, replay reserves one GPU for QOS while
        # the evaluator itself still runs on CPU.
        "eval_gres": latent_gres,
        "eval_replay_gres": replay_gres,
        "eval_replay_partition": ",".join(replay_partitions),
        "eval_replay_qos": replay_qos,
        # The replay decodes two frames per occurrence and writes two PNGs; the
        # 8 CPU / 64G defaults belong to the GPU encoding prepass. Asking for
        # less lets far more replay tasks fit each selected node at once.
        "eval_replay_cpus": int(
            _at(config, "slurm", "replay_cpus", _at(config, "slurm", "cpus", 8))
        ),
        "eval_replay_mem": str(
            _at(config, "slurm", "replay_memory", _at(config, "slurm", "memory", "64G"))
        ),
        # Checkpoints replayed by ONE array task. Start-up (torch + lerobot
        # imports) costs minutes while a checkpoint's replay costs seconds to a
        # few minutes, so batching a few together is faster in wall clock AND
        # wastes far fewer node-minutes than one task per checkpoint.
        "eval_checkpoints_per_job": checkpoints_per_job,
        "eval_latents_time": str(_at(config, "slurm", "latents_time", "04:00:00")),
        "eval_cpus_per_task": int(_at(config, "slurm", "cpus", 8)),
        "eval_mem": str(_at(config, "slurm", "memory", "64G")),
        "eval_time": str(_at(config, "slurm", "time", "12:00:00")),
        "eval_nodelist": str(get_value(config, "train_nodelist", "")),
        "eval_exclude_nodes": ",".join(exclude),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--expected-checkpoints",
        default=None,
        help=(
            "Space-separated checkpoint list frozen by the submitting shell. "
            "Overrides fsq_eval_checkpoint so a job's expected epoch tags cannot "
            "grow while training keeps writing new checkpoints."
        ),
    )
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    settings = build_settings(
        load_config(args.config),
        checkpoint_override=args.checkpoint,
        run_override=args.run_name,
        checkpoint_list_override=(
            args.expected_checkpoints.split()
            if args.expected_checkpoints is not None
            else None
        ),
    )
    if args.shell:
        print_shell(settings)
    else:
        for key, value in settings.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
