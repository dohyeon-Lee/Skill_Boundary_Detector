"""Describe a bounded ABC evaluation directly from native dataset metadata."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _source_rows(cfg):
    from lerobot.datasets.abc_native_io import load_abc_native_episodes

    root = Path(cfg["project_root"])
    dataset_root = Path(str(cfg.get("dataset_root", "dataset_ABC"))).expanduser()
    dataset = (dataset_root if dataset_root.is_absolute() else root / dataset_root) / cfg["target_dataset"]
    tasks = cfg["source_tasks"]
    if not isinstance(tasks, list) or not tasks or any(not str(task).strip() for task in tasks):
        raise ValueError("source_tasks must be a non-empty list of official ABC task names")
    rows = load_abc_native_episodes(dataset)
    missing = set(tasks) - set(rows["source_task"])
    if missing:
        raise ValueError(f"Unknown ABC source_tasks: {sorted(missing)}")
    return rows[rows["source_task"].isin(tasks)]


def resolve_task_ids(cfg):
    """Select all or the first N language prompts from official source tasks."""
    rows = _source_rows(cfg)
    ids = sorted({int(task_id) for values in rows["task_indices"] for task_id in values})
    if not ids:
        raise ValueError("source_tasks selects no ABC language tasks")
    count = cfg.get("n_tasks", "all")
    if isinstance(count, str) and count.strip().lower() == "all":
        return ids
    if isinstance(count, bool) or not (
        isinstance(count, int) or isinstance(count, str) and count.strip().isdigit()
    ):
        raise ValueError("n_tasks must be all or a positive integer")
    count = int(count)
    if count < 1 or count > len(ids):
        raise ValueError(f"n_tasks must be between 1 and {len(ids)}, or all")
    return ids[:count]


def _training_pca_episodes(checkpoint, dataset, rows, marker):
    """Resolve the saved DP dataset filters, independently of eval prompts."""
    path = checkpoint / "train_config.json"
    if not path.is_file():
        raise ValueError("pca_scope: dataset requires the checkpoint's train_config.json to recover its DP training episodes")
    training = json.loads(path.read_text())["dataset"]
    names = {Path(str(training.get("root") or "")).name,
             str(training.get("repo_id") or "").rsplit("/", 1)[-1]}
    if dataset.name not in names:
        raise ValueError("PCA dataset differs from the dataset recorded in the DP training config")
    tasks = training.get("tasks")
    episodes = training.get("episodes")
    selection = training.get("episode_selection_manifest")
    identity = {"train_config_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if episodes is not None and selection is not None:
        raise ValueError("Saved DP episodes and episode_selection_manifest are mutually exclusive")
    if selection is not None:
        selection_path = Path(selection).expanduser()
        if not selection_path.is_absolute():
            selection_path = dataset / selection_path
        elif not selection_path.is_file():
            # Checkpoints copied from another server retain that server's root.
            old_root = Path(str(training.get("root") or ""))
            try:
                selection_path = dataset / selection_path.relative_to(old_root)
            except ValueError:
                pass
        saved = json.loads(selection_path.read_text())
        if saved.get("format") != "sbd_abc_sim_episode_selection/v1":
            raise ValueError("Invalid DP episode selection manifest format")
        request = marker.get("request_sha256")
        if request and saved.get("dataset_request_sha256") and request != saved["dataset_request_sha256"]:
            raise ValueError("DP episode selection belongs to a different ABC dataset build")
        episodes = saved.get("episodes", [])
        if len(episodes) != saved.get("episode_count"):
            raise ValueError("Invalid DP episode selection count")
        manifest_tasks = saved.get("tasks", [])
        effective_tasks = tasks if tasks is not None else marker.get("default_source_tasks")
        if manifest_tasks and effective_tasks is not None and list(effective_tasks) != manifest_tasks:
            raise ValueError("Saved DP task filter differs from its episode selection manifest")
        tasks = effective_tasks if effective_tasks is not None else (manifest_tasks or None)
        identity["episode_selection_sha256"] = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    if tasks is None:
        tasks = marker.get("default_source_tasks") or None
    if episodes is None:
        episodes = marker.get("default_episode_ids") or None
    selected = rows
    if episodes is not None:
        ids = list(map(int, episodes))
        if not ids or len(ids) != len(set(ids)) or set(ids) - set(map(int, rows["episode_index"])):
            raise ValueError("Saved DP training episode IDs are empty, duplicated, or missing from this dataset")
        selected = selected[selected["episode_index"].isin(ids)]
    if tasks is not None:
        if not tasks or set(tasks) - set(rows["source_task"]):
            raise ValueError("Saved DP training task filter is empty or unavailable in this dataset")
        selected = selected[selected["source_task"].isin(tasks)]
    ids = sorted(map(int, selected["episode_index"]))
    if not ids:
        raise ValueError("Saved DP training filters select no PCA episodes")
    return ids, identity


def make_source(cfg: dict, settings: dict) -> dict:
    from lerobot.datasets.abc_native_io import (
        load_abc_native_episodes,
        load_abc_native_marker,
    )

    root = Path(settings["project_root"])
    dataset_root = Path(str(cfg.get("dataset_root", "dataset_ABC"))).expanduser()
    dataset = (dataset_root if dataset_root.is_absolute() else root / dataset_root) / cfg["target_dataset"]
    marker = load_abc_native_marker(dataset)
    checkpoint = Path(settings["policy_checkpoint_override"]).expanduser()
    if not settings["policy_checkpoint_override"]:
        raise ValueError("Standalone DP evaluation requires policy_checkpoint")
    checkpoint = (checkpoint if checkpoint.is_absolute() else root / checkpoint).resolve()
    policy = json.loads((checkpoint / "config.json").read_text())
    if policy.get("type") != "diffusion" or not policy.get("state_only"):
        raise ValueError("Standalone SBD evaluation currently requires a state-only diffusion checkpoint")
    if settings["task_id_space"] != "dataset":
        raise ValueError("Standalone ABC evaluation requires task_id_space: dataset")
    if settings["threshold_calibration"]["source"] != "selected":
        raise ValueError("Standalone evaluation calibrates only on selected episodes")
    tasks = cfg.get("source_tasks", [settings["target_task"]])
    if not isinstance(tasks, list) or not tasks or any(not str(task).strip() for task in tasks):
        raise ValueError("source_tasks must be a non-empty list of official ABC task names")
    rows = load_abc_native_episodes(dataset)
    all_rows = rows
    missing_tasks = set(tasks) - set(rows["source_task"])
    if missing_tasks:
        raise ValueError(f"Unknown ABC source_tasks: {sorted(missing_tasks)}")
    rows = rows[rows["source_task"].isin(tasks)].sort_values("episode_index")
    episodes = []
    task_ids = settings["task_ids"]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task_ids must not contain duplicates")
    for task_id in task_ids:
        selected = rows[rows["task_indices"].map(lambda values: task_id in values)]
        count = settings["n_episodes"]
        if selected.empty:
            raise ValueError(f"Task {task_id} has no episodes in {tasks}")
        # n_episodes is a per-prompt cap: sparse prompts use every available
        # episode, without duplicating samples or dropping the prompt.
        for row in selected.head(count).to_dict("records"):
            episodes.append({
                "episode_index": int(row["episode_index"]),
                "task_index": task_id,
                "source_task": str(row["source_task"]),
                "length": int(row["length"]),
                "states_actions_path": str(row["states_actions_path"]),
            })
    if len({row["episode_index"] for row in episodes}) != len(episodes):
        raise ValueError("Requested language tasks overlap on the same episodes")
    expected = settings["threshold_calibration"]["expected_episodes"]
    if expected is not None and expected != len(episodes):
        raise ValueError(f"Threshold calibration expects {expected} episodes; selected {len(episodes)}")
    settings["threshold_calibration"]["expected_episodes"] = len(episodes)
    action_dim = int(policy["output_features"]["action"]["shape"][0])
    if action_dim != int(marker["action_dim"]):
        raise ValueError("Checkpoint and ABC dataset action dimensions differ")
    info = json.loads((dataset / "meta/info.json").read_text())
    names = info["features"]["action"]["names"]
    grippers = [index for index, name in enumerate(names) if "gripper" in name.lower()]
    modes = [mode for flag, mode in (
        ("use_delta_joint_actions", "delta_joint"),
        ("use_delta_pose_actions", "delta_pose"),
        ("use_relative_actions", "anchor_relative"),
    ) if policy.get(flag)]
    if len(modes) > 1:
        raise ValueError("Checkpoint has conflicting action transforms")
    pca_scope = str(cfg.get("pca_scope", "selected"))
    if pca_scope not in {"selected", "source_tasks", "dataset"}:
        raise ValueError("pca_scope must be selected|source_tasks|dataset")
    training_ids, training_source = (
        _training_pca_episodes(checkpoint, dataset, all_rows, marker)
        if pca_scope == "dataset" else (None, None)
    )
    files = {}
    required = ["config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"]
    for name in required:
        path = checkpoint / name
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(checkpoint.glob("policy_*processor*.safetensors")):
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "evaluation_source": "abc_native_subset",
        "dataset_name": dataset.name,
        "dataset_dir": str(dataset.resolve()),
        "dataset_marker_sha256": hashlib.sha256((dataset / "native_dataset.json").read_bytes()).hexdigest(),
        "source_tasks": tasks,
        "policy_path": str(checkpoint),
        "checkpoint_sha256": files,
        "image_key": settings["report_image_key"],
        "evaluation_episodes": episodes,
        "pca_scope": pca_scope,
        **({"pca_episode_ids": sorted(row["episode_index"] for row in episodes)}
           if pca_scope == "selected" else {}),
        **({"pca_episode_ids": training_ids, "pca_source_tasks": [], "pca_training_source": training_source}
           if pca_scope == "dataset" else {}),
        "action": {
            "dim": action_dim,
            "mode": modes[0] if modes else "dataset",
            "relative_exclude_joints": ["gripper"],
            "gripper_mode": "continuous",
            "gripper_indices": grippers,
            "gripper_values": [-1, 1],
            "gripper_threshold": 0.0,
        },
        "probe": {"exclude_indices": grippers},
        "detector": {
            "noise_scheduler_type": "DDIM",
            "num_inference_steps": 10,
            "eval_at_step": 7,
            "replan_interval": 8,
            "n_gmm_components": 5,
            "min_skill_len": 15,
        },
    }


def prepare_source(settings: dict) -> None:
    """Create only an immutable evaluation manifest; never a training skillset."""
    manifest = settings.get("source_manifest")
    if manifest is None:
        return
    path = Path(settings["skillset_dir"]) / "skillset_manifest.json"
    if path.is_file() and json.loads(path.read_text()) != manifest:
        raise ValueError(
            f"Evaluation source changed at {path}; choose a new output_name "
            "for a different checkpoint or episode selection"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(manifest, indent=2) + "\n")
        temp.replace(path)
