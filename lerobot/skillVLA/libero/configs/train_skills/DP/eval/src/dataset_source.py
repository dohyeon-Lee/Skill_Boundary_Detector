"""Select scene-exact LIBERO episodes directly; no FSQ skillset is required."""
from __future__ import annotations

import hashlib
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _full_dataset_pca_reference(cfg, settings, dataset, checkpoint, policy):
    """Find the same checkpoint's compatible original full-dataset PCA."""
    import numpy as np

    fsq_root = Path(str(cfg.get("fsq_dataset_root", "FSQ_dataset")))
    if not fsq_root.is_absolute():
        fsq_root = dataset.parent / fsq_root
    inputs = fsq_root / dataset.name / str(cfg.get("fsq_inputs_name", "FSQ_inputs"))
    dim = int(policy["output_features"]["action"]["shape"][0])
    probe = settings["probe_variants"][0]
    # A temporal-mean basis across all action dimensions is the original
    # LIBERO recipe. Other representations are fitted by the normal CPU path.
    if probe.get("mode") != "std" or probe.get("descriptor_pca_representation", "mean") != "mean":
        return None
    expected = {
        "dataset": dataset.name, "action_dim": dim,
        "action_mode": "anchor_relative" if policy.get("use_relative_actions") else "dataset",
        "descriptor_horizon": int(probe["pca_temporal_length"]),
        "stride": int(probe["pca_stride"]), "pca_scale_mode": "std",
        "variance_threshold": float(probe["pca_variance"]),
        "probe_action_indices": list(range(dim)),
    }
    for manifest_path in sorted(inputs.glob("*/skillset/skillset_manifest.json")):
        pca_path = manifest_path.parent / "action_probe_pca.npz"
        try:
            manifest = json.loads(manifest_path.read_text())
            if tuple(Path(manifest.get("policy_path", "")).parts[-4:]) != tuple(checkpoint.parts[-4:]):
                continue
            with np.load(pca_path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"]))
            if any(metadata.get(key) != value for key, value in expected.items()):
                continue
            if not metadata.get("normalizer_stats_sha256"):
                continue
            return {"path": str(pca_path.resolve()), "sha256": _sha256(pca_path)}
        except (OSError, ValueError, KeyError):
            continue
    return None


def make_source(cfg, settings):
    import pandas as pd

    helpers = Path(__file__).resolve().parents[3] / "skill_eval/src"
    sys.path.insert(0, str(helpers))
    from codebook_visualizer import _load_episodes_meta
    from dp_skillset_eval import select_episodes, select_stage1_suite_episodes

    root = Path(settings["project_root"])
    dataset_root = Path(cfg["dataset_root"]).expanduser()
    dataset = ((dataset_root if dataset_root.is_absolute() else root / dataset_root)
               / cfg["target_dataset"]).resolve()
    checkpoint = Path(settings["policy_checkpoint_override"])
    policy = json.loads((checkpoint / "config.json").read_text())
    if policy.get("type") != "diffusion" or not policy.get("state_only"):
        raise ValueError("LIBERO DP/eval requires a state-only diffusion checkpoint")
    if settings["threshold_calibration"]["source"] != "selected":
        raise ValueError("Partial evaluation calibrates thresholds on selected episodes")
    metadata = _load_episodes_meta(dataset).sort_values("episode_index")
    tasks = pd.read_parquet(dataset / "meta/tasks.parquet")
    if "task" not in tasks.columns:
        tasks = tasks.reset_index()
    names = dict(zip(tasks["task"].astype(str), tasks["task_index"].astype(int)))
    ep_task = {}
    by_episode = {}
    for row in metadata.to_dict("records"):
        indices = list(row.get("task_indices", [])) or [names[str(task)] for task in row["tasks"]]
        if len(indices) != 1:
            raise ValueError("LIBERO subset selection requires one task per episode")
        episode = int(row["episode_index"])
        ep_task[episode] = int(indices[0])
        by_episode[episode] = row
    ids, count = settings["task_ids"], settings["n_episodes"]
    if len(ids) != len(set(ids)):
        raise ValueError("task_ids must not contain duplicates")
    # Selection diagnostics must not contaminate --shell's exported settings.
    with redirect_stdout(sys.stderr):
        if settings["task_id_space"] == "suite":
            groups = select_stage1_suite_episodes(
                ep_task, ids, suite_name=settings["target_task"], dataset_dir=dataset, n_episodes=count,
            )
        else:
            groups = select_episodes(ep_task, ids, count)
    episodes = []
    for task, selected in groups:
        if len(selected) != count:
            raise ValueError(f"Task {task} has {len(selected)} episodes; requested {count}")
        episodes.extend({
            "episode_index": int(episode), "task_index": ep_task[episode],
            "selection_task_id": int(task), "length": int(by_episode[episode]["length"]),
        } for episode in selected)
    if {task for task, _ in groups} != set(ids):
        raise ValueError("Some requested LIBERO tasks have no episodes")
    if len({row["episode_index"] for row in episodes}) != len(episodes):
        raise ValueError("Requested LIBERO tasks overlap on the same episodes")
    expected = settings["threshold_calibration"]["expected_episodes"]
    if expected is not None and expected != len(episodes):
        raise ValueError(f"Threshold calibration expects {expected} episodes; selected {len(episodes)}")
    settings["threshold_calibration"]["expected_episodes"] = len(episodes)
    info = json.loads((dataset / "meta/info.json").read_text())
    action_dim = int(policy["output_features"]["action"]["shape"][0])
    if action_dim != int(info["features"]["action"]["shape"][0]):
        raise ValueError("Checkpoint and LIBERO dataset action dimensions differ")
    if policy.get("use_delta_joint_actions") or policy.get("use_delta_pose_actions"):
        raise ValueError("LIBERO eval uses stored delta-EEF actions or anchor-relative actions")
    action_names = info["features"]["action"].get("names") or []
    grippers = [i for i, name in enumerate(action_names) if "gripper" in str(name).lower()]
    if not grippers:
        grippers = [action_dim - 1]
    scope = str(cfg.get("pca_scope", "selected"))
    if scope not in {"selected", "dataset"}:
        raise ValueError("LIBERO pca_scope must be selected|dataset")
    checkpoint_files = [checkpoint / name for name in [
        "config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json",
    ]]
    checkpoint_files += sorted(checkpoint.glob("policy_*processor*.safetensors"))
    source_files = [dataset / "meta/info.json", dataset / "meta/tasks.parquet"]
    source_files += sorted((dataset / "meta/episodes").rglob("file-*.parquet"))
    if settings["task_id_space"] == "suite":
        source_files.append(dataset.parent / "skillvla_dataset" / dataset.name / "eval_init_states.npz")
    identity = {str(path.relative_to(dataset.parent)): _sha256(path) for path in source_files}
    reference = _full_dataset_pca_reference(cfg, settings, dataset, checkpoint, policy) if scope == "dataset" else None
    return {
        "schema_version": 1, "evaluation_source": "libero_subset",
        "dataset_name": dataset.name, "dataset_dir": str(dataset),
        "dataset_marker_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        "source_tasks": [], "policy_path": str(checkpoint),
        "checkpoint_sha256": {path.name: _sha256(path) for path in checkpoint_files},
        "image_key": settings["report_image_key"], "evaluation_episodes": episodes,
        "pca_scope": scope,
        **({"pca_reference": reference} if reference else {}),
        "pca_episode_ids": sorted(row["episode_index"] for row in episodes) if scope == "selected" else sorted(ep_task),
        "action": {"dim": action_dim, "mode": "anchor_relative" if policy.get("use_relative_actions") else "dataset",
                   "relative_exclude_joints": ["gripper"], "gripper_mode": "discrete",
                   "gripper_indices": grippers, "gripper_values": [-1, 1], "gripper_threshold": 0.0},
        "probe": {"exclude_indices": grippers},
        "detector": {"noise_scheduler_type": "DDIM", "num_inference_steps": 10,
                     "eval_at_step": 7, "replan_interval": 5, "n_gmm_components": 5, "min_skill_len": 10},
    }


def prepare_source(settings):
    manifest = settings.get("source_manifest")
    if manifest is None:
        return
    path = Path(settings["skillset_dir"]) / "skillset_manifest.json"
    if path.is_file() and json.loads(path.read_text()) != manifest:
        raise ValueError(f"Evaluation source changed at {path}; choose a new output_name")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(path)
