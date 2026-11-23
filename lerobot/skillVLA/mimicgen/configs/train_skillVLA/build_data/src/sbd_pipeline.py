"""Build missing SBD prerequisites inside the existing SkillVLA GPU process."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys

import yaml

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "lerobot/src/lerobot").is_dir())
SHARED = ROOT / "lerobot/skillVLA/libero/configs"
sys.path.insert(0, str(SHARED / "train_skills/src"))
sys.path.insert(0, str(ROOT / "lerobot/skillVLA/mimicgen/configs/src"))
from train_skills_config import build_data_settings
from data_modes import configure_data_mode, validate_skillset_mode


def recipe(cfg):
    if "sbd_recipe" in cfg:
        return deepcopy(cfg["sbd_recipe"])
    path = ROOT / "lerobot/skillVLA/mimicgen/configs/train_skills/build_data/build_data_config.yaml"
    return yaml.safe_load(path.read_text())


def settings(cfg):
    # The SBD YAML supplies the detector recipe, while this submission owns the
    # target, storage, PT/FT mode and PT reference. No live YAML is read by workers.
    source = {**cfg, **recipe(cfg)}
    source.pop("sbd_recipe", None)
    source.pop("skillset_seg_name", None)
    for key in ("project_root", "dataset_root", "outputs_root", "server", "workflow_domain",
                "target_dataset", "skillvla_data_mode"):
        if key in cfg:
            source[key] = cfg[key]
    reference = cfg.get("skillvla_pt_reference", cfg.get("fsq_snap_reference"))
    if reference is not None:
        source["skillvla_pt_reference"] = deepcopy(reference)
    source = configure_data_mode(source)
    resolved = build_data_settings(source)
    cpus = int(cfg["slurm"]["cpus"])
    workers = int(resolved["skillset_single_gpu_workers"])
    if workers < 1 or resolved["skillset_cpus_per_task"] > cpus:
        raise ValueError("Automatic SBD needs positive skillset_single_gpu_workers and SBD CPUs <= SkillVLA slurm.cpus")
    return source, resolved


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_manifest(cfg, resolved):
    directory = Path(resolved["skillset_dir"])
    path = directory / "skillset_manifest.json"
    if not path.is_file():
        return
    manifest = json.loads(path.read_text())
    if Path(manifest["dataset_dir"]).resolve() != Path(resolved["raw_dataset_dir"]).resolve():
        raise ValueError("Existing SBD belongs to another dataset")
    if Path(manifest["policy_path"]).parts[-4:] != Path(resolved["dp_policy_path"]).parts[-4:]:
        raise ValueError("Existing SBD uses another DP checkpoint")
    validate_skillset_mode(cfg, manifest, directory)
    expected = {
        "detector": {
            "eval_at_step": "skillset_dn_step", "replan_interval": "skillset_replan_interval",
            "n_gmm_components": "skillset_n_gmm", "nms_dist": "skillset_nms_dist",
            "smooth_window": "skillset_smooth_window", "savgol_polyorder": "skillset_savgol_polyorder",
            "score_power": "skillset_score_power", "score_metric": "skillset_score_metric",
            "terminal_mask_frames": "skillset_terminal_mask_frames", "min_skill_len": "skillset_min_skill_len",
            "min_skills": "skillset_min_skills", "gmm_covariance": "skillset_gmm_covariance",
            "gmm_n_init": "skillset_gmm_n_init", "gmm_max_iter": "skillset_gmm_max_iter",
            "gmm_weighted": "skillset_gmm_weighted", "min_effective_samples": "skillset_min_effective_samples",
        },
        "probe": {
            "count": "skillset_probe_count", "generation": "skillset_probe_generation", "alpha": "skillset_probe_alpha",
            "gaussian_sampling": "skillset_gaussian_sampling", "gaussian_include_mean": "skillset_gaussian_include_mean",
            "gaussian_seed": "skillset_gaussian_seed", "denoise_steps": "skillset_denoise_steps",
            "denoise_output": "skillset_denoise_output", "pca_variance": "skillset_pca_variance",
            "pca_stride": "skillset_pca_stride", "pca_scale_mode": "skillset_pca_scale_mode",
        },
        "action": {"mode": "skillset_action_mode", "gripper_mode": "skillset_gripper_mode"},
    }
    for section, fields in expected.items():
        for name, key in fields.items():
            if manifest[section].get(name) != resolved[key]:
                raise ValueError(f"Existing SBD differs at {section}.{name}; select another SBD output suffix")
    policy = json.loads((Path(resolved["dp_policy_path"]) / "config.json").read_text())
    initial = int(policy["n_obs_steps"]) if resolved["skillset_initial_mask"] else 0
    if int(manifest["detector"].get("initial_mask_frames", 0)) != initial:
        raise ValueError("Existing SBD initial mask differs from this build")
    if manifest["probe"].get("gaussian_center", "current_gt") != resolved["skillset_gaussian_center"]:
        raise ValueError("Existing SBD Gaussian center differs from this build")
    for section, name, key in [
        ("probe", "exclude_indices", "skillset_probe_exclude_indices"),
        ("action", "gripper_indices", "skillset_gripper_indices"),
        ("action", "gripper_values", "skillset_gripper_values"),
    ]:
        expected_values = [float(value) for value in resolved[key].split(",") if value]
        if manifest[section].get(name) != expected_values:
            raise ValueError(f"Existing SBD differs at {section}.{name}")
    if manifest.get("mode") != resolved["skillset_mode"]:
        raise ValueError("Existing SBD probe mode differs from this build")
    if float(manifest["detector"].get("boundary_threshold_scale", 1)) != resolved["skillset_boundary_threshold_scale"]:
        raise ValueError("Existing SBD threshold scale differs from this build")


def is_complete(resolved):
    directory = Path(resolved["skillset_dir"])
    if not (directory / ".complete").is_file():
        return False
    verifier = _load(SHARED / "train_skillVLA/build_data/src/verify_skillset.py", "coffee_sbd_verify")
    mapping = verifier.task_to_episodes(Path(resolved["raw_dataset_dir"]), None)
    return (any(mapping.values()) and any((directory / "skills").rglob("*.npz"))
            and all(verifier.is_done(directory / "skills", ep, task)
                    for task, episodes in mapping.items() for ep in episodes))


def _env_value(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return ",".join(map(str, value))
    return str(value)


def progress_metadata(resolved, output):
    """Publish the actual pipeline paths for the Desk without exposing the process environment."""
    import pandas as pd
    raw = Path(resolved["raw_dataset_dir"])
    total = int(json.loads((raw / "meta/info.json").read_text())["total_episodes"])
    tasks = pd.read_parquet(raw / "meta/tasks.parquet").reset_index()
    task_ids = sorted(int(value) for value in tasks["task_index"].unique())
    if total < 1 or not task_ids:
        raise ValueError("Source dataset has no tasks/episodes")
    env = {key.upper(): _env_value(value) for key, value in resolved.items()}
    env.update(LIBERO_EXPECTED_EPISODES=str(total), LIBERO_TASK_INDICES_JSON=json.dumps(task_ids),
               LIBERO_SINGLE_GPU_BUILD="true")
    output = Path(output)
    staging = output.parent / (".build-" + hashlib.sha256(output.name.encode()).hexdigest()[:16])
    return {"kind": "skillvla_build", "run_id": os.environ.get("SBD_TRAINING_PARENT_ID", ""),
            "environment": env, "output_dir": str(output), "fsq_latents_path": str(staging / "skill_latents.npz")}


def publish_progress(metadata):
    status = os.environ.get("SBD_TRAINING_STATUS_PATH")
    if status:
        path = Path(status).with_name("sbd_build_progress.json")
        temporary = path.with_suffix(f".tmp-{os.getpid()}")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(path)


def ensure(cfg, output):
    source, resolved = settings(cfg)
    validate_manifest(source, resolved)
    metadata = progress_metadata(resolved, output)
    publish_progress(metadata)
    if is_complete(resolved):
        print("[1/3] Verified full SBD already complete; reusing it", flush=True)
        return
    for path in (resolved["raw_dataset_dir"], resolved["dp_policy_path"]):
        if not Path(path).is_dir():
            raise FileNotFoundError(path)
    work = Path(output).parent / (".sbd-" + Path(output).name)
    work.mkdir(parents=True, exist_ok=True)
    config_path = work / "sbd_build_config.yaml"
    config_path.write_text(yaml.safe_dump(source, sort_keys=False, allow_unicode=True))
    import pandas as pd
    rows = pd.concat([pd.read_parquet(p, columns=["episode_index"])
                      for p in (Path(resolved["raw_dataset_dir"]) / "meta/episodes").rglob("file-*.parquet")])
    if len(rows) != int(metadata["environment"]["LIBERO_EXPECTED_EPISODES"]):
        raise ValueError("Source episode count differs from its metadata")
    env = dict(os.environ)
    env.update(metadata["environment"])
    env.update(TRAIN_SKILLS_CONFIG=str(config_path), TRAIN_DATA=source["target_dataset"],
               LIBERO_EXPECTED_EPISODES=str(len(rows)), LIBERO_SINGLE_GPU_BUILD="true",
               SBD_SRUN_SESSION="1")
    runner = _load(SHARED / "train_skills/build_data/src/run_single_gpu_workers.py", "coffee_sbd_runner")
    builder = runner.LIBEROSharedGPUBuilder(env, log_dir=work / "logs")
    previous = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, lambda *_: builder.stop("Coffee build cancelled"))
        print(f"[1/3] Building full SBD inside this GPU allocation: {len(rows)} episodes, {builder.workers} workers", flush=True)
        builder.run()
    finally:
        builder.stop()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    validate_manifest(source, resolved)
    if not is_complete(resolved):
        raise RuntimeError("SBD is incomplete; FSQ/SkillVLA stages were not started")
