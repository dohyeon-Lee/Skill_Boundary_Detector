"""Resolve and validate the three checkpoint contracts before requesting a GPU."""
from __future__ import annotations

import json
import math
from pathlib import Path

CAMERA_KEYS = ["observation.images.top", "observation.images.left_wrist", "observation.images.right_wrist"]


def resolve_skillvla(policy: dict, global_config: dict, project_root: Path) -> dict:
    settings = dict(policy.get("skillvla") or {})
    settings["condition_source"] = str(settings.get("condition_source", "predictor")).lower()
    if settings["condition_source"] not in {"predictor", "gt"}:
        raise ValueError("condition_source must be predictor or gt")
    root = Path(str(global_config.get("outputs_root", "outputs_ABC")))
    if not root.is_absolute():
        root = project_root / root
    contracts = {}
    for component, directory in (("vsa", "VSA"), ("predictor", "Predictor"), ("terminator", "Terminator")):
        if component == "predictor" and settings["condition_source"] == "gt":
            settings.pop("predictor", None)
            continue
        spec = settings.get(component)
        if isinstance(spec, str):
            path = Path(spec).expanduser()
            if not path.is_absolute():
                path = project_root / path
        elif isinstance(spec, dict):
            run = str(spec.get("run_name", ""))
            if not run or Path(run).name != run or run in {".", ".."}:
                raise ValueError(f"policy.skillvla.{component}.run_name must be one folder name")
            step = str(spec.get("checkpoint", "last"))
            if step != "last":
                if not step.isdigit() or int(step) <= 0:
                    raise ValueError(f"invalid {component} checkpoint: {step}")
                step = f"{int(step):06d}"
            path = root / "skillVLA_PT" / directory / run / "checkpoints" / step / "pretrained_model"
        else:
            raise ValueError(f"policy.skillvla.{component} needs run_name/checkpoint")
        path = path.resolve(strict=True)  # Pin 'last' to a numeric checkpoint.
        if not (path / "model.safetensors").is_file():
            raise FileNotFoundError(f"incomplete {component} checkpoint: {path}")
        contract = json.loads((path / "config.json").read_text())
        expected_type = "skill_expert" if component == "vsa" else "skill_aux"
        if contract.get("type") != expected_type or not contract.get("bimanual_spatial_auxiliary"):
            raise ValueError(f"{component} must be an ABC bimanual {expected_type} checkpoint")
        if contract.get("skill_image_keys") != CAMERA_KEYS:
            raise ValueError(f"{component} camera order must be top/left wrist/right wrist")
        if contract.get("proprio_grounding", "none") != "none":
            raise ValueError(f"ABC {component} requires absolute joint coordinates")
        if component == "vsa":
            if contract["output_features"]["action"]["shape"] != [14]:
                raise ValueError("ABC VSA must output 14 absolute joint actions")
            if contract.get("skill_plan_state_dim", contract.get("chunk_end_state_dim")) != 14:
                raise ValueError("ABC full-skill plan must have 14 joint channels")
            if len(contract.get("chunk_end_state_q01", [])) != 14 or len(contract.get("chunk_end_state_q99", [])) != 14:
                raise ValueError("full-skill visualization requires all 14 checkpoint normalization bounds")
        elif not contract.get("train_skill_predictor" if component == "predictor" else "train_terminator"):
            raise ValueError(f"checkpoint does not contain the requested {component}")
        contracts[component] = contract
        settings[component] = str(path)
    identity = lambda c: (c.get("skill_code_space_id"), c.get("skill_fsq_levels"))
    if not identity(contracts["vsa"])[0] or any(identity(c) != identity(contracts["vsa"]) for c in contracts.values()):
        raise ValueError("VSA, Predictor and Terminator must use the same FSQ code space and levels")
    settings["chunk_size"] = int(contracts["vsa"]["chunk_size"])
    settings["termination_threshold"] = float(settings.get("termination_threshold", 0.5))
    if not math.isfinite(settings["termination_threshold"]) or not 0 < settings["termination_threshold"] <= 1:
        raise ValueError("termination_threshold must be in (0,1]")
    settings["min_skill_steps"] = int(settings.get("min_skill_steps", 0))
    settings["max_skill_steps"] = int(settings.get("max_skill_steps", 300))
    if not 0 <= settings["min_skill_steps"] < settings["max_skill_steps"]:
        raise ValueError("need 0 <= min_skill_steps < max_skill_steps")
    settings["fullskill_overlay"] = bool(settings.get("fullskill_overlay", True))
    # Give newly captured auxiliaries/clean originals a separate auto output;
    # old rollouts did not save these readouts and must not masquerade as new.
    settings["video_overlay_version"] = "abc_bimanual_aux_top_plan_v1"
    if settings["condition_source"] == "gt":
        dataset = settings.get("gt_dataset")
        if not dataset:
            raise ValueError("condition_source: gt requires gt_dataset (built SkillVLA dataset folder)")
        dataset = Path(dataset).expanduser()
        if not dataset.is_absolute():
            dataset = project_root / dataset
        dataset = dataset.resolve(strict=True)
        settings["gt_dataset"] = str(dataset)
        info = json.loads((dataset / "meta/info.json").read_text())
        if info.get("skill_fsq_levels") != contracts["vsa"]["skill_fsq_levels"]:
            raise ValueError("GT dataset FSQ levels differ from the VSA checkpoint")
        if dataset.parent.name != contracts["vsa"]["skill_code_space_id"]:
            raise ValueError("GT dataset must belong to the VSA checkpoint's FSQ code space")
        for filename in ("native_dataset.json", "../skill_latents.npz", "../skill_spatial_targets.npz"):
            if not (dataset / filename).is_file():
                raise FileNotFoundError(f"GT evaluation artifact missing: {dataset / filename}")
        episode_ids = settings.get("gt_episode_ids")
        if episode_ids is not None:
            if not isinstance(episode_ids, list) or not episode_ids or any(isinstance(x, bool) or int(x) != x for x in episode_ids):
                raise ValueError("gt_episode_ids must be null or a nonempty list of integer episode IDs")
            settings["gt_episode_ids"] = [int(x) for x in episode_ids]
            if len(set(settings["gt_episode_ids"])) != len(episode_ids):
                raise ValueError("gt_episode_ids cannot contain duplicates")
    return settings
