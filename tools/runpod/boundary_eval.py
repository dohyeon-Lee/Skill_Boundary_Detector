"""Resolve portable inputs for the existing skill-boundary evaluator."""
import importlib.util
import json
from pathlib import Path
import sys

RELATIVE = Path("lerobot/skillVLA/libero/configs/train_skills/DP/eval")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configuration(root, config):
    resolver = load("boundary_hf_config", root / RELATIVE / "src/config.py")
    evaluator = load("boundary_hf_eval", root / RELATIVE / "src/evaluate.py")
    evaluator.PROJECT_ROOT = root
    merged = resolver.load_config(config)
    settings = json.loads(resolver.build_settings(str(config))["dp_ablation_settings_json"])
    skillset = Path(settings["skillset_dir"])
    manifest = settings["source_manifest"]
    dataset = evaluator._portable_path(manifest["dataset_dir"], "dataset_filtered")
    override = settings.get("policy_checkpoint_override")
    checkpoint = evaluator._portable_path(
        str(root / override) if override and not Path(override).is_absolute() else override or manifest["policy_path"],
        "outputs_filtered")
    selected = [int(row["episode_index"]) for row in manifest["evaluation_episodes"]]
    calibration = selected
    paths = {checkpoint, dataset / "data", dataset / "meta"}
    if settings["task_id_space"] == "suite":
        paths.add(dataset.parent / "skillvla_dataset" / dataset.name / "eval_init_states.npz")
    # Step data is needed for the complete calibration/PCA fit. Report video
    # chunks are selected through metadata; retain canonical shared HF paths.
    from codebook_visualizer import _load_episodes_meta, _resolve_image_key, _video_path
    meta = _load_episodes_meta(dataset)
    image_key = _resolve_image_key(meta, settings["report_image_key"])
    paths.update(_video_path(dataset, meta, episode, image_key) for episode in selected)
    for path in paths:
        path.relative_to(root)
        if not path.exists():
            raise ValueError("Missing boundary evaluation input: " + str(path))
    print(f"SBD: calibration {len(calibration)} episodes / report {len(selected)} episodes", flush=True)
    return {"input_paths": sorted(str(path) for path in paths),
            "models_json": "[]", "boundary_eval": True}, merged


def upload_groups(paths):
    """Avoid queuing one Desk job per small skill NPZ or metadata file."""
    grouped = set()
    for path in paths:
        parts = Path(path).parts
        if "skillset" in parts and "skills" in parts:
            grouped.add(Path(*parts[:parts.index("skills") + 1]).as_posix())
        elif len(parts) > 2 and parts[2] in {"meta", "data"}:
            grouped.add(Path(*parts[:3]).as_posix())
        else:
            grouped.add(path)
    return sorted(grouped)
