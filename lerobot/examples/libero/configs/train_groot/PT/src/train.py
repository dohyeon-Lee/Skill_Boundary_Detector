#!/usr/bin/env python3
"""Run the pinned HF LeRobot GR00T policy/trainer in its isolated venv."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from groot_config import build_settings, load_config, trainer_args, validate_inputs


def prepare_environment(settings: dict) -> None:
    source = Path(settings["source"]).resolve()
    vendor = source / "src"
    if Path(sys.prefix).resolve() != Path(settings["python"]).parent.parent.resolve():
        raise RuntimeError(f"Use the GR00T virtual environment: {settings['python']} (got {sys.prefix})")
    sys.path[:0] = [str(vendor)]
    os.environ["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parent), str(vendor)])
    os.environ["LEROBOT_DATA_ROOT"] = settings["dataset_root"]
    os.environ["LEROBOT_VIDEO_BACKEND"] = "pyav"
    os.environ["WANDB_PROJECT"] = settings["wandb_project"]
    os.environ["WANDB_ENTITY"] = settings["wandb_entity"]
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import lerobot
    if not Path(lerobot.__file__).resolve().is_relative_to(vendor.resolve()):
        raise RuntimeError(f"Wrong LeRobot package imported: {lerobot.__file__}; expected {vendor}")


def check_data(settings: dict) -> None:
    """Read real frames and an action chunk without a model download or GPU."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    root = Path(settings["dataset_dir"])
    meta = LeRobotDatasetMetadata(repo_id=settings["dataset"], root=root)
    ds = LeRobotDataset(
        repo_id=settings["dataset"], root=root, episodes=[0], video_backend="pyav",
        delta_timestamps={"action": [i / meta.fps for i in range(settings["horizon"])]},
    )
    if settings["proprio_grounding"] == "episode_start_xyz":
        from grounding_adapter import ground_dataset
        ds = ground_dataset(ds, settings["proprio_grounding"])
    sample = ds[0]
    shapes = {
        "observation.images.image": (3, 256, 256), "observation.images.wrist_image": (3, 256, 256),
        "observation.state": (8,), "action": (settings["horizon"], 7),
    }
    for key, expected in shapes.items():
        value = sample[key]
        if tuple(value.shape) != expected:
            raise ValueError(f"{key}: expected {expected}, got {tuple(value.shape)}")
        print(f"{key}: {tuple(value.shape)} {value.dtype}", flush=True)
    print(f"Task: {sample.get('task')}", flush=True)
    print(f"Proprio grounding: {settings['proprio_grounding']}; first XYZ: {sample['observation.state'][:3].tolist()}", flush=True)
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.lerobot_types import TransitionKey as K
    from lerobot.policies.groot.configuration_groot import GrootConfig
    from lerobot.policies.groot.processor_groot import GrootN17PackInputsStep
    from processor_adapter import make_dataset_processors
    import torch
    cfg = GrootConfig(device="cpu", base_model_path=settings["base_checkpoint"], embodiment_tag="libero_sim",
                      action_decode_transform="none",
                      chunk_size=settings["horizon"], n_action_steps=settings["horizon"],
                      output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))})
    pre, post = make_dataset_processors(cfg, ds.meta.stats, ds.meta, horizon=settings["horizon"])
    pack = next(step for step in pre.steps if isinstance(step, GrootN17PackInputsStep))
    transition = {K.OBSERVATION: {key: sample[key].unsqueeze(0) for key in shapes if key != "action"},
                  K.ACTION: sample["action"].unsqueeze(0),
                  K.COMPLEMENTARY_DATA: {"task": [sample["task"]],
                                         "action_is_pad": sample["action_is_pad"].unsqueeze(0)}}
    with torch.no_grad():
        packed = pack(transition)
        decoded = post(packed[K.ACTION])
    mask = packed[K.COMPLEMENTARY_DATA]["action_mask"]
    if packed[K.ACTION].shape != (1, 40, 132) or decoded.shape != (1, settings["horizon"], 7):
        raise ValueError("Unexpected GR00T padded/decoded action shapes.")
    # Shapes alone did not catch a gripper sign inversion in the old bridge.
    torch.testing.assert_close(decoded[..., -1], sample["action"][None, ..., -1], rtol=0, atol=1e-6)
    print(f"Native pack/decode: action {tuple(packed[K.ACTION].shape)}, valid entries {int(mask.sum())}, decoded {tuple(decoded.shape)}", flush=True)
    print("Dataset sample check passed (episode 0, both cameras, PyAV).", flush=True)


def write_provenance(settings: dict) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    signature_keys = ("stage", "dataset_dir", "initial_checkpoint", "base_checkpoint", "horizon", "tuning", "checkpointing", "proprio_grounding", "lr_mode", "backend",
                      "batch_size", "micro_batch_size", "gpus", "accumulation_steps", "lr", "warmup_steps")
    signature = {key: settings[key] for key in signature_keys}
    signature["normalization_stats"] = settings.get("normalization_stats", "ft" if settings["stage"] == "ft" else "dataset")
    path = output / "sbd_launch.json"
    if path.exists():
        previous = json.loads(path.read_text())
        previous["signature"].setdefault(
            "normalization_stats", "ft" if settings["stage"] == "ft" else "dataset",
        )
        if previous["signature"] != signature:
            raise ValueError(f"{output} belongs to a different training recipe. Use another run.suffix.")
        return
    try:
        revision = subprocess.check_output(
            ["git", "-C", settings["source"], "rev-parse", "HEAD"], text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    versions = {}
    for package in ("torch", "torchvision", "transformers", "lerobot", "accelerate", "datasets", "av"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    payload = {"signature": signature, "settings": settings,
               "lerobot_revision": revision, "versions": versions,
               "step_units": "YAML/checkpoint folders/W&B: optimizer updates; native state/config: micro-batches"}
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true", help="20 steps, batch 1, workers 0, W&B disabled, separate output")
    parser.add_argument("--check-data", action="store_true", help="CPU-only real sample check; does not require model weights")
    parser.add_argument("--upstream-help", action="store_true", help="test official trainer imports/CLI, without CUDA or model loading")
    args = parser.parse_args()
    settings = build_settings(load_config(args.config), smoke=args.smoke)
    validate_inputs(settings, model=not (args.check_data or args.upstream_help))
    prepare_environment(settings)
    from lerobot.scripts import lerobot_train as native
    from training_adapter import install
    install(native, settings, write_provenance)
    if args.check_data:
        check_data(settings)
        return
    if args.upstream_help:
        sys.argv = ["lerobot-train", "--help"]
    else:
        native_args = trainer_args(settings)
        sys.argv = ["lerobot-train", *native_args]
        print(f"HF LeRobot GR00T trainer: {native.__file__}", flush=True)
        print(f"Global batch {settings['batch_size']} = micro {settings['micro_batch_size']} x "
              f"{settings['gpus']} GPU(s) x {settings['accumulation_steps']} accumulation", flush=True)
        print(f"Output: {settings['output_dir']}", flush=True)
        print(f"Checkpoint retention: latest {settings['keep_checkpoints']} + protected steps "
              f"{settings['protect_checkpoints']} (-1 means keep all)", flush=True)
    native.main()


if __name__ == "__main__":
    main()
