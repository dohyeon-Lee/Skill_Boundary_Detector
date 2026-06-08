#!/usr/bin/env python3
# Inputs:
#   original HDF5 : {project_root}/libero_original_dataset/{suite}/*.hdf5
#   config        : ../training_dataset_config.yaml
# Reference model:
#   none
# Outputs:
#   LeRobot data  : {project_root}/libero_original_dataset/{output_name}
"""Convert original LIBERO HDF5 demos into the local LeRobot dataset format.

The converter keeps the current SBD conventions:
  - 20 Hz episodes
  - agentview -> observation.images.image
  - eye-in-hand -> observation.images.wrist_image
  - state = [ee_states(6), gripper_states(2)]
  - action = original LIBERO OSC delta action, with gripper kept in {-1, 1}
  - LeRobot codebase_version is created by the local LeRobotDataset writer

Examples:
  python convert_original_libero_to_lerobot.py --suite libero_10
  python convert_original_libero_to_lerobot.py --suite libero_90 --output-name libero_90_full_full

Smoke test:
  python convert_original_libero_to_lerobot.py --suite libero_10 \
    --output-name libero_10_smoke --max-tasks 1 --max-episodes-per-task 1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
GENERATE_DIR = THIS_DIR.parent
PROJECT_ROOT_FALLBACK = Path("/data2/dohyeon/SBD")
CONFIG_PATH = GENERATE_DIR / "training_dataset_config.yaml"

sys.path.insert(0, str(GENERATE_DIR / "src"))
try:
    from training_dataset_config import dataset_root_path, load_config, project_root
except ImportError:
    dataset_root_path = None
    load_config = None
    project_root = None


FPS = 20
REPO_PREFIX = "dohyeon"
DEFAULT_IMAGE_SIZE = 256
TASK_MAP_PATH = "tools/lerobot-libero/libero/libero/benchmark/libero_suite_task_map.py"

FEATURES = {
    "observation.images.image": {
        "dtype": "image",
        "shape": (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.wrist_image": {
        "dtype": "image",
        "shape": (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": {
            "motors": [
                "x",
                "y",
                "z",
                "axis_angle1",
                "axis_angle2",
                "axis_angle3",
                "left_finger",
                "right_finger",
            ]
        },
    },
    "observation.states.ee_state": {
        "dtype": "float32",
        "shape": (6,),
        "names": {"motors": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3"]},
    },
    "observation.states.joint_state": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": [f"joint_{i}" for i in range(7)]},
    },
    "observation.states.gripper_state": {
        "dtype": "float32",
        "shape": (2,),
        "names": {"motors": ["left_finger", "right_finger"]},
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {
            "motors": [
                "x",
                "y",
                "z",
                "axis_angle1",
                "axis_angle2",
                "axis_angle3",
                "gripper",
            ]
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--suite", choices=["libero_90", "libero_10"], required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Root containing original HDF5 folders. Default: {project_root}/libero_original_dataset",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root where converted LeRobot datasets are written. Default: {project_root}/libero_original_dataset",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output dataset folder name. Default: {suite}_full_full",
    )
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--image-writer-threads", type=int, default=10)
    parser.add_argument("--image-writer-processes", type=int, default=5)
    parser.add_argument(
        "--vcodec",
        default="libsvtav1",
        help="Video codec: libsvtav1 matches current datasets; auto may use GPU NVENC on GPU nodes.",
    )
    parser.add_argument(
        "--encoder-threads",
        type=int,
        default=None,
        help="Threads per video encoder. None lets LeRobot/FFmpeg choose.",
    )
    parser.add_argument(
        "--batch-encoding-size",
        type=int,
        default=1,
        help="Accumulate this many episodes before video encoding. 1 encodes each episode immediately.",
    )
    parser.add_argument(
        "--streaming-encoding",
        action="store_true",
        help="Encode video as frames are added. Useful for reducing temp image overhead.",
    )
    parser.add_argument("--encoder-queue-maxsize", type=int, default=30)
    parser.add_argument(
        "--schema",
        choices=["match-current", "rich"],
        default="match-current",
        help=(
            "match-current mirrors the feature schema of an existing libero_dataset/{suite} "
            "when present; rich always includes observation.states.* helper columns."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None, help="Debug limit.")
    parser.add_argument("--max-episodes-per-task", type=int, default=None, help="Debug limit.")
    return parser.parse_args()


def load_paths(config_path: Path) -> tuple[Path, Path]:
    if load_config is None or project_root is None or dataset_root_path is None:
        root = PROJECT_ROOT_FALLBACK
        return root, root / "libero_dataset"
    cfg = load_config(config_path)
    root = project_root(cfg)
    return root, dataset_root_path(cfg)


def import_lerobot(project_dir: Path) -> None:
    lerobot_src = project_dir / "lerobot" / "src"
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))


def load_official_task_names(project_dir: Path, suite: str) -> list[str] | None:
    task_map_path = project_dir / TASK_MAP_PATH
    if not task_map_path.exists():
        return None

    spec = importlib.util.spec_from_file_location("libero_suite_task_map", task_map_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.libero_task_map[suite])


def hdf5_task_stem(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]
    return stem


def language_from_file(path: Path, h5: h5py.File) -> str:
    data_attrs = h5["data"].attrs
    if "problem_info" in data_attrs:
        info = json.loads(data_attrs["problem_info"])
        if "language_instruction" in info:
            return str(info["language_instruction"])

    stem = hdf5_task_stem(path)
    for prefix in ("KITCHEN_", "LIVING_ROOM_", "STUDY_"):
        if stem.startswith(prefix):
            parts = stem.split("_", 2)
            if len(parts) == 3:
                return parts[2].replace("_", " ")
    return stem.replace("_", " ")


def ordered_demo_keys(data_group: h5py.Group) -> list[str]:
    def demo_index(name: str) -> int:
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return 10**9

    return sorted((k for k in data_group.keys() if k.startswith("demo_")), key=demo_index)


def resize_rgb(image: np.ndarray, size: int) -> np.ndarray:
    if image.shape[:2] == (size, size):
        return image
    pil = Image.fromarray(image)
    return np.asarray(pil.resize((size, size), resample=Image.BILINEAR), dtype=np.uint8)


def normalized_action(action: np.ndarray) -> np.ndarray:
    out = action.astype(np.float32, copy=True)
    grip = out[..., 6]
    unique = np.unique(np.round(grip, decimals=6))
    if set(unique.tolist()).issubset({0.0, 1.0}):
        out[..., 6] = np.where(grip > 0.5, 1.0, -1.0)
    elif set(unique.tolist()).issubset({-1.0, 1.0}):
        out[..., 6] = np.where(grip > 0.0, 1.0, -1.0)
    return out


def task_files(source_dir: Path, project_dir: Path, suite: str) -> list[Path]:
    files_by_stem = {hdf5_task_stem(path): path for path in source_dir.glob("*.hdf5")}
    files_by_stem.update({hdf5_task_stem(path): path for path in source_dir.glob("*.h5")})
    official = load_official_task_names(project_dir, suite)

    if official is None:
        return [files_by_stem[key] for key in sorted(files_by_stem)]

    missing = [name for name in official if name not in files_by_stem]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} official {suite} files under {source_dir}: {preview}")
    return [files_by_stem[name] for name in official]


def current_feature_specs(output_root: Path, suite: str) -> dict[str, Any] | None:
    info_path = output_root / suite / "meta" / "info.json"
    if not info_path.exists():
        return None
    with open(info_path) as f:
        info = json.load(f)
    features = info.get("features")
    if not isinstance(features, dict):
        return None
    allowed = set(FEATURES)
    return {key: value for key, value in features.items() if key in allowed}


def normalize_feature_spec(spec: dict[str, Any]) -> dict[str, Any]:
    out = dict(spec)
    if isinstance(out.get("shape"), list):
        out["shape"] = tuple(out["shape"])
    if isinstance(out.get("info"), dict):
        out["info"] = dict(out["info"])
    return out


def codec_metadata_name(vcodec: str) -> str:
    if vcodec == "libsvtav1":
        return "av1"
    if vcodec.startswith("h264"):
        return "h264"
    if vcodec.startswith("hevc"):
        return "hevc"
    return vcodec


def update_feature_shapes(
    image_size: int,
    base_features: dict[str, Any] | None = None,
    resolved_vcodec: str = "libsvtav1",
) -> dict[str, Any]:
    features = {key: dict(value) for key, value in FEATURES.items()}
    if base_features is not None:
        features = {key: normalize_feature_spec(value) for key, value in base_features.items()}
    for key in ("observation.images.image", "observation.images.wrist_image"):
        if key in features:
            features[key] = dict(features[key])
            features[key]["shape"] = (image_size, image_size, 3)
            if "info" in features[key]:
                features[key]["info"] = dict(features[key]["info"])
                features[key]["info"]["video.height"] = image_size
                features[key]["info"]["video.width"] = image_size
                features[key]["info"]["video.fps"] = FPS
                features[key]["info"]["video.codec"] = codec_metadata_name(resolved_vcodec)
    return features


def main() -> None:
    args = parse_args()
    project_dir, configured_output_root = load_paths(args.config)
    source_root = args.source_root or (project_dir / "libero_original_dataset")
    output_root = args.output_root or source_root
    schema_root = project_dir / "libero_dataset"
    source_dir = source_root / args.suite
    output_name = args.output_name or f"{args.suite}_full_full"
    output_dir = output_root / output_name

    if not source_dir.exists():
        raise FileNotFoundError(f"Original suite folder not found: {source_dir}")
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)

    import_lerobot(project_dir)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import resolve_vcodec

    files = task_files(source_dir, project_dir, args.suite)
    if args.max_tasks is not None:
        files = files[: args.max_tasks]
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under {source_dir}")

    print("Convert original LIBERO to LeRobot")
    print(f"  suite        : {args.suite}")
    print(f"  source       : {source_dir}")
    print(f"  output       : {output_dir}")
    print(f"  task files   : {len(files)}")
    print(f"  image size   : {args.image_size}")

    base_features = None
    if args.schema == "match-current":
        base_features = current_feature_specs(schema_root, args.suite)
        if base_features is not None:
            print(f"  schema       : match-current ({schema_root / args.suite})")
        else:
            print("  schema       : match-current requested, no current dataset found; using rich schema")
    else:
        print("  schema       : rich")
    resolved_vcodec = resolve_vcodec(args.vcodec)
    print(f"  codec        : requested={args.vcodec} resolved={resolved_vcodec}")
    features = update_feature_shapes(
        args.image_size,
        base_features=base_features,
        resolved_vcodec=resolved_vcodec,
    )

    dataset = LeRobotDataset.create(
        repo_id=f"{REPO_PREFIX}/{output_name}",
        fps=FPS,
        root=output_dir,
        robot_type="franka",
        features=features,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
        vcodec=resolved_vcodec,
        encoder_threads=args.encoder_threads,
        batch_encoding_size=args.batch_encoding_size,
        streaming_encoding=args.streaming_encoding,
        encoder_queue_maxsize=args.encoder_queue_maxsize,
    )

    total_episodes = 0
    total_frames = 0
    for task_idx, path in enumerate(files):
        with h5py.File(path, "r") as h5:
            task = language_from_file(path, h5)
            demos = ordered_demo_keys(h5["data"])
            if args.max_episodes_per_task is not None:
                demos = demos[: args.max_episodes_per_task]

            print(f"  task{task_idx:02d}: episodes={len(demos):3d} | {task}")
            for demo_key in demos:
                demo = h5["data"][demo_key]
                obs = demo["obs"]

                actions = normalized_action(demo["actions"][()])
                ee_states = obs["ee_states"][()].astype(np.float32)
                gripper_states = obs["gripper_states"][()].astype(np.float32)
                joint_states = obs["joint_states"][()].astype(np.float32)
                agentview = obs["agentview_rgb"]
                wrist = obs["eye_in_hand_rgb"]

                length = int(actions.shape[0])
                for t in range(length):
                    state = np.concatenate([ee_states[t], gripper_states[t]], axis=0).astype(np.float32)
                    frame = {
                        # Original LIBERO HDF5 stores agentview/eye-in-hand frames rotated 180
                        # (mujoco renders bottom-to-top). Rotate H+W to match the upright
                        # convention used by libero_dataset and eval's LiberoProcessorStep.
                        "observation.images.image": resize_rgb(agentview[t][::-1, ::-1], args.image_size),
                        "observation.images.wrist_image": resize_rgb(wrist[t][::-1, ::-1], args.image_size),
                        "observation.state": state,
                        "action": actions[t],
                        "task": task,
                    }
                    if "observation.states.ee_state" in features:
                        frame["observation.states.ee_state"] = ee_states[t]
                    if "observation.states.joint_state" in features:
                        frame["observation.states.joint_state"] = joint_states[t]
                    if "observation.states.gripper_state" in features:
                        frame["observation.states.gripper_state"] = gripper_states[t]
                    dataset.add_frame(frame)

                dataset.save_episode()
                total_episodes += 1
                total_frames += length
                if total_episodes % 100 == 0:
                    print(f"    saved episodes={total_episodes} frames={total_frames}")

    dataset.finalize()
    print("")
    print("DONE")
    print(f"  output         : {output_dir}")
    print(f"  total tasks    : {len(files)}")
    print(f"  total episodes : {total_episodes}")
    print(f"  total frames   : {total_frames}")


if __name__ == "__main__":
    main()
