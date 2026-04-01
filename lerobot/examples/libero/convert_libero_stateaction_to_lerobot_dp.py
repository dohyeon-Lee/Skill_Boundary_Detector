"""
Convert LIBERO RLDS dataset to LeRobot format (SBD lerobot) with state-based delta actions.
Compatible with SBD lerobot DiffusionPolicy (DP).

Action space (7D): [delta_eef_pos/0.05 (3), delta_eef_ori/0.5 (3), gripper (1)]
  - delta_eef = next_eef_state - curr_eef_state  (actual robot movement, dynamics-filtered)
  - Divided by OSC output scale (pos: 0.05m, ori: 0.5rad) → maps to ~[-1, 1]
  - gripper: original commanded gripper value (binary 0/1)

State space (8D): [eef_pos (3), eef_ori/axis-angle (3), left_finger (1), right_finger (1)]
  - Raw observation state from LIBERO, unchanged

Usage:
  PYTHONPATH=/data2/dohyeon/SBD/lerobot/src \
    /data2/dohyeon/SBD/.venv/bin/python \
    /data2/dohyeon/SBD/lerobot/examples/libero/convert_libero_stateaction_to_lerobot_dp.py \
    --data_dir /data2/dohyeon/libero_data \
    --dataset libero_90_openvla_processed

  PYTHONPATH=/data2/dohyeon/SBD/lerobot/src \
    /data2/dohyeon/SBD/.venv/bin/python \
    /data2/dohyeon/SBD/lerobot/examples/libero/convert_libero_stateaction_to_lerobot_dp.py \
    --data_dir /data2/dohyeon/libero_data \
    --dataset libero_10_no_noops
"""

import shutil
from dataclasses import dataclass

import numpy as np
import tensorflow_datasets as tfds
import tyro

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

# OSC output scale (robosuite osc_pose.json: output_max)
OSC_POS_SCALE = 0.05  # meters
OSC_ORI_SCALE = 0.5   # radians

FPS = 20  # LIBERO runs at 20 Hz

REPO_PREFIX = "dohyeon"

FEATURES = {
    "observation.images.image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.wrist_image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": {
            "motors": [
                "x", "y", "z",
                "axis_angle1", "axis_angle2", "axis_angle3",
                "left_finger", "right_finger",
            ]
        },
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {
            "motors": [
                "x", "y", "z",
                "axis_angle1", "axis_angle2", "axis_angle3",
                "gripper",
            ]
        },
    },
}


@dataclass
class Args:
    data_dir: str
    """Path to the RLDS dataset root (e.g. /data2/dohyeon/libero_data)"""
    dataset: str = "libero_90_openvla_processed"
    """RLDS dataset name (e.g. libero_90_no_noops, libero_10_no_noops)"""
    push_to_hub: bool = False


def main(args: Args) -> None:
    # repo name: e.g. dohyeon/libero_90_no_noops_stateaction_dp
    repo_name = f"{REPO_PREFIX}/{args.dataset}_stateaction_dp"

    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        fps=FPS,
        root=output_path,
        robot_type="franka",
        features=FEATURES,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    print(f"Processing {args.dataset} ...")
    raw_dataset = tfds.load(args.dataset, data_dir=args.data_dir, split="train")

    for ep_idx, episode in enumerate(raw_dataset):
            steps = list(episode["steps"].as_numpy_iterator())

            for i, step in enumerate(steps):
                next_step = steps[i + 1] if i + 1 < len(steps) else step

                # Compute actual EEF delta (what the robot physically realized)
                curr_eef = step["observation"]["state"][:6].astype(np.float32)
                next_eef = next_step["observation"]["state"][:6].astype(np.float32)
                delta_eef = next_eef - curr_eef

                # Normalize to ~[-1, 1] to match OSC controller input convention
                delta_eef[:3]  /= OSC_POS_SCALE
                delta_eef[3:6] /= OSC_ORI_SCALE

                # Gripper: use original commanded value (binary 0/1 from teleoperation)
                gripper = step["action"][6:7].astype(np.float32)
                action = np.concatenate([delta_eef, gripper])

                dataset.add_frame({
                    "observation.images.image":       step["observation"]["image"],
                    "observation.images.wrist_image": step["observation"]["wrist_image"],
                    "observation.state": step["observation"]["state"].astype(np.float32),
                    "action": action,
                    # 'task' is required by the LeRobot API but NOT fed to DP as a model input
                    "task": step["language_instruction"].decode(),
                })

            dataset.save_episode()

            if (ep_idx + 1) % 100 == 0:
                print(f"  Saved {ep_idx + 1} episodes")

    dataset.finalize()
    print(f"Done. Dataset saved to {output_path}")
    print(f"  Total episodes: {dataset.meta.total_episodes}")
    print(f"  Total frames:   {dataset.meta.total_frames}")

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "franka", "rlds", "stateaction"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
