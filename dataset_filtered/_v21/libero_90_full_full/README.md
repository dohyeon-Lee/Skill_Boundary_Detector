---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
- LeRobot
- libero
- franka
- libero_90_no_noops
configs:
- config_name: default
  data_files: data/*/*.parquet
---

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## Dataset Description



- **Homepage:** [More Information Needed]
- **Paper:** [More Information Needed]
- **License:** apache-2.0

## Dataset Structure

[meta/info.json](meta/info.json):
```json
{
    "codebase_version": "v2.1",
    "robot_type": "franka",
    "total_episodes": 3921,
    "total_frames": 569249,
    "total_tasks": 73,
    "total_videos": 7842,
    "total_chunks": 3,
    "chunks_size": 1000,
    "fps": 20,
    "splits": {
        "train": "0:3921"
    },
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": {
        "observation.images.image": {
            "dtype": "video",
            "shape": [
                256,
                256,
                3
            ],
            "names": [
                "height",
                "width",
                "rgb"
            ],
            "info": {
                "video.height": 256,
                "video.width": 256,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "video.fps": 20,
                "video.channels": 3,
                "has_audio": false
            }
        },
        "observation.images.wrist_image": {
            "dtype": "video",
            "shape": [
                256,
                256,
                3
            ],
            "names": [
                "height",
                "width",
                "rgb"
            ],
            "info": {
                "video.height": 256,
                "video.width": 256,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "video.fps": 20,
                "video.channels": 3,
                "has_audio": false
            }
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [
                8
            ],
            "names": {
                "motors": [
                    "x",
                    "y",
                    "z",
                    "roll",
                    "pitch",
                    "yaw",
                    "gripper",
                    "gripper"
                ]
            }
        },
        "observation.states.ee_state": {
            "dtype": "float32",
            "shape": [
                6
            ],
            "names": {
                "motors": [
                    "x",
                    "y",
                    "z",
                    "roll",
                    "pitch",
                    "yaw"
                ]
            }
        },
        "observation.states.joint_state": {
            "dtype": "float32",
            "shape": [
                7
            ],
            "names": {
                "motors": [
                    "joint_0",
                    "joint_1",
                    "joint_2",
                    "joint_3",
                    "joint_4",
                    "joint_5",
                    "joint_6"
                ]
            }
        },
        "observation.states.gripper_state": {
            "dtype": "float32",
            "shape": [
                2
            ],
            "names": {
                "motors": [
                    "gripper",
                    "gripper"
                ]
            }
        },
        "action": {
            "dtype": "float32",
            "shape": [
                7
            ],
            "names": {
                "motors": [
                    "x",
                    "y",
                    "z",
                    "roll",
                    "pitch",
                    "yaw",
                    "gripper"
                ]
            }
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": null
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "task_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        }
    }
}
```


## Citation

**BibTeX:**

```bibtex
[More Information Needed]
```