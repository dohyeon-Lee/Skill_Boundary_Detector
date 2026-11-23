# Coffee D0 → LIBERO-compatible LeRobot data

Run from `mimicgen/configs/generate_training_dataset/download_dataset/`:

```bash
./prepare_dataset.sh                  # download, then convert; CPU only
./prepare_dataset.sh --download-only
./prepare_dataset.sh --convert-only
./prepare_dataset.sh --dry-run
```

Edit the single `dataset_config.yaml`. No virtualenv activation is required:
the launcher uses the main `.venv` for parquet processing and invokes the
isolated `.venv_mimicgen` only to reconstruct simulator geometry. It does not
submit a Slurm allocation or start training. `max_episodes: all` selects all
500 demonstrations; a positive integer selects the first N in episode order.
This cap affects conversion; the complete raw snapshot is downloaded because
its parquet/video files contain multiple episodes.

Paths follow `mimicgen/configs/global_config.yaml` and the shared server profile:

* Raw snapshot: `dataset_mimicgen/_raw/coffee_d0/`
* Converted dataset: `dataset_mimicgen/coffee_d0_full_full/`

Downloads use an immutable Hugging Face revision, bounded HTTP ranges,
preserved `.part` files, and size/SHA-256 verification for LFS files. Repeating
the command reuses complete files and matching completed conversions. A changed
conversion request requires a different `output_name`; existing data is never
deleted. Partial conversion writes go into `.OUTPUT.building/` and become the
final directory only after validation. A raw-directory lock prevents concurrent
preparation commands. `--convert-only` requires a previously downloaded snapshot.
Hard-linked videos share disk blocks with the raw files; treat both as immutable.
Across filesystems the converter copies them instead.

## Feature contract

The reference LIBERO dataset supplies exact feature names, shapes, video codec,
resolution, and FPS. The adapter refuses incompatible reference settings.

| Output | Coffee source / transformation |
| --- | --- |
| `observation.images.image` | `observation.images.agentview`, original AV1 pixels |
| `observation.images.wrist_image` | `observation.images.wrist`, original AV1 pixels |
| `observation.state` (8) | EEF xyz + xyzw quaternion converted to axis-angle **radians** + two finger qpos |
| `observation.states.ee_state` (6) | EEF xyz + axis-angle radians |
| `observation.states.joint_state` (7) | Seven Panda joint angles |
| `observation.states.gripper_state` (2) | First two actual gripper qpos; four padding values must be zero |
| `action` (7) | Original normalized OSC delta command, `-1=open / +1=close` |

The original two-dimensional `observation.state` is a zero placeholder and is
discarded. Actions are neither integrated nor converted to joint actions.
Quaternion conversion matches LIBERO's actual processor, including negative
quaternion `w`: it does not replace rotations above pi with the equivalent
opposite-axis shorter rotation. This matters for pretrained DP/FSQ inputs.
In particular, do **not** apply the LIBERO hub 0/1 gripper remapping here.
Video file offsets are preserved while all metadata feature names are renamed.
Selected episodes and frame indexes are contiguous, with source IDs saved in
`conversion_manifest.json`. Numeric statistics (including quantiles) are
recomputed from converted values, both globally and per episode. Image
statistics are aggregated from the selected source episode statistics.

## Camera geometry and limits

`meta/mimicgen_camera.json` contains a fixed `agentview` world-to-pixel matrix
and the wrist camera's transform relative to the EEF position / hand rotation.
`meta/mimicgen_model.xml` and `eval_env_meta/demo.hdf5` retain the reconstructed
model and original environment settings. Building these requires no rendering
or GPU. A 32-frame check compares recorded EEF poses against forward kinematics
at the recorded Panda joint angles and rejects material frame/asset mismatches.

This geometry is reconstructed from the pinned environment, **not recorded
per-demonstration calibration**. Image/projection alignment is explicitly
marked unverified until a Coffee-specific overlay check is performed. Coffee
keeps its original video orientation; do not apply LIBERO's HDF5 rotation or
the LIBERO projection helper's horizontal flip without checking it.

The HF snapshot does not contain complete simulator states or per-demo model
XML. It supports observation/action preprocessing, but cannot provide exact
training-demonstration resets. The converted data has the tensor/storage
contract needed for shared LIBERO model checkpoints; Coffee-specific DP/FSQ
workflow YAMLs, spatial-target generation, FT, and rollout integration are
subsequent stages and are not launched by this command.

Source: [Coffee D0 HF dataset](https://huggingface.co/datasets/OliverHausdoerfer/demo_src_coffee_task_D0_robot_Panda_gripper_PandaGripper),
[official MimicGen](https://github.com/NVlabs/mimicgen).
The raw source README is preserved in the download folder.
