# DROID data preparation and skill training

This workflow converts **existing local raw DROID** episodes to LeRobot v3.
It does not download data, allocate GPUs, or change source files. The current
source is a public DROID success subset, not robot demonstrations collected by
this project. LIBERO, ABC and MimicGen data/configurations stay separate.

```text
droid/
└── configs/
    ├── global_config.yaml
    ├── src/global_config_loader.py
    ├── train_skills/
    │   ├── DP/             # dp_config.yaml, submit_train_dp.sh, src/train_dp.sbatch
    │   ├── build_data/     # SBD using the shared single-GPU episode workers
    │   ├── FSQ/            # fsq_config.yaml, submit_train_fsq.sh, src/train_fsq.sbatch
    │   └── src/            # DROID resolver, frozen-config submission, shared trainers
    └── generate_training_dataset/
        ├── download_dataset/
        │   ├── dataset_config.yaml
        │   ├── prepare_dataset.sh
        │   └── src/          # converter, geometry, existing-allocation launcher
        └── visualized_dataset/
            ├── visualized_dataset_config.yaml
            ├── visualize_training_dataset.sh
            └── src/visualize_training_dataset.py
```

## DP → SBD → FSQ

The initial controller contract follows the **released pi05_droid**:

| Item | Initial value |
| --- | --- |
| Control frequency | 15Hz |
| Action | Recorded joint velocity 7 + absolute gripper position 1 (0=open, 1=closed) |
| Proprioception | Measured joint position 7 + measured gripper state 1 |
| Future VLA action chunk | 15 steps (1 second) |
| Future VLA open-loop execution / replanning | 8 steps (~0.53 seconds) |
| Our state-only SBD DP history | 30 observations (2 seconds), transformer |
| Our SBD DP future prediction | 18 actions (1.2 seconds), small diffusion U-Net |
| FSQ | 333, normalized controller-action sequence, reconstruction only |

The 30-step DP history and FSQ architecture are our defaults, not OpenPI
training recipe claims. The released `pi05_droid` config has horizon 15;
OpenPI's separate approximate full-DROID retraining example has horizon 16
and joint-position targets. We follow the released velocity-action interface.
References: [model config](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py),
[robot control example](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/main.py),
[policy inputs](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/droid_policy.py).

Controller defaults and the eight `idle_filter` switches live in
`configs/global_config.yaml`. DP and FSQ choices live in their own YAMLs.
FSQ chunk length inherits global `vla_chunk_size` unless explicitly overridden.
Future VLA launchers should use `vla_chunk_size` and `vla_replan_steps`;
VLA training/deployment launchers are not implemented here yet.

DP demonstration-boundary evaluation is available in
[`configs/train_skills/DP/eval`](configs/train_skills/DP/eval/README.md).
It uses the shared LIBERO SBD workflow with power2, covariance-gated cosine,
GMM5, SG7/4 and replan8, plus DROID's 8D joint-velocity/gripper contract.
NMS is 23 frames, rounding chunk15 × 1.5 upward.
It supports existing-allocation submission, episode workers, resumed caches,
one global threshold and video/skill-bar/curve HTML reports.

```bash
cd lerobot/skillVLA/droid/configs/train_skills/DP
./submit_train_dp.sh --dry-run
./submit_train_dp.sh --75000       # explicit existing allocation, or omit for server-default submission

cd ../build_data
./submit_build_data.sh --dry-run
./submit_build_data.sh --75000

cd ../FSQ
./submit_train_fsq.sh --dry-run
./submit_train_fsq.sh --75000
```

Dry runs only print plans, including missing prerequisites; they never submit
jobs or create a training view. Actual submission fails before GPU scheduling
if conversion/skillset prerequisites are missing. Nothing chains automatically.
Training uses shared server profiles, srun/tmux or the explicit existing job,
normal Slurm Desk records and the server's memory guard. Each submission freezes
the component YAML, global switches and server profiles. DP resumes its own
`checkpoints/last` model/optimizer/scheduler/RNG, and FSQ resumes its latest
periodic checkpoint with the shared trainer's resume semantics. Changed
data/model/filter contracts cannot silently reuse the same run.

DP creates `dataset_DROID/droid_success_5k_joint_velocity` from the completed
conversion on its worker. Only numeric parquet action/state columns and their
statistics are rewritten to 8D; videos are referenced, not re-encoded. All
three cameras, measured EEF XYZ/pose, source action fields and calibration stay
available. The source conversion remains untouched. Missing completion markers
or a changed frame rate fail rather than reading an in-progress conversion or
silently resampling it. This view is also the input to SBD and FSQ.

DP alone can select `dp_action_space` in `configs/train_skills/DP/dp_config.yaml`:

| Value | DP action | DP proprioception |
| --- | --- | --- |
| `joint_velocity` (default) | Recorded joint velocity 7 + absolute gripper (8D) | Measured joint positions 7 + gripper (8D) |
| `delta_pose` | EEF translation delta 3 + rotation delta 3 + absolute gripper (7D) | Measured EEF XYZ + rotation vector + gripper (7D) |

The velocity option uses recorded velocity commands, not joint-position
differences. New velocity-run names use the requested `delta_joint` label
(for example `droid_success_5k_delta_joint_future_only_obs30_future18_...`),
while their stored training view remains named `joint_velocity`. Old frozen
requests without `dp_action_space` retain their original run name for resume.
EEF deltas use the recorded Cartesian position targets: the first
target is relative to the current measured EEF pose, and subsequent targets
are relative to the preceding target. Translation is in the robot base frame;
rotation is the SO(3) rotation vector of `R_target * inverse(R_previous)`, not
Euler-angle subtraction. Gripper values remain absolute (0 open, 1 closed).

`delta_pose` automatically creates a separate `droid_success_5k_delta_pose`
view and run name. This view retains absolute XYZ/rotvec targets for runtime
conversion; selected-corpus, horizon-specific delta-action statistics are
stored alongside the run contract. It reuses the original videos. Existing
joint-velocity checkpoints and the global FSQ/VLA action convention stay
unchanged. Switching representation starts a separate DP run; it cannot resume
an incompatible checkpoint. To analyze that DP later, explicitly select its
EEF view and corresponding checkpoint in the SBD configuration.

Outputs default to `outputs_DROID/DP/<run>` and `outputs_DROID/FSQ/<run>`.
SBD uses `dataset_DROID/FSQ_dataset/<dataset>/FSQ_inputs/<seg>/skillset`.
Its default Gaussian probes, dataset-global threshold, power2/SG7-order4,
replan8/NMS23/min-skill10 are initial DROID choices that can be edited in
`build_data_config.yaml`; action mode is `dataset`, never delta-joint.
Gripper snapping uses [0,1] with threshold 0.5. If selecting another DP
experiment or changing its horizon, update `dp_run_name` in build_data and
the corresponding `skillset_seg_name` in FSQ as with the other environments.

## Training-only idle filtering

`idle_filter` has only eight user switches: DP/FSQ default **false**, and
PT_VSA/PT_PRED/PT_TERM/FT_VSA/FT_PRED/FT_TERM default **true**.
The eight stages are independent. Missing switches in an old frozen config are
off. Non-booleans or misspelled stage names fail rather than silently enabling
or disabling a filter. A module YAML cannot override these global switches.

The common CPU implementation follows OpenPI's velocity-command *difference*
test: all seven differences <1e-3, runs >=7 removed from allowed starts,
remaining ranges >=16, trimming their last 10 starts. It does not test zero
speed and does not protect gripper-only motion automatically.
[Official rule](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/compute_droid_nonidle_ranges.py).

DP filters only DataLoader start indices after episode-tail selection. Action
chunks/history still read the original converted timeline. FSQ opt-in selects
training **skill occurrences** with eligible starts; it never truncates their
trajectories, removes them from validation, or modifies the source skillset
used by subsequent full-data encoding. Both opt-ins append `_idle` to the
training output name. With false, their original sampling path is unchanged.
SBD segmentation and encoding never apply a training filter, even when the
training switches are true.

Raw commands are read per original episode, and masks are mapped through
`source.frame_index`, so removing inactive collection frames during conversion
does not alter the official range calculation. Exact filtering needs the
original H5 commands to remain accessible. TERM frame filtering requires
explicit boundary/positive protection, including tagged final-frame positives;
the common adapter protects start/end neighbours and configured post-end
supervision. Full skill targets, reference observations and labels stay intact.
PT/FT stage APIs are ready, but their DROID launchers still need to be built;
grouped/jittered observation sampling must filter the final observation choices,
and unsupported grouped paths explicitly fail rather than pretend to filter.

## Dataset families and episode groups

Select the dataset and its episode group in `configs/global_config.yaml`:

```yaml
droid_dataset:
  name: success_5k
  group: without_deformable  # all to keep the complete selected dataset
```

The same block's `sources` lists each family's **raw_root**, **calibration_root**, optional
**annotations_path**, **converted_dataset** folder and **training_dataset** folder.
Conversion, the gallery, DP, SBD builds/eval and FSQ inherit this family. Add a
new entry with different output folder names after obtaining another raw dataset,
then change `name`. Source paths may be under your own `dataset_DROID` or another
accessible directory; they are not tied to the current data owner's directory.
A raw root must have the supported DROID download
manifest/trajectory/camera structure; arbitrary RLDS folders are not inputs.
Explicit paths in a local YAML or an old resolved snapshot still take priority.
Checkpoint steps and model/horizon choices remain explicit in their stage YAMLs;
dataset placeholders in DP references follow the selected family.

The shipped default remains `group: all`. The
`without_deformable` manifest is an allowlist of **55,865 original episode UUIDs**
from the public 75,144-episode language annotation file. It excludes any episode
whose three descriptions mention fabric/clothing, paper/cardboard/foil,
bags/pouches, cords/cables/elastics, sponge/plush/foam and the related broad terms
recorded in the manifest. Episodes without a matching annotated UUID are excluded.
This is language screening, not visually verified rigid-object ground truth.
Of the current 4,760 converted episodes, **3,116** match this group.

The built-in UUID manifest chooses the list; its internal SHA256 verifies its
unchanged content. These implementation details are not user settings in the
global YAML. Only `name` and `group` normally need editing. For a different annotation release
or exclusion rule, create a new versioned manifest and update its internal SHA;
do not silently replace a running job's list. Selected UUIDs are mapped to each
dataset's numeric IDs without renumbering or shortening trajectories. New data
can therefore reuse the group when its UUIDs occur in the screened release;
otherwise extend/rebuild the screen before treating unknown episodes as eligible.

Conversion preserves the selected family's full trajectories/cameras. The
episode group is a **read-time corpus selection** for learning and subsequent
builds, independent of `idle_filter` (which only filters training starts).
DP receives the selected episode IDs. FSQ removes excluded episodes before
train/validation splitting. SBD worker shards partition only selected episodes;
all workers share PCA fitted on the full group and one global threshold after
all selected curves finish. Evaluation also limits PCA to the selected group.
No original data or existing skillset is deleted or rewritten.

Group outputs carry `_without_deformable`; selected DP references and FSQ's
skillset references follow that naming. Manually specifying another checkpoint
still explicitly overrides the reference. A private copy of the UUID manifest
is bundled into each selected job snapshot. Resume contracts reject changed
selection. Old snapshots without the selector retain their all-episode behavior.

Shared VSA/Predictor/Terminator **PT/FT config resolvers** also use this episode
selection API and separate their output names. The DROID-specific VLA dataset
builders/model contracts/launchers are still a separate implementation step;
this selection support alone does not make those robot workflows runnable.

## Run locally on CPU

From `configs/generate_training_dataset/download_dataset`:

```bash
./prepare_dataset.sh --dry-run             # inspect selection and calibration
./prepare_dataset.sh                       # convert/resume this exact request
./prepare_dataset.sh --verify              # verify a completed dataset, decode every video
./prepare_dataset.sh --config /path/to/dataset_config.yaml
./prepare_dataset.sh --workers 8           # local CPU worker override
```

To use CPUs inside an **existing RUNNING managed Slurm GPU allocation**:

```bash
./prepare_dataset.sh --75000 --16
./prepare_dataset.sh --75000 --32
./prepare_dataset.sh --75000 --64          # only if allocation 75000 has >=64 CPUs
./prepare_dataset.sh --75000 --16 --dry-run
# Equivalent explicit options:
./prepare_dataset.sh --job-id 75000 --workers 16
```

The first numeric option selects the existing job; the second overrides YAML
episode workers. Without a worker override, the YAML value is used. CPU budget
is `workers * encoder_threads` and may not exceed the selected allocation's
CPU count. All allocated CPUs may be selected; no hidden reserve is imposed for
other training. Those trainers can temporarily slow down while sharing CPUs.

This uses the shared original-allocation worker, **not** a new GPU allocation,
additional Slurm step, or compute-node SSH. The owner, RUNNING state and
allocation identity are checked. Missing/expired allocations fail rather than
falling back to local execution or requesting another GPU. Only the conversion
child hides CUDA and restricts its CPU affinity; peers are not reconfigured.
Conversion settings are frozen per launch. Logs and the frozen config are
printed under `.cache/droid_conversions/<id>/`; the normal shared Slurm status
records track the process. A dry run performs local allocation/selection checks
and **does not enqueue a conversion**.

Set `raw_root`, `calibration_root` and optionally `annotations_path` in the YAML.
`raw_root` must contain the existing `download_manifest.jsonl`, metadata JSONs,
`trajectory.h5` and three single-view left-stereo MP4s per episode. This adapter
does not read RLDS shards. Calibration requires `intrinsics.json` and, by
default, `cam2base_extrinsic_superset.json`.

`max_episodes` applies after selection and calibration filtering; `episode_ids`
can restrict conversion to original DROID UUIDs. `missing_calibration: skip`
records exclusions in `conversion_manifest.json`; `error` stops instead.
Missing/truncated source files are errors, not calibration exclusions.
Official language annotations are optional: otherwise the raw `current_task`
text is retained, which can be generic. Alternative annotations and their
source are preserved per episode.

The destination defaults to `dataset_DROID/droid_success_5k`. Work is resumable
by episode, with a lock and completion markers under the destination. The raw
source is never locked or modified. Changed selection, action conventions or
calibration require a new `output_name`; changing CPU worker count does not.
Completed caches with damaged files cause an error rather than deletion.

## Actions and measured poses

`action_mode` controls only the primary `action` and `observation.state`.
The six raw action fields listed below are also saved, independently of that choice.

| `action_mode` | Primary action | Primary observation state |
| --- | --- | --- |
| `cartesian_velocity` (default) | Recorded normalized EEF velocity 6 + gripper position | Measured EEF XYZ/rotation vector 6 + gripper position |
| `cartesian_position` | Absolute EEF XYZ/rotation vector 6 + gripper position | Measured EEF XYZ/rotation vector 6 + gripper position |
| `joint_velocity` | Recorded joint velocity 7 + gripper position | Joint positions 7 + gripper position |
| `joint_position` | Joint position targets 7 + gripper position | Joint positions 7 + gripper position |

The DROID gripper command is **absolute: 0=open, 1=closed**, not MimicGen's
signed gripper command. Recorded velocity actions are preserved numerically;
they are not relabelled as metres/radians per second or converted to OSC deltas.

Every frame additionally contains:

- `observation.states.ee_state`: **measured** XYZ in robot-base metres + rotation vector in radians.
- `observation.states.ee_state_euler`: measured original XYZ + Euler `xyz` radians.
- `observation.states.joint_state`: seven measured joint positions.
- `observation.states.gripper_state`: measured gripper position.
- `source.action.cartesian_velocity`, `source.action.cartesian_position`,
  `source.action.joint_velocity`, `source.action.joint_position`,
  `source.action.gripper_position`, `source.action.gripper_velocity`:
  original numeric commands, including original Euler pose commands.
- `source.frame_index`, `source.control_timestamp_ms`, `source.skip_action`,
  `source.movement_enabled`: mapping back to the original control stream.

Thus EEF positions needed for later UV projection are retained even when the
primary action is joint based. No artificial two-finger state is synthesized.

The official OpenPI `pi05_droid` robot interface uses **joint velocity 7 +
gripper position 1**, with joint-position/gripper state. Choose
`action_mode: joint_velocity` for that data convention. This converter is not a
DROID deployment adapter: checkpoint-specific image naming, normalization,
padding and robot execution still need to match that checkpoint.

## Cameras, time alignment and UV projection

`exterior_1` and `exterior_2` are two external cameras, **not necessarily top
views**. `wrist` is mounted on the gripper. This workflow uses one RGB image
from the left sensor of each stereo camera, matching the downloaded MP4s:

| Raw camera | Training image feature |
| --- | --- |
| `exterior_1` | `observation.images.image` |
| `wrist` | `observation.images.wrist_image` |
| `exterior_2` | `observation.images.image2` |

Images are RGB. The default `image_transform: letterbox` preserves aspect ratio
and adds centred black padding, without cropping or flipping. For example,
1280x720 becomes 256x144 with 56 black rows above and below. Original and output
intrinsics and the pixel transform (including padding offsets) are retained.
The same resize geometry drives both RGB encoding and camera calibration.
Legacy `stretch` is available explicitly; old frozen configs that omitted the
key retain their old stretch convention and cannot be mixed into a letterbox
conversion. The selected transform is fingerprinted and recorded per episode.
This calibration source packs
`cameraMatrix` as **[fx, cx, fy, cy]**. Nonzero distortion and mismatched
calibration/video resolution are rejected; this converter does not silently
undistort or split already single-view videos.

Per-episode calibration lives in:

```text
meta/droid/episodes/000000.json  # original UUID, metadata, language, serials, GT/Pred quality provenance
meta/droid/episodes/000000.npz   # synchronized camera geometry and timestamps
```

NPZ keys for each camera are `{camera}.intrinsics`, `.intrinsics_original`,
`.pixel_transform`, `.camera_to_base`, `.recorded_pose_xyz_euler` and available
raw camera timestamps. `source_frame_index` matches the parquet exactly.
`camera_to_base` is an `(N,4,4)` transform, including the moving wrist camera.
The wrist's recorded pose is already in the robot base frame and must **not**
be multiplied by the EEF pose again. External cameras use calibrated GT/Pred
poses by default, while original recorded poses remain available.

For EEF projection use measured `ee_state[:3]` in the robot base frame:
`p_camera = inverse(camera_to_base) @ [XYZ,1]`, then project with the resized
intrinsics. `src/camera_geometry.py` provides this operation and returns depth
as well as UV. Points behind the camera or outside the image are not clamped.
Calibration presence/projection mathematics do not establish visual alignment;
the manifest explicitly keeps `image_projection_alignment_verified: false`.

DROID control runs at **15 Hz**, while these raw MP4 headers can say 60 fps.
Conversion maps raw video frame *i* to control step *i*, then encodes selected
frames at 15 fps. The default removes paused/skipped steps across all
modalities together. Dense output timestamps are `frame_index / 15`; original
timestamps are retained to identify removed intervals. By default a single
missing final video frame drops the corresponding final control sample; larger
gaps or different camera lengths fail. Each episode records these counts.

## Inspect converted data

From `configs/generate_training_dataset/visualized_dataset`:

```bash
./visualize_training_dataset.sh --dry-run
./visualize_training_dataset.sh --list-tasks
./visualize_training_dataset.sh
```

The gallery reuses the existing LIBERO visualizer with DROID-local settings and
outputs. This stage implements dataset conversion and inspection only; DROID
FSQ/DP/SkillVLA training, robot control and a UV-overlay viewer are not provided
by this folder yet.

Reference conventions:
[DROID platform](https://droid-dataset.github.io/),
[DROID transforms](https://github.com/droid-dataset/droid/blob/main/droid/misc/transformations.py),
[OpenPI DROID conversion](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/convert_droid_data_to_lerobot.py).
