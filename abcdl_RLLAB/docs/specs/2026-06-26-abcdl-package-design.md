# abcdl_RLLAB — Design Spec

**Date:** 2026-06-26
**Repo:** `~/jellyho/abcdl_RLLAB` (new, standalone; import package name `abcdl`)
**Status:** Design — awaiting user review before implementation planning.

---

## 1. Goal

Build a reusable, pip-installable Python package (`abcdl`) that owns the **ABC dataset
data layer** — saving, loading, converting, and publishing bimanual manipulation
episodes — so it can be dropped into multiple training stacks (our YAM pipeline and
openpi, which today read RLDS / LeRobot datasets).

One sentence: *make ABC-style data (the `abcdl` MP4+binary format and the ABC-130k MCAP
release format) easy to write from YAM, convert to/from LeRobot, stream from
HuggingFace, and load efficiently behind a LeRobot-compatible interface.*

## 2. Background (grounded in the paper + real data)

The ABC stack has **two distinct data layers** — this distinction drove the whole design
and was a source of confusion worth stating plainly:

| Layer | Format | Role |
|---|---|---|
| **ABC-130k release** | **MCAP** (`episode.mcap` [+ `annotation.mcap`]) | raw distribution: full-res H.264/H.265 camera streams, state/action at native timing, self-describing (embedded protobuf schemas), Foxglove/ROS2 interop |
| **abcdl** | **MP4 + binary** (per episode: `combined_camera-images-rgb.mp4` + `states_actions.bin` + `episode_metadata.json`) | training format + loader. Paper Appendix B.1: *"abcdl (A Behavior Cloning Dataloader)"*. Fixed-clock, downscaled, fast random-frame access |

`export_mcap.py` in the ABC repo converts **MCAP → abcdl** (resample to a fixed 30 Hz
causal clock, vertically stack cameras, strict re-encode). `abcdl` is **not** MCAP; it is
the MP4+binary dataloader format. ABC-130k *is* distributed as MCAP.

### 2.1 Validated facts (from `XDOF/ABC-130k` sample `open_the_umbrella/episode_…edf8/episode.mcap`)

- **Channels / schemas** (RealSense station, 3 cameras): `/{left,right}-arm-{state,action}`
  → `RobotState`; `/{left,right}-ee-{state,action}` → `GripperState`;
  `/top-camera`, `/{left,right}-wrist-camera` → `foxglove.CompressedVideo`; sibling
  `…-info` → `foxglove.CameraCalibration`; `/instruction` → `Instructions`.
- **Custom protobuf schemas are embedded** in the MCAP (proto3, reconstructable from the
  file's `FileDescriptorSet`):
  - `RobotState { Timestamp timestamp=1; repeated double position=2, velocity=3, acceleration=4, torque=5, pose=6 }`
    — observed: `position`=6, `velocity`=7 (6 joints + gripper), `torque`=7, `pose`=16 (EE 4×4 row-major).
  - `GripperState { Timestamp timestamp=1; repeated double position=2, velocity=3, acceleration=4, torque=5 }`
    — `position`=[aperture], 0=closed, 1=open.
  - `Instructions { Timestamp timestamp=1; string data=2 }` — task name.
  - `foxglove.CompressedVideo { …; bytes data; string format }` — `format`="h264", one Annex-B frame per message.
- **Metadata record drift:** the sample uses a record named **`episode-metadata`** with
  underscore keys (`session_id`, `operator_id`, `task_name`, `duration`, per-camera
  `*_width/_height/_type/_polling_fps/_auto_exposure`). The spec doc `YAM_DATA_FORMAT.md`
  and `export_mcap.py` instead expect `session-metadata` / `session-uuid`. **Both must be
  accepted** on read.
- **Two station types** (do not hard-code cameras): RealSense (mono `/top-camera`, all
  H.264, 640×480) vs ZED-X (stereo `/top-left-camera` + `/top-right-camera`, H.265 top +
  H.264 wrists, 1920×1200). Branch on each stream's `format` and `…-info` width/height.
- **abcdl efficient-loading encoding** (paper Fig. 17–19, `export_mcap.py`): H.264 with
  **GOP=30 constant keyframes, CFR, `+faststart`, no B-frames**, timebase **1/15360**,
  **512 ticks/frame** (30 fps). This lets the loader synthesize the frame index
  **analytically** (torchcodec `custom_frame_mappings`, `pts=512*k`, keyframe at
  `k % 30 == 0`) instead of scanning the file — ~70× less read per decode
  (9.75 MB → 0.14 MB). Loads from local FS **or** S3/HTTP by swapping the backend.

## 3. Scope

### In scope (v1)
- The standalone `abcdl` package: core format I/O, MCAP I/O, converters, native loader,
  HF/cloud integration, and a writer API.
- Support **both** formats (abcdl MP4+bin and MCAP) with **bidirectional** conversion.
- LeRobot ↔ abcdl conversion and a **LeRobotDataset-compatible** native loader.
- HuggingFace upload/download with **format = branch, version = tag**, plus S3/HTTP
  streaming.
- `EpisodeWriter` API suitable for live recording (so YAM can later call it).

### Out of scope (v1, explicitly deferred)
- Wiring `abcdl` into the i2rt_rllab YAM recorder (`dataset_writer.py`). v1 ships the
  writer API only; integration is a follow-up.
- Model/training code (DiT/VLA), inference, simulation.
- ABC annotation (`annotation.mcap`) authoring beyond read-through pass-through of
  subtask labels into the `Episode` IR (writing annotations can come later).

### Non-goals
- Byte-for-byte reproduction of ABC's internal encoder beyond what the published format
  requires for correctness and the analytic frame index.

## 4. Architecture

**Central idea — a neutral in-memory `Episode` IR.** Every reader produces an `Episode`;
every writer consumes one; converters compose through it. With N formats this means N
readers + N writers instead of N² direct converters, and the YAM writer targets a single
type.

```
                       ┌──────────────────────────┐
  MCAP  ──reader──▶    │                          │ ──writer──▶  MCAP
  abcdl ──reader──▶    │      Episode  (IR)        │ ──writer──▶  abcdl
  LeRobot ──reader──▶  │                          │ ──writer──▶  LeRobot
  YAM live ─EpisodeWriter─▶ (builds Episode incrementally) ──▶ abcdl / MCAP
                       └──────────────────────────┘
                                   │
                          AbcdlDataset (torch, LeRobot-compatible)
                                   │
                          openpi / ABC-DiT training
```

Data flows:
```
ABC-130k (HF, mcap) ─pull/stream─▶ mcap ─convert─▶ abcdl ─AbcdlDataset─▶ training
YAM record ─EpisodeWriter─▶ abcdl (+ mcap) ─hf.push(format=branch,version=tag)─▶ HF
LeRobot dataset ◀─convert─▶ abcdl
```

## 5. Data model — `Episode` IR (`abcdl/episode.py`)

```
Episode:
  states:      (T, state_dim)  float64        # per-arm: 6 joints + 1 gripper → 14 bimanual
  actions:     (T, action_dim) float64        # commanded, same layout
  timestamps:  (T,) int64 ns                  # absolute (mcap) or fixed-clock (abcdl)
  cameras:     {name: CameraStream}           # name e.g. "top", "left_wrist", "right_wrist"
  meta: EpisodeMeta
      task: str
      fps: float
      cameras: [name…]; camera_resolutions: {name:(w,h)}; camera_codecs: {name:"h264"|"h265"}
      operator_id: str|None
      alignment: str                          # e.g. "fixed_clock_30hz_causal" | "native"
      t0_ns: int|None; tick_ns: int|None
      session_id: str|None
  ee_poses:    {side: (T,4,4) float64}|None    # optional, preserved from RobotState.pose
  extras:      {side: {"velocity":…, "torque":…}}|None   # optional proprio, when present
  subtasks:    [(t_ns, label)]|None            # from annotation.mcap when available

CameraStream:
  frames: list[bytes] | ndarray               # encoded Annex-B chunks (mcap) OR decoded frames
  timestamps: (Tc,) int64 ns
  width, height: int; codec: "h264"|"h265"|"raw"
```

State/action column layout is configurable (a `StateLayout` describing the per-arm and
per-gripper slices) so the package is not hard-wired to YAM 14-D, but ships a YAM default.

## 6. Modules

```
abcdl_RLLAB/                      # repo
  pyproject.toml                  # package "abcdl", optional extras: [lerobot], [s3]
  README.md
  docs/specs/…                    # this spec
  abcdl/
    episode.py                    # Episode / CameraStream / EpisodeMeta / StateLayout (IR)
    format/                       # the abcdl MP4+binary format (namesake)
      encode.py                   # strict x264: GOP30 + CFR + faststart + no-Bframe; timebase 1/15360
      writer.py                   # Episode → episode dir (combined mp4 + states_actions.bin + metadata.json)
      reader.py                   # episode dir → Episode; analytic frame index via torchcodec custom_frame_mappings
    mcap/                         # ABC-130k raw MCAP format
      schemas/                    # vendored RobotState/GripperState/Instructions/Annotation .proto + generated _pb2
      reader.py                   # episode.mcap (+annotation.mcap) → Episode; embedded-schema decode; metadata-drift tolerant
      writer.py                   # Episode → episode.mcap (ABC-130k compatible); foxglove video/calibration
    convert/
      mcap_abcdl.py               # mcap_to_abcdl (port export_mcap: 30Hz causal resample, vstack, strict encode) + abcdl_to_mcap
      lerobot.py                  # lerobot_to_abcdl / abcdl_to_lerobot (via Episode); optional import of lerobot
    dataset.py                    # AbcdlDataset(torch.utils.data.Dataset), LeRobotDataset-compatible surface
    backends.py                   # file access: local FS / S3 / HTTP (so reader can stream)
    hf.py                         # push/pull/list_versions; format→branch, version→tag; dataset card on main
    writer.py                     # EpisodeWriter (incremental, live) → abcdl and/or mcap
    cli.py                        # `abcdl {convert|push|pull|inspect}`
  tests/
```

### 6.1 `format/` — the abcdl MP4+binary format
- **encode.py:** owns the exact ffmpeg parameters that make the analytic index valid
  (`keyint=30:min-keyint=30:scenecut=0`, `fps=30/1:timebase=1/15360:force-cfr=1`,
  `-bf 0`, `+faststart`, `-pix_fmt yuv420p`). One place, asserted via frame-count probe.
- **writer.py:** writes `states_actions.bin` (raw float64, `(T, state_dim+action_dim)`),
  per-camera mp4s vstacked into `combined_camera-images-rgb.mp4`, and
  `episode_metadata.json` (task, cameras, resolutions, alignment, t0_ns, tick_ns,
  num_steps). Validates `nb_read_frames == num_steps` for every mp4.
- **reader.py:** seeks `states_actions.bin` by byte offset for a window; decodes
  `combined…mp4` frames by index with a **synthesized** `custom_frame_mappings`
  (no per-file probe), splits the vertical stack back into named cameras.

### 6.2 `mcap/` — ABC-130k raw format
- **schemas/:** vendor the three custom protos (reconstructed from the embedded
  `FileDescriptorSet`, validated against the sample) plus `Annotation`; depend on
  `foxglove-schemas-protobuf` for `CompressedVideo` / `CameraCalibration`.
- **reader.py:** decodes with `mcap` + `mcap-protobuf-support` (schemas embedded, so no
  local protos needed for reading); tolerant of the `episode-metadata`/`session-metadata`
  drift; branches on per-stream `format` and `…-info` resolution; reads `/instruction`
  and optional `annotation.mcap`.
- **writer.py:** emits an ABC-130k-compatible `episode.mcap` — `RobotState`/`GripperState`
  on the state/action topics, `foxglove.CompressedVideo` (Annex-B H.264) per camera with
  `…-info` calibration, `/instruction`, and an `episode-metadata` record.

### 6.3 `dataset.py` — LeRobot-compatible loader
`AbcdlDataset(root, delta_timestamps=…, chunk=…, camera_keys=…, …)` is a
`torch.utils.data.Dataset` whose surface mirrors what openpi reads off `LeRobotDataset`:
- `__getitem__(i)` → dict: `observation.images.<cam>` (CHW float), `observation.state`
  (S,), `action` (chunk, A), `task` (str), plus timestamp/episode-index fields.
- attributes: `.meta` (with `.features`, `.fps`, `.robot_type`, `.camera_keys`, `.stats`/
  norm-stats), `.num_episodes`, `.num_frames`.
- Internally uses `format/reader.py` for the efficient decode; a `MixtureDataset` helper
  supports hours-weighted real/sim mixing (parity with ABC training).
Compatibility is **duck-typed** (no LeRobot subclassing) to avoid a hard LeRobot dep at
load time; a thin `as_lerobot_compatible()` shim documents the exact contract.

### 6.4 `hf.py` + `backends.py` — HuggingFace & cloud
- **Layout:** one repo per dataset; **branch per format**, **tag per version** (`v1`, `v2`,
  …); `main` carries only the dataset card + a manifest of available formats/versions.
  Branch names: **`mcap`** = full-resolution source recordings (archival; only this can be
  re-processed); **`abcdl_<size>`** = the downscaled training cache at that square
  resolution (e.g. `abcdl_224`, `abcdl_256`) — multiple resolutions coexist as separate
  branches in one repo. Use `abcdl.convert.mcap_abcdl.abcdl_format_name(size)` for the name
  and `mcap_to_abcdl(src, dst, size=...)` to produce a given resolution. mcap is ~7× larger
  than `abcdl_224` (full-res vs 224 downscale) but is the only re-derivable source.
- **API:**
  - `push(repo_id, root, format, version, message=…)` → commit to branch `format`, tag `version`.
  - `pull(repo_id, format, version="latest", dest=…, stream=False)` → download or stream
    via `backends` (HF resolve URL / S3 / local).
  - `list_versions(repo_id, format)` → tags on that branch.
- **backends.py:** uniform open(path|s3://|https://) so `format/reader.py` streams frames
  without a full download (parity with the paper's "swap the backend" claim).

### 6.5 `writer.py` — EpisodeWriter (for YAM later)
```
w = EpisodeWriter(out_dir, formats=["abcdl","mcap"], fps=30,
                  cameras=[("top",(640,480),"h264"), ("left_wrist",…), ("right_wrist",…)],
                  state_layout=YAM_BIMANUAL)
w.add_frame(t_ns, state, action, images={"top":…, "left_wrist":…, "right_wrist":…})
…
w.save(task="open the umbrella", operator_id=…)   # builds Episode, dispatches to selected writers
```
Accepts raw frames (encodes) or pre-encoded H.264 chunks (pass-through for MCAP). v1 is
standalone-testable with synthetic frames; YAM integration is a later step.

### 6.6 `cli.py`
`abcdl inspect <mcap|dir>` (channels/schemas/shapes), `abcdl convert <src> <dst>
--from mcap --to abcdl`, `abcdl push/pull …`.

## 7. Dependencies
- **Python `>=3.10,<3.13`** — must run on 3.10/3.11/3.12 (openpi is 3.11; much of the
  ecosystem is still 3.10). No newer-only syntax; `from __future__ import annotations`
  everywhere; `typing.Optional`/`Union` at runtime.
- Required: `numpy`, `torch`, `torchcodec`, `mcap`, `mcap-protobuf-support`, `protobuf`,
  `foxglove-schemas-protobuf`, `huggingface_hub`; system `ffmpeg`.
- Optional extras: `[lerobot]` (LeRobot conversion/interop), `[s3]` (`s3fs`/`boto3`).

## 8. Testing strategy
- **Round-trip:** Episode → abcdl → Episode and Episode → mcap → Episode preserve
  states/actions (exact) and frame counts; images within codec tolerance.
- **Golden sample:** read the downloaded `open_the_umbrella` episode; assert channel set,
  schema fields, 14-D state/action, 30 Hz, task string.
- **Analytic index:** assert the encoder output yields `nb_read_frames == num_steps` and
  that synthesized `custom_frame_mappings` decode equals a naive full-probe decode.
- **Cross-format:** mcap → abcdl matches a reference produced by ABC's `export_mcap.py`
  (state/action arrays; frame counts).
- **Loader contract:** `AbcdlDataset[i]` keys/shapes match the documented LeRobot surface;
  a tiny synthetic dataset trains one step through an openpi-style data path (smoke).
- **HF (mocked):** `push`/`pull`/`list_versions` target the right branch/tag (no network
  in CI; integration test gated by `HF_TOKEN`).

## 9. Open questions / future
- YAM recorder integration (`dataset_writer.py` backend) — follow-up project.
- Writing `annotation.mcap` (subtask labels) — read supported in v1; authoring later.
- ZED-X (H.265 / stereo) write path — read/convert supported; confirm whether we ever
  need to *produce* ZED-X-style MCAP or only RealSense-style.
- Exact norm-stats parity with ABC (`norm_stats.json` schema) for checkpoint interop.
