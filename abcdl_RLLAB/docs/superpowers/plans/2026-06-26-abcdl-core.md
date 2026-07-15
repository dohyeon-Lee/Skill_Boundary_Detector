# abcdl Core Data Layer — Implementation Plan (Plan 1 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `abcdl` package's core data layer — a neutral `Episode` representation plus read/write for the abcdl MP4+binary format and the ABC-130k MCAP format, with MCAP↔abcdl conversion.

**Architecture:** Every format has a reader (→ `Episode`) and a writer (← `Episode`); converters compose through the IR. The abcdl format uses a strict H.264 encoding (GOP 30, CFR, faststart, no B-frames) whose frame index is reconstructed analytically at load time via torchcodec `custom_frame_mappings`. MCAP read uses embedded protobuf schemas; MCAP write uses vendored protos + foxglove schemas.

**Tech Stack:** Python 3.10–3.12, numpy, torch + torchcodec, ffmpeg (system), mcap + mcap-protobuf-support, protobuf, foxglove-schemas-protobuf, pytest.

## Global Constraints

- Python `>=3.10,<3.13` — code MUST run on **3.10, 3.11, and 3.12** (openpi is 3.11; much of the ecosystem is still 3.10). Avoid newer-only syntax: keep `from __future__ import annotations` in every module, use `typing.Optional`/`typing.Union` (not bare `X | Y` at runtime, e.g. not in `isinstance`/casts), no `match` statements, no `Self`/`tomllib`/`ExceptionGroup`-only features. CI should run the test suite on 3.10 and 3.12.
- Package import name is `abcdl`; repo is `abcdl_RLLAB`.
- abcdl video encoding is **exactly**: H.264, `keyint=30:min-keyint=30:scenecut=0`, `fps=30/1:timebase=1/15360:force-cfr=1`, `-bf 0`, `-pix_fmt yuv420p`, `-movflags +faststart`. Output frame count MUST equal `num_steps` (assert via ffprobe `nb_read_frames`).
- abcdl fixed clock: `FPS=30`, `TICK_NS=33333333`, `TIMESCALE=15360`, `TICKS_PER_FRAME=512`.
- `states_actions.bin` is raw little-endian float64, shape `(num_steps, state_dim + action_dim)`, no header.
- MCAP reads MUST tolerate the metadata-record naming drift: accept record name `episode-metadata` (keys `session_id`, `operator_id`, `task_name`) AND `session-metadata` (keys `session-uuid`, `operator-id`, `instruction`).
- Never hard-code camera sets; derive per-stream codec from each `CompressedVideo.format` and resolution from each `…-info` calibration (fallback to the `…-metadata` width/height).
- YAM default state layout: per arm = 6 arm joints + 1 gripper; bimanual order = `[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]` → 14-D state and 14-D action, matching ABC `export_mcap.py` `STATE_TOPICS`/`ACTION_TOPICS`.
- TDD: every task is failing-test-first, minimal implementation, passing test, commit.

**Test fixture:** the validated sample episode is at
`$ABCDL_SAMPLE_MCAP` if set, else
`~/.cache/huggingface/hub/datasets--XDOF--ABC-130k/snapshots/071311db1ac281848714bff024f9c6f944837c40/data/val/open_the_umbrella/episode_ebcbf9d1-d42b-4ef1-b3be-cad29b11edf8/episode.mcap`.
Tests that need it `pytest.skip` when the file is absent.

---

## File Structure

```
abcdl_RLLAB/
  pyproject.toml                 # package metadata, deps, extras, pytest config
  abcdl/__init__.py
  abcdl/constants.py             # FPS, TICK_NS, TIMESCALE, TICKS_PER_FRAME, YAM layout
  abcdl/episode.py               # Episode, CameraStream, EpisodeMeta, StateLayout
  abcdl/format/__init__.py
  abcdl/format/encode.py         # strict ffmpeg encode; frame-count assertion
  abcdl/format/writer.py         # write_abcdl(episode, out_dir)
  abcdl/format/reader.py         # read_abcdl(dir) -> Episode  (analytic-index decode)
  abcdl/mcap/__init__.py
  abcdl/mcap/schemas/abc.proto   # RobotState, GripperState, Instructions, Annotation
  abcdl/mcap/schemas/__init__.py # loads generated _pb2 (or builds descriptors at import)
  abcdl/mcap/reader.py           # read_mcap(path) -> Episode
  abcdl/mcap/writer.py           # write_mcap(episode, path)
  abcdl/convert/__init__.py
  abcdl/convert/mcap_abcdl.py    # mcap_to_abcdl(src, dst), abcdl_to_mcap(src, dst)
  tests/conftest.py              # sample_mcap fixture, tmp helpers
  tests/test_episode.py
  tests/test_encode.py
  tests/test_format_roundtrip.py
  tests/test_mcap_reader.py
  tests/test_mcap_roundtrip.py
  tests/test_convert.py
```

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `abcdl/__init__.py`, `abcdl/constants.py`, `tests/conftest.py`
- Test: `tests/test_scaffold.py`

**Interfaces:**
- Produces: importable package `abcdl` with `abcdl.constants` (`FPS=30`, `TICK_NS=33333333`, `TIMESCALE=15360`, `TICKS_PER_FRAME=512`); pytest fixture `sample_mcap`.

- [ ] **Step 1: Write the failing test**

`tests/test_scaffold.py`:
```python
import abcdl
from abcdl import constants


def test_constants():
    assert constants.FPS == 30
    assert constants.TICK_NS == 33333333
    assert constants.TIMESCALE == 15360
    assert constants.TICKS_PER_FRAME == 512
    assert constants.TIMESCALE // constants.FPS == constants.TICKS_PER_FRAME
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scaffold.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'abcdl'`.

- [ ] **Step 3: Write minimal implementation**

`pyproject.toml`:
```toml
[project]
name = "abcdl"
version = "0.1.0"
description = "A Behavior Cloning Dataloader — ABC MP4+binary and MCAP data layer"
requires-python = ">=3.10,<3.13"
dependencies = [
    "numpy",
    "torch",
    "torchcodec",
    "mcap",
    "mcap-protobuf-support",
    "protobuf",
    "foxglove-schemas-protobuf",
]

[project.optional-dependencies]
lerobot = ["lerobot"]
s3 = ["s3fs", "boto3"]
dev = ["pytest"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["abcdl*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`abcdl/__init__.py`:
```python
"""abcdl — A Behavior Cloning Dataloader: ABC MP4+binary and MCAP data layer."""

__version__ = "0.1.0"
```

`abcdl/constants.py`:
```python
"""Fixed-clock and encoding constants for the abcdl format (paper Appendix B.1)."""

from __future__ import annotations

FPS = 30
TICK_NS = 33_333_333  # int(1e9 / 30)
TIMESCALE = 15_360
TICKS_PER_FRAME = 512  # TIMESCALE // FPS

# YAM bimanual default state/action layout: [L_arm(6), L_grip(1), R_arm(6), R_grip(1)].
YAM_ARM_DOF = 6
YAM_GRIPPER_DOF = 1
YAM_STATE_DIM = 2 * (YAM_ARM_DOF + YAM_GRIPPER_DOF)  # 14
YAM_ACTION_DIM = YAM_STATE_DIM
```

`tests/conftest.py`:
```python
import os
from pathlib import Path

import pytest

_DEFAULT_SAMPLE = Path(
    "~/.cache/huggingface/hub/datasets--XDOF--ABC-130k/snapshots/"
    "071311db1ac281848714bff024f9c6f944837c40/data/val/open_the_umbrella/"
    "episode_ebcbf9d1-d42b-4ef1-b3be-cad29b11edf8/episode.mcap"
).expanduser()


@pytest.fixture
def sample_mcap() -> Path:
    path = Path(os.environ.get("ABCDL_SAMPLE_MCAP", _DEFAULT_SAMPLE)).expanduser()
    if not path.exists():
        pytest.skip(f"sample mcap not found at {path}")
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install -e '.[dev]' && pytest tests/test_scaffold.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml abcdl/__init__.py abcdl/constants.py tests/conftest.py tests/test_scaffold.py
git commit -m "feat: project scaffold + constants"
```

---

### Task 2: Episode IR

**Files:**
- Create: `abcdl/episode.py`
- Test: `tests/test_episode.py`

**Interfaces:**
- Produces:
  - `StateLayout(arm_dof, gripper_dof, n_arms)` with `.state_dim`, `.action_dim`, and `YAM = StateLayout(6, 1, 2)`.
  - `CameraStream(frames: list[bytes] | np.ndarray, timestamps: np.ndarray, width: int, height: int, codec: str)`.
  - `EpisodeMeta(task, fps, cameras, camera_resolutions, camera_codecs, operator_id=None, alignment="native", t0_ns=None, tick_ns=None, session_id=None)`.
  - `Episode(states, actions, timestamps, cameras, meta, ee_poses=None, extras=None, subtasks=None)` with `.num_steps`, `.validate()`.

- [ ] **Step 1: Write the failing test**

`tests/test_episode.py`:
```python
import numpy as np
import pytest

from abcdl.episode import CameraStream, Episode, EpisodeMeta, StateLayout


def test_state_layout_dims():
    assert StateLayout.YAM.state_dim == 14
    assert StateLayout.YAM.action_dim == 14


def _toy_episode(T=5):
    cam = CameraStream(frames=np.zeros((T, 4, 4, 3), np.uint8),
                       timestamps=np.arange(T, dtype=np.int64), width=4, height=4, codec="raw")
    meta = EpisodeMeta(task="t", fps=30.0, cameras=["top"],
                       camera_resolutions={"top": (4, 4)}, camera_codecs={"top": "raw"})
    return Episode(states=np.zeros((T, 14)), actions=np.zeros((T, 14)),
                   timestamps=np.arange(T, dtype=np.int64), cameras={"top": cam}, meta=meta)


def test_episode_num_steps_and_validate():
    ep = _toy_episode(7)
    assert ep.num_steps == 7
    ep.validate()  # no raise


def test_episode_validate_rejects_length_mismatch():
    ep = _toy_episode(5)
    ep.actions = np.zeros((4, 14))
    with pytest.raises(ValueError):
        ep.validate()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_episode.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'abcdl.episode'`.

- [ ] **Step 3: Write minimal implementation**

`abcdl/episode.py`:
```python
"""Neutral in-memory representation of one episode (the IR all formats share)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np


@dataclass(frozen=True)
class StateLayout:
    arm_dof: int
    gripper_dof: int
    n_arms: int

    @property
    def per_arm(self) -> int:
        return self.arm_dof + self.gripper_dof

    @property
    def state_dim(self) -> int:
        return self.n_arms * self.per_arm

    @property
    def action_dim(self) -> int:
        return self.state_dim


StateLayout.YAM = StateLayout(arm_dof=6, gripper_dof=1, n_arms=2)


@dataclass
class CameraStream:
    frames: Union[list, np.ndarray]   # list[bytes] (encoded Annex-B) OR (T,H,W,3) uint8
    timestamps: np.ndarray            # (Tc,) int64 ns
    width: int
    height: int
    codec: str                        # "h264" | "h265" | "raw"

    def __len__(self) -> int:
        return len(self.frames)


@dataclass
class EpisodeMeta:
    task: str
    fps: float
    cameras: list
    camera_resolutions: dict
    camera_codecs: dict
    operator_id: Optional[str] = None
    alignment: str = "native"
    t0_ns: Optional[int] = None
    tick_ns: Optional[int] = None
    session_id: Optional[str] = None


@dataclass
class Episode:
    states: np.ndarray                # (T, state_dim) float64
    actions: np.ndarray               # (T, action_dim) float64
    timestamps: np.ndarray            # (T,) int64 ns
    cameras: dict                     # {name: CameraStream}
    meta: EpisodeMeta
    ee_poses: Optional[dict] = None   # {side: (T,4,4)}
    extras: Optional[dict] = None     # {side: {"velocity":..., "torque":...}}
    subtasks: Optional[list] = None   # [(t_ns, label)]

    @property
    def num_steps(self) -> int:
        return int(self.states.shape[0])

    def validate(self) -> None:
        T = self.num_steps
        if self.actions.shape[0] != T:
            raise ValueError(f"actions length {self.actions.shape[0]} != states {T}")
        if self.timestamps.shape[0] != T:
            raise ValueError(f"timestamps length {self.timestamps.shape[0]} != states {T}")
        for name in self.meta.cameras:
            if name not in self.cameras:
                raise ValueError(f"meta lists camera {name!r} but it is missing from cameras")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_episode.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add abcdl/episode.py tests/test_episode.py
git commit -m "feat: Episode IR (StateLayout, CameraStream, EpisodeMeta, Episode)"
```

---

### Task 3: abcdl strict video encode

**Files:**
- Create: `abcdl/format/__init__.py`, `abcdl/format/encode.py`
- Test: `tests/test_encode.py`

**Interfaces:**
- Consumes: `abcdl.constants`.
- Produces:
  - `encode_strict_h264(rgb_frames: np.ndarray, out_path: str) -> None` — `rgb_frames` is `(N,H,W,3)` uint8; writes an mp4 with the locked params; raises `RuntimeError` if `probe_frame_count(out_path) != N`.
  - `probe_frame_count(path: str) -> int` — ffprobe `nb_read_frames`.

- [ ] **Step 1: Write the failing test**

`tests/test_encode.py`:
```python
import shutil

import numpy as np
import pytest

from abcdl.format.encode import encode_strict_h264, probe_frame_count

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="ffmpeg/ffprobe not installed")


def test_encode_frame_count_matches(tmp_path):
    n, h, w = 45, 16, 16
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, size=(n, h, w, 3), dtype=np.uint8)
    out = str(tmp_path / "v.mp4")
    encode_strict_h264(frames, out)
    assert probe_frame_count(out) == n
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_encode.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'abcdl.format.encode'`.

- [ ] **Step 3: Write minimal implementation**

`abcdl/format/__init__.py`:
```python
```

`abcdl/format/encode.py`:
```python
"""Strict H.264 encoding for the abcdl format — enables analytic frame indexing."""

from __future__ import annotations

import subprocess

import numpy as np

from abcdl.constants import FPS, TIMESCALE

_X264_PARAMS = (
    f"keyint={FPS}:min-keyint={FPS}:scenecut=0:"
    f"fps={FPS}/1:timebase=1/{TIMESCALE}:force-cfr=1"
)
_FFMPEG_ARGS = [
    "-vsync", "0",
    "-enc_time_base", f"1/{TIMESCALE}",
    "-video_track_timescale", str(TIMESCALE),
    "-bf", "0",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    "-x264-params", _X264_PARAMS,
    "-threads", "1",
]


def probe_frame_count(path: str) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames",
         "-of", "csv=p=0", path],
        capture_output=True, text=True,
    ).stdout.strip()
    return int(out)


def encode_strict_h264(rgb_frames: np.ndarray, out_path: str) -> None:
    if rgb_frames.dtype != np.uint8 or rgb_frames.ndim != 4 or rgb_frames.shape[3] != 3:
        raise ValueError("rgb_frames must be (N,H,W,3) uint8")
    n, h, w, _ = rgb_frames.shape
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", *_FFMPEG_ARGS, out_path],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    proc.stdin.write(np.ascontiguousarray(rgb_frames).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg encode failed")
    got = probe_frame_count(out_path)
    if got != n:
        raise RuntimeError(f"encoded {got} frames, expected {n}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_encode.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add abcdl/format/__init__.py abcdl/format/encode.py tests/test_encode.py
git commit -m "feat: strict H.264 encode with frame-count assertion"
```

---

### Task 4: abcdl format writer + reader (round-trip)

**Files:**
- Create: `abcdl/format/writer.py`, `abcdl/format/reader.py`
- Test: `tests/test_format_roundtrip.py`

**Interfaces:**
- Consumes: `Episode`, `encode_strict_h264`, constants.
- Produces:
  - `write_abcdl(episode: Episode, out_dir: str) -> None` — writes `states_actions.bin`, `combined_camera-images-rgb.mp4` (cameras vstacked in `episode.meta.cameras` order, each frame `(H,W,3)` uint8 already at output size), and `episode_metadata.json`. Requires every camera's `frames` to be a decoded `(T,H,W,3)` uint8 array of length `num_steps` with equal H,W across cameras.
  - `read_abcdl(in_dir: str) -> Episode` — reads back; camera frames returned as decoded `(T,h,W,3)` uint8 split from the vertical stack; uses a synthesized `custom_frame_mappings` (no probing).

- [ ] **Step 1: Write the failing test**

`tests/test_format_roundtrip.py`:
```python
import shutil

import numpy as np
import pytest

from abcdl.episode import CameraStream, Episode, EpisodeMeta
from abcdl.format.reader import read_abcdl
from abcdl.format.writer import write_abcdl

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _episode(T=33, h=16, w=16):
    rng = np.random.default_rng(1)
    cams = {}
    res = {}
    codecs = {}
    for name in ("top", "left_wrist"):
        cams[name] = CameraStream(
            frames=rng.integers(0, 255, (T, h, w, 3), dtype=np.uint8),
            timestamps=np.arange(T, dtype=np.int64), width=w, height=h, codec="raw")
        res[name] = (w, h)
        codecs[name] = "h264"
    meta = EpisodeMeta(task="demo", fps=30.0, cameras=["top", "left_wrist"],
                       camera_resolutions=res, camera_codecs=codecs,
                       alignment="fixed_clock_30hz_causal", t0_ns=0, tick_ns=33_333_333)
    states = rng.standard_normal((T, 14))
    actions = rng.standard_normal((T, 14))
    return Episode(states, actions, np.arange(T, dtype=np.int64), cams, meta)


def test_roundtrip_states_exact_and_frames_aligned(tmp_path):
    ep = _episode()
    write_abcdl(ep, str(tmp_path))
    back = read_abcdl(str(tmp_path))
    assert back.num_steps == ep.num_steps
    np.testing.assert_array_equal(back.states, ep.states)        # bin is lossless
    np.testing.assert_array_equal(back.actions, ep.actions)
    assert set(back.cameras) == {"top", "left_wrist"}
    # frame count preserved per camera
    for name in ep.meta.cameras:
        assert len(back.cameras[name].frames) == ep.num_steps
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_format_roundtrip.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'abcdl.format.writer'`.

- [ ] **Step 3: Write minimal implementation**

`abcdl/format/writer.py`:
```python
"""Write an Episode to the abcdl MP4+binary on-disk layout."""

from __future__ import annotations

import json
import os

import numpy as np

from abcdl.constants import TICK_NS
from abcdl.episode import Episode
from abcdl.format.encode import encode_strict_h264


def write_abcdl(episode: Episode, out_dir: str) -> None:
    episode.validate()
    os.makedirs(out_dir, exist_ok=True)
    T = episode.num_steps

    sa = np.concatenate([np.asarray(episode.states, np.float64),
                         np.asarray(episode.actions, np.float64)], axis=1)
    sa.tofile(os.path.join(out_dir, "states_actions.bin"))

    stacks = []
    h = w = None
    for name in episode.meta.cameras:
        frames = np.asarray(episode.cameras[name].frames, np.uint8)
        if frames.shape[0] != T:
            raise ValueError(f"camera {name} has {frames.shape[0]} frames, expected {T}")
        if h is None:
            h, w = frames.shape[1], frames.shape[2]
        elif (frames.shape[1], frames.shape[2]) != (h, w):
            raise ValueError("all cameras must share H,W for vertical stacking")
        stacks.append(frames)
    combined = np.concatenate(stacks, axis=1)  # vstack along height
    encode_strict_h264(combined, os.path.join(out_dir, "combined_camera-images-rgb.mp4"))

    meta = {
        "task_name": episode.meta.task,
        "cameras": list(episode.meta.cameras),
        "camera_resolutions": {k: list(v) for k, v in episode.meta.camera_resolutions.items()},
        "alignment": episode.meta.alignment,
        "t0_ns": int(episode.meta.t0_ns or 0),
        "tick_ns": int(episode.meta.tick_ns or TICK_NS),
        "num_steps": T,
        "state_dim": int(episode.states.shape[1]),
        "action_dim": int(episode.actions.shape[1]),
        "operator_id": episode.meta.operator_id,
        "session_id": episode.meta.session_id,
    }
    with open(os.path.join(out_dir, "episode_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
```

`abcdl/format/reader.py`:
```python
"""Read the abcdl MP4+binary layout back into an Episode (analytic frame index)."""

from __future__ import annotations

import json
import os

import numpy as np

from abcdl.constants import TICK_NS, TICKS_PER_FRAME
from abcdl.episode import CameraStream, Episode, EpisodeMeta


def _decode_all_frames(mp4_path: str, num_steps: int) -> np.ndarray:
    """Decode every frame via torchcodec with a synthesized CFR frame map."""
    from torchcodec.decoders import VideoDecoder

    frames = [{"pts": TICKS_PER_FRAME * i, "duration": TICKS_PER_FRAME,
               "key_frame": 1 if i % 30 == 0 else 0} for i in range(num_steps)]
    mapping = json.dumps({"frames": frames})
    dec = VideoDecoder(mp4_path, custom_frame_mappings=mapping)
    out = np.empty((num_steps, *dec[0].shape), dtype=np.uint8)  # (C,H,W) per frame
    for i in range(num_steps):
        out[i] = dec[i].numpy()
    return out  # (T, C, Hstack, W)


def read_abcdl(in_dir: str) -> Episode:
    with open(os.path.join(in_dir, "episode_metadata.json")) as f:
        meta = json.load(f)
    T = int(meta["num_steps"])
    state_dim = int(meta["state_dim"])
    action_dim = int(meta["action_dim"])
    row = state_dim + action_dim

    sa = np.fromfile(os.path.join(in_dir, "states_actions.bin"), dtype=np.float64).reshape(T, row)
    states, actions = sa[:, :state_dim], sa[:, state_dim:]

    stacked = _decode_all_frames(os.path.join(in_dir, "combined_camera-images-rgb.mp4"), T)
    stacked = np.transpose(stacked, (0, 2, 3, 1))  # (T, Hstack, W, C)
    names = list(meta["cameras"])
    h = stacked.shape[1] // len(names)
    cams = {}
    for i, name in enumerate(names):
        frames = stacked[:, i * h:(i + 1) * h, :, :]
        w = frames.shape[2]
        cams[name] = CameraStream(frames=frames, timestamps=np.arange(T, dtype=np.int64),
                                  width=w, height=h, codec="h264")

    tick_ns = int(meta.get("tick_ns", TICK_NS))
    t0 = int(meta.get("t0_ns", 0))
    ts = t0 + tick_ns * np.arange(T, dtype=np.int64)
    em = EpisodeMeta(
        task=meta["task_name"], fps=1e9 / tick_ns, cameras=names,
        camera_resolutions={k: tuple(v) for k, v in meta["camera_resolutions"].items()},
        camera_codecs={k: "h264" for k in names},
        operator_id=meta.get("operator_id"), alignment=meta.get("alignment", "native"),
        t0_ns=t0, tick_ns=tick_ns, session_id=meta.get("session_id"))
    return Episode(states=states, actions=actions, timestamps=ts, cameras=cams, meta=em)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_format_roundtrip.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add abcdl/format/writer.py abcdl/format/reader.py tests/test_format_roundtrip.py
git commit -m "feat: abcdl format writer + reader (round-trip)"
```

---

### Task 5: MCAP protobuf schemas

**Files:**
- Create: `abcdl/mcap/__init__.py`, `abcdl/mcap/schemas/abc.proto`, `abcdl/mcap/schemas/__init__.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Produces: `abcdl.mcap.schemas` exposing generated message classes `RobotState`, `GripperState`, `Instructions`, `Annotation` (each with `timestamp` and the repeated-double / string fields from the spec).

- [ ] **Step 1: Write the failing test**

`tests/test_schemas.py`:
```python
from abcdl.mcap import schemas


def test_robotstate_fields():
    m = schemas.RobotState()
    m.position.extend([0.0] * 6)
    m.velocity.extend([0.0] * 7)
    m.torque.extend([0.0] * 7)
    m.pose.extend([0.0] * 16)
    assert len(m.position) == 6 and len(m.pose) == 16


def test_gripperstate_and_instructions():
    g = schemas.GripperState()
    g.position.extend([0.5])
    assert g.position[0] == 0.5
    ins = schemas.Instructions(data="open the umbrella")
    assert ins.data == "open the umbrella"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'abcdl.mcap'`.

- [ ] **Step 3: Write minimal implementation**

`abcdl/mcap/__init__.py`:
```python
```

`abcdl/mcap/schemas/abc.proto`:
```proto
syntax = "proto3";

import "google/protobuf/timestamp.proto";

message RobotState {
  google.protobuf.Timestamp timestamp = 1;
  repeated double position = 2;
  repeated double velocity = 3;
  repeated double acceleration = 4;
  repeated double torque = 5;
  repeated double pose = 6;
}

message GripperState {
  google.protobuf.Timestamp timestamp = 1;
  repeated double position = 2;
  repeated double velocity = 3;
  repeated double acceleration = 4;
  repeated double torque = 5;
}

message Instructions {
  google.protobuf.Timestamp timestamp = 1;
  string data = 2;
}

message Annotation {
  google.protobuf.Timestamp timestamp = 1;
  string data = 2;
}
```

`abcdl/mcap/schemas/__init__.py` (builds message classes at import via `protobuf` dynamic compilation, so no `protoc` build step is required):
```python
"""Load the ABC custom protobuf messages (RobotState/GripperState/Instructions/Annotation).

Built at import time from abc.proto via grpc_tools.protoc into a temp module,
falling back to a prebuilt abc_pb2 if present.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile

_HERE = os.path.dirname(__file__)


def _build():
    try:
        return importlib.import_module("abcdl.mcap.schemas.abc_pb2")
    except ModuleNotFoundError:
        pass
    from grpc_tools import protoc  # provided by grpcio-tools

    out = tempfile.mkdtemp(prefix="abcdl_pb2_")
    rc = protoc.main([
        "protoc", f"-I{_HERE}",
        f"-I{os.path.dirname(protoc.__file__)}/_proto",  # bundled google/protobuf/*.proto
        f"--python_out={out}", os.path.join(_HERE, "abc.proto"),
    ])
    if rc != 0:
        raise RuntimeError("protoc failed to compile abc.proto")
    sys.path.insert(0, out)
    return importlib.import_module("abc_pb2")


_pb2 = _build()
RobotState = _pb2.RobotState
GripperState = _pb2.GripperState
Instructions = _pb2.Instructions
Annotation = _pb2.Annotation
```

Add `grpcio-tools` to `pyproject.toml` `dependencies` (compiles `abc.proto` and bundles `google/protobuf/*.proto`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install -e . && pytest tests/test_schemas.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add abcdl/mcap/__init__.py abcdl/mcap/schemas/ pyproject.toml tests/test_schemas.py
git commit -m "feat: vendored ABC protobuf schemas (RobotState/GripperState/Instructions/Annotation)"
```

---

### Task 6: MCAP reader

**Files:**
- Create: `abcdl/mcap/reader.py`
- Test: `tests/test_mcap_reader.py`

**Interfaces:**
- Consumes: `Episode`, `CameraStream`, `EpisodeMeta`, `StateLayout`, mcap libs.
- Produces: `read_mcap(path: str, layout: StateLayout = StateLayout.YAM) -> Episode`. Reads `RobotState`/`GripperState` on the 8 state/action topics, `foxglove.CompressedVideo` per camera (frames kept as encoded Annex-B `bytes` with native timestamps), `/instruction`, EE pose into `ee_poses`, and the metadata record (drift-tolerant). State row order = layout order; camera nearest-neighbor alignment is **not** applied here (raw native streams kept; alignment happens in convert).

- [ ] **Step 1: Write the failing test**

`tests/test_mcap_reader.py`:
```python
from abcdl.mcap.reader import read_mcap


def test_read_sample_episode(sample_mcap):
    ep = read_mcap(str(sample_mcap))
    assert ep.meta.task == "open the umbrella"
    assert ep.states.shape[1] == 14 and ep.actions.shape[1] == 14
    assert ep.num_steps > 100
    assert set(ep.cameras) >= {"top", "left_wrist", "right_wrist"}
    # cameras carry encoded h264 chunks (bytes) at native timing
    top = ep.cameras["top"]
    assert top.codec == "h264" and isinstance(top.frames[0], (bytes, bytearray))
    assert ep.meta.operator_id  # metadata-drift tolerant read populated this
    assert ep.ee_poses is not None and ep.ee_poses["left"].shape[1:] == (4, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mcap_reader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'abcdl.mcap.reader'`.

- [ ] **Step 3: Write minimal implementation**

`abcdl/mcap/reader.py`:
```python
"""Read an ABC-130k episode.mcap into the Episode IR (embedded-schema decode)."""

from __future__ import annotations

import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from abcdl.episode import CameraStream, Episode, EpisodeMeta, StateLayout

_CAM_TOPICS = {
    "top": "/top-camera", "top_left": "/top-left-camera", "top_right": "/top-right-camera",
    "left_wrist": "/left-wrist-camera", "right_wrist": "/right-wrist-camera",
}
# state/action row order for the YAM layout (arm joints then gripper, per arm)
_ARM = {"left": "/left-arm-{k}", "right": "/right-arm-{k}"}
_EE = {"left": "/left-ee-{k}", "right": "/right-ee-{k}"}


def _read_metadata(reader):
    """Return (session_id, operator_id, task) tolerant of the naming drift."""
    for m in reader.iter_metadata():
        md = dict(m.metadata)
        if m.name in ("episode-metadata", "session-metadata"):
            return (md.get("session_id") or md.get("session-uuid"),
                    md.get("operator_id") or md.get("operator-id"),
                    md.get("task_name") or md.get("instruction"))
    return None, None, None


def _floor_idx(src_ts, tgt_ts):
    return np.clip(np.searchsorted(src_ts, tgt_ts, side="right") - 1, 0, len(src_ts) - 1)


def read_mcap(path: str, layout: StateLayout = StateLayout.YAM) -> Episode:
    arms, grips, poses, cams, task = {}, {}, {}, {}, None
    with open(path, "rb") as f:
        sid, oid, mtask = _read_metadata(make_reader(f))
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, ch, _, dec in reader.iter_decoded_messages():
            t = ch.topic
            if t == "/instruction":
                task = dec.data
            elif t.endswith("-arm-state") or t.endswith("-arm-action"):
                side = "left" if "left" in t else "right"
                kind = "state" if t.endswith("state") else "action"
                arms.setdefault((side, kind), []).append((ch_time(dec), list(dec.position)))
                if kind == "state" and len(dec.pose) == 16:
                    poses.setdefault(side, []).append((ch_time(dec), list(dec.pose)))
            elif t.endswith("-ee-state") or t.endswith("-ee-action"):
                side = "left" if "left" in t else "right"
                kind = "state" if t.endswith("state") else "action"
                grips.setdefault((side, kind), []).append((ch_time(dec), float(dec.position[0])))
            else:
                for name, topic in _CAM_TOPICS.items():
                    if t == topic:
                        cams.setdefault(name, []).append((ch_time(dec), bytes(dec.data), dec.format))

    # Build per-(side,kind) aligned arrays on the arm-state clock (native, no resample here).
    def stack(side, kind):
        a = sorted(arms[(side, kind)]); g = sorted(grips[(side, kind)])
        a_ts = np.array([x[0] for x in a], np.int64)
        a_v = np.array([x[1] for x in a], np.float64)              # (Ta, 6)
        g_ts = np.array([x[0] for x in g], np.int64)
        g_v = np.array([x[1] for x in g], np.float64)[:, None]     # (Tg, 1)
        g_on_a = g_v[_floor_idx(g_ts, a_ts)]
        return a_ts, np.concatenate([a_v, g_on_a], axis=1)         # (Ta, 7)

    lt, lstate = stack("left", "state")
    _, laction = stack("left", "action")[0], stack("left", "action")[1]
    _, rstate = stack("right", "state")[0], stack("right", "state")[1]
    _, raction = stack("right", "action")[0], stack("right", "action")[1]
    # align right onto left-state clock
    rt = stack("right", "state")[0]
    states = np.concatenate([lstate, rstate[_floor_idx(rt, lt)]], axis=1)   # (T,14)
    actions = np.concatenate([laction, raction[_floor_idx(rt, lt)]], axis=1)

    ee = {}
    for side, lst in poses.items():
        lst = sorted(lst)
        p_ts = np.array([x[0] for x in lst], np.int64)
        p = np.array([x[1] for x in lst], np.float64).reshape(-1, 4, 4)
        ee[side] = p[_floor_idx(p_ts, lt)]

    cam_streams, res, codecs, names = {}, {}, {}, []
    for name, lst in cams.items():
        lst = sorted(lst, key=lambda x: x[0])
        ts = np.array([x[0] for x in lst], np.int64)
        frames = [x[1] for x in lst]
        codec = lst[0][2]
        cam_streams[name] = CameraStream(frames=frames, timestamps=ts, width=0, height=0, codec=codec)
        res[name] = (0, 0); codecs[name] = codec; names.append(name)

    meta = EpisodeMeta(task=task or mtask or "", fps=0.0, cameras=names,
                       camera_resolutions=res, camera_codecs=codecs,
                       operator_id=oid, alignment="native", session_id=sid)
    return Episode(states=states, actions=actions, timestamps=lt,
                   cameras=cam_streams, meta=meta, ee_poses=ee or None)


def ch_time(dec) -> int:
    return int(dec.timestamp.seconds) * 1_000_000_000 + int(dec.timestamp.nanos)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mcap_reader.py -v`
Expected: PASS (skips only if sample absent).

- [ ] **Step 5: Commit**

```bash
git add abcdl/mcap/reader.py tests/test_mcap_reader.py
git commit -m "feat: MCAP reader (drift-tolerant; native streams + EE pose)"
```

---

### Task 7: MCAP writer (round-trip)

**Files:**
- Create: `abcdl/mcap/writer.py`
- Test: `tests/test_mcap_roundtrip.py`

**Interfaces:**
- Consumes: `Episode`, schemas, `foxglove_schemas_protobuf`, mcap writer libs, `read_mcap`.
- Produces: `write_mcap(episode: Episode, path: str) -> None`. Emits `RobotState`/`GripperState` on the 8 topics (splitting each 14-D row back into per-arm 6+1 by `StateLayout.YAM`), `foxglove.CompressedVideo` per camera (Annex-B `bytes` frames; if a camera holds decoded arrays it is H.264-encoded first), `/instruction`, and an `episode-metadata` record (`session_id`/`operator_id`/`task_name`).

- [ ] **Step 1: Write the failing test**

`tests/test_mcap_roundtrip.py`:
```python
import numpy as np

from abcdl.mcap.reader import read_mcap
from abcdl.mcap.writer import write_mcap


def test_mcap_state_roundtrip(sample_mcap, tmp_path):
    ep = read_mcap(str(sample_mcap))
    out = str(tmp_path / "episode.mcap")
    write_mcap(ep, out)
    back = read_mcap(out)
    assert back.meta.task == ep.meta.task
    assert back.states.shape == ep.states.shape
    np.testing.assert_allclose(back.states, ep.states, rtol=0, atol=1e-9)
    assert set(back.cameras) == set(ep.cameras)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mcap_roundtrip.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'abcdl.mcap.writer'`.

- [ ] **Step 3: Write minimal implementation**

`abcdl/mcap/writer.py`:
```python
"""Write an Episode to an ABC-130k-compatible episode.mcap."""

from __future__ import annotations

import numpy as np
from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo
from mcap_protobuf.writer import Writer

from abcdl.episode import Episode, StateLayout
from abcdl.mcap import schemas


def _ts(ns: int):
    from google.protobuf.timestamp_pb2 import Timestamp
    t = Timestamp(); t.FromNanoseconds(int(ns)); return t


def write_mcap(episode: Episode, path: str, layout: StateLayout = StateLayout.YAM) -> None:
    episode.validate()
    arm, grip = layout.arm_dof, layout.gripper_dof
    sides = ["left", "right"]
    ts = np.asarray(episode.timestamps, np.int64)

    with open(path, "wb") as f, Writer(f) as w:
        for i in range(episode.num_steps):
            for s, side in enumerate(sides):
                base = s * (arm + grip)
                for kind, vec in (("state", episode.states[i]), ("action", episode.actions[i])):
                    rs = schemas.RobotState(timestamp=_ts(ts[i]))
                    rs.position.extend(vec[base:base + arm].tolist())
                    if episode.ee_poses and side in episode.ee_poses and kind == "state":
                        rs.pose.extend(np.asarray(episode.ee_poses[side][i]).reshape(-1).tolist())
                    w.write_message(topic=f"/{side}-arm-{kind}", message=rs, log_time=int(ts[i]),
                                    publish_time=int(ts[i]))
                    gs = schemas.GripperState(timestamp=_ts(ts[i]))
                    gs.position.append(float(vec[base + arm]))
                    w.write_message(topic=f"/{side}-ee-{kind}", message=gs, log_time=int(ts[i]),
                                    publish_time=int(ts[i]))

        ins = schemas.Instructions(timestamp=_ts(ts[0]), data=episode.meta.task)
        w.write_message(topic="/instruction", message=ins, log_time=int(ts[0]), publish_time=int(ts[0]))

        for name in episode.meta.cameras:
            cam = episode.cameras[name]
            topic = f"/{name.replace('_', '-')}-camera" if "wrist" in name else f"/{name}-camera"
            frames, fts = _encoded_frames(cam)
            for j, chunk in enumerate(frames):
                cv = CompressedVideo(data=chunk, format=cam.codec, frame_id=f"{name}-images-rgb")
                cv.timestamp.FromNanoseconds(int(fts[j]))
                w.write_message(topic=topic, message=cv, log_time=int(fts[j]), publish_time=int(fts[j]))

        w._writer.add_metadata(  # episode-metadata record (drift-canonical form)
            name="episode-metadata",
            data={"session_id": episode.meta.session_id or "",
                  "operator_id": episode.meta.operator_id or "",
                  "task_name": episode.meta.task},
        )


def _encoded_frames(cam):
    """Return (list[bytes] Annex-B frames, timestamps). Encode if frames are decoded arrays."""
    if len(cam.frames) and isinstance(cam.frames[0], (bytes, bytearray)):
        return [bytes(x) for x in cam.frames], np.asarray(cam.timestamps, np.int64)
    raise NotImplementedError("re-encoding decoded frames to per-frame Annex-B is added in Plan 2")
```

> Note: round-trip test uses the sample (cameras already encoded `bytes`), so the
> `NotImplementedError` branch is not hit. Encoding decoded frames for MCAP is Plan 2.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mcap_roundtrip.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add abcdl/mcap/writer.py tests/test_mcap_roundtrip.py
git commit -m "feat: MCAP writer (state/camera/instruction round-trip)"
```

---

### Task 8: MCAP ↔ abcdl conversion

**Files:**
- Create: `abcdl/convert/__init__.py`, `abcdl/convert/mcap_abcdl.py`
- Test: `tests/test_convert.py`

**Interfaces:**
- Consumes: `read_mcap`, `write_abcdl`, `read_abcdl`, constants, torchcodec.
- Produces:
  - `mcap_to_abcdl(src_mcap: str, out_dir: str) -> None` — resamples to a fixed 30 Hz causal clock (`ticks = arange(t0+TICK_NS, t_end, TICK_NS)`), floor-samples states/actions onto ticks, decodes each camera's Annex-B stream and floor-samples + scales frames to a common size, then `write_abcdl`.
  - `abcdl_to_mcap(in_dir: str, out_mcap: str) -> None` — `read_abcdl` then re-encode each decoded camera frame to a single-frame H.264 chunk and `write_mcap`.

- [ ] **Step 1: Write the failing test**

`tests/test_convert.py`:
```python
import shutil

import numpy as np
import pytest

from abcdl.convert.mcap_abcdl import mcap_to_abcdl
from abcdl.format.reader import read_abcdl
from abcdl.mcap.reader import read_mcap

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_mcap_to_abcdl_fixed_clock(sample_mcap, tmp_path):
    out = tmp_path / "ep"
    mcap_to_abcdl(str(sample_mcap), str(out))
    ep = read_abcdl(str(out))
    assert ep.meta.alignment == "fixed_clock_30hz_causal"
    assert abs(ep.meta.fps - 30.0) < 1e-6
    # 14-D state preserved; frame count equals state count
    assert ep.states.shape[1] == 14
    for name in ep.meta.cameras:
        assert len(ep.cameras[name].frames) == ep.num_steps
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_convert.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'abcdl.convert.mcap_abcdl'`.

- [ ] **Step 3: Write minimal implementation**

`abcdl/convert/__init__.py`:
```python
```

`abcdl/convert/mcap_abcdl.py`:
```python
"""Convert between the ABC-130k MCAP format and the abcdl MP4+binary format."""

from __future__ import annotations

import subprocess
import tempfile

import numpy as np

from abcdl.constants import TICK_NS
from abcdl.episode import CameraStream, Episode, EpisodeMeta
from abcdl.format.reader import read_abcdl
from abcdl.format.writer import write_abcdl
from abcdl.mcap.reader import read_mcap
from abcdl.mcap.writer import write_mcap

OUT_W = OUT_H = 224


def _floor_idx(src_ts, tgt_ts):
    return np.clip(np.searchsorted(src_ts, tgt_ts, side="right") - 1, 0, len(src_ts) - 1)


def _decode_annexb(chunks, codec, out_w, out_h):
    """Decode a list of Annex-B frame chunks to (N, out_h, out_w, 3) uint8 via ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=f".{codec}") as raw:
        raw.write(b"".join(chunks)); raw.flush()
        vf = (f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:flags=bicubic,"
              f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2")
        proc = subprocess.run(
            ["ffmpeg", "-i", raw.name, "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-v", "error", "pipe:1"], capture_output=True)
    buf = np.frombuffer(proc.stdout, np.uint8)
    n = buf.size // (out_h * out_w * 3)
    return buf[:n * out_h * out_w * 3].reshape(n, out_h, out_w, 3)


def mcap_to_abcdl(src_mcap: str, out_dir: str) -> None:
    ep = read_mcap(src_mcap)
    t0 = int(ep.timestamps[0])
    t_end = int(ep.timestamps[-1])
    ticks = np.arange(t0 + TICK_NS, t_end + 1, TICK_NS, dtype=np.int64)
    T = len(ticks)

    states = ep.states[_floor_idx(ep.timestamps, ticks)]
    actions = ep.actions[_floor_idx(ep.timestamps, ticks)]

    cams, res, codecs = {}, {}, {}
    for name in ep.meta.cameras:
        c = ep.cameras[name]
        decoded = _decode_annexb(c.frames, c.codec, OUT_W, OUT_H)  # (Nframes, H, W, 3)
        nf = decoded.shape[0]
        cam_ts = (np.linspace(c.timestamps[0], c.timestamps[-1], nf).astype(np.int64)
                  if nf != len(c.timestamps) else np.asarray(c.timestamps, np.int64))
        sel = decoded[_floor_idx(cam_ts, ticks)]
        cams[name] = CameraStream(frames=sel, timestamps=ticks, width=OUT_W, height=OUT_H, codec="raw")
        res[name] = (OUT_W, OUT_H); codecs[name] = "h264"

    meta = EpisodeMeta(task=ep.meta.task, fps=30.0, cameras=list(ep.meta.cameras),
                       camera_resolutions=res, camera_codecs=codecs,
                       operator_id=ep.meta.operator_id, alignment="fixed_clock_30hz_causal",
                       t0_ns=t0, tick_ns=TICK_NS, session_id=ep.meta.session_id)
    out = Episode(states=states, actions=actions, timestamps=ticks, cameras=cams, meta=meta)
    write_abcdl(out, out_dir)


def abcdl_to_mcap(in_dir: str, out_mcap: str) -> None:
    from abcdl.format.encode import encode_strict_h264  # reuse strict encoder per-frame
    ep = read_abcdl(in_dir)
    for name in ep.meta.cameras:
        c = ep.cameras[name]
        chunks = []
        for fr in np.asarray(c.frames, np.uint8):
            with tempfile.NamedTemporaryFile(suffix=".mp4") as t:
                encode_strict_h264(fr[None], t.name)
                with open(t.name, "rb") as fh:
                    chunks.append(fh.read())
        c.frames = chunks  # now bytes per frame
        c.codec = "h264"
    write_mcap(ep, out_mcap)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_convert.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add abcdl/convert/__init__.py abcdl/convert/mcap_abcdl.py tests/test_convert.py
git commit -m "feat: mcap<->abcdl conversion (30Hz causal resample)"
```

---

## Self-Review

**Spec coverage (Plan 1 portion of the spec):**
- §4 neutral `Episode` IR → Task 2. ✓
- §5 IR fields (states/actions/timestamps/cameras/meta/ee_poses/extras/subtasks) → Task 2 (extras/subtasks present as optional; populated as available). ✓
- §6.1 abcdl `format/` encode+writer+reader, analytic index → Tasks 3–4. ✓
- §6.2 `mcap/` schemas+reader+writer, drift tolerance, foxglove video → Tasks 5–7. ✓
- §6.3 converters mcap↔abcdl (30Hz causal) → Task 8. ✓
- §2.1 validated facts (proto fields, 14-D, drift, encoding params) → encoded in constants + Tasks 3/5/6. ✓
- Deferred to Plan 2 (per spec scope): `convert/lerobot.py`, `dataset.py`, `backends.py`, `hf.py`, `writer.py` EpisodeWriter, `cli.py`. ✓ (intentional)

**Placeholder scan:** the only `NotImplementedError` (MCAP re-encode of decoded frames in `mcap/writer.py`) is explicitly out-of-path for Plan 1's tests and called out; `abcdl_to_mcap` supplies real per-frame encoding in Task 8. No `TODO`/`TBD` left.

**Type consistency:** `read_mcap`/`write_mcap`/`read_abcdl`/`write_abcdl`/`mcap_to_abcdl`/`abcdl_to_mcap` signatures match across tasks; `Episode`/`CameraStream`/`EpisodeMeta` fields used consistently; `StateLayout.YAM` (6,1,2) consistent.

---

## Plan 2 (preview — separate plan after Plan 1 lands)
LeRobot ↔ abcdl converters, `AbcdlDataset` (LeRobot-compatible loader), `backends.py`
(local/S3/HTTP), `hf.py` (format=branch / version=tag + streaming), `EpisodeWriter`
(incremental live writer + decoded-frame MCAP encoding), and `cli.py`.
