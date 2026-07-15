# abcdl Consumers — Implementation Plan (Plan 2 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the consumer layer on top of the abcdl core (Plan 1): a LeRobot-compatible loader, LeRobot↔abcdl conversion, HF publishing, and an incremental EpisodeWriter (for live YAM recording). **No CLI** (per user) — HF use is LeRobot-style (`push_to_hub` / `AbcdlDataset(repo_id=...)` auto-load).

**Architecture:** Everything builds on the Plan 1 `Episode` IR and `format`/`mcap` readers/writers. The loader presents a LeRobotDataset-shaped surface so openpi can swap in. HF uses format=branch / version=tag. Backends abstract local/HTTP file access for streaming (no S3).

**Tech Stack:** Python 3.10–3.12, numpy, torch, torchcodec, ffmpeg, mcap stack (Plan 1), plus huggingface_hub, fsspec; optional extra: lerobot (convert + loader feature parity). **S3 is out of scope** (per user) — backends cover local + HTTP only.

## Global Constraints

- Python `>=3.10,<3.13` — must run on 3.10/3.11/3.12. `from __future__ import annotations` in every module; `typing.Optional`/`Union` at runtime; no `match`/`Self`/`tomllib`-only features.
- Run pytest as `PYTHONPATH= python -m pytest <args>` (clears an inherited `/opt/ros` path that loads broken ROS pytest plugins). Env: conda `yam_ws` (Python 3.11); all needed deps present.
- Reuse Plan 1 APIs verbatim — do NOT reimplement: `abcdl.episode` (`Episode`,`CameraStream`,`EpisodeMeta`,`StateLayout`), `abcdl.format.writer.write_abcdl`, `abcdl.format.reader.read_abcdl`, `abcdl.format.encode.encode_strict_h264`, `abcdl.mcap.writer.write_mcap`, `abcdl.mcap.reader.read_mcap`, `abcdl.convert.mcap_abcdl.{mcap_to_abcdl,abcdl_to_mcap}`, `abcdl.constants`.
- HF layout: one repo per dataset; **branch per format** (`abcdl`, `mcap`); **tag per version** (`v1`, `v2`, …); `main` holds only a dataset card/manifest.
- LeRobot-compatible loader surface (target = `lerobot` 0.4.4 `LeRobotDataset`): `len(ds)`==frame count; `ds[i]` returns a dict with `observation.state` (torch.FloatTensor), `action` (FloatTensor; stacked `(chunk, A)` when `delta_timestamps["action"]` is set), `observation.images.<cam>` (CHW float in [0,1]), `task` (str), `timestamp`, `episode_index`, `frame_index`, `index`; plus attributes `ds.meta.features` (dict `{key:{dtype,shape,names}}`), `ds.meta.fps`, `ds.meta.camera_keys`, `ds.meta.video_keys`, `ds.num_frames`, `ds.num_episodes`. Optional extras (lerobot installed) may reuse lerobot helpers, but the loader MUST import and work WITHOUT lerobot installed.
- Optional dependencies must be lazily imported inside the functions that need them (so `import abcdl.<module>` works without the extra). Missing optional dep → a clear `ImportError` with the install hint, only when the feature is invoked.
- TDD: failing-test-first, minimal implementation, passing test, commit. Tests use small synthetic abcdl episodes (built via `write_abcdl`); no network in CI (HF tested against a temp local dir / monkeypatched hub).

**Test fixture:** the validated sample MCAP is at `$ABCDL_SAMPLE_MCAP` or the HF cache path (see Plan 1 conftest). A reusable `tmp_abcdl_episode` fixture (Task 1) builds a small abcdl episode on disk for the other tasks.

---

## File Structure

```
abcdl/
  backends.py            # open_file(uri) -> binary stream; local / http via fsspec (no S3)
  writer.py              # EpisodeWriter: incremental add_frame()/save() -> abcdl and/or mcap
  dataset.py             # AbcdlDataset (LeRobot-compatible torch Dataset) + MixtureDataset
  convert/lerobot.py     # lerobot_to_abcdl / abcdl_to_lerobot (optional lerobot)
  hf.py                  # push / pull / list_versions  (format=branch, version=tag)
  dataset.py             # MODIFY (Task 5): repo_id auto-load + push_to_hub (LeRobot-style)
  mcap/writer.py         # MODIFY: implement decoded-frame -> Annex-B encode (remove NotImplementedError)
tests/
  conftest.py            # MODIFY: add tmp_abcdl_episode fixture
  test_backends.py test_writer.py test_dataset.py test_convert_lerobot.py test_hf.py test_dataset_hub.py
pyproject.toml           # MODIFY: add huggingface_hub, fsspec deps
```

---

### Task 1: backends + shared episode fixture

**Files:**
- Create: `abcdl/backends.py`
- Modify: `tests/conftest.py` (add `tmp_abcdl_episode` fixture), `pyproject.toml` (add `fsspec`, `huggingface_hub` to dependencies)
- Test: `tests/test_backends.py`

**Interfaces:**
- Produces:
  - `open_file(uri: str, mode: str = "rb")` — context manager yielding a binary file-like for `local path`, `file://`, `http(s)://` via fsspec. (S3 is out of scope.)
  - `local_path_for(uri: str, cache_dir: str | None = None) -> str` — return a local filesystem path for `uri` (the path itself if local; download to `cache_dir` for remote HTTP).
  - fixture `tmp_abcdl_episode(tmp_path) -> str` — writes a small abcdl episode dir (T=6, cams `top`,`left_wrist` at 16×16, 14-D state) and returns its path.

- [ ] **Step 1: Write the failing test**

`tests/test_backends.py`:
```python
from abcdl.backends import open_file, local_path_for


def test_open_local_file(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello")
    with open_file(str(p)) as f:
        assert f.read() == b"hello"


def test_local_path_for_passthrough(tmp_path):
    p = tmp_path / "y.bin"
    p.write_bytes(b"z")
    assert local_path_for(str(p)) == str(p)


def test_open_http(monkeypatch):
    # fsspec is used for remote; assert open_file routes http(s) through fsspec.open
    import abcdl.backends as backends

    class _Ctx:
        def __enter__(self): return __import__("io").BytesIO(b"web")
        def __exit__(self, *a): return False

    called = {}
    monkeypatch.setattr(backends, "_fsspec_open", lambda uri, mode: called.setdefault("uri", uri) or _Ctx())
    with open_file("https://example.com/a.bin") as f:
        assert f.read() == b"web"
    assert called["uri"] == "https://example.com/a.bin"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH= python -m pytest tests/test_backends.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'abcdl.backends'`.

- [ ] **Step 3: Write minimal implementation**

`abcdl/backends.py`:
```python
"""Uniform file access for local / HTTP URIs (so readers can stream). S3 out of scope."""

from __future__ import annotations

import contextlib
import os
from typing import Optional


def _is_remote(uri: str) -> bool:
    return "://" in uri and not uri.startswith("file://")


def _fsspec_open(uri: str, mode: str):
    import fsspec
    return fsspec.open(uri, mode)


@contextlib.contextmanager
def open_file(uri: str, mode: str = "rb"):
    if not _is_remote(uri):
        path = uri[len("file://"):] if uri.startswith("file://") else uri
        with open(path, mode) as f:
            yield f
        return
    with _fsspec_open(uri, mode) as f:
        yield f


def local_path_for(uri: str, cache_dir: Optional[str] = None) -> str:
    if not _is_remote(uri):
        return uri[len("file://"):] if uri.startswith("file://") else uri
    import fsspec

    dest_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "abcdl")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(uri.rstrip("/")))
    fs, _, paths = fsspec.get_fs_token_paths(uri)
    fs.get(paths[0], dest)
    return dest
```

Add `tmp_abcdl_episode` to `tests/conftest.py`:
```python
@pytest.fixture
def tmp_abcdl_episode(tmp_path):
    import numpy as np
    from abcdl.episode import CameraStream, Episode, EpisodeMeta
    from abcdl.format.writer import write_abcdl

    T, h, w = 6, 16, 16
    rng = np.random.default_rng(7)
    cams, res, codecs = {}, {}, {}
    for name in ("top", "left_wrist"):
        cams[name] = CameraStream(frames=rng.integers(0, 255, (T, h, w, 3), dtype=np.uint8),
                                  timestamps=np.arange(T, dtype=np.int64), width=w, height=h, codec="raw")
        res[name] = (w, h); codecs[name] = "h264"
    meta = EpisodeMeta(task="demo task", fps=30.0, cameras=["top", "left_wrist"],
                       camera_resolutions=res, camera_codecs=codecs,
                       alignment="fixed_clock_30hz_causal", t0_ns=0, tick_ns=33_333_333)
    ep = Episode(rng.standard_normal((T, 14)), rng.standard_normal((T, 14)),
                 np.arange(T, dtype=np.int64), cams, meta)
    out = tmp_path / "abcdl_ep"
    write_abcdl(ep, str(out))
    return str(out)
```

In `pyproject.toml` `[project] dependencies`, add `"huggingface_hub"` and `"fsspec"`. (No S3 extra — out of scope.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install -e . && PYTHONPATH= python -m pytest tests/test_backends.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add abcdl/backends.py tests/conftest.py tests/test_backends.py pyproject.toml
git commit -m "feat: backends (local/http/s3) + shared abcdl episode fixture"
```

---

### Task 2: EpisodeWriter + decoded-frame MCAP encoding

**Files:**
- Create: `abcdl/writer.py`
- Modify: `abcdl/mcap/writer.py` (replace the `NotImplementedError` branch in `_encoded_frames` with a real decoded-frame→single-frame-Annex-B encode, reusing `encode_strict_h264`; factor the per-frame encode helper so both modules share it)
- Test: `tests/test_writer.py`

**Interfaces:**
- Consumes: `Episode`, `EpisodeMeta`, `CameraStream`, `StateLayout`, `write_abcdl`, `write_mcap`.
- Produces:
  - `EpisodeWriter(out_dir, formats=("abcdl",), fps=30, cameras=None, state_layout=StateLayout.YAM)` with:
    - `add_frame(t_ns: int, state, action, images: dict[str, np.ndarray])` — accumulate one timestep (images are `(H,W,3)` uint8 per camera name).
    - `save(task: str, operator_id=None) -> dict[str,str]` — build an `Episode`, write each requested format (`"abcdl"` → `write_abcdl` into `out_dir`; `"mcap"` → `write_mcap` into `out_dir/episode.mcap`), return `{format: path}`. Camera order = first-seen / `cameras` arg order.
  - `encode_frame_to_annexb(frame_hwc: np.ndarray) -> bytes` (in `abcdl/mcap/writer.py`) — encode one RGB frame to an H.264 byte blob; used by `write_mcap` for decoded camera frames.

- [ ] **Step 1: Write the failing test**

`tests/test_writer.py`:
```python
import shutil

import numpy as np
import pytest

from abcdl.writer import EpisodeWriter
from abcdl.format.reader import read_abcdl

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_episode_writer_abcdl(tmp_path):
    w = EpisodeWriter(str(tmp_path / "ep"), formats=("abcdl",), fps=30,
                      cameras=["top", "left_wrist"])
    rng = np.random.default_rng(0)
    T = 5
    for i in range(T):
        w.add_frame(i * 33_333_333, rng.standard_normal(14), rng.standard_normal(14),
                    {"top": rng.integers(0, 255, (16, 16, 3), np.uint8),
                     "left_wrist": rng.integers(0, 255, (16, 16, 3), np.uint8)})
    paths = w.save(task="t")
    ep = read_abcdl(paths["abcdl"])
    assert ep.num_steps == T
    assert ep.states.shape == (T, 14)
    assert set(ep.cameras) == {"top", "left_wrist"}


def test_episode_writer_mcap_encodes_decoded_frames(tmp_path):
    from abcdl.mcap.reader import read_mcap
    w = EpisodeWriter(str(tmp_path / "ep2"), formats=("mcap",), fps=30, cameras=["top"])
    rng = np.random.default_rng(1)
    for i in range(5):
        w.add_frame(i * 33_333_333, rng.standard_normal(14), rng.standard_normal(14),
                    {"top": rng.integers(0, 255, (16, 16, 3), np.uint8)})
    paths = w.save(task="t2")
    back = read_mcap(paths["mcap"])
    assert back.meta.task == "t2"
    assert back.states.shape == (5, 14)
    assert "top" in back.cameras
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH= python -m pytest tests/test_writer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'abcdl.writer'`.

- [ ] **Step 3: Write minimal implementation**

First, in `abcdl/mcap/writer.py`, add a shared helper and use it in `_encoded_frames`:
```python
def encode_frame_to_annexb(frame_hwc) -> bytes:
    """Encode one (H,W,3) uint8 RGB frame to an H.264 byte blob."""
    import os
    import tempfile
    import numpy as np
    from abcdl.format.encode import encode_strict_h264

    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        encode_strict_h264(np.asarray(frame_hwc, np.uint8)[None], tmp)
        with open(tmp, "rb") as fh:
            return fh.read()
    finally:
        os.unlink(tmp)
```
and replace the `raise NotImplementedError(...)` branch of `_encoded_frames` with:
```python
    return [encode_frame_to_annexb(fr) for fr in cam.frames], np.asarray(cam.timestamps, np.int64)
```

`abcdl/writer.py`:
```python
"""Incremental episode writer — accumulate frames live, then save to abcdl and/or mcap."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from abcdl.constants import TICK_NS
from abcdl.episode import CameraStream, Episode, EpisodeMeta, StateLayout
from abcdl.format.writer import write_abcdl
from abcdl.mcap.writer import write_mcap


class EpisodeWriter:
    def __init__(self, out_dir: str, formats=("abcdl",), fps: int = 30,
                 cameras: Optional[list] = None, state_layout: StateLayout = StateLayout.YAM):
        self.out_dir = out_dir
        self.formats = tuple(formats)
        self.fps = fps
        self.cameras = list(cameras) if cameras else None
        self.layout = state_layout
        self._t: list = []
        self._states: list = []
        self._actions: list = []
        self._frames: dict = {}

    def add_frame(self, t_ns: int, state, action, images: dict) -> None:
        self._t.append(int(t_ns))
        self._states.append(np.asarray(state, np.float64))
        self._actions.append(np.asarray(action, np.float64))
        if self.cameras is None:
            self.cameras = list(images.keys())
        for name in self.cameras:
            self._frames.setdefault(name, []).append(np.asarray(images[name], np.uint8))

    def save(self, task: str, operator_id: Optional[str] = None) -> dict:
        T = len(self._t)
        if T == 0:
            raise ValueError("no frames added")
        ts = np.asarray(self._t, np.int64)
        cams, res, codecs = {}, {}, {}
        for name in self.cameras:
            arr = np.stack(self._frames[name])
            cams[name] = CameraStream(frames=arr, timestamps=ts, width=arr.shape[2],
                                      height=arr.shape[1], codec="raw")
            res[name] = (arr.shape[2], arr.shape[1]); codecs[name] = "h264"
        meta = EpisodeMeta(task=task, fps=float(self.fps), cameras=list(self.cameras),
                           camera_resolutions=res, camera_codecs=codecs, operator_id=operator_id,
                           alignment="fixed_clock_30hz_causal", t0_ns=int(ts[0]), tick_ns=TICK_NS)
        ep = Episode(np.stack(self._states), np.stack(self._actions), ts, cams, meta)
        out = {}
        if "abcdl" in self.formats:
            write_abcdl(ep, self.out_dir); out["abcdl"] = self.out_dir
        if "mcap" in self.formats:
            os.makedirs(self.out_dir, exist_ok=True)
            p = os.path.join(self.out_dir, "episode.mcap")
            write_mcap(ep, p); out["mcap"] = p
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH= python -m pytest tests/test_writer.py tests/test_mcap_roundtrip.py -v`
Expected: PASS (writer tests + the Plan 1 mcap round-trip still green).

- [ ] **Step 5: Commit**

```bash
git add abcdl/writer.py abcdl/mcap/writer.py tests/test_writer.py
git commit -m "feat: EpisodeWriter + decoded-frame MCAP encoding"
```

---

### Task 3: AbcdlDataset (LeRobot-compatible loader)

**Files:**
- Create: `abcdl/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `read_abcdl` (and its efficient decode), `abcdl.constants`.
- Produces:
  - `AbcdlDatasetMeta` with `.features` (`{key:{dtype,shape,names}}`), `.fps`, `.camera_keys` (list), `.video_keys` (list), `.tasks` (list[str]), `.robot_type`.
  - `AbcdlDataset(root, delta_timestamps=None, camera_keys=None)` — `root` is a dir of `episode_*/` abcdl dirs. Scans episodes, builds a global frame index. `len(ds)` == total frames. `ds[i]` returns a dict with `observation.state` (FloatTensor `(S,)`), `action` (FloatTensor `(A,)`, or `(K,A)` when `delta_timestamps["action"]` lists K offsets), `observation.images.<cam>` (FloatTensor CHW in [0,1]), `task` (str), `timestamp` (float s), `episode_index`, `frame_index`, `index` (all int tensors). `.meta` exposes `AbcdlDatasetMeta`. `.num_frames`, `.num_episodes` ints.
  - `MixtureDataset(datasets: list, weights: list)` — `torch.utils.data.Dataset` sampling sub-datasets by hours/weight (weighted index map).
- Decode strategy: cache per-episode decoded frames lazily (decode the episode's combined mp4 once via `read_abcdl`, keep an LRU of N most-recent episodes) so shuffled access doesn't re-decode every item.

- [ ] **Step 1: Write the failing test**

`tests/test_dataset.py`:
```python
import shutil

import numpy as np
import pytest
import torch

from abcdl.dataset import AbcdlDataset

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _make_root(tmp_path, tmp_abcdl_episode):
    # two episodes under a root
    root = tmp_path / "ds"
    root.mkdir()
    for k in range(2):
        shutil.copytree(tmp_abcdl_episode, root / f"episode_{k:04d}")
    return str(root)


def test_dataset_len_and_item(tmp_path, tmp_abcdl_episode):
    root = _make_root(tmp_path, tmp_abcdl_episode)
    ds = AbcdlDataset(root)
    assert ds.num_episodes == 2
    assert ds.num_frames == len(ds) == 12  # 6 frames x 2
    item = ds[0]
    assert item["observation.state"].shape == (14,)
    assert item["action"].shape == (14,)
    assert item["observation.images.top"].shape[0] == 3  # CHW
    assert 0.0 <= float(item["observation.images.top"].max()) <= 1.0
    assert item["task"] == "demo task"
    assert int(item["episode_index"]) in (0, 1)


def test_dataset_action_chunk(tmp_path, tmp_abcdl_episode):
    root = _make_root(tmp_path, tmp_abcdl_episode)
    ds = AbcdlDataset(root, delta_timestamps={"action": [0.0, 1 / 30, 2 / 30]})
    item = ds[0]
    assert item["action"].shape == (3, 14)


def test_meta_features(tmp_path, tmp_abcdl_episode):
    root = _make_root(tmp_path, tmp_abcdl_episode)
    ds = AbcdlDataset(root)
    assert ds.meta.fps == 30.0
    assert set(ds.meta.camera_keys) == {"top", "left_wrist"}
    assert "observation.state" in ds.meta.features
    assert ds.meta.features["observation.state"]["shape"] == (14,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH= python -m pytest tests/test_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'abcdl.dataset'`.

- [ ] **Step 3: Write minimal implementation**

`abcdl/dataset.py`:
```python
"""LeRobot-compatible torch Dataset over a directory of abcdl episodes."""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from abcdl.format.reader import read_abcdl


@dataclass
class AbcdlDatasetMeta:
    features: dict
    fps: float
    camera_keys: list
    video_keys: list
    tasks: list
    robot_type: str = "yam_bimanual"


class AbcdlDataset(Dataset):
    def __init__(self, root: str, delta_timestamps: Optional[dict] = None,
                 camera_keys: Optional[list] = None, cache_episodes: int = 4):
        self.root = root
        self.delta_timestamps = delta_timestamps or {}
        self._dirs = sorted(
            os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d)) and
            os.path.exists(os.path.join(root, d, "states_actions.bin")))
        if not self._dirs:
            raise ValueError(f"no abcdl episodes under {root}")
        # build per-episode length + global index from metadata json
        import json
        self._lengths, self._tasks_per_ep, self._cams = [], [], None
        for d in self._dirs:
            m = json.load(open(os.path.join(d, "episode_metadata.json")))
            self._lengths.append(int(m["num_steps"]))
            self._tasks_per_ep.append(m["task_name"])
            if self._cams is None:
                self._cams = list(m["cameras"])
        self.camera_keys = camera_keys or self._cams
        self._starts = np.cumsum([0] + self._lengths)
        self.num_frames = int(self._starts[-1])
        self.num_episodes = len(self._dirs)
        self._cache: OrderedDict = OrderedDict()
        self._cache_n = cache_episodes
        s_dim, a_dim = self._probe_dims()
        tasks = sorted(set(self._tasks_per_ep))
        feats = {
            "observation.state": {"dtype": "float32", "shape": (s_dim,), "names": None},
            "action": {"dtype": "float32", "shape": (a_dim,), "names": None},
        }
        for cam in self.camera_keys:
            feats[f"observation.images.{cam}"] = {"dtype": "video", "shape": None, "names": None}
        self.meta = AbcdlDatasetMeta(features=feats, fps=self._fps, camera_keys=list(self.camera_keys),
                                     video_keys=[f"observation.images.{c}" for c in self.camera_keys],
                                     tasks=tasks)

    def _probe_dims(self):
        import json
        m = json.load(open(os.path.join(self._dirs[0], "episode_metadata.json")))
        self._fps = 1e9 / int(m.get("tick_ns")) if m.get("tick_ns") else float(m.get("fps", 30.0))
        return int(m["state_dim"]), int(m["action_dim"])

    def __len__(self) -> int:
        return self.num_frames

    def _episode(self, ep_idx: int):
        if ep_idx in self._cache:
            self._cache.move_to_end(ep_idx)
            return self._cache[ep_idx]
        ep = read_abcdl(self._dirs[ep_idx])
        self._cache[ep_idx] = ep
        if len(self._cache) > self._cache_n:
            self._cache.popitem(last=False)
        return ep

    def _locate(self, idx: int):
        ep_idx = int(np.searchsorted(self._starts, idx, side="right") - 1)
        return ep_idx, idx - int(self._starts[ep_idx])

    def __getitem__(self, idx: int) -> dict:
        ep_idx, frame = self._locate(idx)
        ep = self._episode(ep_idx)
        out = {
            "observation.state": torch.from_numpy(ep.states[frame].astype(np.float32)),
            "task": ep.meta.task,
            "timestamp": torch.tensor(frame / self._fps, dtype=torch.float32),
            "episode_index": torch.tensor(ep_idx),
            "frame_index": torch.tensor(frame),
            "index": torch.tensor(idx),
        }
        # action: single or chunked by delta_timestamps
        offs = self.delta_timestamps.get("action")
        if offs:
            n = ep.num_steps
            rows = [min(frame + int(round(o * self._fps)), n - 1) for o in offs]
            out["action"] = torch.from_numpy(ep.actions[rows].astype(np.float32))
        else:
            out["action"] = torch.from_numpy(ep.actions[frame].astype(np.float32))
        for cam in self.camera_keys:
            hwc = ep.cameras[cam].frames[frame]
            chw = torch.from_numpy(np.ascontiguousarray(hwc)).permute(2, 0, 1).float() / 255.0
            out[f"observation.images.{cam}"] = chw
        return out


class MixtureDataset(Dataset):
    def __init__(self, datasets: list, weights: list):
        self.datasets = datasets
        w = np.asarray(weights, np.float64); w = w / w.sum()
        # integer index map proportional to weights, length = sum of dataset lengths
        total = sum(len(d) for d in datasets)
        self._map = []
        for di, (d, wi) in enumerate(zip(datasets, w)):
            k = max(1, int(round(wi * total)))
            local = np.linspace(0, len(d) - 1, k).astype(int)
            self._map.extend((di, int(j)) for j in local)

    def __len__(self) -> int:
        return len(self._map)

    def __getitem__(self, i: int) -> dict:
        di, j = self._map[i]
        return self.datasets[di][j]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH= python -m pytest tests/test_dataset.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add abcdl/dataset.py tests/test_dataset.py
git commit -m "feat: AbcdlDataset (LeRobot-compatible loader) + MixtureDataset"
```

---

### Task 4: LeRobot ↔ abcdl conversion

**Files:**
- Create: `abcdl/convert/lerobot.py`
- Test: `tests/test_convert_lerobot.py`

**Interfaces:**
- Consumes: `read_abcdl`, `Episode`/`EpisodeMeta`/`CameraStream`, lerobot (optional, lazy import).
- Produces:
  - `abcdl_to_lerobot(abcdl_root: str, repo_id: str, lerobot_root: str | None = None, fps: int = 30) -> str` — create a `LeRobotDataset` and append each abcdl episode's frames (`observation.state`, `action`, `observation.images.<cam>`), `save_episode()` per episode, finalize; return the dataset root.
  - `lerobot_to_abcdl(repo_id_or_root: str, out_root: str) -> list[str]` — read a LeRobotDataset, group frames by `episode_index`, build an `Episode` per episode, `write_abcdl` into `out_root/episode_<i>/`; return the list of written dirs.
  - Both raise a clear `ImportError("lerobot conversion needs `pip install abcdl[lerobot]`")` when lerobot is absent.

- [ ] **Step 1: Write the failing test**

`tests/test_convert_lerobot.py`:
```python
import shutil

import numpy as np
import pytest

lerobot = pytest.importorskip("lerobot")
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_abcdl_to_lerobot_roundtrip(tmp_path, tmp_abcdl_episode):
    from abcdl.convert.lerobot import abcdl_to_lerobot, lerobot_to_abcdl
    from abcdl.format.reader import read_abcdl

    root = tmp_path / "ds"; root.mkdir()
    shutil.copytree(tmp_abcdl_episode, root / "episode_0000")

    lr_root = abcdl_to_lerobot(str(root), "test/yam", lerobot_root=str(tmp_path / "lr"))
    back_dirs = lerobot_to_abcdl(lr_root, str(tmp_path / "back"))
    assert len(back_dirs) == 1
    src = read_abcdl(str(root / "episode_0000"))
    back = read_abcdl(back_dirs[0])
    assert back.num_steps == src.num_steps
    np.testing.assert_allclose(back.states, src.states, atol=1e-4)  # parquet float32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH= python -m pytest tests/test_convert_lerobot.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'abcdl.convert.lerobot'`.

- [ ] **Step 3: Write minimal implementation**

> **API VERIFICATION REQUIRED (implementer):** This task depends on `lerobot` 0.4.4's dataset API. BEFORE writing the final code, confirm the actual signatures and feature schema:
> - `from lerobot.datasets.lerobot_dataset import LeRobotDataset` — confirm `LeRobotDataset.create(repo_id, fps, features, root=..., robot_type=..., use_videos=True)` signature and the `features` dict schema (key → `{dtype, shape, names}`; image/video features use `dtype="video"` or `"image"` with `shape=(H,W,C)`/`(C,H,W)` — verify which).
> - `add_frame(frame_dict, task=...)` and `save_episode()` / `finalize()` — confirm names/signatures.
> - Reading: how to iterate frames + `episode_index` (e.g. `ds.hf_dataset`, `ds[i]`, `ds.meta.total_episodes`).
> Run a tiny scratch script against a 2-frame dataset to lock the API, then implement. If the API differs materially from the sketch below, follow the REAL API (the sketch is a starting point, not gospel). If you cannot create a LeRobotDataset in this env, report BLOCKED with the actual error + the `create`/`add_frame` signatures you found.

`abcdl/convert/lerobot.py` (sketch — adapt to the verified API):
```python
"""Convert between LeRobot datasets and the abcdl format (via the Episode IR)."""

from __future__ import annotations

import os

import numpy as np

from abcdl.episode import CameraStream, Episode, EpisodeMeta
from abcdl.format.reader import read_abcdl
from abcdl.format.writer import write_abcdl


def _require_lerobot():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        return LeRobotDataset
    except ImportError as e:
        raise ImportError("lerobot conversion needs `pip install abcdl[lerobot]`") from e


def abcdl_to_lerobot(abcdl_root, repo_id, lerobot_root=None, fps=30):
    LeRobotDataset = _require_lerobot()
    dirs = sorted(os.path.join(abcdl_root, d) for d in os.listdir(abcdl_root)
                  if os.path.exists(os.path.join(abcdl_root, d, "states_actions.bin")))
    first = read_abcdl(dirs[0])
    h, w = first.cameras[first.meta.cameras[0]].frames.shape[1:3]
    features = {
        "observation.state": {"dtype": "float32", "shape": (first.states.shape[1],), "names": None},
        "action": {"dtype": "float32", "shape": (first.actions.shape[1],), "names": None},
    }
    for cam in first.meta.cameras:
        features[f"observation.images.{cam}"] = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channel"]}
    ds = LeRobotDataset.create(repo_id, fps=fps, features=features, root=lerobot_root, use_videos=True)
    for d in dirs:
        ep = read_abcdl(d)
        for i in range(ep.num_steps):
            frame = {"observation.state": ep.states[i].astype(np.float32),
                     "action": ep.actions[i].astype(np.float32)}
            for cam in ep.meta.cameras:
                frame[f"observation.images.{cam}"] = ep.cameras[cam].frames[i]
            ds.add_frame(frame, task=ep.meta.task)
        ds.save_episode()
    if hasattr(ds, "finalize"):
        ds.finalize()
    return str(ds.root)


def lerobot_to_abcdl(repo_id_or_root, out_root):
    LeRobotDataset = _require_lerobot()
    ds = LeRobotDataset(repo_id_or_root) if "/" in repo_id_or_root and not os.path.exists(repo_id_or_root) \
        else LeRobotDataset(repo_id_or_root.rstrip("/").split("/")[-1], root=repo_id_or_root)
    cam_keys = list(ds.meta.camera_keys)
    by_ep: dict = {}
    for i in range(len(ds)):
        item = ds[i]
        by_ep.setdefault(int(item["episode_index"]), []).append(item)
    written = []
    for ep_idx, items in sorted(by_ep.items()):
        T = len(items)
        states = np.stack([it["observation.state"].numpy() for it in items]).astype(np.float64)
        actions = np.stack([it["action"].numpy() for it in items]).astype(np.float64)
        cams, res, codecs = {}, {}, {}
        for cam in cam_keys:
            frames = np.stack([(it[f"observation.images.{cam}"].permute(1, 2, 0).numpy() * 255)
                               .astype(np.uint8) for it in items])
            cams[cam] = CameraStream(frames=frames, timestamps=np.arange(T, dtype=np.int64),
                                     width=frames.shape[2], height=frames.shape[1], codec="raw")
            res[cam] = (frames.shape[2], frames.shape[1]); codecs[cam] = "h264"
        meta = EpisodeMeta(task=items[0]["task"], fps=float(ds.meta.fps), cameras=cam_keys,
                           camera_resolutions=res, camera_codecs=codecs,
                           alignment="fixed_clock_30hz_causal", t0_ns=0,
                           tick_ns=int(1e9 / ds.meta.fps))
        ep = Episode(states, actions, np.arange(T, dtype=np.int64), cams, meta)
        d = os.path.join(out_root, f"episode_{ep_idx:04d}")
        write_abcdl(ep, d); written.append(d)
    return written
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH= python -m pytest tests/test_convert_lerobot.py -v`
Expected: PASS (skips if lerobot absent — it is present in yam_ws, so it runs).

- [ ] **Step 5: Commit**

```bash
git add abcdl/convert/lerobot.py tests/test_convert_lerobot.py
git commit -m "feat: LeRobot <-> abcdl conversion"
```

---

### Task 5: HuggingFace integration — LeRobot-style push_to_hub + repo_id auto-load

**Files:**
- Create: `abcdl/hf.py`
- Modify: `abcdl/dataset.py` (add repo_id auto-load + `push_to_hub` so usage mirrors LeRobot)
- Test: `tests/test_hf.py`, `tests/test_dataset_hub.py`

**Goal (per user):** no CLI. Make it feel like LeRobot — `AbcdlDataset("jellyho/yam_pick")` loads directly (auto-downloads from the Hub), and `ds.push_to_hub("jellyho/yam_pick", "v1")` uploads. The format=branch / version=tag mechanism stays underneath.

**Interfaces:**
- Consumes: `huggingface_hub` (lazy import).
- Produces in `abcdl/hf.py`:
  - `push(repo_id, local_dir, fmt, version, message=None, token=None) -> str` — ensure repo exists (`create_repo(..., repo_type="dataset", exist_ok=True)`), create/use branch `fmt` (`create_branch(..., branch=fmt, exist_ok=True)`), `upload_folder(folder_path=local_dir, repo_id, repo_type="dataset", revision=fmt, commit_message=...)`, then tag `version` on that revision (`create_tag(repo_id, tag=version, revision=fmt, repo_type="dataset")`). Return the revision used.
  - `pull(repo_id, fmt, version="latest", dest=None, token=None) -> str` — resolve revision (the tag `version`, else branch `fmt`), `snapshot_download(repo_id, repo_type="dataset", revision=..., local_dir=dest)`; return the local dir.
  - `list_versions(repo_id, fmt=None, token=None) -> dict` — `list_repo_refs(repo_id, repo_type="dataset")`; return `{"branches":[...], "tags":[...]}` (filtered to `fmt` branch if given).
- Produces in `abcdl/dataset.py` (LeRobot-style ergonomics on `AbcdlDataset`):
  - The first positional arg now means **root-or-repo_id**: if it is an existing local directory, use it as the root (existing behavior); otherwise if it matches `owner/name` (a `/`, not an existing path), treat it as a Hub `repo_id` and **auto-pull** via `hf.pull(repo_id, fmt, version)` into a cache dir, then load. New kwargs: `fmt="abcdl"`, `version="latest"`, `revision_root=None` (download dir; default `~/.cache/abcdl/<owner>__<name>/<version>`). A `from_hub(repo_id, fmt="abcdl", version="latest", root=None)` classmethod does the same explicitly.
  - `AbcdlDataset.push_to_hub(repo_id, version, fmt="abcdl", message=None, token=None) -> str` — uploads `self.root` via `hf.push(...)`. (Mirrors `LeRobotDataset.push_to_hub`.)
- Tests do NOT hit the network: monkeypatch `hf._hub` (for hf.py) and monkeypatch `abcdl.dataset.hf.pull`/`hf.push` (for the dataset wiring) and assert the right `repo_type`/`revision`/`branch`/`tag`/`fmt`/`version` flow.

- [ ] **Step 1: Write the failing test**

`tests/test_hf.py`:
```python
import types

import abcdl.hf as hf


def test_push_uses_branch_and_tag(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr(hf, "_hub", lambda: types.SimpleNamespace(
        create_repo=lambda **k: calls.setdefault("create_repo", k),
        create_branch=lambda **k: calls.setdefault("create_branch", k),
        upload_folder=lambda **k: calls.setdefault("upload_folder", k),
        create_tag=lambda **k: calls.setdefault("create_tag", k),
    ))
    (tmp_path / "f.txt").write_text("x")
    hf.push("jellyho/yam", str(tmp_path), fmt="abcdl", version="v1")
    assert calls["create_branch"]["branch"] == "abcdl"
    assert calls["upload_folder"]["revision"] == "abcdl"
    assert calls["create_tag"]["tag"] == "v1"
    assert calls["create_tag"]["revision"] == "abcdl"
    assert calls["upload_folder"]["repo_type"] == "dataset"


def test_pull_prefers_version_tag(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(hf, "_hub", lambda: types.SimpleNamespace(
        snapshot_download=lambda **k: seen.update(k) or str(tmp_path)))
    out = hf.pull("jellyho/yam", fmt="abcdl", version="v2", dest=str(tmp_path))
    assert seen["revision"] == "v2"
    assert seen["repo_type"] == "dataset"
    assert out == str(tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH= python -m pytest tests/test_hf.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'abcdl.hf'`.

- [ ] **Step 3: Write minimal implementation**

> **API VERIFICATION (implementer):** confirm `huggingface_hub` 0.35.x exposes `create_repo`, `create_branch`, `upload_folder`, `create_tag`, `snapshot_download`, `list_repo_refs` with the kwargs used below (`repo_type`, `revision`, `branch`, `tag`, `exist_ok`). Adjust to the real signatures if they differ; keep the `_hub()` indirection so tests can monkeypatch.

`abcdl/hf.py`:
```python
"""Publish/fetch abcdl & mcap datasets on the HuggingFace Hub (format=branch, version=tag)."""

from __future__ import annotations

from typing import Optional


def _hub():
    try:
        import huggingface_hub as h
    except ImportError as e:
        raise ImportError("HF integration needs `pip install huggingface_hub`") from e
    return h


def push(repo_id: str, local_dir: str, fmt: str, version: str,
         message: Optional[str] = None, token: Optional[str] = None) -> str:
    h = _hub()
    h.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=token)
    h.create_branch(repo_id=repo_id, repo_type="dataset", branch=fmt, exist_ok=True, token=token)
    h.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=local_dir, revision=fmt,
                    commit_message=message or f"upload {fmt} {version}", token=token)
    h.create_tag(repo_id=repo_id, repo_type="dataset", tag=version, revision=fmt, token=token)
    return fmt


def pull(repo_id: str, fmt: str, version: str = "latest", dest: Optional[str] = None,
         token: Optional[str] = None) -> str:
    h = _hub()
    revision = fmt if version == "latest" else version
    return h.snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision,
                               local_dir=dest, token=token)


def list_versions(repo_id: str, fmt: Optional[str] = None, token: Optional[str] = None) -> dict:
    h = _hub()
    refs = h.list_repo_refs(repo_id, repo_type="dataset", token=token)
    branches = [b.name for b in refs.branches]
    tags = [t.name for t in refs.tags]
    if fmt is not None:
        branches = [b for b in branches if b == fmt]
    return {"branches": branches, "tags": tags}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH= python -m pytest tests/test_hf.py -v`
Expected: PASS (2 tests, no network).

- [ ] **Step 5: Commit**

```bash
git add abcdl/hf.py tests/test_hf.py
git commit -m "feat: HuggingFace push/pull/list_versions (format=branch, version=tag)"
```

- [ ] **Step 6: Write the failing test for the LeRobot-style dataset wiring**

`tests/test_dataset_hub.py`:
```python
import abcdl.dataset as dsmod
from abcdl.dataset import AbcdlDataset


def test_repo_id_autopull_then_load(monkeypatch, tmp_abcdl_episode, tmp_path):
    # a repo_id (owner/name, not an existing path) triggers hf.pull, whose returned
    # dir is then loaded as a normal abcdl root. Point pull at a real local episode root.
    import shutil
    root = tmp_path / "pulled"; root.mkdir()
    shutil.copytree(tmp_abcdl_episode, root / "episode_0000")
    seen = {}
    monkeypatch.setattr(dsmod.hf, "pull",
                        lambda **k: seen.update(k) or str(root))
    ds = AbcdlDataset("jellyho/yam_pick", version="v1")
    assert seen["repo_id"] == "jellyho/yam_pick"
    assert seen["fmt"] == "abcdl" and seen["version"] == "v1"
    assert ds.num_episodes == 1 and len(ds) == 6


def test_push_to_hub_calls_hf_push(monkeypatch, tmp_abcdl_episode):
    seen = {}
    monkeypatch.setattr(dsmod.hf, "push", lambda **k: seen.update(k) or k["fmt"])
    ds = AbcdlDataset(tmp_abcdl_episode)  # existing local dir → root
    out = ds.push_to_hub("jellyho/yam_pick", "v2", fmt="abcdl")
    assert seen["repo_id"] == "jellyho/yam_pick"
    assert seen["fmt"] == "abcdl" and seen["version"] == "v2"
    assert seen["local_dir"] == ds.root
```

- [ ] **Step 7: Run test to verify it fails**

Run: `PYTHONPATH= python -m pytest tests/test_dataset_hub.py -v`
Expected: FAIL — `AbcdlDataset` has no `repo_id` handling / no `push_to_hub` (AttributeError / wrong behavior).

- [ ] **Step 8: Wire HF into `abcdl/dataset.py`**

At the top of `abcdl/dataset.py`, add `from abcdl import hf` (module import so tests can monkeypatch `dsmod.hf.pull`/`push`). Then:
- In `AbcdlDataset.__init__`, change the first positional `root` handling: if `root` is NOT an existing directory AND looks like a Hub repo id (`"/" in root` and not `os.path.isdir(root)`), resolve it from the Hub:
```python
        if not os.path.isdir(root) and "/" in root:
            owner_name = root.replace("/", "__")
            dest = revision_root or os.path.join(
                os.path.expanduser("~"), ".cache", "abcdl", owner_name, version)
            os.makedirs(dest, exist_ok=True)
            root = hf.pull(repo_id=root, fmt=fmt, version=version, dest=dest)
        self.root = root
```
  Add the new kwargs to `__init__`: `fmt: str = "abcdl"`, `version: str = "latest"`, `revision_root: Optional[str] = None` (keep existing `delta_timestamps`, `camera_keys`, `cache_episodes`). Record `self.repo_id = root if "/" in <original> ... ` is optional; at minimum keep `self.root` pointing at the loaded dir.
- Add a classmethod:
```python
    @classmethod
    def from_hub(cls, repo_id, fmt="abcdl", version="latest", root=None, **kw):
        return cls(repo_id, fmt=fmt, version=version, revision_root=root, **kw)
```
- Add the upload method:
```python
    def push_to_hub(self, repo_id: str, version: str, fmt: str = "abcdl",
                    message: Optional[str] = None, token: Optional[str] = None) -> str:
        return hf.push(repo_id=repo_id, local_dir=self.root, fmt=fmt, version=version,
                       message=message, token=token)
```
Keep all existing dataset behavior intact (a plain local `root` path must still work exactly as before — the Hub path only triggers for `owner/name` strings that are not existing dirs).

- [ ] **Step 9: Run tests + commit**

Run: `PYTHONPATH= python -m pytest tests/test_dataset_hub.py tests/test_dataset.py -v` then `PYTHONPATH= python -m pytest tests/ -q`
Expected: PASS (new hub-wiring tests + existing dataset tests + full suite).

```bash
git add abcdl/dataset.py tests/test_dataset_hub.py
git commit -m "feat: LeRobot-style repo_id auto-load + push_to_hub on AbcdlDataset"
```

---


## Self-Review

**Spec coverage (Plan 2 portion of spec §6):**
- §6.3 `dataset.py` AbcdlDataset (LeRobot-compatible) + MixtureDataset → Task 3. ✓
- §6.4 `hf.py` (format=branch/version=tag) + `backends.py` (local/HTTP; S3 dropped per user) → Tasks 1, 5. ✓
- §6.5 `writer.py` EpisodeWriter + decoded-frame MCAP encode (resolves Plan 1 `NotImplementedError`) → Task 2. ✓
- §6 `convert/lerobot.py` → Task 4. ✓
- §6.6 `cli.py` → **dropped per user**; replaced by LeRobot-style ergonomics (`push_to_hub` + `AbcdlDataset(repo_id=...)` auto-load) in Task 5. ✓
- Optional-dep laziness (lerobot/hf) → enforced in Tasks 4/5 (lazy import + clear ImportError). ✓

**Placeholder scan:** Tasks 4 and 5 carry explicit "API VERIFICATION REQUIRED" notes because they bind to third-party APIs (lerobot 0.4.4, huggingface_hub 0.35.x); the sketches are starting points and the implementer must confirm the real signatures (the same discipline used in Plan 1 for torchcodec/protoc/mcap). This is intentional, not a placeholder for missing logic.

**Type consistency:** `Episode`/`CameraStream`/`EpisodeMeta` reused from Plan 1; loader returns `observation.state`/`action`/`observation.images.<cam>`/`task` consistently; `hf.push`/`pull` use `fmt`/`version` consistently; `AbcdlDataset(repo_id=...)` calls `hf.pull` and `AbcdlDataset.push_to_hub` calls `hf.push` with matching `repo_id`/`fmt`/`version` kwargs.
