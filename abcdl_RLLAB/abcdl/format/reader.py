"""Read the abcdl MP4+binary layout back into an Episode (analytic frame index)."""

from __future__ import annotations

import json
import os

import numpy as np

from abcdl.constants import FPS, TICK_NS, TICKS_PER_FRAME
from abcdl.episode import CameraStream, Episode, EpisodeMeta


def _synth_frame_map(num_steps: int) -> str:
    """Synthesized constant-frame-rate map (analytic index; no file probing)."""
    frames = [{"pts": TICKS_PER_FRAME * i, "duration": TICKS_PER_FRAME,
               "key_frame": 1 if i % FPS == 0 else 0} for i in range(num_steps)]
    return json.dumps({"frames": frames})


def _decode_all_frames(mp4_path: str, num_steps: int) -> np.ndarray:
    """Decode every frame via torchcodec with a synthesized CFR frame map."""
    from torchcodec.decoders import VideoDecoder

    dec = VideoDecoder(mp4_path, custom_frame_mappings=_synth_frame_map(num_steps))
    out = np.empty((num_steps, *dec[0].shape), dtype=np.uint8)  # (C,H,W) per frame
    for i in range(num_steps):
        out[i] = dec[i].numpy()
    return out  # (T, C, Hstack, W)


class EpisodeHandle:
    """Lightweight random-access handle for one abcdl episode.

    Unlike :func:`read_abcdl` (which decodes the whole video up front), this keeps
    a torchcodec decoder open and decodes ONLY the requested frame on demand — the
    access pattern the abcdl analytic frame index is designed for, and what a
    shuffled training DataLoader needs. States/actions are small and kept in memory.
    """

    def __init__(self, decoder, states: np.ndarray, actions: np.ndarray,
                 cam_names: list, cam_h: int, task: str, fps: float, num_steps: int,
                 keep=None, frame_features: Optional[dict] = None):
        self.decoder = decoder
        self.states = states
        self.actions = actions
        self.cam_names = cam_names
        self.cam_h = cam_h
        self.task = task
        self.fps = fps
        self.num_steps = num_steps
        self._keep = keep  # holds an open remote file object alive (streaming)
        self.frame_features = frame_features or {}  # {name: (T, ...) array}

    def frame(self, i: int) -> dict:
        """Decode ONLY frame *i*; return ``{cam_name: (H, W, 3) uint8}``."""
        stacked = np.transpose(self.decoder[i].numpy(), (1, 2, 0))  # (Hstack, W, C)
        h = self.cam_h
        return {name: stacked[k * h:(k + 1) * h, :, :]
                for k, name in enumerate(self.cam_names)}


def _handle_from(meta: dict, sa: np.ndarray, mp4_source, keep=None,
                 frame_features: Optional[dict] = None, with_video: bool = True) -> EpisodeHandle:
    """Build an EpisodeHandle from metadata, a states+actions array, and an mp4 source
    (a local path OR a seekable file-like — the latter lets torchcodec range-read a
    remote file, i.e. stream only the GOP it needs). When *with_video* is False the
    video decoder is not opened at all (features-only fast path)."""
    T = int(meta["num_steps"])
    sd, ad = int(meta["state_dim"]), int(meta["action_dim"])
    names = list(meta["cameras"])
    cam_h = int(meta["camera_resolutions"][names[0]][1])  # (width, height) -> height
    tick_ns = int(meta.get("tick_ns", TICK_NS))
    fps = float(meta["fps"]) if "fps" in meta else (1e9 / tick_ns if tick_ns else 30.0)
    dec = None
    if with_video:
        from torchcodec.decoders import VideoDecoder
        dec = VideoDecoder(mp4_source, custom_frame_mappings=_synth_frame_map(T))
    return EpisodeHandle(dec, sa[:, :sd], sa[:, sd:], names, cam_h, meta["task_name"],
                         fps, T, keep=keep, frame_features=frame_features)


def _frame_features_local(in_dir: str, meta: dict, T: int) -> Optional[dict]:
    """MEMMAP each per-frame feature bin (no decode, no whole-load — reads only touched
    rows). Falls back to the legacy frame_features.npz for older datasets."""
    spec = meta.get("frame_features")
    if spec:
        out = {}
        for name, s in spec.items():
            shape = (T,) + tuple(s["shape"])
            out[name] = np.memmap(os.path.join(in_dir, f"ff_{name}.bin"),
                                  dtype=s["dtype"], mode="r").reshape(shape)
        return out
    if meta.get("frame_feature_keys") and os.path.exists(os.path.join(in_dir, "frame_features.npz")):
        return dict(np.load(os.path.join(in_dir, "frame_features.npz")))
    return None


def open_episode(in_dir: str, with_video: bool = True) -> EpisodeHandle:
    """Open a LOCAL abcdl episode for efficient random per-frame access (no full decode).

    states/actions and every per-frame feature are MEMMAPed (read only the touched rows;
    the OS page cache is shared across processes — ideal for multi-worker / multi-GPU
    DDP training). Only the video is decoded on demand; pass with_video=False to skip it.
    """
    with open(os.path.join(in_dir, "episode_metadata.json")) as f:
        meta = json.load(f)
    T = int(meta["num_steps"])
    row = int(meta["state_dim"]) + int(meta["action_dim"])
    sa = np.memmap(os.path.join(in_dir, "states_actions.bin"), dtype="<f8", mode="r").reshape(T, row)
    ff = _frame_features_local(in_dir, meta, T)
    mp4 = os.path.join(in_dir, "combined_camera-images-rgb.mp4") if with_video else None
    return _handle_from(meta, sa, mp4, frame_features=ff, with_video=with_video)


def open_episode_streaming(fs, ep_uri: str, with_video: bool = True) -> EpisodeHandle:
    """Open a REMOTE abcdl episode over fsspec, streaming frames via HTTP range reads.

    *fs* is an fsspec filesystem (e.g. ``huggingface_hub.HfFileSystem``); *ep_uri* is the
    episode directory path within it. The small binaries (states/actions + per-frame
    features) are fetched whole (memmap can't span HTTP); the combined mp4 is opened as a
    seekable remote file so torchcodec only fetches the decoded frame's GOP (no download).
    """
    import io

    with fs.open(f"{ep_uri}/episode_metadata.json") as f:
        meta = json.load(f)
    T = int(meta["num_steps"])
    row = int(meta["state_dim"]) + int(meta["action_dim"])
    with fs.open(f"{ep_uri}/states_actions.bin", "rb") as f:
        sa = np.frombuffer(f.read(), dtype="<f8").reshape(T, row)
    ff = None
    spec = meta.get("frame_features")
    if spec:
        ff = {}
        for name, s in spec.items():
            with fs.open(f"{ep_uri}/ff_{name}.bin", "rb") as f:
                ff[name] = np.frombuffer(f.read(), dtype=s["dtype"]).reshape((T,) + tuple(s["shape"]))
    elif meta.get("frame_feature_keys"):  # legacy npz
        with fs.open(f"{ep_uri}/frame_features.npz", "rb") as f:
            ff = dict(np.load(io.BytesIO(f.read())))
    mp4 = fs.open(f"{ep_uri}/combined_camera-images-rgb.mp4", "rb") if with_video else None
    return _handle_from(meta, sa, mp4, keep=mp4, frame_features=ff, with_video=with_video)


def read_abcdl(in_dir: str) -> Episode:
    with open(os.path.join(in_dir, "episode_metadata.json")) as f:
        meta = json.load(f)
    T = int(meta["num_steps"])
    state_dim = int(meta["state_dim"])
    action_dim = int(meta["action_dim"])
    row = state_dim + action_dim

    sa = np.fromfile(os.path.join(in_dir, "states_actions.bin"), dtype="<f8").reshape(T, row)
    states, actions = sa[:, :state_dim], sa[:, state_dim:]

    stacked = _decode_all_frames(os.path.join(in_dir, "combined_camera-images-rgb.mp4"), T)
    stacked = np.transpose(stacked, (0, 2, 3, 1))  # (T, Hstack, W, C)
    names = list(meta["cameras"])
    h = stacked.shape[1] // len(names)

    tick_ns = int(meta.get("tick_ns", TICK_NS))
    t0 = int(meta.get("t0_ns", 0))
    ts = t0 + tick_ns * np.arange(T, dtype=np.int64)

    cams = {}
    for i, name in enumerate(names):
        frames = stacked[:, i * h:(i + 1) * h, :, :]
        w = frames.shape[2]
        cams[name] = CameraStream(frames=frames, timestamps=ts,
                                  width=w, height=h, codec="h264")
    em = EpisodeMeta(
        task=meta["task_name"], fps=1e9 / tick_ns, cameras=names,
        camera_resolutions={k: tuple(v) for k, v in meta["camera_resolutions"].items()},
        camera_codecs={k: "h264" for k in names},
        operator_id=meta.get("operator_id"), alignment=meta.get("alignment", "native"),
        t0_ns=t0, tick_ns=tick_ns, session_id=meta.get("session_id"))
    return Episode(states=states, actions=actions, timestamps=ts, cameras=cams, meta=em)
