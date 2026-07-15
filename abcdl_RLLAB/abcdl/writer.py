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

    def save(self, task: str, operator_id: Optional[str] = None,
             frame_features: Optional[dict] = None) -> dict:
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
        ep = Episode(np.stack(self._states), np.stack(self._actions), ts, cams, meta,
                     frame_features=frame_features)
        out = {}
        if "abcdl" in self.formats:
            write_abcdl(ep, self.out_dir); out["abcdl"] = self.out_dir
        if "mcap" in self.formats:
            os.makedirs(self.out_dir, exist_ok=True)
            p = os.path.join(self.out_dir, "episode.mcap")
            write_mcap(ep, p); out["mcap"] = p
        return out
