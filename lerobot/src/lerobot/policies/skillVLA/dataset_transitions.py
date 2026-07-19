"""STAGE-3 / FT-replay transition dataset — segment-level samples from a prebuilt transitions.npz.

The VLM only runs at skill TRANSITIONS at deployment (predict once per segment, then cached), so the
skill-prediction stages need exactly one training pair per segment: (jittered skill-start images/state,
language) → FSQ code. This dataset serves those pairs straight from the transition pack
(build_data/src/build_transition_pack.py) — no episode-video seeks, segment-uniform sampling (= the
per-transition deployment distribution, instead of frame-uniform's skill-length weighting).

Subclasses SkillVLADataset so the meta/stats/feature plumbing (normalizers, wandb dataset info) is
unchanged; only __len__/__getitem__ are pack-driven. The unused motor-side keys (current obs, state,
actions) are filled with cheap dummies — the SKILL forward path never reads them (and the ones that
ARE formally touched, e.g. actions for the batch size, only need the right shape).

Multiple packs concatenate (FT replay: pass the parent PT pack alongside the new task's) — language
is stored as strings per pack, so packs from different tasks merge freely.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.skillVLA.dataset_skillVLA import (
    CAM_3RD,
    CAM_WRIST,
    SKILL_CODE,
    SKILL_CODE_TRUE,
    SKILL_PROGRESS,
    SKILL_START_IMAGE,
    SKILL_START_STATE,
    SKILL_START_WRIST_IMAGE,
    SkillVLADataset,
)
from lerobot.policies.skillVLA.skill_jitter import sample_offset
from lerobot.policies.skillVLA.skill_jitter import normalize_jitter_distribution


class _Pack:
    """One transitions.npz: flat JPEG buffers + offsets, per-segment code/task/provenance."""

    def __init__(self, path: str | Path):
        z = np.load(str(path))
        self.jpeg = {CAM_3RD: np.asarray(z["jpeg_3rd"]), CAM_WRIST: np.asarray(z["jpeg_wrist"])}
        self.off = {CAM_3RD: np.asarray(z["off_3rd"]), CAM_WRIST: np.asarray(z["off_wrist"])}
        self.skill_code = np.asarray(z["skill_code"])
        if ("start_state" not in z or "jitter_distribution" not in z
                or int(z.get("schema_version", 0)) < 3):
            raise ValueError(
                f"Transition pack {path} predates configurable jitter support. Rebuild it with "
                "build_data/src/build_transition_pack.py."
            )
        self.start_state = np.asarray(z["start_state"], dtype=np.float32)
        self.task_index = np.asarray(z["task_index"])
        self.tasks = [str(t) for t in np.asarray(z["tasks"])]
        self.pmax = int(z["pmax"])
        self.jitter_distribution = normalize_jitter_distribution(str(z["jitter_distribution"]))
        self.n = int(self.skill_code.shape[0])
        self.win = 2 * self.pmax + 1
        assert self.off[CAM_3RD].shape[0] == self.n * self.win + 1, f"corrupt pack: {path}"
        assert self.start_state.shape[:2] == (self.n, self.win), f"corrupt start_state: {path}"

    def image(self, cam: str, seg: int, win_idx: int) -> torch.Tensor:
        from PIL import Image  # noqa: PLC0415

        flat = seg * self.win + win_idx
        raw = self.jpeg[cam][self.off[cam][flat]: self.off[cam][flat + 1]]
        arr = np.asarray(Image.open(io.BytesIO(raw.tobytes())).convert("RGB")).copy()
        return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0     # CHW float [0,1], native res


class SkillTransitionDataset(SkillVLADataset):
    """Segment-level (transition) samples for pt_stage="skill" / FT SKILL batches.

    ``transition_packs``: one or more transitions.npz paths — index = concatenation. Each __getitem__
    picks an offset in [-pmax, +pmax] from the pack's configured distribution and decodes the two
    skill-start JPEGs for that offset; everything motor-side is a dummy."""

    def __init__(self, *args, transition_packs: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)                    # full meta/ISS plumbing of the base dataset
        paths = [p for p in (transition_packs or []) if str(p).strip()]
        if not paths:
            raise ValueError("SkillTransitionDataset needs at least one transitions.npz path.")
        self._packs = [_Pack(p) for p in paths]
        self._cum = np.cumsum([p.n for p in self._packs])
        # dummy shapes from the base dataset's features / the action delta window
        self._state_dim = int(np.prod(self.meta.features["observation.state"]["shape"]))
        n_act = len(self.delta_timestamps["action"]) if self.delta_timestamps and "action" in self.delta_timestamps else 1
        self._act_shape = (n_act, int(np.prod(self.meta.features["action"]["shape"])))
        print(f"[transition-dataset] {len(self._packs)} pack(s), {int(self._cum[-1])} segments "
              f"(window ±{self._packs[0].pmax})")

    def __len__(self) -> int:
        return int(self._cum[-1])

    def __getitem__(self, idx) -> dict:
        idx = int(idx)
        pi = int(np.searchsorted(self._cum, idx, side="right"))
        pack = self._packs[pi]
        seg = idx - (int(self._cum[pi - 1]) if pi > 0 else 0)

        offset = sample_offset(pack.pmax, distribution=pack.jitter_distribution)
        win_idx = pack.pmax + offset
        img3 = pack.image(CAM_3RD, seg, win_idx)
        imgw = pack.image(CAM_WRIST, seg, win_idx)
        start_state = torch.from_numpy(pack.start_state[seg, win_idx].copy())
        code = int(pack.skill_code[seg])

        item: dict = {
            SKILL_START_IMAGE: img3,
            SKILL_START_WRIST_IMAGE: imgw,
            SKILL_START_STATE: start_state,                            # prompt state at the SAME offset
            SKILL_CODE: torch.tensor(code, dtype=torch.long),
            SKILL_CODE_TRUE: torch.tensor(code, dtype=torch.long),
            SKILL_PROGRESS: torch.tensor(0.0),
            # language: the pack's own task string (tokenized by the policy preprocessor)
            "task": pack.tasks[int(pack.task_index[seg])],
            "task_index": torch.tensor(int(pack.task_index[seg]), dtype=torch.long),
            # motor-side dummies (formally touched for shapes only — never supervised on SKILL batches)
            CAM_3RD: img3,                                            # cond image slot (unused content)
            CAM_WRIST: imgw,
            "observation.state": torch.zeros(self._state_dim),
            "action": torch.zeros(self._act_shape),
            "action_is_pad": torch.zeros(self._act_shape[0], dtype=torch.bool),
            # bookkeeping
            "episode_index": torch.tensor(0, dtype=torch.long),
            "frame_index": torch.tensor(0, dtype=torch.long),
            "timestamp": torch.tensor(0.0),
            "index": torch.tensor(idx, dtype=torch.long),
        }
        return item
