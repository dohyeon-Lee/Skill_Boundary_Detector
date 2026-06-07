"""SkillVLA training dataset — LeRobotDataset + Stage-2 skill-start jitter & decode.

On top of the standard (current-frame) sample, each item gains the VLM's *skill-start* inputs,
with transition-timing randomization (jitter) so the VLM is robust to the FSQ terminator firing
the skill transition slightly early/late at inference:

  skill_start_image        : 3rd-person frame decoded at the (jittered) skill start
  skill_start_wrist_image  : wrist frame at the same start
  skill_start_state        : observation.state at that start (from skill_initial_state.npz)
  skill_code               : the (jittered) skill's FSQ code (VLM target + action-expert teacher forcing)

Static ingredients come from build_data:
  parquet columns  : skill_index(k), skill_sequence(SS), skill_ds, skill_de, skill_initial_frame(IFS)
  ISS npz          : per-skill observation.state window (±pmax) — path & pmax read from info.json

The jitter decision is in `skill_jitter.choose_jitter`; here we decode the chosen frame (reusing the
reader's `_query_videos`, which handles v3.0 chunk→file→timestamp mapping) and pull the ISS state.
"""

from __future__ import annotations

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.skillVLA.skill_jitter import choose_jitter

# Batch keys this dataset adds (the model + processor consume these).
SKILL_START_IMAGE = "skill_start_image"
SKILL_START_WRIST_IMAGE = "skill_start_wrist_image"
SKILL_START_STATE = "skill_start_state"
SKILL_CODE = "skill_code"

CAM_3RD = "observation.images.image"
CAM_WRIST = "observation.images.wrist_image"


def _scalar(x) -> int:
    return int(x.item() if torch.is_tensor(x) else np.asarray(x).reshape(-1)[0])


class _ISSStore:
    """skill_initial_state.npz reader: per-skill observation.state window (±pmax), keyed by episode_id.

    Mirrors the skill_latents.npz convention (flat per-skill arrays). `frame_start` is cross-checked
    against the parquet IFS so a wrong episode/skill alignment fails loudly instead of silently."""

    def __init__(self, npz_path: str):
        z = np.load(npz_path)
        self.frame_start = np.asarray(z["frame_start"])
        self.windows = np.asarray(z["iss_windows"])  # (total_skills, 2*pmax+1, state_dim)
        self.pmax = int(z["pmax"])
        epid = np.asarray(z["episode_id"])
        order = np.lexsort((self.frame_start, epid))  # group by episode, then sort by frame_start (= skill order)
        self.by_ep: dict[int, list[int]] = {}
        for i in order:
            self.by_ep.setdefault(int(epid[i]), []).append(int(i))

    def state(self, ep_idx: int, skill_rank: int, iss_index: int, expected_frame_start: int) -> np.ndarray:
        flat = self.by_ep[ep_idx][skill_rank]
        fs = int(self.frame_start[flat])
        if fs != expected_frame_start:
            raise ValueError(
                f"ISS/IFS mismatch (ep={ep_idx}, skill={skill_rank}): npz frame_start={fs} != IFS={expected_frame_start}"
            )
        return self.windows[flat][iss_index].astype(np.float32)


class SkillVLADataset(LeRobotDataset):
    """LeRobotDataset that also yields the VLM's (jittered) skill-start image/state + skill code."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        info = self.meta.info
        iss_path = info.get("skill_initial_state_path")
        if not iss_path:
            raise ValueError(
                "Dataset info.json has no 'skill_initial_state_path'. Rebuild the dataset with the "
                "updated add_skill_latents_to_dataset.py (Stage-2 schema)."
            )
        self._iss = _ISSStore(iss_path)
        self._pmax = int(info.get("skill_pmax", self._iss.pmax))

    def __getitem__(self, idx) -> dict:
        item = super().__getitem__(idx)
        reader = self._ensure_reader()

        ep_idx = _scalar(item["episode_index"])
        k = _scalar(item["skill_index"])
        ds = _scalar(item["skill_ds"])
        de = _scalar(item["skill_de"])
        seq_len = _scalar(item["skill_sequence_len"])
        ss = np.asarray(item["skill_sequence"]).reshape(-1)
        ifs = np.asarray(item["skill_initial_frame"]).reshape(-1)

        # 1) pick the (possibly jittered) skill + start-frame offset
        kp, offset = choose_jitter(k, ds, de, seq_len, self._pmax)
        skill_code = int(ss[kp])
        gt_start = int(ifs[kp])

        # 2) decode the start frame's images (clamp to the episode)
        ep_len = _scalar(self.meta.episodes[ep_idx]["length"])
        start_frame = int(np.clip(gt_start + offset, 0, ep_len - 1))
        start_ts = start_frame / self.fps
        start_imgs = reader._query_videos({CAM_3RD: [start_ts], CAM_WRIST: [start_ts]}, ep_idx)  # noqa: SLF001
        if reader._image_transforms is not None:  # noqa: SLF001  (match current-frame transforms)
            start_imgs = {c: reader._image_transforms(v) for c, v in start_imgs.items()}  # noqa: SLF001

        # 3) start state from the ISS window (offset is clamped into the window by construction)
        iss_index = int(np.clip(self._pmax + offset, 0, 2 * self._pmax))
        start_state = self._iss.state(ep_idx, kp, iss_index, gt_start)

        item[SKILL_START_IMAGE] = start_imgs[CAM_3RD]
        item[SKILL_START_WRIST_IMAGE] = start_imgs[CAM_WRIST]
        item[SKILL_START_STATE] = torch.from_numpy(start_state)
        item[SKILL_CODE] = torch.tensor(skill_code, dtype=torch.long)
        return item
