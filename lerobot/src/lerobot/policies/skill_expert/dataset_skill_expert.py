"""SkillExpert (Stage-1) training dataset — LeRobotDataset + the skill's END-frame inputs.

On top of the standard (current-frame) sample, each item gains the connector's *skill-END* inputs
(the future/goal the action expert must reach), decoded live so the connector can DINO-encode them:

  skill_end_image        : 3rd-person frame at the current skill's last frame (fe-1)
  skill_end_wrist_image  : wrist frame at the same end
  skill_end_state        : observation.state at that end (from skill_final_state.npz / ESS)

Static ingredients come from build_data:
  parquet columns  : skill_index(k), skill_initial_frame(IFS), skill_length_sequence
  ESS npz          : per-skill observation.state window (±pmax) centered on the end frame —
                     path & pmax read from info.json (built by add_skill_latents_to_dataset.py)

The end frame is decoded with the reader's ``_query_videos`` (v3.0 chunk→file→timestamp mapping),
mirroring SkillVLADataset's skill-START path. Transition randomization is NOT applied here yet
(offset=0 → window center = the GT end); the ESS window is pre-built ±pmax so it can be enabled
later (ess_index = pmax + offset) without a rebuild.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Batch keys this dataset adds (the policy's forward consumes these for the connector).
SKILL_END_IMAGE = "skill_end_image"
SKILL_END_WRIST_IMAGE = "skill_end_wrist_image"
SKILL_END_STATE = "skill_end_state"

CAM_3RD = "observation.images.image"
CAM_WRIST = "observation.images.wrist_image"


def _scalar(x) -> int:
    return int(x.item() if torch.is_tensor(x) else np.asarray(x).reshape(-1)[0])


class _ESSStore:
    """skill_final_state.npz reader: per-skill observation.state window (±pmax) centered on the end
    frame, keyed by episode_id. Mirrors _ISSStore (skill order = frame_start order). ``frame_end`` is
    cross-checked against the parquet IFS[k]+length-1 so a wrong alignment fails loudly."""

    def __init__(self, npz_path: str):
        z = np.load(npz_path)
        self.frame_end = np.asarray(z["frame_end"])
        self.windows = np.asarray(z["ess_windows"])  # (total_skills, 2*pmax+1, state_dim)
        self.pmax = int(z["pmax"])
        epid = np.asarray(z["episode_id"])
        # group by episode, then sort by frame_end (= skill order, same as frame_start order)
        order = np.lexsort((self.frame_end, epid))
        self.by_ep: dict[int, list[int]] = {}
        for i in order:
            self.by_ep.setdefault(int(epid[i]), []).append(int(i))

    def state(self, ep_idx: int, skill_rank: int, ess_index: int, expected_frame_end: int) -> np.ndarray:
        flat = self.by_ep[ep_idx][skill_rank]
        fe = int(self.frame_end[flat])
        if fe != expected_frame_end:
            raise ValueError(
                f"ESS/IFS mismatch (ep={ep_idx}, skill={skill_rank}): npz frame_end={fe} "
                f"!= IFS+len-1={expected_frame_end}"
            )
        return self.windows[flat][ess_index].astype(np.float32)


class SkillExpertDataset(LeRobotDataset):
    """LeRobotDataset that also yields the current skill's END image/state (connector inputs)."""

    def __init__(self, *args, **kwargs):
        # Only sample episodes that actually have skills (same reasoning as SkillVLADataset): the
        # segmentation drops <2-skill episodes, which are absent from the ESS npz; sampling one would
        # KeyError in _ESSStore. Restrict `episodes` to the npz-covered set.
        valid = self._episodes_with_skills(args, kwargs)
        requested = kwargs.get("episodes")
        kwargs["episodes"] = sorted(valid) if requested is None else [e for e in requested if e in valid]
        super().__init__(*args, **kwargs)
        info = self.meta.info
        ess_path = self._resolve_ess_path(info.get("skill_final_state_path"), self.root)
        self._ess = _ESSStore(ess_path)
        self._pmax = int(info.get("skill_pmax", self._ess.pmax))

    @staticmethod
    def _resolve_ess_path(ess_path: str | None, root) -> str:
        """info.json의 ``skill_final_state_path`` 해석. 다른 서버에서 빌드된 데이터셋의 절대경로가
        존재하지 않으면 run 폴더(= dataset root의 부모)의 동명 파일로 폴백한다 (ISS와 동일 패턴)."""
        if not ess_path:
            raise ValueError(
                "Dataset info.json has no 'skill_final_state_path'. Rebuild the dataset with the "
                "updated add_skill_latents_to_dataset.py (Stage-1 ESS schema)."
            )
        if Path(ess_path).exists():
            return str(ess_path)
        local = Path(root).resolve().parent / Path(ess_path).name
        if local.exists():
            return str(local)
        raise FileNotFoundError(
            f"skill_final_state npz not found at the recorded path ({ess_path}) "
            f"nor at the run-dir fallback ({local})."
        )

    @staticmethod
    def _episodes_with_skills(args, kwargs) -> set[int]:
        """Episode indices present in the ESS npz (= have >=1 skill), read before super().__init__."""
        repo_id = args[0] if args else kwargs["repo_id"]
        meta = LeRobotDatasetMetadata(repo_id, root=kwargs.get("root"), revision=kwargs.get("revision"))
        ess_path = SkillExpertDataset._resolve_ess_path(meta.info.get("skill_final_state_path"), meta.root)
        with np.load(ess_path) as z:
            return {int(e) for e in np.unique(np.asarray(z["episode_id"]))}

    def __getitem__(self, idx) -> dict:
        item = super().__getitem__(idx)
        reader = self._ensure_reader()

        ep_idx = _scalar(item["episode_index"])
        k = _scalar(item["skill_index"])                              # 0-based current skill
        ifs = np.asarray(item["skill_initial_frame"]).reshape(-1)
        lens = np.asarray(item["skill_length_sequence"]).reshape(-1)

        # End frame of the current skill = IFS[k] + length[k] - 1 (== fe-1). Robust to the "leftover"
        # frames (de=0 past the skill end) where current_frame + skill_de would be wrong.
        gt_end = int(ifs[k]) + int(lens[k]) - 1
        ep_len = _scalar(self.meta.episodes[ep_idx]["length"])
        end_frame = int(np.clip(gt_end, 0, ep_len - 1))
        end_ts = end_frame / self.fps

        end_imgs = reader._query_videos({CAM_3RD: [end_ts], CAM_WRIST: [end_ts]}, ep_idx)  # noqa: SLF001
        if reader._image_transforms is not None:  # noqa: SLF001  (match current-frame transforms)
            end_imgs = {c: reader._image_transforms(v) for c, v in end_imgs.items()}  # noqa: SLF001

        # End state from the ESS window. offset=0 (center) for now — transition randomization, when
        # added, will pass ess_index = pmax + offset (the window is pre-built for it).
        end_state = self._ess.state(ep_idx, k, self._pmax, gt_end)

        item[SKILL_END_IMAGE] = end_imgs[CAM_3RD]
        item[SKILL_END_WRIST_IMAGE] = end_imgs[CAM_WRIST]
        item[SKILL_END_STATE] = torch.from_numpy(end_state)
        return item
