"""SkillVLA training dataset — LeRobotDataset + Stage-2 skill-start jitter & decode.

On top of the standard (current-frame) sample, each item gains the VLM's *skill-start* inputs,
with transition-timing randomization (jitter) so the VLM is robust to the FSQ terminator firing
the skill transition slightly early/late at inference:

  skill_start_image        : 3rd-person frame decoded at the (jittered) skill start
  skill_start_wrist_image  : wrist frame at the same start
  skill_start_state        : observation.state at that start (from skill_initial_state.npz)
  skill_code               : the (jittered) skill's FSQ code (VLM target + action-expert teacher forcing)
  skill_effective_de       : distance to the jittered skill assignment's virtual end

Static ingredients come from build_data:
  parquet columns  : skill_index(k), skill_sequence(SS), skill_ds, skill_de, skill_initial_frame(IFS)
  ISS npz          : per-skill observation.state window (±pmax) — path & pmax read from info.json

The jitter decision is in `skill_jitter.choose_jitter`; here we decode the chosen frame (reusing the
reader's `_query_videos`, which handles v3.0 chunk→file→timestamp mapping) and pull the ISS state.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.skillVLA.foveated_augmentation import (
    FoveatedVisionAugmentationConfig,
    augment_camera_pair,
)
from lerobot.policies.skillVLA.skill_jitter import (
    choose_jitter,
    effective_jittered_skill_de,
    normalize_jitter_distribution,
    resolve_transition_jitter_pmaxes,
)

# Batch keys this dataset adds (the model + processor consume these).
SKILL_START_IMAGE = "skill_start_image"
SKILL_START_WRIST_IMAGE = "skill_start_wrist_image"
SKILL_START_STATE = "skill_start_state"
SKILL_CODE = "skill_code"
SKILL_CODE_TRUE = "skill_code_true"
SKILL_PROGRESS = "skill_progress"
SKILL_EFFECTIVE_DE = "skill_effective_de"
SKILL_PREVIOUS_ACTION = "skill_previous_action"
SKILL_PREVIOUS_ACTION_BOS = "skill_previous_action_bos"
SKILL_CANONICAL_ACTIONS = "skill_canonical_actions"
SKILL_CANONICAL_ACTION_IS_PAD = "skill_canonical_action_is_pad"
SKILL_CANONICAL_ACTION_LENGTH = "skill_canonical_action_length"
SKILL_FOCUS_UV = "skill_focus_uv"
SKILL_FOCUS_UV_PIXELS = "skill_focus_uv_pixels"
SKILL_FOCUS_VALID = "skill_focus_valid"
SKILL_FOCUS_CLIPPED = "skill_focus_clipped"
SAME_SKILL_PAIR_ID = "same_skill_pair_id"
SAME_SKILL_PAIR_FALLBACK = "same_skill_pair_fallback"
LATENT_SKILL_GROUP_ID = "latent_skill_group_id"

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


class _FocusUVStore:
    """Canonical per-skill endpoint focus, keyed by episode and skill rank."""

    def __init__(self, npz_path: str):
        with np.load(npz_path, allow_pickle=False) as z:
            required = {
                "episode_id",
                "skill_index",
                "frame_start",
                "focus_uv",
                "focus_uv_pixels",
                "focus_valid",
                "focus_clipped",
            }
            missing = sorted(required - set(z.files))
            if missing:
                raise ValueError(f"skill_focus_uv.npz is missing {missing}: {npz_path}")
            self.episode_id = np.asarray(z["episode_id"], dtype=np.int64)
            self.skill_index = np.asarray(z["skill_index"], dtype=np.int64)
            self.frame_start = np.asarray(z["frame_start"], dtype=np.int64)
            self.focus_uv = np.asarray(z["focus_uv"], dtype=np.float32)
            self.focus_uv_pixels = np.asarray(z["focus_uv_pixels"], dtype=np.int32)
            self.focus_valid = np.asarray(z["focus_valid"], dtype=np.bool_)
            self.focus_clipped = np.asarray(z["focus_clipped"], dtype=np.bool_)
        lengths = {
            "episode_id": len(self.episode_id),
            "skill_index": len(self.skill_index),
            "frame_start": len(self.frame_start),
            "focus_uv": len(self.focus_uv),
            "focus_uv_pixels": len(self.focus_uv_pixels),
            "focus_valid": len(self.focus_valid),
            "focus_clipped": len(self.focus_clipped),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"skill_focus_uv arrays have inconsistent lengths: {lengths}"
            )
        if self.focus_uv.shape[1:] != (2,) or self.focus_uv_pixels.shape[1:] != (2,):
            raise ValueError(
                "skill_focus_uv coordinates must have shape [num_skills, 2], got "
                f"uv={self.focus_uv.shape}, pixels={self.focus_uv_pixels.shape}."
            )
        order = np.lexsort((self.frame_start, self.episode_id))
        self.by_ep: dict[int, list[int]] = {}
        for index in order:
            self.by_ep.setdefault(int(self.episode_id[index]), []).append(int(index))

    def target(
        self,
        ep_idx: int,
        skill_rank: int,
        expected_frame_start: int,
    ) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        if ep_idx not in self.by_ep or not 0 <= skill_rank < len(self.by_ep[ep_idx]):
            raise KeyError(
                f"No focus target for episode={ep_idx}, skill={skill_rank}."
            )
        flat = self.by_ep[ep_idx][skill_rank]
        recorded_rank = int(self.skill_index[flat])
        recorded_start = int(self.frame_start[flat])
        if recorded_rank != skill_rank or recorded_start != expected_frame_start:
            raise ValueError(
                "Focus/IFS mismatch "
                f"(ep={ep_idx}, skill={skill_rank}): focus skill={recorded_rank}, "
                f"focus frame_start={recorded_start}, IFS={expected_frame_start}."
            )
        return (
            self.focus_uv[flat].copy(),
            self.focus_uv_pixels[flat].copy(),
            bool(self.focus_valid[flat]),
            bool(self.focus_clipped[flat]),
        )


class SkillVLADataset(LeRobotDataset):
    """LeRobotDataset that also yields the VLM's (jittered) skill-start image/state + skill code."""

    def __init__(self, *args, **kwargs):
        jitter_pmax_override = kwargs.pop("jitter_pmax", None)
        self._foveated_vision = FoveatedVisionAugmentationConfig.from_mapping(
            kwargs.pop("foveated_vision_config", None)
        )
        self._include_canonical_skill_actions = bool(
            kwargs.pop("include_canonical_skill_actions", False)
        )
        canonical_max_length_override = kwargs.pop(
            "canonical_skill_action_max_length", None
        )
        directional_overrides = {
            name: kwargs.pop(f"jitter_{name}_pmax", None)
            for name in ("early_start", "late_start", "early_end", "late_end")
        }
        self._include_predictor_start_inputs = bool(
            kwargs.pop("include_predictor_start_inputs", True)
        )
        if (
            self._foveated_vision.enabled
            and not self._include_predictor_start_inputs
        ):
            raise ValueError(
                "Foveated vision requires include_predictor_start_inputs=True so "
                "the focus target follows the selected jittered skill."
            )
        # Sample only episodes that actually have skills. The skill segmentation
        # (build_skill_dataset.py, min_skills=2) drops episodes with <2 detected skills, so they are
        # absent from the ISS npz and carry no Stage-2 supervision — but the LeRobot parquet still
        # keeps them (skill_sequence_len=1, IFS=-1), and sampling one would KeyError in _ISSStore.state.
        # We mirror the segmentation's decision by restricting `episodes` to the npz-covered set.
        valid = self._episodes_with_skills(args, kwargs)
        requested = kwargs.get("episodes")
        kwargs["episodes"] = sorted(valid) if requested is None else [e for e in requested if e in valid]
        super().__init__(*args, **kwargs)
        info = self.meta.info
        self._canonical_skill_action_max_length = int(
            info.get("skill_observed_max_length", 0)
            if canonical_max_length_override is None
            else canonical_max_length_override
        )
        if self._include_canonical_skill_actions:
            dataset_max_length = int(info.get("skill_observed_max_length", 0))
            if dataset_max_length <= 0:
                raise ValueError(
                    "arch0_skill requires a positive skill_observed_max_length "
                    "in the dataset info.json."
                )
            if self._canonical_skill_action_max_length != dataset_max_length:
                raise ValueError(
                    "Configured canonical skill-action length does not match the "
                    "dataset contract: "
                    f"configured={self._canonical_skill_action_max_length}, "
                    f"dataset={dataset_max_length}."
                )
            if not self._include_predictor_start_inputs:
                raise ValueError(
                    "Canonical skill actions require jitter resolution and therefore "
                    "include_predictor_start_inputs=True."
                )
            values = (
                self.hf_dataset.select_columns(["action"])
                .with_format("numpy")[:]["action"]
            )
            self._canonical_action_cache = (
                torch.from_numpy(np.asarray(values, dtype=np.float32))
                .clone()
                .contiguous()
            )
        else:
            self._canonical_action_cache = None
        iss_path = self._resolve_iss_path(info.get("skill_initial_state_path"), self.root)
        self._iss = (
            _ISSStore(iss_path) if self._include_predictor_start_inputs else None
        )
        focus_path = self._resolve_optional_companion_path(
            info.get("skill_focus_uv_path"), self.root
        )
        self._focus_uv = _FocusUVStore(focus_path) if focus_path is not None else None
        if self._foveated_vision.enabled and self._focus_uv is None:
            raise ValueError(
                "Foveated vision is enabled, but the SkillVLA dataset metadata has "
                "no skill_focus_uv_path. Rebuild the dataset with focus_uv.enabled=true."
            )
        default_pmax = self._iss.pmax if self._iss is not None else 0
        dataset_pmax = int(info.get("skill_pmax", default_pmax))
        if self._iss is not None and dataset_pmax != self._iss.pmax:
            raise ValueError(
                "Dataset skill_pmax does not match the ISS window: "
                f"info.json={dataset_pmax}, ISS={self._iss.pmax}."
            )
        dataset_directional = {
            name: int(info.get(f"skill_jitter_{name}_pmax", dataset_pmax))
            for name in ("early_start", "late_start", "early_end", "late_end")
        }
        if any(
            value is not None and int(value) >= 0
            for value in directional_overrides.values()
        ):
            runtime_directional = dict(
                zip(
                    dataset_directional,
                    resolve_transition_jitter_pmaxes(
                        dataset_pmax if jitter_pmax_override is None else int(jitter_pmax_override),
                        early_start_pmax=directional_overrides["early_start"],
                        late_start_pmax=directional_overrides["late_start"],
                        early_end_pmax=directional_overrides["early_end"],
                        late_end_pmax=directional_overrides["late_end"],
                    ),
                    strict=True,
                )
            )
        elif jitter_pmax_override is not None:
            # Historical callers overriding the scalar contract still mean one
            # common runtime window in all four directions.
            runtime_directional = {
                name: int(jitter_pmax_override) for name in dataset_directional
            }
        else:
            runtime_directional = dataset_directional
        oversized = {
            name: value
            for name, value in runtime_directional.items()
            if value < 0 or value > dataset_directional[name]
        }
        if oversized:
            raise ValueError(
                "Requested transition jitter directional windows must be within "
                f"the built dataset contract {dataset_directional}, got {oversized}."
            )
        self._directional_pmaxes = runtime_directional
        self._pmax = max(runtime_directional.values())
        if self._pmax > dataset_pmax:
            raise ValueError(
                "Requested transition jitter exceeds the prebuilt ISS window "
                f"[0, {dataset_pmax}], got {self._pmax}."
            )
        self._jitter_distribution = normalize_jitter_distribution(
            info.get("skill_jitter_distribution", "half_normal"))

    @property
    def jitter_pmax(self) -> int:
        return self._pmax

    @property
    def jitter_directional_pmaxes(self) -> dict[str, int]:
        return dict(self._directional_pmaxes)

    @property
    def jitter_distribution(self) -> str:
        return self._jitter_distribution

    @property
    def canonical_skill_action_max_length(self) -> int:
        return self._canonical_skill_action_max_length

    def _canonical_skill_actions(
        self,
        *,
        episode_index: int,
        skill_start_frame: int,
        skill_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the selected skill's original (non-jittered) padded trajectory."""
        if self._canonical_action_cache is None:
            raise RuntimeError("Canonical skill-action caching is disabled.")
        if not 0 < skill_length <= self._canonical_skill_action_max_length:
            raise ValueError(
                "Canonical skill length is outside the dataset contract: "
                f"length={skill_length}, max={self._canonical_skill_action_max_length}."
            )
        episode = self.meta.episodes[episode_index]
        episode_length = _scalar(episode["length"])
        if skill_start_frame < 0 or skill_start_frame + skill_length > episode_length:
            raise ValueError(
                "Canonical skill range leaves its episode: "
                f"episode={episode_index}, start={skill_start_frame}, "
                f"length={skill_length}, episode_length={episode_length}."
            )
        absolute_start = _scalar(episode["dataset_from_index"]) + skill_start_frame
        absolute_indices = range(absolute_start, absolute_start + skill_length)
        mapping = self.reader._absolute_to_relative_idx  # noqa: SLF001
        relative_indices = (
            list(absolute_indices)
            if mapping is None
            else [mapping[index] for index in absolute_indices]
        )
        selected = self._canonical_action_cache.index_select(
            0, torch.as_tensor(relative_indices, dtype=torch.long)
        )
        padded = selected.new_zeros(
            self._canonical_skill_action_max_length, selected.shape[-1]
        )
        padded[:skill_length] = selected
        is_pad = torch.ones(
            self._canonical_skill_action_max_length, dtype=torch.bool
        )
        is_pad[:skill_length] = False
        return padded, is_pad

    @staticmethod
    def _resolve_iss_path(iss_path: str | None, root) -> str:
        """info.json의 ``skill_initial_state_path`` 해석. 다른 서버에서 빌드된 데이터셋은 그
        서버의 절대경로가 박혀 있으므로, 존재하지 않으면 run 폴더(= dataset root의 부모)의
        동명 파일로 폴백한다 (FSQ.resolve_image_model_path와 같은 이전(移轉) 면역 패턴)."""
        if not iss_path:
            raise ValueError(
                "Dataset info.json has no 'skill_initial_state_path'. Rebuild the dataset with the "
                "updated add_skill_latents_to_dataset.py (Stage-2 schema)."
            )
        if Path(iss_path).exists():
            return str(iss_path)
        local = Path(root).resolve().parent / Path(iss_path).name
        if local.exists():
            return str(local)
        raise FileNotFoundError(
            f"skill_initial_state npz not found at the recorded path ({iss_path}) "
            f"nor at the run-dir fallback ({local})."
        )

    @staticmethod
    def _resolve_optional_companion_path(path: str | None, root) -> str | None:
        """Resolve an optional run-level artifact across server path changes."""
        if not path:
            return None
        recorded = Path(path)
        if recorded.is_file():
            return str(recorded)
        local = Path(root).resolve().parent / recorded.name
        if local.is_file():
            return str(local)
        raise FileNotFoundError(
            f"Optional SkillVLA artifact not found at the recorded path ({recorded}) "
            f"nor at the run-dir fallback ({local})."
        )

    @staticmethod
    def _episodes_with_skills(args, kwargs) -> set[int]:
        """Episode indices present in the ISS npz (= have >=1 skill), read before super().__init__
        so they can be passed as the `episodes` subset. Same npz that _ISSStore later consumes.

        A built SkillVLA dataset may additionally declare complete episodes as
        training-excluded. Keeping that contract in info.json lets every
        predictor/terminator/action loader apply it without duplicating YAML
        episode lists or physically reindexing the source dataset.
        """
        repo_id = args[0] if args else kwargs["repo_id"]
        meta = LeRobotDatasetMetadata(repo_id, root=kwargs.get("root"), revision=kwargs.get("revision"))
        iss_path = SkillVLADataset._resolve_iss_path(meta.info.get("skill_initial_state_path"), meta.root)
        with np.load(iss_path) as z:
            valid = {int(e) for e in np.unique(np.asarray(z["episode_id"]))}
        excluded = {
            int(episode_id)
            for episode_id in meta.info.get("training_excluded_episode_ids", [])
        }
        invalid = excluded - set(range(meta.total_episodes))
        if invalid:
            raise ValueError(
                "Dataset training_excluded_episode_ids contains invalid ids: "
                f"{sorted(invalid)} (total_episodes={meta.total_episodes})."
            )
        return valid - excluded

    def __getitem__(self, idx) -> dict:
        pair_id = -1
        pair_fallback = False
        latent_skill_group_id = -1
        effective_de_override = None
        jitter_override = None
        if isinstance(idx, tuple):
            if len(idx) == 3:
                idx, pair_id, pair_fallback = idx
            elif len(idx) == 5:
                idx, pair_id, pair_fallback, kp_override, offset_override = idx
                jitter_override = (int(kp_override), int(offset_override))
            elif len(idx) == 6:
                (
                    idx,
                    pair_id,
                    pair_fallback,
                    kp_override,
                    offset_override,
                    latent_skill_group_id,
                ) = idx
                jitter_override = (int(kp_override), int(offset_override))
            elif len(idx) == 7:
                (
                    idx,
                    pair_id,
                    pair_fallback,
                    kp_override,
                    offset_override,
                    latent_skill_group_id,
                    effective_de_override,
                ) = idx
                jitter_override = (int(kp_override), int(offset_override))
                effective_de_override = int(effective_de_override)
                if effective_de_override < 0:
                    raise ValueError(
                        "Sampler-provided effective_de must be non-negative, "
                        f"got {effective_de_override}."
                    )
            else:
                raise ValueError(
                    "Expected grouped sample index (index, pair_id, fallback"
                    "[, k_prime, offset[, latent_group[, effective_de]]]), "
                    f"got {idx!r}."
                )
        item_index = int(idx)
        item = super().__getitem__(item_index)

        # Terminator context is the action that produced the current
        # observation. Episode starts receive an exact raw zero placeholder;
        # the terminator adapter turns it into an exact normalized BOS token.
        raw_row = self.hf_dataset[item_index]
        current_action = np.asarray(raw_row["action"], dtype=np.float32)
        frame_index = _scalar(item["frame_index"])
        if frame_index == 0:
            previous_action = np.zeros_like(current_action)
        else:
            previous_action = np.asarray(
                self.hf_dataset[item_index - 1]["action"], dtype=np.float32
            )
        item[SKILL_PREVIOUS_ACTION] = torch.from_numpy(previous_action.copy())
        item[SKILL_PREVIOUS_ACTION_BOS] = torch.tensor(
            frame_index == 0, dtype=torch.bool
        )

        ep_idx = _scalar(item["episode_index"])
        k = _scalar(item["skill_index"])
        ds = _scalar(item["skill_ds"])
        de = _scalar(item["skill_de"])
        seq_len = _scalar(item["skill_sequence_len"])
        ss = np.asarray(item["skill_sequence"]).reshape(-1)
        ifs = np.asarray(item["skill_initial_frame"]).reshape(-1)
        # TRUE current skill's code (un-jittered) — the FSQ terminator co-training (FT) conditions on
        # the actual skill the current frame belongs to, with progress/termination from its ds/de.
        item[SKILL_CODE_TRUE] = torch.tensor(int(ss[k]), dtype=torch.long)
        item[SKILL_EFFECTIVE_DE] = torch.tensor(de, dtype=torch.long)

        reader = None
        ep_len = _scalar(self.meta.episodes[ep_idx]["length"])
        predictor_start_frame = None
        predictor_start_images = None
        if self._include_predictor_start_inputs:
            reader = self._ensure_reader()
            if jitter_override is None:
                kp, offset = choose_jitter(
                    k,
                    ds,
                    de,
                    seq_len,
                    self._pmax,
                    distribution=self._jitter_distribution,
                    early_start_pmax=self._directional_pmaxes["early_start"],
                    late_start_pmax=self._directional_pmaxes["late_start"],
                    early_end_pmax=self._directional_pmaxes["early_end"],
                    late_end_pmax=self._directional_pmaxes["late_end"],
                )
            else:
                kp, offset = jitter_override
                if not 0 <= kp < seq_len - 1 or not -self._pmax <= offset <= self._pmax:
                    raise ValueError(
                        f"Invalid sampler-provided jitter (k'={kp}, offset={offset}) for "
                        f"seq_len={seq_len}, pmax={self._pmax}."
                    )
            skill_code = int(ss[kp])
            gt_start = int(ifs[kp])
            predictor_start_frame = int(np.clip(gt_start + offset, 0, ep_len - 1))
            start_ts = predictor_start_frame / self.fps
            predictor_start_images = reader._query_videos(  # noqa: SLF001
                {CAM_3RD: [start_ts], CAM_WRIST: [start_ts]}, ep_idx
            )
            if reader._image_transforms is not None:  # noqa: SLF001
                predictor_start_images = {
                    camera: reader._image_transforms(image)  # noqa: SLF001
                    for camera, image in predictor_start_images.items()
                }

            if self._iss is None:
                raise RuntimeError("Predictor start inputs require the ISS store.")
            # The ISS array was built with the dataset's original pmax. A
            # smaller runtime jitter window must still index around that center.
            iss_center = self._iss.pmax
            iss_index = int(np.clip(iss_center + offset, 0, 2 * iss_center))
            start_state = self._iss.state(ep_idx, kp, iss_index, gt_start)
            item[SKILL_START_IMAGE] = predictor_start_images[CAM_3RD]
            item[SKILL_START_WRIST_IMAGE] = predictor_start_images[CAM_WRIST]
            item[SKILL_START_STATE] = torch.from_numpy(start_state)
            item[SKILL_CODE] = torch.tensor(skill_code, dtype=torch.long)
            focus_uv_tensor = None
            if self._focus_uv is not None:
                focus_uv, focus_pixels, focus_valid, focus_clipped = (
                    self._focus_uv.target(ep_idx, kp, gt_start)
                )
                focus_uv_tensor = torch.from_numpy(focus_uv)
                item[SKILL_FOCUS_UV] = focus_uv_tensor
                item[SKILL_FOCUS_UV_PIXELS] = torch.from_numpy(focus_pixels)
                item[SKILL_FOCUS_VALID] = torch.tensor(
                    focus_valid, dtype=torch.bool
                )
                item[SKILL_FOCUS_CLIPPED] = torch.tensor(
                    focus_clipped, dtype=torch.bool
                )
            if (
                self._foveated_vision.enabled
                or self._foveated_vision.randomization_enabled
            ):
                # The current VSA camera pair is augmented independently on
                # every sampled frame. Color and input blur share one draw
                # across top/wrist; crop jitter and endpoint foveation are
                # top-view only. Predictor skill-start frames stay intact.
                item[CAM_3RD], item[CAM_WRIST] = augment_camera_pair(
                    item[CAM_3RD],
                    item[CAM_WRIST],
                    focus_uv_tensor,
                    self._foveated_vision,
                )

            # GT progress of the current frame within the chosen (possibly
            # jittered) predictor skill.
            lens = np.asarray(item["skill_length_sequence"]).reshape(-1)
            if self._include_canonical_skill_actions:
                canonical_actions, canonical_is_pad = self._canonical_skill_actions(
                    episode_index=ep_idx,
                    skill_start_frame=gt_start,
                    skill_length=int(lens[kp]),
                )
                item[SKILL_CANONICAL_ACTIONS] = canonical_actions
                item[SKILL_CANONICAL_ACTION_IS_PAD] = canonical_is_pad
                item[SKILL_CANONICAL_ACTION_LENGTH] = torch.tensor(
                    int(lens[kp]), dtype=torch.long
                )
            current_frame = int(ifs[k]) + int(ds)
            effective_de = (
                effective_de_override
                if effective_de_override is not None
                else effective_jittered_skill_de(
                    k=k,
                    k_prime=kp,
                    ds=ds,
                    de=de,
                    skill_initial_frames=ifs,
                    skill_lengths=lens,
                    offset=offset,
                )
            )
            item[SKILL_EFFECTIVE_DE] = torch.tensor(
                effective_de, dtype=torch.long
            )
            progress = (current_frame - int(ifs[kp])) / max(int(lens[kp]) - 1, 1)
            item[SKILL_PROGRESS] = torch.tensor(
                float(np.clip(progress, 0.0, 1.0)), dtype=torch.float32
            )
            item[SAME_SKILL_PAIR_ID] = torch.tensor(int(pair_id), dtype=torch.long)
            item[SAME_SKILL_PAIR_FALLBACK] = torch.tensor(
                bool(pair_fallback), dtype=torch.bool
            )
            item[LATENT_SKILL_GROUP_ID] = torch.tensor(
                int(latent_skill_group_id), dtype=torch.long
            )

        return item
