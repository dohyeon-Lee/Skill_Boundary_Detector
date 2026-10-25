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
    causal_skill_start,
    choose_jitter,
    choose_jitter_percent,
    effective_jittered_skill_de,
    normalize_jitter_distribution,
    resolve_transition_jitter_pmaxes,
    sample_p,
)

# Batch keys this dataset adds (the model + processor consume these).
SKILL_START_IMAGE = "skill_start_image"
SKILL_START_WRIST_IMAGE = "skill_start_wrist_image"
SKILL_START_STATE = "skill_start_state"
SKILL_START_STATE_NORMALIZED = "skill_start_state_normalized"
SKILL_START_FRAME = "skill_start_frame"
SKILL_START_REQUESTED_FRAME = "skill_start_requested_frame"
SKILL_START_FOCUS_UV = "skill_start_focus_uv"
SKILL_START_FOCUS_VALID = "skill_start_focus_valid"
SKILL_CODE = "skill_code"
SKILL_CODE_TRUE = "skill_code_true"
SKILL_PREVIOUS_CODE = "skill_previous_code"  # -1 = episode BOS, never a real FSQ code
SKILL_PROGRESS = "skill_progress"
SKILL_EFFECTIVE_DE = "skill_effective_de"
SKILL_PREVIOUS_ACTION = "skill_previous_action"
SKILL_PREVIOUS_ACTION_BOS = "skill_previous_action_bos"
TERMINATOR_START_OFFSET = "terminator_start_offset"
TERMINATOR_START_STATE = "terminator_start_state"
TERMINATOR_START_STATE_NORMALIZED = "terminator_start_state_normalized"
TERMINATOR_PREDICTOR_START_IMAGE = "terminator_predictor_start_image"
TERMINATOR_PREDICTOR_START_WRIST_IMAGE = "terminator_predictor_start_wrist_image"
SKILL_CANONICAL_ACTIONS = "skill_canonical_actions"
SKILL_CANONICAL_ACTION_IS_PAD = "skill_canonical_action_is_pad"
SKILL_CANONICAL_ACTION_LENGTH = "skill_canonical_action_length"
SKILL_BSPLINE_STATES = "skill_bspline_states"
SKILL_FOCUS_UV = "skill_focus_uv"
SKILL_FOCUS_UV_PIXELS = "skill_focus_uv_pixels"
SKILL_FOCUS_VALID = "skill_focus_valid"
SKILL_FOCUS_CLIPPED = "skill_focus_clipped"
SKILL_END_XYZ = "skill_end_xyz"
SKILL_END_XYZ_VALID = "skill_end_xyz_valid"
SKILL_END_STATE = "skill_end_state"
SKILL_END_STATE_VALID = "skill_end_state_valid"
SKILL_CHUNK_END_STATE = "skill_chunk_end_state"
SKILL_CHUNK_END_STATE_VALID = "skill_chunk_end_state_valid"
SKILL_CHUNK_END_FOCUS_UV = "skill_chunk_end_focus_uv"
SKILL_CHUNK_END_FOCUS_VALID = "skill_chunk_end_focus_valid"
SAME_SKILL_PAIR_ID = "same_skill_pair_id"
SAME_SKILL_PAIR_FALLBACK = "same_skill_pair_fallback"
LATENT_SKILL_GROUP_ID = "latent_skill_group_id"

CAM_3RD = "observation.images.image"
CAM_WRIST = "observation.images.wrist_image"


def _scalar(x) -> int:
    return int(x.item() if torch.is_tensor(x) else np.asarray(x).reshape(-1)[0])


def _post_end_predictor_images(reader, episode, starts, skill_indices, fps, templates):
    """Decode only valid past anchors; fixed slots permit ordinary collation."""
    images = {camera: [torch.zeros_like(templates[camera]) for _ in skill_indices]
              for camera in (CAM_3RD, CAM_WRIST)}
    for slot in np.flatnonzero(skill_indices >= 0):
        timestamp = int(starts[int(skill_indices[slot])]) / fps
        decoded = reader._query_videos({camera: [timestamp] for camera in images}, episode)
        for camera in images:
            image = decoded[camera]
            if reader._image_transforms is not None:
                image = reader._image_transforms(image)
            images[camera][slot] = image
    return {camera: torch.stack(slots) for camera, slots in images.items()}


def _terminator_start_randomization_plan(
    *,
    item_index: int,
    frame_index: int,
    distance_from_start: int,
    distance_to_end: int,
    offset: int,
    shift_current_observation: bool,
) -> tuple[int, int, int, int]:
    """Resolve causal late-jitter capacity and current-sample fields.

    Returns ``(max_late_offset, current_item_index, expected_frame, skill_de)``.
    Legacy paired-shift profiles move the current sample toward the fixed skill
    endpoint.  Start-anchor-only profiles keep the complete current sample and
    its labels fixed, and prevent a late start anchor from crossing current time.
    """
    if shift_current_observation:
        return (
            distance_to_end,
            item_index + offset,
            frame_index + offset,
            distance_to_end - offset,
        )
    return distance_from_start, item_index, frame_index, distance_to_end


def _should_randomize_terminator_start(enabled: bool, probability: float) -> bool:
    """Sample the augmentation gate without perturbing legacy p=1 RNG streams."""
    if not enabled or probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    return bool(np.random.random() < probability)


class _ISSStore:
    """skill_initial_state.npz reader: per-skill observation.state window (±pmax), keyed by episode_id.

    Mirrors the skill_latents.npz convention (flat per-skill arrays). `frame_start` is cross-checked
    against the parquet IFS so a wrong episode/skill alignment fails loudly instead of silently."""

    def __init__(self, npz_path: str):
        z = np.load(npz_path)
        self.frame_start = np.asarray(z["frame_start"])
        self.windows = np.asarray(z["iss_windows"])  # (total_skills, 2*pmax+1, state_dim)
        self.pmax = int(z["pmax"])
        self.max_percent = float(z["max_percent"]) if "max_percent" in z else 0.0
        self.gripper_validity = bool(z["gripper_validity"]) if "gripper_validity" in z else False
        self.gripper_valid_offsets = (
            np.asarray(z["gripper_valid_offsets"], dtype=bool)
            if "gripper_valid_offsets" in z
            else None
        )
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

    def gripper_offset_valid(
        self, ep_idx: int, skill_rank: int, direction: str, magnitude: int
    ) -> bool:
        if not self.gripper_validity or self.gripper_valid_offsets is None:
            raise RuntimeError("ISS has no precomputed gripper-validity contract.")
        direction_index = {
            "early_start": 0,
            "late_start": 1,
            "early_end": 2,
            "late_end": 3,
        }[direction]
        if magnitude < 0 or magnitude > self.pmax:
            return False
        flat = self.by_ep[ep_idx][skill_rank]
        return bool(self.gripper_valid_offsets[flat, direction_index, magnitude])


class _SkillEndStateStore:
    """Canonical skill endpoints, independent of image UV/camera calibration."""

    index_name = "Skill endpoint"

    def __init__(self, episode_id, skill_index, frame_start, frame_end):
        self.episode_id = np.asarray(episode_id, dtype=np.int64)
        self.skill_index = np.asarray(skill_index, dtype=np.int64)
        self.frame_start = np.asarray(frame_start, dtype=np.int64)
        self.frame_end = np.asarray(frame_end, dtype=np.int64)
        arrays = (self.episode_id, self.skill_index, self.frame_start, self.frame_end)
        if any(array.ndim != 1 for array in arrays) or len({len(a) for a in arrays}) != 1:
            raise ValueError("Skill endpoint arrays must have matching one-dimensional shapes.")
        order = np.lexsort((self.frame_start, self.episode_id))
        self.by_ep: dict[int, list[int]] = {}
        for index in order:
            self.by_ep.setdefault(int(self.episode_id[index]), []).append(int(index))
        self.end_state: np.ndarray | None = None

    @classmethod
    def from_dataset(cls, dataset: "SkillVLADataset") -> "_SkillEndStateStore":
        # The builder repeats these canonical sequences in every episode row.
        # Read one row per selected episode, without decoding images or relying
        # on skill_latents.npz (which cleanup_intermediate may have removed).
        columns = dataset.hf_dataset.select_columns(
            ["skill_initial_frame", "skill_length_sequence", "skill_sequence_len"]
        ).with_format("numpy")
        mapping = dataset.reader._absolute_to_relative_idx  # noqa: SLF001
        episodes = dataset.episodes if dataset.episodes is not None else range(dataset.meta.total_episodes)
        identities = [[], [], [], []]
        for ep_idx in episodes:
            episode = dataset.meta.episodes[int(ep_idx)]
            absolute_start = _scalar(episode["dataset_from_index"])
            row_index = absolute_start if mapping is None else mapping[absolute_start]
            row = columns[int(row_index)]
            starts = np.asarray(row["skill_initial_frame"]).reshape(-1)
            lengths = np.asarray(row["skill_length_sequence"]).reshape(-1)
            count = _scalar(row["skill_sequence_len"]) - 1  # exclude EOS/PAD
            if not 0 <= count <= min(len(starts), len(lengths)):
                raise ValueError(f"Invalid canonical skill sequence for episode={ep_idx}.")
            starts, lengths = starts[:count], lengths[:count]
            if np.any(starts < 0) or np.any(lengths <= 0) or np.any(np.diff(starts) <= 0):
                raise ValueError(f"Invalid canonical skill boundaries for episode={ep_idx}.")
            identities[0].extend([int(ep_idx)] * count)
            identities[1].extend(range(count))
            identities[2].extend(starts)
            identities[3].extend(starts + lengths)
        return cls(*identities)

    def cache_end_state(self, dataset: "SkillVLADataset") -> None:
        """Read canonical skill endpoints once from the existing state parquet.

        Use the historical exclusive frame_end convention: an internal skill
        targets the boundary state, and the episode's last endpoint is clamped.
        """
        states = np.asarray(
            dataset.hf_dataset.select_columns(["observation.state"])
            .with_format("numpy")[:]["observation.state"],
            dtype=np.float32,
        )
        if states.ndim != 2 or states.shape[1] < 3:
            raise ValueError(f"SkillVLA observation.state must contain XYZ, got {states.shape}.")
        end_state = np.full((len(self.episode_id), states.shape[1]), np.nan, dtype=np.float32)
        selected_episodes = (
            set(self.by_ep) if dataset.episodes is None else set(dataset.episodes)
        )
        absolute_to_relative = dataset.reader._absolute_to_relative_idx  # noqa: SLF001
        for ep_idx in selected_episodes:
            episode = dataset.meta.episodes[int(ep_idx)]
            episode_length = _scalar(episode["length"])
            absolute_start = _scalar(episode["dataset_from_index"])
            for flat in self.by_ep[int(ep_idx)]:
                end_frame = min(int(self.frame_end[flat]), episode_length - 1)
                if end_frame < int(self.frame_start[flat]):
                    raise ValueError(
                        f"Invalid skill endpoint: episode={ep_idx}, skill={self.skill_index[flat]}, "
                        f"start={self.frame_start[flat]}, end={self.frame_end[flat]}."
                    )
                absolute_index = absolute_start + end_frame
                relative_index = (
                    absolute_index if absolute_to_relative is None
                    else absolute_to_relative[absolute_index]
                )
                end_state[flat] = states[relative_index]
        self.end_state = end_state

    def cache_end_xyz(self, dataset: "SkillVLADataset") -> None:
        """Keep the Arch7 caller name while sharing the full-state cache."""
        self.cache_end_state(dataset)

    def _flat_index(self, ep_idx: int, skill_rank: int, expected_frame_start: int) -> int:
        if ep_idx not in self.by_ep or not 0 <= skill_rank < len(self.by_ep[ep_idx]):
            raise KeyError(f"No {self.index_name} target for episode={ep_idx}, skill={skill_rank}.")
        flat = self.by_ep[ep_idx][skill_rank]
        recorded_rank = int(self.skill_index[flat])
        recorded_start = int(self.frame_start[flat])
        if recorded_rank != skill_rank or recorded_start != expected_frame_start:
            raise ValueError(
                f"{self.index_name}/IFS mismatch "
                f"(ep={ep_idx}, skill={skill_rank}): focus skill={recorded_rank}, "
                f"focus frame_start={recorded_start}, IFS={expected_frame_start}."
            )
        return flat

    def target_xyz(self, ep_idx: int, skill_rank: int, expected_frame_start: int) -> tuple[np.ndarray, bool]:
        state, valid = self.target_state(ep_idx, skill_rank, expected_frame_start)
        return state[:3], valid

    def target_state(self, ep_idx: int, skill_rank: int, expected_frame_start: int) -> tuple[np.ndarray, bool]:
        if self.end_state is None:
            raise RuntimeError("Skill-end state cache is not initialized.")
        flat = self._flat_index(ep_idx, skill_rank, expected_frame_start)
        state = self.end_state[flat].copy()
        return state, bool(np.isfinite(state).all())


class _FocusUVStore(_SkillEndStateStore):
    """Legacy UV artifact plus the same canonical skill-end state lookup."""

    index_name = "Focus"

    def __init__(self, npz_path: str):
        with np.load(npz_path, allow_pickle=False) as z:
            required = {
                "episode_id", "skill_index", "frame_start", "frame_end",
                "focus_uv", "focus_uv_pixels", "focus_valid", "focus_clipped",
            }
            missing = sorted(required - set(z.files))
            if missing:
                raise ValueError(f"skill_focus_uv.npz is missing {missing}: {npz_path}")
            super().__init__(z["episode_id"], z["skill_index"], z["frame_start"], z["frame_end"])
            self.focus_uv = np.asarray(z["focus_uv"], dtype=np.float32)
            self.focus_uv_pixels = np.asarray(z["focus_uv_pixels"], dtype=np.int32)
            self.focus_valid = np.asarray(z["focus_valid"], dtype=np.bool_)
            self.focus_clipped = np.asarray(z["focus_clipped"], dtype=np.bool_)
        if any(len(a) != len(self.episode_id) for a in (
            self.focus_uv, self.focus_uv_pixels, self.focus_valid, self.focus_clipped,
        )):
            raise ValueError("skill_focus_uv arrays have inconsistent lengths.")
        if self.focus_uv.shape[1:] != (2,) or self.focus_uv_pixels.shape[1:] != (2,):
            raise ValueError(
                "skill_focus_uv coordinates must have shape [num_skills, 2], got "
                f"uv={self.focus_uv.shape}, pixels={self.focus_uv_pixels.shape}."
            )

    def target(
        self,
        ep_idx: int,
        skill_rank: int,
        expected_frame_start: int,
    ) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        flat = self._flat_index(ep_idx, skill_rank, expected_frame_start)
        return (
            self.focus_uv[flat].copy(),
            self.focus_uv_pixels[flat].copy(),
            bool(self.focus_valid[flat]),
            bool(self.focus_clipped[flat]),
        )


class SkillVLADataset(LeRobotDataset):
    """LeRobotDataset with skill-start or current-frame predictor inputs."""

    def __init__(self, *args, **kwargs):
        self._terminator_goal_uv = bool(kwargs.pop("terminator_goal_uv", False))
        self._goal_xyz_uv_noise = bool(kwargs.pop("goal_xyz_uv_noise", False))
        self._terminator_post_end_steps = int(kwargs.pop("terminator_post_end_steps", 0))
        if self._terminator_post_end_steps < 0:
            raise ValueError("terminator_post_end_steps must be non-negative.")
        jitter_pmax_override = kwargs.pop("jitter_pmax", None)
        jitter_pmax_percent = float(kwargs.pop("jitter_pmax_percent", 0.0))
        jitter_preserve_gripper = bool(kwargs.pop("jitter_preserve_gripper", False))
        jitter_distribution_override = kwargs.pop("jitter_distribution", None)
        self._foveated_vision = FoveatedVisionAugmentationConfig.from_mapping(
            kwargs.pop("foveated_vision_config", None)
        )
        self._include_canonical_skill_actions = bool(
            kwargs.pop("include_canonical_skill_actions", False)
        )
        self._residual_action_horizon = int(kwargs.pop("residual_action_horizon", 0))
        self._bspline_skill_state_tokens = int(kwargs.pop("bspline_skill_state_tokens", 0))
        if self._bspline_skill_state_tokens < 0 or self._bspline_skill_state_tokens == 1:
            raise ValueError("bspline_skill_state_tokens must be 0 (disabled) or >=2.")
        self._bspline_target_cache = {}
        if self._residual_action_horizon < 0:
            raise ValueError("residual_action_horizon must be non-negative.")
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
        # Current-only VSA still needs jittered skill selection/start XYZ,
        # but neither of the extra start-camera decodes.
        self._decode_predictor_start_images = bool(kwargs.pop("decode_predictor_start_images", True))
        self._include_preskill_plan_inputs = bool(kwargs.pop("include_preskill_plan_inputs", False))
        self._predictor_wrist_input = bool(kwargs.pop("predictor_wrist_input", True))
        self._include_terminator_start_inputs = bool(
            kwargs.pop("include_terminator_start_inputs", False)
        )
        self._separate_terminator_start_inputs = bool(
            kwargs.pop("separate_terminator_start_inputs", False)
        )
        self._include_terminator_predictor_inputs = bool(
            kwargs.pop("include_terminator_predictor_inputs", False)
        )
        self._terminator_start_randomization = bool(
            kwargs.pop("terminator_start_randomization", False)
        )
        self._terminator_start_randomization_early_frames = int(
            kwargs.pop("terminator_start_randomization_early_frames", 0)
        )
        self._terminator_start_randomization_late_frames = int(
            kwargs.pop("terminator_start_randomization_late_frames", 0)
        )
        self._terminator_start_randomization_distribution = normalize_jitter_distribution(
            kwargs.pop(
                "terminator_start_randomization_distribution", "half_normal"
            )
        )
        self._terminator_start_randomization_probability = float(
            kwargs.pop("terminator_start_randomization_probability", 1.0)
        )
        self._terminator_start_randomization_shift_current_observation = bool(
            kwargs.pop(
                "terminator_start_randomization_shift_current_observation", False
            )
        )
        if (
            self._include_predictor_start_inputs
            and self._include_terminator_start_inputs
            and not self._separate_terminator_start_inputs
        ):
            raise ValueError(
                "Predictor and terminator start-input sampling require separate training jobs."
            )
        if self._include_terminator_predictor_inputs and not self._separate_terminator_start_inputs:
            raise ValueError(
                "Terminator Predictor inputs require separate_terminator_start_inputs=True."
            )
        if self._terminator_start_randomization and not self._include_terminator_start_inputs:
            raise ValueError(
                "Terminator start randomization requires include_terminator_start_inputs=True."
            )
        if not np.isfinite(self._terminator_start_randomization_probability) or not (
            0.0 <= self._terminator_start_randomization_probability <= 1.0
        ):
            raise ValueError(
                "terminator_start_randomization_probability must be finite and between 0 and 1."
            )
        self._include_skill_end_xyz_target = bool(
            kwargs.pop("include_skill_end_xyz_target", False)
        )
        self._include_skill_end_state_target = bool(
            kwargs.pop("include_skill_end_state_target", False)
        )
        self._include_chunk_end_patch_target = bool(
            kwargs.pop("include_chunk_end_patch_target", False)
        )
        self._include_chunk_end_state_target = bool(
            kwargs.pop("include_chunk_end_state_target", False)
        ) or self._include_chunk_end_patch_target
        self._chunk_end_state_horizon = int(
            kwargs.pop("chunk_end_state_horizon", 0)
        )
        self._chunk_end_state_respects_skill_end = bool(
            kwargs.pop("chunk_end_state_respects_skill_end", True)
        )
        if self._include_chunk_end_state_target and self._chunk_end_state_horizon <= 0:
            raise ValueError("chunk_end_state_horizon must be positive when its target is enabled.")
        self._include_terminator_goal_target = bool(
            kwargs.pop("include_terminator_goal_target", False)
        )
        self._terminator_history_mask_padding = bool(kwargs.pop("terminator_history_mask_padding", False))
        self._terminator_history_reset_on_skill = bool(kwargs.pop("terminator_history_reset_on_skill", False))
        self._terminator_history_length = int(kwargs.pop("terminator_history_length", 20))
        if self._terminator_history_reset_on_skill and not self._terminator_history_mask_padding:
            raise ValueError("Skill-reset history requires padding masks.")
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
        if self._include_canonical_skill_actions or self._residual_action_horizon:
            dataset_max_length = int(info.get("skill_observed_max_length", 0))
            if dataset_max_length <= 0:
                raise ValueError(
                    "arch0_skill requires a positive skill_observed_max_length "
                    "in the dataset info.json."
                )
            # Trajectories are zero-padded to the configured length and the padding is masked
            # out of the loss, so a checkpoint trained with a LONGER horizon can keep it on a
            # dataset whose skills are shorter (NewTask FT). A shorter horizon cannot hold them.
            if self._include_canonical_skill_actions and self._canonical_skill_action_max_length < dataset_max_length:
                raise ValueError(
                    "Configured canonical skill-action length cannot hold this dataset's "
                    "longest skill: "
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
        if self._bspline_skill_state_tokens and not self._include_predictor_start_inputs:
            raise ValueError("Bspline targets require include_predictor_start_inputs=True for skill selection.")
        if (self._include_chunk_end_state_target or self._bspline_skill_state_tokens
                or self._include_preskill_plan_inputs):
            values = (
                self.hf_dataset.select_columns(["observation.state"])
                .with_format("numpy")[:]["observation.state"]
            )
            chunk_end_states = np.asarray(values, dtype=np.float32)
            if chunk_end_states.ndim != 2 or chunk_end_states.shape[1] != 8:
                raise ValueError(
                    "LIT chunk-end supervision requires grounded observation.state "
                    f"with exactly 8 values, got {chunk_end_states.shape}."
                )
            self._chunk_end_state_cache = (
                torch.from_numpy(chunk_end_states).clone().contiguous()
            )
        else:
            self._chunk_end_state_cache = None
        iss_path = self._resolve_iss_path(info.get("skill_initial_state_path"), self.root)
        self._iss = (
            _ISSStore(iss_path)
            if self._include_predictor_start_inputs
            or self._include_terminator_start_inputs
            or self._include_terminator_predictor_inputs
            else None
        )
        focus_path = self._resolve_optional_companion_path(
            info.get("skill_focus_uv_path"), self.root
        )
        self._focus_uv = _FocusUVStore(focus_path) if focus_path is not None else None
        if self._terminator_goal_uv or self._goal_xyz_uv_noise:
            if focus_path is None:
                raise ValueError("XYZ-to-UV goal projection needs skill_focus_uv.npz with recorded Top-camera calibration.")
            if self._foveated_vision.enabled:
                raise ValueError("XYZ-to-UV goal projection expects uncropped Top-view images.")
            from .chunk_patch_target import load_dataset_top_camera_transforms
            from lerobot.policies.skill_aux.terminator_uv import normalized_top_camera_projection

            matrices, height, width = load_dataset_top_camera_transforms(self, focus_path)
            self._terminator_camera_projections = {
                ep: torch.from_numpy(normalized_top_camera_projection(matrix, height=height, width=width))
                for ep, matrix in matrices.items()
            }
        self._skill_end_state = self._focus_uv
        self._chunk_end_focus_cache = None
        self._chunk_end_focus_valid_cache = None
        if self._include_chunk_end_patch_target:
            if focus_path is None:
                raise ValueError("Chunk-end patch alignment requires skill_focus_uv.npz camera metadata.")
            if self._foveated_vision.enabled:
                raise ValueError("Chunk-end patch targets require the original uncropped camera images.")
            from .chunk_patch_target import cache_top_focus

            self._chunk_end_focus_cache, self._chunk_end_focus_valid_cache = cache_top_focus(
                self, focus_path, self._chunk_end_state_cache
            )
        if self._foveated_vision.enabled and self._focus_uv is None:
            raise ValueError(
                "Foveated vision is enabled, but the SkillVLA dataset metadata has "
                "no skill_focus_uv_path. Rebuild the dataset with focus_uv.enabled=true."
            )
        if (
            self._include_skill_end_xyz_target
            or self._include_skill_end_state_target
            or self._include_terminator_goal_target
        ):
            if self._skill_end_state is None:
                self._skill_end_state = _SkillEndStateStore.from_dataset(self)
            if (
                (self._include_skill_end_xyz_target or self._include_skill_end_state_target)
                and not self._include_predictor_start_inputs
            ):
                raise ValueError(
                    "Predictor/action skill-end supervision requires predictor-start inputs."
                )
            self._skill_end_state.cache_end_state(self)
        default_pmax = self._iss.pmax if self._iss is not None else 0
        dataset_pmax = int(info.get("skill_pmax", default_pmax))
        if self._iss is not None and dataset_pmax != self._iss.pmax:
            raise ValueError(
                "Dataset skill_pmax does not match the ISS window: "
                f"info.json={dataset_pmax}, ISS={self._iss.pmax}."
            )
        dataset_max_percent = float(info.get("skill_jitter_max_percent", 0.0))
        if not 0.0 <= jitter_pmax_percent <= 100.0:
            raise ValueError("jitter_pmax_percent must be in [0, 100].")
        if jitter_pmax_percent > dataset_max_percent:
            raise ValueError(
                f"Requested jitter percent {jitter_pmax_percent:g} exceeds the "
                f"built maximum {dataset_max_percent:g}."
            )
        if jitter_pmax_percent > 0.0 and jitter_pmax_override not in (None, 0):
            raise ValueError("Step and percentage jitter are mutually exclusive.")
        if jitter_preserve_gripper and (
            self._iss is None or not self._iss.gripper_validity
        ):
            raise ValueError(
                "Gripper-preserving jitter requires a dataset built with its "
                "gripper-validity mask."
            )
        self._jitter_pmax_percent = jitter_pmax_percent
        self._jitter_preserve_gripper = jitter_preserve_gripper
        dataset_directional = {
            name: int(info.get(f"skill_jitter_{name}_pmax", dataset_pmax))
            for name in ("early_start", "late_start", "early_end", "late_end")
        }
        requested_start_ranges = {
            "early_start": self._terminator_start_randomization_early_frames,
            "late_start": self._terminator_start_randomization_late_frames,
        }
        invalid_start_ranges = {
            name: value
            for name, value in requested_start_ranges.items()
            if value < 0 or value > dataset_directional[name]
        }
        if invalid_start_ranges:
            raise ValueError(
                "Terminator start randomization exceeds the built dataset contract "
                f"{dataset_directional}: {invalid_start_ranges}."
            )
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
        self._storage_pmax = dataset_pmax
        if self._pmax > dataset_pmax:
            raise ValueError(
                "Requested transition jitter exceeds the prebuilt ISS window "
                f"[0, {dataset_pmax}], got {self._pmax}."
            )
        self._jitter_distribution = normalize_jitter_distribution(
            info.get("skill_jitter_distribution", "half_normal")
            if jitter_distribution_override in (None, "")
            else jitter_distribution_override
        )

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
    def jitter_pmax_percent(self) -> float:
        return self._jitter_pmax_percent

    @property
    def jitter_preserve_gripper(self) -> bool:
        return self._jitter_preserve_gripper

    def jitter_offset_valid(
        self, ep_idx: int, skill_rank: int, direction: str, magnitude: int
    ) -> bool:
        if not self._jitter_preserve_gripper:
            return True
        return self._iss.gripper_offset_valid(
            ep_idx, skill_rank, direction, magnitude
        )

    @property
    def canonical_skill_action_max_length(self) -> int:
        return self._canonical_skill_action_max_length

    def _bspline_skill_states(self, *, episode_index, skill_start_frame, skill_length):
        """Selected original GT skill, as with canonical flow; jitter changes selection only."""
        from .proprio_spline import resample_skill_proprio

        key = (episode_index, skill_start_frame, skill_length)
        if key not in self._bspline_target_cache:
            episode = self.meta.episodes[episode_index]
            if skill_length <= 0 or skill_start_frame < 0 or skill_start_frame + skill_length > _scalar(episode["length"]):
                raise ValueError(f"Bspline skill range leaves episode: {key}.")
            start = _scalar(episode["dataset_from_index"]) + skill_start_frame
            indices = range(start, start + skill_length)
            mapping = self.reader._absolute_to_relative_idx
            indices = list(indices) if mapping is None else [mapping[index] for index in indices]
            selected = self._chunk_end_state_cache[indices].numpy()
            self._bspline_target_cache[key] = torch.from_numpy(
                resample_skill_proprio(selected, self._bspline_skill_state_tokens)
            )
        return self._bspline_target_cache[key].clone()

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

    def _residual_skill_actions(self, *, episode_index, current_frame, skill_start_frame, skill_length):
        """Past H + current H + future 2H; loss mask outside selected GT skill.

        Transition jitter selects the skill first; its original canonical GT
        interval remains the base target. No action crosses an episode boundary.
        """
        horizon = self._residual_action_horizon
        cache = self._canonical_action_cache
        if horizon <= 0 or cache is None:
            raise RuntimeError("Residual action caching is disabled.")
        episode = self.meta.episodes[episode_index]
        episode_length = _scalar(episode["length"])
        frames = torch.arange(current_frame - horizon, current_frame + 3 * horizon)
        valid = ((frames >= skill_start_frame) & (frames < skill_start_frame + skill_length)
                 & (frames >= 0) & (frames < episode_length))
        actions = cache.new_zeros(4 * horizon, cache.shape[-1])
        absolute = (frames[valid] + _scalar(episode["dataset_from_index"])).tolist()
        mapping = self.reader._absolute_to_relative_idx  # noqa: SLF001
        relative = absolute if mapping is None else [mapping[index] for index in absolute]
        actions[valid] = cache.index_select(0, torch.as_tensor(relative, dtype=torch.long))
        return actions, ~valid

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
        terminal_repeat = False
        pair_id = -1
        pair_fallback = False
        latent_skill_group_id = -1
        effective_de_override = None
        jitter_override = None
        if isinstance(idx, tuple):
            if len(idx) == 2 and self._terminator_post_end_steps > 0:
                idx, terminal_repeat = idx
            elif len(idx) == 3:
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
        terminator_start = None
        if self._include_terminator_start_inputs:
            original = self.hf_dataset[item_index]
            original_frame = _scalar(original["frame_index"])
            original_ep = _scalar(original["episode_index"])
            original_k = _scalar(original["skill_index"])
            original_ds = _scalar(original["skill_ds"])
            original_de = _scalar(original["skill_de"])
            original_seq_len = _scalar(original["skill_sequence_len"])
            original_ss = np.asarray(original["skill_sequence"]).reshape(-1)
            original_ifs = np.asarray(original["skill_initial_frame"]).reshape(-1)
            gt_start = int(original_ifs[original_k])
            offset = 0
            max_late_offset, _, _, _ = _terminator_start_randomization_plan(
                item_index=item_index,
                frame_index=original_frame,
                distance_from_start=original_ds,
                distance_to_end=original_de,
                offset=0,
                shift_current_observation=(
                    self._terminator_start_randomization_shift_current_observation
                ),
            )
            if _should_randomize_terminator_start(
                self._terminator_start_randomization,
                self._terminator_start_randomization_probability,
            ):
                choices = []
                early_limit = min(
                    self._terminator_start_randomization_early_frames,
                    gt_start,
                )
                late_limit = min(
                    self._terminator_start_randomization_late_frames,
                    max_late_offset,
                )
                if early_limit > 0:
                    choices.append((-1, early_limit))
                if late_limit > 0:
                    choices.append((1, late_limit))
                if choices:
                    direction, limit = choices[
                        int(np.random.randint(0, len(choices)))
                    ]
                    offset = direction * sample_p(
                        limit,
                        distribution=self._terminator_start_randomization_distribution,
                    )
            (
                _,
                current_item_index,
                expected_frame,
                current_skill_de,
            ) = _terminator_start_randomization_plan(
                item_index=item_index,
                frame_index=original_frame,
                distance_from_start=original_ds,
                distance_to_end=original_de,
                offset=offset,
                shift_current_observation=(
                    self._terminator_start_randomization_shift_current_observation
                ),
            )
            terminator_start = {
                "episode_index": original_ep,
                "skill_index": original_k,
                "skill_ds": original_ds,
                "skill_de": current_skill_de,
                "skill_sequence_len": original_seq_len,
                "skill_sequence": original_ss,
                "skill_initial_frame": original_ifs,
                "gt_start": gt_start,
                "offset": offset,
                "expected_frame": expected_frame,
            }
            item_index = current_item_index
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
        if getattr(self, "_goal_xyz_uv_noise", False):
            from lerobot.policies.skill_aux.terminator_uv import TOP_CAMERA
            item[TOP_CAMERA] = self._terminator_camera_projections[ep_idx].clone()
        k = _scalar(item["skill_index"])
        ds = _scalar(item["skill_ds"])
        de = _scalar(item["skill_de"])
        seq_len = _scalar(item["skill_sequence_len"])
        ss = np.asarray(item["skill_sequence"]).reshape(-1)
        ifs = np.asarray(item["skill_initial_frame"]).reshape(-1)
        if terminator_start is not None:
            if ep_idx != terminator_start["episode_index"]:
                raise RuntimeError("Terminator start jitter crossed an episode boundary.")
            if frame_index != terminator_start["expected_frame"]:
                raise RuntimeError(
                    "Terminator start jitter decoded the wrong current frame: "
                    f"expected={terminator_start['expected_frame']}, got={frame_index}."
                )
            k = int(terminator_start["skill_index"])
            ds = int(terminator_start["skill_ds"])
            de = int(terminator_start["skill_de"])
            seq_len = int(terminator_start["skill_sequence_len"])
            ss = terminator_start["skill_sequence"]
            ifs = terminator_start["skill_initial_frame"]
            item["skill_index"] = torch.tensor(k, dtype=torch.long)
            item["skill_ds"] = torch.tensor(ds, dtype=torch.long)
            item["skill_de"] = torch.tensor(de, dtype=torch.long)
            item["skill_sequence_len"] = torch.tensor(seq_len, dtype=torch.long)
            item[TERMINATOR_START_OFFSET] = torch.tensor(
                int(terminator_start["offset"]), dtype=torch.long
            )
            if self._iss is None:
                raise RuntimeError("Terminator start proprio requires the ISS store.")
            iss_center = self._iss.pmax
            iss_index = int(
                np.clip(
                    iss_center + int(terminator_start["offset"]),
                    0,
                    2 * iss_center,
                )
            )
            start_state = self._iss.state(
                ep_idx,
                k,
                iss_index,
                int(terminator_start["gt_start"]),
            )
            start_key = (
                TERMINATOR_START_STATE
                if self._separate_terminator_start_inputs
                else SKILL_START_STATE
            )
            item[start_key] = torch.from_numpy(start_state)
        # TRUE current skill's code (un-jittered) — the FSQ terminator co-training (FT) conditions on
        # the actual skill the current frame belongs to, with progress/termination from its ds/de.
        item[SKILL_CODE_TRUE] = torch.tensor(int(ss[k]), dtype=torch.long)
        item[SKILL_EFFECTIVE_DE] = torch.tensor(de, dtype=torch.long)
        if self._terminator_history_mask_padding:
            from lerobot.policies.skillVLA.newterm_supervision import HISTORY_VALID, history_valid_mask
            if item["observation.state"].ndim != 2 or item["observation.state"].shape[0] != self._terminator_history_length:
                raise ValueError("Masked history requires a full observation.state temporal window.")
            # Keep raw episode history intact for past-skill conditions. The
            # model masks values before projection/attention, separately for
            # each current/past skill's canonical start.
            item[HISTORY_VALID] = history_valid_mask(
                frame_index, int(ifs[k]), self._terminator_history_length,
                reset_on_skill=self._terminator_history_reset_on_skill,
            )
        if self._terminator_post_end_steps > 0:
            from lerobot.policies.skillVLA.newterm_supervision import (
                POST_END_CODES, POST_END_VALID, POST_END_START_STATE,
                POST_END_GOAL_XYZ, POST_END_GOAL_VALID, post_end_conditions,
            )

            item[POST_END_CODES], item[POST_END_VALID], past_skill_indices = post_end_conditions(
                frame_index, k, ifs, ss, self._terminator_post_end_steps,
                terminal_repeat=terminal_repeat, return_skill_indices=True,
            )
            if self._include_terminator_predictor_inputs:
                previous_codes = np.full(self._terminator_post_end_steps, -1, dtype=np.int64)
                for slot in np.flatnonzero(past_skill_indices > 0):
                    previous_codes[slot] = int(ss[int(past_skill_indices[slot]) - 1])
                item["terminator_post_end_previous_codes"] = torch.from_numpy(previous_codes)
            if self._terminator_history_mask_padding:
                from lerobot.policies.skillVLA.newterm_supervision import POST_END_HISTORY_VALID
                past_masks = torch.zeros(
                    self._terminator_post_end_steps, self._terminator_history_length, dtype=torch.bool
                )
                for slot in np.flatnonzero(past_skill_indices >= 0):
                    previous = int(past_skill_indices[slot])
                    past_masks[slot] = history_valid_mask(
                        frame_index, int(ifs[previous]), self._terminator_history_length,
                        reset_on_skill=self._terminator_history_reset_on_skill,
                    )
                item[POST_END_HISTORY_VALID] = past_masks
            if self._include_terminator_start_inputs:
                # Reuse the same ISS store as the current skill's start token.
                # Current observations stay fixed; only the past condition's
                # skill code and canonical skill-start proprio change.
                past_start_states = np.zeros(
                    (self._terminator_post_end_steps, len(start_state)), dtype=np.float32
                )
                for slot in np.flatnonzero(past_skill_indices >= 0):
                    previous = int(past_skill_indices[slot])
                    past_start_states[slot] = self._iss.state(
                        ep_idx, previous, self._iss.pmax, int(ifs[previous])
                    )
                item[POST_END_START_STATE] = torch.from_numpy(past_start_states)
            if self._include_terminator_goal_target:
                # Use the same endpoint artifact/cache as the current skill.
                # These raw metre-valued goals are normalized by the FSQ model.
                past_goals = np.zeros((self._terminator_post_end_steps, 3), dtype=np.float32)
                past_goal_valid = np.zeros(self._terminator_post_end_steps, dtype=np.bool_)
                for slot in np.flatnonzero(past_skill_indices >= 0):
                    previous = int(past_skill_indices[slot])
                    previous_end, previous_valid = self._skill_end_state.target_state(
                        ep_idx, previous, int(ifs[previous])
                    )
                    past_goals[slot] = previous_end[:3]
                    past_goal_valid[slot] = previous_valid and np.isfinite(previous_end[:3]).all()
                item[POST_END_GOAL_XYZ] = torch.from_numpy(past_goals)
                item[POST_END_GOAL_VALID] = torch.from_numpy(past_goal_valid)
            # skill_sequence_len includes EOS in existing datasets; check the
            # actual episode endpoint rather than comparing against that length.
            episode_last_frame = _scalar(self.meta.episodes[ep_idx]["length"]) - 1
            if self._include_terminator_predictor_inputs:
                from lerobot.policies.skillVLA.newterm_supervision import TERMINAL_FRAME
                item[TERMINAL_FRAME] = torch.tensor(frame_index == episode_last_frame)
            if terminal_repeat and (frame_index != episode_last_frame or de != 0):
                raise RuntimeError("Newterm terminal repeats must be final-skill endpoints.")

        if self._include_terminator_goal_target:
            current_start = int(ifs[k])
            if getattr(self, "_terminator_goal_uv", False):
                from lerobot.policies.skill_aux.terminator_uv import TOP_CAMERA
                item[TOP_CAMERA] = self._terminator_camera_projections[ep_idx].clone()
            end_state, end_state_valid = self._skill_end_state.target_state(
                ep_idx, k, current_start
            )
            item[SKILL_END_STATE] = torch.from_numpy(end_state)
            item[SKILL_END_STATE_VALID] = torch.tensor(
                end_state_valid, dtype=torch.bool
            )
            if self._focus_uv is not None:
                focus_uv, focus_pixels, focus_valid, focus_clipped = self._focus_uv.target(
                    ep_idx, k, current_start
                )
                item[SKILL_FOCUS_UV] = torch.from_numpy(focus_uv)
                item[SKILL_FOCUS_UV_PIXELS] = torch.from_numpy(focus_pixels)
                item[SKILL_FOCUS_VALID] = torch.tensor(focus_valid, dtype=torch.bool)
                item[SKILL_FOCUS_CLIPPED] = torch.tensor(focus_clipped, dtype=torch.bool)

        reader = None
        ep_len = _scalar(self.meta.episodes[ep_idx]["length"])
        predictor_start_frame = None
        predictor_start_images = None
        if self._include_terminator_predictor_inputs:
            item["terminator_previous_skill_code"] = torch.tensor(
                int(ss[k - 1]) if k > 0 else -1, dtype=torch.long
            )
            if terminator_start is None:
                raise RuntimeError(
                    "Terminator Predictor inputs require a resolved Terminator start."
                )
            reader = self._ensure_reader()
            terminator_predictor_frame = int(
                np.clip(
                    int(terminator_start["gt_start"])
                    + int(terminator_start["offset"]),
                    0,
                    ep_len - 1,
                )
            )
            start_ts = terminator_predictor_frame / self.fps
            terminator_images = reader._query_videos(  # noqa: SLF001
                {CAM_3RD: [start_ts], CAM_WRIST: [start_ts]}, ep_idx
            )
            if reader._image_transforms is not None:  # noqa: SLF001
                terminator_images = {
                    camera: reader._image_transforms(image)  # noqa: SLF001
                    for camera, image in terminator_images.items()
                }
            item[TERMINATOR_PREDICTOR_START_IMAGE] = terminator_images[CAM_3RD]
            item[TERMINATOR_PREDICTOR_START_WRIST_IMAGE] = terminator_images[CAM_WRIST]
            if self._terminator_post_end_steps > 0:
                from lerobot.policies.skillVLA.newterm_supervision import (
                    POST_END_PREDICTOR_START_IMAGE, POST_END_PREDICTOR_START_WRIST_IMAGE,
                )
                past_images = _post_end_predictor_images(
                    reader, ep_idx, ifs, past_skill_indices, self.fps, terminator_images,
                )
                item[POST_END_PREDICTOR_START_IMAGE] = past_images[CAM_3RD]
                item[POST_END_PREDICTOR_START_WRIST_IMAGE] = past_images[CAM_WRIST]
        if self._include_predictor_start_inputs:
            if jitter_override is None:
                if self._jitter_pmax_percent > 0.0:
                    lengths = np.asarray(item["skill_length_sequence"]).reshape(-1)
                    kp, offset = choose_jitter_percent(
                        k,
                        ds,
                        de,
                        seq_len,
                        lengths,
                        self._jitter_pmax_percent,
                        distribution=self._jitter_distribution,
                        valid=(
                            lambda rank, direction, magnitude: self.jitter_offset_valid(
                                ep_idx, rank, direction, magnitude
                            )
                        ),
                    )
                else:
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
                if not 0 <= kp < seq_len - 1 or not -self._storage_pmax <= offset <= self._storage_pmax:
                    raise ValueError(
                        f"Invalid sampler-provided jitter (k'={kp}, offset={offset}) for "
                        f"seq_len={seq_len}, pmax={self._pmax}."
                    )
            skill_code = int(ss[kp])
            item[SKILL_PREVIOUS_CODE] = torch.tensor(
                int(ss[kp - 1]) if kp > 0 else -1, dtype=torch.long
            )
            gt_start = int(ifs[kp])
            if self._include_skill_end_xyz_target or self._include_skill_end_state_target:
                end_state, end_state_valid = self._skill_end_state.target_state(ep_idx, kp, gt_start)
                if self._include_skill_end_xyz_target:
                    item[SKILL_END_XYZ] = torch.from_numpy(end_state[:3].copy())
                    item[SKILL_END_XYZ_VALID] = torch.tensor(end_state_valid, dtype=torch.bool)
                if self._include_skill_end_state_target:
                    item[SKILL_END_STATE] = torch.from_numpy(end_state)
                    item[SKILL_END_STATE_VALID] = torch.tensor(end_state_valid, dtype=torch.bool)
            # Images and ISS state share the same causal jittered start.
            predictor_start_frame, observation_offset = causal_skill_start(
                gt_start, offset, frame_index, ep_len
            )
            if getattr(self, "_decode_predictor_start_images", True):
                reader = self._ensure_reader()
                start_ts = predictor_start_frame / self.fps
                cameras = (CAM_3RD, CAM_WRIST) if getattr(self, "_predictor_wrist_input", True) else (CAM_3RD,)
                predictor_start_images = reader._query_videos(  # noqa: SLF001
                    {camera: [start_ts] for camera in cameras}, ep_idx
                )
                if reader._image_transforms is not None:  # noqa: SLF001
                    predictor_start_images = {
                        camera: reader._image_transforms(image)  # noqa: SLF001
                        for camera, image in predictor_start_images.items()
                    }
            if self._iss is None:
                raise RuntimeError("Predictor start inputs require the ISS store.")
            # A smaller runtime window still indexes around the storage center.
            iss_center = self._iss.pmax
            iss_index = int(np.clip(iss_center + observation_offset, 0, 2 * iss_center))
            start_state = self._iss.state(ep_idx, kp, iss_index, gt_start)
            item[SKILL_START_FRAME] = torch.tensor(predictor_start_frame, dtype=torch.long)
            item[SKILL_START_REQUESTED_FRAME] = torch.tensor(
                gt_start + offset,
                dtype=torch.long,
            )
            if predictor_start_images is not None:
                item[SKILL_START_IMAGE] = predictor_start_images[CAM_3RD]
                if getattr(self, "_predictor_wrist_input", True):
                    item[SKILL_START_WRIST_IMAGE] = predictor_start_images[CAM_WRIST]
            item[SKILL_START_STATE] = (
                start_state.clone() if torch.is_tensor(start_state)
                else torch.from_numpy(start_state)
            )
            item[SKILL_CODE] = torch.tensor(skill_code, dtype=torch.long)
            if getattr(self, "_include_preskill_plan_inputs", False):
                self._add_preskill_plan_inputs(item, ep_idx, kp, predictor_start_frame, ifs, ss)
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
                # Start images are not passed through the current-frame crop/
                # foveation augmentor. Keep their static top-camera target.
                item[SKILL_START_FOCUS_UV] = item[SKILL_FOCUS_UV].clone()
                item[SKILL_START_FOCUS_VALID] = item[SKILL_FOCUS_VALID].clone()
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
            if getattr(self, "_bspline_skill_state_tokens", 0):
                item[SKILL_BSPLINE_STATES] = self._bspline_skill_states(
                    episode_index=ep_idx, skill_start_frame=gt_start, skill_length=int(lens[kp]),
                )
            if getattr(self, "_residual_action_horizon", 0):
                window, window_pad = self._residual_skill_actions(
                    episode_index=ep_idx, current_frame=int(ifs[k]) + int(ds),
                    skill_start_frame=gt_start, skill_length=int(lens[kp]),
                )
                item[SKILL_CANONICAL_ACTIONS] = window
                item[SKILL_CANONICAL_ACTION_IS_PAD] = window_pad
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

        if self._include_chunk_end_state_target:
            self._add_chunk_end_targets(
                item, item_index=item_index, frame_index=frame_index, episode_length=ep_len
            )
        return item

    def _add_preskill_plan_inputs(self, item, episode_index, rank, start_frame, starts, codes):
        from . import preskill_plan as memory

        frame = memory.previous_observation_frame(rank, start_frame)
        valid = frame >= 0
        previous_end = np.zeros(8, dtype=np.float32)
        if valid:
            previous_end, endpoint_valid = self._skill_end_state.target_state(
                episode_index, rank - 1, int(starts[rank - 1]),
            )
            valid = bool(endpoint_valid)
        item[memory.VALID] = torch.tensor(valid, dtype=torch.bool)
        item[memory.FRAME] = torch.tensor(frame if valid else -1, dtype=torch.long)
        item[memory.CODE] = torch.tensor(int(codes[rank - 1]) if valid else 0, dtype=torch.long)
        item[memory.END_STATE] = torch.as_tensor(previous_end).clone()
        uv, uv_valid = np.zeros(2, dtype=np.float32), False
        if valid and getattr(self, "_focus_uv", None) is not None:
            uv, _, uv_valid, _ = self._focus_uv.target(episode_index, rank - 1, int(starts[rank - 1]))
        item[memory.FOCUS_UV] = torch.as_tensor(uv).clone()
        item[memory.FOCUS_VALID] = torch.tensor(bool(uv_valid), dtype=torch.bool)
        if not valid:
            for target, source in ((memory.IMAGE, CAM_3RD), (memory.WRIST_IMAGE, CAM_WRIST)):
                item[target] = torch.zeros_like(item[source])
            item[memory.STATE] = torch.zeros_like(item["observation.state"])
            item[memory.START_STATE] = torch.zeros_like(item["observation.state"])
            return
        reader = self._ensure_reader()
        images = reader._query_videos({camera: [frame / self.fps] for camera in (CAM_3RD, CAM_WRIST)}, episode_index)
        if reader._image_transforms is not None:
            images = {key: reader._image_transforms(value) for key, value in images.items()}
        if self._foveated_vision.enabled or self._foveated_vision.randomization_enabled:
            focus = None
            if self._focus_uv is not None:
                uv, _, _, _ = self._focus_uv.target(episode_index, rank - 1, int(starts[rank - 1]))
                focus = torch.as_tensor(uv)
            images[CAM_3RD], images[CAM_WRIST] = augment_camera_pair(
                images[CAM_3RD], images[CAM_WRIST], focus, self._foveated_vision,
            )
        item[memory.IMAGE], item[memory.WRIST_IMAGE] = images[CAM_3RD], images[CAM_WRIST]
        # Early jitter can precede the previous canonical start for short skills.
        # Causally clip that start too; never fetch a future proprio anchor.
        previous_start = min(int(starts[rank - 1]), frame)
        episode = self.meta.episodes[episode_index]
        origin = _scalar(episode["dataset_from_index"])
        mapping = reader._absolute_to_relative_idx
        for target, index in ((memory.STATE, frame), (memory.START_STATE, previous_start)):
            absolute = origin + index
            relative = mapping[absolute] if mapping is not None else absolute
            item[target] = self._chunk_end_state_cache[relative].clone()

    def _add_chunk_end_targets(self, item, *, item_index: int, frame_index: int, episode_length: int):
        """State and both patch labels must follow the same moving, valid action endpoint."""
        if self._chunk_end_state_cache is None:
            raise RuntimeError("Chunk-end state cache is not initialized.")
        horizon = self._chunk_end_state_horizon
        action_is_pad = item.get("action_is_pad")
        if action_is_pad is None:
            valid_actions = horizon
        else:
            padding = torch.as_tensor(action_is_pad).reshape(-1)[:horizon].bool()
            # Delta timestamps pad only the tail; a hole makes the endpoint ambiguous.
            if bool(((~padding[1:]) & padding[:-1]).any()):
                raise ValueError("action_is_pad must contain one valid prefix then padding.")
            valid_actions = int((~padding).sum().item())
        if self._chunk_end_state_respects_skill_end:
            valid_actions = min(valid_actions, int(item[SKILL_EFFECTIVE_DE].item()) + 1)
        available = max(episode_length - 1 - frame_index, 0)
        advance = min(valid_actions, available)
        target_valid = valid_actions > 0 and advance == valid_actions
        target_index = item_index + advance
        item[SKILL_CHUNK_END_STATE] = self._chunk_end_state_cache[target_index].clone()
        item[SKILL_CHUNK_END_STATE_VALID] = torch.tensor(target_valid, dtype=torch.bool)
        if self._include_chunk_end_patch_target:
            item[SKILL_CHUNK_END_FOCUS_UV] = self._chunk_end_focus_cache[target_index].clone()
            item[SKILL_CHUNK_END_FOCUS_VALID] = (
                self._chunk_end_focus_valid_cache[target_index].clone() & target_valid
            )
