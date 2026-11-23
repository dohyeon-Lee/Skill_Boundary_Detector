#!/usr/bin/env python

"""ABC-native SkillVLA view with coherent runtime transition jitter.

The storage-light ABC build keeps demonstrations in the official files and
stores only annotations, ISS windows, and exact MuJoCo projection geometry.
This adapter selects one jittered skill per sample and derives every dependent
target from that same selection: code, start proprio, endpoint XYZ/UV, and the
fixed-length full-skill proprio trajectory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.abc_native_dataset import ABCNativeDataset
from lerobot.policies.skillVLA.dataset_skillVLA import (
    LATENT_SKILL_GROUP_ID,
    SAME_SKILL_PAIR_FALLBACK,
    SAME_SKILL_PAIR_ID,
    SKILL_BSPLINE_STATES,
    SKILL_CODE,
    SKILL_CODE_TRUE,
    SKILL_EFFECTIVE_DE,
    SKILL_END_XYZ,
    SKILL_END_XYZ_VALID,
    SKILL_FOCUS_CLIPPED,
    SKILL_FOCUS_UV,
    SKILL_FOCUS_UV_PIXELS,
    SKILL_FOCUS_VALID,
    SKILL_PREVIOUS_ACTION,
    SKILL_PREVIOUS_ACTION_BOS,
    SKILL_PREVIOUS_CODE,
    SKILL_PROGRESS,
    SKILL_START_FRAME,
    SKILL_START_REQUESTED_FRAME,
    SKILL_START_STATE,
    SKILL_WRIST_FOCUS_CLIPPED,
    SKILL_WRIST_FOCUS_UV,
    SKILL_WRIST_FOCUS_UV_PIXELS,
    SKILL_WRIST_FOCUS_VALID,
)
from lerobot.policies.skillVLA.proprio_spline import resample_skill_proprio
from lerobot.policies.skillVLA.skill_jitter import (
    causal_skill_start,
    choose_jitter,
    effective_jittered_skill_de,
    normalize_jitter_distribution,
    resolve_transition_jitter_pmaxes,
)


def _scalar(value) -> int:
    return int(value.item() if torch.is_tensor(value) else np.asarray(value).reshape(-1)[0])


def _companion_path(recorded: str | None, dataset_root: Path, name: str) -> Path:
    candidates = []
    if recorded:
        candidates.append(Path(recorded).expanduser())
    candidates.extend((dataset_root.parent / name, dataset_root / name))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"ABC SkillVLA companion {name!r} was not found; checked "
        + ", ".join(str(path) for path in candidates)
    )


class _ABCISSStore:
    def __init__(self, path: Path) -> None:
        archive = np.load(path, allow_pickle=False)
        self.episode_id = np.asarray(archive["episode_id"], dtype=np.int64)
        self.frame_start = np.asarray(archive["frame_start"], dtype=np.int64)
        self.windows = np.asarray(archive["iss_windows"], dtype=np.float32)
        self.pmax = int(archive["pmax"])
        self.distribution = normalize_jitter_distribution(str(archive["jitter_distribution"]))
        order = np.lexsort((self.frame_start, self.episode_id))
        self.by_episode: dict[int, list[int]] = {}
        for flat in order:
            self.by_episode.setdefault(int(self.episode_id[flat]), []).append(int(flat))

    def state(self, episode_id: int, rank: int, offset: int, expected_start: int) -> np.ndarray:
        flat = self.by_episode[int(episode_id)][int(rank)]
        if int(self.frame_start[flat]) != int(expected_start):
            raise ValueError(
                "ABC ISS/annotation mismatch: "
                f"episode={episode_id}, rank={rank}, "
                f"ISS={int(self.frame_start[flat])}, annotation={expected_start}."
            )
        index = int(np.clip(self.pmax + int(offset), 0, 2 * self.pmax))
        return self.windows[flat, index].copy()


class _ABCSpatialStore:
    def __init__(self, path: Path) -> None:
        archive = np.load(path, allow_pickle=False)
        if str(archive["schema_name"]) != "abc_bimanual_skill_spatial/v1":
            raise ValueError(f"Unsupported ABC spatial artifact: {path}")
        self.image_height = int(archive["image_height"])
        self.image_width = int(archive["image_width"])
        self.episode_ids = np.asarray(archive["episode_id"], dtype=np.int64)
        self.frame_offsets = np.asarray(archive["frame_offsets"], dtype=np.int64)
        self.skill_episode_id = np.asarray(archive["skill_episode_id"], dtype=np.int64)
        self.skill_index = np.asarray(archive["skill_index"], dtype=np.int64)
        self.frame_start = np.asarray(archive["frame_start"], dtype=np.int64)
        self.skill_end_xyz = np.asarray(archive["skill_end_xyz"], dtype=np.float32)
        self.top_projection = np.asarray(archive["top_projection"], dtype=np.float32)
        self.wrist_projection = np.asarray(archive["wrist_projection"], dtype=np.float32)
        self.episode_position = {
            int(episode_id): position for position, episode_id in enumerate(self.episode_ids)
        }
        self.by_episode: dict[int, list[int]] = {}
        order = np.lexsort((self.frame_start, self.skill_episode_id))
        for flat in order:
            self.by_episode.setdefault(int(self.skill_episode_id[flat]), []).append(int(flat))

    def _flat_skill(self, episode_id: int, rank: int, expected_start: int) -> int:
        flat = self.by_episode[int(episode_id)][int(rank)]
        # Annotations use the rank among retained skills. The spatial artifact
        # preserves source SBD indices, which can have gaps after short skills
        # are filtered. Match the same sorted episode/start identity as ISS.
        if int(self.frame_start[flat]) != int(expected_start):
            raise ValueError(
                "ABC spatial/annotation mismatch: "
                f"episode={episode_id}, rank={rank}, "
                f"spatial={int(self.frame_start[flat])}, annotation={expected_start}."
            )
        return flat

    def targets(
        self, episode_id: int, rank: int, expected_start: int, current_frame: int
    ) -> dict[str, np.ndarray]:
        episode_position = self.episode_position[int(episode_id)]
        flat = self._flat_skill(episode_id, rank, expected_start)
        endpoint = self.skill_end_xyz[flat]
        frame_flat = int(self.frame_offsets[episode_position]) + int(current_frame)
        matrices = np.stack(
            (
                self.top_projection[episode_position],
                self.top_projection[episode_position],
                self.wrist_projection[frame_flat, 0],
                self.wrist_projection[frame_flat, 1],
            )
        )
        points = np.stack((endpoint[0], endpoint[1], endpoint[0], endpoint[1]))
        homogeneous = np.concatenate(
            (points, np.ones((4, 1), dtype=np.float32)), axis=-1
        )
        projected = np.einsum("nij,nj->ni", matrices, homogeneous)
        depth = projected[:, 2]
        valid = np.isfinite(projected).all(axis=-1) & (depth > 1e-8)
        raw = np.full((4, 2), np.nan, dtype=np.float32)
        np.divide(projected[:, :2], depth[:, None], out=raw, where=valid[:, None])
        clipped = valid & (
            (raw[:, 0] < 0)
            | (raw[:, 0] > self.image_width - 1)
            | (raw[:, 1] < 0)
            | (raw[:, 1] > self.image_height - 1)
        )
        bounded = raw.copy()
        bounded[:, 0] = np.clip(bounded[:, 0], 0, self.image_width - 1)
        bounded[:, 1] = np.clip(bounded[:, 1], 0, self.image_height - 1)
        normalized = np.zeros((4, 2), dtype=np.float32)
        normalized[:, 0] = 2 * bounded[:, 0] / max(self.image_width - 1, 1) - 1
        normalized[:, 1] = 2 * bounded[:, 1] / max(self.image_height - 1, 1) - 1
        normalized[~valid] = 0
        pixels = np.full((4, 2), -1, dtype=np.int64)
        pixels[valid] = np.rint(bounded[valid]).astype(np.int64)
        return {
            "endpoint": endpoint.copy(),
            "uv": normalized,
            "pixels": pixels,
            "valid": valid,
            "clipped": clipped,
        }


class ABCNativeSkillVLADataset(ABCNativeDataset):
    """ABC-native frame dataset extended with SkillVLA supervision."""

    def __init__(
        self,
        *args,
        jitter_pmax: int | None = None,
        jitter_pmax_percent: float = 0.0,
        jitter_preserve_gripper: bool = False,
        jitter_early_start_pmax: int = -1,
        jitter_late_start_pmax: int = -1,
        jitter_early_end_pmax: int = -1,
        jitter_late_end_pmax: int = -1,
        jitter_distribution: str | None = None,
        bspline_skill_state_tokens: int = 0,
        include_predictor_start_inputs: bool = False,
        terminator_post_end_steps: int = 0,
        vlm_termination_enabled: bool = False,
        vlm_termination_post_end_steps: int = 10,
        **kwargs,
    ) -> None:
        if float(jitter_pmax_percent) != 0.0:
            raise ValueError("ABC SkillVLA currently uses step-based transition jitter only.")
        if jitter_preserve_gripper:
            raise ValueError("ABC transition jitter does not use LIBERO gripper-validity gating.")
        super().__init__(*args, **kwargs)
        if self._skill_annotations is None:
            raise ValueError("ABCNativeSkillVLADataset requires built skill annotations.")
        info = self.meta.info
        iss_path = _companion_path(
            info.get("skill_initial_state_path"), self.root, "skill_initial_state.npz"
        )
        spatial_path = _companion_path(
            info.get("skill_spatial_targets_path"), self.root, "skill_spatial_targets.npz"
        )
        self._iss = _ABCISSStore(iss_path)
        self._spatial = _ABCSpatialStore(spatial_path)
        storage_pmax = int(info.get("skill_pmax", self._iss.pmax))
        if storage_pmax != self._iss.pmax:
            raise ValueError("ABC ISS pmax disagrees with dataset metadata.")
        runtime_pmax = storage_pmax if jitter_pmax is None else int(jitter_pmax)
        if not 0 <= runtime_pmax <= storage_pmax:
            raise ValueError(
                f"ABC runtime jitter pmax must be within [0,{storage_pmax}], got {runtime_pmax}."
            )
        directional = resolve_transition_jitter_pmaxes(
            runtime_pmax,
            early_start_pmax=jitter_early_start_pmax,
            late_start_pmax=jitter_late_start_pmax,
            early_end_pmax=jitter_early_end_pmax,
            late_end_pmax=jitter_late_end_pmax,
        )
        if max(directional) > storage_pmax:
            raise ValueError("ABC directional jitter exceeds the built ISS window.")
        self._runtime_pmax = runtime_pmax
        self._directional_pmaxes = dict(
            zip(("early_start", "late_start", "early_end", "late_end"), directional, strict=True)
        )
        self._jitter_distribution = normalize_jitter_distribution(
            jitter_distribution or str(info.get("skill_jitter_distribution", self._iss.distribution))
        )
        self._bspline_tokens = int(bspline_skill_state_tokens)
        if self._bspline_tokens not in (0,) and self._bspline_tokens < 2:
            raise ValueError("bspline_skill_state_tokens must be 0 or >=2.")
        self._bspline_cache: dict[tuple[int, int, int], torch.Tensor] = {}
        self.include_predictor_start_inputs = bool(include_predictor_start_inputs)
        self.terminator_post_end_steps = int(terminator_post_end_steps)
        self._vlm_termination_enabled = bool(vlm_termination_enabled)
        self._vlm_termination_post_end_steps = int(vlm_termination_post_end_steps)
        if self._vlm_termination_post_end_steps < 0:
            raise ValueError("vlm_termination_post_end_steps must be non-negative.")

    @property
    def jitter_directional_pmaxes(self) -> dict[str, int]:
        return dict(self._directional_pmaxes)

    @property
    def jitter_distribution(self) -> str:
        return self._jitter_distribution

    @property
    def jitter_preserve_gripper(self) -> bool:
        return False

    @property
    def native_episode_end_indices(self) -> np.ndarray:
        return self._cumulative_lengths - 1

    @property
    def jitter_pmax(self) -> int:
        return self._runtime_pmax

    @property
    def jitter_pmax_percent(self) -> float:
        return 0.0

    def _skill_states(
        self, row: dict, episode_id: int, start: int, length: int
    ) -> torch.Tensor:
        key = (int(episode_id), int(start), int(length))
        cached = self._bspline_cache.get(key)
        if cached is None:
            if length <= 0 or start < 0 or start + length > int(row["length"]):
                raise ValueError(f"ABC full-skill range leaves episode: {key}.")
            states = np.asarray(
                self._memmap(row)[start : start + length, : self.state_dim],
                dtype=np.float32,
            )
            cached = torch.from_numpy(
                resample_skill_proprio(
                    states,
                    self._bspline_tokens,
                    gripper_axes=(6, 13),
                )
            )
            self._bspline_cache[key] = cached
        return cached.clone()

    def __getitem__(self, index) -> dict:
        pair_id, pair_fallback, latent_group = -1, False, -1
        jitter_override = None
        effective_de_override = None
        terminal_repeat = False
        if isinstance(index, tuple):
            if len(index) == 2:
                index, terminal_repeat = index
            elif len(index) == 3:
                index, pair_id, pair_fallback = index
            elif len(index) in (5, 6, 7):
                values = index
                index, pair_id, pair_fallback = values[:3]
                jitter_override = (int(values[3]), int(values[4]))
                if len(values) >= 6:
                    latent_group = int(values[5])
                if len(values) == 7:
                    effective_de_override = int(values[6])
            else:
                raise ValueError(f"Unsupported ABC SkillVLA grouped index: {index!r}")
        selected_position, row, current_frame = self._locate(int(index))
        del selected_position
        episode_id = int(row["episode_index"])
        item = super().__getitem__(int(index))
        k = _scalar(item["skill_index"])
        ds = _scalar(item["skill_ds"])
        de = _scalar(item["skill_de"])
        seq_len = _scalar(item["skill_sequence_len"])
        codes = np.asarray(item["skill_sequence"]).reshape(-1)
        starts = np.asarray(item["skill_initial_frame"]).reshape(-1)
        lengths = np.asarray(item["skill_length_sequence"]).reshape(-1)
        if jitter_override is None:
            kp, offset = choose_jitter(
                k,
                ds,
                de,
                seq_len,
                self._runtime_pmax,
                distribution=self._jitter_distribution,
                early_start_pmax=self._directional_pmaxes["early_start"],
                late_start_pmax=self._directional_pmaxes["late_start"],
                early_end_pmax=self._directional_pmaxes["early_end"],
                late_end_pmax=self._directional_pmaxes["late_end"],
            )
        else:
            kp, offset = jitter_override
        if not 0 <= kp < seq_len - 1 or abs(offset) > self._iss.pmax:
            raise ValueError(
                f"Invalid ABC jitter selection k'={kp}, offset={offset}, seq_len={seq_len}."
            )
        gt_start = int(starts[kp])
        causal_frame, causal_offset = causal_skill_start(
            gt_start, offset, current_frame, int(row["length"])
        )
        start_state = self._iss.state(episode_id, kp, causal_offset, gt_start)
        spatial = self._spatial.targets(episode_id, kp, gt_start, current_frame)

        item[SKILL_CODE_TRUE] = torch.tensor(int(codes[k]), dtype=torch.long)
        item[SKILL_CODE] = torch.tensor(int(codes[kp]), dtype=torch.long)
        item[SKILL_PREVIOUS_CODE] = torch.tensor(
            int(codes[kp - 1]) if kp > 0 else -1, dtype=torch.long
        )
        item[SKILL_START_FRAME] = torch.tensor(causal_frame, dtype=torch.long)
        item[SKILL_START_REQUESTED_FRAME] = torch.tensor(
            gt_start + offset, dtype=torch.long
        )
        item[SKILL_START_STATE] = torch.from_numpy(start_state)
        item[SKILL_END_XYZ] = torch.from_numpy(spatial["endpoint"])
        item[SKILL_END_XYZ_VALID] = torch.from_numpy(
            np.isfinite(spatial["endpoint"]).all(axis=-1)
        )
        item[SKILL_FOCUS_UV] = torch.from_numpy(spatial["uv"][:2])
        item[SKILL_FOCUS_UV_PIXELS] = torch.from_numpy(spatial["pixels"][:2])
        item[SKILL_FOCUS_VALID] = torch.from_numpy(spatial["valid"][:2])
        item[SKILL_FOCUS_CLIPPED] = torch.from_numpy(spatial["clipped"][:2])
        item[SKILL_WRIST_FOCUS_UV] = torch.from_numpy(spatial["uv"][2:])
        item[SKILL_WRIST_FOCUS_UV_PIXELS] = torch.from_numpy(spatial["pixels"][2:])
        item[SKILL_WRIST_FOCUS_VALID] = torch.from_numpy(spatial["valid"][2:])
        item[SKILL_WRIST_FOCUS_CLIPPED] = torch.from_numpy(spatial["clipped"][2:])
        if self._bspline_tokens:
            item[SKILL_BSPLINE_STATES] = self._skill_states(
                row, episode_id, gt_start, int(lengths[kp])
            )
        item[SKILL_EFFECTIVE_DE] = torch.tensor(
            effective_jittered_skill_de(
                k=k,
                k_prime=kp,
                ds=ds,
                de=de,
                skill_initial_frames=starts,
                skill_lengths=lengths,
                offset=offset,
            ),
            dtype=torch.long,
        )
        if effective_de_override is not None:
            item[SKILL_EFFECTIVE_DE] = torch.tensor(effective_de_override, dtype=torch.long)
        if self._vlm_termination_enabled:
            self._add_vlm_termination_targets(
                item, episode_id, current_frame, kp, starts, lengths, codes, int(row["length"])
            )
        progress = (current_frame - gt_start) / max(int(lengths[kp]) - 1, 1)
        item[SKILL_PROGRESS] = torch.tensor(
            float(np.clip(progress, 0, 1)), dtype=torch.float32
        )
        values = self._memmap(row)
        if current_frame == 0:
            previous_action = np.zeros(self.action_dim, dtype=np.float32)
        else:
            previous_action = np.asarray(
                values[current_frame - 1, self.state_dim :], dtype=np.float32
            ).copy()
        item[SKILL_PREVIOUS_ACTION] = torch.from_numpy(previous_action)
        item[SKILL_PREVIOUS_ACTION_BOS] = torch.tensor(current_frame == 0)
        item[SAME_SKILL_PAIR_ID] = torch.tensor(int(pair_id), dtype=torch.long)
        item[SAME_SKILL_PAIR_FALLBACK] = torch.tensor(bool(pair_fallback))
        item[LATENT_SKILL_GROUP_ID] = torch.tensor(int(latent_group), dtype=torch.long)
        if self.include_predictor_start_inputs:
            keys = ["observation.images.top", "observation.images.left_wrist", "observation.images.right_wrist"]
            images = self._query_cameras(row, causal_frame, {}, camera_keys=keys)
            if self.image_transforms is not None:
                images = {key: self.image_transforms(image) for key, image in images.items()}
            item["skill_start_image"], item["skill_start_wrist_image"], item["skill_start_right_wrist_image"] = (
                images[key] for key in keys
            )
            # Text proprio and instruction must describe exactly this invocation,
            # including calls jittered to the preceding/following prompt segment.
            item["observation.state"] = torch.from_numpy(start_state.copy())
            item["task_index"], item["task"] = self._prompt_at(row, causal_frame)

        if self.terminator_post_end_steps:
            from lerobot.policies.skillVLA.newterm_supervision import (
                post_end_conditions, POST_END_CODES, POST_END_VALID,
                POST_END_START_STATE, TERMINAL_FRAME,
            )
            steps = self.terminator_post_end_steps
            past_codes, past_valid, ranks = post_end_conditions(
                current_frame, k, starts, codes, steps,
                terminal_repeat=terminal_repeat, return_skill_indices=True,
            )
            past_starts = np.zeros((steps, self.state_dim), dtype=np.float32)
            past_xyz = np.zeros((steps, 2, 3), dtype=np.float32)
            past_uv = np.zeros((steps, 2, 2), dtype=np.float32)
            past_uv_valid = np.zeros((steps, 2), dtype=bool)
            for slot in np.flatnonzero(past_valid.numpy()):
                rank = int(ranks[slot]); start = int(starts[rank])
                past_starts[slot] = self._iss.state(episode_id, rank, 0, start)
                target = self._spatial.targets(episode_id, rank, start, current_frame)
                past_xyz[slot] = target["endpoint"]
                past_uv[slot] = target["uv"][:2]
                past_uv_valid[slot] = target["valid"][:2]
            item[POST_END_CODES], item[POST_END_VALID] = past_codes, past_valid
            item[POST_END_START_STATE] = torch.from_numpy(past_starts)
            item["terminator_post_end_goal_uv"] = torch.from_numpy(past_uv)
            item["terminator_post_end_goal_uv_valid"] = torch.from_numpy(past_uv_valid)
            item["terminator_post_end_bimanual_xyz"] = torch.from_numpy(past_xyz)
            item[TERMINAL_FRAME] = torch.tensor(terminal_repeat)
            position = self._spatial.episode_position[episode_id]
            item["terminator_top_camera_projection"] = torch.from_numpy(self._spatial.top_projection[position].copy())
        return item

    def _add_vlm_termination_targets(self, item, episode, frame, selected, starts, lengths, codes, episode_length):
        """Reuse LIBERO's GT endpoints for the jitter-selected skill and past conditions.

        Each extra condition retains its own two top-view UVs and arm validity.
        These targets use the existing spatial sidecar, so no data rebuild or
        extra video decode is needed, and VSA frame sampling stays unchanged.
        """
        from .vlm_termination_supervision import (
            DISTANCE, TERMINAL, PAST_CODES, PAST_VALID, PAST_UV, PAST_UV_VALID,
            PAST_TERMINAL, PAST_DISTANCE, termination_conditions,
        )
        distance, terminal, indices = termination_conditions(
            frame, selected, starts, lengths, codes, self._vlm_termination_post_end_steps,
            episode_length=episode_length,
        )
        slots = len(indices)
        past_codes = torch.zeros(slots, dtype=torch.long)
        valid = torch.zeros(slots, dtype=torch.bool)
        uv = torch.zeros(slots, 2, 2)
        uv_valid = torch.zeros(slots, 2, dtype=torch.bool)
        past_terminal = torch.zeros(slots, dtype=torch.bool)
        past_distance = torch.zeros(slots, dtype=torch.long)
        for slot, rank in enumerate(indices):
            if rank < 0:
                continue
            start = int(starts[rank])
            target = self._spatial.targets(episode, int(rank), start, frame)
            point = torch.as_tensor(target["uv"][:2]).float()
            finite = torch.isfinite(point).all(dim=-1)
            past_codes[slot] = int(codes[rank])
            valid[slot] = True
            uv[slot] = torch.where(finite[:, None], point, torch.zeros_like(point)).clamp(-1, 1)
            uv_valid[slot] = torch.as_tensor(target["valid"][:2]).bool() & finite
            endpoint = start + int(lengths[rank]) - 1
            past_distance[slot] = endpoint - frame
            past_terminal[slot] = frame == endpoint == episode_length - 1
        item.update({
            DISTANCE: torch.tensor(distance, dtype=torch.long),
            TERMINAL: torch.tensor(terminal), PAST_CODES: past_codes, PAST_VALID: valid,
            PAST_UV: uv, PAST_UV_VALID: uv_valid, PAST_TERMINAL: past_terminal,
            PAST_DISTANCE: past_distance,
        })
