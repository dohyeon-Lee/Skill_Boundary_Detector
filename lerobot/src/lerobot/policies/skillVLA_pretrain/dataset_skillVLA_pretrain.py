"""Segment-level SkillVLA pretraining dataset with precomputed variable-length FAST tokens."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lerobot.policies.skillVLA.dataset_skillVLA import SKILL_START_STATE
from lerobot.policies.skillVLA.dataset_transitions import SkillTransitionDataset
from lerobot.utils.constants import OBS_STATE

PRETRAIN_FAST_TOKENS = "pretrain_fast_tokens"
PRETRAIN_FAST_TOKEN_MASK = "pretrain_fast_token_mask"
PRETRAIN_TRAJECTORY_LENGTH = "pretrain_trajectory_length"


class _FastTargets:
    def __init__(self, path: str | Path):
        z = np.load(str(path))
        if int(z.get("schema_version", 0)) < 1:
            raise ValueError(f"Pretrain FAST target pack has an unsupported schema: {path}")
        self.tokens = np.asarray(z["fast_tokens"], dtype=np.int64)
        self.offsets = np.asarray(z["fast_token_offsets"], dtype=np.int64)
        self.trajectory_length = np.asarray(z["trajectory_length"], dtype=np.int64)
        self.skill_code = np.asarray(z["skill_code"], dtype=np.int64)
        self.episode_id = np.asarray(z["episode_id"], dtype=np.int64)
        self.frame_start = np.asarray(z["frame_start"], dtype=np.int64)
        self.frame_end = np.asarray(z["frame_end"], dtype=np.int64)
        self.n = int(self.trajectory_length.shape[0])
        if self.offsets.shape != (self.n + 1,) or int(self.offsets[-1]) != len(self.tokens):
            raise ValueError(f"Corrupt FAST token offsets in {path}")
        self.max_tokens = int(np.diff(self.offsets).max(initial=0))

    def sequence(self, index: int) -> np.ndarray:
        return self.tokens[self.offsets[index] : self.offsets[index + 1]]


class SkillVLAPretrainDataset(SkillTransitionDataset):
    """One sample per skill: start observation/language -> FSQ code + full-trajectory FAST tokens."""

    def __init__(
        self,
        *args,
        transition_packs: list[str] | None = None,
        pretrain_target_packs: list[str] | None = None,
        max_fast_tokens: int = 384,
        transition_randomization: bool = True,
        **kwargs,
    ):
        super().__init__(
            *args,
            transition_packs=transition_packs,
            transition_randomization=transition_randomization,
            **kwargs,
        )
        target_paths = [p for p in (pretrain_target_packs or []) if str(p).strip()]
        if len(target_paths) != len(self._packs):
            raise ValueError(
                "SkillVLA pretrain needs one FAST target pack per transition pack: "
                f"{len(target_paths)} target(s) for {len(self._packs)} transition pack(s)."
            )
        self._fast_targets = [_FastTargets(path) for path in target_paths]
        self._max_fast_tokens = int(max_fast_tokens)
        if self._max_fast_tokens <= 0:
            raise ValueError("max_fast_tokens must be positive.")

        for transition, target, path in zip(self._packs, self._fast_targets, target_paths, strict=True):
            if transition.n != target.n:
                raise ValueError(
                    f"Transition/FAST target segment count mismatch for {path}: "
                    f"{transition.n} != {target.n}."
                )
            contracts = (
                ("skill_code", transition.skill_code, target.skill_code),
                ("episode_id", transition.episode_id, target.episode_id),
                ("frame_start", transition.frame_start, target.frame_start),
                ("frame_end", transition.frame_end, target.frame_end),
            )
            for name, expected, actual in contracts:
                if not np.array_equal(expected, actual):
                    raise ValueError(f"Transition/FAST target {name} mismatch for {path}.")
            if target.max_tokens > self._max_fast_tokens:
                raise ValueError(
                    f"FAST target pack {path} needs {target.max_tokens} tokens, but "
                    f"max_fast_tokens={self._max_fast_tokens}. Increase tokenizers.max_fast_tokens; "
                    "targets are never truncated."
                )

    def __getitem__(self, idx) -> dict:
        item = super().__getitem__(idx)
        index = int(idx)
        pack_index = int(np.searchsorted(self._cum, index, side="right"))
        segment = index - (int(self._cum[pack_index - 1]) if pack_index > 0 else 0)
        target = self._fast_targets[pack_index]
        tokens = target.sequence(segment)
        count = len(tokens)

        padded = torch.zeros(self._max_fast_tokens, dtype=torch.long)
        mask = torch.zeros(self._max_fast_tokens, dtype=torch.bool)
        if count:
            padded[:count] = torch.from_numpy(tokens.copy())
            mask[:count] = True
        item[PRETRAIN_FAST_TOKENS] = padded
        item[PRETRAIN_FAST_TOKEN_MASK] = mask
        item[PRETRAIN_TRAJECTORY_LENGTH] = torch.tensor(
            int(target.trajectory_length[segment]), dtype=torch.long
        )
        # The PI0-FAST prompt reads observation.state; for pretraining this is the same start-state
        # selected from the transition window (GT center or randomized offset).
        item[OBS_STATE] = item[SKILL_START_STATE].clone()
        return item
