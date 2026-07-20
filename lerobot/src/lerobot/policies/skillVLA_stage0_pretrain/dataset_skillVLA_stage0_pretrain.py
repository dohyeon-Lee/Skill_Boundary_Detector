"""Frame-level Stage-0 motor data plus a segment-uniform autoregressive transition sample."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lerobot.policies.skillVLA.dataset_skillVLA import CAM_3RD, CAM_WRIST, SkillVLADataset
from lerobot.policies.skillVLA.dataset_transitions import _Pack
from lerobot.policies.skillVLA.skill_jitter import sample_offset
from lerobot.policies.skillVLA_pretrain.dataset_skillVLA_pretrain import _FastTargets

AR_IMAGE = "stage0_pretrain_image"
AR_WRIST_IMAGE = "stage0_pretrain_wrist_image"
AR_STATE = "stage0_pretrain_state"
AR_SKILL_CODE = "stage0_pretrain_skill_code"
AR_TASK = "stage0_pretrain_task"
AR_FAST_TOKENS = "stage0_pretrain_fast_tokens"
AR_FAST_TOKEN_MASK = "stage0_pretrain_fast_token_mask"
AR_TRAJECTORY_LENGTH = "stage0_pretrain_trajectory_length"


class SkillVLAStage0PretrainDataset(SkillVLADataset):
    """Regular Stage-0 frame samples paired with independent skill-transition AR samples.

    The frame index is mapped modulo the transition count. Since the training loader shuffles frame
    indices, every epoch covers transition segments uniformly (counts differ by at most one), avoiding
    the skill-length weighting that would result from attaching the current frame's segment target.
    """

    def __init__(
        self,
        *args,
        transition_packs: list[str] | None = None,
        pretrain_target_packs: list[str] | None = None,
        max_fast_tokens: int = 384,
        transition_randomization: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        transition_paths = [Path(path) for path in (transition_packs or []) if str(path).strip()]
        target_paths = [Path(path) for path in (pretrain_target_packs or []) if str(path).strip()]
        if not transition_paths or len(transition_paths) != len(target_paths):
            raise ValueError(
                "Stage0-pretrain needs matching transition/FAST target packs, got "
                f"{len(transition_paths)} and {len(target_paths)}."
            )
        self._ar_packs = [_Pack(path) for path in transition_paths]
        self._ar_targets = [_FastTargets(path) for path in target_paths]
        self._ar_cum = np.cumsum([pack.n for pack in self._ar_packs])
        self._ar_randomization = bool(transition_randomization)
        self._ar_max_fast_tokens = int(max_fast_tokens)
        if self._ar_max_fast_tokens <= 0:
            raise ValueError("max_fast_tokens must be positive.")

        for pack, target, path in zip(
            self._ar_packs, self._ar_targets, target_paths, strict=True
        ):
            if pack.n != target.n:
                raise ValueError(f"Transition/FAST segment mismatch for {path}: {pack.n} != {target.n}.")
            for name, expected, actual in (
                ("skill_code", pack.skill_code, target.skill_code),
                ("episode_id", pack.episode_id, target.episode_id),
                ("frame_start", pack.frame_start, target.frame_start),
                ("frame_end", pack.frame_end, target.frame_end),
            ):
                if not np.array_equal(expected, actual):
                    raise ValueError(f"Transition/FAST {name} mismatch for {path}.")
            if target.max_tokens > self._ar_max_fast_tokens:
                raise ValueError(
                    f"{path} needs {target.max_tokens} FAST tokens, but max_fast_tokens="
                    f"{self._ar_max_fast_tokens}. Targets are never truncated."
                )
        print(
            f"[stage0-pretrain-dataset] frame motor + {int(self._ar_cum[-1])} "
            f"segment-uniform AR targets (randomization={self._ar_randomization})"
        )

    def __getitem__(self, idx) -> dict:
        item = super().__getitem__(idx)
        ar_index = int(idx) % int(self._ar_cum[-1])
        pack_index = int(np.searchsorted(self._ar_cum, ar_index, side="right"))
        previous = int(self._ar_cum[pack_index - 1]) if pack_index else 0
        segment = ar_index - previous
        pack = self._ar_packs[pack_index]
        target = self._ar_targets[pack_index]

        offset = (
            sample_offset(pack.pmax, distribution=pack.jitter_distribution)
            if self._ar_randomization
            else 0
        )
        window_index = pack.pmax + offset
        tokens = target.sequence(segment)
        count = len(tokens)
        padded = torch.zeros(self._ar_max_fast_tokens, dtype=torch.long)
        mask = torch.zeros(self._ar_max_fast_tokens, dtype=torch.bool)
        if count:
            padded[:count] = torch.from_numpy(tokens.copy())
            mask[:count] = True

        item.update(
            {
                AR_IMAGE: pack.image(CAM_3RD, segment, window_index),
                AR_WRIST_IMAGE: pack.image(CAM_WRIST, segment, window_index),
                AR_STATE: torch.from_numpy(pack.start_state[segment, window_index].copy()),
                AR_SKILL_CODE: torch.tensor(int(pack.skill_code[segment]), dtype=torch.long),
                AR_TASK: pack.tasks[int(pack.task_index[segment])],
                AR_FAST_TOKENS: padded,
                AR_FAST_TOKEN_MASK: mask,
                AR_TRAJECTORY_LENGTH: torch.tensor(
                    int(target.trajectory_length[segment]), dtype=torch.long
                ),
            }
        )
        return item
