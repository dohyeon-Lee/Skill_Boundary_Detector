#!/usr/bin/env python
"""Multi-checkpoint closed-loop LIBERO evaluation for Stage 1 and Stage 2."""

import copy
import gc
import json
import logging
import math
import os
import sys
from collections import deque
from contextlib import nullcontext
from itertools import cycle, islice
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.skill_expert.configuration_skill_expert import (
    COND_GEMMA_ARCHITECTURE,
    COND_GEMMA_ARCHITECTURE_REVISION,
    SkillExpertConfig,
    normalize_conditioning_route,
)
from lerobot.policies.skill_expert.modeling_utils import build_fsq_terminator
from lerobot.scripts.lerobot_skillvla_eval import (
    _libero_task_descriptions,
    close_envs,
    eval_policy_all,
)
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
    STAGE2_VLM_CACHE_ID,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.random_utils import set_seed

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from eval_oracle import (  # noqa: E402
    load_episode_exact_data,
    load_sequences_by_language,
    map_sequences_to_tasks,
)

RAW_STATE = "skill_decoder_state"
RAW_IMAGE = "skill_decoder_image"
RAW_WRIST = "skill_decoder_wrist"
CURRENT_IMAGE = "observation.images.image"
CURRENT_WRIST = "observation.images.wrist_image"
# Bump this string whenever the Stage-2 VLM snapshot semantics change so
# completed eval artifacts are never silently reused under a different input
# contract.
STAGE2_VLM_START_CONTRACT = (
    "skill_boundary_image_state_external_predictor_base_v2"
)
log = logging.getLogger(__name__)

_INLINE_CUDA_GUARD_EXIT_CODE = 86


def _run_inline_cuda_guard() -> None:
    """Validate CUDA in the real evaluator process to avoid a second torch import."""
    if os.environ.get("LEROBOT_INLINE_CUDA_GUARD", "0") != "1":
        return
    if torch.cuda.is_available():
        return

    marker = os.environ.get("LEROBOT_CUDA_GUARD_FAILURE_MARKER", "")
    if marker:
        try:
            Path(marker).write_text(
                "torch.cuda.is_available()=false\n", encoding="utf-8"
            )
        except OSError as error:
            print(
                f"GPU GUARD: could not write failure marker {marker}: {error}",
                flush=True,
            )
    print(
        "GPU GUARD: torch.cuda.is_available() is false; refusing CPU fallback.",
        flush=True,
    )
    raise SystemExit(_INLINE_CUDA_GUARD_EXIT_CODE)


def _mark_startup_ready() -> None:
    """Tell the Slurm parent that imports and CUDA initialization completed."""
    marker = os.environ.get("LEROBOT_STARTUP_READY_MARKER", "")
    if marker:
        Path(marker).touch()


def _normalize_skill_source(value: str) -> str:
    aliases = {
        "gt": "gt",
        "oracle": "gt",
        "own": "own",
        "pred": "own",
        "predicted": "own",
        "predictor": "own",
        "external": "external",
    }
    normalized = aliases.get(str(value).strip().lower())
    if normalized is None:
        raise ValueError(f"skill_source must be external|own|gt, got {value!r}.")
    return normalized


def _normalize_advance_mode(value: str) -> str:
    aliases = {
        "gt": "gt",
        "own": "own",
        "terminator": "own",
        "external": "external",
        "original": "original",
    }
    normalized = aliases.get(str(value).strip().lower())
    if normalized is None:
        raise ValueError(
            f"advance_mode must be external|own|original|gt, got {value!r}."
        )
    return normalized


def _normalize_terminator_variant(value: str) -> str:
    aliases = {
        "normal": "state_image",
        "state_image": "state_image",
        "state+image": "state_image",
        "image": "image_only",
        "image_only": "image_only",
        "image-only": "image_only",
    }
    normalized = aliases.get(str(value).strip().lower())
    if normalized is None:
        raise ValueError(
            f"terminator_variant must be state_image|image_only, got {value!r}."
        )
    return normalized


class CheckpointTerminator:
    """Inference adapter around the terminator currently attached to a policy."""

    use_wrist = True

    def __init__(self, policy, variant: str = "state_image"):
        self.variant = _normalize_terminator_variant(variant)
        self.requires_state = self.variant == "state_image"
        if self.requires_state:
            if policy.model.fsq_term_train is None:
                raise ValueError("The policy checkpoint has no co-trained terminator.")
        elif getattr(policy.model, "fsq_image_term_train", None) is None:
            raise ValueError("The policy has no attached image-only terminator.")
        self.policy = policy
        module = (
            policy.model.fsq_term_train
            if self.requires_state
            else policy.model.fsq_image_term_train
        )
        # Exposed so progress-gated advance modes can reject a terminator whose
        # progress output is a constant zero by construction.
        self.termination_only = bool(getattr(module, "termination_only", False))
        self.context_mode = str(getattr(module, "context_mode", "proprio"))

    @torch.no_grad()
    def terminate(
        self,
        codes: torch.Tensor,
        state: torch.Tensor | None,
        image: torch.Tensor,
        wrist_image: torch.Tensor,
        previous_action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.variant == "image_only":
            progress, logits = self.policy.image_only_terminator_predict(
                codes, image, wrist_image
            )
        elif self.context_mode == "prev_action":
            module = self.policy.model.fsq_term_train
            if previous_action is None:
                context = torch.zeros(
                    codes.shape[0],
                    int(module.state_dim),
                    device=codes.device,
                    dtype=next(module.parameters()).dtype,
                )
            else:
                context = module.normalize_previous_action(previous_action)
            z_q = self.policy.model._code_to_zq(
                codes.to(self.policy.model._fsq_strides.device)
            ).to(device=next(module.parameters()).device, dtype=next(module.parameters()).dtype)
            progress, logits = module(
                z_q,
                context.to(device=z_q.device, dtype=z_q.dtype),
                image.to(device=z_q.device, dtype=z_q.dtype),
                wrist_image.to(device=z_q.device, dtype=z_q.dtype),
            )
        else:
            progress, logits = self.policy.model.terminator_predict(
                codes, state, image, wrist_image
            )
        return progress, torch.sigmoid(logits)


def _attach_original_terminator(policy, fsq_path: str | Path) -> None:
    """Replace any co-trained terminator with the pristine one saved by FSQ."""
    path = Path(fsq_path)
    if not path.is_file():
        raise FileNotFoundError(f"Original FSQ terminator checkpoint not found: {path}")
    terminator = build_fsq_terminator(path)
    device = next(policy.parameters()).device
    terminator.to(device=device, dtype=torch.float32)
    terminator.requires_grad_(False).eval()
    policy.model.fsq_term_train = terminator


# Predictor-driven panels keep the GT sequence only as a per-env placeholder: the
# skill comes from the predictor and the boundary from the terminator. Evaluating a
# suite the skill dataset never covered (e.g. libero_10 against libero_90 skills)
# therefore only needs a stand-in of the right shape, flagged so the HTML report
# does not render it as a real GT timeline.
_SYNTHETIC_SKILL = {"token": 0, "gt_length": 0, "synthetic": True}


def _is_synthetic_sequences(sequences) -> bool:
    return any(
        isinstance(skill, dict) and skill.get("synthetic")
        for sequence in sequences
        for skill in sequence
    )


def _parse_sequences(sequences) -> tuple[list[list[int]], list[list[int]]]:
    codes, lengths = [], []
    for sequence in sequences:
        episode_codes, episode_lengths = [], []
        for skill in sequence:
            episode_codes.append(
                int(skill["token"] if isinstance(skill, dict) else skill)
            )
            episode_lengths.append(
                int(skill.get("gt_length", 0)) if isinstance(skill, dict) else 0
            )
        if not episode_codes:
            raise ValueError("Every reference skill sequence must be non-empty.")
        codes.append(episode_codes)
        lengths.append(episode_lengths)
    return codes, lengths


class Stage1OraclePolicy(PreTrainedPolicy):
    """Run a skill-conditioned policy with GT skills or its learned predictor."""

    config_class = SkillExpertConfig
    name = "stage1_oracle"

    def __init__(
        self,
        policy,
        terminator,
        *,
        skill_source: str = "gt",
        advance_mode: str,
        end_mode: str,
        end_threshold: float,
        progress_threshold: float,
        max_skill_length: int,
        n_action_steps: int,
        immediate_replan_on_skill_end: bool = False,
        gt_termination_min_fraction: float = 0.0,
    ):
        super().__init__(policy.config)
        skill_source = _normalize_skill_source(skill_source)
        advance_mode = _normalize_advance_mode(advance_mode)
        if advance_mode != "gt" and terminator is None:
            raise ValueError(
                f"advance_mode={advance_mode} requires a terminator."
            )
        allowed_end_modes = {"termination", "progress", "or", "and"}
        # Terminator-free skill/noise evaluation uses the wrapper only to load
        # the action policy. Its outer rollout loop enforces the per-occurrence
        # GT-length cap, so no end signal is queried in this mode.
        if advance_mode == "gt":
            allowed_end_modes.add("max_length")
        if end_mode not in allowed_end_modes:
            raise ValueError(
                "end_mode must be termination|progress|or|and"
                f"{'|max_length' if advance_mode == 'gt' else ''}, "
                f"got {end_mode!r}."
            )
        self.policy = policy
        self.terminator = terminator
        self.skill_source = skill_source
        self.advance_mode = advance_mode
        self.end_mode = end_mode
        self.end_threshold = float(end_threshold)
        self.progress_threshold = float(progress_threshold)
        self.max_skill_length = int(max_skill_length)
        self.n_action_steps = int(n_action_steps)
        self.immediate_replan_on_skill_end = bool(immediate_replan_on_skill_end)
        self.gt_termination_min_fraction = float(gt_termination_min_fraction)
        if not 0.0 <= self.gt_termination_min_fraction <= 1.0:
            raise ValueError(
                "gt_termination_min_fraction must be between 0 and 1."
            )
        self._sequences: list[list[int]] | None = None
        self._gt_lengths: list[list[int]] | None = None
        self._references: list[list[int]] | None = None
        self._reference_lengths: list[list[int]] | None = None
        self._references_synthetic = False
        self._action_queue: deque = deque(maxlen=self.n_action_steps)
        self.reset()

    def set_forced_skill_token_sequences(self, sequences) -> None:
        self._sequences, self._gt_lengths = _parse_sequences(sequences)
        self.reset()

    def set_reference_skill_token_sequences(self, sequences) -> None:
        self._references_synthetic = _is_synthetic_sequences(sequences)
        self._references, self._reference_lengths = _parse_sequences(sequences)
        self.reset()

    def reset(self) -> None:
        if hasattr(self, "policy"):
            self.policy.reset()
        runtime_preprocessor = getattr(self, "_runtime_preprocessor", None)
        if runtime_preprocessor is not None:
            runtime_preprocessor.reset()
        self._action_queue.clear()
        source = self._sequences if self.skill_source == "gt" else self._references
        count = len(source) if source is not None else 0
        self._cursor = [0] * count
        self._skill_step = [0] * count
        self._skill_order = [-1] * count
        self._active_trace = [None] * count
        self._pending_advance: set[int] = set()
        self._pending_episode_done: set[int] = set()
        self._episode_done = [False] * count
        self._skill_end_fired = [False] * count
        self._predicted_codes: torch.Tensor | None = None
        self._stage2_vlm_start: dict[str, torch.Tensor] | None = None
        # Updated through ``record_executed_action`` after both the policy and
        # environment postprocessors have run.  A prev-action terminator was
        # trained on this raw action space, not on the policy-normalized chunk.
        self._last_executed_action: torch.Tensor | None = None
        self._trace: list[dict] = []
        self._episode_step = 0
        self._started = False

    def record_executed_action(self, action: torch.Tensor) -> None:
        """Remember the action actually sent to the environment for obs_(t+1)."""
        self._last_executed_action = action.detach().clone()

    def _predict_codes(self, batch: dict) -> torch.Tensor:
        return self.policy.predict_skill_code(batch).view(-1).long()

    def _current_codes(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.skill_source != "gt":
            if self._predicted_codes is None or len(self._predicted_codes) != batch_size:
                raise RuntimeError("Predictor skill codes have not been initialized.")
            return self._predicted_codes.to(device)
        if self._sequences is None or len(self._sequences) != batch_size:
            raise RuntimeError(
                "set_forced_skill_token_sequences must provide one sequence per environment."
            )
        return torch.tensor(
            [
                self._sequences[index][self._cursor[index]]
                for index in range(batch_size)
            ],
            dtype=torch.long,
            device=device,
        )

    def _start_skill(self, batch_index: int, codes: torch.Tensor) -> None:
        self._skill_order[batch_index] += 1
        self._trace.append(
            {
                "batch_index": batch_index,
                "codebook_token": int(codes[batch_index]),
                "skill_index": self._skill_order[batch_index],
                "episode_timestep": self._episode_step,
                "length": 0,
                "end_probs": [],
                "skill_source": self.skill_source,
            }
        )
        self._active_trace[batch_index] = len(self._trace) - 1

    def _capture_stage2_vlm_start(
        self, batch: dict, batch_indices: list[int]
    ) -> None:
        """Snapshot one VLM condition per skill; keep the VSA observation live."""
        if getattr(self.policy, "name", None) != "skill_vla_stage2":
            return
        source_keys = (
            CURRENT_IMAGE,
            CURRENT_WRIST,
            OBS_LANGUAGE_TOKENS,
            OBS_LANGUAGE_ATTENTION_MASK,
        )
        missing = [key for key in source_keys if key not in batch]
        if missing:
            raise ValueError(
                "Stage-2 evaluation requires current images and state-tokenized "
                f"language at each skill boundary; missing={missing}."
            )
        if self._stage2_vlm_start is None:
            self._stage2_vlm_start = {
                key: batch[key].detach().clone() for key in source_keys
            }
            return
        for key in source_keys:
            current = batch[key]
            cached = self._stage2_vlm_start[key]
            if current.shape[0] != cached.shape[0]:
                raise ValueError(
                    "Stage-2 evaluation batch size changed within an episode: "
                    f"{current.shape[0]} != {cached.shape[0]}."
                )
            index = torch.as_tensor(batch_indices, device=current.device, dtype=torch.long)
            cached[index] = current[index].detach()

    def _apply_stage2_vlm_start(self, action_batch: dict) -> None:
        if self._stage2_vlm_start is None:
            return
        action_batch["skill_start_image"] = self._stage2_vlm_start[CURRENT_IMAGE]
        action_batch["skill_start_wrist_image"] = self._stage2_vlm_start[CURRENT_WRIST]
        # These tokens already contain the discretized proprio state. Replacing
        # them here keeps the state prompt fixed while current OBS_STATE remains
        # available to the VSA.
        action_batch[OBS_LANGUAGE_TOKENS] = self._stage2_vlm_start[
            OBS_LANGUAGE_TOKENS
        ]
        action_batch[OBS_LANGUAGE_ATTENTION_MASK] = self._stage2_vlm_start[
            OBS_LANGUAGE_ATTENTION_MASK
        ]

    def _terminator_fired(self, progress: float, probability: float) -> bool:
        progress_high = progress >= self.progress_threshold
        termination_high = probability >= self.end_threshold
        if self.end_mode == "progress":
            return progress_high
        if self.end_mode == "termination":
            return termination_high
        if self.end_mode == "and":
            return progress_high and termination_high
        return progress_high or termination_high

    def _can_advance(self, batch_index: int) -> bool:
        if self.skill_source == "gt":
            return self._cursor[batch_index] < len(self._sequences[batch_index]) - 1
        if self.advance_mode == "gt":
            return self._cursor[batch_index] < len(self._references[batch_index]) - 1
        return True

    def _activate_pending_advances(
        self, batch: dict, device: torch.device
    ) -> set[int]:
        indices = sorted(self._pending_advance)
        if not indices:
            return set()
        if self.skill_source == "gt":
            for batch_index in indices:
                self._cursor[batch_index] += 1
        else:
            if self.advance_mode == "gt":
                for batch_index in indices:
                    self._cursor[batch_index] += 1
            new_codes = self._predict_codes(batch).to(device)
            self._predicted_codes[indices] = new_codes[indices]
        codes = self._current_codes(len(self._cursor), device)
        self._capture_stage2_vlm_start(batch, indices)
        for batch_index in indices:
            self._skill_step[batch_index] = 0
            self._start_skill(batch_index, codes)
        self._pending_advance.clear()
        return set(indices)

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        device = next(self.policy.parameters()).device
        batch_size = int(batch[OBS_STATE].shape[0])
        if len(self._episode_done) != batch_size:
            self._episode_done = [False] * batch_size
        self._skill_end_fired = [False] * batch_size
        if not self._started:
            if self.skill_source != "gt":
                if self._references is None or len(self._references) != batch_size:
                    raise RuntimeError(
                        "Predictor evaluation needs one GT reference sequence per environment."
                    )
                self._predicted_codes = self._predict_codes(batch).to(device)
            codes = self._current_codes(batch_size, device)
            for batch_index in range(batch_size):
                self._start_skill(batch_index, codes)
            self._capture_stage2_vlm_start(batch, list(range(batch_size)))
            self._started = True

        # Fixed mode activates a previously detected boundary only after every
        # queued action has run. Immediate mode discards the shared batch queue
        # and replans from the current observation instead.
        activated_at_start = set()
        if self._pending_advance and (
            not self._action_queue or self.immediate_replan_on_skill_end
        ):
            if self.immediate_replan_on_skill_end:
                self._action_queue.clear()
            activated_at_start = self._activate_pending_advances(batch, device)
        if self._pending_episode_done and (
            not self._action_queue or self.immediate_replan_on_skill_end
        ):
            if self.immediate_replan_on_skill_end:
                self._action_queue.clear()
            for batch_index in self._pending_episode_done:
                self._episode_done[batch_index] = True
            self._pending_episode_done.clear()

        codes = self._current_codes(batch_size, device)
        progress = probability = None
        if self.advance_mode != "gt":
            required = [RAW_IMAGE, RAW_WRIST]
            if getattr(self.terminator, "requires_state", True):
                required.insert(0, RAW_STATE)
            missing = [key for key in required if key not in batch]
            if missing:
                raise ValueError(
                    "The saved policy preprocessor must preserve raw terminator inputs; "
                    f"missing={missing}."
                )
            progress, probability = self.terminator.terminate(
                codes,
                batch.get(RAW_STATE),
                batch[RAW_IMAGE],
                batch[RAW_WRIST],
                previous_action=self._last_executed_action,
            )

        for batch_index in range(batch_size):
            trace = self._trace[self._active_trace[batch_index]]
            if self.advance_mode != "gt":
                trace["end_probs"].append(
                    {
                        "episode_timestep": self._episode_step,
                        "skill_step": self._skill_step[batch_index],
                        "prob": float(probability[batch_index]),
                        "progress": float(progress[batch_index]),
                    }
                )
            self._skill_step[batch_index] += 1
            trace["length"] = self._skill_step[batch_index]
            if batch_index in activated_at_start:
                continue
            if batch_index in self._pending_advance:
                continue
            if self.advance_mode == "gt":
                lengths = (
                    self._gt_lengths
                    if self.skill_source == "gt"
                    else self._reference_lengths
                )
                target = lengths[batch_index][self._cursor[batch_index]]
                fired = self._skill_step[batch_index] >= max(1, int(target))
            else:
                fired = self._terminator_fired(
                    float(progress[batch_index]), float(probability[batch_index])
                )
                # GT-skill panels know the canonical skill duration. Ignore a
                # noisy terminator firing during the configurable initial
                # fraction of that duration, while leaving predicted/external
                # skill panels untouched.
                if (
                    fired
                    and self.skill_source == "gt"
                    and self.gt_termination_min_fraction > 0.0
                ):
                    target = self._gt_lengths[batch_index][self._cursor[batch_index]]
                    minimum_step = max(
                        1,
                        int(
                            math.ceil(
                                max(1, int(target))
                                * self.gt_termination_min_fraction
                            )
                        ),
                    )
                    if self._skill_step[batch_index] < minimum_step:
                        fired = False
                if self.max_skill_length > 0:
                    fired |= self._skill_step[batch_index] >= self.max_skill_length

            if not fired:
                continue
            self._skill_end_fired[batch_index] = True
            if self._can_advance(batch_index):
                self._pending_advance.add(batch_index)
            elif self.immediate_replan_on_skill_end or not self._action_queue:
                self._episode_done[batch_index] = True
            else:
                self._pending_episode_done.add(batch_index)

        # Immediate mode interrupts the current action chunk as soon as this
        # observation fires the boundary. Fixed mode preserves the old behavior:
        # activate only when the queue has naturally reached a replanning point.
        if self._pending_advance and (
            not self._action_queue or self.immediate_replan_on_skill_end
        ):
            if self.immediate_replan_on_skill_end:
                self._action_queue.clear()
            self._activate_pending_advances(batch, device)

        if not self._action_queue:
            codes = self._current_codes(batch_size, device)
            action_batch = dict(batch)
            action_batch["skill_code"] = codes
            action_batch["skill_sequence"] = codes[:, None]
            action_batch["skill_index"] = torch.zeros(
                batch_size, dtype=torch.long, device=device
            )
            self._apply_stage2_vlm_start(action_batch)
            if getattr(self.policy, "name", None) == "skill_vla_stage2":
                action_batch[STAGE2_VLM_CACHE_ID] = torch.as_tensor(
                    self._skill_order, dtype=torch.long, device=device
                )
            chunk = self.policy.predict_action_chunk(action_batch)
            self._action_queue.extend(
                chunk[:, : self.n_action_steps].transpose(0, 1)
            )
        self._episode_step += 1
        return self._action_queue.popleft()

    def get_skill_trace(self) -> list[dict]:
        return self._trace

    def get_episode_done(self) -> list[bool]:
        """Return final-skill completion flags for the current batch."""
        return list(self._episode_done)

    def get_skill_end_fired(self) -> list[bool]:
        """Return terminator/GT-boundary events detected on the current observation."""
        return list(self._skill_end_fired)

    def get_progress_threshold(self) -> float:
        return self.progress_threshold

    def get_end_threshold(self) -> float:
        return self.end_threshold

    def get_gt_timeline(self) -> dict[int, list[dict]]:
        if self.skill_source != "gt" and self._references_synthetic:
            return {}
        sequences = self._sequences if self.skill_source == "gt" else self._references
        lengths = self._gt_lengths if self.skill_source == "gt" else self._reference_lengths
        if sequences is None or lengths is None:
            return {}
        return {
            batch_index: [
                {"token": int(code), "length": int(length)}
                for code, length in zip(
                    sequences[batch_index], lengths[batch_index], strict=True
                )
            ]
            for batch_index in range(len(sequences))
        }

    def get_optim_params(self):
        return self.policy.get_optim_params()

    def forward(self, batch, *args, **kwargs):
        return self.policy.forward(batch, *args, **kwargs)

    def predict_action_chunk(self, batch, **kwargs):
        return self.policy.predict_action_chunk(batch, **kwargs)


def _repeat_to_length(sequences: list, count: int) -> list:
    return list(islice(cycle(sequences), count)) if sequences else []


def _language_oracle_maps(
    envs: dict,
    specs: list[dict],
    task_descriptions: dict[int, str],
    n_episodes: int,
) -> list[dict]:
    per_model = [
        map_sequences_to_tasks(
            task_descriptions,
            load_sequences_by_language(spec["skill_dataset_dir"]),
        )
        for spec in specs
    ]
    # Every panel that reads its skill from the predictor and its boundary from the
    # terminator ignores the GT sequence, so a suite with no GT coverage can still be
    # evaluated on a synthetic placeholder instead of losing the task outright.
    gt_optional = all(
        _normalize_skill_source(spec["skill_source"]) != "gt"
        and _normalize_advance_mode(spec["advance_mode"]) != "gt"
        for spec in specs
    )
    maps = [dict() for _ in specs]
    for task_group, group in envs.items():
        for task_id in list(group):
            sequences = [model.get(int(task_id), []) for model in per_model]
            if not all(sequences):
                if not gt_optional:
                    log.warning("task_id=%s is absent from at least one model dataset; dropping it.", task_id)
                    entry = group.pop(task_id)
                    if hasattr(entry, "close"):
                        entry.close()
                    continue
                log.warning(
                    "task_id=%s has no GT skills; evaluating it on predictor/terminator only.",
                    task_id,
                )
                sequences = [
                    model_sequences or [[dict(_SYNTHETIC_SKILL)]]
                    for model_sequences in sequences
                ]
            for index, model_sequences in enumerate(sequences):
                maps[index][(task_group, int(task_id))] = _repeat_to_length(
                    model_sequences, n_episodes
                )
    return maps


def _episode_exact_oracle_maps(
    envs: dict,
    specs: list[dict],
    suite_name: str,
    n_episodes: int,
    init_state_arrays: dict[tuple[str, int], np.ndarray],
) -> list[dict]:
    episode_data = [
        load_episode_exact_data(
            spec["skill_dataset_dir"], spec["eval_init_states_path"], suite_name
        )
        for spec in specs
    ]
    maps = [dict() for _ in specs]
    for task_group, group in envs.items():
        for task_id in list(group):
            indexed = [
                {record["episode_index"]: record for record in data.get(int(task_id), [])}
                for data in episode_data
            ]
            common = sorted(set.intersection(*(set(model) for model in indexed))) if all(indexed) else []
            if len(common) < n_episodes:
                log.warning(
                    "task_id=%s has %d shared exact episodes; dropping it.",
                    task_id,
                    len(common),
                )
                entry = group.pop(task_id)
                if hasattr(entry, "close"):
                    entry.close()
                continue
            common = common[:n_episodes]
            init_states = np.stack(
                [indexed[0][episode]["init_state"] for episode in common]
            ).astype(np.float64)
            # Applied by the lazy env factory when this task's env is created.
            init_state_arrays[(task_group, int(task_id))] = init_states
            for index, model in enumerate(indexed):
                maps[index][(task_group, int(task_id))] = [
                    model[episode]["skills"] for episode in common
                ]
    return maps


def _reset_init_state_ids(envs: dict) -> None:
    for group in envs.values():
        for vector_env in group.values():
            for sub_env in getattr(vector_env, "envs", []):
                base = sub_env.unwrapped
                base.init_state_id = base.episode_index


def _lazy_env_factory(
    cfg,
    suite_name: str,
    task_id: int,
    init_state_arrays: dict[tuple[str, int], np.ndarray],
):
    def build():
        env_cfg = copy.deepcopy(cfg.env)
        env_cfg.task = suite_name
        env_cfg.task_ids = [task_id]
        vec = make_env(
            env_cfg,
            n_envs=cfg.eval.batch_size,
            use_async_envs=cfg.eval.use_async_envs,
            trust_remote_code=cfg.trust_remote_code,
        )[suite_name][task_id]
        init_states = init_state_arrays.get((suite_name, int(task_id)))
        sub_envs = getattr(vec, "envs", None)
        if init_states is not None and sub_envs is None:
            raise RuntimeError("Episode-exact eval requires SyncVectorEnv.")
        for sub_env in sub_envs or []:
            base = sub_env.unwrapped
            if init_states is not None:
                base.init_states = True
                base._init_states = init_states
            base.init_state_id = base.episode_index
        return vec

    return build


def _make_lazy_envs(
    cfg, init_state_arrays: dict[tuple[str, int], np.ndarray]
) -> dict[str, dict[int, object]]:
    """Build {suite: {task_id: env_factory}} without instantiating simulators.

    Every LIBERO env keeps an EGL render context on the GPU, so instantiating a
    multi-task chunk upfront stacks batch_size contexts per task next to the
    policy weights and can OOM. eval_policy_all calls each factory right before
    its task runs and closes the env right after, so only one task's contexts
    are resident at a time. init_state_arrays is filled by the episode-exact
    oracle stage before any factory runs and is read at env-creation time.
    """
    from lerobot.envs.libero import _get_suite, _select_task_ids

    suite_names = [s.strip() for s in str(cfg.env.task).split(",") if s.strip()]
    envs: dict[str, dict[int, object]] = {}
    for suite_name in suite_names:
        total = len(_get_suite(suite_name).tasks)
        for tid in _select_task_ids(total, cfg.env.task_ids):
            envs.setdefault(suite_name, {})[tid] = _lazy_env_factory(
                cfg, suite_name, tid, init_state_arrays
            )
    return envs


def _policy_config(spec: dict, base, device: torch.device):
    config = PreTrainedConfig.from_pretrained(spec["policy_path"])
    if getattr(config, "type", getattr(config, "model_type", "")) == "skill_vla_stage2":
        expected_stage2_mode = str(
            spec.get("stage2_mode", "likelihood")
        ).strip().lower()
        loaded_stage2_mode = str(
            getattr(config, "stage2_mode", "likelihood")
        ).strip().lower()
        if loaded_stage2_mode != expected_stage2_mode:
            raise RuntimeError(
                "Checkpoint contract changed while starting evaluation: "
                f"stage2_mode resolved={expected_stage2_mode}, "
                f"loaded={loaded_stage2_mode} at {spec['policy_path']}"
            )
        expected_noise_mode = str(
            spec.get("dsbc_noise_output_mode", "shared")
        ).strip().lower()
        loaded_noise_mode = str(
            getattr(config, "dsbc_noise_output_mode", "shared")
        ).strip().lower()
        if loaded_noise_mode != expected_noise_mode:
            raise RuntimeError(
                "Checkpoint contract changed while starting evaluation: "
                f"dsbc_noise_output_mode resolved={expected_noise_mode}, "
                f"loaded={loaded_noise_mode} at {spec['policy_path']}"
            )
        for field, default in (
            ("dsbc_frs_num_steps", 10),
            ("dsbc_anchor_seed", 0),
            ("dsbc_latent_timesteps", 2),
        ):
            expected = int(spec.get(field, default))
            loaded = int(getattr(config, field, default))
            if loaded != expected:
                raise RuntimeError(
                    "Checkpoint contract changed while starting evaluation: "
                    f"{field} resolved={expected}, loaded={loaded} "
                    f"at {spec['policy_path']}"
                )
        for field, default in (
            ("dsbc_reader", "final"),
            ("dsbc_latent_predictor_enabled", False),
            ("dsbc_latent_loss_weight", 1.0),
        ):
            expected = spec.get(field, default)
            loaded = getattr(config, field, default)
            if loaded != expected:
                raise RuntimeError(
                    "Checkpoint contract changed while starting evaluation: "
                    f"{field} resolved={expected!r}, loaded={loaded!r} "
                    f"at {spec['policy_path']}"
                )
    architecture = str(spec.get("architecture", "vsa_perceiver_crossattn"))
    architecture_label = str(spec.get("architecture_label", ""))
    loaded_architecture_label = str(getattr(config, "architecture_label", ""))
    historical_arch0_alias = (
        architecture == COND_GEMMA_ARCHITECTURE
        and str(spec.get("architecture_revision", ""))
        == COND_GEMMA_ARCHITECTURE_REVISION
        and loaded_architecture_label == "arch1"
        and architecture_label == "arch0"
    )
    historical_arch2_alias = (
        architecture == "vsa_perceiver_crossattn"
        and str(spec.get("architecture_revision", ""))
        == "interleaved_direct1024_v3"
        and str(spec.get("vision_conditioning_mode", ""))
        == "interleaved_cross_attention"
        and loaded_architecture_label == "arch2"
        and architecture_label == "arch2_2"
    )
    if (
        loaded_architecture_label
        and loaded_architecture_label != architecture_label
        and not historical_arch0_alias
        and not historical_arch2_alias
    ):
        raise RuntimeError(
            "Checkpoint contract changed while starting evaluation: "
            f"architecture_label resolved={architecture_label}, "
            f"loaded={loaded_architecture_label} at {spec['policy_path']}"
        )
    config.architecture_label = architecture_label
    config.eval_legacy_vsa = bool(spec.get("eval_legacy_vsa", False))
    config.eval_vsa_revision = str(spec.get("eval_vsa_revision", ""))
    if architecture == COND_GEMMA_ARCHITECTURE:
        loaded_architecture = str(getattr(config, "architecture", ""))
        if (
            not bool(spec.get("architecture_inferred", False))
            and loaded_architecture != COND_GEMMA_ARCHITECTURE
        ):
            raise RuntimeError(
                "Checkpoint contract changed while starting evaluation: "
                f"architecture resolved={COND_GEMMA_ARCHITECTURE}, "
                f"loaded={loaded_architecture} at {spec['policy_path']}"
            )
        expected_route = normalize_conditioning_route(
            spec.get("conditioning_route", "state_cond")
        )
        actual_route = normalize_conditioning_route(
            getattr(config, "conditioning_route", "state_cond")
        )
        if actual_route != expected_route:
            raise RuntimeError(
                "Checkpoint contract changed while starting evaluation: "
                f"conditioning_route resolved={expected_route}, loaded={actual_route} "
                f"at {spec['policy_path']}"
            )
        # skillVLA_real checkpoints predate the explicit architecture fields.
        # The resolver verified the raw config; materialize that contract before
        # construction so the current branch builds the exact Cond-Gemma path.
        config.architecture = COND_GEMMA_ARCHITECTURE
        config.architecture_revision = str(
            spec.get(
                "architecture_revision", COND_GEMMA_ARCHITECTURE_REVISION
            )
        )
        config.conditioning_route = expected_route
        config.num_visual_latents_per_camera = int(
            spec.get("num_visual_latents_per_camera", 32)
        )
        config.visual_perceiver_width = int(
            spec.get("visual_perceiver_width", 1024)
        )
    else:
        if not config.eval_legacy_vsa:
            expected_mode = str(
                spec.get("vision_conditioning_mode", "interleaved_cross_attention")
            )
            actual_mode = str(
                getattr(config, "vision_conditioning_mode", "interleaved_cross_attention")
            )
            if actual_mode != expected_mode:
                raise RuntimeError(
                    "Checkpoint contract changed while starting evaluation: "
                    f"vision_conditioning_mode resolved={expected_mode}, "
                    f"loaded={actual_mode} at {spec['policy_path']}"
                )
        config.num_visual_latents_per_camera = int(
            spec.get(
                "num_visual_latents_per_camera",
                8 if config.eval_vsa_revision == "legacy_alternating_v1" else 32,
            )
        )
        config.visual_perceiver_width = int(
            spec.get(
                "visual_perceiver_width",
                384 if config.eval_legacy_vsa else 1024,
            )
        )
        for field in (
            "include_state_in_visual_crossattn",
            "include_skill_in_visual_crossattn",
        ):
            expected = bool(spec.get(field, False))
            if config.eval_legacy_vsa:
                setattr(config, field, expected)
                continue
            actual = bool(getattr(config, field, False))
            if actual != expected:
                raise RuntimeError(
                    f"Checkpoint contract changed while starting evaluation: {field} "
                    f"resolved={expected}, loaded={actual} at {spec['policy_path']}"
                )
    config.pretrained_path = Path(spec["policy_path"])
    config.device = str(device)
    config.use_amp = base.use_amp
    config.n_action_steps = base.n_action_steps
    config.compile_model = False
    config.gradient_checkpointing = False
    for field in (
        "fsq_path",
        "dino_model_path",
        "tokenizer_path",
    ):
        setattr(config, field, spec[field])
    # The VSA/cond DINO may intentionally differ from the FSQ terminator DINO.
    # Ignore legacy policy overrides and reconstruct the terminator from FSQ.pt.
    config.terminator_dino_model_path = None
    return config


def _ensure_skill_runtime_steps(
    preprocessor,
    policy_config,
    *,
    needs_predictor: bool,
    needs_terminator: bool,
) -> None:
    """Add runtime-only steps absent from predictor/terminator-free checkpoints."""
    from lerobot.policies.skillVLA.processor_skillVLA import (  # noqa: PLC0415
        SkillVLAPrepareStateTokenizerProcessorStep,
        SkillVLAPreserveRawStateProcessorStep,
    )
    from lerobot.policies.skill_expert.processor_skill_expert import (  # noqa: PLC0415
        EpisodeStartXYZGroundingProcessorStep,
    )
    from lerobot.processor import (  # noqa: PLC0415
        DeviceProcessorStep,
        NormalizerProcessorStep,
        TokenizerProcessorStep,
    )

    steps = list(preprocessor.steps)
    proprio_grounding = str(
        getattr(policy_config, "proprio_grounding", "none") or "none"
    ).strip().lower().replace("-", "_")
    grounded_policy_types = {"skill_expert", "skill_vla_stage2"}
    if (
        policy_config.type in grounded_policy_types
        and proprio_grounding == "episode_start_xyz"
    ):
        if not any(
            isinstance(step, EpisodeStartXYZGroundingProcessorStep)
            for step in steps
        ):
            normalizer_index = next(
                (
                    index
                    for index, step in enumerate(steps)
                    if isinstance(step, NormalizerProcessorStep)
                ),
                None,
            )
            if normalizer_index is None:
                raise ValueError(
                    "Cannot add episode-start proprio grounding: saved "
                    "preprocessor has no normalizer step."
                )
            preserve_index = next(
                (
                    index
                    for index, step in enumerate(steps)
                    if isinstance(step, SkillVLAPreserveRawStateProcessorStep)
                ),
                normalizer_index,
            )
            steps.insert(
                min(normalizer_index, preserve_index),
                EpisodeStartXYZGroundingProcessorStep(),
            )
    elif proprio_grounding != "none" and policy_config.type in grounded_policy_types:
        raise ValueError(
            f"Unsupported skill-policy proprio_grounding={proprio_grounding!r}."
        )
    elif proprio_grounding != "none":
        raise ValueError(
            "episode-start proprio grounding is implemented only for "
            "skill_expert and skill_vla_stage2 evaluation; refusing to evaluate "
            f"policy.type={policy_config.type!r} with an ungrounded input."
        )
    if needs_terminator and not any(
        isinstance(step, SkillVLAPreserveRawStateProcessorStep) for step in steps
    ):
        normalizer_index = next(
            (
                index
                for index, step in enumerate(steps)
                if isinstance(step, NormalizerProcessorStep)
            ),
            None,
        )
        if normalizer_index is None:
            raise ValueError(
                "Cannot add external terminator inputs: saved preprocessor has no "
                "normalizer step."
            )
        steps.insert(normalizer_index, SkillVLAPreserveRawStateProcessorStep())

    if needs_predictor:
        prepare_index = next(
            (
                index
                for index, step in enumerate(steps)
                if isinstance(step, SkillVLAPrepareStateTokenizerProcessorStep)
            ),
            None,
        )
        if prepare_index is None:
            device_index = next(
                (
                    index
                    for index, step in enumerate(steps)
                    if isinstance(step, DeviceProcessorStep)
                ),
                len(steps),
            )
            steps.insert(
                device_index,
                SkillVLAPrepareStateTokenizerProcessorStep(
                    max_state_dim=policy_config.max_state_dim
                ),
            )
            prepare_index = device_index

        tokenizer_index = next(
            (
                index
                for index, step in enumerate(steps)
                if isinstance(step, TokenizerProcessorStep)
            ),
            None,
        )
        tokenizer_path = str(policy_config.tokenizer_path)
        if tokenizer_index is None:
            steps.insert(
                prepare_index + 1,
                TokenizerProcessorStep(
                    tokenizer_name=tokenizer_path,
                    max_length=policy_config.tokenizer_max_length,
                    padding_side="right",
                    padding="max_length",
                ),
            )
        elif str(steps[tokenizer_index].tokenizer_name) != tokenizer_path:
            steps[tokenizer_index] = TokenizerProcessorStep(
                tokenizer_name=tokenizer_path,
                max_length=policy_config.tokenizer_max_length,
                padding_side="right",
                padding="max_length",
            )

    preprocessor.steps = steps


def _saved_preprocessor_step_names(pretrained_path: str | Path) -> set[str]:
    config_path = (
        Path(pretrained_path) / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
    )
    config = json.loads(config_path.read_text())
    return {
        str(step.get("registry_name") or step.get("class_name") or "")
        for step in config.get("steps", [])
    }


def _build_context(spec: dict, cfg, device: torch.device) -> dict:
    skill_source = _normalize_skill_source(spec["skill_source"])
    advance_mode = _normalize_advance_mode(spec["advance_mode"])
    terminator_variant = _normalize_terminator_variant(
        spec.get("terminator_variant", "state_image")
    )
    external_skill_model = str(spec.get("external_skill_model") or "").strip()
    # The predictor and terminator overlays may come from different checkpoints;
    # both fall back to external_skill_model when a role is not split out.
    external_predictor_model = str(
        spec.get("external_predictor_model") or external_skill_model
    ).strip()
    external_terminator_model = str(
        spec.get("external_terminator_model") or external_skill_model
    ).strip()
    policy_config = _policy_config(spec, cfg.policy, device)
    log.info(
        "[%s] proprio grounding=%s (checkpoint contract; no eval YAML override).",
        spec["label"],
        getattr(policy_config, "proprio_grounding", "none"),
    )
    if policy_config.type == "skill_vla_stage2":
        stage2_mode = str(getattr(policy_config, "stage2_mode", "likelihood"))
        dsbc_detail = (
            ", noise_output="
            f"{getattr(policy_config, 'dsbc_noise_output_mode', 'shared')}"
            if stage2_mode == "dsbc"
            else ""
        )
        log.info(
            "[%s] Stage-2 mode=%s%s architecture=%s revision=%s "
            "conditioning_route=%s; VLM start=%s.",
            spec["label"],
            stage2_mode,
            dsbc_detail,
            spec.get("architecture"),
            spec.get("architecture_revision"),
            spec.get("conditioning_route"),
            STAGE2_VLM_START_CONTRACT,
        )
    elif spec.get("architecture") == COND_GEMMA_ARCHITECTURE:
        log.info(
            "[%s] Stage-1 %s architecture=%s revision=%s conditioning_route=%s, "
            "loss=%s%s.",
            spec["label"],
            spec.get("architecture_label"),
            spec.get("architecture"),
            spec.get("architecture_revision"),
            spec.get("conditioning_route"),
            spec.get("action_loss_mode"),
            " (inferred from skillVLA_real metadata)"
            if spec.get("architecture_inferred")
            else "",
        )
    else:
        log.info(
            "[%s] Stage-1 %s architecture=%s revision=%s mode=%s, "
            "visual cross-attention queries=%s, loss=%s.",
            spec["label"],
            spec.get("architecture_label"),
            spec.get("architecture"),
            spec.get("architecture_revision"),
            spec.get("vision_conditioning_mode"),
            spec.get("visual_crossattn_queries"),
            spec.get("action_loss_mode"),
        )
    policy = make_policy(
        cfg=policy_config, env_cfg=cfg.env, rename_map=cfg.rename_map
    )
    if skill_source == "external":
        if not external_predictor_model:
            raise ValueError(
                f"[{spec['label']}] skill_source=external requires "
                "external_predictor_model or external_skill_model."
            )
        if policy_config.type not in {"skill_expert", "skill_vla_stage2"}:
            raise ValueError(
                f"[{spec['label']}] external predictor override is supported only "
                "for skill_expert and skill_vla_stage2 checkpoints."
            )
        policy.load_external_skill_predictor(external_predictor_model)
        log.info(
            "[%s] overlaid external predictor from %s.",
            spec["label"],
            external_predictor_model,
        )
    if advance_mode == "external":
        if not external_terminator_model:
            raise ValueError(
                f"[{spec['label']}] advance_mode=external requires "
                "external_terminator_model or external_skill_model."
            )
        if policy_config.type not in {"skill_expert", "skill_vla_stage2"}:
            raise ValueError(
                f"[{spec['label']}] external terminator override is supported only "
                "for skill_expert and skill_vla_stage2 checkpoints."
            )
        if terminator_variant == "image_only":
            policy.load_external_image_only_terminator(external_terminator_model)
        else:
            policy.load_external_terminator(external_terminator_model)
        log.info(
            "[%s] overlaid external %s terminator from %s.",
            spec["label"],
            terminator_variant,
            external_terminator_model,
        )
    elif advance_mode == "original":
        if policy_config.type not in {"skill_expert", "skill_vla_stage2"}:
            raise ValueError(
                f"[{spec['label']}] original FSQ terminator is supported only "
                "for skill_expert and skill_vla_stage2 checkpoints."
            )
        if terminator_variant != "state_image":
            raise ValueError(
                f"[{spec['label']}] advance_mode=original supports only "
                "terminator_variant=state_image."
            )
        _attach_original_terminator(policy, spec["fsq_path"])
        log.info(
            "[%s] attached pristine FSQ terminator from %s.",
            spec["label"],
            spec["fsq_path"],
        )
    policy.eval()
    terminator = (
        CheckpointTerminator(policy, terminator_variant)
        if advance_mode != "gt"
        else None
    )
    # A Stage-1 GT run does not need its predictor/VLM and can release it.  Stage 2
    # is different: the likelihood blocks always consume the pristine frozen VLM
    # memory owned by skill_predictor, even when the injected skill itself is GT.
    if (
        skill_source == "gt"
        and policy_config.type == "skill_expert"
        and policy.model.skill_predictor is not None
    ):
        policy.model.skill_predictor = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
        log.info("[%s] released unused predictor.", spec["label"])
    if skill_source != "gt" and policy.model.skill_predictor is None:
        raise ValueError(f"[{spec['label']}] checkpoint predictor is unavailable.")

    wrapper = Stage1OraclePolicy(
        policy,
        terminator,
        skill_source=skill_source,
        advance_mode=advance_mode,
        end_mode=os.environ["SKILL_END_MODE"],
        end_threshold=float(os.environ["SKILL_END_THRESHOLD"]),
        progress_threshold=float(os.environ["SKILL_END_PROGRESS_THRESHOLD"]),
        max_skill_length=int(os.environ["INFERENCE_SKILL_MAX_LENGTH"]),
        n_action_steps=policy_config.n_action_steps,
        immediate_replan_on_skill_end=(
            os.environ.get("IMMEDIATE_REPLAN_ON_SKILL_END", "false").lower()
            == "true"
        ),
        gt_termination_min_fraction=float(
            os.environ.get("GT_TERMINATION_MIN_FRACTION", "0")
        ),
    )
    wrapper.eval()
    overrides = {
        "device_processor": {"device": str(device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    saved_steps = _saved_preprocessor_step_names(policy_config.pretrained_path)
    if "tokenizer_processor" in saved_steps:
        # Imported checkpoints can retain the source server's absolute tokenizer
        # path inside policy_preprocessor.json. The resolver already relocated the
        # corresponding config.json path. Apply it even for GT-skill evaluation:
        # the saved pipeline constructs every configured step before inference.
        overrides["tokenizer_processor"] = {
            "tokenizer_name": str(policy_config.tokenizer_path)
        }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=policy_config.pretrained_path,
        preprocessor_overrides=overrides,
    )
    _ensure_skill_runtime_steps(
        preprocessor,
        policy_config,
        needs_predictor=skill_source != "gt",
        needs_terminator=advance_mode != "gt",
    )
    # rollout() resets the policy but not its separately-owned preprocessor.
    # Let the wrapper clear the cached episode-start xyz at every rollout reset.
    wrapper._runtime_preprocessor = preprocessor
    return {
        "policy": wrapper,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "config": policy_config,
    }


def _panel_dir(index: int, label: str) -> str:
    safe = "".join(character if character.isalnum() or character in "._-" else "-" for character in label)
    return f"{index:02d}_{safe.strip('-_') or 'model'}"


def _panel_task_names(oracle_map: dict) -> set[str]:
    return {
        f"{task_group}_{int(task_id)}"
        for task_group, task_id in oracle_map
    }


def _panel_artifacts_complete(
    panel_root: Path,
    task_names: set[str],
    cfg,
) -> bool:
    expected_videos = min(
        int(cfg.eval.n_episodes), int(cfg.eval.max_videos_per_task)
    )
    has_checkable_output = expected_videos > 0 or bool(cfg.eval.skill_html)
    if not has_checkable_output:
        return False
    for task_name in task_names:
        if expected_videos > 0:
            videos = list((panel_root / "videos" / task_name).glob("eval_episode_*.mp4"))
            if len(videos) < expected_videos or any(
                video.stat().st_size == 0 for video in videos
            ):
                return False
        if cfg.eval.skill_html:
            task_group, task_id = task_name.rsplit("_", 1)
            html = (
                panel_root
                / "skill_html"
                / f"{task_group}_task{int(task_id):02d}"
                / "skill_trace.html"
            )
            if not html.is_file() or html.stat().st_size == 0:
                return False
    return True


def _eval_info_name() -> str:
    task_tag = os.environ.get("TASK_TAG", "").strip()
    return f"eval_info_{task_tag}.json" if task_tag else "eval_info.json"


def _panel_cache_path(panel_root: Path) -> Path:
    output_root = panel_root.parent.parent
    return output_root / "metrics" / "panel_cache" / panel_root.name / _eval_info_name()


def _legacy_panel_cache_path(panel_root: Path) -> Path:
    """Location used before eval JSON files were collected under metrics/."""
    return panel_root / _eval_info_name()


def _panel_signature(spec: dict, task_names: set[str], cfg) -> dict:
    return {
        "policy_path": spec["policy_path"],
        "external_skill_model": spec.get("external_skill_model") or "",
        "external_predictor_model": spec.get("external_predictor_model") or "",
        "external_terminator_model": spec.get("external_terminator_model") or "",
        "skill_source": spec["skill_source"],
        "advance_mode": spec["advance_mode"],
        "terminator_variant": spec.get("terminator_variant", "state_image"),
        "architecture": spec.get("architecture"),
        "architecture_label": spec.get("architecture_label"),
        "architecture_revision": spec.get("architecture_revision"),
        "conditioning_route": spec.get("conditioning_route"),
        "proprio_grounding": spec.get("proprio_grounding", "none"),
        "stage2_mode": spec.get("stage2_mode"),
        "dsbc_noise_output_mode": spec.get("dsbc_noise_output_mode"),
        "dsbc_frs_num_steps": spec.get("dsbc_frs_num_steps"),
        "dsbc_anchor_seed": spec.get("dsbc_anchor_seed"),
        "stage2_vlm_start_contract": (
            STAGE2_VLM_START_CONTRACT
            if spec.get("mode") == "stage2"
            else None
        ),
        "previous_checkpoint": spec.get("previous_checkpoint", False),
        "vision_conditioning_mode": spec.get("vision_conditioning_mode"),
        "include_state_in_visual_crossattn": spec.get(
            "include_state_in_visual_crossattn", False
        ),
        "include_skill_in_visual_crossattn": spec.get(
            "include_skill_in_visual_crossattn", False
        ),
        "tasks": sorted(task_names),
        "n_episodes": int(cfg.eval.n_episodes),
        "n_action_steps": int(cfg.policy.n_action_steps),
        "seed": int(cfg.seed),
        "replanning_mode": (
            "immediate_skill_end_v1"
            if os.environ.get("IMMEDIATE_REPLAN_ON_SKILL_END", "false").lower()
            == "true"
            else "fixed_chunk_v1"
        ),
        "immediate_replan_on_skill_end": os.environ.get(
            "IMMEDIATE_REPLAN_ON_SKILL_END", "false"
        ).lower()
        == "true",
        "skill_end_mode": os.environ["SKILL_END_MODE"],
        "skill_end_threshold": os.environ["SKILL_END_THRESHOLD"],
        "skill_end_progress_threshold": os.environ[
            "SKILL_END_PROGRESS_THRESHOLD"
        ],
        "gt_termination_min_fraction": os.environ.get(
            "GT_TERMINATION_MIN_FRACTION", "0"
        ),
        "inference_skill_max_length": os.environ["INFERENCE_SKILL_MAX_LENGTH"],
    }


def _load_resumed_panel_info(
    panel_root: Path,
    spec: dict,
    task_names: set[str],
    cfg,
) -> tuple[dict | None, str | None]:
    if not _panel_artifacts_complete(panel_root, task_names, cfg):
        return None, None
    signature = _panel_signature(spec, task_names, cfg)
    cache_paths = (
        (_panel_cache_path(panel_root), "metrics cache"),
        (_legacy_panel_cache_path(panel_root), "legacy metrics cache"),
    )
    found_signature_cache = False
    for cache_path, source in cache_paths:
        if cache_path.is_file():
            found_signature_cache = True
            cached = json.loads(cache_path.read_text())
            if cached.get("signature") == signature and isinstance(
                cached.get("info"), dict
            ):
                return cached["info"], source
    if found_signature_cache:
        # A cache from a known evaluator exists but its contract changed (for
        # example likelihood -> DSBC or live-frame -> skill-start VLM input).
        # Never fall through to the pre-signature artifact compatibility path.
        return None, None
    # Compatibility for panels completed before per-panel caches were introduced.
    return (
        {
            "overall": {},
            "per_task": [],
            "resume": {
                "status": "reused_existing_artifacts",
                "metrics_available": False,
                "tasks": sorted(task_names),
            },
        },
        "existing videos/HTML (metrics unavailable)",
    )


def _save_panel_info(
    panel_root: Path,
    spec: dict,
    task_names: set[str],
    cfg,
    info: dict,
) -> None:
    cache_path = _panel_cache_path(panel_root)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "signature": _panel_signature(spec, task_names, cfg),
                "info": info,
            },
            indent=2,
        )
    )


def _stitch_panels(
    panels: list[tuple[Path, str]],
    output_dir: Path,
    columns: int,
    *,
    task_names: set[str] | None = None,
) -> None:
    if len(panels) < 2:
        return
    grid_columns = columns if columns > 0 else len(panels)
    try:
        import imageio.v2 as imageio
        from compare_videos import even, label_bar, load_font, make_panel, read_video
    except Exception as error:  # noqa: BLE001
        log.warning("Side-by-side video generation skipped: %s", error)
        return

    height = 256
    bar_height = even(max(20, height // 9))
    font = load_font(int(bar_height * 0.62))
    first_dir = panels[0][0]
    written = 0
    for task_dir in sorted(path for path in first_dir.glob("*") if path.is_dir()):
        if task_names is not None and task_dir.name not in task_names:
            continue
        for first_video in sorted(task_dir.glob("eval_episode_*.mp4")):
            videos = [directory / task_dir.name / first_video.name for directory, _ in panels]
            if not all(video.is_file() for video in videos):
                continue
            destination = output_dir / task_dir.name / first_video.name
            # Concurrent fanout jobs may both reach a completed task; the first
            # finished stitch wins and later jobs skip it.
            if destination.is_file() and destination.stat().st_size > 0:
                continue
            reads = [read_video(video) for video in videos]
            frame_sets = [read[0] for read in reads]
            if any(not frames for frames in frame_sets):
                continue
            bars = []
            for (_, label), frames in zip(panels, frame_sets, strict=True):
                frame_height, frame_width = frames[0].shape[:2]
                width = even(max(2, round(frame_width * height / frame_height)))
                bars.append(label_bar(width, bar_height, label, font))
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(
                f"{destination.stem}.tmp{os.getpid()}.mp4"
            )
            writer = imageio.get_writer(
                str(temporary),
                fps=reads[0][1],
                codec="libx264",
                quality=8,
                macro_block_size=None,
            )
            for frame_index in range(max(len(frames) for frames in frame_sets)):
                tiles = [
                    make_panel(frames[min(frame_index, len(frames) - 1)], height, bar)
                    for frames, bar in zip(frame_sets, bars, strict=True)
                ]
                rows = [
                    np.hstack(tiles[start : start + grid_columns])
                    for start in range(0, len(tiles), grid_columns)
                ]
                max_width = max(row.shape[1] for row in rows)
                rows = [
                    row
                    if row.shape[1] == max_width
                    else np.pad(row, ((0, 0), (0, max_width - row.shape[1]), (0, 0)))
                    for row in rows
                ]
                frame = np.vstack(rows)
                frame = frame[
                    : frame.shape[0] - frame.shape[0] % 2,
                    : frame.shape[1] - frame.shape[1] % 2,
                ]
                writer.append_data(frame)
            writer.close()
            temporary.replace(destination)
            written += 1
    log.info("Wrote %d side-by-side videos to %s.", written, output_dir)


def _maybe_log_wandb(cfg, infos: dict[str, dict], specs: list[dict]) -> None:
    if not cfg.wandb_project:
        return
    try:
        import wandb

        wandb.init(
            project=cfg.wandb_project,
            name=cfg.job_name,
            config={
                "models": [
                    {
                        "label": spec["label"],
                        "policy_path": spec["policy_path"],
                        "skill_source": spec["skill_source"],
                        "advance_mode": spec["advance_mode"],
                        "terminator_variant": spec.get(
                            "terminator_variant", "state_image"
                        ),
                        "external_skill_model": (
                            spec.get("external_skill_model") or "unused"
                        ),
                        "external_predictor_model": (
                            spec.get("external_predictor_model") or "unused"
                        ),
                        "external_terminator_model": (
                            spec.get("external_terminator_model") or "unused"
                        ),
                        "architecture": spec.get("architecture"),
                        "architecture_label": spec.get("architecture_label"),
                        "architecture_revision": spec.get("architecture_revision"),
                        "conditioning_route": spec.get("conditioning_route"),
                        "proprio_grounding": spec.get(
                            "proprio_grounding", "none"
                        ),
                        "stage2_mode": spec.get("stage2_mode"),
                        "dsbc_noise_output_mode": spec.get(
                            "dsbc_noise_output_mode"
                        ),
                        "dsbc_frs_num_steps": spec.get("dsbc_frs_num_steps"),
                        "dsbc_anchor_seed": spec.get("dsbc_anchor_seed"),
                        "stage2_vlm_start_contract": (
                            STAGE2_VLM_START_CONTRACT
                            if spec.get("mode") == "stage2"
                            else None
                        ),
                        "previous_checkpoint": spec.get(
                            "previous_checkpoint", False
                        ),
                        "vision_conditioning_mode": spec.get(
                            "vision_conditioning_mode"
                        ),
                        "include_state_in_visual_crossattn": spec.get(
                            "include_state_in_visual_crossattn", False
                        ),
                        "include_skill_in_visual_crossattn": spec.get(
                            "include_skill_in_visual_crossattn", False
                        ),
                        "visual_crossattn_queries": spec.get(
                            "visual_crossattn_queries"
                        ),
                        "action_loss_mode": spec.get("action_loss_mode"),
                    }
                    for spec in specs
                ],
                "n_episodes": cfg.eval.n_episodes,
                "n_action_steps": int(cfg.policy.n_action_steps),
                "replanning_mode": (
                    "immediate_skill_end_v1"
                    if os.environ.get(
                        "IMMEDIATE_REPLAN_ON_SKILL_END", "false"
                    ).lower()
                    == "true"
                    else "fixed_chunk_v1"
                ),
            },
        )
        payload = {}
        for label, info in infos.items():
            for key, value in info.get("overall", {}).items():
                if isinstance(value, (int, float)):
                    payload[f"{label}/overall/{key}"] = float(value)
            for task in info.get("per_task", []):
                task_id = task.get("task_id")
                metrics = task.get("metrics", task)
                success = metrics.get("pc_success", metrics.get("success_rate"))
                if task_id is not None and success is not None:
                    payload[f"{label}/task_{int(task_id):02d}/success"] = float(success)
        wandb.log(payload)
        wandb.finish()
    except Exception as error:  # noqa: BLE001
        log.warning("wandb logging failed: %s", error)


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    _run_inline_cuda_guard()
    _mark_startup_ready()
    supported = {"skill_expert", "skill_vla_stage2"}
    if cfg.policy is None or cfg.policy.type not in supported:
        raise ValueError(
            "Skill-conditioned eval requires a skill_expert or "
            "skill_vla_stage2 checkpoint."
        )
    specs = json.loads(os.environ.get("MODELS_JSON", "") or "[]")
    if not specs:
        raise ValueError("MODELS_JSON is empty; resolve the eval config first.")

    device = get_safe_torch_device(cfg.policy.device, log=True)
    set_seed(cfg.seed)
    # Env factories instead of live envs: eval_policy_all creates and closes
    # one task's env at a time, so multi-task chunks don't stack simulator
    # render contexts on the GPU next to the policy weights.
    init_state_arrays: dict[tuple[str, int], np.ndarray] = {}
    envs = _make_lazy_envs(cfg, init_state_arrays)
    try:
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=cfg.env, policy_cfg=cfg.policy
        )
        task_descriptions = _libero_task_descriptions(cfg.env.task)
        episode_exact = all(spec.get("eval_init_states_path") for spec in specs)
        oracle_maps = (
            _episode_exact_oracle_maps(
                envs, specs, cfg.env.task, cfg.eval.n_episodes, init_state_arrays
            )
            if episode_exact
            else _language_oracle_maps(
                envs, specs, task_descriptions, cfg.eval.n_episodes
            )
        )
        if not oracle_maps or not oracle_maps[0]:
            raise RuntimeError("No requested LIBERO tasks matched every model dataset.")

        output_dir = Path(cfg.output_dir)
        infos = {}
        # Fanout jobs evaluate a subset of panels; every job still stitches over
        # the full panel directory list, so whichever job finishes a task last
        # completes that task's side-by-side clips.
        video_panels = [
            (
                output_dir / "panels" / _panel_dir(index, spec["label"]) / "videos",
                spec["label"],
            )
            for index, spec in enumerate(specs)
        ]
        panel_filter = {
            int(value)
            for value in os.environ.get("PANEL_INDICES", "").split(",")
            if value.strip()
        }
        resume = os.environ.get("EVAL_RESUME", "false").lower() == "true"
        current_task_names = _panel_task_names(oracle_maps[0])
        for index, (spec, oracle_map) in enumerate(zip(specs, oracle_maps, strict=True)):
            if panel_filter and index not in panel_filter:
                continue
            panel_root = output_dir / "panels" / _panel_dir(index, spec["label"])
            task_names = _panel_task_names(oracle_map)
            if resume:
                resumed_info, resume_source = _load_resumed_panel_info(
                    panel_root, spec, task_names, cfg
                )
                if resumed_info is not None:
                    infos[spec["label"]] = resumed_info
                    log.warning(
                        "[%s] resume: skipping completed panel from %s.",
                        spec["label"],
                        resume_source,
                    )
                    continue
            log.info(
                "[%s] loading %s (stage2_mode=%s, skill_source=%s, "
                "advance_mode=%s, terminator_variant=%s, predictor=%s, "
                "terminator=%s).",
                spec["label"],
                spec["policy_path"],
                spec.get("stage2_mode", "prior_or_stage1"),
                spec["skill_source"],
                spec["advance_mode"],
                spec.get("terminator_variant", "state_image"),
                spec.get("external_predictor_model")
                or spec.get("external_skill_model")
                or "unused",
                spec.get("external_terminator_model")
                or spec.get("external_skill_model")
                or "unused",
            )
            context = _build_context(spec, cfg, device)
            try:
                _reset_init_state_ids(envs)
                use_gt = spec["skill_source"] == "gt"
                with torch.no_grad(), (
                    torch.autocast(device_type=device.type)
                    if context["config"].use_amp
                    else nullcontext()
                ):
                    info = eval_policy_all(
                        envs=envs,
                        policy=context["policy"],
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=context["preprocessor"],
                        postprocessor=context["postprocessor"],
                        n_episodes=cfg.eval.n_episodes,
                        max_episodes_rendered=cfg.eval.max_videos_per_task,
                        video_frame_stride=cfg.eval.video_frame_stride,
                        video_fps=cfg.eval.video_fps,
                        videos_dir=panel_root / "videos",
                        return_episode_data=False,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                        forced_skill_token_sequences_by_task=(oracle_map if use_gt else None),
                        reference_skill_token_sequences_by_task=(None if use_gt else oracle_map),
                        skill_html_dir=(panel_root / "skill_html" if cfg.eval.skill_html else None),
                        skill_html_train_samples=cfg.eval.skill_html_train_samples,
                        skill_html_skill_latents_path=spec["skill_latents_path"],
                        skill_html_raw_dataset_dir=spec["raw_dataset_dir"],
                        skill_html_image_key=cfg.eval.skill_html_image_key,
                        task_descriptions=task_descriptions,
                    )
                infos[spec["label"]] = info
                _save_panel_info(panel_root, spec, task_names, cfg, info)
                log.info("[%s] overall=%s", spec["label"], info.get("overall"))
            finally:
                del context
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        _stitch_panels(
            video_panels,
            output_dir / "side_by_side",
            int(os.environ.get("GRID_COLUMNS", "0") or 0),
            task_names=current_task_names,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        info_path = metrics_dir / _eval_info_name()
        info_path.write_text(json.dumps(infos, indent=2))
        for label, info in infos.items():
            print(f"{label}: {info.get('overall', {})}")
        print("Saved:", info_path)
        _maybe_log_wandb(cfg, infos, specs)
    finally:
        close_envs(envs)


if __name__ == "__main__":
    eval_main()
