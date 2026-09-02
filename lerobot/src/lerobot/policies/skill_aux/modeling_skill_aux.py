"""Auxiliary-only training for terminator variants and the skill predictor."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import Tensor, nn

from lerobot.policies.pi05.lora import route_plain_to_base
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.skill_expert.modeling_skill_expert import (
    _PREDICTOR_CHECKPOINT_CONTRACT_FIELDS,
    _load_complete_predictor_parameters,
    _load_learned_predictor_parameters,
)
from lerobot.policies.skill_expert.modeling_skill_predictor import FrozenVLMSkillPredictor
from lerobot.policies.skill_expert.modeling_utils import (
    build_fsq_image_only_terminator,
    build_fsq_wrist_only_terminator,
    build_trainable_fsq_terminator,
)
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from lerobot.policies.skillVLA.dataset_skillVLA import (
    SKILL_PREVIOUS_ACTION,
    SKILL_PREVIOUS_ACTION_BOS,
)

from .configuration_skill_aux import SkillAuxConfig
from .modeling_state_terminator import (
    StateSkillMLPTerminator,
    StateSkillRNNTerminator,
)

log = logging.getLogger(__name__)


class SkillAuxModules(nn.Module):
    """Keep checkpoint prefixes compatible with Stage-1 auxiliary loaders."""

    def __init__(self, config: SkillAuxConfig):
        super().__init__()
        self.skill_predictor = (
            FrozenVLMSkillPredictor(config) if config.train_skill_predictor else None
        )
        self.fsq_term_train = None
        if config.train_terminator:
            terminator = build_trainable_fsq_terminator(
                config.fsq_path,
                termination_only=config.terminator_termination_only,
                context=config.terminator_context,
                cameras=config.terminator_cameras,
                default_arch=config.terminator_arch,
                vision_backbone=config.terminator_vision_backbone,
                freeze_vision_encoder=config.terminator_freeze_vision_encoder,
            )
            terminator.freeze_vision_encoder = bool(config.terminator_freeze_vision_encoder)
            terminator.requires_grad_(True)
            if terminator.freeze_vision_encoder:
                terminator.vision_encoder.requires_grad_(False).eval()
            self.fsq_term_train = terminator.to(dtype=torch.float32)
        self.fsq_image_term_train = None
        if config.train_image_only_terminator:
            terminator = build_fsq_image_only_terminator(
                config.fsq_path,
                termination_only=config.image_only_terminator_termination_only,
            )
            terminator.freeze_vision_encoder = bool(
                config.image_only_terminator_freeze_vision_encoder
            )
            terminator.requires_grad_(True)
            if terminator.freeze_vision_encoder:
                terminator.vision_encoder.requires_grad_(False).eval()
            self.fsq_image_term_train = terminator.to(dtype=torch.float32)
        self.fsq_wrist_term_train = None
        if config.train_wrist_only_terminator:
            terminator = build_fsq_wrist_only_terminator(
                config.fsq_path,
                termination_only=config.wrist_only_terminator_termination_only,
            )
            terminator.freeze_vision_encoder = bool(
                config.wrist_only_terminator_freeze_vision_encoder
            )
            terminator.requires_grad_(True)
            if terminator.freeze_vision_encoder:
                terminator.vision_encoder.requires_grad_(False).eval()
            self.fsq_wrist_term_train = terminator.to(dtype=torch.float32)
        self.fsq_state_term_train = (
            StateSkillMLPTerminator(
                state_dim=config.max_state_dim,
                skill_dim=len(config.skill_fsq_levels),
                hidden_dim=config.state_only_terminator_hidden_dim,
                num_layers=config.state_only_terminator_num_layers,
                termination_only=config.state_only_terminator_termination_only,
            ).to(dtype=torch.float32)
            if config.train_state_only_terminator
            else None
        )
        self.fsq_state_rnn_term_train = (
            StateSkillRNNTerminator(
                state_dim=config.max_state_dim,
                skill_dim=len(config.skill_fsq_levels),
                input_dim=config.state_rnn_terminator_input_dim,
                hidden_dim=config.state_rnn_terminator_hidden_dim,
                num_layers=config.state_rnn_terminator_num_layers,
                dropout=config.state_rnn_terminator_dropout,
                termination_only=config.state_rnn_terminator_termination_only,
            ).to(dtype=torch.float32)
            if config.train_state_rnn_terminator
            else None
        )


class SkillAuxPolicy(PreTrainedPolicy):
    """Train only the selected predictor and/or terminator modules."""

    config_class = SkillAuxConfig
    name = "skill_aux"

    def __init__(self, config: SkillAuxConfig, **kwargs):
        skip_auxiliary_warm_start = bool(
            kwargs.pop("_skip_auxiliary_warm_start", False)
        )
        # The generic policy factory supplies these for policies that build
        # internal normalizers. SkillAux uses the shared preprocessing pipeline.
        kwargs.pop("dataset_stats", None)
        kwargs.pop("dataset_meta", None)
        if kwargs:
            raise TypeError(f"Unexpected SkillAuxPolicy arguments: {sorted(kwargs)}")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = SkillAuxModules(config)
        predictor_checkpoint = str(
            config.skill_predictor_checkpoint_path
            or config.auxiliary_checkpoint_path
            or ""
        ).strip()
        terminator_checkpoint = str(
            config.terminator_checkpoint_path
            or config.auxiliary_checkpoint_path
            or ""
        ).strip()
        if (
            config.train_terminator
            and terminator_checkpoint
            and not skip_auxiliary_warm_start
        ):
            self._load_terminator_warm_start(terminator_checkpoint)
        if (
            config.train_skill_predictor
            and predictor_checkpoint
            and not skip_auxiliary_warm_start
        ):
            self._load_complete_predictor_warm_start(predictor_checkpoint)

        levels = torch.tensor(config.skill_fsq_levels, dtype=torch.long)
        strides = torch.ones_like(levels)
        for index in range(1, len(config.skill_fsq_levels)):
            strides[index] = strides[index - 1] * config.skill_fsq_levels[index - 1]
        self.register_buffer("_fsq_levels", levels, persistent=False)
        self.register_buffer("_fsq_strides", strides, persistent=False)
        self.register_buffer("_fsq_half", (levels - 1).float() / 2.0, persistent=False)

        dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
        if self.model.skill_predictor is not None:
            self.model.skill_predictor.to(dtype=dtype)
            for parameter in self.model.skill_predictor.auxiliary_parameters():
                parameter.requires_grad_(True)
            if config.gradient_checkpointing:
                self.model.skill_predictor.gradient_checkpointing_enable()
        if self.model.fsq_term_train is not None:
            self.model.fsq_term_train.to(dtype=torch.float32)
        if self.model.fsq_image_term_train is not None:
            self.model.fsq_image_term_train.to(dtype=torch.float32)
        if self.model.fsq_wrist_term_train is not None:
            self.model.fsq_wrist_term_train.to(dtype=torch.float32)
        if self.model.fsq_state_term_train is not None:
            self.model.fsq_state_term_train.to(dtype=torch.float32)
        if self.model.fsq_state_rnn_term_train is not None:
            self.model.fsq_state_rnn_term_train.to(dtype=torch.float32)
        self.to(device=config.device)
        log.info(
            "Auxiliary-only policy: terminator=%s (%s/%s/%s/%s), image_only_terminator=%s, "
            "wrist_only_terminator=%s, state_only_terminator=%s, "
            "state_rnn_terminator=%s, skill_predictor=%s",
            config.train_terminator,
            config.terminator_context,
            config.terminator_cameras,
            config.terminator_arch,
            config.terminator_vision_backbone,
            config.train_image_only_terminator,
            config.train_wrist_only_terminator,
            config.train_state_only_terminator,
            config.train_state_rnn_terminator,
            config.train_skill_predictor,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        terminator = self.model.fsq_term_train
        if terminator is not None and terminator.freeze_vision_encoder:
            terminator.vision_encoder.eval()
        image_terminator = self.model.fsq_image_term_train
        if (
            image_terminator is not None
            and image_terminator.freeze_vision_encoder
        ):
            image_terminator.vision_encoder.eval()
        wrist_terminator = self.model.fsq_wrist_term_train
        if (
            wrist_terminator is not None
            and wrist_terminator.freeze_vision_encoder
        ):
            wrist_terminator.vision_encoder.eval()
        return self

    def reset(self) -> None:
        return None

    def _code_to_zq(self, skill_code: Tensor) -> Tensor:
        index = skill_code.reshape(-1, 1).long()
        level_ids = (
            torch.div(index, self._fsq_strides[None], rounding_mode="floor")
            % self._fsq_levels[None]
        )
        return (level_ids.float() - self._fsq_half[None]) / self._fsq_half[None]

    @staticmethod
    def _as_channels_first(image: Tensor) -> Tensor:
        image = image.float()
        if image.ndim == 4 and image.shape[1] != 3 and image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        return image

    def _true_skill_code(self, batch: dict) -> Tensor:
        if "skill_code_true" in batch:
            true_code = batch["skill_code_true"].view(-1).long()
        else:
            missing = [
                key for key in ("skill_sequence", "skill_index") if key not in batch
            ]
            if missing:
                raise ValueError(
                    "Terminator training needs skill_code_true or "
                    f"skill_sequence/skill_index; missing={missing}."
                )
            sequence = batch["skill_sequence"].long()
            index = batch["skill_index"].long().view(-1).clamp(
                0, sequence.shape[1] - 1
            )
            true_code = sequence.gather(1, index[:, None]).squeeze(1)
        return true_code.clamp(0, self.config.skill_vocab_size - 1)

    @staticmethod
    def _termination_targets(
        batch: dict, device: torch.device, sigma: float
    ) -> tuple[Tensor, Tensor]:
        distance_from_start = batch["skill_ds"].float().view(-1).to(device)
        distance_to_end = batch["skill_de"].float().view(-1).to(device)
        progress_target = (
            distance_from_start
            / (distance_from_start + distance_to_end).clamp_min(1.0)
        ).clamp(0.0, 1.0)
        termination_target = (
            torch.exp(-(distance_to_end.square()) / (2.0 * sigma**2))
            if sigma > 0
            else (distance_to_end == 0).float()
        )
        return progress_target, termination_target

    @staticmethod
    def _termination_loss_and_metrics(
        *,
        prefix: str,
        progress_prediction: Tensor,
        termination_logits: Tensor,
        progress_target: Tensor,
        termination_target: Tensor,
        positive_weight: float,
        termination_only: bool = False,
    ) -> tuple[Tensor, dict[str, float]]:
        # In termination-only mode the model returns a detached zero progress, so
        # a smooth-L1 term against it would carry no gradient while still moving
        # the reported loss. Drop the term and its panels instead of scoring a
        # head that is not being trained.
        progress_loss = (
            None
            if termination_only
            else F.smooth_l1_loss(
                progress_prediction, progress_target.to(progress_prediction.dtype)
            )
        )
        termination_loss = F.binary_cross_entropy_with_logits(
            termination_logits,
            termination_target.to(termination_logits.dtype),
            pos_weight=torch.tensor(
                positive_weight,
                device=termination_logits.device,
                dtype=termination_logits.dtype,
            ),
        )
        objective = (
            termination_loss if progress_loss is None else progress_loss + termination_loss
        )
        with torch.no_grad():
            predicted_end = termination_logits.sigmoid() >= 0.5
            target_end = termination_target >= 0.5
            true_positive = (predicted_end & target_end).sum().float()
            predicted_positive = predicted_end.sum().float()
            actual_positive = target_end.sum().float()
            precision = true_positive / predicted_positive.clamp_min(1.0)
            recall = true_positive / actual_positive.clamp_min(1.0)
            f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
            metrics = {
                f"{prefix}/loss": objective.detach().item(),
                f"{prefix}/termination_loss": termination_loss.detach().item(),
                f"{prefix}/termination_only": float(termination_only),
                f"{prefix}/end_accuracy": (
                    (predicted_end == target_end).float().mean().item()
                ),
                f"{prefix}/end_precision": precision.item(),
                f"{prefix}/end_recall": recall.item(),
                f"{prefix}/end_f1": f1.item(),
                f"{prefix}/positive_fraction": target_end.float().mean().item(),
            }
            if progress_loss is not None:
                metrics[f"{prefix}/progress_loss"] = progress_loss.detach().item()
                metrics[f"{prefix}/progress_mae"] = F.l1_loss(
                    progress_prediction, progress_target.to(progress_prediction.dtype)
                ).item()
        return objective, metrics

    def _state_sequence(self, batch: dict, *, label: str) -> Tensor:
        if OBS_STATE not in batch:
            raise ValueError(f"{label} training batch is missing {OBS_STATE!r}.")
        module = (
            self.model.fsq_state_rnn_term_train
            if label == "State-RNN terminator"
            else self.model.fsq_state_term_train
        )
        if module is None:
            raise RuntimeError(f"{label} training is disabled.")
        device = next(module.parameters()).device
        dtype = next(module.parameters()).dtype
        states = batch[OBS_STATE].to(device=device, dtype=dtype)
        if states.ndim == 2:
            states = states[:, None, :]
        if states.ndim != 3:
            raise ValueError(
                f"{label} expects normalized state [B, D] or [B, T, D], "
                f"got {tuple(states.shape)}."
            )
        if states.shape[-1] < self.config.max_state_dim:
            raise ValueError(
                f"{label} state width {states.shape[-1]} is smaller than "
                f"max_state_dim={self.config.max_state_dim}."
            )
        return states[..., : self.config.max_state_dim]

    def _state_sequence_targets(
        self,
        *,
        batch: dict,
        sequence_length: int,
        device: torch.device,
        sigma: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        current_ds = batch["skill_ds"].to(device=device).long().view(-1)
        current_de = batch["skill_de"].to(device=device).long().view(-1)
        unclamped_lengths = current_ds + 1
        if self.config.state_full_skill_supervision:
            if torch.any(current_de != 0):
                raise ValueError(
                    "Full-skill state supervision requires endpoint-anchored "
                    "samples with skill_de == 0."
                )
            if torch.any(unclamped_lengths > sequence_length):
                longest = int(unclamped_lengths.max().item())
                raise ValueError(
                    "state_rnn_terminator.sequence_length does not cover the "
                    f"longest sampled skill: {sequence_length} < {longest}."
                )
        lengths = unclamped_lengths.clamp(1, sequence_length)
        positions = torch.arange(sequence_length, device=device)[None, :]
        valid = positions < lengths[:, None]
        distance_from_current = (lengths[:, None] - 1 - positions).clamp_min(0)
        distance_from_start = (current_ds[:, None] - distance_from_current).clamp_min(0)
        distance_to_end = current_de[:, None] + distance_from_current
        progress_target = (
            distance_from_start.float()
            / (distance_from_start + distance_to_end).float().clamp_min(1.0)
        ).clamp(0.0, 1.0)
        termination_target = (
            torch.exp(-(distance_to_end.float().square()) / (2.0 * sigma**2))
            if sigma > 0
            else (distance_to_end == 0).float()
        )
        return lengths, valid, progress_target, termination_target

    @staticmethod
    def _state_sequence_loss_and_metrics(
        *,
        prefix: str,
        progress_prediction: Tensor,
        termination_logits: Tensor,
        progress_target: Tensor,
        termination_target: Tensor,
        valid: Tensor,
        positive_weight: float,
        balance_positive_negative: bool,
        termination_only: bool,
    ) -> tuple[Tensor, dict[str, float]]:
        if progress_prediction.shape != valid.shape or termination_logits.shape != valid.shape:
            raise ValueError(
                f"{prefix} predictions must match sequence mask {tuple(valid.shape)}, "
                f"got progress={tuple(progress_prediction.shape)}, "
                f"termination={tuple(termination_logits.shape)}."
            )
        valid_count = valid.sum().clamp_min(1)
        progress_errors = F.smooth_l1_loss(
            progress_prediction,
            progress_target.to(progress_prediction.dtype),
            reduction="none",
        )
        progress_loss = (
            None
            if termination_only
            else (progress_errors * valid).sum() / valid_count
        )
        element_loss = F.binary_cross_entropy_with_logits(
            termination_logits,
            termination_target.to(termination_logits.dtype),
            pos_weight=torch.tensor(
                positive_weight,
                device=termination_logits.device,
                dtype=termination_logits.dtype,
            ),
            reduction="none",
        )
        target_end = termination_target >= 0.5
        positive_mask = valid & target_end
        if balance_positive_negative:
            negative_mask = valid & ~target_end
            positive_count = positive_mask.sum(dim=1)
            negative_count = negative_mask.sum(dim=1)
            positive_loss = (
                (element_loss * positive_mask).sum(dim=1)
                / positive_count.clamp_min(1)
            )
            negative_loss = (
                (element_loss * negative_mask).sum(dim=1)
                / negative_count.clamp_min(1)
            )
            has_positive = positive_count > 0
            has_negative = negative_count > 0
            group_count = has_positive.long() + has_negative.long()
            termination_loss = (
                positive_loss * has_positive + negative_loss * has_negative
            ) / group_count.clamp_min(1)
            termination_loss = termination_loss.mean()
        else:
            termination_loss = (element_loss * valid).sum() / valid_count
        objective = (
            termination_loss if progress_loss is None else termination_loss + progress_loss
        )
        with torch.no_grad():
            probabilities = termination_logits.sigmoid()
            predicted_end = probabilities >= 0.5
            true_positive = (predicted_end & target_end & valid).sum().float()
            predicted_positive = (predicted_end & valid).sum().float()
            actual_positive = positive_mask.sum().float()
            precision = true_positive / predicted_positive.clamp_min(1.0)
            recall = true_positive / actual_positive.clamp_min(1.0)
            f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
            metrics = {
                f"{prefix}/loss": objective.detach().item(),
                f"{prefix}/termination_loss": termination_loss.detach().item(),
                f"{prefix}/termination_only": float(termination_only),
                f"{prefix}/balanced_positive_negative": float(
                    balance_positive_negative
                ),
                f"{prefix}/all_step_supervision": 1.0,
                f"{prefix}/end_accuracy": (
                    ((predicted_end == target_end) & valid).sum().float() / valid_count
                ).item(),
                f"{prefix}/end_precision": precision.item(),
                f"{prefix}/end_recall": recall.item(),
                f"{prefix}/end_f1": f1.item(),
                f"{prefix}/positive_fraction": (actual_positive / valid_count).item(),
                f"{prefix}/predicted_positive_fraction": (
                    predicted_positive / valid_count
                ).item(),
                f"{prefix}/termination_probability_mean": (
                    (probabilities * valid).sum() / valid_count
                ).item(),
            }
            if progress_loss is not None:
                metrics[f"{prefix}/progress_loss"] = progress_loss.detach().item()
                metrics[f"{prefix}/progress_mae"] = (
                    (progress_prediction - progress_target).abs() * valid
                ).sum().div(valid_count).item()
        return objective, metrics

    def _terminator_objective(self, batch: dict) -> tuple[Tensor, dict[str, float]]:
        required = ["skill_ds", "skill_de"]
        terminator = self.model.fsq_term_train
        if terminator is None:
            raise RuntimeError("Terminator training is disabled.")
        context_mode = str(getattr(terminator, "context_mode", "proprio"))
        camera_mode = str(getattr(terminator, "camera_mode", "both"))
        if camera_mode in {"both", "top"}:
            required.append("observation.images.image")
        if camera_mode in {"both", "wrist"}:
            required.append("observation.images.wrist_image")
        if context_mode == "prev_action":
            required.extend((SKILL_PREVIOUS_ACTION, SKILL_PREVIOUS_ACTION_BOS))
        elif context_mode == "proprio":
            required.append("skill_decoder_state")
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"Terminator training batch is missing {missing}.")
        device = next(terminator.parameters()).device
        dtype = next(terminator.parameters()).dtype
        true_code = self._true_skill_code(batch)
        if context_mode == "prev_action":
            action = batch[SKILL_PREVIOUS_ACTION].to(device=device, dtype=dtype)
            if action.ndim == 3 and action.shape[1] == 1:
                action = action[:, 0]
            if action.ndim != 2 or action.shape[-1] < int(terminator.state_dim):
                raise ValueError(
                    "Previous terminator action must be (B,A) with "
                    f"A>={terminator.state_dim}, got {tuple(action.shape)}."
                )
            context = terminator.normalize_previous_action(
                action[..., : int(terminator.state_dim)]
            )
            bos = batch[SKILL_PREVIOUS_ACTION_BOS].to(
                device=device, dtype=torch.bool
            ).view(-1)
            context = context.clone()
            context[bos] = 0.0
        elif context_mode == "proprio":
            context = batch["skill_decoder_state"].to(device=device, dtype=dtype)[
                ..., : int(terminator.state_dim)
            ]
            if context.ndim == 3:
                context = context[:, -1]
        else:
            context = None
        z_q = self._code_to_zq(true_code.to(self._fsq_strides.device)).to(
            device=device, dtype=dtype
        )
        third = batch.get("observation.images.image")
        wrist = batch.get("observation.images.wrist_image")
        progress_prediction, termination_logits = terminator(
            z_q,
            context,
            None
            if third is None
            else self._as_channels_first(third).to(device=device, dtype=dtype),
            None
            if wrist is None
            else self._as_channels_first(wrist).to(device=device, dtype=dtype),
        )
        progress_target, termination_target = self._termination_targets(
            batch, device, self.config.terminator_end_target_sigma
        )
        return self._termination_loss_and_metrics(
            prefix="terminator",
            progress_prediction=progress_prediction,
            termination_logits=termination_logits,
            progress_target=progress_target,
            termination_target=termination_target,
            positive_weight=self.config.terminator_end_pos_weight,
            termination_only=self.config.terminator_termination_only,
        )

    def _image_only_terminator_objective(
        self, batch: dict
    ) -> tuple[Tensor, dict[str, float]]:
        required = (
            "skill_ds",
            "skill_de",
            "observation.images.image",
            "observation.images.wrist_image",
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"Image-only terminator training batch is missing {missing}.")
        terminator = self.model.fsq_image_term_train
        if terminator is None:
            raise RuntimeError("Image-only terminator training is disabled.")
        device = next(terminator.parameters()).device
        dtype = next(terminator.parameters()).dtype
        z_q = self._code_to_zq(
            self._true_skill_code(batch).to(self._fsq_strides.device)
        ).to(device=device, dtype=dtype)
        progress_prediction, termination_logits = terminator(
            z_q,
            self._as_channels_first(batch["observation.images.image"]).to(
                device=device, dtype=dtype
            ),
            self._as_channels_first(batch["observation.images.wrist_image"]).to(
                device=device, dtype=dtype
            ),
        )
        progress_target, termination_target = self._termination_targets(
            batch, device, self.config.image_only_terminator_end_target_sigma
        )
        return self._termination_loss_and_metrics(
            prefix="image_terminator",
            progress_prediction=progress_prediction,
            termination_logits=termination_logits,
            progress_target=progress_target,
            termination_target=termination_target,
            positive_weight=self.config.image_only_terminator_end_pos_weight,
            termination_only=self.config.image_only_terminator_termination_only,
        )

    def _wrist_only_terminator_objective(
        self, batch: dict
    ) -> tuple[Tensor, dict[str, float]]:
        required = (
            "skill_ds",
            "skill_de",
            "observation.images.wrist_image",
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"Wrist-only terminator training batch is missing {missing}.")
        terminator = self.model.fsq_wrist_term_train
        if terminator is None:
            raise RuntimeError("Wrist-only terminator training is disabled.")
        device = next(terminator.parameters()).device
        dtype = next(terminator.parameters()).dtype
        z_q = self._code_to_zq(
            self._true_skill_code(batch).to(self._fsq_strides.device)
        ).to(device=device, dtype=dtype)
        progress_prediction, termination_logits = terminator(
            z_q,
            self._as_channels_first(batch["observation.images.wrist_image"]).to(
                device=device, dtype=dtype
            ),
        )
        progress_target, termination_target = self._termination_targets(
            batch, device, self.config.wrist_only_terminator_end_target_sigma
        )
        return self._termination_loss_and_metrics(
            prefix="wrist_terminator",
            progress_prediction=progress_prediction,
            termination_logits=termination_logits,
            progress_target=progress_target,
            termination_target=termination_target,
            positive_weight=self.config.wrist_only_terminator_end_pos_weight,
            termination_only=self.config.wrist_only_terminator_termination_only,
        )

    def _state_only_terminator_objective(
        self, batch: dict
    ) -> tuple[Tensor, dict[str, float]]:
        required = ("skill_ds", "skill_de")
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"State-only terminator training batch is missing {missing}.")
        terminator = self.model.fsq_state_term_train
        if terminator is None:
            raise RuntimeError("State-only terminator training is disabled.")
        states = self._state_sequence(batch, label="State-only terminator")
        z_q = self._code_to_zq(
            self._true_skill_code(batch).to(self._fsq_strides.device)
        ).to(device=states.device, dtype=states.dtype)
        lengths, valid, progress_target, termination_target = (
            self._state_sequence_targets(
                batch=batch,
                sequence_length=states.shape[1],
                device=states.device,
                sigma=self.config.state_only_terminator_end_target_sigma,
            )
        )
        compact_states = StateSkillRNNTerminator.compact_valid_suffix(states, lengths)
        batch_size, sequence_length, state_dim = compact_states.shape
        sequence_z_q = z_q[:, None, :].expand(-1, sequence_length, -1)
        progress, logits = terminator.forward_outputs(
            sequence_z_q.reshape(batch_size * sequence_length, -1),
            compact_states.reshape(batch_size * sequence_length, state_dim),
        )
        progress = progress.view(batch_size, sequence_length)
        logits = logits.view(batch_size, sequence_length)
        return self._state_sequence_loss_and_metrics(
            prefix="state_terminator",
            progress_prediction=progress,
            termination_logits=logits,
            progress_target=progress_target,
            termination_target=termination_target,
            valid=valid,
            positive_weight=self.config.state_only_terminator_end_pos_weight,
            balance_positive_negative=(
                self.config.state_only_terminator_balance_positive_negative
            ),
            termination_only=self.config.state_only_terminator_termination_only,
        )

    def _state_rnn_terminator_objective(
        self, batch: dict
    ) -> tuple[Tensor, dict[str, float]]:
        required = ("skill_ds", "skill_de")
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"State-RNN terminator training batch is missing {missing}.")
        terminator = self.model.fsq_state_rnn_term_train
        if terminator is None:
            raise RuntimeError("State-RNN terminator training is disabled.")
        states = self._state_sequence(batch, label="State-RNN terminator")
        if states.shape[1] != self.config.state_rnn_terminator_sequence_length:
            raise ValueError(
                "State-RNN dataset window does not match sequence_length: "
                f"batch={states.shape[1]}, "
                f"config={self.config.state_rnn_terminator_sequence_length}."
            )
        z_q = self._code_to_zq(
            self._true_skill_code(batch).to(self._fsq_strides.device)
        ).to(device=states.device, dtype=states.dtype)
        valid_lengths, valid, progress_target, termination_target = (
            self._state_sequence_targets(
                batch=batch,
                sequence_length=states.shape[1],
                device=states.device,
                sigma=self.config.state_rnn_terminator_end_target_sigma,
            )
        )
        progress, logits, _ = terminator.forward_all_outputs(
            z_q,
            states,
            lengths=valid_lengths,
        )
        objective, metrics = self._state_sequence_loss_and_metrics(
            prefix="state_rnn_terminator",
            progress_prediction=progress,
            termination_logits=logits,
            progress_target=progress_target,
            termination_target=termination_target,
            valid=valid,
            positive_weight=self.config.state_rnn_terminator_end_pos_weight,
            balance_positive_negative=(
                self.config.state_rnn_terminator_balance_positive_negative
            ),
            termination_only=self.config.state_rnn_terminator_termination_only,
        )
        metrics["state_rnn_terminator/mean_valid_length"] = (
            valid_lengths.float().mean().item()
        )
        return objective, metrics

    @torch.no_grad()
    def state_only_terminator_predict(
        self,
        skill_code: Tensor,
        normalized_state: Tensor,
    ) -> Tensor:
        """Return state-only boundary probabilities for normalized proprio."""
        _, probability = self.state_only_terminator_predict_outputs(
            skill_code,
            normalized_state,
        )
        return probability

    @torch.no_grad()
    def state_only_terminator_predict_outputs(
        self,
        skill_code: Tensor,
        normalized_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(progress, boundary_probability)`` for normalized proprio."""
        terminator = self.model.fsq_state_term_train
        if terminator is None:
            raise RuntimeError("State-only terminator is not attached.")
        device = next(terminator.parameters()).device
        dtype = next(terminator.parameters()).dtype
        z_q = self._code_to_zq(skill_code.to(self._fsq_strides.device)).to(
            device=device,
            dtype=dtype,
        )
        state = normalized_state.to(device=device, dtype=dtype)[
            ..., : self.config.max_state_dim
        ]
        progress, logits = terminator.forward_outputs(z_q, state)
        return progress, logits.sigmoid()

    @torch.no_grad()
    def state_rnn_terminator_predict(
        self,
        skill_code: Tensor,
        normalized_state: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Online recurrent boundary prediction with explicit hidden I/O."""
        _, probability, next_hidden = self.state_rnn_terminator_predict_outputs(
            skill_code,
            normalized_state,
            hidden,
        )
        return probability, next_hidden

    @torch.no_grad()
    def state_rnn_terminator_predict_outputs(
        self,
        skill_code: Tensor,
        normalized_state: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return online ``(progress, boundary_probability, next_hidden)``."""
        terminator = self.model.fsq_state_rnn_term_train
        if terminator is None:
            raise RuntimeError("State-RNN terminator is not attached.")
        device = next(terminator.parameters()).device
        dtype = next(terminator.parameters()).dtype
        z_q = self._code_to_zq(skill_code.to(self._fsq_strides.device)).to(
            device=device,
            dtype=dtype,
        )
        state = normalized_state.to(device=device, dtype=dtype)[
            ..., : self.config.max_state_dim
        ]
        if hidden is not None:
            hidden = hidden.to(device=device, dtype=dtype)
        progress, logits, next_hidden = terminator.step_outputs(z_q, state, hidden)
        return progress, logits.sigmoid(), next_hidden

    def _skill_predictor_objective(self, batch: dict) -> tuple[Tensor, dict[str, float]]:
        required = (
            "skill_start_image",
            "skill_start_wrist_image",
            "skill_code",
            OBS_LANGUAGE_TOKENS,
            OBS_LANGUAGE_ATTENTION_MASK,
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"Skill predictor training batch is missing {missing}.")
        predictor = self.model.skill_predictor
        if predictor is None:
            raise RuntimeError("Skill predictor training is disabled.")
        device = next(predictor.parameters()).device
        target = batch["skill_code"].to(device).view(-1).long().clamp(
            0, self.config.skill_vocab_size - 1
        )
        raw_loss, accuracy = predictor.loss(
            [
                self._as_channels_first(batch["skill_start_image"]).to(device),
                self._as_channels_first(batch["skill_start_wrist_image"]).to(device),
            ],
            batch[OBS_LANGUAGE_TOKENS].to(device),
            batch[OBS_LANGUAGE_ATTENTION_MASK].to(device),
            target,
        )
        return raw_loss, {
            "skill_predictor/loss": raw_loss.detach().item(),
            "skill_predictor/skill_accuracy": float(accuracy),
            "skill_predictor/all_layers": float(self.config.skill_predictor_all_layers),
            "skill_predictor/lora_layers": float(predictor.lora_layer_count),
            "skill_predictor/deadzone_frac": self.config.skill_predictor_deadzone_frac,
        }

    def forward(self, batch: dict, reduction: str = "mean"):
        if reduction != "mean":
            raise ValueError("skill_aux does not support per-sample/RABC reduction.")
        objectives: list[Tensor] = []
        metrics: dict[str, float] = {}
        if self.config.train_terminator:
            objective, output = self._terminator_objective(batch)
            objectives.append(objective)
            metrics.update(output)
        if self.config.train_image_only_terminator:
            objective, output = self._image_only_terminator_objective(batch)
            objectives.append(objective)
            metrics.update(output)
        if self.config.train_wrist_only_terminator:
            objective, output = self._wrist_only_terminator_objective(batch)
            objectives.append(objective)
            metrics.update(output)
        if self.config.train_state_only_terminator:
            objective, output = self._state_only_terminator_objective(batch)
            objectives.append(objective)
            metrics.update(output)
        if self.config.train_state_rnn_terminator:
            objective, output = self._state_rnn_terminator_objective(batch)
            objectives.append(objective)
            metrics.update(output)
        if self.config.train_skill_predictor:
            objective, output = self._skill_predictor_objective(batch)
            objectives.append(objective)
            metrics.update(output)
        if not objectives:
            raise RuntimeError("No auxiliary objective is enabled.")
        return torch.stack(objectives).sum(), metrics

    def get_optim_params(self) -> list[dict]:
        groups: list[dict] = []
        terminator = self.model.fsq_term_train
        if terminator is not None:
            params = [parameter for parameter in terminator.parameters() if parameter.requires_grad]
            if params:
                groups.append(
                    {
                        "params": params,
                        "lr": self.config.optimizer_lr * self.config.terminator_lr_scale,
                        "lr_scale": self.config.terminator_lr_scale,
                        "group_name": "terminator",
                    }
                )
        image_terminator = self.model.fsq_image_term_train
        if image_terminator is not None:
            params = [
                parameter
                for parameter in image_terminator.parameters()
                if parameter.requires_grad
            ]
            if params:
                groups.append(
                    {
                        "params": params,
                        "lr": self.config.optimizer_lr
                        * self.config.image_only_terminator_lr_scale,
                        "lr_scale": self.config.image_only_terminator_lr_scale,
                        "group_name": "image_terminator",
                    }
                )
        wrist_terminator = self.model.fsq_wrist_term_train
        if wrist_terminator is not None:
            params = [
                parameter
                for parameter in wrist_terminator.parameters()
                if parameter.requires_grad
            ]
            if params:
                groups.append(
                    {
                        "params": params,
                        "lr": self.config.optimizer_lr
                        * self.config.wrist_only_terminator_lr_scale,
                        "lr_scale": self.config.wrist_only_terminator_lr_scale,
                        "group_name": "wrist_terminator",
                    }
                )
        state_terminator = self.model.fsq_state_term_train
        if state_terminator is not None:
            params = [
                parameter
                for parameter in state_terminator.parameters()
                if parameter.requires_grad
            ]
            if params:
                groups.append(
                    {
                        "params": params,
                        "lr": self.config.optimizer_lr
                        * self.config.state_only_terminator_lr_scale,
                        "lr_scale": self.config.state_only_terminator_lr_scale,
                        "group_name": "state_terminator",
                    }
                )
        state_rnn_terminator = self.model.fsq_state_rnn_term_train
        if state_rnn_terminator is not None:
            params = [
                parameter
                for parameter in state_rnn_terminator.parameters()
                if parameter.requires_grad
            ]
            if params:
                groups.append(
                    {
                        "params": params,
                        "lr": self.config.optimizer_lr
                        * self.config.state_rnn_terminator_lr_scale,
                        "lr_scale": self.config.state_rnn_terminator_lr_scale,
                        "group_name": "state_rnn_terminator",
                    }
                )
        predictor = self.model.skill_predictor
        if predictor is not None:
            reader_head = [
                parameter
                for parameter in predictor.reader_head_parameters()
                if parameter.requires_grad
            ]
            if reader_head:
                groups.append(
                    {
                        "params": reader_head,
                        "lr": self.config.optimizer_lr * self.config.skill_predictor_lr_scale,
                        "lr_scale": self.config.skill_predictor_lr_scale,
                        "group_name": "skill_predictor_reader_head",
                    }
                )
            lora = [
                parameter for parameter in predictor.lora_parameters() if parameter.requires_grad
            ]
            if lora:
                groups.append(
                    {
                        "params": lora,
                        "lr": self.config.optimizer_lr
                        * self.config.skill_predictor_lora_lr_scale,
                        "lr_scale": self.config.skill_predictor_lora_lr_scale,
                        "group_name": "skill_predictor_lora",
                    }
                )
        if not groups:
            raise RuntimeError("The auxiliary optimizer has no trainable parameters.")
        return groups

    def isolated_main_optimizer_grad_groups(self) -> dict[str, list[nn.Parameter]]:
        groups: dict[str, list[nn.Parameter]] = {}
        terminator = self.model.fsq_term_train
        if terminator is not None:
            params = [parameter for parameter in terminator.parameters() if parameter.requires_grad]
            if params:
                groups["terminator"] = params
        image_terminator = self.model.fsq_image_term_train
        if image_terminator is not None:
            params = [
                parameter
                for parameter in image_terminator.parameters()
                if parameter.requires_grad
            ]
            if params:
                groups["image_terminator"] = params
        wrist_terminator = self.model.fsq_wrist_term_train
        if wrist_terminator is not None:
            params = [
                parameter
                for parameter in wrist_terminator.parameters()
                if parameter.requires_grad
            ]
            if params:
                groups["wrist_terminator"] = params
        state_terminator = self.model.fsq_state_term_train
        if state_terminator is not None:
            params = [
                parameter
                for parameter in state_terminator.parameters()
                if parameter.requires_grad
            ]
            if params:
                groups["state_terminator"] = params
        state_rnn_terminator = self.model.fsq_state_rnn_term_train
        if state_rnn_terminator is not None:
            params = [
                parameter
                for parameter in state_rnn_terminator.parameters()
                if parameter.requires_grad
            ]
            if params:
                groups["state_rnn_terminator"] = params
        predictor = self.model.skill_predictor
        if predictor is not None:
            params = [
                parameter
                for parameter in predictor.auxiliary_parameters()
                if parameter.requires_grad
            ]
            if params:
                groups["skill_predictor"] = params
        return groups

    def optimizer_metrics(self, optimizer) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for group in optimizer.param_groups:
            name = group.get("group_name")
            if name == "terminator":
                metrics["terminator/lr"] = float(group["lr"])
            elif name == "image_terminator":
                metrics["image_terminator/lr"] = float(group["lr"])
            elif name == "wrist_terminator":
                metrics["wrist_terminator/lr"] = float(group["lr"])
            elif name == "state_terminator":
                metrics["state_terminator/lr"] = float(group["lr"])
            elif name == "state_rnn_terminator":
                metrics["state_rnn_terminator/lr"] = float(group["lr"])
            elif name == "skill_predictor_reader_head":
                metrics["skill_predictor/reader_head_lr"] = float(group["lr"])
            elif name == "skill_predictor_lora":
                metrics["skill_predictor/lora_lr"] = float(group["lr"])
        return metrics

    def parameter_counts(self) -> dict[str, int]:
        def count(module: nn.Module | None, trainable: bool = False) -> int:
            if module is None:
                return 0
            return sum(
                parameter.numel()
                for parameter in module.parameters()
                if not trainable or parameter.requires_grad
            )

        return {
            "total": count(self),
            "trainable": count(self, trainable=True),
            "terminator": count(self.model.fsq_term_train),
            "image_terminator": count(self.model.fsq_image_term_train),
            "wrist_terminator": count(self.model.fsq_wrist_term_train),
            "state_terminator": count(self.model.fsq_state_term_train),
            "state_rnn_terminator": count(self.model.fsq_state_rnn_term_train),
            "skill_predictor": count(self.model.skill_predictor),
        }

    def _load_terminator_warm_start(self, checkpoint_path: str | Path) -> None:
        terminator = self.model.fsq_term_train
        if terminator is None:
            return
        path = Path(checkpoint_path)
        config_path = path / "config.json"
        weights_path = path / "model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                f"Incomplete auxiliary warm-start checkpoint: {path}"
            )
        source = json.loads(config_path.read_text())
        if source.get("type") != "skill_aux" or not source.get(
            "train_terminator", False
        ):
            raise ValueError(
                "Terminator warm-start requires a skill_aux checkpoint with "
                "train_terminator=true."
            )
        expected_contract = {
            "skill_fsq_levels": self.config.skill_fsq_levels,
            "skill_vocab_size": self.config.skill_vocab_size,
            "terminator_context": self.config.terminator_context,
            "terminator_cameras": self.config.terminator_cameras,
            "terminator_arch": self.config.terminator_arch,
            "terminator_vision_backbone": self.config.terminator_vision_backbone,
            "terminator_termination_only": self.config.terminator_termination_only,
        }
        mismatches = []
        for field, value in expected_contract.items():
            checkpoint_value = source.get(
                field, "both" if field == "terminator_cameras" else None
            )
            if checkpoint_value != value:
                mismatches.append(
                    f"{field}: checkpoint={checkpoint_value!r}, current={value!r}"
                )
        source_space = str(source.get("skill_code_space_id", "") or "").strip()
        current_space = str(self.config.skill_code_space_id or "").strip()
        if source_space and current_space and source_space != current_space:
            mismatches.append(
                "skill_code_space_id: "
                f"checkpoint={source_space!r}, current={current_space!r}"
            )
        if mismatches:
            raise ValueError(
                "Terminator warm-start contract mismatch: " + "; ".join(mismatches)
            )

        prefix = "model.fsq_term_train."
        target_state = terminator.state_dict()
        with safe_open(str(weights_path), framework="pt", device="cpu") as checkpoint:
            source_keys = {
                key.removeprefix(prefix)
                for key in checkpoint.keys()
                if key.startswith(prefix)
            }
            missing = set(target_state) - source_keys
            unexpected = source_keys - set(target_state)
            if missing or unexpected:
                raise RuntimeError(
                    "Terminator warm-start tensor mismatch: "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )
            with torch.no_grad():
                for key, target in target_state.items():
                    value = checkpoint.get_tensor(prefix + key)
                    if value.shape != target.shape:
                        raise RuntimeError(
                            f"Terminator warm-start shape mismatch for {key}: "
                            f"checkpoint={tuple(value.shape)}, current={tuple(target.shape)}"
                        )
                    target.copy_(value.to(device=target.device, dtype=target.dtype))
        log.info(
            "Loaded %d terminator tensors from %s.", len(target_state), path
        )

    def _load_pi05_predictor_base(self, pretrained_path: str | Path) -> None:
        predictor = self.model.skill_predictor
        if predictor is None:
            return
        path = Path(pretrained_path)
        weights_path = path if path.is_file() else path / "model.safetensors"
        if not weights_path.is_file():
            raise FileNotFoundError(f"pi0.5 predictor base not found: {weights_path}")
        raw_prefix = "paligemma_with_expert.paligemma.model."
        target_prefix = "model.skill_predictor.vlm."
        expected = {
            target_prefix + key
            for key in predictor.vlm.state_dict()
            if ".adapters." not in key
        }
        mapped: dict[str, Tensor] = {}
        with safe_open(str(weights_path), framework="pt", device="cpu") as checkpoint:
            keys = set(checkpoint.keys())
            for key in keys:
                if key.startswith(raw_prefix):
                    target = target_prefix + key.removeprefix(raw_prefix)
                    if target in expected or target.replace(".weight", ".base.weight") in expected:
                        mapped[target] = checkpoint.get_tensor(key)
            embedding = target_prefix + "language_model.embed_tokens.weight"
            lm_head = "paligemma_with_expert.paligemma.lm_head.weight"
            if embedding not in mapped and lm_head in keys:
                mapped[embedding] = checkpoint.get_tensor(lm_head)
        mapped, routed = route_plain_to_base(mapped, set(self.state_dict()))
        missing = expected - set(mapped)
        if missing:
            raise RuntimeError(
                "The pi0.5 checkpoint is incomplete for the predictor VLM; "
                f"missing={sorted(missing)[:20]}."
            )
        dtype = torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float32
        unexpected = self.load_state_dict(
            {key: value.to(dtype) for key, value in mapped.items()}, strict=False
        ).unexpected_keys
        if unexpected:
            raise RuntimeError(f"Unexpected pi0.5 predictor tensors: {sorted(unexpected)}")
        log.info("Loaded pi0.5 predictor base (%d tensors, %d LoRA routes).", len(mapped), routed)

    def _load_predictor_warm_start(self, checkpoint_path: str | Path) -> None:
        predictor = self.model.skill_predictor
        if predictor is None:
            return
        path = Path(checkpoint_path)
        config_path = path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Predictor warm-start config not found: {config_path}")
        source = json.loads(config_path.read_text())
        if source.get("type") not in {"skill_expert", "skill_aux"}:
            raise ValueError(f"Unsupported predictor checkpoint type {source.get('type')!r}.")
        if not source.get("train_skill_predictor", False):
            raise ValueError("Predictor warm-start checkpoint has no trained predictor.")
        mismatches = [
            f"{field}: checkpoint={source.get(field)!r}, current={getattr(self.config, field)!r}"
            for field in _PREDICTOR_CHECKPOINT_CONTRACT_FIELDS
            if source.get(field) != getattr(self.config, field)
        ]
        if mismatches:
            raise ValueError("Predictor warm-start contract mismatch: " + "; ".join(mismatches))
        loaded = _load_learned_predictor_parameters(predictor, path)
        log.info("Loaded %d learned predictor tensors from %s.", loaded, path)

    def _load_complete_predictor_warm_start(
        self, checkpoint_path: str | Path
    ) -> None:
        predictor = self.model.skill_predictor
        if predictor is None:
            return
        path = Path(checkpoint_path)
        config_path = path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Predictor warm-start config not found: {config_path}"
            )
        source = json.loads(config_path.read_text())
        if source.get("type") != "skill_aux" or not source.get(
            "train_skill_predictor", False
        ):
            raise ValueError(
                "Complete predictor warm-start requires a skill_aux checkpoint "
                "with train_skill_predictor=true."
            )
        mismatches = [
            f"{field}: checkpoint={source.get(field)!r}, current={getattr(self.config, field)!r}"
            for field in _PREDICTOR_CHECKPOINT_CONTRACT_FIELDS
            if source.get(field) != getattr(self.config, field)
        ]
        source_space = str(source.get("skill_code_space_id", "") or "").strip()
        current_space = str(self.config.skill_code_space_id or "").strip()
        if source_space and current_space and source_space != current_space:
            mismatches.append(
                "skill_code_space_id: "
                f"checkpoint={source_space!r}, current={current_space!r}"
            )
        if mismatches:
            raise ValueError(
                "Predictor warm-start contract mismatch: " + "; ".join(mismatches)
            )
        loaded = _load_complete_predictor_parameters(predictor, path)
        log.info("Loaded %d complete predictor tensors from %s.", loaded, path)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path,
        *,
        config=None,
        strict: bool = False,
        **kwargs,
    ):
        config_path = Path(pretrained_name_or_path) / "config.json"
        source_type = None
        if config_path.is_file():
            source_type = json.loads(config_path.read_text()).get("type")
        if source_type == "skill_aux":
            kwargs["_skip_auxiliary_warm_start"] = True
            return super().from_pretrained(
                pretrained_name_or_path,
                config=config,
                strict=strict,
                **kwargs,
            )
        if config is None:
            raise ValueError("A SkillAuxConfig is required when initializing from pi0.5.")
        policy = cls(config, **kwargs)
        if config.train_skill_predictor:
            policy._load_pi05_predictor_base(pretrained_name_or_path)
            predictor_checkpoint = str(
                config.skill_predictor_checkpoint_path
                or config.auxiliary_checkpoint_path
                or ""
            ).strip()
            if predictor_checkpoint:
                policy._load_predictor_warm_start(predictor_checkpoint)
        return policy

    def predict_action_chunk(self, batch: dict, **kwargs) -> Tensor:
        del batch, kwargs
        raise RuntimeError("skill_aux is training-only and has no action policy.")

    def select_action(self, batch: dict, **kwargs) -> Tensor:
        del batch, kwargs
        raise RuntimeError("skill_aux is training-only and has no action policy.")
