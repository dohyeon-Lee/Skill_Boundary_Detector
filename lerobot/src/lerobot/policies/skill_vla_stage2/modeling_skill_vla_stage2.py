"""BayesVLA-style Stage-2 likelihood refinement for the frozen Stage-1 VSA prior."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import (
    OPENPI_ATTENTION_MASK_VALUE,
    make_att_2d_masks,
    pad_vector,
)
from lerobot.policies.pi_gemma import (
    GemmaAttention,
    GemmaMLP,
    PiGemmaRMSNorm,
    _gated_residual,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.skill_expert.modeling_skill_expert import (
    SkillExpertPolicy,
    SkillExpertPytorch,
    _load_pretrained_state_dict,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from .configuration_skill_vla_stage2 import SkillVLAStage2Config

log = logging.getLogger(__name__)


class GemmaCrossAttention(nn.Module):
    """Cross-attention with gemma_300m's 8-query/1-KV-head geometry."""

    def __init__(self, config):
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        width = int(config.hidden_size)
        use_bias = bool(config.attention_bias)
        self.q_proj = nn.Linear(width, self.num_heads * self.head_dim, bias=use_bias)
        self.k_proj = nn.Linear(width, self.num_kv_heads * self.head_dim, bias=use_bias)
        self.v_proj = nn.Linear(width, self.num_kv_heads * self.head_dim, bias=use_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, width, bias=use_bias)

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        key_padding_mask: Tensor,
    ) -> Tensor:
        batch, query_tokens = query.shape[:2]
        memory_tokens = memory.shape[1]
        q = self.q_proj(query).view(
            batch, query_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(memory).view(
            batch, memory_tokens, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(memory).view(
            batch, memory_tokens, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        if self.num_kv_groups != 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        scores = scores.masked_fill(
            key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min
        )
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        output = torch.matmul(weights, v).transpose(1, 2).contiguous()
        return self.o_proj(output.view(batch, query_tokens, -1))


def _identity_adarms(norm: PiGemmaRMSNorm) -> PiGemmaRMSNorm:
    """Make a fresh gated residual branch exactly zero at initialization."""
    if norm.dense is None:
        raise ValueError("Stage-2 likelihood norms must be adaptive.")
    nn.init.zeros_(norm.dense.weight)
    nn.init.zeros_(norm.dense.bias)
    return norm


class LikelihoodBlock(nn.Module):
    """Action self-attention -> frozen-VLM cross-attention -> Gemma FFN."""

    def __init__(self, config, layer_index: int):
        super().__init__()
        width = int(config.hidden_size)
        eps = float(config.rms_norm_eps)
        self.self_norm = _identity_adarms(PiGemmaRMSNorm(width, eps=eps, cond_dim=width))
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_index)
        self.cross_norm = _identity_adarms(PiGemmaRMSNorm(width, eps=eps, cond_dim=width))
        self.cross_attn = GemmaCrossAttention(config)
        self.ffn_norm = _identity_adarms(PiGemmaRMSNorm(width, eps=eps, cond_dim=width))
        self.mlp = GemmaMLP(config)

    def forward(
        self,
        hidden: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Tensor,
        expert_condition: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> Tensor:
        residual = hidden
        normalized, gate = self.self_norm(hidden, cond=expert_condition)
        attended, _ = self.self_attn(
            normalized,
            attention_mask=None,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        hidden = _gated_residual(residual, attended, gate)

        residual = hidden
        normalized, gate = self.cross_norm(hidden, cond=expert_condition)
        attended = self.cross_attn(normalized, memory, memory_key_padding_mask)
        hidden = _gated_residual(residual, attended, gate)

        residual = hidden
        normalized, gate = self.ffn_norm(hidden, cond=expert_condition)
        transformed = self.mlp(normalized)
        return _gated_residual(residual, transformed, gate)


class SkillVLAStage2Pytorch(SkillExpertPytorch):
    """Stage-1 model plus four action-only likelihood blocks."""

    def __init__(self, config: SkillVLAStage2Config):
        super().__init__(config)
        if self.skill_predictor is None:
            raise RuntimeError("Stage 2 requires the Stage-1 frozen VLM/predictor.")
        expert_config = self.gemma_expert.model.config
        vlm_width = int(self.skill_predictor.vlm.language_model.config.hidden_size)
        self.vlm_to_expert_projection = nn.Linear(vlm_width, self.width)
        first_index = int(expert_config.num_hidden_layers)
        self.likelihood_blocks = nn.ModuleList(
            LikelihoodBlock(expert_config, first_index + index)
            for index in range(config.likelihood_num_layers)
        )
        self._likelihood_gradient_checkpointing = False
        self._freeze_stage1_prior()

    def gradient_checkpointing_enable(self) -> None:
        # The frozen 18-layer prior runs under no_grad. Checkpoint only the four
        # trainable likelihood blocks and an explicitly continued predictor.
        self._likelihood_gradient_checkpointing = True
        if self.config.finetune_skill_predictor and self.skill_predictor is not None:
            self.skill_predictor.gradient_checkpointing_enable()

    def _freeze_stage1_prior(self) -> None:
        self.requires_grad_(False)
        self.vlm_to_expert_projection.requires_grad_(True)
        self.likelihood_blocks.requires_grad_(True)
        self.action_out_proj.requires_grad_(True)
        if self.config.finetune_terminator and self.fsq_term_train is not None:
            self.fsq_term_train.requires_grad_(True)
            if self.fsq_term_train.freeze_vision_encoder:
                self.fsq_term_train.vision_encoder.requires_grad_(False)

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        # Stage 2 must not change stochastic behavior or running state in the
        # frozen prior. Optional auxiliaries retain Stage-1's isolated behavior.
        frozen_modules = (
            self.dino,
            self.image_proj,
            self.state_proj,
            self.skill_proj,
            self.action_in_proj,
            self.time_mlp_in,
            self.time_mlp_out,
            self.cond_encoder,
            self.gemma_expert,
        )
        for module in frozen_modules:
            if module is not None:
                module.eval()
        if self.config.finetune_skill_predictor:
            self.skill_predictor.train(mode)
        else:
            self.skill_predictor.eval()
        if self.fsq_term_train is not None:
            self.fsq_term_train.train(
                mode and self.config.finetune_terminator
            )
            if self.fsq_term_train.freeze_vision_encoder:
                self.fsq_term_train.vision_encoder.eval()
        self.vlm_to_expert_projection.train(mode)
        self.likelihood_blocks.train(mode)
        self.action_out_proj.train(mode)
        return self

    def _likelihood_velocity(
        self,
        prior_hidden: Tensor,
        vlm_hidden: Tensor,
        vlm_key_padding_mask: Tensor,
        expert_condition: Tensor,
    ) -> Tensor:
        hidden = prior_hidden.to(self.working_dtype)
        memory = self.vlm_to_expert_projection(
            vlm_hidden.to(self.vlm_to_expert_projection.weight.dtype)
        )
        position_ids = torch.arange(
            hidden.shape[1], device=hidden.device, dtype=torch.long
        )[None].expand(hidden.shape[0], -1)
        position_embeddings = self.gemma_expert.model.rotary_emb(hidden, position_ids)
        use_checkpoint = self._likelihood_gradient_checkpointing and self.training
        for block in self.likelihood_blocks:
            if use_checkpoint:
                hidden = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden,
                    memory,
                    vlm_key_padding_mask,
                    expert_condition,
                    position_embeddings,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                hidden = block(
                    hidden,
                    memory,
                    vlm_key_padding_mask,
                    expert_condition,
                    position_embeddings,
                )
        return self.action_out_proj(hidden.to(self.working_dtype)).float()

    def forward(
        self,
        images: list[Tensor],
        state: Tensor,
        skill_code: Tensor,
        actions: Tensor,
        language_tokens: Tensor,
        language_mask: Tensor,
        *,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        batch_size = actions.shape[0]
        time = self.sample_time(batch_size, actions.device) if time is None else time
        source = self.sample_noise(actions.shape, actions.device) if noise is None else noise
        source = source.to(actions.dtype)
        x_t = time[:, None, None] * source + (1.0 - time[:, None, None]) * actions
        target_velocity = source - actions
        expert_condition = self._expert_condition(time, state)

        with torch.no_grad():
            prior_hidden = self._run_joint_hidden(
                self._condition_tokens(images),
                x_t,
                expert_condition,
                self._skill_broadcast(skill_code),
            )
            vlm_hidden, vlm_key_padding_mask = (
                self.skill_predictor.encode_last_hidden(
                    images, language_tokens, language_mask
                )
            )
        predicted_velocity = self._likelihood_velocity(
            prior_hidden,
            vlm_hidden,
            vlm_key_padding_mask,
            expert_condition,
        )
        return target_velocity - predicted_velocity

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[Tensor],
        state: Tensor,
        skill_code: Tensor,
        language_tokens: Tensor,
        language_mask: Tensor,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        num_steps = self.config.num_inference_steps if num_steps is None else num_steps
        batch_size, device = state.shape[0], state.device
        if noise is None:
            noise = self.sample_noise(
                (batch_size, self.config.chunk_size, self.config.max_action_dim), device
            )
        condition_tokens = self._condition_tokens(images)
        n_condition = condition_tokens.shape[1]
        n_action = noise.shape[1]

        condition_padding = torch.ones(
            batch_size, n_condition, dtype=torch.bool, device=device
        )
        condition_attention = make_att_2d_masks(
            condition_padding, torch.zeros_like(condition_padding)
        )[:, None]
        condition_attention = torch.where(
            condition_attention, 0.0, OPENPI_ATTENTION_MASK_VALUE
        )
        condition_positions = torch.cumsum(condition_padding, dim=1) - 1
        condition_cache = self.cond_encoder.model.forward(
            inputs_embeds=condition_tokens,
            attention_mask=condition_attention,
            position_ids=condition_positions,
            past_key_values=None,
            use_cache=True,
            adarms_cond=None,
        ).past_key_values

        action_padding = torch.ones(batch_size, n_action, dtype=torch.bool, device=device)
        action_blocks = torch.tensor(
            [1] + [0] * (n_action - 1), dtype=torch.bool, device=device
        )[None].expand(batch_size, -1)
        action_attention = make_att_2d_masks(action_padding, action_blocks)
        condition_visible = condition_padding[:, None].expand(
            batch_size, n_action, n_condition
        )
        full_attention = torch.cat((condition_visible, action_attention), dim=2)[:, None]
        full_attention = torch.where(full_attention, 0.0, OPENPI_ATTENTION_MASK_VALUE)
        action_positions = n_condition + torch.cumsum(action_padding, dim=1) - 1
        vlm_hidden, vlm_key_padding_mask = self.skill_predictor.encode_last_hidden(
            images, language_tokens, language_mask
        )

        dt = -1.0 / num_steps
        x_t = noise
        skill_broadcast = self._skill_broadcast(skill_code)
        for step in range(num_steps):
            time = torch.full(
                (batch_size,), 1.0 + step * dt, dtype=torch.float32, device=device
            )
            expert_condition = self._expert_condition(time, state)
            prior_hidden = self._action_hidden_with_condition_cache(
                x_t,
                expert_condition,
                skill_broadcast,
                condition_cache,
                full_attention,
                action_positions,
            )
            velocity = self._likelihood_velocity(
                prior_hidden,
                vlm_hidden,
                vlm_key_padding_mask,
                expert_condition,
            )
            x_t = x_t + dt * velocity
        return x_t


_STAGE1_CONTRACT_FIELDS = (
    "action_expert_variant",
    "cond_encoder_variant",
    "chunk_size",
    "n_action_steps",
    "max_state_dim",
    "max_action_dim",
    "num_inference_steps",
    "time_sampling_beta_alpha",
    "time_sampling_beta_beta",
    "time_sampling_scale",
    "time_sampling_offset",
    "min_period",
    "max_period",
    "vision_backbone",
    "dino_image_size",
    "state_cond_mode",
    "skill_vocab_size",
    "skill_fsq_levels",
    "transition_jitter_pmax",
    "transition_jitter_distribution",
    "skill_predictor_vlm_variant",
    "skill_predictor_image_size",
    "skill_predictor_reader_tokens",
    "skill_predictor_reader_depth",
    "skill_predictor_reader_heads",
    "skill_predictor_all_layers",
    "skill_predictor_detach_vlm",
    "skill_predictor_lora",
    "skill_predictor_lora_targets",
    "skill_predictor_lora_rank",
    "skill_predictor_lora_alpha",
    "skill_predictor_lora_dropout",
    "skill_predictor_deadzone_frac",
    "skill_predictor_attend_image",
    "skill_predictor_attend_language",
    "tokenizer_max_length",
)


class SkillVLAStage2Policy(SkillExpertPolicy):
    """Likelihood policy with optional parameter-disjoint Stage-1 auxiliaries."""

    config_class = SkillVLAStage2Config
    name = "skill_vla_stage2"

    def __init__(
        self,
        config: SkillVLAStage2Config,
        *,
        initialize_from_stage1: bool = True,
        **kwargs,
    ):
        del kwargs
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.model = SkillVLAStage2Pytorch(config)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(device=config.device, dtype=self._torch_dtype())
        if self.model.fsq_term_train is not None:
            self.model.fsq_term_train.to(dtype=torch.float32)
        if initialize_from_stage1:
            self._initialize_from_stage1(config.stage1_checkpoint_path)
        self.model._freeze_stage1_prior()
        self.reset()

    def _initialize_from_stage1(self, checkpoint_path: str | Path | None) -> None:
        path = Path(str(checkpoint_path or ""))
        config_path = path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Stage-1 config not found: {config_path}")
        stage1_config = json.loads(config_path.read_text())
        if stage1_config.get("type") != "skill_expert":
            raise ValueError(
                f"Stage 2 requires a skill_expert checkpoint, got {stage1_config.get('type')!r}."
            )
        if not stage1_config.get("train_skill_predictor", False):
            raise ValueError("Stage-1 checkpoint has no trained skill predictor/VLM.")
        if not stage1_config.get("train_terminator", False):
            raise ValueError("Stage-1 checkpoint has no co-trained terminator.")
        mismatches = []
        for field in _STAGE1_CONTRACT_FIELDS:
            # These fields were introduced with the Stage3-A predictor. Their
            # absence unambiguously means the old detached, non-LoRA contract.
            if field.startswith("skill_predictor_lora") and field not in stage1_config:
                continue
            expected = stage1_config.get(field)
            actual = getattr(self.config, field)
            if expected != actual:
                mismatches.append(f"{field}: stage1={expected!r}, stage2={actual!r}")
        if mismatches:
            raise ValueError("Stage-1 architecture mismatch: " + "; ".join(mismatches))

        loaded = _load_pretrained_state_dict(path, {})
        if loaded is None:
            raise FileNotFoundError(f"Stage-1 model weights not found: {path}")
        state_dict, is_pi05 = loaded
        if is_pi05:
            raise ValueError("Stage 2 cannot initialize directly from a pi0.5 checkpoint.")
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        allowed_prefixes = (
            "model.vlm_to_expert_projection.",
            "model.likelihood_blocks.",
        )
        invalid_missing = [
            key for key in missing if not key.startswith(allowed_prefixes)
        ]
        if invalid_missing or unexpected:
            raise RuntimeError(
                "Stage-1 checkpoint mismatch: "
                f"missing={sorted(invalid_missing)}, unexpected={sorted(unexpected)}"
            )
        log.info(
            "Stage 2 <- Stage 1: loaded=%d, fresh_likelihood=%d.",
            len(state_dict),
            len(missing),
        )

    def get_optim_params(self) -> list[dict]:
        main_parameters = [
            parameter
            for module in (
                self.model.vlm_to_expert_projection,
                self.model.likelihood_blocks,
                self.model.action_out_proj,
            )
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        terminator = self.model.fsq_term_train
        terminator_parameters = (
            [
                parameter
                for parameter in terminator.parameters()
                if parameter.requires_grad
            ]
            if self.config.finetune_terminator and terminator is not None
            else []
        )
        expected = {id(parameter) for parameter in main_parameters + terminator_parameters}
        actual = {
            id(parameter)
            for parameter in self.parameters()
            if parameter.requires_grad
        }
        if actual != expected:
            raise RuntimeError("Stage-2 trainable-parameter freeze contract was violated.")
        groups = [{"params": main_parameters}]
        if terminator_parameters:
            groups.append(
                {
                    "params": terminator_parameters,
                    "lr": self.config.optimizer_lr
                    * self.config.terminator_lr_scale,
                }
            )
        return groups

    def isolated_auxiliary_step(
        self,
        batch: dict,
        accelerator,
        grad_clip_norm: float,
        current_lr: float | None = None,
    ) -> dict:
        """Optionally continue predictor reader/head training; its VLM stays frozen."""
        if not self.config.finetune_skill_predictor:
            return {}
        return SkillExpertPolicy.isolated_auxiliary_step(
            self,
            batch,
            accelerator,
            grad_clip_norm,
            current_lr=current_lr,
        )

    def isolated_main_optimizer_grad_groups(self) -> dict:
        terminator = self.model.fsq_term_train
        if not self.config.finetune_terminator or terminator is None:
            return {}
        parameters = [
            parameter for parameter in terminator.parameters() if parameter.requires_grad
        ]
        return {"terminator": parameters} if parameters else {}

    @torch.no_grad()
    def _predicted_training_skill_code(self, batch: dict) -> Tensor:
        predictor = self.model.skill_predictor
        if predictor is None:
            raise RuntimeError("Stage 2 has no loaded skill predictor.")
        device = next(self.parameters()).device
        return predictor.predict(
            self._predictor_start_images(batch),
            batch[OBS_LANGUAGE_TOKENS].to(device),
            batch[OBS_LANGUAGE_ATTENTION_MASK].to(device),
        ).long()

    def _training_skill_code(self, batch: dict) -> Tensor:
        if self.config.training_skill_source == "gt":
            return self._skill_code(batch)
        predicted = self._predicted_training_skill_code(batch)
        self._last_transition_jitter_fraction = torch.zeros(
            (), device=predicted.device
        )
        return predicted.clamp(0, self.config.skill_vocab_size - 1)

    def forward(self, batch: dict, reduction: str = "mean"):
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        real_dim = self.config.output_features[ACTION].shape[0]
        actions = self._hold_after_boundary(actions, batch, real_dim)
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        device = actions.device
        skill_code = self._training_skill_code(batch)
        residual = self.model(
            self._collect_images(batch),
            state,
            skill_code,
            actions,
            batch[OBS_LANGUAGE_TOKENS].to(device),
            batch[OBS_LANGUAGE_ATTENTION_MASK].to(device),
        )[..., :real_dim]
        squared_error = residual.square()
        per_sample = squared_error.mean(dim=(1, 2))
        loss_dict = {
            "action_loss": squared_error.mean().detach().item(),
            "loss_per_dim": squared_error.mean(dim=(0, 1)).detach().cpu().tolist(),
            "stage2/skill_source_predictor": float(
                self.config.training_skill_source == "predictor"
            ),
            "regime/transition_jitter_fraction": (
                self._last_transition_jitter_fraction.detach().item()
            ),
        }
        terminator_loss = None
        if self.config.finetune_terminator:
            terminator_loss, progress_loss, termination_loss = self._terminator_loss(
                batch
            )
            loss_dict.update(
                {
                    "terminator/loss": terminator_loss.detach().item(),
                    "terminator/progress": progress_loss.detach().item(),
                    "terminator/termination": termination_loss.detach().item(),
                }
            )
        if reduction == "none":
            return per_sample, loss_dict
        total = per_sample.mean()
        if terminator_loss is not None:
            total = total + terminator_loss
            loss_dict["loss_total"] = total.detach().item()
        return total, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, **kwargs) -> Tensor:
        self.eval()
        device = next(self.parameters()).device
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        actions = self.model.sample_actions(
            self._collect_images(batch),
            state,
            self._skill_code(batch),
            batch[OBS_LANGUAGE_TOKENS].to(device),
            batch[OBS_LANGUAGE_ATTENTION_MASK].to(device),
            **kwargs,
        )
        real_dim = self.config.output_features[ACTION].shape[0]
        return actions[..., :real_dim]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path,
        *,
        config=None,
        strict: bool = True,
        **kwargs,
    ):
        """Load a complete, self-contained Stage-2 checkpoint."""
        if config is None:
            config = PreTrainedConfig.from_pretrained(pretrained_name_or_path, **kwargs)
        return PreTrainedPolicy.from_pretrained.__func__(
            cls,
            pretrained_name_or_path,
            config=config,
            strict=strict,
            initialize_from_stage1=False,
            **kwargs,
        )
