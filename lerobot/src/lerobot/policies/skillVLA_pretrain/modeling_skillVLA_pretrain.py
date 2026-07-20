from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.lora import (
    inject_named_lora,
    route_plain_to_base,
    set_active_adapters,
    target_names_from_spec,
)
from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy, PI0FastPytorch
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.skill_expert.modeling_skill_expert import _load_raw_state_dict
from lerobot.utils.constants import (
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

from .configuration_skillVLA_pretrain import SkillVLAPretrainConfig

log = logging.getLogger(__name__)


class SkillVLAPretrainPytorch(PI0FastPytorch):
    def __init__(self, config: SkillVLAPretrainConfig, *, paligemma_tokenizer):
        super().__init__(config, rtc_processor=None, paligemma_tokenizer=paligemma_tokenizer)
        self.config = config
        self._build_token_contract(paligemma_tokenizer)

        self.token_input_delta = None
        self.token_output_delta = None
        if config.training_mode == "lora":
            for parameter in self.paligemma_with_expert.paligemma.parameters():
                parameter.requires_grad_(False)
            names = target_names_from_spec(config.pretrain_lora_targets)
            count = inject_named_lora(
                self.paligemma_with_expert.paligemma.model.language_model,
                names,
                "pretrain",
                config.pretrain_lora_rank,
                config.pretrain_lora_alpha,
                config.pretrain_lora_dropout,
            )
            hidden = int(
                self.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight.shape[1]
            )
            dtype = self.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight.dtype
            self.token_input_delta = nn.Embedding(len(self._custom_token_ids), hidden, dtype=dtype)
            self.token_output_delta = nn.Parameter(torch.zeros(len(self._custom_token_ids), hidden, dtype=dtype))
            nn.init.zeros_(self.token_input_delta.weight)
            log.info(
                "SkillVLA pretrain LoRA: %d layers, rank=%d, targets=%s; %d selected token rows trainable.",
                count,
                config.pretrain_lora_rank,
                sorted(names),
                len(self._custom_token_ids),
            )
        else:
            for parameter in self.paligemma_with_expert.paligemma.parameters():
                parameter.requires_grad_(True)
            log.info("SkillVLA pretrain FULL: complete PaliGemma vision/projector/LLM/head trainable.")

    def _build_token_contract(self, tokenizer) -> None:
        levels = [int(level) for level in self.config.skill_fsq_levels]
        next_unused = int(self.config.skill_unused_start)
        skill_rows = []
        for dim, level_count in enumerate(levels):
            row = []
            for _ in range(level_count):
                token = f"<unused{next_unused}>"
                token_id = int(tokenizer.convert_tokens_to_ids(token))
                if tokenizer.convert_ids_to_tokens(token_id) != token:
                    raise ValueError(
                        f"PaliGemma tokenizer does not expose the requested skill token {token!r}."
                    )
                row.append(token_id)
                next_unused += 1
            skill_rows.append(row)

        text_vocab = int(tokenizer.vocab_size)
        fast_high = text_vocab - 1 - int(self.config.fast_skip_tokens)
        fast_low = fast_high - int(self.config.fast_vocab_size) + 1
        if fast_low < 0:
            raise ValueError(
                f"FAST token range underflows text vocab: vocab={text_vocab}, "
                f"skip={self.config.fast_skip_tokens}, fast_vocab={self.config.fast_vocab_size}."
            )
        fast_ids = list(range(fast_low, fast_high + 1))
        flat_skill = [token for row in skill_rows for token in row]
        custom = flat_skill + fast_ids
        if len(set(custom)) != len(custom):
            raise ValueError("Skill and FAST token ranges overlap.")

        max_levels = max(levels)
        skill_table = torch.full((len(levels), max_levels), -1, dtype=torch.long)
        for dim, row in enumerate(skill_rows):
            skill_table[dim, : len(row)] = torch.tensor(row, dtype=torch.long)
        strides = torch.ones(len(levels), dtype=torch.long)
        for dim in range(1, len(levels)):
            strides[dim] = strides[dim - 1] * levels[dim - 1]

        token_to_slot = torch.full((text_vocab,), -1, dtype=torch.long)
        token_to_slot[torch.tensor(custom, dtype=torch.long)] = torch.arange(len(custom))
        token_to_skill_dim = torch.full((text_vocab,), -1, dtype=torch.long)
        for dim, row in enumerate(skill_rows):
            token_to_skill_dim[torch.tensor(row, dtype=torch.long)] = dim

        self.register_buffer("_skill_token_table", skill_table, persistent=True)
        self.register_buffer("_skill_levels", torch.tensor(levels, dtype=torch.long), persistent=True)
        self.register_buffer("_skill_strides", strides, persistent=True)
        self.register_buffer("_custom_token_ids", torch.tensor(custom, dtype=torch.long), persistent=True)
        self.register_buffer("_token_to_slot", token_to_slot, persistent=False)
        self.register_buffer("_token_to_skill_dim", token_to_skill_dim, persistent=False)
        self.register_buffer("_fast_token_ids", torch.tensor(fast_ids, dtype=torch.long), persistent=True)
        log.info(
            "Skill token contract: levels=%s ids=%s; FAST text IDs=[%d,%d].",
            levels, skill_rows, fast_low, fast_high,
        )

    def _skill_tokens(self, flat_code: Tensor) -> Tensor:
        code = flat_code.long().reshape(-1, 1)
        indices = torch.div(code, self._skill_strides[None], rounding_mode="floor")
        indices = indices % self._skill_levels[None]
        dims = torch.arange(len(self._skill_levels), device=code.device)[None].expand(code.shape[0], -1)
        return self._skill_token_table[dims, indices]

    def _combined_targets(
        self, skill_code: Tensor, action_tokens: Tensor, action_masks: Tensor
    ) -> tuple[Tensor, Tensor]:
        if action_tokens.ndim != 2 or action_masks.ndim != 2:
            raise ValueError("FAST action tokens/masks must be batched rank-2 tensors.")
        bos = int(self._paligemma_tokenizer.bos_token_id)
        if not bool((action_tokens[:, 0] == bos).all()):
            raise ValueError("FAST processor target must begin with the PaliGemma BOS token.")
        skill = self._skill_tokens(skill_code)
        targets = torch.cat([action_tokens[:, :1], skill, action_tokens[:, 1:]], dim=1)
        masks = torch.cat(
            [action_masks[:, :1].bool(), torch.ones_like(skill, dtype=torch.bool), action_masks[:, 1:].bool()],
            dim=1,
        )
        return targets, masks

    def _add_input_delta(self, embeddings: Tensor, target_ids: Tensor) -> Tensor:
        if self.token_input_delta is None:
            return embeddings
        slots = self._token_to_slot[target_ids]
        valid = slots >= 0
        delta = self.token_input_delta(slots.clamp_min(0))
        # PI0-FAST scales token embeddings by sqrt(hidden); keep this exactly equivalent to
        # changing only the selected rows in the frozen embedding table.
        delta = delta * math.sqrt(delta.shape[-1])
        return embeddings + delta.to(embeddings.dtype) * valid.unsqueeze(-1).to(embeddings.dtype)

    def _output_delta_logits(self, hidden: Tensor) -> Tensor | None:
        if self.token_output_delta is None:
            return None
        return F.linear(hidden.to(self.token_output_delta.dtype), self.token_output_delta).float()

    def _token_ce(self, base_logits: Tensor, hidden: Tensor, targets: Tensor) -> tuple[Tensor, Tensor | None]:
        if self.token_output_delta is None:
            loss = F.cross_entropy(
                base_logits.float().reshape(-1, base_logits.shape[-1]),
                targets.reshape(-1),
                reduction="none",
            ).reshape_as(targets)
            return loss, None

        delta_logits = self._output_delta_logits(hidden)
        base = base_logits.float()
        base_lse = torch.logsumexp(base, dim=-1)
        base_custom = base.index_select(-1, self._custom_token_ids)
        adjusted_custom = base_custom + delta_logits

        custom_mass = torch.exp(base_custom - base_lse.unsqueeze(-1)).sum(dim=-1)
        custom_mass = custom_mass.clamp(max=1.0 - 1e-7)
        noncustom_lse = base_lse + torch.log1p(-custom_mass)
        adjusted_lse = torch.logaddexp(noncustom_lse, torch.logsumexp(adjusted_custom, dim=-1))

        target_logits = base.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        slots = self._token_to_slot[targets]
        selected = slots >= 0
        target_delta = delta_logits.gather(-1, slots.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        target_logits = target_logits + target_delta * selected.to(target_delta.dtype)
        return adjusted_lse - target_logits, delta_logits

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
        return (values * mask.to(values.dtype)).sum() / mask.sum().clamp(min=1)

    def _selected_scores(
        self, base_logits: Tensor, delta_logits: Tensor | None, token_ids: Tensor
    ) -> Tensor:
        scores = base_logits.float().index_select(-1, token_ids)
        if delta_logits is not None:
            slots = self._token_to_slot[token_ids]
            scores = scores + delta_logits.index_select(-1, slots)
        return scores

    def forward(
        self,
        images,
        img_masks,
        language_tokens: Tensor,
        language_masks: Tensor,
        action_tokens: Tensor,
        action_masks: Tensor,
        skill_code: Tensor,
    ) -> dict[str, Tensor]:
        if self.config.training_mode == "lora":
            set_active_adapters({"pretrain"})
        targets, target_masks = self._combined_targets(skill_code, action_tokens, action_masks)
        embeddings, pad_masks, att_masks, _, _ = self.embed_prefix_fast(
            images,
            img_masks,
            language_tokens,
            language_masks,
            fast_action_tokens=targets,
            fast_action_masks=target_masks,
        )
        target_len = targets.shape[1]
        target_emb = self._add_input_delta(embeddings[:, -target_len:], targets)
        embeddings = torch.cat([embeddings[:, :-target_len], target_emb], dim=1)
        if self.paligemma_with_expert.paligemma.model.language_model.layers[
            0
        ].self_attn.q_proj.weight.dtype == torch.bfloat16:
            embeddings = embeddings.to(torch.bfloat16)

        positions = torch.cumsum(pad_masks, dim=1) - 1
        attention = self._prepare_attention_masks_4d(att_masks, dtype=embeddings.dtype)
        (hidden, _), _ = self.paligemma_with_expert.forward(
            attention_mask=attention,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[embeddings, None],
            use_cache=False,
            adarms_cond=[None, None],
        )
        target_hidden = hidden[:, -target_len:]
        pred_hidden = target_hidden[:, :-1]
        next_targets = targets[:, 1:]
        next_masks = target_masks[:, 1:]
        base_logits = self.paligemma_with_expert.paligemma.lm_head(pred_hidden)
        token_loss, delta_logits = self._token_ce(base_logits, pred_hidden, next_targets)

        skill_dims = self._token_to_skill_dim[next_targets]
        skill_mask = next_masks & (skill_dims >= 0)
        fast_low, fast_high = int(self._fast_token_ids[0]), int(self._fast_token_ids[-1])
        fast_mask = next_masks & (next_targets >= fast_low) & (next_targets <= fast_high)
        structure_mask = next_masks & ~skill_mask & ~fast_mask
        skill_loss = self._masked_mean(token_loss, skill_mask)
        fast_loss = self._masked_mean(token_loss, fast_mask)
        structure_loss = self._masked_mean(token_loss, structure_mask)
        loss = (
            float(self.config.skill_loss_weight) * skill_loss
            + float(self.config.fast_loss_weight) * fast_loss
            + float(self.config.structure_loss_weight) * structure_loss
        )

        with torch.no_grad():
            skill_correct = []
            for dim, level_count in enumerate(self._skill_levels.tolist()):
                ids = self._skill_token_table[dim, :level_count]
                scores = self._selected_scores(
                    base_logits[:, dim], None if delta_logits is None else delta_logits[:, dim], ids
                )
                pred = ids[scores.argmax(dim=-1)]
                skill_correct.append(pred == next_targets[:, dim])
            skill_correct = torch.stack(skill_correct, dim=1)
            skill_token_acc = skill_correct.float().mean()
            skill_exact_acc = skill_correct.all(dim=1).float().mean()

            fast_scores = self._selected_scores(base_logits, delta_logits, self._fast_token_ids)
            fast_pred = self._fast_token_ids[fast_scores.argmax(dim=-1)]
            fast_token_acc = self._masked_mean((fast_pred == next_targets).float(), fast_mask)

        return {
            "loss": loss,
            "skill_ce": skill_loss,
            "fast_ce": fast_loss,
            "structure_ce": structure_loss,
            "skill_token_acc": skill_token_acc,
            "skill_exact_acc": skill_exact_acc,
            "fast_token_acc": fast_token_acc,
        }


class SkillVLAPretrainPolicy(PI0FastPolicy):
    config_class = SkillVLAPretrainConfig
    name = "skill_vla_pretrain"

    def __init__(self, config: SkillVLAPretrainConfig, **kwargs):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        from transformers import AutoTokenizer

        self._paligemma_tokenizer = AutoTokenizer.from_pretrained(
            config.text_tokenizer_name,
            trust_remote_code=True,
            add_eos_token=True,
            add_bos_token=False,
        )
        self.rtc_processor = None
        self.model = SkillVLAPretrainPytorch(
            config, paligemma_tokenizer=self._paligemma_tokenizer
        )
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)
        self.reset()
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.parameters())
        log.info(
            "SkillVLA pretrain mode=%s trainable=%0.1fM / total=%0.1fM.",
            config.training_mode, trainable / 1e6, total / 1e6,
        )

    @classmethod
    def from_pretrained(
        cls: type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        if config is None:
            config = PreTrainedConfig.from_pretrained(pretrained_name_or_path, **kwargs)
        policy = cls(config, **kwargs)
        raw = _load_raw_state_dict(pretrained_name_or_path, kwargs)
        if raw is None:
            raise FileNotFoundError(f"No model weights found at {pretrained_name_or_path}")

        # Native pretrain checkpoint: restore its LoRA/token deltas exactly.
        if any(key.startswith("model.") for key in raw):
            missing, unexpected = policy.load_state_dict(raw, strict=strict)
            if unexpected:
                raise RuntimeError(f"Unexpected pretrain checkpoint tensors: {sorted(unexpected)}")
            log.info(
                "Restored SkillVLA pretrain checkpoint: %d tensors, %d missing.",
                len(raw), len(missing),
            )
            return policy

        mapped = {}
        prefix = "paligemma_with_expert.paligemma."
        lm_head = None
        for key, value in raw.items():
            if key.startswith(prefix):
                mapped[f"model.{key}"] = value
            if key == f"{prefix}lm_head.weight":
                lm_head = value
        embed_key = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        if embed_key not in mapped and lm_head is not None:
            mapped[embed_key] = lm_head.clone()

        model_keys = set(policy.state_dict())
        expected = {
            key.replace(".base.", ".")
            for key in model_keys
            if key.startswith("model.paligemma_with_expert.paligemma.")
            and ".adapters." not in key
        }
        missing_base = expected - set(mapped)
        if missing_base:
            raise RuntimeError(
                "pi05 VLM checkpoint is incomplete for pretraining; missing: "
                f"{sorted(missing_base)[:20]}{' ...' if len(missing_base) > 20 else ''}"
            )
        mapped = {key: value for key, value in mapped.items() if key in expected}
        mapped, routed = route_plain_to_base(mapped, model_keys)
        _, unexpected = policy.load_state_dict(mapped, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected pi05 VLM tensors: {sorted(unexpected)}")
        log.info(
            "Initialized SkillVLA pretrain VLM from pi05: %d tensors (%d routed into LoRA bases).",
            len(mapped), routed,
        )
        return policy

    def get_optim_params(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        images, img_masks = self._preprocess_images(batch)
        action_tokens = batch.get(ACTION_TOKENS)
        action_masks = batch.get(ACTION_TOKEN_MASK)
        skill_code = batch.get("skill_code_true")
        if action_tokens is None or action_masks is None:
            raise ValueError("FAST action tokens and masks are missing from the preprocessed batch.")
        if skill_code is None:
            raise ValueError("Missing 'skill_code_true'; the segment dataset must supply GT skill codes.")
        values = self.model.forward(
            images,
            img_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            action_tokens,
            action_masks,
            skill_code,
        )
        loss = values["loss"]
        return loss, {key: value.detach().item() for key, value in values.items()}
