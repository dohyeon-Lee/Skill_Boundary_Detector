from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoTokenizer

from lerobot.policies.pi05.lora import (
    NamedLoRALinear,
    inject_named_lora,
    route_plain_to_base,
    set_active_adapters,
    target_names_from_spec,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.skillVLA.modeling_skillVLA import SkillVLAPolicy, SkillVLAPytorch
from lerobot.policies.skill_expert.modeling_skill_expert import _load_raw_state_dict
from lerobot.utils.constants import OPENPI_ATTENTION_MASK_VALUE

from .configuration_skillVLA_stage0_pretrain import SkillVLAStage0PretrainConfig
from .dataset_skillVLA_stage0_pretrain import AR_IMAGE, AR_SKILL_CODE, AR_WRIST_IMAGE
from .processor_skillVLA_stage0_pretrain import (
    AR_ACTION_TOKEN_MASK,
    AR_ACTION_TOKENS,
    AR_LANGUAGE_MASK,
    AR_LANGUAGE_TOKENS,
)

log = logging.getLogger(__name__)


class SkillVLAStage0PretrainPytorch(SkillVLAPytorch):
    def __init__(self, config, stage1_config, *, paligemma_tokenizer, rtc_processor=None):
        self._paligemma_tokenizer = paligemma_tokenizer
        self._motor_skill_code: Tensor | None = None
        super().__init__(config, stage1_config, rtc_processor=rtc_processor)
        self._build_token_contract(paligemma_tokenizer)

        self.token_input_delta = None
        self.token_output_delta = None
        if config.pretrain_training_mode == "lora":
            hidden = int(self._vlm.embed_tokens.weight.shape[1])
            dtype = self._vlm.embed_tokens.weight.dtype
            device = self._vlm.embed_tokens.weight.device
            self.token_input_delta = nn.Embedding(
                len(self._custom_token_ids), hidden, dtype=dtype, device=device
            )
            self.token_output_delta = nn.Parameter(
                torch.zeros(len(self._custom_token_ids), hidden, dtype=dtype, device=device)
            )
            nn.init.zeros_(self.token_input_delta.weight)

    def _inject_lora_adapters(self) -> None:
        config = self.config
        if config.pretrain_training_mode == "lora":
            names = target_names_from_spec(config.pretrain_lora_targets)
            count = inject_named_lora(
                self._vlm,
                names,
                "pretrain",
                int(config.pretrain_lora_rank),
                float(config.pretrain_lora_alpha),
                float(config.pretrain_lora_dropout),
            )
            if count == 0:
                raise ValueError("No VLM layers matched the pretrained LoRA target set.")
            print(
                "[stage0-pretrain LoRA] "
                f"restored pretrain@VLM={count} r={config.pretrain_lora_rank} "
                f"targets={sorted(names)}"
            )
        super()._inject_lora_adapters()

    def _active_adapters(self, adapters=()) -> set[str]:
        active = super()._active_adapters(adapters)
        if self.config.pretrain_training_mode == "lora":
            active.add("pretrain")
        return active

    def _set_stage0_trainability(self, regime: str) -> None:
        super()._set_stage0_trainability(regime)
        if regime not in {"stage0_a", "stage0_b"}:
            return
        train_vlm = "vlm" in self._stage0_components(regime)
        for module in self.modules():
            if isinstance(module, NamedLoRALinear) and "pretrain" in module.adapters:
                for parameter in module.adapters["pretrain"].parameters():
                    parameter.requires_grad_(train_vlm)
        for module in (
            self.paligemma_with_expert.paligemma.lm_head,
            self.token_input_delta,
        ):
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(train_vlm)
        if self.token_output_delta is not None:
            self.token_output_delta.requires_grad_(train_vlm)

    def _build_token_contract(self, tokenizer) -> None:
        levels = [int(level) for level in self.config.skill_fsq_levels]
        next_unused = int(self.config.skill_unused_start)
        rows = []
        for level_count in levels:
            row = []
            for _ in range(level_count):
                token = f"<unused{next_unused}>"
                token_id = int(tokenizer.convert_tokens_to_ids(token))
                if tokenizer.convert_ids_to_tokens(token_id) != token:
                    raise ValueError(f"Tokenizer does not expose required skill token {token!r}.")
                row.append(token_id)
                next_unused += 1
            rows.append(row)

        text_vocab = int(tokenizer.vocab_size)
        fast_high = text_vocab - 1 - int(self.config.fast_skip_tokens)
        fast_low = fast_high - int(self.config.fast_vocab_size) + 1
        if fast_low < 0:
            raise ValueError("FAST token range underflows the PaliGemma vocabulary.")
        fast_ids = list(range(fast_low, fast_high + 1))
        flat_skill = [token for row in rows for token in row]
        custom = flat_skill + fast_ids
        if len(set(custom)) != len(custom):
            raise ValueError("Skill and FAST token ranges overlap.")

        table = torch.full((len(levels), max(levels)), -1, dtype=torch.long)
        for dim, row in enumerate(rows):
            table[dim, : len(row)] = torch.tensor(row, dtype=torch.long)
        token_to_slot = torch.full((text_vocab,), -1, dtype=torch.long)
        token_to_slot[torch.tensor(custom)] = torch.arange(len(custom))
        token_to_skill_dim = torch.full((text_vocab,), -1, dtype=torch.long)
        for dim, row in enumerate(rows):
            token_to_skill_dim[torch.tensor(row)] = dim

        self.register_buffer("_skill_token_table", table, persistent=True)
        self.register_buffer("_custom_token_ids", torch.tensor(custom), persistent=True)
        self.register_buffer("_fast_token_ids", torch.tensor(fast_ids), persistent=True)
        self.register_buffer("_token_to_slot", token_to_slot, persistent=False)
        self.register_buffer("_token_to_skill_dim", token_to_skill_dim, persistent=False)
        action_prefix = tokenizer.encode("Action: ", add_special_tokens=False)
        action_stop = tokenizer.encode("|", add_special_tokens=False)
        if not action_prefix:
            raise ValueError("Tokenizer produced an empty 'Action: ' prefix.")
        if len(action_stop) != 1:
            raise ValueError(
                "Predicted FAST context requires '|' to be exactly one tokenizer token; "
                f"got {action_stop}."
            )
        if int(action_stop[0]) in set(fast_ids):
            raise ValueError("FAST token range overlaps the action stop token '|'.")
        self.register_buffer(
            "_action_prefix_ids", torch.tensor(action_prefix, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_action_stop_id", torch.tensor(int(action_stop[0]), dtype=torch.long), persistent=False
        )

    def _skill_tokens(self, flat_code: Tensor) -> Tensor:
        code = flat_code.long().reshape(-1, 1)
        indices = torch.div(code, self._fsq_strides[None], rounding_mode="floor")
        indices = indices % self._fsq_levels[None]
        dims = torch.arange(len(self._fsq_levels), device=code.device)[None].expand(
            code.shape[0], -1
        )
        return self._skill_token_table[dims, indices]

    def _add_input_delta(self, embeddings: Tensor, token_ids: Tensor) -> Tensor:
        if self.token_input_delta is None:
            return embeddings
        slots = self._token_to_slot[token_ids]
        valid = slots >= 0
        delta = self.token_input_delta(slots.clamp_min(0)) * math.sqrt(embeddings.shape[-1])
        return embeddings + delta.to(embeddings.dtype) * valid.unsqueeze(-1).to(embeddings.dtype)

    def _output_delta_logits(self, hidden: Tensor) -> Tensor | None:
        if self.token_output_delta is None:
            return None
        return F.linear(hidden.to(self.token_output_delta.dtype), self.token_output_delta).float()

    def _token_ce(self, base_logits: Tensor, hidden: Tensor, targets: Tensor):
        if self.token_output_delta is None:
            loss = F.cross_entropy(
                base_logits.float().reshape(-1, base_logits.shape[-1]),
                targets.reshape(-1),
                reduction="none",
            ).reshape_as(targets)
            return loss, None
        delta = self._output_delta_logits(hidden)
        base = base_logits.float()
        base_lse = torch.logsumexp(base, dim=-1)
        base_custom = base.index_select(-1, self._custom_token_ids)
        adjusted_custom = base_custom + delta
        custom_mass = torch.exp(base_custom - base_lse.unsqueeze(-1)).sum(dim=-1)
        custom_mass = custom_mass.clamp(max=1.0 - 1e-7)
        noncustom_lse = base_lse + torch.log1p(-custom_mass)
        adjusted_lse = torch.logaddexp(
            noncustom_lse, torch.logsumexp(adjusted_custom, dim=-1)
        )
        target_logits = base.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        slots = self._token_to_slot[targets]
        selected = slots >= 0
        target_delta = delta.gather(-1, slots.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        target_logits = target_logits + target_delta * selected.to(target_delta.dtype)
        return adjusted_lse - target_logits, delta

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
        return (values * mask.to(values.dtype)).sum() / mask.sum().clamp(min=1)

    def _selected_scores(self, base_logits: Tensor, delta_logits: Tensor | None, ids: Tensor):
        scores = base_logits.float().index_select(-1, ids)
        if delta_logits is not None:
            scores = scores + delta_logits.index_select(-1, self._token_to_slot[ids])
        return scores

    def _next_fast_context_token(self, hidden: Tensor, *, allow_stop: bool) -> Tensor:
        """Greedy FAST-or-stop choice, including pretrained selected-row output deltas."""
        base_logits = self.paligemma_with_expert.paligemma.lm_head(hidden[:, None])[:, 0]
        delta = self._output_delta_logits(hidden[:, None])
        delta = None if delta is None else delta[:, 0]
        scores = self._selected_scores(base_logits, delta, self._fast_token_ids)
        candidates = self._fast_token_ids
        if allow_stop:
            stop_id = self._action_stop_id.reshape(1)
            scores = torch.cat([scores, base_logits.index_select(-1, stop_id).float()], dim=-1)
            candidates = torch.cat([candidates, stop_id])
        return candidates[scores.argmax(dim=-1)]

    @torch.no_grad()
    def _predict_fast_context(
        self,
        prefix_embeds: Tensor,
        prefix_pad: Tensor,
        skill_tokens: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Generate FAST IDs from the current VLM without ever exposing GT action tokens."""
        batch_size = prefix_embeds.shape[0]
        bos = torch.full_like(skill_tokens[:, :1], int(self._paligemma_tokenizer.bos_token_id))
        action_prefix = self._action_prefix_ids[None].expand(batch_size, -1)
        seed_ids = torch.cat([bos, skill_tokens, action_prefix], dim=1)
        seed_embeds = self._vlm.embed_tokens(seed_ids) * math.sqrt(prefix_embeds.shape[-1])
        seed_embeds = self._add_input_delta(seed_embeds, seed_ids).to(prefix_embeds.dtype)
        embeddings = torch.cat([prefix_embeds.detach(), seed_embeds], dim=1)
        seed_pad = torch.ones_like(seed_ids, dtype=torch.bool)
        current_pad = torch.cat([prefix_pad, seed_pad], dim=1)

        prefix_len, seed_len = prefix_embeds.shape[1], seed_ids.shape[1]
        allow = torch.zeros(
            batch_size,
            prefix_len + seed_len,
            prefix_len + seed_len,
            dtype=torch.bool,
            device=embeddings.device,
        )
        allow[:, :prefix_len, :prefix_len] = True
        allow[:, prefix_len:, :prefix_len] = True
        allow[:, prefix_len:, prefix_len:] = torch.tril(
            torch.ones(seed_len, seed_len, dtype=torch.bool, device=embeddings.device)
        )
        allow &= current_pad[:, None, :] & current_pad[:, :, None]
        attention = torch.where(
            allow[:, None], 0.0, OPENPI_ATTENTION_MASK_VALUE
        ).to(embeddings.dtype)
        positions = torch.cumsum(current_pad, dim=1) - 1

        gradient_checkpointing = getattr(self._vlm, "gradient_checkpointing", None)
        if gradient_checkpointing is not None:
            self._vlm.gradient_checkpointing = False
        try:
            output = self._vlm.forward(
                inputs_embeds=embeddings,
                attention_mask=attention,
                position_ids=positions,
                past_key_values=None,
                use_cache=True,
                adarms_cond=None,
            )
            cache = output.past_key_values
            if cache is None:
                raise RuntimeError("VLM returned no KV cache while generating predicted FAST context.")
            hidden = output.last_hidden_state[:, -1]
            active = torch.ones(batch_size, dtype=torch.bool, device=embeddings.device)
            generated, generated_masks = [], []
            max_tokens = int(self.config.max_action_tokens)

            for step in range(max_tokens):
                was_active = active
                next_token = self._next_fast_context_token(hidden, allow_stop=step > 0)
                next_token = torch.where(was_active, next_token, self._action_stop_id)
                stopped = was_active & (next_token == self._action_stop_id)
                fast_valid = was_active & ~stopped
                if bool(fast_valid.any()):
                    context_token = torch.where(
                        fast_valid, next_token, self._fast_token_ids[0]
                    )
                    generated.append(context_token)
                    generated_masks.append(fast_valid)
                active = was_active & ~stopped
                if not bool(active.any()) or step + 1 == max_tokens:
                    break

                next_embeds = self._vlm.embed_tokens(next_token[:, None])
                next_embeds = next_embeds * math.sqrt(next_embeds.shape[-1])
                next_embeds = self._add_input_delta(next_embeds, next_token[:, None]).to(
                    embeddings.dtype
                )
                current_pad = torch.cat([current_pad, was_active[:, None]], dim=1)
                step_attention = torch.where(
                    current_pad[:, None, None, :], 0.0, OPENPI_ATTENTION_MASK_VALUE
                ).to(next_embeds.dtype)
                step_positions = current_pad.sum(dim=1, keepdim=True).long() - 1
                output = self._vlm.forward(
                    inputs_embeds=next_embeds,
                    attention_mask=step_attention,
                    position_ids=step_positions,
                    past_key_values=cache,
                    use_cache=True,
                    adarms_cond=None,
                )
                cache = output.past_key_values
                hidden = output.last_hidden_state[:, -1]
        finally:
            if gradient_checkpointing is not None:
                self._vlm.gradient_checkpointing = gradient_checkpointing

        if not generated:
            raise RuntimeError("Predicted FAST context produced no FAST tokens.")
        tokens = torch.stack(generated, dim=1)
        masks = torch.stack(generated_masks, dim=1)
        terminated = ~active
        return tokens, masks, terminated

    def _combined_targets(
        self,
        skill_code: Tensor,
        action_tokens: Tensor | None,
        action_masks: Tensor | None,
        include_fast: bool,
    ) -> tuple[Tensor, Tensor]:
        bos = int(self._paligemma_tokenizer.bos_token_id)
        skill = self._skill_tokens(skill_code)
        if not include_fast:
            targets = torch.cat(
                [torch.full_like(skill[:, :1], bos), skill], dim=1
            )
            return targets, torch.ones_like(targets, dtype=torch.bool)
        if action_tokens is None or action_masks is None:
            raise ValueError("FAST CE is enabled but its action tokens/masks are missing.")
        if not bool((action_tokens[:, 0] == bos).all()):
            raise ValueError("FAST target sequence must begin with BOS.")
        targets = torch.cat([action_tokens[:, :1], skill, action_tokens[:, 1:]], dim=1)
        masks = torch.cat(
            [
                action_masks[:, :1].bool(),
                torch.ones_like(skill, dtype=torch.bool),
                action_masks[:, 1:].bool(),
            ],
            dim=1,
        )
        return targets, masks

    def _ar_hidden(
        self,
        start_images: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        targets: Tensor,
        target_masks: Tensor,
    ) -> tuple[Tensor, Tensor]:
        saved = self._motor_skill_code
        self._motor_skill_code = None
        try:
            prefix, prefix_pad, _ = super()._vlm_tokens(
                start_images, lang_tokens, lang_masks
            )
        finally:
            self._motor_skill_code = saved
        target_emb = self._vlm.embed_tokens(targets) * math.sqrt(self._vlm.embed_tokens.weight.shape[1])
        target_emb = self._add_input_delta(target_emb, targets).to(self._wdtype)
        embeddings = torch.cat([prefix, target_emb], dim=1)
        pad = torch.cat([prefix_pad, target_masks.bool()], dim=1)
        prefix_len, target_len = prefix.shape[1], targets.shape[1]
        allow = torch.zeros(
            embeddings.shape[0], embeddings.shape[1], embeddings.shape[1],
            dtype=torch.bool, device=embeddings.device,
        )
        allow[:, :prefix_len, :prefix_len] = True
        allow[:, prefix_len:, :prefix_len] = True
        allow[:, prefix_len:, prefix_len:] = torch.tril(
            torch.ones(target_len, target_len, dtype=torch.bool, device=embeddings.device)
        )
        allow &= pad[:, None, :] & pad[:, :, None]
        attention = torch.where(allow[:, None], 0.0, OPENPI_ATTENTION_MASK_VALUE).to(
            embeddings.dtype
        )
        positions = torch.cumsum(pad, dim=1) - 1
        output = self._vlm.forward(
            inputs_embeds=embeddings,
            attention_mask=attention,
            position_ids=positions,
            past_key_values=None,
            use_cache=False,
            adarms_cond=None,
        ).last_hidden_state
        return output[:, -target_len:], pad

    def autoregressive_loss(
        self,
        start_images: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        skill_code: Tensor,
        action_tokens: Tensor | None,
        action_masks: Tensor | None,
    ) -> dict[str, Tensor]:
        include_fast = bool(self.config.ar_fast_loss)
        targets, target_masks = self._combined_targets(
            skill_code, action_tokens, action_masks, include_fast
        )
        target_hidden, _ = self._ar_hidden(
            start_images, lang_tokens, lang_masks, targets, target_masks
        )
        pred_hidden = target_hidden[:, :-1]
        next_targets = targets[:, 1:]
        next_masks = target_masks[:, 1:]
        base_logits = self.paligemma_with_expert.paligemma.lm_head(pred_hidden)
        token_loss, delta_logits = self._token_ce(base_logits, pred_hidden, next_targets)

        skill_dims = self._token_to_skill_dim[next_targets]
        skill_mask = next_masks & (skill_dims >= 0)
        fast_mask = next_masks & (next_targets >= int(self._fast_token_ids[0])) & (
            next_targets <= int(self._fast_token_ids[-1])
        )
        structure_mask = next_masks & ~skill_mask & ~fast_mask
        skill_loss = self._masked_mean(token_loss, skill_mask)
        fast_loss = self._masked_mean(token_loss, fast_mask)
        structure_loss = self._masked_mean(token_loss, structure_mask)
        total = float(self.config.ar_skill_loss_weight) * skill_loss
        if include_fast:
            total = (
                total
                + float(self.config.ar_fast_loss_weight) * fast_loss
                + float(self.config.ar_structure_loss_weight) * structure_loss
            )

        with torch.no_grad():
            correct = []
            for dim, level_count in enumerate(self._fsq_levels.tolist()):
                ids = self._skill_token_table[dim, :level_count]
                scores = self._selected_scores(
                    base_logits[:, dim],
                    None if delta_logits is None else delta_logits[:, dim],
                    ids,
                )
                pred = ids[scores.argmax(dim=-1)]
                correct.append(pred == next_targets[:, dim])
            correct = torch.stack(correct, dim=1)
        return {
            "loss": total,
            "skill_ce": skill_loss,
            "fast_ce": fast_loss,
            "structure_ce": structure_loss,
            "skill_token_acc": correct.float().mean(),
            "skill_exact_acc": correct.all(dim=1).float().mean(),
        }

    def _vlm_tokens(self, start_images, lang_tokens, lang_masks, *, predictor=False):
        embeds, pad, xattn_block = super()._vlm_tokens(
            start_images, lang_tokens, lang_masks, predictor=predictor
        )
        if predictor or self._motor_skill_code is None:
            self._vlm_is_skill = torch.zeros(
                embeds.shape[1], dtype=torch.bool, device=embeds.device
            )
            self._vlm_is_fast = torch.zeros_like(self._vlm_is_skill)
            self._vlm_is_causal = torch.zeros_like(self._vlm_is_skill)
            return embeds, pad, xattn_block
        skill = self._skill_tokens(self._motor_skill_code)
        bos = torch.full_like(skill[:, :1], int(self._paligemma_tokenizer.bos_token_id))
        skill_ids = torch.cat([bos, skill], dim=1)
        token_ids = skill_ids
        token_pad = torch.ones_like(skill_ids, dtype=torch.bool)
        skill_width = skill_ids.shape[1]
        action_prefix_width = 0
        fast_width = 0
        if bool(self.config.attend_fast):
            fast_ids, fast_pad, terminated = self._predict_fast_context(
                embeds.detach(), pad, skill
            )
            action_prefix = self._action_prefix_ids[None].expand(skill.shape[0], -1)
            token_ids = torch.cat([skill_ids, action_prefix, fast_ids], dim=1)
            token_pad = torch.cat(
                [
                    torch.ones_like(skill_ids, dtype=torch.bool),
                    torch.ones_like(action_prefix, dtype=torch.bool),
                    fast_pad,
                ],
                dim=1,
            )
            action_prefix_width = action_prefix.shape[1]
            fast_width = fast_ids.shape[1]
            self._last_fast_context_lengths = fast_pad.sum(dim=1)
            self._last_fast_context_terminated = terminated
        token_embeds = self._vlm.embed_tokens(token_ids) * math.sqrt(embeds.shape[-1])
        token_embeds = self._add_input_delta(token_embeds, token_ids).to(embeds.dtype)
        skill_mask = torch.zeros(token_ids.shape[1], dtype=torch.bool, device=embeds.device)
        skill_mask[:skill_width] = True
        fast_mask = torch.zeros_like(skill_mask)
        if fast_width:
            fast_mask[skill_width + action_prefix_width :] = True
        causal_mask = torch.ones_like(skill_mask)
        embeds = torch.cat([embeds, token_embeds], dim=1)
        pad = torch.cat(
            [pad, token_pad.to(device=pad.device)], dim=1
        )
        self._vlm_is_lang = torch.cat(
            [self._vlm_is_lang, torch.zeros_like(skill_mask)], dim=0
        )
        self._vlm_is_skill = torch.cat(
            [torch.zeros(xattn_block.shape[0], dtype=torch.bool, device=embeds.device), skill_mask]
        )
        self._vlm_is_fast = torch.cat(
            [torch.zeros(xattn_block.shape[0], dtype=torch.bool, device=embeds.device), fast_mask]
        )
        self._vlm_is_causal = torch.cat(
            [torch.zeros(xattn_block.shape[0], dtype=torch.bool, device=embeds.device), causal_mask]
        )
        token_block = torch.ones_like(skill_mask)
        token_block[skill_mask] = not bool(self.config.attend_skill)
        token_block[fast_mask] = not bool(self.config.attend_fast)
        return embeds, pad, torch.cat([xattn_block, token_block], dim=0)

    def _motor_vlm_self_mask(self, pad: Tensor) -> Tensor:
        causal = self._vlm_is_causal
        prefix = ~causal
        allow = prefix[None, :, None] & prefix[None, None, :]
        allow = allow.expand(pad.shape[0], -1, -1).clone()
        allow |= causal[None, :, None] & prefix[None, None, :]
        positions = torch.arange(causal.shape[0], device=causal.device)
        causal_target = (
            causal[:, None] & causal[None, :] & (positions[:, None] >= positions[None, :])
        )
        allow |= causal_target[None]
        return allow & pad[:, None, :] & pad[:, :, None]

    def _mask_branch_A(self, nc, vlm_pad, vlm_xattn_block, na, drop_vlm=False):
        attention, valid = super()._mask_branch_A(
            nc, vlm_pad, vlm_xattn_block, na, drop_vlm=drop_vlm
        )
        if getattr(self, "_vlm_is_causal", None) is not None and bool(self._vlm_is_causal.any()):
            block = self._motor_vlm_self_mask(vlm_pad)
            attention[:, 0, nc : nc + vlm_pad.shape[1], nc : nc + vlm_pad.shape[1]] = torch.where(
                block, 0.0, OPENPI_ATTENTION_MASK_VALUE
            ).to(attention.dtype)
        return attention, valid

    def _prefix_self_attention_mask(self, layers, pad: Tensor) -> Tensor:
        if (
            layers is self._vlm.layers
            and getattr(self, "_vlm_is_causal", None) is not None
            and self._vlm_is_causal.shape[0] == pad.shape[1]
            and bool(self._vlm_is_causal.any())
        ):
            return self._motor_vlm_self_mask(pad)
        return super()._prefix_self_attention_mask(layers, pad)

    def forward(self, *args, **kwargs):
        self._last_fast_context_lengths = None
        self._last_fast_context_terminated = None
        self._motor_skill_code = args[5] if len(args) > 5 else kwargs.get("skill_code")
        try:
            return super().forward(*args, **kwargs)
        finally:
            self._motor_skill_code = None

    @torch.no_grad()
    def sample_actions(self, *args, **kwargs):
        skill_code = kwargs.get("skill_code")
        if skill_code is None and len(args) > 5:
            skill_code = args[5]
        if skill_code is None:
            skill_code = self.predict_skill_code(args[1], args[2], args[3])
            kwargs["skill_code"] = skill_code
        self._motor_skill_code = skill_code
        try:
            return super().sample_actions(*args, **kwargs)
        finally:
            self._motor_skill_code = None

    @torch.no_grad()
    def predict_skill_code(self, start_images, lang_tokens, lang_masks) -> Tensor:
        set_active_adapters(self._active_adapters({"vlm_lora"}))
        batch_size = lang_tokens.shape[0]
        generated = torch.full(
            (batch_size, 1),
            int(self._paligemma_tokenizer.bos_token_id),
            dtype=torch.long,
            device=lang_tokens.device,
        )
        masks = torch.ones_like(generated, dtype=torch.bool)
        level_indices = []
        for dim, level_count in enumerate(self._fsq_levels.tolist()):
            hidden, _ = self._ar_hidden(
                start_images, lang_tokens, lang_masks, generated, masks
            )
            last = hidden[:, -1:]
            base_logits = self.paligemma_with_expert.paligemma.lm_head(last)[:, 0]
            delta = self._output_delta_logits(last)
            delta = None if delta is None else delta[:, 0]
            ids = self._skill_token_table[dim, :level_count]
            scores = self._selected_scores(base_logits, delta, ids)
            index = scores.argmax(dim=-1)
            level_indices.append(index)
            generated = torch.cat([generated, ids[index, None]], dim=1)
            masks = torch.ones_like(generated, dtype=torch.bool)
        multi = torch.stack(level_indices, dim=1)
        return (multi * self._fsq_strides[None]).sum(dim=1).long()


class SkillVLAStage0PretrainPolicy(SkillVLAPolicy):
    config_class = SkillVLAStage0PretrainConfig
    name = "skill_vla_stage0_pretrain"

    def __init__(self, config: SkillVLAStage0PretrainConfig, stage1_config=None, **kwargs):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        if stage1_config is None:
            stage1_config = self._load_stage1_config(config)
        self.stage1_config = stage1_config
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_path,
            trust_remote_code=True,
            add_eos_token=True,
            add_bos_token=False,
        )
        self.model = SkillVLAStage0PretrainPytorch(
            config,
            stage1_config,
            paligemma_tokenizer=tokenizer,
            rtc_processor=self.rtc_processor,
        )
        expert_head = getattr(self.model.paligemma_with_expert.gemma_expert, "lm_head", None)
        if expert_head is not None:
            expert_head.requires_grad_(False)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self._apply_freezes()
        self.model.to(config.device)
        self.reset()

    def _apply_continual_freezes(self) -> None:
        super()._apply_continual_freezes()
        train = self.model._stage0_components("stage0_a") | self.model._stage0_components(
            "stage0_b"
        )
        train_vlm = "vlm" in train
        for module in self.model.modules():
            if isinstance(module, NamedLoRALinear) and "pretrain" in module.adapters:
                module.adapters["pretrain"].requires_grad_(train_vlm)
        self.model.paligemma_with_expert.paligemma.lm_head.requires_grad_(train_vlm)
        if self.model.token_input_delta is not None:
            self.model.token_input_delta.requires_grad_(train_vlm)
        if self.model.token_output_delta is not None:
            self.model.token_output_delta.requires_grad_(train_vlm)

    def _ar_images(self, batch: dict, count: int) -> list[Tensor]:
        return [
            self._preprocess_vlm_tensor(batch[key][:count])
            for key in (AR_IMAGE, AR_WRIST_IMAGE)
        ]

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean"):
        if reduction != "mean":
            raise ValueError("Stage0-pretrain joint objective supports reduction='mean' only.")
        motor_loss, metrics = super().forward(batch, reduction=reduction)
        fast_lengths = self.model._last_fast_context_lengths
        fast_terminated = self.model._last_fast_context_terminated
        if fast_lengths is not None:
            metrics["fast_context/length_mean"] = fast_lengths.float().mean().item()
            metrics["fast_context/length_max"] = fast_lengths.max().item()
            metrics["fast_context/terminated_frac"] = fast_terminated.float().mean().item()
        regime = getattr(self.model, "_last_regime", None)
        # Preserve the adapter set used by the checkpointed motor graph. B has no VLM graph, so the
        # VLM-only adapter can safely be active for its AR auxiliary without changing B recomputation.
        ar_adapters = {"vlm_lora", "cond_lora"} if regime == "stage0_a" else {"vlm_lora"}
        set_active_adapters(self.model._active_adapters(ar_adapters))
        count = min(int(self.config.ar_batch_size), int(batch[AR_SKILL_CODE].shape[0]))
        ar = self.model.autoregressive_loss(
            self._ar_images(batch, count),
            batch[AR_LANGUAGE_TOKENS][:count],
            batch[AR_LANGUAGE_MASK][:count],
            batch[AR_SKILL_CODE][:count].view(-1).long(),
            batch[AR_ACTION_TOKENS][:count] if self.config.ar_fast_loss else None,
            batch[AR_ACTION_TOKEN_MASK][:count] if self.config.ar_fast_loss else None,
        )
        total = motor_loss + ar["loss"]
        metrics["motor_objective_loss"] = metrics.get("loss", float(motor_loss.detach()))
        metrics["ar/loss"] = ar["loss"].detach().item()
        metrics["ar/skill_ce"] = ar["skill_ce"].detach().item()
        metrics["ar/skill_token_acc"] = ar["skill_token_acc"].detach().item()
        metrics["ar/skill_exact_acc"] = ar["skill_exact_acc"].detach().item()
        if self.config.ar_fast_loss:
            metrics["ar/fast_ce"] = ar["fast_ce"].detach().item()
            metrics["ar/structure_ce"] = ar["structure_ce"].detach().item()
        metrics["loss"] = total.detach().item()
        metrics["loss_total"] = total.detach().item()
        return total, metrics

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, config=None, strict=False, **kwargs):
        policy = super().from_pretrained(
            pretrained_name_or_path, config=config, strict=strict, **kwargs
        )
        parent_raw = _load_raw_state_dict(pretrained_name_or_path, kwargs)
        if parent_raw and any("cond_encoder." in key or ".skill_reader." in key for key in parent_raw):
            return policy
        path = str(policy.config.pretrain_checkpoint_path)
        raw = _load_raw_state_dict(path, kwargs)
        if raw is None:
            raise FileNotFoundError(f"No pretraining checkpoint weights found at {path}")
        model_keys = set(policy.state_dict())
        selected = {}
        prefixes = (
            "model.paligemma_with_expert.paligemma.",
            "model.token_input_delta.",
            "model.token_output_delta",
            "model._skill_token_table",
            "model._custom_token_ids",
            "model._fast_token_ids",
        )
        for key, value in raw.items():
            normalized = key if key.startswith("model.") else f"model.{key}"
            if normalized.startswith(prefixes):
                selected[normalized] = value.to(policy._torch_dtype()) if value.is_floating_point() else value
        if not selected:
            raise RuntimeError(f"No pretraining VLM tensors found in {path}")

        expected_base = {
            key.replace(".base.", ".")
            for key in model_keys
            if key.startswith("model.paligemma_with_expert.paligemma.")
            and ".adapters." not in key
        }
        source_base = {
            key.replace(".base.", ".")
            for key in selected
            if key.startswith("model.paligemma_with_expert.paligemma.")
            and ".adapters." not in key
        }
        missing = expected_base - source_base
        if missing:
            raise RuntimeError(
                "Pretraining checkpoint has an incomplete VLM contract; missing "
                f"{sorted(missing)[:20]}{' ...' if len(missing) > 20 else ''}"
            )
        selected, routed = route_plain_to_base(selected, model_keys)
        _, unexpected = policy.load_state_dict(selected, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected pretraining tensors: {sorted(unexpected)}")
        log.info(
            "Stage0-pretrain restored %d VLM/token tensors from %s (%d plain weights routed); "
            "FSQ expert/cond initialization remains from direct Stage-0.",
            len(selected), path, routed,
        )
        return policy
