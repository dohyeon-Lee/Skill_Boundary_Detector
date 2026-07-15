"""SkillExpert — Stage-1 standalone action expert (no VLM, no language).

Flow-matching action chunk predictor conditioned on:
  - current 3rd-person + wrist images (trainable DINOv3, shared weights),
  - robot state,
  - GT FSQ skill code as its grid coordinate z_q ∈ [-1,1]^D (one Linear token, constant
    within a skill — neighboring codes stay neighboring),
All tokens self-attend; only the action-token hidden states are decoded into actions.
Stage 2 (`skill_vla`) adds the VLM and can init its action expert from a Stage-1 checkpoint.
"""

from __future__ import annotations

import copy
import logging
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call
from transformers import AutoModel
from transformers.models.auto import CONFIG_MAPPING

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import (
    OPENPI_ATTENTION_MASK_VALUE,
    compute_layer_complete,
    create_sinusoidal_pos_embedding,
    get_gemma_config,
    layernorm_forward,
    make_att_2d_masks,
    pad_vector,
    sample_beta,
)
from lerobot.policies.pi05.lora import (
    NamedLoRALinear,
    inject_named_lora,
    route_plain_to_base,
    set_active_adapters,
    target_names_from_spec,
)
from lerobot.policies.pi_gemma import PiGemmaForCausalLM
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from .configuration_skill_expert import SkillExpertConfig

log = logging.getLogger(__name__)


_ACTION_PROJECTION_LORA_NAMES = frozenset({"action_in_proj", "action_out_proj"})
_FSQ_ACTION_EXPERT_PREFIXES = (
    "gemma_expert.", "state_proj.", "skill_proj.", "action_in_proj.",
    "action_out_proj.", "time_mlp_in.", "time_mlp_out.",
)


def _build_siglip_vision_tower(image_size: int):
    """Standalone SigLIP vision tower matching the pi05 PaliGemma vision_tower (So400m, patch 14)
    so the pi05 checkpoint's `...vision_tower.vision_model.*` weights load 1:1 (verified)."""
    from transformers import SiglipVisionModel  # noqa: PLC0415

    vlm_cfg = CONFIG_MAPPING["paligemma"]()
    vc = vlm_cfg.vision_config
    vc.image_size = image_size
    vc.intermediate_size = 4304
    vc.projection_dim = 2048
    vc.projector_hidden_act = "gelu_fast"
    return SiglipVisionModel(vc)


def _build_gemma(variant: str, *, use_adarms: bool) -> PiGemmaForCausalLM:
    """A bare PiGemma transformer (our projections feed it, so no vocab embed table / lm_head).
    use_adarms=True for the action expert (AdaRMS on the flow timestep); False for the cond-encoder."""
    cfg = get_gemma_config(variant)
    hf = CONFIG_MAPPING["gemma"](
        head_dim=cfg.head_dim,
        hidden_size=cfg.width,
        intermediate_size=cfg.mlp_dim,
        num_attention_heads=cfg.num_heads,
        num_hidden_layers=cfg.depth,
        num_key_value_heads=cfg.num_kv_heads,
        vocab_size=257152,
        hidden_activation="gelu_pytorch_tanh",
        dtype="float32",
        use_adarms=use_adarms,
        adarms_cond_dim=cfg.width if use_adarms else None,
    )
    model = PiGemmaForCausalLM(config=hf)
    model.model.embed_tokens = None  # tokens come from our projections, not a vocab table
    model.lm_head = None             # unused (we decode actions via action_out_proj)
    model.model.config._attn_implementation = "eager"  # noqa: SLF001  (custom 4D mask)
    return model


class SkillExpertPytorch(nn.Module):
    """The Stage-1 action expert network (see module docstring)."""

    def __init__(self, config: SkillExpertConfig):
        super().__init__()
        self.config = config
        expert_cfg = get_gemma_config(config.action_expert_variant)
        self.width = expert_cfg.width

        # ── Vision encoder, shared across the two cameras (DINOv3 or SigLIP) ──
        self.vision_backbone = config.vision_backbone
        self.dino = None
        self.siglip = None
        self.n_register = 0
        if config.vision_backbone == "dino":
            self.dino = AutoModel.from_pretrained(config.dino_model_path)
            vis_dim = int(self.dino.config.hidden_size)
            self.n_register = int(getattr(self.dino.config, "num_register_tokens", 0))
            self.vision_image_size = config.dino_image_size
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet
        elif config.vision_backbone == "siglip":
            # Warm-started from pi05's vision_tower in from_pretrained (params separate from the VLM).
            self.siglip = _build_siglip_vision_tower(config.siglip_image_size)
            vis_dim = int(self.siglip.config.hidden_size)
            self.vision_image_size = config.siglip_image_size
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]  # SigLIP: [0,1] → [-1,1]
        else:
            raise ValueError(f"vision_backbone must be 'dino' or 'siglip', got {config.vision_backbone!r}")
        # Statically freeze the SELECTED vision backbone (one flag for whichever vision_backbone is active).
        if config.freeze_vision_encoder:
            for p in (self.dino if self.dino is not None else self.siglip).parameters():
                p.requires_grad_(False)
        self.register_buffer("_img_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_img_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

        # ── Token projections (all → expert width) ──
        self.image_proj = nn.Linear(vis_dim, self.width)                # image token → expert token
        # State: QUANTILE-normalized [-1,1] state vector → width (pi0-style continuous projection). It
        # rides the ACTION expert's flow-time AdaRMS (NOT the image-dominated cond stream, where it was
        # starved) in BOTH modes (state / state_skill); see _expert_cond.
        self.state_proj = nn.Linear(config.max_state_dim, self.width)
        # Skill: flat code → FSQ grid coordinate z_q (codebook's little-endian frame, the same
        # value the FSQ decoder consumes), normalized per dim to [-1, 1] → ONE token, constant
        # within a skill.
        levels = torch.tensor(config.skill_fsq_levels, dtype=torch.long)
        strides = torch.ones_like(levels)
        for i in range(1, len(config.skill_fsq_levels)):
            strides[i] = strides[i - 1] * config.skill_fsq_levels[i - 1]
        self.register_buffer("_fsq_levels", levels, persistent=False)
        self.register_buffer("_fsq_strides", strides, persistent=False)
        self.register_buffer("_fsq_half", (levels - 1).float() / 2.0, persistent=False)
        self.skill_proj = nn.Linear(len(config.skill_fsq_levels), self.width)  # z_q → 1 token

        # ── Flow-matching action head (mirrors PI05) ──
        self.action_in_proj = nn.Linear(config.max_action_dim, self.width)
        self.action_out_proj = nn.Linear(self.width, config.max_action_dim)
        self.time_mlp_in = nn.Linear(self.width, self.width)
        self.time_mlp_out = nn.Linear(self.width, self.width)

        # ── Action expert transformer (Gemma, AdaRMS conditioned on the flow timestep) ──
        self.gemma_expert = _build_gemma(config.action_expert_variant, use_adarms=True)
        self._inject_expert_lora()
        # Populated only for full action-expert FT with B batches. The tensors are non-persistent
        # buffers: they follow the model across devices but never double a Stage-1 checkpoint.
        self._fsq_reference_buffers: dict[str, str] = {}

        # ── Conditioning (joint): a SEPARATE cond-encoder encodes the scene; the action expert takes
        # [skill, progress, action] and reads the cond stream via PI05-style block attention (cond⊥action).
        self._grad_ckpt = False
        variant = config.cond_encoder_variant or config.action_expert_variant
        # The cond-encoder is image-only and never gets AdaRMS (plain RMSNorm); state rides the ACTION
        # expert's AdaRMS instead (see _expert_cond), escaping the image-token starvation it suffered
        # in the cond stream.
        self.cond_encoder = _build_gemma(variant, use_adarms=False)

        # ── Optional co-trained FSQ terminator (skill-end timing for eval; gradient-disjoint) ──
        self.fsq_term_train = None
        if getattr(config, "train_terminator", False):
            self._build_terminator_trainable(config.fsq_path)

    @property
    def _wdtype(self) -> torch.dtype:
        """Working dtype of the expert stream (drives the cast at every token boundary)."""
        return self.action_in_proj.weight.dtype

    def _inject_expert_lora(self) -> None:
        """Stage-1's single named adapter follows Stage-2's action-expert ``expert`` adapter exactly.

        ``lora_targets`` selects frozen FSQ action-side linears. Gemma attention/MLP aliases are injected
        only into ``gemma_expert``; ``action_in`` / ``action_out`` select the two flow-head projections
        outside Gemma. Its active set is switched per A/B forward below; the condition encoder itself is
        never wrapped and stays the trainable image route on A batches.
        """
        if not self.config.lora_expert:
            return
        names = target_names_from_spec(self.config.lora_targets)
        action_projection_names = names & _ACTION_PROJECTION_LORA_NAMES
        gemma_names = names - _ACTION_PROJECTION_LORA_NAMES
        count = 0
        if gemma_names:
            count += inject_named_lora(
                self.gemma_expert, gemma_names, "expert", self.config.lora_rank,
                self.config.lora_alpha, self.config.lora_dropout,
            )
        if action_projection_names:
            # Inject only these top-level flow projections. Passing the full target set here would also
            # recurse into the condition encoder's q/k/v/o linears, which must remain the normal A-only
            # image path rather than part of the frozen expert adapter.
            count += inject_named_lora(
                self, action_projection_names, "expert", self.config.lora_rank,
                self.config.lora_alpha, self.config.lora_dropout,
            )
        if count == 0:
            raise ValueError(
                f"No action-expert Linears matched lora_targets={self.config.lora_targets!r}."
            )
        print(
            "[skill_expert LoRA] "
            f"expert@action-expert={count} r={self.config.lora_rank} "
            f"alpha={self.config.lora_alpha} dropout={self.config.lora_dropout} "
            f"targets={sorted(names)}"
        )

    def _set_expert_lora_active(self, active: bool) -> None:
        """Keep Stage-1's named adapter choice explicit across checkpoint recomputation and inference."""
        if self.config.lora_expert:
            set_active_adapters({"expert"} if active else set())
        else:
            set_active_adapters(None)

    def set_fsq_reference(self, state: dict[str, Tensor]) -> None:
        """Keep an immutable FSQ action-side teacher for full-FT B batches.

        LoRA B can turn its adapter off in the same model. Full fine-tuning cannot: its base
        parameters move on A batches, so it needs the original FSQ tensors as a separate teacher.
        Buffers are explicitly non-persistent to avoid duplicating them in every training checkpoint.
        """
        if self.config.lora_expert:
            return
        expected = {
            key for key in self.state_dict()
            if key.startswith(_FSQ_ACTION_EXPERT_PREFIXES)
        }
        if set(state) != expected:
            raise RuntimeError(
                "FSQ reference and Stage-1 action-expert tensor contracts are not identical: "
                f"missing={sorted(expected - set(state))}, "
                f"extra={sorted(set(state) - expected)}"
            )
        for buffer_name in self._fsq_reference_buffers.values():
            delattr(self, buffer_name)
        self._fsq_reference_buffers.clear()
        for index, (key, value) in enumerate(sorted(state.items())):
            buffer_name = f"_fsq_reference_{index}"
            reference = value.detach().clone()
            if reference.is_floating_point():
                reference = reference.to(dtype=self._wdtype)
            self.register_buffer(buffer_name, reference, persistent=False)
            self._fsq_reference_buffers[key] = buffer_name

    def _fsq_reference_module_state(self, prefix: str) -> dict[str, Tensor]:
        if not self._fsq_reference_buffers:
            raise RuntimeError(
                "Full-FT image-free B batches need an FSQ reference teacher, but it was not initialized."
            )
        return {
            key.removeprefix(prefix): getattr(self, buffer_name)
            for key, buffer_name in self._fsq_reference_buffers.items()
            if key.startswith(prefix)
        }

    def _fsq_reference_linear(self, module: nn.Module, prefix: str, value: Tensor) -> Tensor:
        return functional_call(
            module, self._fsq_reference_module_state(prefix), (value,), strict=False
        )

    def _fsq_reference_action_prefix(self, skill_code: Tensor) -> Tensor | None:
        if self.config.state_cond_mode == "state_skill":
            return None
        z_q = self._code_to_zq(skill_code).to(self._wdtype)
        return self._fsq_reference_linear(self.skill_proj, "skill_proj.", z_q).unsqueeze(1)

    def _fsq_reference_expert_cond(
        self, timestep: Tensor, state: Tensor, skill_code: Tensor
    ) -> Tensor:
        t = create_sinusoidal_pos_embedding(
            timestep, self.width, self.config.min_period, self.config.max_period, device=timestep.device
        ).to(self._wdtype)
        t = F.silu(self._fsq_reference_linear(self.time_mlp_in, "time_mlp_in.", t))
        t = F.silu(self._fsq_reference_linear(self.time_mlp_out, "time_mlp_out.", t))
        c = t + self._fsq_reference_linear(self.state_proj, "state_proj.", state.to(self._wdtype))
        if self.config.state_cond_mode == "state_skill":
            z_q = self._code_to_zq(skill_code).to(self._wdtype)
            c = c + self._fsq_reference_linear(self.skill_proj, "skill_proj.", z_q)
        return c

    def gradient_checkpointing_enable(self) -> None:
        self._grad_ckpt = True  # _run_joint checkpoints its layer loop
        if hasattr(self.gemma_expert, "gradient_checkpointing_enable"):
            self.gemma_expert.gradient_checkpointing_enable()
        if hasattr(self.cond_encoder, "gradient_checkpointing_enable"):
            self.cond_encoder.gradient_checkpointing_enable()
        enc = self.dino if self.vision_backbone == "dino" else self.siglip
        if not self.config.freeze_vision_encoder and hasattr(enc, "gradient_checkpointing_enable"):
            enc.gradient_checkpointing_enable()

    # ── Flow-matching samplers (copied from PI05Pytorch) ──
    def sample_noise(self, shape, device) -> Tensor:
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device) -> Tensor:
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    # ── Tokenization ──
    def _image_features(self, image: Tensor) -> Tensor:
        """image (B, C, H, W) in [0, 1] → (B, n_tokens, vis_dim) image tokens.
        DINO: CLS + patches (registers dropped). SigLIP: 256 patch tokens (no CLS)."""
        x = image.to(dtype=torch.float32)
        size = (self.vision_image_size, self.vision_image_size)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        x = (x - self._img_mean.float()) / self._img_std.float()
        if self.vision_backbone == "dino":
            x = x.to(dtype=next(self.dino.parameters()).dtype)
            out = self.dino(x).last_hidden_state        # (B, 1 + n_register + num_patches, dino_dim)
            cls = out[:, :1, :]
            patches = out[:, 1 + self.n_register :, :]  # drop CLS + register tokens, keep patches
            return torch.cat([cls, patches], dim=1)
        x = x.to(dtype=next(self.siglip.parameters()).dtype)
        return self.siglip(pixel_values=x).last_hidden_state  # (B, 256, siglip_dim)

    def _code_to_zq(self, code: Tensor) -> Tensor:
        """Flat FSQ code (B,) → normalized grid coordinate z_q/half ∈ [-1, 1]^D (little-endian
        strides — the FSQ codebook's own convention, matching the FSQ decoder's z input)."""
        idx = code.view(-1, 1).long()
        level_ids = torch.div(idx, self._fsq_strides[None, :], rounding_mode="floor") % self._fsq_levels[None, :]
        return (level_ids.float() - self._fsq_half[None, :]) / self._fsq_half[None, :]

    @staticmethod
    def _construct_fsq(path: str, dino_model_path: str | None = None):
        """Instantiate and load only ``terminator.*`` from a v2 FSQ checkpoint.

        The trajectory encoder and 300M action expert are never allocated on this path.
        ``dino_model_path`` overrides a checkpoint-local DINO path after moving servers."""
        import sys  # noqa: PLC0415
        sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "examples" / "libero"))
        from FSQ import load_fsq_terminator  # noqa: PLC0415

        terminator, _ = load_fsq_terminator(path, device="cpu", dino_model_path=dino_model_path)
        return terminator

    def _build_terminator_trainable(self, path: str) -> None:
        """Co-train a TRAINABLE terminator on this dataset's GT signals — warm-started from the FSQ
        checkpoint; the checkpoint loader instantiates no encoder or action expert.
        Registered as a submodule (joins the optimizer, checkpointed); its loss runs on a disjoint
        graph (GT/precomputed inputs only) → no SkillExpert effect."""
        terminator = self._construct_fsq(path, getattr(self.config, "terminator_dino_model_path", None))
        terminator.requires_grad_(True).train()
        if terminator.freeze_vision_encoder:
            terminator.vision_encoder.requires_grad_(False).eval()
        self.fsq_term_train = terminator.to(device=next(self.parameters()).device, dtype=torch.float32)
        n_tr = sum(p.numel() for p in terminator.parameters())
        log.info("Built TRAINABLE v2 FSQ terminator from %s (%d params).", path, n_tr)

    def terminator_predict(self, true_code: Tensor, state: Tensor, image: Tensor,
                           wrist_image: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """GT skill/state + current raw third/wrist frames → progress and end logits."""
        terminator = self.fsq_term_train
        dev = next(terminator.parameters()).device
        # Run in the terminator's ACTUAL weight dtype: it is built fp32, but the policy-wide cast
        # (policy.dtype=bfloat16) turns these params bf16 afterwards, so forcing fp32 inputs would
        # mismatch the (bf16) Linear weights. levels are small ints → exact in bf16.
        tdtype = next(terminator.parameters()).dtype
        st = state.to(device=dev, dtype=tdtype)
        st = st[..., : int(terminator.state_dim)]
        z_norm = self._code_to_zq(true_code.to(self._fsq_strides.device)).to(device=dev, dtype=tdtype)
        if wrist_image is None:
            raise ValueError("FSQ terminator requires observation.images.wrist_image.")
        third = image.to(device=dev, dtype=tdtype)
        wrist = wrist_image.to(device=dev, dtype=tdtype)
        return terminator(z_norm, st, third, wrist)

    def _cond_tokens(self, images: list[Tensor]) -> Tensor:
        """Scene conditioning tokens → (B, M, width): [img1 patches, img2 patches] — image ONLY.
        State and skill (z_q) are NOT here — they ride the action expert (see _action_prefix / _expert_cond),
        so they escape the image-dominated cond stream where state was starved."""
        tokens = [self.image_proj(self._image_features(image).to(self._wdtype)) for image in images]
        return torch.cat(tokens, dim=1)

    def _action_prefix(self, skill_code: Tensor, state: Tensor) -> Tensor | None:
        """The skill token prepended to the action stream (read by action tokens; prefix ⊥ action).
        state_cond_mode="state" → the skill (z_q) is a prefix token (state itself rides the expert AdaRMS,
        not a token); "state_skill" → NO prefix at all (state + skill ride the action AdaRMS). (state is
        unused here — it never becomes a token in either mode; kept for call-site symmetry.)"""
        if self.config.state_cond_mode == "state_skill":
            return None
        return self.skill_proj(self._code_to_zq(skill_code).to(self._wdtype)).unsqueeze(1)   # (B, 1, width)

    def _time_cond(self, timestep: Tensor) -> Tensor:
        t = create_sinusoidal_pos_embedding(
            timestep, self.width, self.config.min_period, self.config.max_period, device=timestep.device
        ).to(self._wdtype)
        t = F.silu(self.time_mlp_in(t))
        t = F.silu(self.time_mlp_out(t))
        return t

    def _state_cond(self, state: Tensor) -> Tensor:
        """State→AdaRMS conditioning vector (state_proj(state) → (B, width)): drives the action expert
        in BOTH modes (state / state_skill fuse state into the flow-time AdaRMS)."""
        return self.state_proj(state.to(self._wdtype))

    def _expert_cond(self, timestep: Tensor, state: Tensor, skill_code: Tensor) -> Tensor:
        """AdaRMS conditioning for the ACTION stream — each signal through its own projection, SUMMED
        (DiT-style ⊕): flow-time always; + state in BOTH modes; + skill (z_q) only in state_skill (which
        carries NO prefix tokens — the skill rides the AdaRMS instead)."""
        c = self._time_cond(timestep) + self._state_cond(state)   # state → expert AdaRMS in both modes
        if self.config.state_cond_mode == "state_skill":
            c = c + self.skill_proj(self._code_to_zq(skill_code).to(self._wdtype))
        return c

    def _run_joint(self, cond_tokens: Tensor, x_t: Tensor, expert_cond: Tensor,
                   action_prefix: Tensor | None = None, cond_cond: Tensor | None = None) -> Tensor:
        """Joint block attention over two streams (PI05 VLM↔expert pattern, cond-encoder as prefix).
        Three blocks: cond (scene, image-only) ⊥ everything; the conditioning prefix ([skill, progress?],
        present only in mode "state") reads cond + itself but NOT the action tokens (pi0 prefix⊥action);
        the action tokens read cond + prefix + action. Only the action positions (last K) decode the
        velocity. `expert_cond` is the fused flow-time + state(+skill/progress) AdaRMS conditioning for
        the action stream; `cond_cond` is the cond stream's AdaRMS conditioning (always None now — the
        cond-encoder uses plain RMSNorm)."""
        action_tokens = self.action_in_proj(x_t.to(self._wdtype))
        n_chunk = action_tokens.shape[1]
        n_prefix = action_prefix.shape[1] if action_prefix is not None else 0
        if action_prefix is not None:
            action_tokens = torch.cat([action_prefix, action_tokens], dim=1)  # [skill, progress?, action×K]
        embeds = [cond_tokens, action_tokens]
        bsize, n_cond = cond_tokens.shape[:2]
        n_act = action_tokens.shape[1]
        device = cond_tokens.device

        pad_masks = torch.ones(bsize, n_cond + n_act, dtype=torch.bool, device=device)
        # cond block (0…0, bidirectional, ⊥action); prefix block (1,0…) reads cond+prefix only;
        # action block (1,0…) reads cond+prefix+action. (pi0: conditioning prefix does not see actions.)
        ar_list = [0] * n_cond
        if n_prefix > 0:
            ar_list += [1] + [0] * (n_prefix - 1)
        ar_list += [1] + [0] * (n_chunk - 1)
        ar = torch.tensor(ar_list, dtype=torch.bool, device=device)
        att_masks = ar[None, :].expand(bsize, n_cond + n_act)
        att_2d_4d = make_att_2d_masks(pad_masks, att_masks)[:, None, :, :]
        att_2d_4d = torch.where(att_2d_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        time_cond = expert_cond
        # cond stream: plain RMSNorm (cond_cond=None); action stream: AdaRMS(time + state [+ skill/progress])
        adarms_cond = [cond_cond, time_cond]
        # Reuse PI05's two-stream layer; cond_encoder plays the "prefix" model via a tiny shim.
        shim = SimpleNamespace(model=SimpleNamespace(language_model=self.cond_encoder.model))
        use_ckpt = self._grad_ckpt and self.training
        for layer_idx in range(self.cond_encoder.model.config.num_hidden_layers):
            if use_ckpt:
                embeds = torch.utils.checkpoint.checkpoint(
                    compute_layer_complete, layer_idx, embeds, att_2d_4d, position_ids, adarms_cond,
                    use_reentrant=False, preserve_rng_state=False,
                    paligemma=shim, gemma_expert=self.gemma_expert,
                )
            else:
                embeds = compute_layer_complete(
                    layer_idx, embeds, att_2d_4d, position_ids, adarms_cond,
                    paligemma=shim, gemma_expert=self.gemma_expert,
                )
        action_hidden, _ = layernorm_forward(self.gemma_expert.model.norm, embeds[1], time_cond)
        return self.action_out_proj(action_hidden[:, -n_chunk:].to(self._wdtype)).float()  # action positions only

    def _run_expert_only(self, x_t: Tensor, expert_cond: Tensor,
                         action_prefix: Tensor | None = None) -> Tensor:
        """Image-free action-expert forward for B batches.

        There is deliberately no dummy image or zero cond token here: the condition encoder is not invoked
        and the action Gemma receives only its skill prefix (state mode) plus noisy-action tokens. The
        resulting adapter-on vs adapter-off velocity difference is therefore a direct measure of a
        state/skill-only LoRA shortcut.
        """
        action_tokens = self.action_in_proj(x_t.to(self._wdtype))
        n_chunk = action_tokens.shape[1]
        n_prefix = action_prefix.shape[1] if action_prefix is not None else 0
        if action_prefix is not None:
            action_tokens = torch.cat([action_prefix, action_tokens], dim=1)
        bsize, n_act = action_tokens.shape[:2]
        device = action_tokens.device

        # Same prefix/action causality as _run_joint, with the cond block removed entirely.
        ar_list = ([1] + [0] * (n_prefix - 1)) if n_prefix else []
        ar_list += [1] + [0] * (n_chunk - 1)
        att = torch.tensor(ar_list, dtype=torch.bool, device=device)[None, :].expand(bsize, n_act)
        pad = torch.ones(bsize, n_act, dtype=torch.bool, device=device)
        att_4d = torch.where(make_att_2d_masks(pad, att)[:, None], 0.0, OPENPI_ATTENTION_MASK_VALUE)
        pos = torch.cumsum(pad, dim=1) - 1
        action_hidden = self.gemma_expert.model.forward(
            inputs_embeds=action_tokens,
            attention_mask=att_4d,
            position_ids=pos,
            past_key_values=None,
            use_cache=False,
            adarms_cond=expert_cond,
        ).last_hidden_state
        return self.action_out_proj(action_hidden[:, -n_chunk:].to(self._wdtype)).float()

    def _run_fsq_reference_expert_only(
        self, x_t: Tensor, timestep: Tensor, state: Tensor, skill_code: Tensor
    ) -> Tensor:
        """Frozen FSQ teacher velocity for B when the main action expert is fully trainable."""
        action_tokens = self._fsq_reference_linear(
            self.action_in_proj, "action_in_proj.", x_t.to(self._wdtype)
        )
        action_prefix = self._fsq_reference_action_prefix(skill_code)
        n_chunk = action_tokens.shape[1]
        n_prefix = action_prefix.shape[1] if action_prefix is not None else 0
        if action_prefix is not None:
            action_tokens = torch.cat([action_prefix, action_tokens], dim=1)
        bsize, n_act = action_tokens.shape[:2]
        device = action_tokens.device

        ar_list = ([1] + [0] * (n_prefix - 1)) if n_prefix else []
        ar_list += [1] + [0] * (n_chunk - 1)
        att = torch.tensor(ar_list, dtype=torch.bool, device=device)[None, :].expand(bsize, n_act)
        pad = torch.ones(bsize, n_act, dtype=torch.bool, device=device)
        att_4d = torch.where(make_att_2d_masks(pad, att)[:, None], 0.0, OPENPI_ATTENTION_MASK_VALUE)
        pos = torch.cumsum(pad, dim=1) - 1
        action_hidden = functional_call(
            self.gemma_expert.model,
            self._fsq_reference_module_state("gemma_expert.model."),
            (),
            {
                "inputs_embeds": action_tokens,
                "attention_mask": att_4d,
                "position_ids": pos,
                "past_key_values": None,
                "use_cache": False,
                "adarms_cond": self._fsq_reference_expert_cond(timestep, state, skill_code),
            },
            strict=False,
        ).last_hidden_state
        return self._fsq_reference_linear(
            self.action_out_proj, "action_out_proj.", action_hidden[:, -n_chunk:].to(self._wdtype)
        ).float()

    # ── Training / inference ──
    def forward(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        state: Tensor,
        skill_code: Tensor,
        actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        """Flow-matching: returns the SIGNED velocity residual (u_t - v_t): (B, chunk_size, max_action_dim).
        Its square is the per-(b,t,dim) flow MSE; it ALSO equals the implied per-step action error
        (â - a = (source - v_t) - (source - u_t) = u_t - v_t), so cumsum over the chunk gives the
        cumulative-position error used by the policy's optional cumulative-position loss."""
        self._set_expert_lora_active(True)
        cond = self._cond_tokens(images)
        action_prefix = self._action_prefix(skill_code, state)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        source = self.sample_noise(actions.shape, actions.device) if noise is None else noise
        source = source.to(dtype=actions.dtype)
        time_exp = time[:, None, None]
        x_t = time_exp * source + (1 - time_exp) * actions
        u_t = source - actions
        expert_cond = self._expert_cond(time, state, skill_code)
        v_t = self._run_joint(cond, x_t, expert_cond, action_prefix, None)  # cond stream: plain RMSNorm
        return u_t - v_t

    def image_free_lora_anchor(
        self,
        state: Tensor,
        skill_code: Tensor,
        actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """B batch: anchor image-free action velocity to frozen FSQ, never train on GT actions.

        With LoRA, the teacher is the same frozen base with its adapter disabled. With full expert
        fine-tuning, the teacher is the original FSQ action-side snapshot. Both paths receive identical
        ``(x_t, time, state, skill)``; gradients can only affect the current image-free expert path.
        """
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        source = self.sample_noise(actions.shape, actions.device) if noise is None else noise
        source = source.to(dtype=actions.dtype)
        x_t = time[:, None, None] * source + (1 - time[:, None, None]) * actions
        action_prefix = self._action_prefix(skill_code, state)
        expert_cond = self._expert_cond(time, state, skill_code)
        if self.config.lora_expert:
            self._set_expert_lora_active(False)
            with torch.no_grad():
                base_v = self._run_expert_only(x_t, expert_cond, action_prefix)
            self._set_expert_lora_active(True)
        else:
            with torch.no_grad():
                base_v = self._run_fsq_reference_expert_only(x_t, time, state, skill_code)
        expert_v = self._run_expert_only(x_t, expert_cond, action_prefix)
        # Diagnostic only: (source - expert_v) is the image-free action prediction, so subtract the
        # GT action to obtain the same flow-MSE residual used by A. The optimized B objective above
        # remains exclusively the current-to-frozen-FSQ velocity anchor.
        return expert_v - base_v, source - actions - expert_v

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        state: Tensor,
        skill_code: Tensor,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        bsize, device = state.shape[0], state.device
        if noise is None:
            noise = self.sample_noise((bsize, self.config.chunk_size, self.config.max_action_dim), device)
        self._set_expert_lora_active(True)
        # cond⊥action, so the cond stream (image-only) is identical every step → encode it once, cache
        # its per-layer K/V, and run only the action stream ([prefix, action]) against the cache.
        cond = self._cond_tokens(images)
        action_prefix = self._action_prefix(skill_code, state)
        return self._sample_joint_cached(cond, noise, num_steps, action_prefix, state, skill_code)

    @torch.no_grad()
    def sample_actions_image_free(
        self,
        state: Tensor,
        skill_code: Tensor,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        """B-mode inference: active LoRA, but no image tokens or condition-stream attention.

        This mirrors the image-free training branch's action-side topology while running the
        normal Euler flow sampler. State, skill, and time remain in the expert AdaRMS path.
        """
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        bsize, device = state.shape[0], state.device
        if noise is None:
            noise = self.sample_noise((bsize, self.config.chunk_size, self.config.max_action_dim), device)
        self._set_expert_lora_active(True)
        action_prefix = self._action_prefix(skill_code, state)
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = torch.full((bsize,), 1.0 + step * dt, dtype=torch.float32, device=device)
            velocity = self._run_expert_only(
                x_t, self._expert_cond(time, state, skill_code), action_prefix
            )
            x_t = x_t + dt * velocity
        return x_t

    def _sample_joint_cached(self, cond_tokens: Tensor, noise: Tensor, num_steps: int,
                             action_prefix: Tensor | None = None, state: Tensor | None = None,
                             skill_code: Tensor | None = None) -> Tensor:
        """Joint-mode inference with a cached cond stream (mirrors PI05 prefix-cache / denoise_step).
        The cond-encoder runs ONCE → per-layer K/V cache; each denoising step runs only the action
        stream ([skill?, action]) against the cache (cond⊥action), with the conditioning prefix ⊥ the
        action tokens (pi0) and state-fused AdaRMS via _expert_cond."""
        bsize, n_cond = cond_tokens.shape[:2]
        n_chunk = noise.shape[1]
        n_prefix = action_prefix.shape[1] if action_prefix is not None else 0
        n_act = n_prefix + n_chunk
        device = cond_tokens.device

        # Prefix (cond): one bidirectional block, encoded once → past_key_values.
        prefix_pad = torch.ones(bsize, n_cond, dtype=torch.bool, device=device)
        prefix_att = torch.zeros(bsize, n_cond, dtype=torch.bool, device=device)  # all-0 → bidirectional
        prefix_4d = torch.where(make_att_2d_masks(prefix_pad, prefix_att)[:, None], 0.0, OPENPI_ATTENTION_MASK_VALUE)
        prefix_pos = torch.cumsum(prefix_pad, dim=1) - 1
        past_key_values = self.cond_encoder.model.forward(
            inputs_embeds=cond_tokens, attention_mask=prefix_4d, position_ids=prefix_pos,
            past_key_values=None, use_cache=True, adarms_cond=None,  # cond stream: plain RMSNorm
        ).past_key_values

        # Suffix: conditioning prefix block ([skill?]) reads cond+prefix only; action block reads
        # cond+prefix+action (prefix ⊥ action, pi0). All suffix tokens see all cond.
        suffix_pad = torch.ones(bsize, n_act, dtype=torch.bool, device=device)
        suffix_ar_list = ([1] + [0] * (n_prefix - 1)) if n_prefix > 0 else []
        suffix_ar_list += [1] + [0] * (n_chunk - 1)
        suffix_ar = torch.tensor(suffix_ar_list, dtype=torch.bool, device=device)
        suffix_att = suffix_ar[None, :].expand(bsize, n_act)
        prefix_pad_2d = prefix_pad[:, None, :].expand(bsize, n_act, n_cond)
        full_2d = torch.cat([prefix_pad_2d, make_att_2d_masks(suffix_pad, suffix_att)], dim=2)
        full_4d = torch.where(full_2d[:, None], 0.0, OPENPI_ATTENTION_MASK_VALUE)
        suffix_pos = n_cond + torch.cumsum(suffix_pad, dim=1) - 1

        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            t = torch.full((bsize,), 1.0 + step * dt, dtype=torch.float32, device=device)
            action_tokens = self.action_in_proj(x_t.to(self._wdtype))
            if action_prefix is not None:
                action_tokens = torch.cat([action_prefix, action_tokens], dim=1)  # [skill?, action×K]
            action_hidden = self.gemma_expert.model.forward(
                inputs_embeds=action_tokens, attention_mask=full_4d, position_ids=suffix_pos,
                past_key_values=copy.deepcopy(past_key_values), use_cache=False,
                adarms_cond=self._expert_cond(t, state, skill_code),
            ).last_hidden_state
            v_t = self.action_out_proj(action_hidden[:, -n_chunk:].to(self._wdtype)).float()  # action positions only
            x_t = x_t + dt * v_t
        return x_t


# ── Checkpoint loading helpers (Stage-1 init from a PI05 checkpoint, or resume) ──────────

def _map_pi05_key(key: str, vision_backbone: str = "dino") -> str | None:
    """Map a PI05 checkpoint key to the SkillExpert model key, or None to drop it.

    Always keeps the action expert: the Gemma transformer (incl. AdaRMS norms), the action
    in/out projections, and the time MLP. When vision_backbone="siglip", ALSO warm-start the
    expert's vision tower from PI05's `vision_tower` (robot-adapted prior). Everything else
    (PaliGemma LLM, multi_modal_projector, expert lm_head, discretized-state path) is dropped.
    """
    if key.startswith("paligemma_with_expert.gemma_expert."):
        rest = key[len("paligemma_with_expert.gemma_expert.") :]
        if rest.startswith("lm_head"):
            return None
        return f"model.gemma_expert.{rest}"
    # SigLIP backbone: PI05 vision_tower → expert's own SigLIP (keys are vision_model.*).
    if vision_backbone == "siglip":
        vt = "paligemma_with_expert.paligemma.model.vision_tower."
        if key.startswith(vt):
            return f"model.siglip.{key[len(vt):]}"
    if key.startswith("paligemma_with_expert."):
        return None
    for proj in ("action_in_proj.", "action_out_proj.", "time_mlp_in.", "time_mlp_out."):
        if key.startswith(proj):
            return f"model.{key}"
    # openpi legacy naming for the time MLP
    if key.startswith("action_time_mlp_in."):
        return "model.time_mlp_in." + key[len("action_time_mlp_in.") :]
    if key.startswith("action_time_mlp_out."):
        return "model.time_mlp_out." + key[len("action_time_mlp_out.") :]
    return None


def _build_state_dict(raw: dict, vision_backbone: str = "dino") -> dict:
    """Remap a raw checkpoint to SkillExpert keys. Handles a PI05 checkpoint (partial init)
    and a SkillExpert checkpoint (resume), keyed by whether PI05 prefixes are present."""
    is_pi05 = any("paligemma_with_expert" in k for k in raw)
    out = {}
    for k, v in raw.items():
        nk = _map_pi05_key(k, vision_backbone) if is_pi05 else (k if k.startswith("model.") else f"model.{k}")
        if nk is not None:
            out[nk] = v
    return out


def _load_raw_state_dict(path: str | Path, kwargs: dict) -> dict | None:
    """Load model.safetensors from a local file/dir or a HF repo id."""
    from safetensors.torch import load_file

    p = Path(path)
    if p.is_file():
        return load_file(str(p))
    if (p / "model.safetensors").is_file():
        return load_file(str(p / "model.safetensors"))
    try:
        from transformers.utils import cached_file

        resolved = cached_file(
            str(path),
            "model.safetensors",
            cache_dir=kwargs.get("cache_dir"),
            token=kwargs.get("token"),
            revision=kwargs.get("revision"),
            local_files_only=kwargs.get("local_files_only", False),
        )
        return load_file(resolved)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not resolve weights at %s: %s", path, exc)
        return None


class SkillExpertPolicy(PreTrainedPolicy):
    """Stage-1 action-expert policy (no VLM).

    Trains the SkillExpert by flow matching on (image, skill, state, action). Initializing
    from a PI05 checkpoint (e.g. lerobot/pi05_base) reuses the action expert's motion prior;
    the DINO encoder and the image/state/skill projections start fresh. A Stage-1 checkpoint
    can later seed the Stage-2 `skill_vla` action expert.
    """

    config_class = SkillExpertConfig
    name = "skill_expert"

    def __init__(self, config: SkillExpertConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = SkillExpertPytorch(config)
        self._apply_frozen_expert_lora_regime()
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(device=config.device, dtype=self._torch_dtype())
        self.reset()

    def _torch_dtype(self) -> torch.dtype:
        return torch.bfloat16 if str(self.config.dtype) == "bfloat16" else torch.float32

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def _apply_frozen_expert_lora_regime(self) -> None:
        """Select frozen-LoRA or full-fine-tuning behavior for the FSQ action side.

        ``NamedLoRALinear`` already freezes each wrapped base projection, but the rest of Gemma plus the
        action/time/state/skill projections also belong to the FSQ reconstructor contract and must remain
        immutable in the LoRA regime. The cond side is intentionally not included: it is the trainable
        image route on A. Without LoRA, that same complete action side is trainable at the base LR.
        """
        action_modules = (
            self.model.gemma_expert,
            self.model.action_in_proj,
            self.model.action_out_proj,
            self.model.time_mlp_in,
            self.model.time_mlp_out,
            self.model.state_proj,
            self.model.skill_proj,
        )
        if not self.config.lora_expert:
            for module in action_modules:
                for param in module.parameters():
                    param.requires_grad_(True)
            n = sum(param.numel() for module in action_modules for param in module.parameters())
            log.info("Stage-1 full FSQ action-expert fine-tuning: trainable parameters=%d.", n)
            return

        for module in action_modules:
            for param in module.parameters():
                param.requires_grad_(False)

        adapter_params = []
        for module in self.model.modules():
            if isinstance(module, NamedLoRALinear) and "expert" in module.adapters:
                for param in module.adapters["expert"].parameters():
                    param.requires_grad_(True)
                    adapter_params.append(param)
        if not adapter_params:
            raise RuntimeError("lora_expert=True but no named 'expert' adapter was injected.")
        n = sum(param.numel() for param in adapter_params)
        log.info("Stage-1 frozen FSQ expert: trainable expert-LoRA parameters=%d.", n)

    def _init_image_free_fsq_reference(self) -> None:
        """Load the immutable FSQ teacher required by full-FT image-free B batches."""
        if self.config.lora_expert or self.config.image_free_lora_prob <= 0.0:
            return
        if not self.config.fsq_path:
            raise ValueError("Full-FT image-free B batches need fsq_path for their frozen teacher.")
        import sys  # noqa: PLC0415

        sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "examples" / "libero"))
        from FSQ import load_fsq_action_expert_state  # noqa: PLC0415

        state, _ = load_fsq_action_expert_state(self.config.fsq_path)
        self.model.set_fsq_reference(state)
        n = sum(tensor.numel() for tensor in state.values())
        log.info("Stage-1 full-FT B teacher <- v3 FSQ %s (%d tensors, %d params).", self.config.fsq_path, len(state), n)

    def _load_fsq_action_expert(self, path: str) -> None:
        """Warm-start Stage-1 from the jointly trained v3 FSQ components.

        The action expert always transfers. If ``terminator_arch=cond``, its
        plain-RMS Gemma condition encoder, shared image projection, and selected
        vision tower transfer too. ``small`` intentionally transfers no
        terminator tensors.
        """
        import sys  # noqa: PLC0415

        sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "examples" / "libero"))
        from FSQ import load_fsq_action_expert_state, load_fsq_cond_warmstart_state  # noqa: PLC0415

        state, fsq_cfg = load_fsq_action_expert_state(path)
        checks = {
            "action_expert_variant": self.config.action_expert_variant,
            "state_cond_mode": self.config.state_cond_mode,
            "max_state_dim": self.config.max_state_dim,
            "max_action_dim": self.config.max_action_dim,
            "chunk_size": self.config.chunk_size,
            "min_period": self.config.min_period,
            "max_period": self.config.max_period,
            "time_sampling_beta_alpha": self.config.time_sampling_beta_alpha,
            "time_sampling_beta_beta": self.config.time_sampling_beta_beta,
            "time_sampling_scale": self.config.time_sampling_scale,
            "time_sampling_offset": self.config.time_sampling_offset,
        }
        for name, expected in checks.items():
            actual = getattr(fsq_cfg, name)
            if actual != expected:
                raise ValueError(
                    f"FSQ/Stage-1 expert mismatch for {name}: checkpoint={actual!r}, "
                    f"Stage-1={expected!r}."
                )
        if list(fsq_cfg.fsq_levels) != list(self.config.skill_fsq_levels):
            raise ValueError(
                f"FSQ/Stage-1 levels mismatch: {fsq_cfg.fsq_levels} != "
                f"{self.config.skill_fsq_levels}."
            )
        # LoRA wraps targeted action projections as ``<proj>.base.*`` and adds ``.adapters.expert.*``.
        # FSQ is intentionally adapter-free, so compare its contract with the base-key canonicalization.
        expected_keys = {
            key.replace(".base.", ".")
            for key in self.model.state_dict()
            if key.startswith(_FSQ_ACTION_EXPERT_PREFIXES) and ".adapters." not in key
        }
        if set(state) != expected_keys:
            raise RuntimeError(
                "FSQ and Stage-1 action-expert tensor contracts are not identical: "
                f"missing={sorted(expected_keys - set(state))}, "
                f"extra={sorted(set(state) - expected_keys)}"
            )
        routed = {f"model.{key}": value.to(self._torch_dtype()) for key, value in state.items()}
        routed, n_routed = route_plain_to_base(routed, set(self.state_dict().keys()))
        _, unexpected = self.load_state_dict(routed, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected v3 FSQ action-expert keys: {sorted(unexpected)}")
        log.info(
            "Stage-1 action expert <- v3 FSQ %s (%d tensors, %d routed into LoRA bases).",
            path, len(routed), n_routed,
        )

        cond_state, cond_cfg = load_fsq_cond_warmstart_state(path)
        if cond_state is None:
            log.info("FSQ terminator_arch=small: Stage-1 cond keeps its normal initialization.")
            return
        cond_checks = {
            "vision_backbone": self.config.vision_backbone,
            "cond_encoder_variant": self.config.cond_encoder_variant or self.config.action_expert_variant,
            "dino_image_size": self.config.dino_image_size,
            "siglip_image_size": self.config.siglip_image_size,
        }
        for name, expected in cond_checks.items():
            actual = getattr(cond_cfg, name)
            if actual != expected:
                raise ValueError(
                    f"FSQ/Stage-1 cond mismatch for {name}: checkpoint={actual!r}, Stage-1={expected!r}."
                )
        cond_prefixes = ("cond_encoder.", "image_proj.", f"{cond_cfg.vision_backbone}.")
        expected_cond = {key for key in self.model.state_dict() if key.startswith(cond_prefixes)}
        if set(cond_state) != expected_cond:
            raise RuntimeError(
                "FSQ and Stage-1 cond tensor contracts are not identical: "
                f"missing={sorted(expected_cond - set(cond_state))}, "
                f"extra={sorted(set(cond_state) - expected_cond)}"
            )
        routed_cond = {
            f"model.{key}": value.to(self._torch_dtype()) for key, value in cond_state.items()
        }
        _, unexpected = self.load_state_dict(routed_cond, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected v3 FSQ cond keys: {sorted(unexpected)}")
        log.info("Stage-1 cond/vision <- v3 FSQ %s (%d tensors).", path, len(routed_cond))

    def get_optim_params(self):
        # Optional separate LRs for: LoRA (lora_lr_scale), vision (dino_lr/siglip_lr), and the
        # co-trained terminator (terminator_lr_scale). Everything else uses the base optimizer_lr.
        base = float(self.config.optimizer_lr)
        term = ([p for p in self.model.fsq_term_train.parameters() if p.requires_grad]
                if getattr(self.model, "fsq_term_train", None) is not None else [])
        term_ids = {id(p) for p in term}

        lora = []
        lora_ids = set()
        for module in self.model.modules():
            if isinstance(module, NamedLoRALinear):
                for param in module.adapters.parameters():
                    if param.requires_grad and id(param) not in lora_ids:
                        lora_ids.add(id(param))
                        lora.append(param)

        vis_module = self.model.dino if self.model.vision_backbone == "dino" else self.model.siglip
        vis_lr = self.config.dino_lr if self.model.vision_backbone == "dino" else self.config.siglip_lr
        groups: list[dict] = []
        excluded = set(term_ids) | lora_ids
        if vis_lr is not None:
            vis = [p for p in vis_module.parameters() if p.requires_grad]
            if vis:
                groups.append({"params": vis, "lr": vis_lr})
                excluded |= {id(p) for p in vis_module.parameters()}
        others = [p for p in self.parameters() if p.requires_grad and id(p) not in excluded]
        if others:
            groups.insert(0, {"params": others})
        if lora:
            groups.append({"params": lora, "lr": base * float(self.config.lora_lr_scale)})
        if term:
            groups.append({"params": term, "lr": base * float(self.config.terminator_lr_scale)})
        return groups

    # ── batch → model inputs ──
    def _collect_images(self, batch: dict) -> tuple[list[Tensor], list[Tensor]]:
        """Gather present camera images as [0, 1] float tensors. The vision encoder's own
        normalization (ImageNet for DINO, [-1,1] for SigLIP) is applied inside the model."""
        device = next(self.parameters()).device
        images, masks = [], []
        present = [k for k in self.config.image_features if k in batch]
        if not present:
            raise ValueError(f"No image features in batch. Expected one of {list(self.config.image_features)}.")
        for key in present:
            img = batch[key].to(device=device)
            if img.dtype != torch.float32:
                img = img.float()
            if img.ndim == 4 and img.shape[1] != 3 and img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2)  # channels-last → channels-first
            images.append(img)
            masks.append(torch.ones(img.shape[0], dtype=torch.bool, device=device))
        return images, masks

    def _skill_code(self, batch: dict) -> Tensor:
        """Current GT FSQ skill code = skill_sequence[skill_index]."""
        seq = batch["skill_sequence"].long()
        idx = batch["skill_index"].long().view(-1, 1).clamp(0, seq.shape[1] - 1)
        code = seq.gather(1, idx).squeeze(1)
        return code.clamp(0, self.config.skill_vocab_size - 1)

    def _use_image_free_lora_batch(self, device: torch.device) -> bool:
        """Sample one A/B regime for the WHOLE distributed batch.

        Every DDP rank must take the same branch: A touches cond parameters while B deliberately leaves
        them out of the graph. Rank zero's coin is broadcast to avoid an unused-parameter reducer mismatch.
        """
        prob = float(self.config.image_free_lora_prob)
        if not self.training or prob <= 0.0:
            return False
        coin = torch.rand((), device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(coin, src=0)
        return bool(coin < prob)

    def forward(self, batch: dict, reduction: str = "mean"):
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        real_dim = self.config.output_features[ACTION].shape[0]
        bsize, K = actions.shape[:2]
        dev = actions.device

        # Boundary steps = episode-end padding (action_is_pad — clamped-last repeats) OR steps that spill past
        # the CURRENT skill's end (offset k > skill_de). Instead of MASKING these out of the loss, REPLACE the
        # target with a HOLD action — arm deltas → 0 (stop moving), gripper → the LAST valid step's gripper
        # (absolute, held) — and SUPERVISE them. So each skill learns to STOP + HOLD at its boundary.
        valid = torch.ones(bsize, K, dtype=torch.bool, device=dev)
        pad = batch.get("action_is_pad")
        if pad is not None:
            valid &= ~pad.to(dev).bool()
        if "skill_de" in batch:
            de = batch["skill_de"].to(dev).long().view(bsize, 1)        # frames from t to skill's last frame
            valid &= torch.arange(K, device=dev).view(1, K) <= de
        idx = torch.arange(K, device=dev).view(1, K).expand(bsize, K)
        last_valid = torch.where(valid, idx, torch.full_like(idx, -1)).max(dim=1).values  # (B,), -1 if none
        anchor = last_valid.clamp(min=0)                                # (B,) last within-skill step (= skill end)
        rows = torch.arange(bsize, device=dev)
        g = real_dim - 1                                                # gripper = last real dim (absolute)
        hold = torch.zeros_like(actions[:, 0])                          # (B, max_action_dim): arm + pad = 0
        hold[:, g] = actions[rows, anchor, g]                           # gripper holds the last valid value
        actions = torch.where(valid.unsqueeze(-1), actions, hold.unsqueeze(1))   # boundary steps → hold target

        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)   # → state_proj's fixed width (pi0)
        skill_code = self._skill_code(batch)
        time = self.model.sample_time(bsize, dev)
        noise = self.model.sample_noise(actions.shape, dev)
        image_free_lora = self._use_image_free_lora_batch(dev)
        if image_free_lora:
            # B: no image collection, DINO/SigLIP, image projection, or cond Gemma. The objective is ONLY
            # the current-expert -> frozen-FSQ velocity anchor; GT actions merely define the same x_t.
            objective_resid, diagnostic_resid = self.model.image_free_lora_anchor(
                state, skill_code, actions, noise=noise, time=time)
        else:
            # A: normal image-conditioned flow matching. Cond and LoRA are the only trainable main path.
            images, img_masks = self._collect_images(batch)
            diagnostic_resid = self.model.forward(
                images, img_masks, state, skill_code, actions, noise=noise, time=time)
            objective_resid = diagnostic_resid

        diagnostic_resid = diagnostic_resid[:, :, :real_dim]
        objective_resid = objective_resid[:, :, :real_dim]
        losses = diagnostic_resid ** 2                                  # diagnostic GT action flow MSE
        objective_losses = objective_resid ** 2                         # A=action loss, B=current-to-FSQ anchor

        # ALL K steps are supervised (boundary steps carry the hold target) — no masking. wandb `loss` = the
        # plain uniform action MSE (comparison metric; overrides the trainer's backpropped scalar via
        # output_dict — see lerobot_train). NEVER (necessarily) the optimized objective.
        plain_action = losses.mean()
        per_sample = losses.mean(dim=(1, 2))                            # (B,) per-CHUNK action loss
        objective_per_sample = objective_losses.mean(dim=(1, 2))
        # action_loss = the PLAIN (uniform) action MSE — ALWAYS computed & logged (display-only when
        # action_weight; never the backpropped value in weighted mode). The cross-run comparison axis.
        loss_dict = {"loss_per_dim": losses.mean(dim=(0, 1)).detach().cpu().numpy().tolist(),
                     "action_loss": plain_action.detach().item()}
        if image_free_lora:
            loss_dict["lora_anchor_loss"] = objective_losses.mean().detach().item()
            loss_dict["lora_regime/image_free"] = 1.0
            # ``regime/`` is routed by lerobot_train into its own WandB ``train_regime`` panel. Do not
            # emit an A loss here: sparse curves make each point unambiguously belong to its sampled regime.
            loss_dict["regime/B_anchor_loss"] = objective_losses.mean().detach().item()
            loss_dict["regime/B_image_free_action_loss"] = plain_action.detach().item()
            loss_dict["regime/A_active"] = 0.0
            loss_dict["regime/B_active"] = 1.0
        else:
            loss_dict["lora_regime/image_free"] = 0.0
            loss_dict["regime/A_flow_loss"] = plain_action.detach().item()
            loss_dict["regime/A_active"] = 1.0
            loss_dict["regime/B_active"] = 0.0
        if reduction == "none":
            return objective_per_sample, loss_dict

        # ── Diagnostic: plain per-chunk MSE bucketed by the chunk's ENDPOINT within-skill progress
        # (skill start 0 → skill end 1). action_weight up-weights high-progress chunks, so its benefit is a
        # LOWER loss in the TOP bucket — invisible in the aggregate `action_loss`. Display-only (not
        # backpropped); logged for BOTH action_weight modes so true/false runs compare bucket-by-bucket.
        # Grouped into the wandb `action_loss_prog/*` panel; NaN = no samples fell in that bucket this step. ──
        with torch.no_grad():
            if "skill_ds" in batch and "skill_de" in batch:
                ds_e = batch["skill_ds"].to(dev).float().view(bsize)
                de_e = batch["skill_de"].to(dev).float().view(bsize)
                prog_end = ((ds_e + anchor.float()) / (ds_e + de_e).clamp(min=1.0)).clamp(0.0, 1.0)
            else:
                prog_end = (anchor.float() / max(K - 1, 1)).clamp(0.0, 1.0)
            valid_s = last_valid >= 0                                    # samples with ≥1 within-skill step
            for lo, hi in ((0.0, 0.5), (0.5, 0.9), (0.9, 1.01)):
                m = valid_s & (prog_end >= lo) & (prog_end < hi)
                key = f"action_loss_prog/{int(lo * 100):02d}-{int(min(hi, 1.0) * 100):02d}"
                loss_dict[key] = per_sample[m].mean().item() if bool(m.any()) else float("nan")
            loss_dict["prog_end_mean"] = prog_end[valid_s].mean().item() if bool(valid_s.any()) else float("nan")

        cfg = self.config
        r = float(cfg.skill_end_loss_weight)

        # ── A objective = action MSE. action_weight=False → plain (uniform) MSE. action_weight=True → per-SAMPLE
        # endpoint-weighted MSE: sw = 1 + (R-1)·prog_at_endpoint (anchor = the chunk's last valid step = skill
        # end), so chunks landing near a skill's END count up to R×. B never consumes this GT-action
        # objective: it only anchors image-free expert velocity to frozen FSQ. ──
        if image_free_lora:
            loss = objective_per_sample.mean() * float(cfg.image_free_lora_anchor_weight)
            loss_dict["lora_anchor_weighted_loss"] = loss.detach().item()
            loss_dict["regime/B_objective_loss"] = loss.detach().item()
        elif cfg.action_weight:
            # Per-step within-skill progress (0 at skill start → 1 at last frame); endpoint anchor → sw.
            # Falls back to a linear position if ds/de absent.
            if "skill_ds" in batch and "skill_de" in batch:
                ds = batch["skill_ds"].to(dev).float().view(bsize, 1)
                de_f = batch["skill_de"].to(dev).float().view(bsize, 1)
                kk = torch.arange(K, device=dev).float().view(1, K)
                prog = ((ds + kk) / (ds + de_f).clamp(min=1.0)).clamp(0.0, 1.0)
            else:
                prog = (torch.arange(K, device=dev).float().view(1, K) / max(K - 1, 1)).expand(bsize, K)
            sw = (1.0 + (r - 1.0) * prog[rows, anchor]) * (last_valid >= 0).float()  # per-sample; 0 if no valid step
            loss = (per_sample * sw).sum() / sw.sum().clamp(min=1.0)
            loss_dict["action_weighted_loss"] = loss.detach().item()   # the weighted objective (weighted mode only)
            loss_dict["regime/A_objective_loss"] = loss.detach().item()
        else:
            loss = plain_action
            loss_dict["regime/A_objective_loss"] = loss.detach().item()
        # The terminator below is an INDEPENDENT (gradient-disjoint) head. B intentionally skips it as well:
        # the image-free regime must not move image inputs or the terminator.

        # ── Co-trained FSQ terminator (skill-end timing for eval). DISJOINT graph (GT/precomputed
        # inputs + its own params) → summed into the backpropped scalar; main-model grads untouched. ──
        if (self.training and not image_free_lora and cfg.train_terminator
                and self.model.fsq_term_train is not None):
            for k in ("skill_ds", "skill_de"):
                if k not in batch:
                    raise ValueError(f"train_terminator=True needs '{k}' in the batch.")
            img_3rd = batch.get(f"{OBS_IMAGES}.image")               # observation.images.image
            img_wrist = batch.get(f"{OBS_IMAGES}.wrist_image")
            if img_3rd is None or img_wrist is None:
                raise ValueError("train_terminator=True needs both third-person and wrist images.")
            # RAW state (skill_decoder_state, snapshotted pre-normalization) — the FSQ terminator
            # normalizes internally; feeding the already-normalized OBS_STATE would double-normalize
            # and diverge from closed-loop eval (which uses raw skill_decoder_state). FAIL-FAST (no
            # OBS_STATE fallback): resuming with an OLD saved processor lacking the preserve step would
            # silently re-introduce the double-normalization bug — surface it loudly instead.
            raw_state = batch.get("skill_decoder_state")
            if raw_state is None:
                raise ValueError(
                    "train_terminator=True needs RAW 'skill_decoder_state' in the batch (snapshotted "
                    "pre-normalization by SkillExpertPreserveRawStateProcessorStep). It is missing — likely "
                    "resuming with an OLD pre-processor from before that step existed. Rebuild the "
                    "pre-processor; feeding normalized observation.state would double-normalize the FSQ input.")
            prog_pred, term_logits = self.model.terminator_predict(
                self._skill_code(batch), raw_state, img_3rd, wrist_image=img_wrist)
            ds = batch["skill_ds"].float().view(-1).to(prog_pred.device)
            de = batch["skill_de"].float().view(-1).to(prog_pred.device)
            prog_tgt = (ds / (ds + de).clamp_min(1.0)).clamp(0.0, 1.0)                 # within-skill progress
            sigma = float(cfg.terminator_end_target_sigma)
            term_tgt = torch.exp(-(de ** 2) / (2.0 * sigma ** 2)) if sigma > 0 else (de == 0).float()
            pos_w = torch.tensor(float(cfg.terminator_end_pos_weight), device=term_logits.device, dtype=term_logits.dtype)
            prog_l = F.smooth_l1_loss(prog_pred, prog_tgt.to(prog_pred.dtype))              # within-skill progress regression
            term_l = F.binary_cross_entropy_with_logits(                                    # skill-END detection (BCE)
                term_logits, term_tgt.to(term_logits.dtype), pos_weight=pos_w)
            term_loss = prog_l + term_l
            loss = loss + term_loss                       # backpropped (disjoint) — NOT in loss_total
            # Logged under the "terminator/" prefix → routed to a SEPARATE wandb panel (train_terminator/*).
            loss_dict["terminator/loss"] = term_loss.detach().item()
            loss_dict["terminator/progress"] = prog_l.detach().item()
            loss_dict["terminator/termination"] = term_l.detach().item()
        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, **kwargs) -> Tensor:
        self.eval()
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)   # → state_proj's fixed width (pi0)
        skill_code = self._skill_code(batch)
        if self.config.eval_image_free_expert:
            actions = self.model.sample_actions_image_free(state, skill_code, **kwargs)
        else:
            images, img_masks = self._collect_images(batch)
            actions = self.model.sample_actions(images, img_masks, state, skill_code, **kwargs)
        real_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :real_dim]

    @torch.no_grad()
    def select_action(self, batch: dict, **kwargs) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, config=None, strict: bool = False, **kwargs):
        """Build the model and partially load weights. A PI05 checkpoint seeds only the action
        expert (motion prior); a SkillExpert checkpoint resumes fully. DINO/skill/state/image
        projections that are absent stay freshly initialized (reported as missing keys)."""
        if config is None:
            config = PreTrainedConfig.from_pretrained(pretrained_name_or_path, **kwargs)
        model = cls(config, **kwargs)
        raw = _load_raw_state_dict(pretrained_name_or_path, kwargs)
        if raw is None:
            log.warning("SkillExpert: no weights loaded, using fresh init.")
            if config.fsq_path:
                model._load_fsq_action_expert(config.fsq_path)
            model._init_image_free_fsq_reference()
            return model
        is_pi05 = any("paligemma_with_expert" in key for key in raw)
        state_dict = {k: v.to(model._torch_dtype()) for k, v in _build_state_dict(raw, config.vision_backbone).items()}
        state_dict, n_routed = route_plain_to_base(state_dict, set(model.state_dict().keys()))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        log.info(
            "SkillExpert weights: %d mapped & loaded, %d missing (fresh init), %d unexpected, %d routed into LoRA bases.",
            len(state_dict), len(missing), len(unexpected), n_routed,
        )
        # Ordering is intentional: PI05 establishes the generic motion prior first;
        # FSQ then replaces exactly that expert with its skill-conditioned motion prior.
        # A real Stage-1 resume already contains the trained expert and must not be overwritten.
        if is_pi05 and config.fsq_path:
            model._load_fsq_action_expert(config.fsq_path)
        model._init_image_free_fsq_reference()
        return model
