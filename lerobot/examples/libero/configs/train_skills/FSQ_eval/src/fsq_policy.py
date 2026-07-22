"""Original FSQ action expert and terminator adapters for closed-loop eval."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch
from torch import Tensor, nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.skill_expert.configuration_skill_expert import SkillExpertConfig
from lerobot.utils.constants import ACTION, OBS_STATE

_HERE = Path(__file__).resolve()
_LIBERO_EXAMPLES = _HERE.parents[4]
_STAGE1_EVAL_SRC = _HERE.parents[3] / "train_skillVLA" / "stage1_eval" / "src"
sys.path.insert(0, str(_LIBERO_EXAMPLES))
sys.path.insert(0, str(_STAGE1_EVAL_SRC))

from FSQ import load_fsq_model  # noqa: E402


def _load_stage1_eval_module():
    module_name = "_fsq_eval_stage1_eval_run"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _STAGE1_EVAL_SRC / "run_eval.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Stage-1 eval helpers from {_STAGE1_EVAL_SRC}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_STAGE1_EVAL = _load_stage1_eval_module()
OracleSkillExpertPolicy = _STAGE1_EVAL.OracleSkillExpertPolicy
override_init_states = _STAGE1_EVAL._override_init_states

_IMAGE_KEY = "observation.images.image"
_WRIST_KEY = "observation.images.wrist_image"


class FSQExpertPolicy(nn.Module):
    """Full original FSQ checkpoint exposed through the common eval API."""

    def __init__(
        self,
        fsq_path: str | Path,
        device: torch.device,
        *,
        broadcast_scale: float,
        n_action_steps: int,
        dino_model_path: str,
    ):
        super().__init__()
        if not math.isfinite(broadcast_scale) or broadcast_scale < 0.0:
            raise ValueError(f"broadcast_scale must be finite and >= 0, got {broadcast_scale}.")
        self.broadcast_scale = float(broadcast_scale)
        self.model, self.fsq_cfg = load_fsq_model(
            fsq_path, device=device, dino_model_path=dino_model_path
        )
        if self.fsq_cfg.state_cond_mode != "broadcast" and self.broadcast_scale != 1.0:
            raise ValueError(
                f"broadcast_scale={self.broadcast_scale} is only valid for broadcast checkpoints; "
                f"{fsq_path} uses state_cond_mode={self.fsq_cfg.state_cond_mode!r}."
            )
        self.real_state_dim = int(self.fsq_cfg.state_dim)
        self.real_action_dim = int(self.fsq_cfg.action_dim)
        self.num_inference_steps = 10
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.eval()

        self.config = SkillExpertConfig(
            input_features={OBS_STATE: PolicyFeature(FeatureType.STATE, (self.real_state_dim,))},
            output_features={ACTION: PolicyFeature(FeatureType.ACTION, (self.real_action_dim,))},
            action_expert_variant=self.fsq_cfg.action_expert_variant,
            state_cond_mode=self.fsq_cfg.state_cond_mode,
            skill_vocab_size=int(self.model.fsq.codebook_size),
            skill_fsq_levels=list(self.fsq_cfg.fsq_levels),
            max_state_dim=int(self.fsq_cfg.max_state_dim),
            max_action_dim=int(self.fsq_cfg.max_action_dim),
            chunk_size=int(self.fsq_cfg.chunk_size),
            n_action_steps=min(int(n_action_steps), int(self.fsq_cfg.chunk_size)),
            num_inference_steps=self.num_inference_steps,
            device=str(device),
            dtype="bfloat16" if self.fsq_cfg.expert_dtype == "bfloat16" else "float32",
            fsq_path=str(fsq_path),
        )

    def dataset_stats(self) -> dict[str, dict[str, Tensor]]:
        return {
            OBS_STATE: {
                "q01": torch.as_tensor(self.fsq_cfg.state_q01, dtype=torch.float32),
                "q99": torch.as_tensor(self.fsq_cfg.state_q99, dtype=torch.float32),
            },
            ACTION: {
                "q01": torch.as_tensor(self.fsq_cfg.action_q01, dtype=torch.float32),
                "q99": torch.as_tensor(self.fsq_cfg.action_q99, dtype=torch.float32),
            },
        }

    def reset(self) -> None:
        return None

    def _skill_code(self, batch: dict) -> Tensor:
        sequence = batch["skill_sequence"].long()
        index = batch["skill_index"].long().view(-1, 1).clamp(0, sequence.shape[1] - 1)
        return sequence.gather(1, index).squeeze(1).clamp(0, self.model.fsq.codebook_size - 1)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, **_: object) -> Tensor:
        device = next(self.model.parameters()).device
        normalized = batch[OBS_STATE].to(device=device, dtype=torch.float32)
        state = torch.zeros(
            normalized.shape[0], self.fsq_cfg.max_state_dim, device=device, dtype=torch.float32
        )
        state[:, : self.real_state_dim] = normalized[:, : self.real_state_dim]
        code = self._skill_code(batch).to(device)
        z_norm = self.model.fsq.code_to_normalized(code).to(device)
        actions = self.model.action_expert.sample_actions(
            state,
            z_norm,
            num_steps=self.num_inference_steps,
            broadcast_scale=self.broadcast_scale,
        )
        return actions[:, :, : self.real_action_dim]

    def forward(self, batch: dict, *args, **kwargs):  # pragma: no cover
        return self.predict_action_chunk(batch, **kwargs)


class FSQTerminator:
    """Terminator view sharing the policy's already-loaded original FSQ model."""

    use_wrist = True

    def __init__(self, policy: FSQExpertPolicy):
        self.policy = policy

    @torch.no_grad()
    def terminate(
        self,
        codes: Tensor,
        state: Tensor,
        image: Tensor,
        wrist_image: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if wrist_image is None:
            raise ValueError("FSQ terminator requires the current wrist image.")
        device = next(self.policy.model.parameters()).device
        z_norm = self.policy.model.fsq.code_to_normalized(codes.to(device).long())
        raw_state = state.to(device=device, dtype=torch.float32)[:, : self.policy.real_state_dim]
        progress, logits = self.policy.model.terminator(
            z_norm,
            raw_state,
            image.to(device),
            wrist_image.to(device),
        )
        return progress, torch.sigmoid(logits)
