"""Image/cond-free FSQ action expert adapted to the Stage-1 oracle eval API."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.skill_expert.configuration_skill_expert import SkillExpertConfig
from lerobot.utils.constants import ACTION, OBS_STATE


class FSQExpertOnlyPolicy(nn.Module):
    """Load only ``action_expert.*`` from FSQ.pt; no vision or cond modules exist."""

    def __init__(self, fsq_path: str | Path, device: torch.device, *, n_action_steps: int):
        super().__init__()
        from FSQ import FSQ, VSAFlowExpert, load_fsq_action_expert_state  # noqa: PLC0415

        state, fcfg = load_fsq_action_expert_state(fsq_path)
        self.fsq_cfg = fcfg
        self.real_state_dim = int(fcfg.state_dim)
        self.real_action_dim = int(fcfg.action_dim)
        self.num_inference_steps = 10
        self.expert = VSAFlowExpert(
            variant=fcfg.action_expert_variant,
            fsq_levels=list(fcfg.fsq_levels),
            state_cond_mode=fcfg.state_cond_mode,
            max_state_dim=fcfg.max_state_dim,
            max_action_dim=fcfg.max_action_dim,
            chunk_size=fcfg.chunk_size,
            min_period=fcfg.min_period,
            max_period=fcfg.max_period,
            time_sampling_beta_alpha=fcfg.time_sampling_beta_alpha,
            time_sampling_beta_beta=fcfg.time_sampling_beta_beta,
            time_sampling_scale=fcfg.time_sampling_scale,
            time_sampling_offset=fcfg.time_sampling_offset,
        )
        self.expert.load_state_dict(state, strict=True)
        self.fsq = FSQ(list(fcfg.fsq_levels))
        self.to(device)
        if device.type == "cuda" and fcfg.expert_dtype == "bfloat16":
            self.expert.to(dtype=torch.bfloat16)
        elif device.type == "cuda" and fcfg.expert_dtype == "float16":
            self.expert.to(dtype=torch.float16)
        self.eval()

        # This config is used only by the common oracle wrapper and the existing
        # Stage-1 processor factory. The FSQ expert itself is the module above.
        self.config = SkillExpertConfig(
            input_features={OBS_STATE: PolicyFeature(FeatureType.STATE, (self.real_state_dim,))},
            output_features={ACTION: PolicyFeature(FeatureType.ACTION, (self.real_action_dim,))},
            action_expert_variant=fcfg.action_expert_variant,
            state_cond_mode=fcfg.state_cond_mode,
            skill_vocab_size=int(self.fsq.codebook_size),
            skill_fsq_levels=list(fcfg.fsq_levels),
            max_state_dim=int(fcfg.max_state_dim),
            max_action_dim=int(fcfg.max_action_dim),
            chunk_size=int(fcfg.chunk_size),
            n_action_steps=min(int(n_action_steps), int(fcfg.chunk_size)),
            num_inference_steps=self.num_inference_steps,
            device=str(device),
            dtype="bfloat16" if fcfg.expert_dtype == "bfloat16" else "float32",
            fsq_path=str(fsq_path),
        )

    def dataset_stats(self) -> dict[str, dict[str, Tensor]]:
        """Quantile stats expected by the unmodified Stage-1 pre/post processors."""
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
        seq = batch["skill_sequence"].long()
        idx = batch["skill_index"].long().view(-1, 1).clamp(0, seq.shape[1] - 1)
        return seq.gather(1, idx).squeeze(1).clamp(0, self.fsq.codebook_size - 1)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, **_: object) -> Tensor:
        device = next(self.expert.parameters()).device
        raw = batch[OBS_STATE].to(device=device, dtype=torch.float32)
        state = torch.zeros(
            raw.shape[0], self.fsq_cfg.max_state_dim, device=device, dtype=torch.float32
        )
        state[:, : self.real_state_dim] = raw[:, : self.real_state_dim]
        code = self._skill_code(batch).to(device)
        z_norm = self.fsq.code_to_normalized(code).to(device)
        actions = self.expert.sample_actions(
            state, z_norm, num_steps=self.num_inference_steps
        )
        return actions[:, :, : self.real_action_dim]

    def forward(self, batch: dict, *args, **kwargs):  # pragma: no cover - eval uses predict_action_chunk
        return self.predict_action_chunk(batch, **kwargs)
