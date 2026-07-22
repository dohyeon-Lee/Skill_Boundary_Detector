"""FSQ_new A/B/C action and terminator adapters for closed-loop oracle eval."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor, nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.skill_expert.configuration_skill_expert import SkillExpertConfig
from lerobot.utils.constants import ACTION, OBS_STATE

_HERE = Path(__file__).resolve()
_FSQ_NEW_SRC = _HERE.parents[2] / "FSQ_new" / "src"
_STAGE1_EVAL_SRC = _HERE.parents[3] / "train_skillVLA" / "stage1_eval" / "src"
sys.path.insert(0, str(_FSQ_NEW_SRC))
sys.path.insert(0, str(_STAGE1_EVAL_SRC))

from FSQ_new import load_fsq_model  # noqa: E402

from oracle_data import GoalImageStore  # noqa: E402


def _load_stage1_eval_module():
    module_name = "_skillvla_stage1_eval_run"
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


class FSQNewExpertPolicy(nn.Module):
    """Full FSQ_new checkpoint exposed through the common SkillExpert eval API."""

    def __init__(
        self,
        fsq_path: str | Path,
        device: torch.device,
        *,
        mode: str,
        n_action_steps: int,
        dino_model_path: str,
        raw_dataset_dir: str | Path,
    ):
        super().__init__()
        mode = str(mode).lower()
        if mode not in {"a", "b", "c"}:
            raise ValueError(f"FSQ_new eval mode must be a|b|c, got {mode!r}.")
        self.mode = mode
        self.model, self.fsq_cfg = load_fsq_model(
            fsq_path, device=device, dino_model_path=dino_model_path
        )
        self.real_state_dim = int(self.fsq_cfg.state_dim)
        self.real_action_dim = int(self.fsq_cfg.action_dim)
        self.num_inference_steps = 10
        self.goal_store = GoalImageStore(raw_dataset_dir) if mode == "c" else None
        self.goal_index_provider: Callable[[int], Tensor] | None = None
        self._current_features: tuple[Tensor, Tensor] | None = None
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
        self._current_features = None

    def set_current_features(self, features: tuple[Tensor, Tensor]) -> None:
        self._current_features = features

    def _skill_code(self, batch: dict) -> Tensor:
        sequence = batch["skill_sequence"].long()
        index = batch["skill_index"].long().view(-1, 1).clamp(0, sequence.shape[1] - 1)
        return sequence.gather(1, index).squeeze(1).clamp(0, self.model.fsq.codebook_size - 1)

    def _contexts(self, batch: dict, batch_size: int, device: torch.device):
        image_context = goal_context = None
        if self.mode in {"b", "c"}:
            features = self._current_features
            if features is None:
                features = self.model.terminator.image_features(
                    batch[_IMAGE_KEY].to(device), batch[_WRIST_KEY].to(device)
                )
            image_context = self.model.image_context(*features)
        if self.mode == "c":
            if self.goal_store is None or self.goal_index_provider is None:
                raise RuntimeError("Oracle-C requires a goal image store and active goal indices.")
            goal_indices = self.goal_index_provider(batch_size)
            goal_third, goal_wrist = self.goal_store.get(goal_indices)
            goal_features = self.model.terminator.image_features(
                goal_third.to(device), goal_wrist.to(device)
            )
            goal_context = self.model.goal_context(*goal_features)
        return image_context, goal_context

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
        image_context, goal_context = self._contexts(batch, normalized.shape[0], device)

        if self.mode == "a":
            skill_scale, image_scale, goal_scale = self.fsq_cfg.a_skill_scale, 0.0, 0.0
        elif self.mode == "b":
            skill_scale = self.fsq_cfg.b_skill_scale
            image_scale, goal_scale = self.fsq_cfg.b_image_scale, 0.0
        else:
            skill_scale = self.fsq_cfg.c_skill_scale
            image_scale, goal_scale = self.fsq_cfg.c_image_scale, self.fsq_cfg.c_goal_scale

        actions = self.model.action_expert.sample_actions(
            state,
            z_norm,
            num_steps=self.num_inference_steps,
            skill_scale=float(skill_scale),
            image_context=image_context,
            goal_context=goal_context,
            image_scale=float(image_scale),
            goal_scale=float(goal_scale),
        )
        return actions[:, :, : self.real_action_dim]

    def forward(self, batch: dict, *args, **kwargs):  # pragma: no cover
        return self.predict_action_chunk(batch, **kwargs)


class FSQNewTerminator:
    """Terminator view sharing the policy's loaded FSQ_new model and DINO features."""

    use_wrist = True

    def __init__(self, policy: FSQNewExpertPolicy):
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
            raise ValueError("FSQ_new terminator requires the current wrist image.")
        device = next(self.policy.model.parameters()).device
        z_norm = self.policy.model.fsq.code_to_normalized(codes.to(device).long())
        raw_state = state.to(device=device, dtype=torch.float32)[:, : self.policy.real_state_dim]
        features = self.policy.model.terminator.image_features(
            image.to(device), wrist_image.to(device)
        )
        self.policy.set_current_features(features)
        progress, logits = self.policy.model.terminator(
            z_norm,
            raw_state,
            image.to(device),
            wrist_image.to(device),
            image_features=features,
        )
        return progress, torch.sigmoid(logits)


class FSQNewOraclePolicy(OracleSkillExpertPolicy):
    """Existing oracle cursor/terminator logic plus per-skill endpoint goal indices."""

    def __init__(self, policy: FSQNewExpertPolicy, terminator: FSQNewTerminator, **kwargs):
        super().__init__(policy, terminator, **kwargs)
        self._goal_indices: list[list[int]] = []
        policy.goal_index_provider = self._current_goal_indices

    def set_forced_skill_token_sequences(self, sequences) -> None:
        self._goal_indices = [
            [int(item.get("goal_index", -1)) if isinstance(item, dict) else -1 for item in sequence]
            for sequence in sequences
        ]
        super().set_forced_skill_token_sequences(sequences)

    def _current_goal_indices(self, batch_size: int) -> Tensor:
        indices = [
            self._goal_indices[batch][min(self._cursors[batch], len(self._goal_indices[batch]) - 1)]
            for batch in range(batch_size)
        ]
        if any(index < 0 for index in indices):
            raise RuntimeError("Oracle-C sequence is missing an endpoint goal frame index.")
        return torch.tensor(indices, dtype=torch.long)
