#!/usr/bin/env python3
"""PI0.5 policy adapter for native ABC Sim observations and absolute actions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from eval_config import EvalConfig


ABC_CAMERA_TO_PI05 = {
    "top": "observation.images.base_0_rgb",
    "left": "observation.images.left_wrist_0_rgb",
    "right": "observation.images.right_wrist_0_rgb",
}


def quantile_transform(
    value: Any, q01: Any, q99: Any, *, inverse: bool = False, eps: float = 1e-8
):
    """Match LeRobot's QUANTILES normalization without importing its dataset stack."""
    import torch

    value = torch.as_tensor(value)
    low = torch.as_tensor(q01, device=value.device, dtype=value.dtype)
    high = torch.as_tensor(q99, device=value.device, dtype=value.dtype)
    denominator = high - low
    denominator = torch.where(
        denominator == 0,
        torch.tensor(eps, device=value.device, dtype=value.dtype),
        denominator,
    )
    if inverse:
        return (value + 1.0) * denominator / 2.0 + low
    return 2.0 * (value - low) / denominator - 1.0


def format_pi05_prompts(prompts: list[str], normalized_states: np.ndarray) -> list[str]:
    """Reproduce Pi05PrepareStateTokenizerProcessorStep exactly."""
    if normalized_states.ndim != 2 or normalized_states.shape[0] != len(prompts):
        raise ValueError(
            f"states must have shape (B,D) matching {len(prompts)} prompts, "
            f"got {normalized_states.shape}"
        )
    bins = np.linspace(-1, 1, 256 + 1)[:-1]
    discrete = np.digitize(normalized_states, bins=bins) - 1
    result = []
    for prompt, state in zip(prompts, discrete, strict=True):
        cleaned = str(prompt).strip().replace("_", " ").replace("\n", " ")
        state_text = " ".join(map(str, state))
        result.append(f"Task: {cleaned}, State: {state_text};\nAction: ")
    return result


def _load_quantiles(dataset_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    path = dataset_dir / "meta" / "stats.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read ABC normalization statistics: {path}") from error
    result = {}
    for key in ("observation.state", "action"):
        stats = payload.get(key) or {}
        if "q01" not in stats or "q99" not in stats:
            raise ValueError(f"ABC statistics for {key} need q01 and q99: {path}")
        low = np.asarray(stats["q01"], dtype=np.float32)
        high = np.asarray(stats["q99"], dtype=np.float32)
        if low.shape != (14,) or high.shape != (14,):
            raise ValueError(f"ABC {key} quantiles must be 14D, got {low.shape}/{high.shape}")
        result[key] = {"q01": low, "q99": high}
    return result


class Pi05PolicyAdapter:
    """Translate ABC's batched CHW observations to a LeRobot PI0.5 action chunk."""

    def __init__(self, config: EvalConfig):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from transformers import AutoTokenizer

        if config.policy_checkpoint is None or config.policy_tokenizer_path is None:
            raise ValueError("PI0.5 evaluation needs a checkpoint and tokenizer path")
        if not torch.cuda.is_available() and config.policy_device.startswith("cuda"):
            raise RuntimeError("PI0.5 ABC evaluation requested CUDA, but torch cannot use the GPU")

        self.config = config
        self.torch = torch
        self.device = torch.device(config.policy_device)
        self.stats = _load_quantiles(config.dataset_dir)

        # The distributed base checkpoint records ``device: mps``. Permit the
        # config parser to rebase that metadata on this host, then immediately
        # enforce the explicitly configured CUDA device below. This flag is not
        # left enabled while the model is constructed or evaluated.
        fallback_before = os.environ.get("LEROBOT_ALLOW_CPU_FALLBACK")
        os.environ["LEROBOT_ALLOW_CPU_FALLBACK"] = "1"
        try:
            policy_config = PreTrainedConfig.from_pretrained(config.policy_checkpoint)
        finally:
            if fallback_before is None:
                os.environ.pop("LEROBOT_ALLOW_CPU_FALLBACK", None)
            else:
                os.environ["LEROBOT_ALLOW_CPU_FALLBACK"] = fallback_before
        if not isinstance(policy_config, PI05Config):
            raise TypeError(
                f"policy checkpoint is {type(policy_config).__name__}, expected PI05Config"
            )
        policy_config.device = str(self.device)
        policy_config.dtype = config.policy_dtype
        policy_config.chunk_size = config.policy_model_chunk_size
        policy_config.n_action_steps = config.action_chunk_size
        policy_config.num_inference_steps = config.policy_num_inference_steps
        policy_config.tokenizer_path = str(config.policy_tokenizer_path)
        policy_config.pretrained_path = config.policy_checkpoint
        policy_config.proprio_grounding = "none"

        expected_images = set(ABC_CAMERA_TO_PI05.values())
        actual_images = set(policy_config.image_features)
        if actual_images != expected_images:
            raise ValueError(
                "PI0.5 checkpoint camera contract does not match ABC: "
                f"expected {sorted(expected_images)}, got {sorted(actual_images)}. "
                "Use models/pi05_base or an ABC-trained checkpoint."
            )
        policy_config.input_features["observation.state"] = PolicyFeature(
            type=FeatureType.STATE, shape=(14,)
        )
        policy_config.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(14,))
        }

        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        self.policy = PI05Policy.from_pretrained(
            config.policy_checkpoint,
            config=policy_config,
            strict=True,
        )
        self.policy.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.policy_tokenizer_path,
            local_files_only=True,
        )
        self.reset()

    def reset(self) -> None:
        self.policy.reset()

    def describe(self) -> dict[str, Any]:
        return {
            "type": "pi05",
            "checkpoint": str(self.config.policy_checkpoint),
            "tokenizer_path": str(self.config.policy_tokenizer_path),
            "model_chunk_size": self.config.policy_model_chunk_size,
            "execution_chunk_size": self.config.action_chunk_size,
            "num_inference_steps": self.config.policy_num_inference_steps,
            "dtype": self.config.policy_dtype,
            "device": str(self.device),
            "action_space": "absolute_joint_position_14d",
        }

    def _image(self, value: Any):
        torch = self.torch
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=self.device, dtype=torch.float32)
        else:
            tensor = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4 or tensor.shape[1] != 3:
            raise ValueError(f"ABC image must be CHW or BCHW, got {tuple(tensor.shape)}")
        return tensor.div_(255.0) if float(tensor.max()) > 1.0 else tensor

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        torch = self.torch
        state = torch.as_tensor(
            observation["state"], device=self.device, dtype=torch.float32
        )
        single = state.ndim == 1
        if single:
            state = state.unsqueeze(0)
        if state.ndim != 2 or state.shape[1] != 14:
            raise ValueError(f"ABC state must have shape (14,) or (B,14), got {tuple(state.shape)}")

        prompt_value = observation.get("prompt", "")
        prompts = [str(prompt_value)] if isinstance(prompt_value, str) else list(prompt_value)
        if len(prompts) != state.shape[0]:
            raise ValueError(f"got {len(prompts)} prompts for batch size {state.shape[0]}")
        state_stats = self.stats["observation.state"]
        normalized_state = quantile_transform(
            state, state_stats["q01"], state_stats["q99"]
        )
        full_prompts = format_pi05_prompts(
            prompts, normalized_state.detach().cpu().numpy()
        )
        tokenized = self.tokenizer(
            full_prompts,
            max_length=self.policy.config.tokenizer_max_length,
            truncation=True,
            padding="max_length",
            padding_side="right",
            return_tensors="pt",
        )
        batch = {
            key: self._image(observation["images"][camera])
            for camera, key in ABC_CAMERA_TO_PI05.items()
        }
        batch["observation.state"] = normalized_state
        batch["observation.language.tokens"] = tokenized["input_ids"].to(self.device)
        batch["observation.language.attention_mask"] = tokenized["attention_mask"].to(
            self.device, dtype=torch.bool
        )

        with torch.inference_mode():
            normalized_actions = self.policy.predict_action_chunk(batch)
        action_stats = self.stats["action"]
        actions = quantile_transform(
            normalized_actions,
            action_stats["q01"],
            action_stats["q99"],
            inverse=True,
        )[:, : self.config.action_chunk_size, :14]
        if not torch.isfinite(actions).all():
            raise RuntimeError("PI0.5 produced non-finite ABC actions")
        result = actions.detach().to(dtype=torch.float32, device="cpu").numpy()
        return result[0] if single else result
