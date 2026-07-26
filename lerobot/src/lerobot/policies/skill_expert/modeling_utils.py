"""Small model-building helpers shared by Stage 0 and the Stage-1 VSA policy."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from transformers.models.auto import CONFIG_MAPPING

from lerobot.policies.pi05.modeling_pi05 import get_gemma_config
from lerobot.policies.pi_gemma import PaliGemmaModelWithPiGemma, PiGemmaForCausalLM

log = logging.getLogger(__name__)


def build_siglip_vision_tower(image_size: int):
    """Build the standalone SigLIP tower used by the preserved Stage-0 implementation."""
    from transformers import SiglipVisionModel  # noqa: PLC0415

    vlm_config = CONFIG_MAPPING["paligemma"]()
    vision_config = vlm_config.vision_config
    vision_config.image_size = image_size
    vision_config.intermediate_size = 4304
    vision_config.projection_dim = 2048
    vision_config.projector_hidden_act = "gelu_fast"
    return SiglipVisionModel(vision_config)


def build_gemma(variant: str, *, use_adarms: bool) -> PiGemmaForCausalLM:
    """Build a projection-fed Gemma transformer without token embeddings or an LM head."""
    config = get_gemma_config(variant)
    hf_config = CONFIG_MAPPING["gemma"](
        head_dim=config.head_dim,
        hidden_size=config.width,
        intermediate_size=config.mlp_dim,
        num_attention_heads=config.num_heads,
        num_hidden_layers=config.depth,
        num_key_value_heads=config.num_kv_heads,
        vocab_size=257152,
        hidden_activation="gelu_pytorch_tanh",
        dtype="float32",
        use_adarms=use_adarms,
        adarms_cond_dim=config.width if use_adarms else None,
    )
    model = PiGemmaForCausalLM(config=hf_config)
    model.model.embed_tokens = None
    model.lm_head = None
    model.model.config._attn_implementation = "eager"  # noqa: SLF001
    return model


def build_paligemma_model(variant: str, *, image_size: int) -> PaliGemmaModelWithPiGemma:
    """Build only pi0.5's PaliGemma model trunk (no unused LM head or action expert)."""
    config = get_gemma_config(variant)
    hf_config = CONFIG_MAPPING["paligemma"]()
    hf_config._vocab_size = 257152  # noqa: SLF001
    hf_config.image_token_index = 257152
    hf_config.text_config.hidden_size = config.width
    hf_config.text_config.intermediate_size = config.mlp_dim
    hf_config.text_config.num_attention_heads = config.num_heads
    hf_config.text_config.head_dim = config.head_dim
    hf_config.text_config.num_hidden_layers = config.depth
    hf_config.text_config.num_key_value_heads = config.num_kv_heads
    hf_config.text_config.hidden_activation = "gelu_pytorch_tanh"
    hf_config.text_config.dtype = "float32"
    hf_config.text_config.vocab_size = 257152
    hf_config.text_config.use_adarms = False
    hf_config.text_config.adarms_cond_dim = None
    hf_config.vision_config.image_size = image_size
    hf_config.vision_config.intermediate_size = 4304
    hf_config.vision_config.projection_dim = 2048
    hf_config.vision_config.projector_hidden_act = "gelu_fast"
    hf_config.vision_config.dtype = "float32"
    return PaliGemmaModelWithPiGemma(hf_config)


def load_raw_state_dict(path: str | Path, kwargs: dict) -> dict | None:
    """Load ``model.safetensors`` from a local file/directory or a Hugging Face repo."""
    from safetensors.torch import load_file  # noqa: PLC0415

    local_path = Path(path)
    if local_path.is_file():
        return load_file(str(local_path))
    if (local_path / "model.safetensors").is_file():
        return load_file(str(local_path / "model.safetensors"))
    try:
        from transformers.utils import cached_file  # noqa: PLC0415

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


def build_fsq_terminator(path: str | Path, *, dino_model_path: str | None = None):
    """Load only the v2 FSQ terminator, without allocating its action expert/encoder."""
    examples_root = Path(__file__).resolve().parents[4] / "examples" / "libero"
    if str(examples_root) not in sys.path:
        sys.path.insert(0, str(examples_root))
    from FSQ import load_fsq_terminator  # noqa: PLC0415

    terminator, _ = load_fsq_terminator(
        str(path), device="cpu", dino_model_path=dino_model_path
    )
    return terminator
