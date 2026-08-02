from __future__ import annotations

import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "lerobot/examples/libero"))

from FSQ import DtypeAlignedRMSNorm, FSQQueryTerminator  # noqa: E402


class _CountingDino(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = 0

    def forward(self, images: torch.Tensor) -> SimpleNamespace:
        self.calls += 1
        pooled = images.mean(dim=(-2, -1))
        token = torch.cat([pooled, pooled.mean(dim=-1, keepdim=True)], dim=-1)
        # CLS, one register token, then two patch tokens.
        hidden = torch.stack([token, token + 10.0, token + 20.0, token + 30.0], dim=1)
        return SimpleNamespace(last_hidden_state=hidden)


def _terminator_frontend() -> FSQQueryTerminator:
    # Avoid loading an external DINO checkpoint: these tests exercise only the
    # preprocessing/shared-forward/projection fast path.
    module = FSQQueryTerminator.__new__(FSQQueryTerminator)
    nn.Module.__init__(module)
    module.vision_backbone = "dino"
    module.freeze_vision_encoder = True
    module.dino = _CountingDino()
    module.siglip = None
    module.n_register = 1
    module.vision_image_size = 4
    module.register_buffer("_img_mean", torch.zeros(1, 3, 1, 1), persistent=False)
    module.register_buffer("_img_std", torch.ones(1, 3, 1, 1), persistent=False)
    module.image_proj = nn.Linear(4, 5, bias=False)
    return module


def test_top_and_wrist_share_one_dino_call_without_changing_token_order() -> None:
    module = _terminator_frontend().eval()
    third = torch.linspace(0.0, 1.0, 2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
    wrist = torch.flip(third, dims=(-1,))

    # Historical reference: one shared tower called separately for each camera.
    third_features = module._image_features(third)
    wrist_features = module._image_features(wrist)
    expected = torch.cat(
        [
            module.image_proj(third_features.to(module.image_proj.weight.dtype)),
            module.image_proj(wrist_features.to(module.image_proj.weight.dtype)),
        ],
        dim=1,
    )
    assert module.dino.calls == 2

    module.dino.calls = 0
    actual = module._prepare_image_tokens(third, wrist)

    assert module.dino.calls == 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_uint8_and_zero_one_float_image_contracts_match() -> None:
    module = _terminator_frontend().eval()
    uint8_image = torch.arange(48, dtype=torch.uint8).reshape(1, 3, 4, 4)

    integer_input = module._preprocess_image(uint8_image)
    float_input = module._preprocess_image(uint8_image.float() / 255.0)

    torch.testing.assert_close(integer_input, float_input, rtol=0, atol=0)


def test_dtype_aligned_rmsnorm_avoids_mixed_dtype_fallback_warning() -> None:
    norm = DtypeAlignedRMSNorm(8)
    values = torch.randn(2, 4, 8, dtype=torch.bfloat16)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output = norm(values)

    assert output.dtype == values.dtype
    assert not any("Mismatch dtype between input and weight" in str(item.message) for item in caught)
    assert norm.weight.dtype == torch.float32
    assert set(norm.state_dict()) == {"weight"}
