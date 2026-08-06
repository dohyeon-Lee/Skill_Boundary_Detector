from __future__ import annotations

import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "lerobot/examples/libero"))

import FSQ as fsq_module  # noqa: E402
from FSQ import (  # noqa: E402
    DtypeAlignedRMSNorm,
    FSQQueryTerminator,
    FSQStartComparisonQueryTerminator,
    FSQWristOnlyQueryTerminator,
    SplineFSQAEConfig,
)


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


def _terminator_frontend(
    terminator_cls=FSQQueryTerminator,
) -> FSQQueryTerminator:
    # Avoid loading an external DINO checkpoint: these tests exercise only the
    # preprocessing/shared-forward/projection fast path.
    module = terminator_cls.__new__(terminator_cls)
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


def test_wrist_only_frontend_encodes_only_wrist_tokens() -> None:
    module = _terminator_frontend(FSQWristOnlyQueryTerminator).eval()
    wrist = torch.linspace(0.0, 1.0, 2 * 3 * 4 * 4).reshape(2, 3, 4, 4)

    expected = module.image_proj(
        module._image_features(wrist).to(module.image_proj.weight.dtype)
    )
    assert module.dino.calls == 1

    module.dino.calls = 0
    actual = module._prepare_wrist_tokens(wrist)

    assert module.dino.calls == 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_start_comparison_frontend_builds_change_tokens_and_query_masks() -> None:
    module = _terminator_frontend(FSQStartComparisonQueryTerminator).eval()
    width = module.image_proj.out_features
    module.hidden_dim = width
    module.start_third_type = nn.Parameter(torch.zeros(1, 1, width))
    module.current_third_type = nn.Parameter(torch.zeros(1, 1, width))
    module.current_wrist_type = nn.Parameter(torch.zeros(1, 1, width))
    module.change_type = nn.Parameter(torch.zeros(1, 1, width))
    module.change_mlp = nn.Sequential(nn.Linear(width * 3, width))
    image = torch.linspace(0.0, 1.0, 2 * 3 * 4 * 4).reshape(2, 3, 4, 4)

    tokens, image_allow, query_allow = module._prepare_comparison_tokens(
        image,
        torch.flip(image, dims=(-1,)),
        torch.flip(image, dims=(-2,)),
    )

    # _CountingDino exposes three tokens after its register token is removed.
    assert module.dino.calls == 1
    assert tokens.shape == (2, 9, width)
    assert image_allow.shape == (9, 9)
    assert query_allow.shape == (2, 9)
    assert query_allow[0].tolist() == [True] * 3 + [False] * 3 + [True] * 3
    assert query_allow[1].all()
    # Current rows cannot read comparison tokens; comparison rows can read all.
    assert not image_allow[0, 3:6].any()
    assert image_allow[3:6].all()


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


def test_image_only_builder_uses_fsq_config_but_no_fsq_model_weights(
    monkeypatch,
) -> None:
    config = SplineFSQAEConfig(
        vision_backbone="dino",
        dino_model_path="pretrained-dino",
    )

    class _FreshImageTerminator(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.anchor = nn.Parameter(torch.zeros(()))

    monkeypatch.setattr(
        fsq_module.torch,
        "load",
        lambda *args, **kwargs: {"cfg": config},
    )
    monkeypatch.setattr(
        fsq_module,
        "FSQImageOnlyQueryTerminator",
        _FreshImageTerminator,
    )
    monkeypatch.setattr(
        fsq_module,
        "load_fsq_terminator",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("image-only builder must not load FSQ terminator weights")
        ),
    )

    terminator, loaded_config = fsq_module.build_fsq_image_only_terminator(
        "FSQ.pt"
    )

    assert loaded_config is config
    assert terminator.kwargs["dino_model_path"] == "pretrained-dino"
    assert terminator.training is False


def test_start_comparison_builder_warm_starts_shared_fsq_weights(
    monkeypatch,
) -> None:
    config = SplineFSQAEConfig(
        vision_backbone="dino",
        dino_model_path="pretrained-dino",
    )

    class _ComparisonTerminator(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.shared = nn.Parameter(torch.zeros(()))
            self.start_third_type = nn.Parameter(torch.zeros(1))
            self.current_third_type = nn.Parameter(torch.zeros(1))
            self.current_wrist_type = nn.Parameter(torch.zeros(1))
            self.change_type = nn.Parameter(torch.zeros(1))
            self.change_mlp = nn.Linear(1, 1)

    monkeypatch.setattr(
        fsq_module.torch,
        "load",
        lambda *args, **kwargs: {
            "cfg": config,
            "model_state": {"terminator.shared": torch.tensor(3.0)},
        },
    )
    monkeypatch.setattr(
        fsq_module,
        "FSQStartComparisonQueryTerminator",
        _ComparisonTerminator,
    )

    terminator, loaded_config = fsq_module.build_fsq_start_comparison_terminator(
        "FSQ.pt"
    )

    assert loaded_config is config
    assert terminator.shared.item() == 3.0
    assert terminator.kwargs["dino_model_path"] == "pretrained-dino"
    assert terminator.training is False


def test_wrist_only_builder_uses_fsq_config_but_no_fsq_model_weights(
    monkeypatch,
) -> None:
    config = SplineFSQAEConfig(
        vision_backbone="dino",
        dino_model_path="pretrained-dino",
    )

    class _FreshWristTerminator(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.anchor = nn.Parameter(torch.zeros(()))

    monkeypatch.setattr(
        fsq_module.torch,
        "load",
        lambda *args, **kwargs: {"cfg": config},
    )
    monkeypatch.setattr(
        fsq_module,
        "FSQWristOnlyQueryTerminator",
        _FreshWristTerminator,
    )
    monkeypatch.setattr(
        fsq_module,
        "load_fsq_terminator",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("wrist-only builder must not load FSQ terminator weights")
        ),
    )

    terminator, loaded_config = fsq_module.build_fsq_wrist_only_terminator(
        "FSQ.pt"
    )

    assert loaded_config is config
    assert terminator.kwargs["dino_model_path"] == "pretrained-dino"
    assert terminator.training is False
