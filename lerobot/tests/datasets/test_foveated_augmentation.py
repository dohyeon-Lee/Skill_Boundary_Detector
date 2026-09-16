from types import SimpleNamespace

import torch

import lerobot.policies.skillVLA.foveated_augmentation as foveation
from lerobot.policies.skillVLA.foveated_augmentation import (
    FoveatedVisionAugmentationConfig,
    augment_camera_pair,
)


def _checkerboard(size: int = 32) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    pattern = ((xx + yy) % 2).float()
    return torch.stack((pattern, 1.0 - pattern, pattern), dim=0)


def test_color_and_input_blur_share_one_draw_across_top_and_wrist(
    monkeypatch,
) -> None:
    draws = iter((1.15, 0.9, 1.2, 0.1, 1.5))
    calls: list[tuple[float, float]] = []

    def fixed_sample(bounds: tuple[float, float]) -> float:
        calls.append(bounds)
        return next(draws)

    monkeypatch.setattr(foveation, "_sample_uniform", fixed_sample)
    image = _checkerboard()
    config = FoveatedVisionAugmentationConfig(
        enabled=True,
        randomization_enabled=True,
        peripheral_blur_radius=0.0,
        color_enabled=True,
        input_blur_enabled=True,
    )

    top, wrist = augment_camera_pair(
        image.clone(), image.clone(), torch.tensor([0.0, 0.0]), config
    )

    assert len(calls) == 5
    torch.testing.assert_close(top, wrist)


def test_crop_jitter_and_foveation_do_not_modify_wrist(monkeypatch) -> None:
    monkeypatch.setattr(foveation, "_sample_integer", lambda bounds: bounds[1])
    top_input = _checkerboard()
    wrist_input = _checkerboard()
    config = FoveatedVisionAugmentationConfig(
        enabled=True,
        randomization_enabled=True,
        sharp_size=8,
        feather=0,
        peripheral_blur_radius=4.0,
        crop_enabled=True,
        crop_offset_px=(5, 5),
    )

    top, wrist = augment_camera_pair(
        top_input, wrist_input, torch.tensor([0.0, 0.0]), config
    )

    assert not torch.equal(top, top_input)
    torch.testing.assert_close(wrist, wrist_input)


def test_disabled_foveation_is_an_exact_noop() -> None:
    top_input = torch.rand(3, 16, 16)
    wrist_input = torch.rand(3, 16, 16)

    top, wrist = augment_camera_pair(
        top_input,
        wrist_input,
        torch.tensor([0.0, 0.0]),
        FoveatedVisionAugmentationConfig(enabled=False),
    )

    assert top is top_input
    assert wrist is wrist_input


def test_partial_fov_can_black_out_periphery_without_changing_wrist() -> None:
    top_input = torch.full((3, 32, 32), 0.5)
    wrist_input = _checkerboard()
    config = FoveatedVisionAugmentationConfig.from_mapping(
        {
            "enabled": True,
            "mode": "partial_fov",
            "sharp_size": 8,
            "feather": 20,
            "peripheral_mode": "black",
        }
    )

    top, wrist = augment_camera_pair(
        top_input, wrist_input, torch.tensor([0.0, 0.0]), config
    )

    assert torch.count_nonzero(top[:, 0, 0]) == 0
    assert torch.count_nonzero(top[:, 16, 21]) == 0
    torch.testing.assert_close(top[:, 16, 16], top_input[:, 16, 16], atol=1 / 255, rtol=0)
    torch.testing.assert_close(wrist, wrist_input)


def test_black_periphery_is_only_valid_for_partial_fov() -> None:
    import pytest

    with pytest.raises(ValueError, match="only supported for partial_fov"):
        FoveatedVisionAugmentationConfig.from_mapping(
            {"enabled": True, "mode": "crop", "peripheral_mode": "black"}
        )


def test_eval_policy_restore_disables_training_randomization() -> None:
    config = FoveatedVisionAugmentationConfig.from_policy(
        SimpleNamespace(
            foveated_vision_enabled=True,
            foveation_randomization_enabled=True,
            foveation_mode="crop",
            foveation_crop_size=24,
            foveation_output_size=16,
            foveation_inner_box_enabled=True,
            foveation_inner_box_mode="blur",
            foveation_inner_box_size=8,
        ),
        randomization_enabled=False,
    )

    assert config.enabled is True
    assert config.mode == "crop"
    assert config.crop_size == 24
    assert config.output_size == 16
    assert config.randomization_enabled is False


def test_eval_policy_restores_black_periphery() -> None:
    config = FoveatedVisionAugmentationConfig.from_policy(
        SimpleNamespace(
            foveated_vision_enabled=True,
            foveation_mode="partial_fov",
            foveation_peripheral_mode="black",
        ),
        randomization_enabled=False,
    )

    assert config.peripheral_mode == "black"
    assert config.randomization_enabled is False


def test_randomization_can_run_without_foveation_or_focus(monkeypatch) -> None:
    monkeypatch.setattr(foveation, "_sample_uniform", lambda bounds: bounds[1])
    top_input = torch.full((3, 32, 32), 0.4)
    wrist_input = top_input.clone()
    config = FoveatedVisionAugmentationConfig(
        enabled=False,
        randomization_enabled=True,
        color_enabled=True,
        brightness=(1.5, 1.5),
        contrast=(1.0, 1.0),
        saturation=(1.0, 1.0),
        hue=(0.0, 0.0),
        crop_enabled=True,
    )

    top, wrist = augment_camera_pair(top_input, wrist_input, None, config)

    assert not torch.equal(top, top_input)
    torch.testing.assert_close(top, wrist)


def test_crop_inner_blur_changes_top_shape_but_not_wrist() -> None:
    top_input = _checkerboard(32)
    wrist_input = _checkerboard(32)
    config = FoveatedVisionAugmentationConfig(
        enabled=True,
        mode="crop",
        crop_size=24,
        output_size=16,
        inner_box_enabled=True,
        inner_box_mode="blur",
        inner_box_size=8,
        feather=2,
        peripheral_blur_radius=3.0,
    )

    top, wrist = augment_camera_pair(
        top_input,
        wrist_input,
        torch.tensor([0.0, 0.0]),
        config,
    )

    assert top.shape == (3, 16, 16)
    assert not torch.equal(top[:, :8, :8], top_input[:, :8, :8])
    torch.testing.assert_close(wrist, wrist_input)


def test_outer_crop_and_inner_focus_jitter_are_independent(monkeypatch) -> None:
    draws = iter((5, 5, 1, 1))
    captured: dict[str, tuple[int, int]] = {}

    monkeypatch.setattr(foveation, "_sample_integer", lambda bounds: next(draws))

    def capture_crop(image, *, crop_center_xy, inner_center_xy, config):
        del config
        captured["crop"] = crop_center_xy
        captured["inner"] = inner_center_xy
        return image

    monkeypatch.setattr(foveation, "_crop_focus", capture_crop)
    image = _checkerboard(32)
    config = FoveatedVisionAugmentationConfig(
        enabled=True,
        mode="crop",
        crop_size=24,
        output_size=16,
        inner_box_enabled=True,
        randomization_enabled=True,
        crop_enabled=True,
        crop_offset_px=(5, 5),
        inner_box_offset_px=(1, 1),
    )

    augment_camera_pair(image, image.clone(), torch.tensor([0.0, 0.0]), config)

    assert captured == {"crop": (21, 21), "inner": (17, 17)}
