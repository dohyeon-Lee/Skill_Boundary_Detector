from types import SimpleNamespace

import torch

from lerobot.policies.skill_expert.predictor_image_crop import PredictorTopCrop
from lerobot.policies.skill_expert.vsa_top_crop import map_uv, prepare_top_crop


def test_one_geometry_for_top_pixels_conditions_aux_and_past_termination():
    cfg = SimpleNamespace(vsa_top_crop_enabled=True, vsa_top_crop_zoom=1.1, dino_image_size=100)
    image = torch.zeros(2, 3, 100, 100)
    crop = PredictorTopCrop(100, 100, 110, 110, torch.tensor([0, 10]), torch.tensor([10, 0]), 100)
    uv = torch.zeros(2, 2)
    xyz, wrist = torch.randn(2, 3), torch.rand_like(image)
    batch = {"observation.images.image": image, "observation.images.wrist_image": wrist,
             "observation.state": xyz, "skill_end_xyz": xyz, "skill_chunk_end_state": xyz,
             "skill_focus_uv": uv, "skill_focus_valid": torch.ones(2, dtype=torch.bool),
             "skill_chunk_end_focus_uv": uv.clone(),
             "vlm_termination_past_uv": uv[:, None].expand(-1, 3, -1).clone(),
             "skill_start_focus_uv": uv.clone()}
    result, actual = prepare_top_crop(batch, cfg, training=True, crop=crop)
    # The centered source point moves by +/-5 native pixels with these crops.
    expected = torch.tensor([[-10 / 99, 10 / 99], [10 / 99, -10 / 99]])
    torch.testing.assert_close(result["skill_focus_uv"], expected)
    torch.testing.assert_close(result["skill_chunk_end_focus_uv"], expected)
    torch.testing.assert_close(result["vlm_termination_past_uv"], expected[:, None].expand(-1, 3, -1))
    torch.testing.assert_close(map_uv(actual, result["skill_focus_uv"], inverse=True), uv, atol=2e-7, rtol=0)
    assert result["skill_end_xyz"] is xyz and result["skill_chunk_end_state"] is xyz
    assert result["observation.images.wrist_image"] is wrist
    torch.testing.assert_close(batch["skill_focus_uv"], uv)  # caller remains in original camera frame
    torch.testing.assert_close(result["skill_start_focus_uv"], uv)  # uncropped start-image target
    assert prepare_top_crop(result, cfg, training=True)[0] is result


def test_center_crop_is_repeatable_and_invalid_uv_is_not_made_valid():
    cfg = SimpleNamespace(vsa_top_crop_enabled=True, vsa_top_crop_zoom=1.1, dino_image_size=100)
    batch = {"observation.images.image": torch.rand(2, 3, 100, 100),
             "skill_focus_uv": torch.tensor([[0., 0.], [float('nan'), 0.]]),
             "skill_focus_valid": torch.ones(2, dtype=torch.bool)}
    a, crop = prepare_top_crop(batch, cfg, training=False)
    b, _ = prepare_top_crop(batch, cfg, training=False)
    assert crop.x.tolist() == [5, 5] and crop.y.tolist() == [5, 5]
    torch.testing.assert_close(a["observation.images.image"], b["observation.images.image"])
    assert a["skill_focus_valid"].tolist() == [True, False]


def test_old_checkpoint_disables_crop_without_consuming_rng():
    batch = {"observation.images.image": torch.zeros(1, 3, 100, 100)}
    before = torch.get_rng_state()
    result, crop = prepare_top_crop(batch, SimpleNamespace(), training=True)
    assert result is batch and crop is None
    assert torch.equal(before, torch.get_rng_state())
