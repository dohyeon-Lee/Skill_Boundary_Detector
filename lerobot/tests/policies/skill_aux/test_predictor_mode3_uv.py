"""Absolute cropped UV supervision, exact geometric inverse, and mode3 compatibility."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.skill_predictor_modes import PREDICTOR_MODES
from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch
from lerobot.policies.skill_aux.configuration_skill_aux import SkillAuxConfig
from lerobot.policies.skill_aux.modeling_skill_aux import SkillAuxPolicy
from lerobot.policies.skill_expert import modeling_skill_predictor as model
from lerobot.policies.skill_expert.configuration_skill_expert import SkillExpertConfig
from lerobot.policies.skill_expert.predictor_image_crop import PredictorTopCrop


MODES = PREDICTOR_MODES[4:]


class TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.config = SimpleNamespace(hidden_size=16)
        self.language_model.embed_tokens = nn.Embedding(32, 16)
        self.images = []

    def get_image_features(self, image):
        self.images.append(image.detach().clone())
        values = image.mean((1, 2, 3))[:, None, None].expand(-1, 2, 16)
        return SimpleNamespace(pooler_output=values)


def _config(mode):
    return SkillAuxConfig(
        device="cpu", dtype="float32", train_skill_predictor=True, train_terminator=False,
        skill_predictor_mode=mode, skill_predictor_end_state_mode="off",
        skill_predictor_focus_uv_enabled=True, skill_predictor_focus_uv_loss_weight=1.,
        skill_predictor_freeze_vlm=False, skill_predictor_lora=False,
        skill_predictor_reader_heads=2, skill_predictor_reader_depth=1,
    )


def _predictor(monkeypatch, mode):
    monkeypatch.setattr(model, "build_paligemma_model", lambda *a, **kw: TinyVLM())
    p = model.FrozenVLMSkillPredictor(_config(mode))
    for parameter in p.reader_head_parameters():
        parameter.requires_grad_(True)
    return p


@pytest.mark.parametrize("mode", MODES)
def test_mode3_uv_contract_and_head_roundtrip(monkeypatch, mode, tmp_path):
    cfg = _config(mode)
    cfg.save_pretrained(tmp_path)
    loaded = PreTrainedConfig.from_pretrained(tmp_path)
    assert loaded.skill_predictor_mode == mode
    assert loaded.skill_predictor_focus_uv_enabled
    assert loaded.skill_predictor_end_state_mode == "off"
    # UV-based VSA/external Predictor configs also need to accept the contract.
    SkillExpertConfig(
        device="cpu", skill_predictor_mode=mode, skill_predictor_end_state_mode="off",
        skill_predictor_focus_uv_enabled=True,
    )
    p = _predictor(monkeypatch, mode)
    assert p.uses_top_crop and not p.relative_xyz
    assert p.focus_uv_head is not None and p.end_state_head is None
    if p.uses_previous_skill:
        assert p.previous_skill_projection is not None and p.previous_skill_bos is not None
    else:
        assert p.previous_skill_projection is None and p.previous_skill_bos is None


@pytest.mark.parametrize("size", [(256, 256), (255, 257), (180, 320)])
@pytest.mark.parametrize("training", [False, True])
def test_exact_uv_roundtrip_includes_integer_zoom_crop_resize_and_padding(size, training):
    h, w = size
    image = torch.zeros(3, 3, h, w)
    crop = PredictorTopCrop.sample(image, zoom=1.1, training=training, model_size=224)
    original = torch.tensor([[0., 0.], [.2, -.2], [-.3, .1]])
    transformed, valid = crop.to_model_uv(original)
    assert valid.all()
    torch.testing.assert_close(crop.to_original_uv(transformed), original, rtol=0, atol=5e-7)
    assert crop.expanded_width == round(w * 1.1)
    assert crop.expanded_height == round(h * 1.1)


@pytest.mark.parametrize("size", [(256, 256), (255, 257), (180, 320)])
def test_coordinate_inverse_matches_pixels_in_the_actual_resized_image(size):
    h, w = size
    # R/G encode actual original-frame x/y. Bilinear image interpolation gives
    # an independent geometric oracle for the UV inverse, including half pixels.
    xx = torch.arange(w).float()[None].expand(h, -1) / (w - 1)
    yy = torch.arange(h).float()[:, None].expand(-1, w) / (h - 1)
    image = torch.stack((xx, yy, torch.zeros_like(xx)))[None]
    crop = PredictorTopCrop.sample(image, zoom=1.1, training=False, model_size=224)
    rendered = resize_with_pad_torch(crop.apply(image), 224, 224)
    positions = torch.tensor([[111., 108.], [90., 105.], [140., 120.]])
    for xy in positions:
        uv = xy[None] * 2 / 223 - 1
        expected = rendered[0, :2, int(xy[1]), int(xy[0])][None] * 2 - 1
        torch.testing.assert_close(crop.to_original_uv(uv), expected, rtol=0, atol=8e-7)


@pytest.mark.parametrize("mode", MODES)
def test_one_draw_is_shared_by_pixels_and_absolute_uv_loss(monkeypatch, mode):
    p = _predictor(monkeypatch, mode)
    b = 2
    crop = PredictorTopCrop(256, 256, 282, 282, torch.tensor([0, 26]), torch.tensor([26, 0]), 224)
    monkeypatch.setattr(p, "top_crop", lambda *a, **kw: crop)
    uv = torch.tensor([[.2, -.3], [-.25, .1]])
    target, valid = crop.to_model_uv(uv)
    assert valid.all()
    predicted = nn.Parameter(target.clone())
    memory, mask = torch.randn(b, 4, 16), torch.zeros(b, 4, dtype=torch.bool)
    received = []

    def read(images, tokens, token_mask, *, top_crop):
        assert top_crop is crop
        received.append(len(images))
        p._embed_prefix(images, tokens, token_mask, top_crop=top_crop)
        return memory, mask

    monkeypatch.setattr(p, "reader_memory", read)
    monkeypatch.setattr(p, "predict_focus_uv_from_memory", lambda *a: predicted)
    images = [torch.rand(b, 3, 256, 256)]
    if p.uses_wrist:
        images.append(torch.rand_like(images[0]))
    objective, metrics = p.loss_with_focus_uv(
        images, torch.ones(b, 2, dtype=torch.long), torch.ones(b, 2, dtype=torch.bool),
        torch.tensor([1, 2]), uv, torch.ones(b, dtype=torch.bool),
        previous_skill_code=torch.tensor([-1, 0]),
    )
    assert received == [len(images)]
    assert metrics["focus_uv_loss"] == pytest.approx(0, abs=1e-12)
    assert metrics["focus_uv_crop_masked_fraction"] == 0
    assert metrics["focus_uv_valid_fraction"] == 1
    torch.testing.assert_close(p.vlm.images[0], p._preprocess_image(crop.apply(images[0])))
    if p.uses_wrist:
        torch.testing.assert_close(p.vlm.images[1], p._preprocess_image(images[1]))
    objective.backward()
    if p.uses_previous_skill:
        assert p.previous_skill_bos.grad is not None
        assert p.previous_skill_projection[0].weight.grad is not None


@pytest.mark.parametrize("mode", MODES)
def test_out_of_crop_targets_clamp_to_border_and_preserve_uv_learning(monkeypatch, mode):
    p = _predictor(monkeypatch, mode)
    crop = PredictorTopCrop(256, 256, 282, 282, torch.tensor([13, 13]), torch.tensor([13, 13]), 224)
    monkeypatch.setattr(p, "top_crop", lambda *a, **kw: crop)
    memory, mask = torch.randn(2, 4, 16), torch.zeros(2, 4, dtype=torch.bool)
    monkeypatch.setattr(p, "reader_memory", lambda *a, **kw: (memory, mask))
    predicted = nn.Parameter(torch.zeros(2, 2))
    monkeypatch.setattr(p, "predict_focus_uv_from_memory", lambda *a: predicted)
    # The first point is outside the new crop and must clamp to its nearest
    # border.  Only the corrupt/non-finite second point remains invalid.
    objective, metrics = p.loss_with_focus_uv(
        [torch.rand(2, 3, 256, 256)], torch.ones(2, 2, dtype=torch.long),
        torch.ones(2, 2, dtype=torch.bool), torch.tensor([0, 1]),
        torch.tensor([[-1., -1.], [float("nan"), 0.]]), torch.tensor([True, False]),
        previous_skill_code=torch.tensor([-1, 0]),
    )
    assert torch.isfinite(objective)
    assert metrics["focus_uv_loss"] > 0
    assert metrics["focus_uv_valid_fraction"] == .5
    assert metrics["focus_uv_crop_clipped_fraction"] == .5
    assert metrics["focus_uv_crop_masked_fraction"] == 0
    assert metrics["skill_loss"] > 0
    objective.backward()
    assert predicted.grad is not None and predicted.grad[0].count_nonzero() > 0
    assert predicted.grad[1].count_nonzero() == 0
    assert any(param.grad is not None and param.grad.count_nonzero() for param in p.head.parameters())


@pytest.mark.parametrize("mode", MODES)
def test_finite_originally_clipped_uv_still_receives_loss(monkeypatch, mode):
    p = _predictor(monkeypatch, mode)
    crop = PredictorTopCrop(256, 256, 282, 282, torch.tensor([13]), torch.tensor([13]), 224)
    monkeypatch.setattr(p, "top_crop", lambda *a, **kw: crop)
    memory, mask = torch.randn(1, 4, 16), torch.zeros(1, 4, dtype=torch.bool)
    monkeypatch.setattr(p, "reader_memory", lambda *a, **kw: (memory, mask))
    predicted = nn.Parameter(torch.zeros(1, 2))
    monkeypatch.setattr(p, "predict_focus_uv_from_memory", lambda *a: predicted)
    objective, metrics = p.loss_with_focus_uv(
        [torch.rand(1, 3, 256, 256)], torch.ones(1, 2, dtype=torch.long),
        torch.ones(1, 2, dtype=torch.bool), torch.tensor([0]),
        torch.tensor([[1., -.4]]), torch.tensor([False]),
        previous_skill_code=torch.tensor([-1]),
    )
    assert torch.isfinite(objective)
    assert metrics["focus_uv_valid_fraction"] == 1
    assert metrics["focus_uv_original_valid_fraction"] == 0
    assert metrics["focus_uv_clipped_supervision_fraction"] == 1
    assert metrics["focus_uv_loss"] > 0
    objective.backward()
    assert predicted.grad is not None and predicted.grad.count_nonzero() > 0


@pytest.mark.parametrize("mode", MODES)
def test_public_uv_inference_is_centered_deterministic_and_restores_original_frame(monkeypatch, mode):
    p = _predictor(monkeypatch, mode).eval()
    image = torch.rand(2, 3, 256, 256)
    original = torch.tensor([[.15, -.23], [-.12, .25]])
    memory, mask = torch.randn(2, 4, 16), torch.zeros(2, 4, dtype=torch.bool)
    crops = []

    def read(images, *args, top_crop):
        crops.append(top_crop)
        assert top_crop.x.tolist() == [13, 13]
        assert top_crop.y.tolist() == [13, 13]
        return memory, mask

    monkeypatch.setattr(p, "reader_memory", read)
    monkeypatch.setattr(p, "predict_focus_uv_from_memory", lambda *a: crops[-1].to_model_uv(original)[0])
    monkeypatch.setattr(torch, "randint", lambda *a, **kw: pytest.fail("inference sampled a random crop"))
    tokens, mask_tokens = torch.ones(2, 2, dtype=torch.long), torch.ones(2, 2, dtype=torch.bool)
    for _ in range(2):
        _, result = p.predict_focus_uv(
            [image], tokens, mask_tokens, previous_skill_code=torch.tensor([-1, 0]),
        )
        torch.testing.assert_close(result, original, rtol=0, atol=3e-7)


@pytest.mark.parametrize("mode", MODES)
def test_auxiliary_uv_training_needs_no_xyz_origin(monkeypatch, mode):
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    p = _predictor(monkeypatch, mode)
    memory, mask = torch.randn(2, 4, 16), torch.zeros(2, 4, dtype=torch.bool)
    monkeypatch.setattr(p, "reader_memory", lambda *a, **kw: (memory, mask))
    policy = object.__new__(SkillAuxPolicy)
    nn.Module.__init__(policy)
    policy.config, policy.model = p.config, nn.Module()
    policy.model.skill_predictor = p
    batch = {
        "skill_start_image": torch.rand(2, 3, 256, 256),
        "skill_code": torch.tensor([1, 2]), "skill_previous_code": torch.tensor([-1, 0]),
        "skill_focus_uv": torch.tensor([[.1, .2], [-.1, -.2]]),
        "skill_focus_valid": torch.ones(2, dtype=torch.bool),
        OBS_LANGUAGE_TOKENS: torch.ones(2, 2, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 2, dtype=torch.bool),
    }
    if p.uses_wrist:
        batch["skill_start_wrist_image"] = torch.rand_like(batch["skill_start_image"])
    if not p.uses_previous_skill:
        batch.pop("skill_previous_code")
    objective, metrics = policy._skill_predictor_objective(batch)
    objective.backward()
    assert metrics["skill_predictor/relative_xyz"] == 0
    assert metrics["skill_predictor/top_zoom"] == 1.1
    assert metrics["skill_predictor/uv_crop_coordinates"] == 1
    assert "skill_predictor/focus_uv_clipped_supervision_fraction" in metrics
    assert "skill_predictor/focus_uv_crop_clipped_fraction" in metrics


@pytest.mark.parametrize("mode", MODES)
def test_uv_based_vsa_receives_restored_uv_once_and_keeps_preskill_history(monkeypatch, mode):
    from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    p = _predictor(monkeypatch, mode).eval()
    memory, mask = torch.randn(1, 3, 16), torch.zeros(1, 3, dtype=torch.bool)
    crop = p.top_crop(torch.zeros(1, 3, 256, 256), training=False)
    original = torch.tensor([[.3, -.25]])
    monkeypatch.setattr(p, "reader_memory", lambda *a, **kw: (memory, mask))
    monkeypatch.setattr(p.head, "decode", lambda *a: torch.tensor([2]))
    monkeypatch.setattr(p, "predict_focus_uv_from_memory", lambda *a: crop.to_model_uv(original)[0])
    policy = object.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.model, policy.config = nn.Module(), SimpleNamespace(architecture_label="arch8")
    policy.model.skill_predictor = p
    batch = {
        "observation.images.image": torch.zeros(1, 3, 256, 256), "skill_previous_code": torch.tensor([-1]),
        OBS_LANGUAGE_TOKENS: torch.ones(1, 2, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 2, dtype=torch.bool),
    }
    if p.uses_wrist:
        batch["observation.images.wrist_image"] = torch.ones(1, 3, 256, 256)
    codes, uv = policy.predict_skill_code_and_focus_uv(batch)
    assert codes.tolist() == [2]
    torch.testing.assert_close(uv, original, rtol=0, atol=3e-7)
    batch.pop("skill_previous_code")
    if p.uses_previous_skill:
        with pytest.raises(ValueError, match="skill_previous_code"):
            policy.predict_skill_code_and_focus_uv(batch)
    else:
        codes_without_history, uv_without_history = policy.predict_skill_code_and_focus_uv(batch)
        torch.testing.assert_close(codes_without_history, codes)
        torch.testing.assert_close(uv_without_history, uv)
