from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from lerobot.configs.policies import PreTrainedConfig

from lerobot.policies.skill_aux.configuration_skill_aux import SkillAuxConfig
from lerobot.policies.skill_expert import modeling_skill_predictor as model
from lerobot.configs.skill_predictor_modes import (
    PREDICTOR_MODES, PREDICTOR_START_XYZ, normalize_predictor_mode,
    predictor_uses_previous_skill, predictor_uses_proprio, predictor_uses_wrist,
)
from lerobot.policies.skillVLA.processor_skillVLA import (
    SKILL_START_STATE, SkillPredictorPreserveStartXYZProcessorStep,
    SkillVLAPrepareStateTokenizerProcessorStep,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import OBS_STATE


class TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.config = SimpleNamespace(hidden_size=16)
        self.language_model.embed_tokens = nn.Embedding(32, 16)
        self.images = []

    def get_image_features(self, image):
        self.images.append(image.detach().clone())
        value = image.mean((1, 2, 3))[:, None, None].expand(-1, 2, 16)
        return SimpleNamespace(pooler_output=value)


def config(mode="mode3_preskill", **kw):
    return SkillAuxConfig(
        device="cpu", dtype="float32", train_skill_predictor=True,
        skill_predictor_mode=mode, skill_predictor_end_state_mode="xyz",
        skill_predictor_freeze_vlm=False, skill_predictor_lora=False,
        skill_predictor_reader_heads=2, skill_predictor_reader_depth=1,
        skill_predictor_image_size=8, **kw,
    )


def predictor(monkeypatch, mode="mode3_preskill"):
    monkeypatch.setattr(model, "build_paligemma_model", lambda *a, **k: TinyVLM())
    p = model.FrozenVLMSkillPredictor(config(mode))
    for parameter in p.reader_head_parameters():
        parameter.requires_grad_(True)
    return p


@pytest.mark.parametrize("mode", PREDICTOR_MODES)
def test_mode_contract_roundtrip_and_history(mode, tmp_path):
    cfg = config(mode)
    cfg.save_pretrained(tmp_path)
    loaded = PreTrainedConfig.from_pretrained(tmp_path)
    assert loaded.skill_predictor_mode == mode
    assert predictor_uses_previous_skill(mode) == ("preskill" in mode)
    assert predictor_uses_wrist(mode) == ("wristX" not in mode)
    assert predictor_uses_proprio(mode) == ("proprioX" not in mode)
    assert normalize_predictor_mode(mode.lower()) == mode


@pytest.mark.parametrize("mode", PREDICTOR_MODES[4:])
def test_mode3_keeps_skill_conditioning_and_optional_history(monkeypatch, mode):
    p = predictor(monkeypatch, mode).eval()
    assert p.end_state_skill_projection is not None
    assert p.relative_xyz
    assert p.start_proprio_skill_projection is None
    memory = torch.randn(2, 4, 16)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    if p.uses_previous_skill:
        assert p.previous_skill_projection is not None and p.previous_skill_bos is not None
        with pytest.raises(ValueError, match="Preskill"):
            p.skill_reader_hidden(memory, mask)
    else:
        assert p.previous_skill_projection is None and p.previous_skill_bos is None
        assert not any("previous_skill" in key for key in p.state_dict())
        expected = p.skill_reader_hidden(memory, mask)
        torch.testing.assert_close(
            p.skill_reader_hidden(memory, mask, previous_skill_code=torch.tensor([7, 8])), expected,
        )
    out = p.skill_reader_hidden(memory, mask, previous_skill_code=torch.tensor([-1, 3]))
    out.sum().backward()
    if p.uses_previous_skill:
        assert p.previous_skill_bos.grad is not None
        assert p.previous_skill_projection[0].weight.grad is not None


def test_mode3_rejects_non_xyz_and_proprio_adapter():
    with pytest.raises(ValueError, match="requires spatial_target xyz"):
        SkillAuxConfig(device="cpu", train_skill_predictor=True, skill_predictor_mode="mode3_preskill")
    with pytest.raises(ValueError, match="proprioX"):
        config("mode3_preskill_wristX_proprioX", skill_predictor_start_proprio=True)


def test_relative_goal_reconstruction_does_not_round_measured_origin_to_bfloat16(monkeypatch):
    p = predictor(monkeypatch)
    delta = torch.tensor([[.01, -.02, .03]], dtype=torch.bfloat16)
    origin = torch.tensor([[.123456, .234567, .345678]], dtype=torch.float32)
    result = p.absolute_end_state(delta, origin)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, delta.float() + origin, rtol=0, atol=0)


@pytest.mark.parametrize("mode", PREDICTOR_MODES[4:])
def test_relative_target_uses_jittered_origin_and_is_translation_invariant(monkeypatch, mode):
    p = predictor(monkeypatch, mode)
    memory, ignore = torch.randn(2, 3, 16), torch.zeros(2, 3, dtype=torch.bool)
    monkeypatch.setattr(p, "reader_memory", lambda *a: (memory, ignore))
    monkeypatch.setattr(p.head, "loss", lambda h, c: h.sum() * 0)
    monkeypatch.setattr(p.head, "decode", lambda h: torch.tensor([0, 1]))
    delta = nn.Parameter(torch.tensor([[.2, -.1, .3], [.1, .4, -.2]]))
    captured = []

    def spatial(mem, mask, code):
        captured.append(code.clone())
        return delta

    monkeypatch.setattr(p, "predict_end_state_from_memory", spatial)
    origin = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    endpoint = origin + delta.detach()
    args = ([], torch.zeros(2, 1, dtype=torch.long), torch.ones(2, 1, dtype=torch.bool),
            torch.tensor([0, 1]), endpoint, torch.ones(2, dtype=torch.bool))
    kwargs = {"previous_skill_code": torch.tensor([-1, 0]), "start_xyz": origin}
    loss, metrics = p.loss_with_end_state(*args, **kwargs)
    assert metrics["end_xyz_mae"] == pytest.approx(0, abs=1e-6)
    assert metrics["end_state_skill_conditioned"] == 1
    torch.testing.assert_close(captured[0], args[3])
    torch.testing.assert_close(p.absolute_end_state(delta, origin), endpoint)
    shift = torch.tensor([.7, -.3, .5])
    shifted = (*args[:4], endpoint + shift, args[5])
    _, translated = p.loss_with_end_state(*shifted, **{**kwargs, "start_xyz": origin + shift})
    assert translated["end_xyz_mae"] == pytest.approx(0, abs=1e-6)
    loss, metrics = p.loss_with_end_state(*args, **{**kwargs, "start_xyz": origin + .2})
    assert metrics["end_xyz_mae"] == pytest.approx(.2, abs=1e-6)
    loss.backward()
    assert delta.grad is not None and delta.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="raw jittered"):
        p.loss_with_end_state(*args, previous_skill_code=kwargs["previous_skill_code"])


def test_zoom_uniform_range_independent_samples_and_center_eval(monkeypatch):
    p = predictor(monkeypatch)
    image = torch.arange(256 * 256, dtype=torch.float32).reshape(1, 1, 256, 256).expand(2, 3, -1, -1)
    enlarged = F.interpolate(image, (282, 282), mode="bilinear", align_corners=False)
    draws = []

    def randint(high, size, *, device):
        assert high == 27 and size == (2,)
        result = torch.tensor([0, 26] if not draws else [26, 0], device=device)
        draws.append(result)
        return result

    monkeypatch.setattr(torch, "randint", randint)
    cropped = p.prepare_top_image(image)
    torch.testing.assert_close(cropped[0], enlarged[0, :, :256, 26:282])
    torch.testing.assert_close(cropped[1], enlarged[1, :, 26:282, :256])
    assert len(draws) == 2
    p.eval()
    rng = torch.get_rng_state().clone()
    torch.testing.assert_close(p.prepare_top_image(image), enlarged[:, :, 13:269, 13:269])
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("mode", PREDICTOR_MODES[4:])
def test_wrist_omitted_before_encoding_and_top_only_augmented(monkeypatch, mode):
    p = predictor(monkeypatch, mode).eval()
    top, wrist = torch.rand(2, 3, 20, 20), torch.rand(2, 3, 20, 20)
    p._embed_prefix([top, wrist], torch.zeros(2, 1, dtype=torch.long), torch.ones(2, 1, dtype=torch.bool))
    assert len(p.vlm.images) == (2 if p.uses_wrist else 1)
    if p.uses_wrist:
        torch.testing.assert_close(p.vlm.images[1], p._preprocess_image(wrist))


def test_proprioX_prompt_contains_no_state_and_survives_serialization():
    step = SkillVLAPrepareStateTokenizerProcessorStep(include_state=False)
    first = {TransitionKey.OBSERVATION: {OBS_STATE: torch.randn(1, 8)},
             TransitionKey.COMPLEMENTARY_DATA: {"task": ["pick_up_object"], SKILL_START_STATE: torch.rand(1, 8)}}
    second = {TransitionKey.COMPLEMENTARY_DATA: {"task": ["pick_up_object"]}}
    assert step(first)[TransitionKey.COMPLEMENTARY_DATA]["task"] == step(second)[TransitionKey.COMPLEMENTARY_DATA]["task"]
    prompt = step(first)[TransitionKey.COMPLEMENTARY_DATA]["task"][0]
    assert prompt == "Task: pick up object;\nAction: " and "State:" not in prompt
    assert SkillVLAPrepareStateTokenizerProcessorStep(**step.get_config()).include_state is False


def test_start_xyz_preserved_before_normalization_uses_actual_start_not_current():
    step = SkillPredictorPreserveStartXYZProcessorStep()
    actual = torch.tensor([[.15, .25, .35, 0.]])
    transition = {TransitionKey.OBSERVATION: {OBS_STATE: torch.ones(1, 4) * 99},
                  TransitionKey.COMPLEMENTARY_DATA: {SKILL_START_STATE: actual}}
    result = step(transition)[TransitionKey.COMPLEMENTARY_DATA][PREDICTOR_START_XYZ]
    torch.testing.assert_close(result, actual[:, :3])
    assert result.data_ptr() != actual.data_ptr()
    runtime = {TransitionKey.OBSERVATION: {OBS_STATE: actual}}
    torch.testing.assert_close(step(runtime)[TransitionKey.COMPLEMENTARY_DATA][PREDICTOR_START_XYZ], result)


@pytest.mark.parametrize("mode", PREDICTOR_MODES[4:])
def test_runtime_returns_absolute_goal_once_and_preserves_history(monkeypatch, mode):
    from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

    p = predictor(monkeypatch, mode).eval()
    memory, mask = torch.randn(1, 3, 16), torch.zeros(1, 3, dtype=torch.bool)
    monkeypatch.setattr(p, "reader_memory", lambda *a: (memory, mask))
    monkeypatch.setattr(p.head, "decode", lambda h: torch.tensor([2]))
    delta = torch.tensor([[.1, .2, -.3]])
    monkeypatch.setattr(p, "predict_end_state_from_memory", lambda *a: delta)
    policy = object.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.model = nn.Module()
    policy.model.skill_predictor = p
    policy.config = SimpleNamespace(architecture_label="both_3")
    origin = torch.tensor([[.3, .4, .5]])
    batch = {
        "observation.images.image": torch.zeros(1, 3, 8, 8),
        "skill_previous_code": torch.tensor([-1]), PREDICTOR_START_XYZ: origin,
        OBS_STATE: torch.full((1, 8), 999.),  # normalized state must NEVER be the origin
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 2, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 2, dtype=torch.bool),
    }
    if p.uses_wrist:
        batch["observation.images.wrist_image"] = torch.zeros(1, 3, 8, 8)
    codes, absolute = policy.predict_skill_code_and_end_state(batch)
    assert codes.tolist() == [2]
    torch.testing.assert_close(absolute, origin + delta)
    _, gt, predicted = policy.predict_skill_and_end_state_diagnostics(batch, torch.tensor([1]))
    torch.testing.assert_close(gt, origin + delta)
    torch.testing.assert_close(predicted, origin + delta)
    batch.pop(PREDICTOR_START_XYZ)
    with pytest.raises(ValueError, match="raw skill-start XYZ"):
        policy.predict_skill_code_and_end_state(batch)


@pytest.mark.parametrize("mode", PREDICTOR_MODES[4:])
def test_auxiliary_training_step_all_mode3_variants(monkeypatch, mode):
    from lerobot.policies.skill_aux.modeling_skill_aux import SkillAuxPolicy
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

    p = predictor(monkeypatch, mode)
    memory, mask = torch.randn(2, 4, 16), torch.zeros(2, 4, dtype=torch.bool)
    image_counts = []

    def reader_memory(images, *args):
        image_counts.append(len(images))
        return memory, mask

    monkeypatch.setattr(p, "reader_memory", reader_memory)
    policy = object.__new__(SkillAuxPolicy)
    nn.Module.__init__(policy)
    policy.config = p.config
    policy.model = nn.Module()
    policy.model.skill_predictor = p
    batch = {
        "skill_start_image": torch.rand(2, 3, 8, 8),
        SKILL_START_STATE: torch.randn(2, 8), "skill_end_state": torch.randn(2, 8),
        "skill_end_state_valid": torch.ones(2, dtype=torch.bool),
        "skill_code": torch.tensor([0, 1]), "skill_previous_code": torch.tensor([-1, 0]),
        OBS_LANGUAGE_TOKENS: torch.zeros(2, 2, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 2, dtype=torch.bool),
    }
    if p.uses_wrist:
        batch["skill_start_wrist_image"] = torch.rand(2, 3, 8, 8)
    if not p.uses_previous_skill:
        batch.pop("skill_previous_code")
    loss, metrics = policy._skill_predictor_objective(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert p.end_state_head[-1].weight.grad is not None
    if p.uses_previous_skill:
        assert p.previous_skill_bos.grad is not None
    assert image_counts == [2 if p.uses_wrist else 1]
    assert metrics["skill_predictor/end_state_skill_conditioned"] == 1
    assert metrics["skill_predictor/proprio_input_enabled"] == float(p.uses_proprio)
    assert metrics["skill_predictor/relative_xyz"] == 1
