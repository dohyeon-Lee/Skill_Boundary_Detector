"""Pretrained VLM prefix, unequal stream widths and existing aux3 routing."""
from pathlib import Path
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.uv_fullskill import UV_BSPLINE_SKILL_PI05_VLM_BASE_LABELS as LABELS
from lerobot.policies.skill_expert import cond_gemma, modeling_utils, pi05_vlm_vision as vlm
from lerobot.policies.skill_expert.modeling_skill_expert import (
    SkillExpertPolicy, _allowed_pi05_missing_key, _build_state_dict, _load_pretrained_state_dict,
    _map_pi05_condition_vlm_key,
)
from test_stage1_vsa import _skill_config


class Tokenizer:
    padding_side = "right"
    extra_padding = 0
    def __call__(self, prompts, **kwargs):
        assert kwargs["truncation"] is False
        sequences = [[ord(character) for character in prompt] for prompt in prompts]
        length = max(map(len, sequences)) + self.extra_padding
        return {
            "input_ids": torch.tensor([seq + [0] * (length - len(seq)) for seq in sequences]),
            "attention_mask": torch.tensor([[1] * len(seq) + [0] * (length - len(seq)) for seq in sequences]),
        }


def tiny_vlm_model(monkeypatch, label, checkpoint=False, dtype=torch.float32, freeze=False, bridge_layers=9):
    def dimensions(variant):
        width = 64 if variant == "gemma_2b" else 32
        return SimpleNamespace(width=width, depth=18, mlp_dim=width * 2,
                               num_heads=4, num_kv_heads=2, head_dim=8)
    monkeypatch.setattr(modeling_utils, "get_gemma_config", dimensions)
    monkeypatch.setattr(cond_gemma, "get_gemma_config", dimensions)

    class Trunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = modeling_utils.build_gemma("gemma_2b", use_adarms=False, attention_backend="sdpa").model
            self.language_model.embed_tokens = nn.Embedding(256, 64)
            self.vision_tower = nn.Conv2d(3, 64, 2, stride=2)
            self.multi_modal_projector = nn.Linear(64, 64)
        def get_image_features(self, image):
            features = self.vision_tower(image.to(self.vision_tower.weight.dtype)).flatten(2).transpose(1, 2)
            return SimpleNamespace(pooler_output=self.multi_modal_projector(features))

    monkeypatch.setattr(vlm, "build_paligemma_model", lambda *args, **kwargs: Trunk())
    monkeypatch.setattr(vlm.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(cond_gemma.AutoModel, "from_pretrained", lambda *args, **kwargs: pytest.fail("VLM must not load DINO"))
    from lerobot.configs.uv_fullskill import VANILA_PI05_VLM_BASE_LABELS
    from lerobot.configs.uv_fullskill import DUAL_BOTTLENECK_LABEL
    if label == DUAL_BOTTLENECK_LABEL:
        from lerobot.policies.skill_expert.dual_bottleneck import DualBottleneckSkillExpert
        config_factory, model_class = _skill_config, DualBottleneckSkillExpert
    elif label in VANILA_PI05_VLM_BASE_LABELS:
        from test_vanila import config_for
        config_factory, model_class = config_for, vlm.Pi05VLMVanilaSkillExpert
    else:
        config_factory, model_class = _skill_config, vlm.Pi05VLMIndependentSkillPlannerExpert
    config = config_factory(label, tokenizer_path="unused", tokenizer_max_length=300,
                           visual_bridge_last_n_layers=bridge_layers, dino_image_size=4,
                           attention_backend="sdpa", freeze_vision_encoder=freeze)
    config.visual_bottleneck_width, config.visual_bridge_heads = 16, 4
    model = model_class(config).to(dtype=dtype)
    if checkpoint:
        model.gradient_checkpointing_enable()
    return model


@pytest.mark.parametrize("label,baseline", LABELS.items())
def test_config_roundtrip_and_pretraining_requirements(tmp_path, label, baseline):
    config = _skill_config(label)
    assert config.uses_pi05_vlm_vision and config.cond_encoder_variant == "gemma_2b"
    assert config.vision_backbone == "pi05vlm"
    assert config.action_expert_variant == "gemma_300m"
    assert config.skill_planner_mode == "denoising" and not config.reads_bspline_plan
    assert config.uses_skill_prefix_vision == ("_wovis" not in label)
    assert config.omits_skill_expert_uv_goal == ("_nouv" in label)
    assert config.uses_triple_auxiliary_prediction and config.uses_joint_bottleneck_self_attention
    assert config.architecture_revision != _skill_config(baseline).architecture_revision
    assert not _allowed_pi05_missing_key("model.cond_encoder.vlm.language_model.norm.weight", config)
    config.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.uses_pi05_vlm_vision and restored.architecture_revision == config.architecture_revision


def test_prompt_preserves_pi05_state_and_clips_uv():
    state = torch.tensor([[-1.1, -1., 0., 1., 1.1]] * 2)
    prompts = vlm.conditioning_prompts(state, torch.tensor([[1., 0., -1.]] * 2),
                                      torch.tensor([[-2., 2.], [0., 0.]]), torch.tensor([1., 0.]))
    assert "State: -1 0 128 255 255;\nAction: " in prompts[0]
    assert "Skill: 1 0 -1; Gripper goal in top-view image: 0 255" in prompts[0]
    assert "Gripper goal in top-view image: none" in prompts[1]


@pytest.mark.parametrize("label", LABELS)
@pytest.mark.parametrize("checkpoint,dtype", [(False, torch.float32), (True, torch.bfloat16)])
def test_actual_constructor_prefix_and_gradients(monkeypatch, label, checkpoint, dtype):
    torch.manual_seed(12)
    model = tiny_vlm_model(monkeypatch, label, checkpoint, dtype)
    model.train()
    assert model.dino is None and model.width == 32
    assert len(model.cond_encoder.model.layers) == 18
    assert model.start_bridge_layer == 8 and model.current_bridge_layers == (17,)
    assert model.prefix_vision == ("_wovis" not in label)
    images = [torch.rand(2, 3, 4, 4), torch.rand(2, 3, 4, 4)]
    goal = torch.tensor([[.2, -.2, 1., .1, .1, .1]] * 2)
    condition = model._project_condition_state(torch.zeros(2, 32), skill_code=torch.tensor([1, 2]), end_pose=goal)
    image_tokens = model._condition_tokens(images)
    assert image_tokens.shape == (2, 8, 64)
    latents = model._encode_layerwise_latents(image_tokens, condition)
    assert len(latents) == 18 and latents[-1].shape == (2, 100, 16)
    assert model._final_agent_patch_tokens.shape == (2, 4, 64)  # no CLS removed
    assert model._final_wrist_patch_tokens.shape == (2, 4, 64)
    patch_logits = model.agent_patch_align_head(latents[-1][:, :1], model._final_agent_patch_tokens)
    assert patch_logits.shape == (2, 4)
    loss = patch_logits.square().mean() + model.agent_end_state_head(latents[-1][:, 2]).square().mean()
    # A second encoding before backward must not corrupt checkpoint replay.
    changed = list(condition)
    changed[2] = torch.full_like(changed[2], -.9)
    second = model._encode_layerwise_latents(model._condition_tokens(images), tuple(changed))
    assert not torch.allclose(latents[-1], second[-1])
    (loss + second[-1].square().mean()).backward()
    assert model.cond_encoder.model.layers[0].self_attn.q_proj.weight.grad is not None
    assert model.cond_encoder.model.embed_tokens.weight.grad is not None
    assert model.cond_encoder.vlm.vision_tower.weight.grad is not None
    assert model.layerwise_condition_readers[0].self_attention.in_proj_weight.grad is not None


def test_frozen_image_tower_and_truncation_guard(monkeypatch):
    model = tiny_vlm_model(monkeypatch, next(iter(LABELS)), freeze=True)
    model.train()
    assert not model.cond_encoder.vlm.vision_tower.training
    assert not model.cond_encoder.vlm.vision_tower.weight.requires_grad
    assert model.cond_encoder.model.embed_tokens.weight.requires_grad
    model.config.tokenizer_max_length = 1
    with pytest.raises(ValueError, match="truncate"):
        model._embed_condition_text((torch.zeros(1, 32), torch.zeros(1, 3), torch.zeros(1, 2), torch.ones(1)), "cpu")


@pytest.mark.parametrize("label", LABELS)
@pytest.mark.parametrize("bridge_layers", [9, 1, 17])
def test_action_fullskill_training_and_inference(monkeypatch, label, bridge_layers):
    model = tiny_vlm_model(monkeypatch, label, bridge_layers=bridge_layers)
    assert model.split_layer == 18 - bridge_layers
    assert model.start_bridge_layer == model.split_layer - 1
    assert model.current_bridge_layers == (17,)
    if bridge_layers == 1:
        # The late segment is L18 alone, so last/all late-vision routes coincide.
        assert model.current_bridge_layers == tuple(range(model.split_layer, 18))
    policy = SkillExpertPolicy.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.config, policy.model = model.config, model
    groups = policy._get_cond_gemma_optim_params()
    image_group = next(group for group in groups if group["group_name"] == "siglip")
    assert image_group["lr_scale"] == model.config.dino_lr_scale
    image_ids = {id(parameter) for parameter in image_group["params"]}
    assert image_ids == {id(parameter) for parameter in model.cond_encoder.vlm.vision_tower.parameters()}
    assert all(id(parameter) not in image_ids for group in groups if group["group_name"] != "siglip" for parameter in group["params"])
    images, state, skill = [torch.rand(1, 3, 4, 4)] * 2, torch.zeros(1, 32), torch.tensor([1])
    goal = torch.tensor([[.2, -.2, 1., .1, .1, .1]])
    condition = model._project_condition_state(state, skill_code=skill, end_pose=goal)
    start = model._encode_layerwise_latents(model._condition_tokens(images), condition)[-1]
    model.set_skill_start_latent(start)
    prefix_calls = []
    original_layer = model._expert_layer_with_latent_bridge
    def record_prefix(*args):
        prefix_calls.append(args[0])
        if not model.prefix_vision:
            assert args[6] is None
        else:
            assert (args[6] is start) == (args[0] == model.start_bridge_layer)
        return original_layer(*args)
    model._expert_layer_with_latent_bridge = record_prefix
    plan, prediction, fullskill = model.generate_independent_plan(skill, goal, target=torch.randn(1, 30, 8))
    assert prefix_calls == list(range(18 - bridge_layers))
    model._expert_layer_with_latent_bridge = original_layer
    assert plan is None and prediction is None
    residual = model(images, state, skill, torch.randn(1, 10, 32), end_pose=goal)
    (residual.square().mean() + fullskill.mean()).backward()
    assert model.action_out_proj.weight.grad is not None
    assert model.bspline_noise_input.weight.grad is not None
    assert model.cond_encoder.model.layers[0].self_attn.q_proj.weight.grad is not None
    model.eval()
    actions = model.sample_actions(images, state, skill, num_steps=1, end_pose=goal)
    assert actions.shape == (1, 10, 32) and torch.isfinite(actions).all()


@pytest.mark.parametrize("label", [label for label in LABELS if "_wovis" not in label])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_scheduled_debug_preserves_losses_and_gradients_with_tuple_conditions(monkeypatch, label, checkpoint):
    model = tiny_vlm_model(monkeypatch, label, bridge_layers=17)
    for name in ("action_expert", "skill_planner", "vlm"):
        setattr(model.config, name + "_checkpointing", checkpoint)
    model.config.vision_checkpointing = False
    model.configure_stream_checkpointing()
    model.config.vsa_debug_schedule = [1]
    images = [torch.rand(2, 3, 4, 4), torch.rand(2, 3, 4, 4)]
    state, skill = torch.zeros(2, 32), torch.tensor([1, 2])
    goal = torch.tensor([[.2, -.2, 1., .1, .1, .1], [-.3, .4, 1., -.1, .2, .3]])
    actions, target = torch.randn(2, 10, 32), torch.randn(2, 30, 8)
    parameters = [model.action_out_proj.weight, model.bspline_noise_input.weight,
                  model.cond_encoder.model.layers[0].self_attn.q_proj.weight,
                  model.agent_patch_align_head.query_proj.weight]
    reference_loss, reference_gradients = None, None
    for step in (2, 1):
        model.zero_grad(set_to_none=True)
        model.set_training_step(step)
        torch.manual_seed(123)
        condition = model._project_condition_state(state, skill_code=skill, end_pose=goal)
        start = model._encode_layerwise_latents(model._condition_tokens(images), condition)[-1]
        model.set_skill_start_latent(start)
        _, _, flow = model.generate_independent_plan(skill, goal, target=target)
        residual = model(images, state, skill, actions, end_pose=goal)
        top, wrist, xyz = model.predict_training_top_wrist_patch_and_xyz()
        loss = residual.square().mean() + flow.mean() + sum(x.square().mean() for x in (top, wrist, xyz))
        loss.backward()
        gradients = [parameter.grad.clone() for parameter in parameters]
        if step == 2:
            reference_loss, reference_gradients = loss.detach(), gradients
        else:
            torch.testing.assert_close(loss.detach(), reference_loss)
            for actual, expected in zip(gradients, reference_gradients, strict=True):
                torch.testing.assert_close(actual, expected)
            assert "activation/state_input_rms" in model._last_vsa_debug_stats
            assert "activation/goal_uv_input_rms" in model._last_vsa_debug_stats
            assert "sensitivity/state_shuffle/output_delta_rms" in model._last_vsa_debug_stats
            assert all(math.isfinite(value) for value in model._last_vsa_debug_stats.values())


def test_padding_does_not_leak_into_images(monkeypatch):
    model = tiny_vlm_model(monkeypatch, next(iter(LABELS))).eval()
    condition = (torch.zeros(1, 32), torch.zeros(1, 3), torch.zeros(1, 2), torch.ones(1))
    images = [torch.rand(1, 3, 4, 4)] * 2
    with torch.no_grad():
        tokens = model._condition_tokens(images)
        first = model._encode_layerwise_latents(tokens, condition)[-1]
        model.prefix_tokenizer.extra_padding = 5
        model.cond_encoder.model.embed_tokens.weight[0].fill_(100.)
        padded = model._encode_layerwise_latents(tokens, condition)[-1]
    torch.testing.assert_close(first, padded, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('action,planner,vlm', [(18, True, False), ([1,18], [1,17], [1,18]),
                                               (False, False, 2)])
def test_independent_checkpoint_controls_cover_expert_and_planner(monkeypatch, tmp_path, action, planner, vlm):
    from lerobot.utils.gradient_checkpointing import checkpoint_indices
    label=next(label for label in LABELS if '_wovis_last_' in label)
    model=tiny_vlm_model(monkeypatch,label,bridge_layers=1)
    for name,value in {'action_expert':action,'skill_planner':planner,'vlm':vlm,'vision':False}.items():
        setattr(model.config,name+'_checkpointing',value)
    model.configure_stream_checkpointing()
    saved_config=_skill_config(label,action_expert_checkpointing=action,skill_planner_checkpointing=planner,
                               vlm_checkpointing=vlm,vision_checkpointing=False)
    saved_config.save_pretrained(tmp_path)
    restored=PreTrainedConfig.from_pretrained(tmp_path, cli_overrides=[
        '--vision_checkpointing=[1,3]', '--vlm_checkpointing=false',
    ])
    assert restored.action_expert_checkpointing==action
    assert restored.skill_planner_checkpointing==planner
    assert restored.vision_checkpointing==[1,3] and restored.vlm_checkpointing is False
    calls=[]
    phase='vlm'
    original=torch.utils.checkpoint.checkpoint
    def checkpoint(function,*args,**kwargs):
        if function.__name__ in ('_expert_layer_with_latent_bridge','_vlm_layer_step'):
            calls.append((phase,function.__name__,args[0]))
        return original(function,*args,**kwargs)
    monkeypatch.setattr(torch.utils.checkpoint,'checkpoint',checkpoint)
    images,state,skill=[torch.rand(1,3,4,4)]*2,torch.zeros(1,32),torch.tensor([1])
    goal=torch.tensor([[.2,-.2,1.,.1,.1,.1]])
    condition=model._project_condition_state(state,skill_code=skill,end_pose=goal)
    start=model._encode_layerwise_latents(model._condition_tokens(images),condition)[-1]
    model.set_skill_start_latent(start)
    phase='planner'
    _,_,loss=model.generate_independent_plan(skill,goal,target=torch.randn(1,30,8))
    phase='action'
    prediction=model(images,state,skill,torch.randn(1,10,32),end_pose=goal)
    for key,depth,spec in [('planner',17,planner),('action',18,action)]:
        selected=[index for p,name,index in calls if p==key and name=='_expert_layer_with_latent_bridge']
        assert selected==sorted(checkpoint_indices(spec,depth))
    assert [index for p,name,index in calls if p=='vlm' and name=='_vlm_layer_step']==sorted(checkpoint_indices(vlm,18))
    (prediction.square().mean()+loss.mean()).backward()
    assert model.action_out_proj.weight.grad is not None
    assert model.bspline_noise_input.weight.grad is not None
    assert model.cond_encoder.model.layers[0].self_attn.q_proj.weight.grad is not None


def test_checkpoint_mapping_streaming_and_shared_embedding(tmp_path):
    from safetensors.torch import save_file
    raw = {"paligemma_with_expert.paligemma.model.language_model.norm.weight": torch.ones(4),
           "paligemma_with_expert.paligemma.lm_head.weight": torch.ones(7, 4),
           "paligemma_with_expert.gemma_expert.model.norm.weight": torch.ones(2)}
    options = dict(architecture="layerwise_cond_bottleneck", include_condition_vlm=True, include_predictor_vlm=True)
    mapped, is_pi05 = _build_state_dict(raw, **options)
    assert is_pi05
    assert "model.cond_encoder.vlm.language_model.norm.weight" in mapped
    assert "model.skill_predictor.vlm.language_model.norm.weight" in mapped
    assert "model.cond_encoder.vlm.language_model.embed_tokens.weight" in mapped
    save_file(raw, tmp_path / "model.safetensors")
    streamed, _ = _load_pretrained_state_dict(tmp_path, {}, **options)
    assert mapped.keys() == streamed.keys()
    for key in mapped:
        torch.testing.assert_close(mapped[key], streamed[key])


def test_local_pi05_checkpoint_covers_entire_real_vlm():
    from safetensors import safe_open
    path = Path(__file__).resolve().parents[4] / "models/pi05_base/model.safetensors"
    if not path.is_file():
        pytest.skip("local pi0.5 checkpoint unavailable")
    with torch.device("meta"):
        trunk = modeling_utils.build_paligemma_model("gemma_2b", image_size=224)
    with safe_open(path, framework="pt", device="cpu") as source:
        shapes = {_map_pi05_condition_vlm_key(key): source.get_slice(key).get_shape()
                  for key in source.keys() if _map_pi05_condition_vlm_key(key) is not None}
        embedding = "model.cond_encoder.vlm.language_model.embed_tokens.weight"
        if embedding not in shapes:
            shapes[embedding] = source.get_slice("paligemma_with_expert.paligemma.lm_head.weight").get_shape()
    for key, value in trunk.state_dict().items():
        assert tuple(shapes["model.cond_encoder.vlm." + key]) == tuple(value.shape), key
