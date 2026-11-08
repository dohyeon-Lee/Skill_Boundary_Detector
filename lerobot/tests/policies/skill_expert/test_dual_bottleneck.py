"""Shared-current VLM, independent final bottlenecks, and prefix-only auxiliary FM."""
from dataclasses import replace

import pytest
import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.configs.uv_fullskill import DUAL_BOTTLENECK_LABEL as LABEL
from lerobot.configs.uv_fullskill import DUAL_BOTTLENECK_LAST_LABEL, DUAL_BOTTLENECK_NAMES
from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy, _allowed_pi05_missing_key
from test_pi05_vlm_vsa import tiny_vlm_model
from test_stage1_vsa import _skill_config
from test_uv_gt_xyz import goal_batch


def inputs():
    return ([torch.rand(2, 3, 4, 4), torch.rand(2, 3, 4, 4)],
            torch.randn(2, 32), torch.tensor([1, 2]), torch.randn(2, 10, 32),
            torch.tensor([[.2, -.3, 1., .1, .2, .3]] * 2))


@pytest.fixture(params=DUAL_BOTTLENECK_NAMES)
def label(request):
    return request.param


def test_config_roundtrip_and_warmstart_contract(tmp_path, label):
    cfg = _skill_config(label)
    if label == DUAL_BOTTLENECK_LAST_LABEL:
        assert cfg.architecture_revision != _skill_config(LABEL).architecture_revision
    assert cfg.uses_dual_bottleneck and cfg.uses_pi05_vlm_vision
    assert cfg.uses_independent_skill_planner and cfg.skill_planner_mode == 'denoising'
    assert cfg.omits_start_vision and not cfg.uses_symmetric_start_vision
    assert not cfg.reads_bspline_plan and not cfg.skill_flow_enabled
    assert cfg.uses_skill_uv_goal_conditioning and cfg.uses_joint_bottleneck_self_attention
    assert cfg.uses_triple_auxiliary_prediction and cfg.predicts_xyz_only
    cfg.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.uses_dual_bottleneck and restored.architecture_revision == cfg.architecture_revision
    for key in ('lower_layerwise_latent_queries', 'lower_layerwise_condition_readers.0.cross_attention.in_proj_weight',
                'lower_agent_end_state_head.1.weight', 'bspline_noise_input.weight'):
        assert _allowed_pi05_missing_key('model.' + key, cfg)
    assert not _allowed_pi05_missing_key('model.cond_encoder.vlm.language_model.norm.weight', cfg)
    with pytest.raises(ValueError, match='9\\+9'):
        replace(cfg, visual_bridge_last_n_layers=17)


@pytest.mark.parametrize('checkpoint,dtype', [(False, torch.float32), (True, torch.bfloat16)])
def test_single_vlm_pass_both_towers_and_prefix_flow_gradients(monkeypatch, checkpoint, dtype, label):
    torch.manual_seed(32)
    model = tiny_vlm_model(monkeypatch, label, dtype=dtype)
    for field in ('action_expert','skill_planner','vlm'):
        setattr(model.config, field + '_checkpointing', checkpoint)
    # This CPU fixture replaces SigLIP's block stack with a single Conv2d.
    model.config.vision_checkpointing = False
    model.configure_stream_checkpointing()
    counts = {'image':0, 'vlm':0}
    def image_count(*args): counts['image'] += 1
    def vlm_count(*args): counts['vlm'] += 1
    hooks = [model.cond_encoder.vlm.vision_tower.register_forward_hook(image_count)]
    hooks.extend(layer.self_attn.q_proj.register_forward_hook(vlm_count)
                 for layer in model.cond_encoder.model.layers)
    images,state,skill,actions,goal = inputs()
    residual = model(images,state,skill,actions,end_pose=goal,fullskill_target=torch.randn(2,30,8))
    assert counts == {'image':2,'vlm':18}
    flow = model.take_fullskill_flow_loss()
    assert flow.shape == (2,) and torch.isfinite(flow).all()
    assert model._last_lower_latent is None and model._fullskill_flow_loss is None
    # The independent fullskill branch traverses L1..9 and never L10..18.
    early = model.gemma_expert.model.layers[0].self_attn.q_proj.weight
    late = model.gemma_expert.model.layers[9].self_attn.q_proj.weight
    early_grad, late_grad = torch.autograd.grad(flow.mean(), (early,late), retain_graph=True, allow_unused=True)
    assert early_grad is not None and early_grad.abs().sum() > 0 and late_grad is None
    upper,lower = model.predict_training_dual_auxiliaries()
    assert [x.shape for x in upper] == [x.shape for x in lower] == [(2,4),(2,4),(2,3)]
    aux = sum(x.square().mean() for x in (*upper,*lower))
    (residual.square().mean() + flow.mean() + .05*aux).backward()
    for parameter in (model.layerwise_latent_queries, model.lower_layerwise_latent_queries,
                      model.lower_visual_bridge_attention.q_proj_weight, model.visual_bridge_attention.q_proj_weight,
                      model.bspline_noise_input.weight, model.lower_agent_end_state_head[-1].weight,
                      model.agent_end_state_head[-1].weight,
                      model.cond_encoder.model.layers[0].self_attn.q_proj.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert model.lower_layerwise_condition_readers[0] is not model.layerwise_condition_readers[0]
    assert not any('start_pose_condition.' in k for k in model.state_dict())
    for hook in hooks: hook.remove()


def test_routes_final_separate_latents_and_no_start_xyz(monkeypatch, label):
    model=tiny_vlm_model(monkeypatch,label)
    images,state,skill,actions,goal=inputs()
    nn.init.normal_(model.bridge_proprio_condition[-1].weight,std=.1)
    nn.init.normal_(model.end_pose_condition[-1].weight,std=.1)
    model._project_condition_state(state,skill_code=skill,end_pose=goal)
    time=torch.full((2,),.4)
    early=model._expert_condition(time,end_pose=goal)
    late=model._action_layer_condition(9,early)
    torch.testing.assert_close(late,model._time_condition(time))
    changed=goal.clone();changed[:,3:]+=100
    torch.testing.assert_close(early,model._expert_condition(time,end_pose=changed))
    model._project_condition_state(state+.5,skill_code=skill,end_pose=goal)
    new_early=model._expert_condition(time,end_pose=goal)
    assert not torch.equal(early,new_early)
    torch.testing.assert_close(late,model._action_layer_condition(9,new_early))
    latents=[torch.randn(2,200,16) for _ in range(18)]
    calls=[]; original=model._expert_layer_with_latent_bridge
    bridges=[]; original_bridge=model._visual_bridge_update
    def capture_bridge(index, *args):
        bridges.append(index)
        return original_bridge(index, *args)
    model._visual_bridge_update=capture_bridge
    def capture(*args): calls.append(args);return original(*args)
    model._expert_layer_with_latent_bridge=capture
    broadcast=model._skill_broadcasts(skill)[1]
    model._run_expert_with_layerwise_latents(actions,new_early,broadcast,latents)
    assert len(calls)==18
    expected = [8, 17] if label == DUAL_BOTTLENECK_LAST_LABEL else list(range(18))
    assert bridges == expected
    assert [i for i in range(18) if model._layer_uses_visual_bridge(i)] == expected
    assert model._active_visual_bridge_gates().numel() == len(expected)
    for index,call in enumerate(calls):
        assert (call[5] is broadcast) == (index<9)
        torch.testing.assert_close(call[4],new_early if index<9 else late)
        torch.testing.assert_close(call[6],latents[-1][:,:100] if index<9 else latents[-1][:,100:])
    bridges.clear()
    model._planner_forward(torch.randn(2,30,model.width), time, skill, goal, latents[-1][:,:100])
    assert bridges == [i for i in expected if i < 9]


def test_aux_target_never_enters_action_and_inference_skips_fullskill(monkeypatch, label):
    model=tiny_vlm_model(monkeypatch,label)
    images,state,skill,actions,goal=inputs()
    noise=torch.randn_like(actions);time=torch.full((2,),.5)
    first=model(images,state,skill,actions,end_pose=goal,noise=noise,time=time,
                fullskill_target=torch.zeros(2,30,8))
    model.take_fullskill_flow_loss()
    second=model(images,state,skill,actions,end_pose=goal,noise=noise,time=time,
                 fullskill_target=torch.ones(2,30,8))
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    monkeypatch.setattr(model,'_flow_readout',lambda *a,**k:pytest.fail('Action inference must skip fullskill'))
    model.eval()
    output=model.sample_actions(images,state,skill,num_steps=2,end_pose=goal)
    assert output.shape==(2,10,32) and torch.isfinite(output).all()


def test_policy_joint_losses_and_uv_noise_without_start_observation(monkeypatch, label):
    model=tiny_vlm_model(monkeypatch,label)
    model.config.output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(7,))}
    policy=SkillExpertPolicy.__new__(SkillExpertPolicy);nn.Module.__init__(policy)
    policy.config,policy.model=model.config,model
    images,state,skill,actions,goal=inputs()
    policy._collect_images=lambda batch:images
    batch=goal_batch();batch.pop('skill_start_state')
    batch.update(action=actions[:,:,:7],skill_code=skill,**{'observation.state':state,
        'skill_decoder_state':torch.zeros(2,8),'skill_bspline_states':torch.randn(2,30,8)})
    packed=policy._skill_delta_goal(batch,require_valid=True)
    torch.testing.assert_close(packed[:,3:6],torch.zeros(2,3))
    loss,metrics=policy(batch)
    assert torch.isfinite(loss)
    assert 'bspline/loss' in metrics
    assert 'vision_aux/upper/agent_patch/raw_loss' in metrics
    assert 'vision_aux/lower/agent_patch/raw_loss' in metrics
    upper=metrics['vision_aux/upper/patch_alignment/raw_sum']
    lower=metrics['vision_aux/lower/patch_alignment/raw_sum']
    assert metrics['patch_alignment/raw_sum']==pytest.approx(.5*(upper+lower))
    loss.backward()
    assert model.bspline_noise_input.weight.grad is not None
    assert model.lower_layerwise_latent_queries.grad is not None
    policy.eval()
    runtime={'skill_focus_uv':torch.zeros(2,2),'skill_focus_valid':torch.ones(2,dtype=torch.bool)}
    assert policy._skill_delta_goal(runtime).shape==(2,6)


def test_policy_dispatch_and_debug_preserves_both_auxiliary_graphs(monkeypatch, label):
    model=tiny_vlm_model(monkeypatch,label)
    from lerobot.policies.skill_expert import dual_bottleneck
    monkeypatch.setattr(dual_bottleneck,'DualBottleneckSkillExpert',lambda cfg:model)
    policy=SkillExpertPolicy(model.config)
    assert policy.model is model
    images,state,skill,actions,goal=inputs()
    model.config.vsa_debug_schedule=[1]
    parameters=[model.lower_layerwise_latent_queries,model.layerwise_latent_queries,
                model.bspline_noise_input.weight]
    baseline=None
    for step in (2,1):
        model.zero_grad(set_to_none=True);model.set_training_step(step)
        torch.manual_seed(841)
        residual=model(images,state,skill,actions,end_pose=goal,fullskill_target=torch.zeros(2,30,8))
        flow=model.take_fullskill_flow_loss()
        readouts=model.predict_training_dual_auxiliaries()
        loss=residual.square().mean()+flow.mean()+.05*sum(x.square().mean() for tower in readouts for x in tower)
        loss.backward()
        result=loss.detach(),[p.grad.clone() for p in parameters]
        if baseline is None: baseline=result
        else:
            torch.testing.assert_close(result[0],baseline[0])
            for actual,expected in zip(result[1],baseline[1],strict=True):
                torch.testing.assert_close(actual,expected)
    assert 'activation/state_input_rms' in model._last_vsa_debug_stats


def test_frozen_expert_ft_keeps_both_towers_and_fullskill_learning(monkeypatch):
    from lerobot.policies.skill_expert.modeling_skill_expert import apply_newtask_ft_freeze
    model=tiny_vlm_model(monkeypatch,LABEL)
    model.config=replace(model.config, visual_bottleneck_width=256,visual_bridge_heads=8,
                         newtask_ft_enabled=True,newtask_ft_freeze_expert=True,
                         newtask_ft_skill_flow_loss=True,newtask_ft_visual_aux_loss=True)
    model.config.visual_bottleneck_width,model.config.visual_bridge_heads=16,4
    report=apply_newtask_ft_freeze(model,model.config)
    assert report['trainable_expert_layers']==0 and report['action_head_trainable']==0
    assert not any(p.requires_grad for p in model.gemma_expert.parameters())
    assert not any(p.requires_grad for p in model.action_out_proj.parameters())
    frozen={name:p.detach().clone() for name,p in model.named_parameters() if not p.requires_grad}
    images,state,skill,actions,goal=inputs()
    residual=model(images,state,skill,actions,end_pose=goal,fullskill_target=torch.randn(2,30,8))
    flow=model.take_fullskill_flow_loss()
    assert flow is not None
    towers=model.predict_training_dual_auxiliaries()
    loss=residual.square().mean()+flow.mean()+.05*sum(x.square().mean() for tower in towers for x in tower)
    loss.backward()
    for p in (model.lower_layerwise_latent_queries,model.layerwise_latent_queries,
              model.cond_encoder.vlm.vision_tower.weight,
              model.lower_agent_end_state_head[-1].weight,model.agent_end_state_head[-1].weight):
        assert p.grad is not None and p.grad.abs().sum()>0
    torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01).step()
    for name,p in model.named_parameters():
        if name in frozen:
            assert p.grad is None
            torch.testing.assert_close(p,frozen[name],rtol=0,atol=0)
