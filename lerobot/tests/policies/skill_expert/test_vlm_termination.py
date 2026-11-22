"""Shared-VLM termination uses the deployed prefix, with no start proprio."""
import torch
import pytest
from types import SimpleNamespace
from dataclasses import replace

from lerobot.configs.uv_fullskill import DUAL_BOTTLENECK_NOPROPRIO_LABEL
from lerobot.policies.skill_expert.modeling_skill_expert import _allowed_pi05_missing_key
from test_pi05_vlm_vsa import tiny_vlm_model
from lerobot.policies.skillVLA.vlm_termination_supervision import (
    DISTANCE, TERMINAL, PAST_CODES, PAST_VALID, PAST_UV, PAST_TERMINAL, PAST_DISTANCE,
    termination_conditions, termination_targets,
)


def test_termination_gradients_share_vlm_but_do_not_train_action_expert(monkeypatch):
    torch.manual_seed(19)
    model = tiny_vlm_model(monkeypatch, DUAL_BOTTLENECK_NOPROPRIO_LABEL,
                           vlm_termination_enabled=True, device="cpu")
    images = [torch.rand(2, 3, 4, 4) for _ in range(2)]
    state, skill = torch.randn(2, 32), torch.tensor([1, 2])
    goal = torch.tensor([[.2, -.3, 1., .1, .2, .3]] * 2)
    model(images, state, skill, torch.randn(2, 10, 32), end_pose=goal)
    logits, memory = model.take_vlm_termination_outputs()
    assert logits.shape == (2,) and memory.shape[:2] == (2, 8)
    # Recondition at the same frame for a past skill. Start proprio must have
    # no effect: it is deliberately absent from the VLM termination prefix.
    other = goal.clone()
    other[:, 3:] += 100
    expected = model.vlm_termination_from_image_tokens(memory, state, skill, goal)
    actual = model.vlm_termination_from_image_tokens(memory, state, skill, other)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    (logits.square().mean() + actual.square().mean()).backward()
    assert model.vlm_termination_head[-1].weight.grad.abs().sum() > 0
    assert model.cond_encoder.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.gemma_expert.parameters())
    assert model.take_vlm_termination_outputs() == (None, None)
    assert _allowed_pi05_missing_key("model.vlm_termination_head.1.weight", model.config)


def test_jitter_selects_condition_but_never_moves_termination_gt_endpoint():
    # A=0..99, B=100..199. B starting early keeps its endpoint at 199.
    early, _, past = termination_conditions(95, 1, [0, 100], [100, 100], [4, 7], 10,
                                           episode_length=200)
    assert early == 104 and past[past >= 0].tolist() == [0]
    # Late-held A has already ended at frame 103, even if its VSA virtual end
    # was delayed to 107. Its GT post-end window ends at 109, not 117.
    distances = torch.tensor([99 - frame for frame in (98, 99, 103, 107, 109, 110)])
    target, valid = termination_targets(distances, sigma=1., post_end_steps=10)
    torch.testing.assert_close(target, torch.tensor([torch.exp(torch.tensor(-.5)), 1, 1, 1, 1, 1]))
    assert valid.tolist() == [True, True, True, True, True, False]
    late, _, past = termination_conditions(103, 0, [0, 100], [100, 100], [4, 7], 10,
                                          episode_length=200)
    assert late == -4 and past[past >= 0].tolist() == [1]  # A main; B retains its GT negative.
    _, _, past = termination_conditions(103, 1, [0, 100], [100, 100], [4, 7], 10,
                                       episode_length=200)
    assert past[past >= 0].tolist() == [0]  # B main, plus A positive.
    _, _, expired = termination_conditions(110, 1, [0, 100], [100, 100], [4, 7], 10,
                                          episode_length=200)
    assert (expired < 0).all()


def test_termination_loss_ignores_expired_holds_and_weights_only_terminal_objective():
    from lerobot.policies.skill_expert.vlm_termination import vlm_termination_loss
    logits = torch.zeros(3, requires_grad=True)
    model = SimpleNamespace(take_vlm_termination_outputs=lambda: (logits, torch.zeros(3, 2, 4)))
    config = SimpleNamespace(vlm_termination_target_sigma=1., vlm_termination_post_end_steps=10,
                             vlm_termination_positive_weight=2.)
    batch = {DISTANCE: torch.tensor([-4, -11, 0]), TERMINAL: torch.tensor([False, False, True]),
             PAST_VALID: torch.zeros(3, 11, dtype=torch.bool)}
    loss, metrics = vlm_termination_loss(model, config, batch, torch.zeros(3, 32))
    expected = torch.log(torch.tensor(2.)) * torch.tensor([2., 0., 22.]) * (3 / 12)
    torch.testing.assert_close(loss, expected)
    loss.mean().backward()
    assert logits.grad[0] < 0 and logits.grad[1] == 0 and logits.grad[2] == 11 * logits.grad[0]
    assert metrics['vlm_termination/post_end_conditions_per_frame'] == 0


@pytest.mark.parametrize('finetune', [False, True])
def test_real_policy_training_loss_and_serialized_head(monkeypatch, tmp_path, finetune):
    from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy, apply_newtask_ft_freeze
    from lerobot.configs.policies import PreTrainedConfig
    from test_uv_gt_xyz import goal_batch
    model = tiny_vlm_model(monkeypatch, DUAL_BOTTLENECK_NOPROPRIO_LABEL,
                           vlm_termination_enabled=True, device="cpu",
                           newtask_ft_enabled=finetune, newtask_ft_freeze_expert=finetune,
                           newtask_ft_skill_flow_loss=False)
    # Build the policy around the same tiny real VLM/Expert used above.
    policy = SkillExpertPolicy(model.config)
    policy.model = model
    if finetune:
        apply_newtask_ft_freeze(model, model.config)
    batch = goal_batch()
    batch.update({'observation.state': torch.zeros(2, 8), 'skill_code': torch.tensor([1, 2]),
                  'skill_decoder_state': torch.zeros(2, 8),
                  'action': torch.randn(2, 10, 7), 'skill_effective_de': torch.tensor([4, 0]),
                  'skill_bspline_states': torch.randn(2, 30, 8),
                  'observation.images.image': torch.rand(2, 3, 4, 4),
                  'observation.images.wrist_image': torch.rand(2, 3, 4, 4),
                  DISTANCE: torch.tensor([-4, 0]), TERMINAL: torch.zeros(2, dtype=torch.bool),
                  PAST_CODES: torch.zeros(2, 11, dtype=torch.long),
                  PAST_VALID: torch.zeros(2, 11, dtype=torch.bool),
                  PAST_UV: torch.zeros(2, 11, 2), PAST_TERMINAL: torch.zeros(2, 11, dtype=torch.bool),
                  PAST_DISTANCE: torch.zeros(2, 11, dtype=torch.long)})
    batch[PAST_VALID][1, 0] = True
    batch[PAST_CODES][1, 0] = 3
    batch[PAST_UV][1, 0] = torch.tensor([.6, -.4])
    observed = []
    original = model.vlm_termination_from_image_tokens
    def record(memory, state, code, goal):
        observed.append((code.detach().clone(), goal.detach().clone()))
        return original(memory, state, code, goal)
    monkeypatch.setattr(model, 'vlm_termination_from_image_tokens', record)
    loss, metrics = policy(batch)
    assert observed[0][0].tolist() == [3]
    torch.testing.assert_close(observed[0][1][:, :2], torch.tensor([[.6, -.4]]))
    assert torch.isfinite(loss) and metrics['vlm_termination/post_end_conditions_per_frame'] == .5
    loss.backward()
    assert model.vlm_termination_head[-1].weight.grad.abs().sum() > 0
    if finetune:
        assert all(not p.requires_grad and p.grad is None for p in model.gemma_expert.parameters())
        assert model.cond_encoder.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    optimizer_ids = {id(p) for group in policy.get_optim_params() for p in group['params']}
    assert all(id(p) in optimizer_ids for p in model.vlm_termination_head.parameters())
    # Tiny fixtures shrink the otherwise fixed bottleneck geometry after
    # validation. Serialize the production geometry for the config roundtrip.
    saved = replace(policy.config, visual_bottleneck_width=256, visual_bridge_heads=8)
    saved.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.vlm_termination_enabled and restored.vlm_termination_post_end_steps == 10
    # The deployed head consumes current cameras/state/skill/UV only.
    batch.pop('skill_start_state')
    probability = policy.predict_termination_probability(batch)
    assert probability.shape == (2,) and torch.isfinite(probability).all()


def test_dataset_attaches_selected_skill_gt_distance_and_past_goal_without_start_input():
    import numpy as np
    from lerobot.policies.skillVLA.dataset_skillVLA import SkillVLADataset
    dataset = object.__new__(SkillVLADataset)
    dataset._vlm_termination_post_end_steps = 10
    dataset._focus_uv = SimpleNamespace(target=lambda episode, rank, start: (
        np.array([rank * .2, .1], dtype=np.float32), None, True, False))
    item = {}
    dataset._add_vlm_termination_targets(item, 0, 103, 0, [0, 100, -1], [100, 100, 0], [4, 7, 27], 200)
    assert item[DISTANCE].item() == -4
    assert item[PAST_CODES][item[PAST_VALID]].tolist() == [7]
    assert item[PAST_DISTANCE][item[PAST_VALID]].tolist() == [96]
    dataset._add_vlm_termination_targets(item, 0, 103, 1, [0, 100, -1], [100, 100, 0], [4, 7, 27], 200)
    assert item[DISTANCE].item() == 96
    assert item[PAST_CODES][item[PAST_VALID]].tolist() == [4]
    assert not any('start_state' in key for key in item)
