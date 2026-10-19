from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/libero/configs/train_skillVLA/PT/eval/src"))
from cond_goal_ablation import cond_goal_settings, install_cond_goal_ablation


class _Both3(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(architecture_label="both_3", skill_end_pose_mode="xyz")
        self.cond_end_pose_condition = nn.Linear(3, 4)
        self.expert_goal = nn.Linear(6, 4)

    def forward(self, goal):
        return self.cond_end_pose_condition(goal[:, :3]), self.expert_goal(goal)


@pytest.mark.parametrize("mode,offset", [("off", None), ("offset", [.05, 0, 0])])
def test_cond_intervention_preserves_expert_goal_weights_and_rng(mode, offset):
    model = _Both3().eval()
    goal = torch.randn(2, 6)
    original = goal.clone()
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    normal, expert = model(goal)
    rng = torch.random.get_rng_state().clone()
    handle = install_cond_goal_ablation(model, mode=mode, offset_m=offset)
    changed, expert_after = model(goal)
    torch.testing.assert_close(expert, expert_after, rtol=0, atol=0)
    torch.testing.assert_close(goal, original, rtol=0, atol=0)
    assert torch.equal(rng, torch.random.get_rng_state())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
    if mode == "off":
        assert torch.equal(changed, torch.zeros_like(changed))
        assert not torch.equal(model.cond_end_pose_condition.bias, torch.zeros(4))
    else:
        expected = nn.functional.linear(goal[:, :3] + goal.new_tensor(offset),
                                        model.cond_end_pose_condition.weight, model.cond_end_pose_condition.bias)
        torch.testing.assert_close(changed, expected)
    handle.remove()
    torch.testing.assert_close(model(goal)[0], normal, rtol=0, atol=0)


def test_normal_mode_is_an_exact_noop_and_training_is_rejected_for_interventions():
    model = _Both3()
    assert install_cond_goal_ablation(model) is None
    with pytest.raises(ValueError, match="evaluation-only"):
        install_cond_goal_ablation(model, mode="off")
    model.eval()
    install_cond_goal_ablation(model, mode="off")
    model.train()
    with pytest.raises(RuntimeError, match="never run during training"):
        model(torch.ones(1, 6))


@pytest.mark.parametrize("mode,offset", [
    (False, None), ("typo", None), ("offset", None), ("normal", [.05, 0, 0]),
    ("off", [.05, 0, 0]), ("offset", [float("nan"), 0, 0]), ("offset", [1, 2]),
])
def test_invalid_cond_intervention_settings_are_not_silently_ignored(mode, offset):
    with pytest.raises(ValueError):
        cond_goal_settings(mode, offset)


def test_goal_free_cond_architecture_cannot_be_ablated_silently():
    model = _Both3().eval()
    model.config.architecture_label = "both_4"
    with pytest.raises(ValueError, match="Both3 XYZ"):
        install_cond_goal_ablation(model, mode="off")
