"""Prove private termination learns without changing the VSA gradient path."""
import pytest
import torch

from lerobot.configs.uv_fullskill import DUAL_BOTTLENECK_NOPROPRIO_LABEL, DUAL_BOTTLENECK_TERM_DETACH_LABEL
from lerobot.policies.skill_expert.modeling_skill_expert import _allowed_pi05_missing_key
from test_pi05_vlm_vsa import tiny_vlm_model


@pytest.mark.parametrize('checkpoint,dtype', [(False, torch.float32), (True, torch.bfloat16)])
def test_private_readers_detach_all_layers_and_preserve_action(monkeypatch, checkpoint, dtype):
    torch.manual_seed(33)
    baseline = tiny_vlm_model(monkeypatch, DUAL_BOTTLENECK_NOPROPRIO_LABEL,
                              checkpoint=checkpoint, dtype=dtype)
    torch.manual_seed(33)
    model = tiny_vlm_model(monkeypatch, DUAL_BOTTLENECK_TERM_DETACH_LABEL,
                           checkpoint=checkpoint, dtype=dtype)
    branch = model.vlm_termination_bottleneck
    assert len(branch.readers) == len(model.cond_encoder.model.layers) == 18
    assert branch.queries.shape == (8, 256)
    shared = dict(baseline.named_parameters())
    for name, parameter in model.named_parameters():
        if name in shared:
            torch.testing.assert_close(parameter, shared[name], rtol=0, atol=0)
    images = [torch.rand(2, 3, 4, 4) for _ in range(2)]
    state, skill, noise = torch.zeros(2, 32), torch.tensor([1, 2]), torch.randn(2, 10, 32)
    goal = torch.tensor([[.2, -.3, 1., .1, .2, .3]] * 2)
    torch.manual_seed(4)
    expected = baseline(images, state, skill, noise, end_pose=goal)
    torch.manual_seed(4)
    actual = model(images, state, skill, noise, end_pose=goal)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    logits, memory = model.take_vlm_termination_outputs()
    assert not memory.requires_grad
    # Current and standalone (past-condition / deployed) routes agree. Neither
    # reads skill-start XYZ or padding; past route must not call any VSA reader.
    model._update_layerwise_latent = lambda *a: pytest.fail('Termination called VSA readers')
    changed = goal.clone()
    changed[:, 3:] += 100
    past = model.vlm_termination_from_image_tokens(memory, state, skill, changed)
    torch.testing.assert_close(logits, past, rtol=0, atol=0)
    (logits.square().mean() + past.square().mean()).backward()
    assert branch.queries.grad.abs().sum() > 0
    assert branch.memory_norm.weight.grad.abs().sum() > 0
    assert branch.head[-1].weight.grad.abs().sum() > 0
    assert all(reader.cross_attention.in_proj_weight is None for reader in branch.readers)
    assert all(reader.cross_attention.q_proj_weight.grad.abs().sum() > 0 for reader in branch.readers)
    assert all(p.grad is None for name, p in model.named_parameters()
               if not name.startswith('vlm_termination_bottleneck.'))
    assert _allowed_pi05_missing_key('model.vlm_termination_bottleneck.queries', model.config)
