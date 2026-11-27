"""Keep SkillVLA loss/update arithmetic and crop-dependent labels unchanged."""
from copy import deepcopy
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.utils.training_performance import (
    cached_training_target, constant_tensor, defer_scalar_metrics,
    materialize_scalar_metrics, reuse_training_targets,
)
from lerobot.policies.skill_expert import wrist_patch_alignment as alignment
from lerobot.policies.skill_expert import wrist_patch_target as geometry
from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy
from lerobot.policies.skill_groot.policy import GrootSkillExpertPolicy
from lerobot.policies.skill_groot.model import CameraMemory


def legacy_patch(logits, label, valid, grid, sigma):
    logits = logits.float()
    if int(valid.sum()) == 0:
        return logits.sum() * 0
    logits, label = logits[valid], label[valid]
    return -(geometry.soft_targets(label, grid=grid, sigma=sigma)
             * logits.log_softmax(-1)).sum(-1).mean() / math.log(grid * grid)


def legacy_xyz(batch, prediction, config):
    target = batch['skill_end_state'][:, :3].float()
    valid = batch['skill_end_state_valid'].bool() & torch.isfinite(target).all(-1)
    q01 = prediction.new_tensor(config.chunk_end_state_q01[:3], dtype=torch.float32)
    q99 = prediction.new_tensor(config.chunk_end_state_q99[:3], dtype=torch.float32)
    target = torch.where(valid[:, None], target, q01)
    target = 2 * (target - q01) / (q99 - q01).clamp_min(1e-6) - 1
    return ((prediction.float() - target).square().mean(-1) * valid).sum() / valid.sum().clamp_min(1)


class Heads(nn.Module):
    def __init__(self, grid):
        super().__init__()
        self.values = nn.ParameterList([nn.Parameter(torch.randn(4, dim))
                                       for dim in [grid * grid, grid * grid, 3] * 2])
        self._camera_memory = None

    @property
    def _aux(self):
        return [tuple(self.values[i:i + 3]) for i in (0, 3)]

    def predict_training_dual_auxiliaries(self, **kwargs):
        return self._aux


def make_policy(backend, grid):
    cls = SkillExpertPolicy if backend == 'pi05' else GrootSkillExpertPolicy
    policy = object.__new__(cls)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        architecture_label='test_dual', bimanual_spatial_auxiliary=False,
        uses_dual_bottleneck=True, trains_agent_patch_alignment=True,
        trains_agent_end_state_prediction=True, uses_triple_auxiliary_prediction=True,
        uses_gemma_xyz_readout=False, dino_image_size=224,
        trains_chunk_end_patch_alignment=False, wrist_patch_align_target_sigma=.7,
        wrist_patch_align_loss_weight=.1, agent_end_state_loss_weight=.1,
        predicts_xyz_only=True, chunk_end_state_q01=[-.5] * 8,
        chunk_end_state_q99=[1.5] * 8,
    )
    policy.model = Heads(grid)
    return policy


def batch(device='cpu'):
    raw = torch.tensor([[0., 0., .3, math.pi, 0., 0., .01, .01]] * 4)
    target = raw.clone()
    target[:, :3] += torch.tensor([[.1, 0., -.25], [0., .1, -.25], [.1, .1, -.25], [0., 0., .1]])
    target[2, 0] = float('nan')  # invalid labels must still contribute no NaN gradients
    data = {'observation.state': raw.clone(), 'skill_decoder_state': raw,
            'skill_end_state': target, 'skill_end_state_valid': torch.tensor([True, True, False, True]),
            'skill_focus_uv': torch.tensor([[.1, -.2], [-.3, .5], [1., 0.], [.7, -.6]]),
            'skill_focus_valid': torch.ones(4, dtype=torch.bool)}
    return {key: value.to(device) for key, value in data.items()}


def legacy_visual(policy, data, backend):
    cfg = policy.config
    grid = math.isqrt(policy.model._aux[0][0].shape[-1])
    raw = data['skill_decoder_state']
    wrist_cell, pixel, wrist_valid = geometry.patch_labels(
        raw[:, :3], raw[:, 3:6], data['skill_end_state'][:, :3],
        geometry.camera_for_batch(data), grid=grid, height=224, width=224)
    top = data['skill_focus_uv']
    goal_valid = data['skill_end_state_valid'].bool()
    if backend != 'pi05':
        top = policy.model._camera_memory.uv(top)
        wrist = policy.model._camera_memory.uv(2 * pixel / 224 - 1, camera=1)
    total = 0.
    for top_logits, wrist_logits, xyz in policy.model._aux:
        valid = goal_valid & torch.isfinite(top).all(-1) & (top.abs() <= 1).all(-1)
        if backend == 'pi05':
            xy = torch.floor((top + 1.) * .5 * grid).long().clamp(0, grid - 1)
            term = cfg.wrist_patch_align_loss_weight * (
                legacy_patch(top_logits, xy[:, 1] * grid + xy[:, 0], valid, grid, .7)
                + legacy_patch(wrist_logits, wrist_cell, wrist_valid, grid, .7))
            term = term + cfg.agent_end_state_loss_weight * legacy_xyz(data, xyz, cfg)
        else:
            term = xyz.sum() * 0
            for logits, uv, usable in [(top_logits, top, goal_valid),
                                       (wrist_logits, wrist, wrist_valid & goal_valid)]:
                usable = usable & torch.isfinite(uv).all(-1) & (uv.abs() <= 1).all(-1)
                xy = ((uv.nan_to_num().clamp(-1, 1) + 1) * .5 * grid).floor().long().clamp(0, grid - 1)
                term = term + cfg.wrist_patch_align_loss_weight * legacy_patch(
                    logits, xy[:, 1] * grid + xy[:, 0], usable, grid, .7)
            term = term + cfg.agent_end_state_loss_weight * legacy_xyz(data, xyz, cfg)
        total = total + .5 * term
    return total


@pytest.mark.parametrize('backend', ['pi05', 'groot_integrate', 'groot_seperate'])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='GPU equivalence is also run inside an existing allocation'))])
def test_both_banks_preserve_losses_gradients_updates_rng_and_new_crops(backend, device, monkeypatch):
    torch.manual_seed(512)
    grid = 14 if backend == 'pi05' else 7
    old = make_policy(backend, grid).to(device)
    new = deepcopy(old)
    optimizers = [torch.optim.AdamW(p.parameters(), lr=.001) for p in (old, new)]
    real_projection = geometry.patch_labels
    from lerobot.policies.skill_expert import modeling_skill_expert as pi
    from lerobot.policies.skill_groot import policy as groot
    calls = []
    def projection(*args, **kwargs):
        calls.append(1)
        return real_projection(*args, **kwargs)
    monkeypatch.setattr(pi, 'patch_labels', projection)
    monkeypatch.setattr(groot, 'patch_labels', projection)
    data = batch(device)
    original_uv = data['skill_focus_uv'].clone()
    for step in range(3):
        # Crop/UV changes on EVERY forward, including when the batch object is reused.
        data['skill_focus_uv'].copy_(original_uv + step * .02)
        offsets = torch.tensor([[step * 10., 26. - step * 10.]] * 4, device=device)
        for p in (old, new):
            p.model._camera_memory = CameraMemory([], [offsets, offsets.flip(-1)])
        for opt in optimizers:
            opt.zero_grad()
        before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state().clone() if device == 'cuda' else None
        expected = legacy_visual(old, data, backend)
        calls.clear()
        with reuse_training_targets(), defer_scalar_metrics():
            actual, metrics = (new._wrist_patch_alignment_loss(data, top_k=1)
                               if backend == 'pi05' else new._visual_loss(data))
        assert len(calls) == 1  # one wrist projection for BOTH banks
        assert torch.equal(before, torch.get_rng_state())
        if cuda_before is not None:
            assert torch.equal(cuda_before, torch.cuda.get_rng_state())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert all(not value.requires_grad for value in metrics.values() if torch.is_tensor(value))
        expected.backward(); actual.backward()
        for a, b in zip(old.parameters(), new.parameters(), strict=True):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        for opt in optimizers:
            opt.step()
        for a, b in zip(old.parameters(), new.parameters(), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_forward_cache_expires_on_exit_exception_and_nested_forward():
    owner = {}
    calls = []
    def build():
        calls.append(1)
        return len(calls)
    with pytest.raises(RuntimeError), reuse_training_targets():
        assert cached_training_target(owner, 'x', build) == 1
        assert cached_training_target(owner, 'x', build) == 1
        with reuse_training_targets():
            assert cached_training_target(owner, 'x', build) == 2
        assert cached_training_target(owner, 'x', build) == 1
        raise RuntimeError()
    assert cached_training_target(owner, 'x', build) == 3
    with reuse_training_targets():
        assert cached_training_target(owner, 'x', build) == 4


def test_constant_cache_tracks_values_dtype_and_camera_and_preserves_rng(monkeypatch):
    state = torch.get_rng_state().clone()
    a = constant_tensor([1., 2.], device='cpu')
    assert a is constant_tensor((1., 2.), device=torch.device('cpu'))
    assert a is not constant_tensor((1., 3.), device='cpu')
    assert a is not constant_tensor((1., 2.), device='cpu', dtype=torch.float64)
    data = batch(); raw = data['skill_decoder_state']
    args = (raw[:, :3], raw[:, 3:6], data['skill_end_state'][:, :3])
    for camera in [geometry.WristCamera(), geometry.WristCamera(position=(.1, .01, -.09))]:
        cached = geometry.project_into_wrist(*args, camera, height=224, width=224)
        with monkeypatch.context() as patch:
            patch.setattr(geometry, 'constant_tensor', lambda values, **kwargs: torch.tensor(values, **kwargs))
            patch.setattr(geometry, 'constant_eye', lambda size, device, dtype: torch.eye(size, device=device, dtype=dtype))
            expected = geometry.project_into_wrist(*args, camera, height=224, width=224)
        for x, y in zip(cached, expected, strict=True):
            # Missing targets project to NaN in both paths; validity masking
            # (covered by the loss/update test) excludes those rows.
            torch.testing.assert_close(x, y, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize('valid', [torch.tensor([False] * 4), torch.tensor([True, False, True, False])])
def test_patch_metrics_preserved_and_targets_built_once(valid, monkeypatch):
    label = torch.tensor([1, -1, 3, -1])
    logits = torch.randn(4, 49, requires_grad=True)
    expected = legacy_patch(logits, label, valid, 7, .7)
    plain, plain_metrics = alignment.wrist_patch_alignment_loss(logits, label, valid, grid=7, sigma=.7)
    builds = []
    original = alignment.soft_targets
    def target(*args, **kwargs):
        builds.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(alignment, 'soft_targets', target)
    with reuse_training_targets(), defer_scalar_metrics():
        a, first = alignment.wrist_patch_alignment_loss(logits, label, valid, grid=7, sigma=.7)
        b, second = alignment.wrist_patch_alignment_loss(logits, label, valid, grid=7, sigma=.7)
    assert len(builds) == int(valid.any())
    assert materialize_scalar_metrics(first) == plain_metrics == materialize_scalar_metrics(second)
    for value in (plain, a, b):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
