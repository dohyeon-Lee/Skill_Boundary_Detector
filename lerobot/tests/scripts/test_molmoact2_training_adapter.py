"""Run with .venv_molmoact2_lerobot: real upstream accumulation and scheduler."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/libero/configs/train_molmoact2/PT/src"))
try:
    from lerobot.configs.accelerator import GradientAccumulationConfig
except ImportError:
    pytest.skip("Requires isolated HF LeRobot Molmo environment", allow_module_level=True)
from training_adapter import UpdateClockSchedulerConfig


def test_native_action_normalization_keeps_gripper_sign_and_magnitude():
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.lerobot_types import TransitionKey as K
    from lerobot.policies.molmoact2 import processor_molmoact2 as pg
    features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    stats = {"action": {
        "q01": torch.tensor([-.4] * 6 + [-1.]), "q99": torch.tensor([.6] * 6 + [1.]),
        "mask": torch.tensor([True] * 6 + [False]),
    }}
    kwargs = dict(features=features, norm_map={FeatureType.ACTION: NormalizationMode.QUANTILES}, stats=stats)
    normalizer = pg.MolmoAct2MaskedNormalizerProcessorStep(**kwargs)
    unnormalizer = pg.MolmoAct2MaskedUnnormalizerProcessorStep(**kwargs)
    action = torch.zeros(1, 6, 7)
    action[0, :, -1] = torch.tensor([-1., 1., -.25, .25, 0., .8])
    normalized = normalizer({K.ACTION: action})
    clamped = pg.MolmoAct2ClampActionProcessorStep()(normalized)
    decoded = unnormalizer(clamped)[K.ACTION]
    torch.testing.assert_close(decoded, action, rtol=0, atol=1e-7)
    torch.testing.assert_close(decoded[..., -1], action[..., -1], rtol=0, atol=0)


def test_warmup_constant_uses_optimizer_updates_and_roundtrips():
    import draccus
    from lerobot.optim.schedulers import LRSchedulerConfig
    config = UpdateClockSchedulerConfig(num_warmup_steps=2, accumulation_steps=4)
    assert [config.factor(i) for i in range(12)] == pytest.approx([1/3]*4 + [2/3]*4 + [1.0]*4)
    encoded = draccus.encode(config, LRSchedulerConfig)
    assert draccus.decode(LRSchedulerConfig, encoded) == config
    parameter = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.SGD([parameter], lr=0.1)
    sched = config.build(opt, 400)
    for _ in range(8):
        opt.step()
        sched.step()
    state = sched.state_dict()
    restored = config.build(opt, 400)
    restored.load_state_dict(state)
    assert restored.last_epoch == 8 and restored.get_last_lr() == sched.get_last_lr()


def test_native_accumulation_updates_once_per_global_batch():
    from accelerate import Accelerator
    from lerobot.scripts.lerobot_train import update_policy
    from lerobot.utils.logging_utils import AverageMeter, MetricsTracker

    class ToyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, batch):
            return ((self.w * batch - 0.5) ** 2).mean(), {}

    accelerator = Accelerator(cpu=True, gradient_accumulation_plugin=GradientAccumulationConfig(steps=4).build_plugin(),
                              step_scheduler_with_optimizer=False)
    policy = ToyPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    schedule = UpdateClockSchedulerConfig(num_warmup_steps=0, accumulation_steps=4).build(optimizer, 8)
    policy, optimizer, schedule = accelerator.prepare(policy, optimizer, schedule)
    meters = {name: AverageMeter(name, ":.3f") for name in ("loss", "grad_norm", "lr", "update_s")}
    tracker = MetricsTracker(1, 100, 1, meters)
    initial = policy.w.item()
    for index in range(8):
        before = policy.w.item()
        update_policy(tracker, policy, torch.ones(1), optimizer, 0, accelerator, schedule)
        if (index + 1) % 4:
            assert policy.w.item() == before
    # Two full-batch SGD updates: w: 1 -> .9 -> .82. Loss scaling is native.
    assert initial == 1.0 and policy.w.item() == pytest.approx(0.82)
    assert schedule.scheduler.last_epoch == 8


def test_grounded_view_is_spawn_pickle_safe():
    import pickle
    from grounding_adapter import GroundedMolmoDataset
    assert pickle.loads(pickle.dumps(GroundedMolmoDataset)) is GroundedMolmoDataset


def test_native_optimizer_roundtrip_preserves_bf16_compensation(tmp_path):
    from lerobot.optim.optimizers import save_optimizer_state, load_optimizer_state
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2AdamWConfig
    parameter = torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16))
    config = MolmoAct2AdamWConfig(lr=2.5e-5)
    optimizer = config.build([parameter])
    for _ in range(3):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad()
    save_optimizer_state(optimizer, tmp_path)
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = config.build([restored_parameter])
    load_optimizer_state(restored, tmp_path)
    assert torch.equal(optimizer.state[parameter]["compensation"], restored.state[restored_parameter]["compensation"])
    for param, opt in ((parameter, optimizer), (restored_parameter, restored)):
        param.grad = torch.ones_like(param)
        opt.step()
    assert torch.equal(parameter, restored_parameter)
    assert torch.equal(optimizer.state[parameter]["exp_avg"], restored.state[restored_parameter]["exp_avg"])
