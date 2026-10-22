"""Run with .venv_groot_lerobot: real upstream accumulation and scheduler."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/libero/configs/train_groot/PT/src"))
try:
    from lerobot.configs.accelerator import GradientAccumulationConfig
except ImportError:
    pytest.skip("Requires isolated HF LeRobot Groot environment", allow_module_level=True)
import importlib.util
_spec = importlib.util.spec_from_file_location("sbd_groot_training_test", Path(__file__).resolve().parents[2] / "examples/libero/configs/train_groot/PT/src/training_adapter.py")
_hooks = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _hooks
_spec.loader.exec_module(_hooks)
UpdateClockSchedulerConfig = _hooks.UpdateClockSchedulerConfig


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
    from grounding_adapter import GroundedGrootDataset
    assert pickle.loads(pickle.dumps(GroundedGrootDataset)) is GroundedGrootDataset


def test_checkpointing_preserves_dropout_outputs_and_gradients():
    import copy
    from types import SimpleNamespace
    blocks = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(.2)) for _ in range(2)])
    reference = copy.deepcopy(blocks)
    policy = SimpleNamespace(_groot_model=SimpleNamespace(
        action_head=SimpleNamespace(model=SimpleNamespace(transformer_blocks=blocks)),
        backbone=SimpleNamespace(model=None)))
    assert _hooks.checkpoint_action_blocks(policy) == 2
    results = []
    for branch in (reference, blocks):
        torch.manual_seed(17)
        value = torch.ones(2, 4, requires_grad=True)
        for block in branch:
            value = block(value)
        value.square().sum().backward()
        results.append((value.detach(), [p.grad.clone() for p in branch.parameters()]))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for left, right in zip(results[0][1], results[1][1], strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_native_optimizer_roundtrip(tmp_path):
    from lerobot.optim.optimizers import save_optimizer_state, load_optimizer_state
    from lerobot.policies.groot.configuration_groot import GrootConfig
    config = GrootConfig(device="cpu").get_optimizer_preset()
    parameter = torch.nn.Parameter(torch.ones(8))
    optimizer = config.build([parameter])
    for _ in range(3):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad()
    save_optimizer_state(optimizer, tmp_path)
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = config.build([restored_parameter])
    load_optimizer_state(restored, tmp_path)
    for param, opt in ((parameter, optimizer), (restored_parameter, restored)):
        param.grad = torch.ones_like(param)
        opt.step()
    torch.testing.assert_close(parameter, restored_parameter, rtol=0, atol=0)


def test_native_processor_statistics_masks_and_save_reload(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.lerobot_types import TransitionKey as K
    from lerobot.policies.groot.configuration_groot import GrootConfig
    from lerobot.policies.groot import processor_groot as pg
    from processor_adapter import make_dataset_processors

    def stats(dim):
        return {"q01": torch.zeros(dim), "q99": torch.ones(dim), "min": torch.full((dim,), -100.),
                "max": torch.full((dim,), 100.), "mean": torch.zeros(dim), "std": torch.ones(dim), "count": torch.tensor([10])}
    config = GrootConfig(device="cpu", embodiment_tag="libero_sim", chunk_size=10, n_action_steps=10,
                         base_model_path=str(tmp_path / "base"), action_decode_transform="none",
                         output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))})
    monkeypatch.setattr(pg, "_load_n1_7_checkpoint_processor_assets", lambda _: None)
    action_stats = stats(7)
    action_stats["q01"][-1] = -1
    action_stats["min"][-1], action_stats["max"][-1] = -1, 1
    pre, post = make_dataset_processors(config, {"observation.state": stats(8), "action": action_stats},
                                        SimpleNamespace(), horizon=10)
    pack = next(step for step in pre.steps if isinstance(step, pg.GrootN17PackInputsStep))
    action = torch.full((1, 10, 7), .25)
    action[0, :, -1] = torch.tensor([-1., 1.] * 5)  # our data: open=-1, close=+1
    def transition():
        return {K.OBSERVATION: {"observation.state": torch.full((1, 8), .25),
                               "observation.images.image": torch.zeros(1, 3, 16, 16),
                               "observation.images.wrist_image": torch.ones(1, 3, 16, 16)},
                K.ACTION: action.clone(),
                K.COMPLEMENTARY_DATA: {"task": ["test"], "action_is_pad": torch.tensor([[False] * 9 + [True]])}}
    with torch.no_grad():
        packed = pack(transition())
    assert packed[K.OBSERVATION]["state"].shape == (1, 1, 132)
    torch.testing.assert_close(packed[K.OBSERVATION]["state"][..., :8], torch.full((1, 1, 8), -.5))
    video = packed[K.OBSERVATION]["video"]
    assert not video[:, :, 0].any() and (video[:, :, 1] == 255).all()
    assert packed[K.ACTION].shape == (1, 40, 132)
    mask = packed[K.COMPLEMENTARY_DATA]["action_mask"]
    assert mask.sum() == 9 * 7 and not mask[:, 9:].any() and not mask[..., 7:].any()
    decoded = post(packed[K.ACTION])
    torch.testing.assert_close(decoded[:, :9], action[:, :9])
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    loaded_pre, loaded_post = pg.make_groot_pre_post_processors_from_pretrained(config, str(tmp_path))
    reloaded = next(step for step in loaded_pre.steps if isinstance(step, pg.GrootN17PackInputsStep))
    with torch.no_grad():
        after = reloaded(transition())
    torch.testing.assert_close(packed[K.ACTION], after[K.ACTION], rtol=0, atol=0)
    torch.testing.assert_close(loaded_post(after[K.ACTION]), decoded, rtol=0, atol=0)

    # Existing checkpoints saved the wrong transform. Repair config in memory
    # from saved statistics, exactly as the isolated evaluation worker does.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/libero/configs/src/baseline_eval"))
    from groot_action_contract import groot_action_decode_transform
    post.steps[0].action_decode_transform = "libero"
    post.save_pretrained(tmp_path)
    assert groot_action_decode_transform(tmp_path) == "none"
    _, repaired_post = pg.make_groot_pre_post_processors_from_pretrained(config, str(tmp_path))
    torch.testing.assert_close(repaired_post(after[K.ACTION]), decoded, rtol=0, atol=0)


def test_actual_native_cli_accepts_generated_training_arguments(tmp_path):
    import draccus
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.groot.configuration_groot import GrootConfig
    import groot_config
    settings = groot_config.build_settings({"stage": "pt", "project_root": str(tmp_path)})
    args = [arg.replace('--policy.device="cuda"', '--policy.device="cpu"')
            for arg in groot_config.trainer_args(settings)]
    config = draccus.parse(TrainPipelineConfig, args=args)
    assert config.policy.type == "groot"
    assert isinstance(config.policy, GrootConfig)
    assert config.policy.dtype == torch.float32 and config.policy.use_bf16
    assert config.policy.base_model_path == settings["base_checkpoint"] and not config.policy.use_relative_actions
    assert config.policy.action_decode_transform is None
    assert config.policy.chunk_size == config.policy.n_action_steps == 10
    optimizer = config.policy.get_optimizer_preset()
    assert optimizer.lr == settings["lr"] == 1e-4
    assert optimizer.betas == (0.9, 0.999)
    assert optimizer.eps == 1e-8 and optimizer.weight_decay == 1e-5
    assert config.accelerator.gradient_accumulation.steps == 16
    assert config.steps == 1600000


def test_resume_keeps_saved_statistics_and_training_mode(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import groot_config
    from lerobot.policies.groot.configuration_groot import GrootConfig
    settings = groot_config.build_settings({"stage": "pt", "project_root": str(tmp_path)})
    monkeypatch.setattr(groot_config, "latest_checkpoint", lambda _: tmp_path / "checkpoint")
    received = {}
    def original_processors(*args, **kwargs):
        received.update(kwargs)
        return "saved processors"
    noop = lambda *args, **kwargs: None
    native = SimpleNamespace(make_train_eval_datasets=noop, make_accelerator=noop,
                             update_last_checkpoint=noop, make_pre_post_processors=original_processors,
                             make_policy=noop, WandBLogger=object)
    original_preset = GrootConfig.get_scheduler_preset
    try:
        _hooks.install(native, settings, noop)
        meta = object()
        config = SimpleNamespace(_runtime_dataset_meta=meta)
        assert native.make_pre_post_processors(policy_cfg=config, pretrained_path="saved") == "saved processors"
        assert received["dataset_meta"] is meta
        assert "dataset_stats" not in received
    finally:
        GrootConfig.get_scheduler_preset = original_preset
