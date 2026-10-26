"""Run in .venv_molmoact2_lerobot; CUDA cases require a Slurm GPU allocation."""
import copy
from dataclasses import asdict
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/libero/configs/train_molmoact2/PT/src"))
try:
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2AdamW, MolmoAct2AdamWConfig
except ImportError:
    pytest.skip("Requires the isolated MolmoAct2 environment", allow_module_level=True)
from optimizer_offload import CPUStateMolmoAct2AdamW, CPUStateMolmoAct2AdamWConfig, offload_optimizer_config


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture
def isolated_accelerator_state():
    from accelerate.state import AcceleratorState, GradientState
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()
    yield
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()


def parameters(device):
    return [torch.nn.Parameter(torch.linspace(-1, 1, 24, device=device, dtype=dtype).reshape(4, 6))
            for dtype in (torch.bfloat16, torch.float32, torch.float64)]


def optimizer(cls, params, *, amsgrad=False):
    return cls([{"params": params[:2], "lr": 2e-3}, {"params": params[2:], "lr": 3e-3}],
               group_grad_clip_norm=.35, betas=(.9, .95), eps=1e-6, weight_decay=.1,
               amsgrad=amsgrad, foreach=False)


def update(params, opt, step):
    for index, param in enumerate(params):
        param.grad = None if index == 2 and step % 2 else torch.full_like(param, .1 * (step + 1) * (index + 1))
    opt.step()
    opt.zero_grad(set_to_none=True)


def assert_match(left, right, opt_left, opt_right):
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert opt_left.state[a].keys() == opt_right.state[b].keys()
        for key in opt_left.state[a]:
            torch.testing.assert_close(opt_left.state[a][key].cpu(), opt_right.state[b][key].cpu(), rtol=0, atol=0)
            assert opt_right.state[b][key].device.type == "cpu"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("amsgrad", [False, True])
def test_updates_match_native_mixed_precision_groups_and_clipping(device, amsgrad):
    baseline = parameters(device)
    offloaded = [torch.nn.Parameter(p.detach().clone()) for p in baseline]
    native = optimizer(MolmoAct2AdamW, baseline, amsgrad=amsgrad)
    cpu = optimizer(CPUStateMolmoAct2AdamW, offloaded, amsgrad=amsgrad)
    original_groups = cpu.param_groups
    for step in range(6):
        update(baseline, native, step)
        update(offloaded, cpu, step)
        assert cpu.param_groups is original_groups
        assert_match(baseline, offloaded, native, cpu)


@pytest.mark.parametrize("device", DEVICES)
def test_native_serialization_and_resume_keeps_cpu_states(tmp_path, device):
    from lerobot.optim.optimizers import save_optimizer_state, load_optimizer_state

    params = parameters(device)
    opt = optimizer(CPUStateMolmoAct2AdamW, params)
    for step in range(3):
        update(params, opt, step)
    save_optimizer_state(opt, tmp_path)
    resumed_params = [torch.nn.Parameter(p.detach().clone()) for p in params]
    resumed = optimizer(CPUStateMolmoAct2AdamW, resumed_params)
    load_optimizer_state(resumed, tmp_path)
    assert_match(params, resumed_params, opt, resumed)
    assert resumed.param_groups[0]["params"][0] is resumed_params[0]
    for step in range(3, 6):
        update(params, opt, step)
        update(resumed_params, resumed, step)
        assert_match(params, resumed_params, opt, resumed)


def test_optimizer_probe_config_roundtrips_and_preserves_hyperparameters():
    import draccus
    from lerobot.optim import OptimizerConfig

    native = MolmoAct2AdamWConfig(lr=2.5e-5, group_grad_clip_norm=.7)
    offload = offload_optimizer_config(native)
    assert asdict(native) == asdict(offload)
    assert isinstance(offload, CPUStateMolmoAct2AdamWConfig)
    assert offload_optimizer_config(offload) is offload
    restored = draccus.decode(OptimizerConfig, draccus.encode(offload, OptimizerConfig))
    assert restored == offload


@pytest.mark.parametrize("device", DEVICES)
def test_nonfinite_gradient_skips_state_initialization_and_updates(device):
    param = torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16, device=device))
    opt = CPUStateMolmoAct2AdamW([param], group_grad_clip_norm=1.)
    param.grad = torch.full_like(param, float("inf"))
    opt.step()
    assert not opt.state and param.grad is None
    torch.testing.assert_close(param, torch.ones_like(param), rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_accumulation_and_scheduler_advance_once_per_effective_batch(device, isolated_accelerator_state):
    from accelerate import Accelerator
    from lerobot.configs.accelerator import GradientAccumulationConfig

    accelerator = Accelerator(cpu=device == "cpu", gradient_accumulation_plugin=GradientAccumulationConfig(steps=4).build_plugin(),
                              step_scheduler_with_optimizer=False)
    model = torch.nn.Linear(4, 1, bias=False).to(device)
    opt = CPUStateMolmoAct2AdamW(model.parameters(), lr=.01, group_grad_clip_norm=1.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)
    model, opt, scheduler = accelerator.prepare(model, opt, scheduler)
    initial = model.weight.detach().clone()
    for index in range(8):
        before = model.weight.detach().clone()
        with accelerator.accumulate(model):
            accelerator.backward(model(torch.ones(1, 4, device=device)).square().mean())
            opt.step(); opt.zero_grad(); scheduler.step()
        if (index + 1) % 4:
            torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
    assert not torch.equal(initial, model.weight)
    inner = opt.optimizer
    assert all(state["step"].item() == 2 for state in inner.state.values())
    assert all(v.device.type == "cpu" for state in inner.state.values() for v in state.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a Slurm CUDA allocation")
def test_cuda_state_residency_and_reload_peak_are_bounded():
    # More than one tensor matters: a whole-model state upload on resume must fail this check.
    params = [torch.nn.Parameter(torch.ones(1024, 1024, dtype=torch.bfloat16, device="cuda")) for _ in range(8)]
    opt = CPUStateMolmoAct2AdamW(params, group_grad_clip_norm=1.)
    for step in range(2):
        update(params, opt, step)
        assert all(v.device.type == "cpu" for state in opt.state.values() for v in state.values())
    saved = copy.deepcopy(opt.state_dict())
    state_bytes = sum(v.numel() * v.element_size() for state in saved["state"].values() for v in state.values())
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    opt.load_state_dict(saved)
    torch.cuda.synchronize()
    peak_extra = torch.cuda.max_memory_allocated() - baseline
    assert peak_extra < state_bytes / 4, (peak_extra, state_bytes)
    assert all(v.device.type == "cpu" for state in opt.state.values() for v in state.values())
