import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


_FSQ_PATH = Path(__file__).resolve().parents[3] / "examples/libero/FSQ.py"
_SPEC = importlib.util.spec_from_file_location("libero_fsq_broadcast_test", _FSQ_PATH)
fsq_module = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = fsq_module
_SPEC.loader.exec_module(fsq_module)


def _stub(mode: str):
    model = SimpleNamespace(
        state_cond_mode=mode,
        working_dtype=torch.float32,
        skill_proj=nn.Linear(3, 4, bias=False),
        state_proj=nn.Linear(2, 4, bias=False),
        _time_cond=lambda time: torch.full((time.shape[0], 4), 2.0),
    )
    with torch.no_grad():
        model.skill_proj.weight.copy_(torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]))
        model.state_proj.weight.zero_()
    return model


def test_fsq_skill_routes_are_mutually_exclusive() -> None:
    z = torch.tensor([[0.2, 0.4, 0.6]])
    state = torch.zeros(1, 2)
    time = torch.tensor([0.5])
    projected = torch.tensor([[0.2, 0.4, 0.6, 1.2]])

    state_model = _stub("state")
    torch.testing.assert_close(
        fsq_module.VSAFlowExpert._action_prefix(state_model, z), projected.unsqueeze(1)
    )
    assert fsq_module.VSAFlowExpert._skill_broadcast(state_model, z) is None
    torch.testing.assert_close(
        fsq_module.VSAFlowExpert._expert_cond(state_model, time, state, z),
        torch.full((1, 4), 2.0),
    )

    adarms_model = _stub("state_skill")
    assert fsq_module.VSAFlowExpert._action_prefix(adarms_model, z) is None
    assert fsq_module.VSAFlowExpert._skill_broadcast(adarms_model, z) is None
    torch.testing.assert_close(
        fsq_module.VSAFlowExpert._expert_cond(adarms_model, time, state, z),
        torch.full((1, 4), 2.0) + projected,
    )

    broadcast_model = _stub("broadcast")
    assert fsq_module.VSAFlowExpert._action_prefix(broadcast_model, z) is None
    torch.testing.assert_close(
        fsq_module.VSAFlowExpert._skill_broadcast(broadcast_model, z), projected
    )
    torch.testing.assert_close(
        fsq_module.VSAFlowExpert._expert_cond(broadcast_model, time, state, z),
        torch.full((1, 4), 2.0),
    )
