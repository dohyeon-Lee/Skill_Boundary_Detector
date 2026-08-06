from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/terminator_eval/src/run_terminator_eval.py"
)
SPEC = importlib.util.spec_from_file_location("run_terminator_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class _DummyTerminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))


def test_termination_display_latches_at_first_threshold_crossing() -> None:
    values = [0.1, 0.49, 0.72, 0.3, 0.91]

    latched = MODULE._latch_termination_trace(values, end_threshold=0.5)

    assert latched == [0.1, 0.49, 0.72, 0.72, 0.72]
    assert values == [0.1, 0.49, 0.72, 0.3, 0.91]


def test_termination_display_latch_ignores_missing_and_nonfinite_values() -> None:
    values = [None, float("nan"), 0.6, None, 0.2]

    latched = MODULE._latch_termination_trace(values, end_threshold=0.5)

    assert latched[0] is None
    assert torch.isnan(torch.tensor(latched[1]))
    assert latched[2:] == [0.6, 0.6, 0.6]


def test_signal_panel_uses_header_and_one_labeled_row_per_terminator() -> None:
    frame = np.zeros((70, 210, 3), dtype=np.uint8)

    annotated = MODULE._annotate_frames(
        [frame],
        progress=[0.4],
        termination=[0.6],
        display_traces=[
            {"label": "STATE30K_bias", "progress": [0.2], "termination": [0.7]}
        ],
        progress_threshold=0.95,
        end_threshold=0.5,
    )

    # camera + header + one display terminator + MAIN
    assert annotated[0].shape == (70 + 3 * 36, 210, 3)


def test_fsq_initial_can_be_attached_as_main_terminator(
    tmp_path: Path, monkeypatch
) -> None:
    raw_fsq = tmp_path / "FSQ.pt"
    raw_fsq.touch()
    action_policy = SimpleNamespace(model=nn.Linear(1, 1))
    wrapper = SimpleNamespace(
        policy=action_policy,
        terminator=None,
        advance_mode="gt",
    )
    context = {
        "policy": wrapper,
        "preprocessor": object(),
        "config": object(),
    }
    built_specs: list[dict] = []
    runtime_step_calls: list[dict] = []

    def build_context(spec, _cfg, _device):
        built_specs.append(spec)
        return context

    monkeypatch.setattr(MODULE, "_build_stage1_context", build_context)
    monkeypatch.setattr(
        MODULE,
        "_ensure_skill_runtime_steps",
        lambda *_args, **kwargs: runtime_step_calls.append(kwargs),
    )
    monkeypatch.setattr(MODULE, "build_fsq_terminator", lambda _path: _DummyTerminator())

    result = MODULE._build_context(
        {
            "label": "ACTION",
            "advance_mode": "external",
            "external_skill_model": str(raw_fsq),
            "external_skill_model_variant": "fsq_initial",
        },
        SimpleNamespace(),
        torch.device("cpu"),
    )

    assert result is context
    assert built_specs[0]["advance_mode"] == "gt"
    assert runtime_step_calls == [
        {"needs_predictor": False, "needs_terminator": True}
    ]
    assert isinstance(wrapper.terminator, MODULE.IndependentTerminator)
    assert wrapper.terminator.variant == "fsq_initial"
    assert wrapper.advance_mode == "external"
    assert not any(
        parameter.requires_grad for parameter in wrapper.terminator.module.parameters()
    )


def test_fsq_initial_loads_raw_fsq_without_checkpoint_overlay(
    tmp_path: Path, monkeypatch
) -> None:
    raw_fsq = tmp_path / "FSQ.pt"
    raw_fsq.touch()
    loaded_paths: list[str] = []

    def build(path):
        loaded_paths.append(str(path))
        return _DummyTerminator()

    monkeypatch.setattr(MODULE, "build_fsq_terminator", build)
    monkeypatch.setattr(
        MODULE,
        "_load_complete_terminator_parameters",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fsq_initial must not overlay auxiliary checkpoint tensors")
        ),
    )
    policy = SimpleNamespace(model=nn.Linear(1, 1))

    terminator = MODULE._load_display_terminator(
        policy,
        {
            "label": "FSQ_INIT",
            "variant": "fsq_initial",
            "path": str(raw_fsq),
        },
        tmp_path / "fallback_should_not_be_used.pt",
    )

    assert loaded_paths == [str(raw_fsq)]
    assert terminator.variant == "fsq_initial"
    assert terminator.module.training is False
    assert not any(parameter.requires_grad for parameter in terminator.module.parameters())
