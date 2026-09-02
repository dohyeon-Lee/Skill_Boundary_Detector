from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
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


class _DummyPolicyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.register_buffer("_fsq_strides", torch.ones(1, dtype=torch.long))

    def _code_to_zq(self, codes: torch.Tensor) -> torch.Tensor:
        return torch.ones(codes.shape[0], 3, device=codes.device)


class _RecordingPrevActionTerminator(nn.Module):
    state_dim = 2
    context_mode = "prev_action"
    termination_only = True

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.raw_actions: list[torch.Tensor] = []
        self.contexts: list[torch.Tensor] = []

    def normalize_previous_action(self, action: torch.Tensor) -> torch.Tensor:
        self.raw_actions.append(action.detach().clone())
        return action[..., :2] * 2.0

    def forward(self, z_q, context, image, wrist):
        del z_q, image, wrist
        self.contexts.append(context.detach().clone())
        return torch.zeros(context.shape[0]), torch.zeros(context.shape[0])


def test_restore_state_synchronizes_controller_before_first_action() -> None:
    events: list[object] = []

    class _Controller:
        use_delta = False

        def update(self, *, force: bool = False) -> None:
            events.append(("controller_update", force))

        def reset_goal(self) -> None:
            events.append("controller_reset_goal")

    class _Env:
        def __init__(self) -> None:
            self.robots = [SimpleNamespace(controller=_Controller())]

        def reset(self) -> None:
            events.append("env_reset")

        def set_init_state(self, state: np.ndarray):
            events.append(("set_init_state", state.copy()))
            return "restored_obs"

    env = _Env()
    base_env = SimpleNamespace(_env=env)
    state = np.array([1.0, 2.0], dtype=np.float32)

    raw_obs = MODULE._restore_state(base_env, state)

    assert raw_obs == "restored_obs"
    assert env.robots[0].controller.use_delta is True
    assert events[0] == "env_reset"
    assert events[1][0] == "set_init_state"
    assert events[1][1].dtype == np.float64
    assert np.array_equal(events[1][1], state)
    assert events[2:] == [("controller_update", True), "controller_reset_goal"]


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


def test_signal_panel_uses_one_row_per_terminator() -> None:
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

    # camera + one display terminator + MAIN
    assert annotated[0].shape == (70 + 2 * 36, 210, 3)


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


class _RecordingStateRNN(nn.Module):
    state_dim = 2
    termination_only = False

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    def step_outputs(self, z_q, state, hidden=None):
        del z_q
        self.calls.append(
            (
                state.detach().clone(),
                None if hidden is None else hidden.detach().clone(),
            )
        )
        next_hidden = torch.ones(1, state.shape[0], 1, device=state.device)
        if hidden is not None:
            next_hidden = hidden + 1
        progress = next_hidden[0, :, 0] / 10
        logits = torch.zeros(state.shape[0], device=state.device)
        return progress, logits, next_hidden


def test_state_rnn_uses_current_state_and_carries_then_resets_hidden() -> None:
    module = _RecordingStateRNN()
    adapter = MODULE.IndependentTerminator(
        SimpleNamespace(model=_DummyPolicyModel()),
        module,
        "state_rnn",
    )
    codes = torch.tensor([2])
    # A singleton history dimension is accepted, but only its current/last
    # state is sent to the online RNN step.
    state = torch.tensor([[[1.0, 2.0]]])
    image = torch.zeros(1, 3, 4, 4)

    adapter.terminate(codes, state, image, image)
    adapter.terminate(codes, state + 1, image, image)

    assert module.calls[0][0].shape == (1, 2)
    assert module.calls[0][1] is None
    torch.testing.assert_close(module.calls[1][1], torch.ones(1, 1, 1))
    adapter.reset()
    adapter.terminate(codes, state + 2, image, image)
    assert module.calls[2][1] is None


def test_state_image_prev_action_adapter_normalizes_previous_action() -> None:
    module = _RecordingPrevActionTerminator()
    adapter = MODULE.IndependentTerminator(
        SimpleNamespace(model=_DummyPolicyModel()), module, "state_image"
    )
    image = torch.zeros(1, 3, 4, 4)

    adapter.terminate(
        torch.tensor([2]),
        torch.full((1, 8), 99.0),
        image,
        image,
        previous_action=torch.tensor([[1.0, 3.0]]),
    )

    torch.testing.assert_close(module.raw_actions[0], torch.tensor([[1.0, 3.0]]))
    torch.testing.assert_close(module.contexts[0], torch.tensor([[2.0, 6.0]]))


def test_state_image_loader_rebuilds_saved_auxiliary_contract(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = tmp_path / "state_image"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "skill_aux",
                "train_terminator": True,
                "skill_fsq_levels": [3, 3, 3],
                "terminator_context": "prev_action",
                "terminator_arch": "fusion",
                "terminator_vision_backbone": "resnet",
                "terminator_freeze_vision_encoder": False,
                "terminator_termination_only": True,
            }
        )
    )
    built: list[tuple] = []
    module = _RecordingPrevActionTerminator()

    def build(path, termination_only=None, **kwargs):
        built.append((path, termination_only, kwargs))
        return module

    monkeypatch.setattr(MODULE, "build_trainable_fsq_terminator", build)
    monkeypatch.setattr(
        MODULE, "_load_complete_terminator_parameters", lambda *_args, **_kwargs: 1
    )
    adapter = MODULE._load_display_terminator(
        SimpleNamespace(model=_DummyPolicyModel()),
        {"variant": "state_image", "path": str(checkpoint)},
        tmp_path / "FSQ.pt",
    )

    assert adapter.module is module
    assert adapter.context_mode == "prev_action"
    assert built == [
        (
            tmp_path / "FSQ.pt",
            True,
            {
                    "context": "prev_action",
                    "cameras": "both",
                    "default_arch": "fusion",
                "vision_backbone": "resnet",
                "freeze_vision_encoder": False,
            },
        )
    ]


def test_gt_rollout_seeds_and_advances_previous_action(monkeypatch) -> None:
    seen: list[np.ndarray | None] = []

    def query(**kwargs):
        previous = kwargs.get("previous_action")
        seen.append(None if previous is None else np.asarray(previous).copy())
        return {}, np.zeros(2, dtype=np.float32), 0.0, 0.0, []

    class _Env:
        def __init__(self) -> None:
            self._env = self

        @staticmethod
        def step(_action):
            return object(), 0.0, False, {}

    monkeypatch.setattr(MODULE, "_query_terminator", query)
    monkeypatch.setattr(MODULE, "_restore_state", lambda *_args: object())
    monkeypatch.setattr(
        MODULE, "_render", lambda *_args: np.zeros((2, 2, 3), dtype=np.uint8)
    )
    result = MODULE._run_gt_actions(
        base_env=_Env(),
        state=np.zeros(2, dtype=np.float32),
        actions=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        token=1,
        context={
            "policy": SimpleNamespace(terminator=SimpleNamespace(reset=lambda: None)),
            "display_terminators": [],
        },
        env_preprocessor=object(),
        initial_previous_action=np.asarray([9.0, 8.0], dtype=np.float32),
    )

    assert result["steps"] == 2
    np.testing.assert_array_equal(seen[0], [9.0, 8.0])
    np.testing.assert_array_equal(seen[1], [1.0, 2.0])
    np.testing.assert_array_equal(seen[2], [3.0, 4.0])


def test_rollout_max_skill_length_scales_each_occurrence_and_early_start() -> None:
    base = MODULE._rollout_max_skill_length(
        gt_length=45,
        mode="gt_scale",
        fixed_length=1,
        scale=1.5,
    )

    assert base == 68
    assert MODULE._branch_max_skill_length(
        base_max_skill_length=base,
        branch="policy_early",
        time_shift_offset=15,
    ) == 83
    assert MODULE._branch_max_skill_length(
        base_max_skill_length=base,
        branch="policy",
        time_shift_offset=15,
    ) == 68


def test_policy_rollout_records_but_does_not_stop_on_task_success(monkeypatch) -> None:
    class _Policy(nn.Linear):
        def __init__(self) -> None:
            super().__init__(1, 1)

        def reset(self) -> None:
            pass

        def predict_action_chunk(self, _batch):
            return torch.zeros(1, 1, 7)

    class _Env:
        def __init__(self) -> None:
            self._env = self
            self.steps = 0

        def step(self, _action):
            self.steps += 1
            # LIBERO's raw env reports done on task success. That signal must
            # not terminate a learned-terminator rollout.
            return object(), 0.0, self.steps >= 1, {}

        def check_success(self) -> bool:
            return self.steps >= 1

    signals = iter([0.0, 0.0, 1.0])

    def query(**_kwargs):
        termination = next(signals)
        return (
            {MODULE.RAW_IMAGE: torch.zeros(1, 3, 8, 8)},
            np.zeros(7, dtype=np.float32),
            0.0,
            termination,
            [],
        )

    monkeypatch.setattr(MODULE, "_query_terminator", query)
    monkeypatch.setattr(MODULE, "_restore_state", lambda *_args: object())
    monkeypatch.setattr(
        MODULE, "_render", lambda *_args: np.zeros((8, 8, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(
        MODULE,
        "_postprocess_action",
        lambda *_args: np.zeros(7, dtype=np.float32),
    )
    result = MODULE._run_policy(
        base_env=_Env(),
        state=np.zeros(7, dtype=np.float32),
        expected_filtered_state=np.zeros(7, dtype=np.float32),
        token=1,
        context={
            "policy": SimpleNamespace(
                policy=_Policy(),
                terminator=SimpleNamespace(reset=lambda: None),
            ),
            "postprocessor": object(),
            "display_terminators": [],
        },
        env_preprocessor=object(),
        env_postprocessor=object(),
        max_skill_length=10,
        n_action_steps=1,
        end_mode="termination",
        end_threshold=0.5,
        progress_threshold=0.95,
        finish_action_chunk_on_end=False,
        seed=1,
    )

    assert result["task_success_seen"] is True
    assert result["task_success_step"] == 1
    assert result["steps"] == 2
    assert result["stop_reason"] == "predicted_end"
    assert result["environment_done_step"] is None


@pytest.mark.parametrize(
    ("variant", "expected_type", "expected_prefix"),
    [
        (
            "state_only",
            MODULE.StateSkillMLPTerminator,
            "model.fsq_state_term_train.",
        ),
        (
            "state_rnn",
            MODULE.StateSkillRNNTerminator,
            "model.fsq_state_rnn_term_train.",
        ),
    ],
)
def test_state_display_terminator_rebuilds_checkpoint_architecture(
    tmp_path: Path,
    monkeypatch,
    variant: str,
    expected_type,
    expected_prefix: str,
) -> None:
    checkpoint = tmp_path / variant
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "skill_aux",
                "skill_fsq_levels": [3, 3, 3],
                "max_state_dim": 8,
                "state_only_terminator_hidden_dim": 16,
                "state_only_terminator_num_layers": 2,
                "state_only_terminator_termination_only": False,
                "state_rnn_terminator_input_dim": 12,
                "state_rnn_terminator_hidden_dim": 16,
                "state_rnn_terminator_num_layers": 1,
                "state_rnn_terminator_dropout": 0.0,
                "state_rnn_terminator_termination_only": False,
            }
        )
    )
    loaded: list[tuple[type[nn.Module], str]] = []

    def load(module, _path, *, prefix, label):
        del label
        loaded.append((type(module), prefix))
        return len(module.state_dict())

    monkeypatch.setattr(MODULE, "_load_complete_terminator_parameters", load)
    adapter = MODULE._load_display_terminator(
        SimpleNamespace(model=_DummyPolicyModel()),
        {"variant": variant, "path": str(checkpoint)},
        tmp_path / "unused_fsq.pt",
    )

    assert isinstance(adapter.module, expected_type)
    assert adapter.termination_only is False
    assert loaded == [(expected_type, expected_prefix)]
