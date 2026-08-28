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
    / "examples/libero/configs/train_skillVLA/stage1_skill_eval/src/run_skill_eval.py"
)
SPEC = importlib.util.spec_from_file_location("run_stage1_skill_eval", SCRIPT)
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


def test_policy_episode_units_use_all_ten_workers() -> None:
    assignments = [
        MODULE._worker_model_episode_units(
            model_count=5,
            selected={3: [30, 31]},
            worker_index=worker_index,
            worker_count=10,
        )
        for worker_index in range(10)
    ]

    units = [
        (model_index, episode_id)
        for assignment in assignments
        for model_index, episode_ids in assignment.items()
        for episode_id in episode_ids
    ]
    assert units == [
        (model_index, episode_id)
        for model_index in range(5)
        for episode_id in (30, 31)
    ]
    assert all(sum(map(len, assignment.values())) == 1 for assignment in assignments)


def test_policy_episode_units_group_each_workers_models() -> None:
    assignments = [
        MODULE._worker_model_episode_units(
            model_count=3,
            selected={0: [10], 2: [20, 21]},
            worker_index=worker_index,
            worker_count=4,
        )
        for worker_index in range(4)
    ]

    units = {
        (model_index, episode_id)
        for assignment in assignments
        for model_index, episode_ids in assignment.items()
        for episode_id in episode_ids
    }
    assert units == {
        (model_index, episode_id)
        for model_index in range(3)
        for episode_id in (10, 20, 21)
    }
    assert sum(
        len(episode_ids)
        for assignment in assignments
        for episode_ids in assignment.values()
    ) == 9


def test_fsq_initial_is_attached_as_main_terminator(tmp_path: Path, monkeypatch) -> None:
    fsq_path = tmp_path / "FSQ.pt"
    fsq_path.touch()
    action_policy = SimpleNamespace(model=nn.Linear(1, 1))
    wrapper = SimpleNamespace(policy=action_policy, terminator=None, advance_mode="gt")
    context = {"policy": wrapper, "preprocessor": object(), "config": object()}
    base_specs: list[dict] = []

    monkeypatch.setattr(
        MODULE,
        "_build_stage1_context",
        lambda spec, _cfg, _device: (base_specs.append(spec) or context),
    )
    monkeypatch.setattr(MODULE, "_ensure_skill_runtime_steps", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(MODULE, "build_fsq_terminator", lambda _path: _DummyTerminator())

    result = MODULE._build_context(
        {
            "label": "POLICY",
            "advance_mode": "external",
            "external_skill_model": str(fsq_path),
            "external_skill_model_variant": "fsq_initial",
        },
        SimpleNamespace(),
        torch.device("cpu"),
    )

    assert result is context
    assert base_specs[0]["advance_mode"] == "gt"
    assert isinstance(wrapper.terminator, MODULE.IndependentTerminator)
    assert wrapper.terminator.variant == "fsq_initial"
    assert wrapper.advance_mode == "external"


def test_trained_main_uses_standard_external_terminator_loader(monkeypatch) -> None:
    context = {"policy": object()}
    seen: list[dict] = []
    monkeypatch.setattr(
        MODULE,
        "_build_stage1_context",
        lambda spec, _cfg, _device: (seen.append(spec) or context),
    )

    spec = {
        "label": "POLICY",
        "advance_mode": "external",
        "terminator_variant": "image_only",
        "external_skill_model": "/checkpoint/pretrained_model",
        "external_skill_model_variant": "image_only",
    }
    result = MODULE._build_context(spec, SimpleNamespace(), torch.device("cpu"))

    assert result is context
    assert seen == [spec]


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


def test_same_main_and_display_source_reuses_one_forward(monkeypatch) -> None:
    class _Signal:
        def __init__(self) -> None:
            self.calls = 0

        def terminate(self, *_args, **_kwargs):
            self.calls += 1
            return torch.tensor([0.42]), torch.tensor([0.73])

    signal = _Signal()
    policy = nn.Linear(1, 1)
    batch = {
        MODULE.RAW_STATE: torch.zeros(1, 7),
        MODULE.RAW_IMAGE: torch.zeros(1, 3, 8, 8),
        MODULE.RAW_WRIST: torch.zeros(1, 3, 8, 8),
    }
    context = {
        "policy": SimpleNamespace(policy=policy, terminator=signal),
        "preprocessor": object(),
        "display_terminators": [
            {
                "label": "TRAINED",
                "reuse_main": True,
                "terminator": None,
            }
        ],
    }
    monkeypatch.setattr(
        MODULE,
        "_prepare_observation",
        lambda **_kwargs: (batch, np.zeros(7, dtype=np.float32)),
    )

    _, _, main_progress, main_termination, display = MODULE._query_terminator(
        base_env=object(),
        raw_obs=object(),
        token=1,
        context=context,
        env_preprocessor=object(),
    )

    assert signal.calls == 1
    assert (main_progress, main_termination) == pytest.approx((0.42, 0.73))
    assert len(display) == 1
    assert display[0] == pytest.approx((0.42, 0.73))


def test_main_display_reuse_requires_same_variant_and_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    spec = {
        "external_skill_model": str(checkpoint),
        "external_skill_model_variant": "image_only",
    }

    assert MODULE._display_reuses_main(
        spec,
        {"path": str(checkpoint), "variant": "image_only"},
    )
    assert not MODULE._display_reuses_main(
        spec,
        {"path": str(checkpoint), "variant": "state_image"},
    )
    assert not MODULE._display_reuses_main(
        spec,
        {"path": str(tmp_path / "other"), "variant": "image_only"},
    )


def test_rollout_max_skill_length_scales_each_gt_occurrence() -> None:
    assert MODULE._rollout_max_skill_length(
        gt_length=45,
        mode="gt_scale",
        fixed_length=1,
        scale=1.1,
    ) == 50
    assert MODULE._rollout_max_skill_length(
        gt_length=63,
        mode="gt_scale",
        fixed_length=1,
        scale=1.1,
    ) == 70
    assert MODULE._rollout_max_skill_length(
        gt_length=45,
        mode="fixed",
        fixed_length=150,
        scale=0.0,
    ) == 150


def test_early_start_gets_time_shift_added_to_rollout_budget() -> None:
    base = MODULE._rollout_max_skill_length(
        gt_length=100,
        mode="gt_scale",
        fixed_length=1,
        scale=1.5,
    )

    assert base == 150
    assert MODULE._branch_max_skill_length(
        base_max_skill_length=base,
        branch="policy_early",
        time_shift_offset=15,
    ) == 165
    for branch in ("gt", "policy", "policy_alt_noise", "policy_late"):
        assert MODULE._branch_max_skill_length(
            base_max_skill_length=base,
            branch=branch,
            time_shift_offset=15,
        ) == 150


def test_query_keeps_main_and_display_termination_values_separate(monkeypatch) -> None:
    class _Signal:
        def __init__(self, progress: float, termination: float) -> None:
            self.progress = progress
            self.termination = termination

        def terminate(self, *_args, **_kwargs):
            return torch.tensor([self.progress]), torch.tensor([self.termination])

    policy = nn.Linear(1, 1)
    batch = {
        MODULE.RAW_STATE: torch.zeros(1, 7),
        MODULE.RAW_IMAGE: torch.zeros(1, 3, 8, 8),
        MODULE.RAW_WRIST: torch.zeros(1, 3, 8, 8),
    }
    context = {
        "policy": SimpleNamespace(
            policy=policy,
            terminator=_Signal(progress=0.91, termination=0.99),
        ),
        "preprocessor": object(),
        "display_terminators": [
            {"label": "TRAINED", "terminator": _Signal(0.22, 0.33)}
        ],
    }
    monkeypatch.setattr(
        MODULE,
        "_prepare_observation",
        lambda **_kwargs: (batch, np.zeros(7, dtype=np.float32)),
    )

    _, _, main_progress, main_termination, display = MODULE._query_terminator(
        base_env=object(),
        raw_obs=object(),
        token=1,
        context=context,
        env_preprocessor=object(),
    )

    assert main_progress == pytest.approx(0.91)
    assert main_termination == pytest.approx(0.99)
    assert display[0][0] == pytest.approx(0.22)
    assert display[0][1] == pytest.approx(0.33)


def test_display_value_is_frozen_at_first_fsq_main_boundary(monkeypatch) -> None:
    class _Policy(nn.Linear):
        def __init__(self) -> None:
            super().__init__(1, 1)

        def reset(self) -> None:
            pass

        def predict_action_chunk(self, _batch):
            return torch.zeros(1, 3, 7)

    class _BaseEnv:
        def __init__(self) -> None:
            self._env = self

        def step(self, _action):
            return object(), 0.0, False, {}

    signals = iter(
        [
            (0.1, 0.1, [(0.8, 0.8)]),
            (0.9, 0.95, [(0.4, 0.4)]),
            (0.2, 0.2, [(0.2, 0.2)]),
            (0.1, 0.1, [(0.1, 0.1)]),
        ]
    )

    def query(**_kwargs):
        progress, termination, display = next(signals)
        return (
            {MODULE.RAW_IMAGE: torch.zeros(1, 3, 8, 8)},
            np.zeros(7, dtype=np.float32),
            progress,
            termination,
            display,
        )

    monkeypatch.setattr(MODULE, "_query_terminator", query)
    monkeypatch.setattr(MODULE, "_restore_state", lambda *_args: object())
    monkeypatch.setattr(MODULE, "_render", lambda *_args: np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(
        MODULE,
        "_postprocess_action",
        lambda *_args: np.zeros(7, dtype=np.float32),
    )
    context = {
        "policy": SimpleNamespace(policy=_Policy()),
        "postprocessor": object(),
        "display_terminators": [{"label": "TRAINED"}],
    }

    result = MODULE._run_policy(
        base_env=_BaseEnv(),
        state=np.zeros(7, dtype=np.float32),
        expected_filtered_state=np.zeros(7, dtype=np.float32),
        token=1,
        context=context,
        env_preprocessor=object(),
        env_postprocessor=object(),
        max_skill_length=10,
        n_action_steps=3,
        end_mode="termination",
        end_threshold=0.9,
        progress_threshold=0.9,
        finish_action_chunk_on_end=True,
        seed=1,
    )

    assert result["steps"] == 3
    assert result["main_boundary"]["step"] == 1
    assert result["main_boundary"]["termination"] == pytest.approx(0.95)
    assert result["main_boundary"]["display_terminators"][0]["termination"] == pytest.approx(0.4)
    assert result["main_boundary"]["display_terminators"][0]["would_fire"] is False
    assert (
        result["main_boundary"]["display_terminators"][0][
            "fired_by_main_boundary"
        ]
        is True
    )
    assert result["display_traces"][0]["termination"] == pytest.approx(
        [0.8, 0.4, 0.4, 0.4]
    )
    assert result["display_traces"][0]["fired"] == [True, True, True, True]


def test_display_firing_adds_translucent_green_tint_to_later_frames() -> None:
    frames = [
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.zeros((8, 8, 3), dtype=np.uint8),
    ]
    trace = {
        "label": "TRAINED",
        "end_mode": "termination",
        "end_threshold": 0.9,
        "progress_threshold": 0.95,
        "progress": [0.1, 0.1],
        "termination": [0.1, 0.95],
        "fired": [False, True],
    }

    annotated = MODULE._annotate_frames(
        frames,
        progress=[0.1, 0.1],
        termination=[0.1, 0.1],
        display_traces=[trace],
        progress_threshold=0.95,
        end_threshold=0.5,
    )

    assert annotated[0][0, 0].tolist() == [0, 0, 0]
    tinted_pixel = annotated[1][0, 0]
    assert int(tinted_pixel[1]) > int(tinted_pixel[2]) > int(tinted_pixel[0]) > 0
