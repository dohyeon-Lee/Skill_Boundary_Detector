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


def test_restore_state_synchronizes_controller_before_first_action(monkeypatch) -> None:
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

        def seed(self, value: int) -> None:
            events.append(("env_seed", value))

        def reset(self) -> None:
            events.append("env_reset")

        def set_init_state(self, state: np.ndarray):
            events.append(("set_init_state", state.copy()))
            return "restored_obs"

    env = _Env()
    base_env = SimpleNamespace(_env=env)
    state = np.array([1.0, 2.0], dtype=np.float32)
    monkeypatch.setattr(
        MODULE,
        "_apply_seeded_fixture_layout",
        lambda _base_env, seed: events.append(("fixture_layout", seed)),
    )

    raw_obs = MODULE._restore_state(base_env, state, layout_seed=123)

    assert raw_obs == "restored_obs"
    assert env.robots[0].controller.use_delta is True
    assert events[0] == ("env_seed", 123)
    assert events[1] == "env_reset"
    assert events[2] == ("fixture_layout", 123)
    assert events[3][0] == "set_init_state"
    assert events[3][1].dtype == np.float64
    assert np.array_equal(events[3][1], state)
    assert events[4:] == [("controller_update", True), "controller_reset_goal"]


def test_exact_restore_loads_recorded_xml_before_frame_state() -> None:
    events: list[object] = []

    class _Controller:
        use_delta = False

        def update(self, *, force: bool = False) -> None:
            events.append(("controller_update", force))

        def reset_goal(self) -> None:
            events.append("controller_reset_goal")

    class _Env:
        robots = [SimpleNamespace(controller=_Controller())]

        @staticmethod
        def reset_from_xml_string(xml: str) -> None:
            events.append(("xml", xml))

        @staticmethod
        def set_init_state(state: np.ndarray):
            events.append(("state", state.copy()))
            return "exact_obs"

    result = MODULE._restore_state(
        SimpleNamespace(_env=_Env()),
        np.asarray([4.0, 5.0], dtype=np.float32),
        model_xml="<mujoco/>",
    )

    assert result == "exact_obs"
    assert events[0] == ("xml", "<mujoco/>")
    assert events[1][0] == "state"
    assert events[1][1].dtype == np.float64


def test_langgap_restore_uses_task_init_id_and_replays_actions() -> None:
    init_states = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    events: list[object] = []

    class _RawEnv:
        def __init__(self) -> None:
            self.env = SimpleNamespace(
                _get_observations=lambda: {"step": len(events)}
            )

        @staticmethod
        def step(action):
            events.append(("step", np.asarray(action).copy()))
            return {"step": len(events)}, 0.0, False, {}

    class _BaseEnv:
        def __init__(self) -> None:
            self._init_states = init_states
            self._env = _RawEnv()
            self.init_state_id = -1
            self.task_id = 7

        def reset(self, *, seed, _advance):
            events.append(("reset", seed, _advance, self.init_state_id))
            return {}, {}

    base_env = _BaseEnv()
    replay = np.asarray(
        [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
         [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, -1.0]],
        dtype=np.float32,
    )
    result = MODULE._restore_skill_start(
        base_env,
        init_states[1],
        exact_init_state_index=1,
        replay_actions=replay,
    )

    assert events[0] == ("reset", 0, False, 1)
    np.testing.assert_array_equal(events[1][1], replay[0])
    np.testing.assert_array_equal(events[2][1], replay[1])
    assert result == {"step": 3}


def test_langgap_dataset_builds_episode_replay_plan(tmp_path: Path) -> None:
    import pandas as pd
    import skill_data

    dataset_dir = tmp_path / "skill_dataset"
    (dataset_dir / "meta" / "episodes").mkdir(parents=True)
    (dataset_dir / "data" / "chunk-000").mkdir(parents=True)
    pd.DataFrame(
        {
            "episode_index": [9],
            "data/chunk_index": [0],
            "data/file_index": [0],
            "length": [3],
            "tasks": [["task"]],
        }
    ).to_parquet(dataset_dir / "meta" / "episodes" / "chunk.parquet")
    pd.DataFrame(
        {"task_index": [0]},
        index=pd.Index(["task"], name="task"),
    ).to_parquet(dataset_dir / "meta" / "tasks.parquet")
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)
    states = np.arange(24, dtype=np.float32).reshape(3, 8)
    pd.DataFrame(
        {
            "episode_index": [9, 9, 9],
            "frame_index": [0, 1, 2],
            "action": list(actions),
            "observation.state": list(states),
        }
    ).to_parquet(dataset_dir / "data" / "chunk-000" / "file-000.parquet")
    latents_path = tmp_path / "skill_latents.npz"
    np.savez(
        latents_path,
        tokens=np.asarray([2]),
        episode_id=np.asarray([9]),
        skill_index=np.asarray([0]),
        frame_start=np.asarray([2]),
        frame_end=np.asarray([3]),
    )
    init_state = np.asarray([8.0, 9.0, 10.0], dtype=np.float64)
    exact_path = tmp_path / "eval_init_states.npz"
    np.savez(
        exact_path,
        episode_index=np.asarray([9]),
        dataset_task_id=np.asarray([0]),
        init_states=np.asarray([init_state]),
        suite_name=np.asarray(["langgap_ext"]),
        suite_task_id=np.asarray([5]),
        init_state_index=np.asarray([17]),
    )

    dataset = skill_data.SkillEvaluationDataset(
        skill_dataset_dir=dataset_dir,
        skill_latents_path=latents_path,
        eval_init_states_path=exact_path,
        original_dataset_dir="",
        suite_name="langgap_ext",
    )
    aligned = dataset.load_aligned_episode(9)
    state, replay, init_index = aligned.restoration_at(2)

    assert dataset.uses_langgap_replay is True
    assert aligned.requires_episode_replay is True
    assert aligned.source.task_id == 0
    assert aligned.source.env_task_id == 5
    assert init_index == 17
    np.testing.assert_array_equal(state, init_state)
    np.testing.assert_array_equal(replay, actions[:2])
    np.testing.assert_array_equal(aligned.episode_start_xyz, states[0, :3])


def test_grounded_langgap_uses_raw_episode_start_xyz(tmp_path: Path) -> None:
    import json
    import pandas as pd
    import skill_data

    skill_dir = tmp_path / "skill_dataset"
    raw_dir = tmp_path / "raw_dataset"
    for dataset_dir in (skill_dir, raw_dir):
        (dataset_dir / "meta" / "episodes").mkdir(parents=True)
        (dataset_dir / "data" / "chunk-000").mkdir(parents=True)
        pd.DataFrame(
            {
                "episode_index": [9],
                "data/chunk_index": [0],
                "data/file_index": [0],
                "length": [3],
                "tasks": [["task"]],
            }
        ).to_parquet(dataset_dir / "meta" / "episodes" / "chunk.parquet")
        pd.DataFrame(
            {"task_index": [0]},
            index=pd.Index(["task"], name="task"),
        ).to_parquet(dataset_dir / "meta" / "tasks.parquet")

    raw_states = np.asarray(
        [
            [-0.2, 0.01, 1.15, 3.14, 0.0, 0.0, 0.04, -0.04],
            [-0.1, 0.03, 1.05, 3.14, 0.0, 0.0, 0.04, -0.04],
            [0.0, 0.05, 0.95, 3.14, 0.0, 0.0, 0.04, -0.04],
        ],
        dtype=np.float32,
    )
    grounded_states = raw_states.copy()
    grounded_states[:, :3] -= raw_states[0, :3]
    actions = np.zeros((3, 7), dtype=np.float32)
    for dataset_dir, states in (
        (skill_dir, grounded_states),
        (raw_dir, raw_states),
    ):
        pd.DataFrame(
            {
                "episode_index": [9, 9, 9],
                "frame_index": [0, 1, 2],
                "action": list(actions),
                "observation.state": list(states),
            }
        ).to_parquet(dataset_dir / "data" / "chunk-000" / "file-000.parquet")
    (skill_dir / "meta" / "info.json").write_text(
        json.dumps({"proprio_grounding": "episode_start_xyz"})
    )

    latents_path = tmp_path / "skill_latents.npz"
    np.savez(
        latents_path,
        tokens=np.asarray([2]),
        episode_id=np.asarray([9]),
        skill_index=np.asarray([0]),
        frame_start=np.asarray([2]),
        frame_end=np.asarray([3]),
    )
    init_state = np.asarray([8.0, 9.0, 10.0], dtype=np.float64)
    exact_path = tmp_path / "eval_init_states.npz"
    np.savez(
        exact_path,
        episode_index=np.asarray([9]),
        dataset_task_id=np.asarray([0]),
        init_states=np.asarray([init_state]),
        suite_name=np.asarray(["langgap_ext"]),
        suite_task_id=np.asarray([5]),
        init_state_index=np.asarray([17]),
    )

    dataset = skill_data.SkillEvaluationDataset(
        skill_dataset_dir=skill_dir,
        skill_latents_path=latents_path,
        eval_init_states_path=exact_path,
        original_dataset_dir=None,
        suite_name="langgap_ext",
        raw_dataset_dir=raw_dir,
    )
    aligned = dataset.load_aligned_episode(9)

    np.testing.assert_allclose(aligned.filtered_states, grounded_states)
    np.testing.assert_allclose(aligned.episode_start_xyz, raw_states[0, :3])


def test_grounded_langgap_rejects_missing_raw_dataset(tmp_path: Path) -> None:
    import json
    import pandas as pd
    import skill_data

    dataset_dir = tmp_path / "skill_dataset"
    (dataset_dir / "meta" / "episodes").mkdir(parents=True)
    (dataset_dir / "data" / "chunk-000").mkdir(parents=True)
    pd.DataFrame(
        {
            "episode_index": [9],
            "data/chunk_index": [0],
            "data/file_index": [0],
            "length": [1],
            "tasks": [["task"]],
        }
    ).to_parquet(dataset_dir / "meta" / "episodes" / "chunk.parquet")
    pd.DataFrame(
        {"task_index": [0]}, index=pd.Index(["task"], name="task")
    ).to_parquet(dataset_dir / "meta" / "tasks.parquet")
    pd.DataFrame(
        {
            "episode_index": [9],
            "frame_index": [0],
            "action": [np.zeros(7, dtype=np.float32)],
            "observation.state": [np.zeros(8, dtype=np.float32)],
        }
    ).to_parquet(dataset_dir / "data" / "chunk-000" / "file-000.parquet")
    (dataset_dir / "meta" / "info.json").write_text(
        json.dumps({"proprio_grounding": "episode_start_xyz"})
    )
    latents_path = tmp_path / "skill_latents.npz"
    np.savez(
        latents_path,
        tokens=np.asarray([2]),
        episode_id=np.asarray([9]),
        skill_index=np.asarray([0]),
        frame_start=np.asarray([0]),
        frame_end=np.asarray([1]),
    )
    exact_path = tmp_path / "eval_init_states.npz"
    np.savez(
        exact_path,
        episode_index=np.asarray([9]),
        dataset_task_id=np.asarray([0]),
        init_states=np.asarray([[8.0, 9.0, 10.0]]),
        suite_name=np.asarray(["langgap_ext"]),
        suite_task_id=np.asarray([5]),
        init_state_index=np.asarray([17]),
    )
    dataset = skill_data.SkillEvaluationDataset(
        skill_dataset_dir=dataset_dir,
        skill_latents_path=latents_path,
        eval_init_states_path=exact_path,
        original_dataset_dir=None,
        suite_name="langgap_ext",
    )

    with pytest.raises(RuntimeError, match="raw source dataset"):
        dataset.load_aligned_episode(9)


def test_partial_skill_uses_explicit_parent_episode_grounding_reference() -> None:
    from lerobot.policies.skill_expert.processor_skill_expert import (
        EpisodeStartXYZGroundingProcessorStep,
    )
    from lerobot.types import TransitionKey
    from lerobot.utils.constants import OBS_STATE

    step = EpisodeStartXYZGroundingProcessorStep()
    context = {
        "config": SimpleNamespace(proprio_grounding="episode_start_xyz"),
        "preprocessor": SimpleNamespace(steps=[step]),
    }
    MODULE._set_episode_grounding_reference(
        context, np.array([1.0, 2.0, 3.0], dtype=np.float32)
    )

    transition = step(
        {
            TransitionKey.OBSERVATION: {
                OBS_STATE: torch.tensor([[4.0, 6.0, 8.0, 9.0]])
            }
        }
    )
    torch.testing.assert_close(
        transition[TransitionKey.OBSERVATION][OBS_STATE],
        torch.tensor([[3.0, 4.0, 5.0, 9.0]]),
    )


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


@pytest.mark.parametrize("advance_mode", ["external", "original"])
def test_fsq_initial_is_attached_as_main_terminator(
    tmp_path: Path, monkeypatch, advance_mode: str
) -> None:
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
            "advance_mode": advance_mode,
            "external_skill_model": str(fsq_path),
            "external_skill_model_variant": "fsq_initial",
        },
        SimpleNamespace(),
        torch.device("cpu"),
    )

    assert result is context
    assert base_specs[0]["advance_mode"] == "gt"
    assert base_specs[0]["terminator_variant"] == "state_image"
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
    monkeypatch.setattr(
        MODULE, "_restore_state", lambda *_args, **_kwargs: object()
    )
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
            self.steps = 0

        def step(self, _action):
            self.steps += 1
            return object(), 0.0, self.steps >= 1, {}

        def check_success(self) -> bool:
            return self.steps >= 1

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
    monkeypatch.setattr(
        MODULE, "_restore_state", lambda *_args, **_kwargs: object()
    )
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
    assert result["stop_reason"] == "predicted_end"
    assert result["task_success_seen"] is True
    assert result["task_success_step"] == 1
    assert result["environment_done_step"] is None
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
