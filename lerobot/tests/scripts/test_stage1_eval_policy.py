import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


_EVAL_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/PT/eval/src"
)
sys.path.insert(0, str(_EVAL_SRC))

import eval_oracle
import run_eval
import lerobot.scripts.lerobot_skillvla_eval as skillvla_eval
from goal_noise import GoalNoisePerturber
from run_eval import CheckpointTerminator, Stage1OraclePolicy
from lerobot.policies.skill_expert import modeling_skill_expert
from lerobot.policies.skill_expert.configuration_skill_expert import SkillExpertConfig
from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy
from lerobot.policies.skill_expert.processor_skill_expert import (
    EpisodeStartXYZGroundingProcessorStep,
)
from lerobot.scripts.lerobot_skillvla_eval import (
    _annotate_eval_video,
    _latent_values_from_trace,
    _make_presentation_eval_video,
    _policy_completion_vector,
    _progress_values_from_trace,
    _skill_banner_color,
    _skill_ids_from_trace,
    _skill_video_segments_from_trace,
    _termination_values_from_trace,
)
from lerobot.utils.constants import (
    SKILL_FLOW_MODE_LATENT_OVERRIDE,
    SKILL_FLOW_NOISE_OVERRIDE,
    STAGE2_MODE_LATENT_OVERRIDE,
    STAGE2_VLM_CACHE_ID,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import OBS_STATE


def test_motion_plan_capture_toggle_validates_window_and_clears_pending():
    model = SimpleNamespace()
    wrapper = SimpleNamespace(policy=SimpleNamespace(config=SimpleNamespace(uses_motion_window=True), model=model))
    Stage1OraclePolicy.set_capture_motion_plans(wrapper, True, 4)
    assert wrapper.capture_motion_plans and model.capture_motion_plans
    Stage1OraclePolicy.set_capture_motion_plans(wrapper, True, 4, independent_base=True)
    assert model.capture_independent_motion_base
    wrapper._pending_motion_plan = {"chunk": 1}
    assert Stage1OraclePolicy.pop_motion_plan(wrapper) == {"chunk": 1}
    assert Stage1OraclePolicy.pop_motion_plan(wrapper) is None
    Stage1OraclePolicy.set_capture_motion_plans(wrapper, False)
    assert not model.capture_motion_plans
    assert not model.capture_independent_motion_base
    wrapper.policy.config.uses_motion_window = False
    with pytest.raises(ValueError, match="4H"):
        Stage1OraclePolicy.set_capture_motion_plans(wrapper, True)


def test_bspline_motion_capture_reads_actual_head_without_inference_cache_or_rng_changes():
    wrapper = Stage1OraclePolicy.__new__(Stage1OraclePolicy)
    nn.Module.__init__(wrapper)
    prediction = torch.randn(1, 30, 8)
    wrapper.policy = SimpleNamespace(
        config=SimpleNamespace(uses_motion_window=False, uses_bspline_plan=True,
                               proprio_grounding="episode_start_xyz", num_inference_steps=10,
                               chunk_end_state_q01=[-1.] * 8, chunk_end_state_q99=[1.] * 8),
        model=SimpleNamespace(_bspline_prediction=prediction),
    )
    wrapper._runtime_preprocessor = SimpleNamespace(steps=[SimpleNamespace(_reference_xyz=torch.tensor([[.1, .2, .3]]))])
    wrapper._skill_order, wrapper._active_trace = [0], [0]
    wrapper._trace = [{"codebook_token": 13}]
    wrapper._episode_step, wrapper.n_action_steps = 5, 2
    wrapper.set_capture_motion_plans(True)
    assert not wrapper.policy.model.capture_motion_plans
    rng = torch.get_rng_state().clone()
    chunk = torch.ones(1, 10, 7)
    wrapper._capture_motion_plan({"skill_start_state": torch.zeros(1, 8)}, chunk)
    plan = wrapper.pop_motion_plan()
    assert plan["kind"] == "bspline" and plan["flow_time"] == .1
    np.testing.assert_allclose(plan["grounding_offset"], [[.1, .2, .3]])
    torch.testing.assert_close(plan["proprio_normalized"], prediction, rtol=0, atol=0)
    assert wrapper.policy.model._bspline_prediction is prediction
    assert plan["proprio_normalized"].data_ptr() != prediction.data_ptr()
    assert torch.equal(torch.get_rng_state(), rng)
    torch.testing.assert_close(chunk, torch.ones_like(chunk), rtol=0, atol=0)


def test_bspline_motion_capture_rejects_missing_runtime_origin():
    wrapper = SimpleNamespace(
        policy=SimpleNamespace(config=SimpleNamespace(uses_bspline_plan=True, proprio_grounding="episode_start_xyz"),
                               model=SimpleNamespace(_bspline_prediction=torch.zeros(1, 30, 8))),
        get_grounding_offset=lambda: None,
    )
    with pytest.raises(RuntimeError, match="grounding origin"):
        Stage1OraclePolicy._capture_motion_plan(wrapper, {}, torch.zeros(1, 10, 7))


def test_canonical_motion_capture_caches_each_occurrence_and_preserves_rng():
    from collections import deque

    class CanonicalExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(uses_motion_window=False, skill_flow_enabled=True,
                                          skill_flow_target="canonical", skill_flow_max_length=12)
            self.model = SimpleNamespace()
            self.samples = []
            self._last_eval_mode_latent = torch.tensor([[.1, .2], [.3, .4]])

        def reset(self):
            pass

        def predict_skill_only_action_chunk(self, batch, *, horizon, mode_latent):
            self.samples.append((horizon, batch["skill_code"].clone(), mode_latent.clone()))
            return torch.rand(1, horizon, 7)

    wrapper = Stage1OraclePolicy.__new__(Stage1OraclePolicy)
    nn.Module.__init__(wrapper)
    wrapper.policy = CanonicalExpert()
    wrapper.skill_source = "gt"
    wrapper._gt_lengths = [[3, 5], [4]]
    wrapper._skill_order = [0, 0]
    wrapper._active_trace = [0, 1]
    wrapper._trace = [{"codebook_token": 3}, {"codebook_token": 8}]
    wrapper._episode_step = 0
    wrapper.n_action_steps = 2
    wrapper.set_capture_motion_plans(True)
    assert not wrapper.policy.model.capture_motion_plans
    batch = {"skill_code": torch.tensor([3, 8]), "observation.state": torch.zeros(2, 8)}
    chunk = torch.ones(2, 2, 7)
    torch.manual_seed(71)
    before = torch.get_rng_state().clone()
    wrapper._capture_motion_plan(batch, chunk)
    assert torch.equal(torch.get_rng_state(), before)
    first = wrapper.pop_motion_plan()
    with torch.random.fork_rng(devices=[]):
        next_real_noise = torch.rand(1, 3, 7)
    assert not torch.equal(first["full_skills"][0]["actions"], next_real_noise[0])
    assert [sample[0] for sample in wrapper.policy.samples] == [3, 4]
    assert first["kind"] == "canonical"
    torch.testing.assert_close(first["chunk"], chunk, rtol=0, atol=0)
    torch.testing.assert_close(wrapper.policy.samples[1][2], torch.tensor([[.3, .4]]))

    wrapper._episode_step = 2
    wrapper._capture_motion_plan(batch, chunk)
    second = wrapper.pop_motion_plan()
    assert len(wrapper.policy.samples) == 2
    assert second["full_skills"][0] is first["full_skills"][0]
    assert second["full_skills"][0]["skill_start_step"] == 0

    wrapper._skill_order[0] = 1
    wrapper._capture_motion_plan(batch, chunk)
    third = wrapper.pop_motion_plan()
    assert len(wrapper.policy.samples) == 3
    assert third["full_skills"][0]["gt_length"] == 5
    assert third["full_skills"][0]["skill_start_step"] == 2
    assert torch.equal(torch.get_rng_state(), before)

    # Resetting an episode must not reuse a previous episode's canonical samples.
    wrapper.terminator = None
    wrapper._action_queue = deque()
    wrapper._sequences = [[3, 3], [8]]
    wrapper.reset()
    assert wrapper._motion_full_skill_cache == {}


def test_canonical_motion_capture_requires_gt_length_not_a_silent_248_fallback():
    wrapper = Stage1OraclePolicy.__new__(Stage1OraclePolicy)
    nn.Module.__init__(wrapper)
    wrapper.policy = SimpleNamespace(config=SimpleNamespace(uses_motion_window=False, skill_flow_max_length=12))
    wrapper._motion_full_skill_cache = {}
    wrapper._gt_lengths = [[13]]
    wrapper._skill_order = [0]
    with pytest.raises(ValueError, match="outside trained horizon"):
        wrapper._capture_motion_plan({}, torch.ones(1, 2, 7))


def test_paired_policy_rng_restarts_every_episode_seed() -> None:
    skillvla_eval._reset_paired_policy_rng(True, 123)
    first = (random.random(), np.random.rand(), torch.rand(3))
    skillvla_eval._reset_paired_policy_rng(True, 123)
    repeated = (random.random(), np.random.rand(), torch.rand(3))
    assert first[0] == repeated[0]
    assert first[1] == repeated[1]
    torch.testing.assert_close(first[2], repeated[2])

    skillvla_eval._reset_paired_policy_rng(True, 124)
    assert not torch.equal(first[2], torch.rand(3))


def test_presentation_video_is_rollout_only_with_skill_tint_and_prompt() -> None:
    frames = np.full((2, 12, 12, 3), 100, dtype=np.uint8)
    rendered = _make_presentation_eval_video(
        frames,
        "put the object away",
        [15, 20],
        output_size=256,
        tint_alpha=0.20,
    )

    # 256px rollout plus the renderer's minimum 54px language bar: no outcome
    # bar or camera-label row is inserted above the simulator image.
    assert rendered.shape == (2, 310, 256, 3)
    assert not np.array_equal(rendered[0, 0, 0], rendered[1, 0, 0])
    assert np.all(rendered[:, 256:, 0] == np.array([20, 20, 20]))
    assert not np.any(np.all(rendered == np.array([255, 64, 64]), axis=-1))


def test_skill_video_segments_keep_repeated_token_occurrences_separate() -> None:
    trace = [
        {
            "batch_index": 0,
            "skill_index": 0,
            "codebook_token": 2,
            "episode_timestep": 0,
        },
        {
            "batch_index": 0,
            "skill_index": 1,
            "codebook_token": 2,
            "episode_timestep": 3,
        },
        {
            "batch_index": 0,
            "skill_index": 2,
            "codebook_token": 7,
            "episode_timestep": 8,
        },
        {
            "batch_index": 1,
            "skill_index": 0,
            "codebook_token": 9,
            "episode_timestep": 0,
        },
    ]

    assert _skill_video_segments_from_trace(
        trace,
        batch_index=0,
        n_video_frames=6,
        video_frame_stride=2,
    ) == [
        (0, 2, 0, 2),
        (1, 2, 2, 4),
        (2, 7, 4, 6),
    ]


def test_paired_policy_rng_requires_sequential_tasks() -> None:
    with pytest.raises(ValueError, match="max_parallel_tasks=1"):
        skillvla_eval.eval_policy_all(
            envs={},
            policy=None,
            env_preprocessor=None,
            env_postprocessor=None,
            preprocessor=None,
            postprocessor=None,
            n_episodes=1,
            start_seed=42,
            paired_policy_rng=True,
            max_parallel_tasks=2,
        )


def test_spec_end_threshold_overrides_global_default(monkeypatch) -> None:
    monkeypatch.setenv("SKILL_END_THRESHOLD", "0.3")

    assert run_eval._spec_end_threshold({}) == 0.3
    assert run_eval._spec_end_threshold({"end_threshold": 0.5}) == 0.5


def test_spec_end_mode_overrides_global_and_blank_inherits(monkeypatch) -> None:
    monkeypatch.setenv("SKILL_END_MODE", "termination")

    assert run_eval._spec_end_mode({}) == "termination"
    assert run_eval._spec_end_mode({"end_mode": ""}) == "termination"
    assert run_eval._spec_end_mode({"end_mode": "progress"}) == "progress"


def test_predictor_diagnostic_summary_aggregates_replans_and_complete_episodes() -> None:
    summary = run_eval._predictor_diagnostic_summary(
        [
            {
                "num_reference_skills": 2,
                "all_observed_skills_correct": True,
                "all_reference_skills_correct": True,
                "replans": [
                    {
                        "predicted_skill_correct": True,
                        "predictor_gt_skill_conditioned_xyz_l2_m": 0.01,
                        "predictor_pred_skill_conditioned_xyz_l2_m": 0.02,
                        "predictor_gt_skill_conditioned_xyz_mae_m": 0.005,
                        "predictor_pred_skill_conditioned_xyz_mae_m": 0.01,
                        "replan_timing_error_steps": -2,
                    },
                    {
                        "predicted_skill_correct": True,
                        "predictor_gt_skill_conditioned_xyz_l2_m": 0.03,
                        "predictor_pred_skill_conditioned_xyz_l2_m": 0.06,
                        "predictor_gt_skill_conditioned_xyz_mae_m": 0.015,
                        "predictor_pred_skill_conditioned_xyz_mae_m": 0.03,
                        "replan_timing_error_steps": 4,
                    },
                ],
            },
            {
                "num_reference_skills": 1,
                "all_observed_skills_correct": False,
                "all_reference_skills_correct": False,
                "replans": [{"predicted_skill_correct": False}],
            },
        ]
    )

    assert summary["predictor_skill_accuracy_pct"] == pytest.approx(200 / 3)
    assert summary["predictor_episode_all_reference_skills_correct_pct"] == 50
    assert summary["predictor_episode_all_skills_correct_pct"] == 50
    assert summary["predictor_gt_skill_conditioned_xyz_l2_mean_m"] == 0.02
    assert summary["predictor_pred_skill_conditioned_xyz_l2_mean_m"] == 0.04
    assert summary["predictor_replan_timing_bias_mean_steps"] == 1
    assert summary["predictor_replan_timing_mae_steps"] == 3


def test_skill_trace_records_predictor_and_boundary_diagnostics() -> None:
    wrapper = Stage1OraclePolicy.__new__(Stage1OraclePolicy)
    nn.Module.__init__(wrapper)
    wrapper.predictor_diagnostics = True
    wrapper.skill_source = "external"
    wrapper._references = [[3, 7]]
    wrapper._reference_lengths = [[5, 8]]
    wrapper._reference_end_states = [[
        np.asarray([0.1, 0.2, 0.3]),
        np.asarray([0.4, 0.5, 0.6]),
    ]]
    wrapper._sequences = wrapper._gt_lengths = wrapper._end_states = None
    wrapper._skill_order = [0]
    wrapper._skill_start_states = [None]
    wrapper._active_trace = [None]
    wrapper._trace = []
    wrapper._episode_step = 6
    wrapper._diagnostic_predicted_codes = torch.tensor([7])
    wrapper._diagnostic_gt_end_states = torch.tensor([[0.41, 0.49, 0.60]])
    wrapper._diagnostic_predicted_end_states = torch.tensor([[0.43, 0.50, 0.58]])
    wrapper._foveated_vision = SimpleNamespace(enabled=False)
    wrapper._requires_focus_uv_condition = False
    wrapper._requires_any_end_pose_condition = False
    wrapper._pending_oracle_skill_metadata = {}

    wrapper._start_skill(0, torch.tensor([7]))

    record = wrapper.get_skill_trace()[0]
    assert record["gt_skill_code"] == 7
    assert record["predicted_skill_correct"] is True
    assert record["actual_replan_timestep"] == 6
    assert record["gt_boundary_timestep"] == 5
    assert record["replan_timing_error_steps"] == 1
    assert record["predictor_gt_skill_conditioned_xyz_l2_m"] == pytest.approx(
        np.sqrt(0.0002)
    )
    episode = wrapper.get_predictor_diagnostics()[0]
    assert episode["all_skills_correct"] is False
    assert episode["all_reference_skills_correct"] is False
    assert episode["all_observed_skills_correct"] is True


def test_exact_skill_sequence_carries_occurrence_focus_uv() -> None:
    skills = eval_oracle._decode_skills(
        {
            "episode_index": 12,
            "skill_sequence": np.asarray([3, 7]),
            "skill_length_sequence": np.asarray([5, 9]),
        },
        27,
        {
            (12, 0): {
                "focus_uv": np.asarray([0.25, -0.5], dtype=np.float32),
                "focus_valid": True,
                "focus_clipped": False,
            },
            (12, 1): {
                "focus_uv": np.asarray([-0.75, 0.5], dtype=np.float32),
                "focus_valid": True,
                "focus_clipped": False,
            },
        },
    )

    np.testing.assert_allclose(skills[0]["focus_uv"], [0.25, -0.5])
    np.testing.assert_allclose(skills[1]["focus_uv"], [-0.75, 0.5])


def test_exact_end_pose_uses_dataset_endpoint_and_keeps_shared_final_frame(tmp_path: Path) -> None:
    import pandas as pd

    data_dir = tmp_path / "skillvla" / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    states = [np.full(8, frame, dtype=np.float32) for frame in range(4)]
    pd.DataFrame({
        "episode_index": [0] * 4,
        "frame_index": list(range(4)),
        "observation.state": states,
    }).to_parquet(data_dir / "file-000.parquet")
    endpoints = eval_oracle._skill_end_states(
        tmp_path / "skillvla",
        {
            (0, 0): {"frame_end": 3},
            (0, 1): {"frame_end": 4},
        },
    )
    np.testing.assert_array_equal(endpoints[(0, 0)], states[3])
    np.testing.assert_array_equal(endpoints[(0, 1)], states[3])


def test_episode_start_grounding_is_stateful_and_resettable() -> None:
    step = EpisodeStartXYZGroundingProcessorStep()

    first = step(
        {
            TransitionKey.OBSERVATION: {
                OBS_STATE: torch.tensor([[10.0, 20.0, 30.0, 4.0]])
            }
        }
    )
    second = step(
        {
            TransitionKey.OBSERVATION: {
                OBS_STATE: torch.tensor([[11.0, 18.0, 33.0, 5.0]])
            }
        }
    )
    torch.testing.assert_close(
        first[TransitionKey.OBSERVATION][OBS_STATE],
        torch.tensor([[0.0, 0.0, 0.0, 4.0]]),
    )
    torch.testing.assert_close(
        second[TransitionKey.OBSERVATION][OBS_STATE],
        torch.tensor([[1.0, -2.0, 3.0, 5.0]]),
    )

    step.reset()
    reset_first = step(
        {
            TransitionKey.OBSERVATION: {
                OBS_STATE: torch.tensor([[100.0, 200.0, 300.0, 6.0]])
            }
        }
    )
    torch.testing.assert_close(
        reset_first[TransitionKey.OBSERVATION][OBS_STATE],
        torch.tensor([[0.0, 0.0, 0.0, 6.0]]),
    )


def test_episode_start_grounding_accepts_explicit_partial_rollout_reference() -> None:
    step = EpisodeStartXYZGroundingProcessorStep()
    step.set_reference_xyz(np.array([10.0, 20.0, 30.0], dtype=np.float32))

    transition = step(
        {
            TransitionKey.OBSERVATION: {
                OBS_STATE: torch.tensor([[12.0, 18.0, 35.0, 7.0]])
            }
        }
    )

    torch.testing.assert_close(
        transition[TransitionKey.OBSERVATION][OBS_STATE],
        torch.tensor([[2.0, -2.0, 5.0, 7.0]]),
    )


def test_stage2_runtime_inserts_episode_start_grounding_before_normalizer() -> None:
    from lerobot.processor import NormalizerProcessorStep

    normalizer = NormalizerProcessorStep(features={}, norm_map={}, stats={})
    preprocessor = SimpleNamespace(steps=[normalizer])
    policy_config = SimpleNamespace(
        type="skill_vla_stage2",
        proprio_grounding="episode_start_xyz",
    )

    run_eval._ensure_skill_runtime_steps(
        preprocessor,
        policy_config,
        needs_predictor=False,
        needs_terminator=False,
    )

    assert isinstance(preprocessor.steps[0], EpisodeStartXYZGroundingProcessorStep)
    assert preprocessor.steps[1] is normalizer


def test_eval_runtime_removes_joint_training_only_terminator_tokenizer() -> None:
    from lerobot.policies.skillVLA.processor_skillVLA import (
        SkillVLATerminatorPredictorTokenizerProcessorStep,
    )

    training_only_tokenizer = object.__new__(
        SkillVLATerminatorPredictorTokenizerProcessorStep
    )
    retained_step = object()
    preprocessor = SimpleNamespace(steps=[retained_step, training_only_tokenizer])
    policy_config = SimpleNamespace(type="skill_expert", proprio_grounding="none")

    run_eval._ensure_skill_runtime_steps(
        preprocessor,
        policy_config,
        needs_predictor=False,
        needs_terminator=False,
    )

    assert preprocessor.steps == [retained_step]


@pytest.mark.parametrize("mode", ["mode3_preskill", "mode3_preskill_wristX_proprioX", "mode3_preskill_proprioX"])
@pytest.mark.parametrize("snapshot_saved", [False, True])
def test_relative_predictor_runtime_preserves_grounded_origin_and_prompt_ablation(mode, snapshot_saved):
    from lerobot.processor import NormalizerProcessorStep, TokenizerProcessorStep
    from lerobot.policies.skillVLA.processor_skillVLA import (
        SkillPredictorPreserveStartXYZProcessorStep, SkillVLAPrepareStateTokenizerProcessorStep,
    )
    tokenizer = object.__new__(TokenizerProcessorStep)
    tokenizer.tokenizer_name = "unused/test-tokenizer"
    normalizer = NormalizerProcessorStep(features={}, norm_map={}, stats={})
    saved = [SkillPredictorPreserveStartXYZProcessorStep()] if snapshot_saved else []
    preprocessor = SimpleNamespace(steps=[*saved, normalizer, tokenizer])
    config = SimpleNamespace(
        type="skill_expert", proprio_grounding="episode_start_xyz", architecture_label="both_3",
        max_state_dim=8, tokenizer_path=tokenizer.tokenizer_name, tokenizer_max_length=200,
    )
    run_eval._ensure_skill_runtime_steps(
        preprocessor, config, needs_predictor=True, needs_terminator=False, predictor_mode=mode,
    )
    steps = preprocessor.steps
    origin = next(step for step in steps if isinstance(step, SkillPredictorPreserveStartXYZProcessorStep))
    prompt = next(step for step in steps if isinstance(step, SkillVLAPrepareStateTokenizerProcessorStep))
    assert isinstance(steps[0], EpisodeStartXYZGroundingProcessorStep)
    assert steps.index(origin) < steps.index(normalizer)
    assert prompt.include_state == ("proprioX" not in mode)


def test_absolute_mode3_uv_runtime_has_no_relative_xyz_requirement():
    from lerobot.processor import NormalizerProcessorStep, TokenizerProcessorStep
    from lerobot.policies.skillVLA.processor_skillVLA import (
        SkillPredictorPreserveStartXYZProcessorStep, SkillVLAPrepareStateTokenizerProcessorStep,
    )
    tokenizer = object.__new__(TokenizerProcessorStep)
    tokenizer.tokenizer_name = "unused/test-tokenizer"
    preprocessor = SimpleNamespace(steps=[NormalizerProcessorStep(features={}, norm_map={}, stats={}), tokenizer])
    config = SimpleNamespace(
        type="skill_expert", proprio_grounding="episode_start_xyz", architecture_label="arch8",
        max_state_dim=8, tokenizer_path=tokenizer.tokenizer_name, tokenizer_max_length=200,
    )
    run_eval._ensure_skill_runtime_steps(
        preprocessor, config, needs_predictor=True, needs_terminator=False,
        predictor_mode="mode3_preskill_proprioX", predictor_spatial_target="uv",
    )
    assert not any(isinstance(step, SkillPredictorPreserveStartXYZProcessorStep) for step in preprocessor.steps)
    prompt = next(step for step in preprocessor.steps if isinstance(step, SkillVLAPrepareStateTokenizerProcessorStep))
    assert prompt.include_state is False


def test_inline_cuda_guard_is_opt_in(monkeypatch, tmp_path: Path) -> None:
    marker = tmp_path / "cuda.failed"
    monkeypatch.delenv("LEROBOT_INLINE_CUDA_GUARD", raising=False)
    monkeypatch.setenv("LEROBOT_CUDA_GUARD_FAILURE_MARKER", str(marker))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    run_eval._run_inline_cuda_guard()

    assert not marker.exists()


def test_inline_cuda_guard_marks_cuda_failure(monkeypatch, tmp_path: Path) -> None:
    marker = tmp_path / "cuda.failed"
    monkeypatch.setenv("LEROBOT_INLINE_CUDA_GUARD", "1")
    monkeypatch.setenv("LEROBOT_CUDA_GUARD_FAILURE_MARKER", str(marker))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(SystemExit) as error:
        run_eval._run_inline_cuda_guard()

    assert error.value.code == 86
    assert marker.read_text() == "torch.cuda.is_available()=false\n"


def test_inline_cuda_guard_accepts_cuda(monkeypatch, tmp_path: Path) -> None:
    marker = tmp_path / "cuda.failed"
    monkeypatch.setenv("LEROBOT_INLINE_CUDA_GUARD", "1")
    monkeypatch.setenv("LEROBOT_CUDA_GUARD_FAILURE_MARKER", str(marker))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    run_eval._run_inline_cuda_guard()

    assert not marker.exists()


class _FakeExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SkillExpertConfig(n_action_steps=2, chunk_size=2)
        self.calls = []
        self.predictions = [torch.tensor([4]), torch.tensor([8])]

    def reset(self):
        return None

    def predict_action_chunk(self, batch):
        self.calls.append(batch["skill_code"].detach().clone())
        batch_size = batch["skill_code"].shape[0]
        code = batch["skill_code"].float().view(batch_size, 1, 1)
        return code.expand(batch_size, 2, 1)

    def predict_skill_code(self, batch):
        del batch
        return self.predictions.pop(0).to(self.anchor.device)

    def get_optim_params(self):
        return []


class _FakeStage2Expert(_FakeExpert):
    name = "skill_vla_stage2"

    def __init__(self, stage2_mode: str = "likelihood"):
        super().__init__()
        self.config.stage2_mode = stage2_mode
        self.vlm_calls = []

    def predict_action_chunk(self, batch):
        self.vlm_calls.append(
            {
                "current_image": batch["observation.images.image"].detach().clone(),
                "start_image": batch["skill_start_image"].detach().clone(),
                "start_wrist": batch["skill_start_wrist_image"].detach().clone(),
                "start_tokens": batch["observation.language.tokens"].detach().clone(),
                "cache_id": batch[STAGE2_VLM_CACHE_ID].detach().clone(),
            }
        )
        return super().predict_action_chunk(batch)


class _FakeOracleStage2Expert(_FakeStage2Expert):
    def __init__(self):
        super().__init__("dsbc")
        self.config.skill_flow_latent_dim = 2
        self.oracle_targets = []
        self.mode_overrides = []

    def select_hindsight_mode_latent(
        self, batch, target_actions, target_valid, *, grid_size, timesteps
    ):
        self.oracle_targets.append(
            {
                "code": batch["skill_code"].detach().clone(),
                "actions": target_actions.detach().clone(),
                "valid": target_valid.detach().clone(),
                "grid_size": grid_size,
                "timesteps": timesteps,
            }
        )
        return torch.tensor([[0.25, -0.5]]), torch.tensor([[0.7, 0.2]])

    def predict_action_chunk(self, batch):
        self.mode_overrides.append(
            batch[STAGE2_MODE_LATENT_OVERRIDE].detach().clone()
        )
        self._last_eval_predicted_mode_latent = torch.tensor([[0.1, 0.2]])
        self._last_eval_mode_latent = batch[
            STAGE2_MODE_LATENT_OVERRIDE
        ].detach().clone()
        return super().predict_action_chunk(batch)


class _FakeOracleSkillStage2Expert(_FakeStage2Expert):
    def __init__(self):
        super().__init__("dsbc")
        self.oracle_skill_targets = []

    def select_hindsight_skill_code(
        self, batch, target_actions, target_valid, *, timesteps, aggregate_windows
    ):
        self.oracle_skill_targets.append(
            {
                "actions": target_actions.detach().clone(),
                "valid": target_valid.detach().clone(),
                "timesteps": timesteps,
                "aggregate_windows": aggregate_windows,
                "start_image": batch["skill_start_image"].detach().clone(),
                "current_image": batch["observation.images.image"].detach().clone(),
                "state": batch["observation.state"].detach().clone(),
            }
        )
        code = 9 + len(self.oracle_skill_targets) - 1
        scores = torch.arange(27, dtype=torch.float32).unsqueeze(0)
        scores[0, code] = -1.0
        return torch.tensor([code]), scores


class _FakeOracleStage1Expert(_FakeExpert):
    name = "skill_expert"

    def __init__(self):
        super().__init__()
        self.config.skill_flow_latent_dim = 2
        self.config.max_action_dim = 4
        self.oracle_targets = []
        self.mode_overrides = []
        self.noise_overrides = []

    def select_hindsight_mode_latent(
        self, batch, target_actions, target_valid, *, grid_size, timesteps
    ):
        self.oracle_targets.append(
            {
                "code": batch["skill_code"].detach().clone(),
                "actions": target_actions.detach().clone(),
                "valid": target_valid.detach().clone(),
                "grid_size": grid_size,
                "timesteps": timesteps,
            }
        )
        self._last_hindsight_mode_noise = torch.full((1, 2, 4), 0.75)
        return torch.tensor([[-0.5, 0.75]]), torch.tensor([[0.6, 0.1]])

    def predict_action_chunk(self, batch):
        self.mode_overrides.append(
            batch[SKILL_FLOW_MODE_LATENT_OVERRIDE].detach().clone()
        )
        self.noise_overrides.append(
            batch[SKILL_FLOW_NOISE_OVERRIDE].detach().clone()
        )
        self._last_eval_baseline_mode_latent = torch.tensor([[0.2, -0.1]])
        self._last_eval_mode_latent = batch[
            SKILL_FLOW_MODE_LATENT_OVERRIDE
        ].detach().clone()
        return super().predict_action_chunk(batch)


def test_attach_original_terminator_replaces_checkpoint_copy(
    monkeypatch, tmp_path: Path
) -> None:
    class _Terminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.ones(()))

    class _Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.model = SimpleNamespace(fsq_term_train=object())

    fsq_path = tmp_path / "FSQ.pt"
    fsq_path.touch()
    original = _Terminator()
    monkeypatch.setattr(run_eval, "build_fsq_terminator", lambda path: original)
    policy = _Policy()

    run_eval._attach_original_terminator(policy, fsq_path)

    assert policy.model.fsq_term_train is original
    assert original.training is False
    assert all(parameter.requires_grad is False for parameter in original.parameters())


@pytest.mark.parametrize("current_proprio,start_proprio,goal_xyz", [
    (None, True, False), (False, True, False), (False, False, True),
])
def test_external_terminator_is_rebuilt_from_its_saved_contract(
    monkeypatch, tmp_path: Path, current_proprio, start_proprio, goal_xyz
) -> None:
    checkpoint = tmp_path / "terminator"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        """{
          "type": "skill_aux",
          "train_terminator": true,
          "skill_fsq_levels": [3, 3, 3],
          "terminator_context": "prev_action",
          "terminator_arch": "fusion",
          "terminator_vision_backbone": "dino",
          "terminator_freeze_vision_encoder": false,
          "terminator_termination_only": true,
          "terminator_start_proprio": true,
          "terminator_proprio_conditioning": "adarms",
          "terminator_proprio_noise_magnitude": 0.02,
          "terminator_proprio_noise_distribution": "uniform",
          "terminator_proprio_noise_exclude_last_n": 2,
          "terminator_proprio_noise_clamp": true
        }"""
    )

    if current_proprio is not None:
        contract_path = checkpoint / "config.json"
        contract = json.loads(contract_path.read_text())
        contract.update(
            terminator_current_proprio=current_proprio,
            terminator_start_proprio=start_proprio,
            terminator_context="proprio",
            terminator_proprio_conditioning="tokens",
            terminator_proprio_noise_magnitude=0.0,
        )
        if goal_xyz:
            contract.update(terminator_goal_xyz=True, terminator_goal_noise_max_m=0.01)
        contract_path.write_text(json.dumps(contract))

    class _Terminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.ones(()))

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.fsq_term_train = _Terminator()

    built = _Terminator()
    builder_calls = []

    def _build(path, **kwargs):
        builder_calls.append((path, kwargs))
        return built

    loaded = []
    monkeypatch.setattr(
        modeling_skill_expert, "build_trainable_fsq_terminator", _build
    )
    monkeypatch.setattr(
        modeling_skill_expert,
        "_load_complete_terminator_parameters",
        lambda module, path: loaded.append((module, path)) or 1,
    )
    owner = SimpleNamespace(
        config=SimpleNamespace(
            fsq_path="/target/FSQ.pt", skill_fsq_levels=[3, 3, 3]
        ),
        model=_Model(),
    )

    SkillExpertPolicy.load_external_terminator(owner, checkpoint)

    assert builder_calls == [
        (
            "/target/FSQ.pt",
            {
                "termination_only": True,
                "progress_detach_backbone": None,
                "context": "prev_action" if current_proprio is None else "proprio",
                "cameras": "both",
                "default_arch": "fusion",
                "vision_backbone": "dino",
                "freeze_vision_encoder": False,
                "start_proprio": start_proprio,
                "proprio_conditioning": "adarms" if current_proprio is None else "tokens",
                "proprio_noise_magnitude": 0.02 if current_proprio is None else 0.0,
                "proprio_noise_distribution": "uniform",
                "proprio_noise_exclude_last_n": 2,
                "proprio_noise_clamp": True,
                **({} if current_proprio is None else {"current_proprio": current_proprio}),
                **({"goal_xyz": True} if goal_xyz else {}),
            },
        )
    ]
    assert loaded == [(built, checkpoint)]
    assert owner.model.fsq_term_train is built
    assert built.training is False


def test_normalize_advance_mode_accepts_original() -> None:
    assert run_eval._normalize_advance_mode("original") == "original"


class _FakeTerminator:
    use_wrist = True

    def __init__(self):
        self.step = 0

    def terminate(self, codes, state, image, wrist, previous_action=None):
        del state, image, wrist, previous_action
        self.step += 1
        probability = (
            torch.ones_like(codes, dtype=torch.float32)
            if self.step == 2
            else torch.zeros_like(codes, dtype=torch.float32)
        )
        return probability, probability


class _ConstantTerminator:
    requires_state = True

    def __init__(self, probability: float = 0.0):
        self.probability = probability

    def terminate(self, codes, state, image, wrist, previous_action=None):
        del state, image, wrist, previous_action
        values = torch.full_like(codes, self.probability, dtype=torch.float32)
        return values, values


def _batch():
    return {
        "observation.state": torch.zeros(1, 8),
        "skill_decoder_state": torch.zeros(1, 8),
        "skill_decoder_image": torch.zeros(1, 3, 8, 8),
        "skill_decoder_wrist": torch.zeros(1, 3, 8, 8),
    }


def _stage2_batch(value: int):
    batch = _batch()
    batch.update(
        {
            "observation.images.image": torch.full((1, 3, 8, 8), float(value)),
            "observation.images.wrist_image": torch.full(
                (1, 3, 8, 8), float(value + 10)
            ),
            "observation.language.tokens": torch.tensor([[value]], dtype=torch.long),
            "observation.language.attention_mask": torch.ones(
                1, 1, dtype=torch.bool
            ),
        }
    )
    return batch


def test_foveated_stage1_uses_exact_gt_focus_only_for_action_policy() -> None:
    class _RecordingExpert(_FakeExpert):
        def __init__(self):
            super().__init__()
            self.config.foveated_vision_enabled = True
            self.config.foveation_randomization_enabled = True
            self.config.foveation_mode = "crop"
            self.config.foveation_crop_size = 6
            self.config.foveation_output_size = 8
            self.config.foveation_inner_box_enabled = True
            self.config.foveation_inner_box_mode = "blur"
            self.config.foveation_inner_box_size = 2
            self.config.foveation_peripheral_blur_radius = 2.0
            self.action_images = []

        def predict_action_chunk(self, batch):
            self.action_images.append(
                (
                    batch["observation.images.image"].detach().clone(),
                    batch["observation.images.wrist_image"].detach().clone(),
                )
            )
            return super().predict_action_chunk(batch)

    expert = _RecordingExpert()
    wrapper = Stage1OraclePolicy(
        expert,
        None,
        advance_mode="gt",
        end_mode="max_length",
        end_threshold=0.5,
        progress_threshold=0.95,
        max_skill_length=0,
        n_action_steps=2,
    )
    wrapper.set_forced_skill_token_sequences(
        [[{"token": 3, "gt_length": 5, "focus_uv": [0.5, -0.5]}]]
    )
    wrapper.set_capture_vsa_top_inputs(True)
    yy, xx = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
    checker = ((xx + yy) % 2).float()
    top = torch.stack((checker, 1.0 - checker, checker))[None]
    wrist = torch.rand(1, 3, 8, 8)
    batch = _batch()
    batch["observation.images.image"] = top.clone()
    batch["observation.images.wrist_image"] = wrist.clone()

    wrapper.select_action(batch)

    policy_top, policy_wrist = expert.action_images[0]
    assert not torch.equal(policy_top, top)
    torch.testing.assert_close(policy_wrist, wrist)
    # The shared preprocessed batch remains untouched for predictors and
    # terminators; foveation is applied to an action-policy copy.
    torch.testing.assert_close(batch["observation.images.image"], top)
    assert wrapper.get_skill_trace()[0]["focus_uv"] == [0.5, -0.5]

    # The queued second action does not invoke the VSA again. Its displayed
    # condition therefore remains the exact top frame used for this plan.
    wrapper.select_action(batch)
    captured = wrapper.get_vsa_top_input_frames()
    captured_wrist = wrapper.get_vsa_wrist_input_frames()
    assert captured.shape == (1, 2, 8, 8, 3)
    assert captured_wrist.shape == (1, 2, 8, 8, 3)
    expected = (
        policy_top[0]
        .permute(1, 2, 0)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    np.testing.assert_array_equal(captured[0, 0], expected)
    np.testing.assert_array_equal(captured[0, 1], expected)
    expected_wrist = (
        policy_wrist[0]
        .permute(1, 2, 0)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    np.testing.assert_array_equal(captured_wrist[0, 0], expected_wrist)
    np.testing.assert_array_equal(captured_wrist[0, 1], expected_wrist)


@pytest.mark.parametrize(
    "architecture_label",
    [
        "arch9_1_skill",
        "arch9_2_skill",
        "arch10_1_skill",
        "arch10_2_skill",
        "arch11_1_skill",
        "arch11_2_skill",
        "arch12_1_skill",
        "arch12_2_skill",
    ],
)
def test_wrist_only_vsa_renders_black_top_input(
    architecture_label: str,
) -> None:
    expert = _FakeExpert()
    expert.config.architecture_label = architecture_label
    wrapper = Stage1OraclePolicy(
        expert,
        None,
        advance_mode="gt",
        end_mode="max_length",
        end_threshold=0.5,
        progress_threshold=0.95,
        max_skill_length=0,
        n_action_steps=2,
    )
    wrapper.set_forced_skill_token_sequences(
        [[{"token": 3, "gt_length": 5, "end_state": [1, 2, 3, 4, 5, 6]}]]
    )
    wrapper.set_capture_vsa_top_inputs(True)
    batch = _batch()
    batch["observation.images.image"] = torch.ones(1, 3, 8, 8)
    batch["observation.images.wrist_image"] = torch.rand(1, 3, 8, 8)

    wrapper.select_action(batch)

    captured_top = wrapper.get_vsa_top_input_frames()
    captured_wrist = wrapper.get_vsa_wrist_input_frames()
    assert captured_top.shape == (1, 1, 8, 8, 3)
    assert captured_wrist.shape == (1, 1, 8, 8, 3)
    assert not np.any(captured_top)
    assert np.any(captured_wrist)


@pytest.mark.parametrize("architecture_label", ["arch8_1_skill", "arch8_2_skill"])
def test_arch8_uses_original_agent_view_and_gt_uv_condition(architecture_label: str) -> None:
    class _RecordingExpert(_FakeExpert):
        def __init__(self):
            super().__init__()
            self.config.architecture_label = architecture_label
            self.action_inputs = []

        def predict_action_chunk(self, batch):
            self.action_inputs.append(
                (batch["observation.images.image"].clone(), batch["skill_focus_uv"].clone())
            )
            return super().predict_action_chunk(batch)

    expert = _RecordingExpert()
    wrapper = Stage1OraclePolicy(
        expert,
        None,
        advance_mode="gt",
        end_mode="max_length",
        end_threshold=0.5,
        progress_threshold=0.95,
        max_skill_length=0,
        n_action_steps=2,
    )
    wrapper.set_forced_skill_token_sequences(
        [[{"token": 3, "gt_length": 5, "focus_uv": [0.5, -0.5]}]]
    )
    batch = _batch()
    batch["observation.images.image"] = torch.rand(1, 3, 8, 8)
    batch["observation.images.wrist_image"] = torch.rand(1, 3, 8, 8)
    wrapper.select_action(batch)
    top, uv = expert.action_inputs[0]
    torch.testing.assert_close(top, batch["observation.images.image"])
    torch.testing.assert_close(uv, torch.tensor([[0.5, -0.5]]))


@pytest.mark.parametrize("architecture_label", ["arch8_1_skill", "arch8_2_skill"])
def test_arch8_uses_joint_predictor_uv_at_skill_start(architecture_label: str) -> None:
    class _PredictingExpert(_FakeExpert):
        def __init__(self):
            super().__init__()
            self.config.architecture_label = architecture_label
            self.received_uv = None

        def predict_skill_code_and_focus_uv(self, batch):
            del batch
            return torch.tensor([4]), torch.tensor([[-0.75, 0.25]])

        def predict_action_chunk(self, batch):
            self.received_uv = batch["skill_focus_uv"].clone()
            return super().predict_action_chunk(batch)

    expert = _PredictingExpert()
    wrapper = Stage1OraclePolicy(
        expert,
        None,
        skill_source="external",
        focus_source="predictor",
        advance_mode="gt",
        end_mode="max_length",
        end_threshold=0.5,
        progress_threshold=0.95,
        max_skill_length=0,
        n_action_steps=2,
    )
    wrapper.set_reference_skill_token_sequences(
        [[{"token": 3, "gt_length": 5, "focus_uv": [0.5, -0.5]}]]
    )
    batch = _batch()
    batch["observation.images.image"] = torch.rand(1, 3, 8, 8)
    batch["observation.images.wrist_image"] = torch.rand(1, 3, 8, 8)
    wrapper.select_action(batch)
    torch.testing.assert_close(expert.received_uv, torch.tensor([[-0.75, 0.25]]))


@pytest.mark.parametrize(
    "label",
    [
        "arch9_1_skill", "arch9_2_skill", "arch10_1_skill", "arch10_2_skill",
        "arch11_1_skill", "arch11_2_skill", "arch12_1_skill", "arch12_2_skill",
    ],
)
def test_arch9_uses_exact_gt_end_pose_for_wrist_only_vsa(label: str) -> None:
    class _RecordingExpert(_FakeExpert):
        def __init__(self):
            super().__init__()
            self.config.architecture_label = label
            self.config.skill_end_pose_mode = "pose"
            self.received = None

        def predict_action_chunk(self, batch):
            self.received = batch["skill_end_state"].clone()
            return super().predict_action_chunk(batch)

    expert = _RecordingExpert()
    wrapper = Stage1OraclePolicy(
        expert, None, advance_mode="gt", end_mode="max_length",
        end_threshold=0.5, progress_threshold=0.95,
        max_skill_length=0, n_action_steps=2,
    )
    wrapper.set_forced_skill_token_sequences(
        [[{"token": 3, "gt_length": 5, "end_state": [1, 2, 3, 4, 5, 6, 7, 8]}]]
    )
    wrapper.select_action(_batch())
    torch.testing.assert_close(expert.received, torch.tensor([[1., 2., 3., 4., 5., 6.]]))


@pytest.mark.parametrize(
    "label",
    [
        "arch9_1_skill", "arch9_2_skill", "arch10_1_skill", "arch10_2_skill",
        "arch11_1_skill", "arch11_2_skill", "arch12_1_skill", "arch12_2_skill",
    ],
)
def test_arch9_predictor_supplies_skill_and_end_xyz_together(label: str) -> None:
    class _PredictingExpert(_FakeExpert):
        def __init__(self):
            super().__init__()
            self.config.architecture_label = label
            self.config.skill_end_pose_mode = "xyz"
            self.received = None

        def predict_skill_code_and_end_state(self, batch):
            del batch
            return torch.tensor([4]), torch.tensor([[0.25, -0.5, 0.75]])

        def predict_action_chunk(self, batch):
            self.received = (batch["skill_code"].clone(), batch["skill_end_state"].clone())
            return super().predict_action_chunk(batch)

    expert = _PredictingExpert()
    wrapper = Stage1OraclePolicy(
        expert, None, skill_source="external", pose_source="predictor",
        advance_mode="gt", end_mode="max_length", end_threshold=0.5,
        progress_threshold=0.95, max_skill_length=0, n_action_steps=2,
    )
    wrapper.set_reference_skill_token_sequences([[{"token": 3, "gt_length": 5}]])
    wrapper.select_action(_batch())
    code, pose = expert.received
    torch.testing.assert_close(code, torch.tensor([4]))
    torch.testing.assert_close(pose, torch.tensor([[0.25, -0.5, 0.75]]))


def test_arch13_uses_gt_end_xyz_and_keeps_top_view() -> None:
    class _RecordingExpert(_FakeExpert):
        def __init__(self):
            super().__init__()
            self.config.architecture_label = "arch13_skill"
            self.config.skill_end_pose_mode = "xyz"
            self.received = None

        def predict_action_chunk(self, batch):
            self.received = (
                batch["skill_end_state"].clone(),
                batch["skill_end_xyz"].clone(),
                batch["observation.images.image"].clone(),
            )
            return super().predict_action_chunk(batch)

    expert = _RecordingExpert()
    wrapper = Stage1OraclePolicy(
        expert, None, advance_mode="gt", end_mode="max_length",
        end_threshold=0.5, progress_threshold=0.95,
        max_skill_length=0, n_action_steps=2,
    )
    wrapper.set_forced_skill_token_sequences(
        [[{"token": 3, "gt_length": 5, "end_state": [1, 2, 3, 4, 5, 6]}]]
    )
    batch = _batch()
    batch["observation.images.image"] = torch.rand(1, 3, 8, 8)
    wrapper.select_action(batch)
    end_state, end_xyz, top = expert.received
    torch.testing.assert_close(end_state, torch.tensor([[1.0, 2.0, 3.0]]))
    torch.testing.assert_close(end_xyz, torch.tensor([[1.0, 2.0, 3.0]]))
    torch.testing.assert_close(top, batch["observation.images.image"])


@pytest.mark.parametrize("label", ["arch19_skill", "arch20_skill"])
def test_arch19_latches_the_skill_start_and_arch20_needs_none(label: str) -> None:
    class _RecordingExpert(_FakeExpert):
        def __init__(self):
            super().__init__()
            self.config.architecture_label = label
            self.config.skill_end_pose_mode = "xyz"
            self.received = []

        def predict_action_chunk(self, batch):
            start = batch.get("skill_start_state")
            self.received.append((
                batch["skill_end_xyz"].clone(),
                None if start is None else start.clone(),
                batch["observation.images.image"].clone(),
            ))
            return super().predict_action_chunk(batch)

    expert = _RecordingExpert()
    wrapper = Stage1OraclePolicy(
        expert, None, advance_mode="gt", end_mode="max_length",
        end_threshold=0.5, progress_threshold=0.95,
        max_skill_length=0, n_action_steps=2,
    )
    wrapper.set_forced_skill_token_sequences(
        [[{"token": 3, "gt_length": 5, "end_state": [1, 2, 3, 4, 5, 6]}]]
    )
    for step in range(3):                       # plans at steps 0 and 2 (two-action chunks)
        batch = _batch()
        batch["skill_decoder_state"] = torch.full((1, 8), float(step + 1))
        batch["observation.images.image"] = torch.rand(1, 3, 8, 8)
        wrapper.select_action(batch)
    assert len(expert.received) == 2
    for end_xyz, start, top in expert.received:
        torch.testing.assert_close(end_xyz, torch.tensor([[1.0, 2.0, 3.0]]))
        assert top.abs().sum() > 0                                   # top view stays visible
        if label.startswith("arch19"):
            # Latched at the skill's first plan and held for its later chunks.
            torch.testing.assert_close(start, torch.full((1, 8), 1.0))
        else:
            assert start is None


@pytest.mark.parametrize("label", [
    "fullskill_vis_1", "fullskill_vis_1_firstlayer", "fullskill_vis_1_lastlayer",
    "fullskill_vis_2_firstlayer", "fullskill_vis_2_lastlayer",
])
def test_fullskill_visual_models_latch_start_cameras_state_and_refresh_equal_codes(label):
    class RecordingExpert(_FakeExpert):
        def __init__(self):
            super().__init__()
            self.config.architecture_label = label
            self.config.skill_end_pose_mode = "xyz"
            self.received = []

        def predict_action_chunk(self, batch):
            self.received.append({key: batch[key].clone() for key in (
                "skill_start_image", "skill_start_wrist_image", "skill_start_state_normalized",
                "skill_start_cache_id", "skill_start_state", "skill_end_state",
                "observation.images.image")})
            return super().predict_action_chunk(batch)

    expert = RecordingExpert()
    wrapper = Stage1OraclePolicy(
        expert, None, advance_mode="gt", end_mode="max_length", end_threshold=.5,
        progress_threshold=.95, max_skill_length=0, n_action_steps=1,
        reject_same_skill_transition=False,
    )
    wrapper.set_forced_skill_token_sequences([[{
        "token": 3, "gt_length": length, "end_state": [1, 2, 3, 4, 5, 6],
    } for length in [3, 4]]])
    for step in range(7):
        batch = _batch()
        batch["skill_decoder_state"] = torch.full((1, 8), float(step + 1))
        batch["observation.state"] = torch.full((1, 8), float(step + 10))
        batch["observation.images.image"] = torch.full((1, 3, 8, 8), step / 10.)
        batch["observation.images.wrist_image"] = torch.full((1, 3, 8, 8), step / 10. + .01)
        wrapper.select_action(batch)
    groups = {}
    for record in expert.received:
        occurrence = record["skill_start_cache_id"].item()
        groups.setdefault(occurrence, []).append(record)
    assert set(groups) == {0, 1}
    for records in groups.values():
        assert len(records) > 1
        for record in records:
            for key in ("skill_start_image", "skill_start_wrist_image", "skill_start_state_normalized", "skill_start_state"):
                torch.testing.assert_close(record[key], records[0][key])
    assert not torch.equal(groups[0][0]["skill_start_image"], groups[1][0]["skill_start_image"])
    assert not torch.equal(groups[0][-1]["observation.images.image"], groups[0][0]["observation.images.image"])


@pytest.mark.parametrize("batch_size", [1, 2])
def test_preskill_plan_records_last_observed_frame_not_last_chunk_plan_and_resets(batch_size):
    from lerobot.policies.skillVLA import preskill_plan as memory

    class RecordingExpert(_FakeExpert):
        def __init__(self):
            super().__init__()
            self.config.architecture_label = "fullskill_bspline_3_lastlayer_lastvis_current_preskill"
            self.config.skill_end_pose_mode = "xyz"
            self.received = []

        def predict_action_chunk(self, batch):
            self.received.append({key: value.clone() for key, value in batch.items()
                                  if key in memory.KEYS or key == "skill_code"})
            return super().predict_action_chunk(batch)

    expert = RecordingExpert()
    wrapper = Stage1OraclePolicy(
        expert, None, advance_mode="gt", end_mode="max_length", end_threshold=.5,
        progress_threshold=.95, max_skill_length=0, n_action_steps=2,
        reject_same_skill_transition=False,
    )
    sequences = [[{"token": code, "gt_length": 4 + 2 * index, "end_state": [code, 2, 3, 4, 5, 6]}
                  for code in [3, 3]] for index in range(batch_size)]
    wrapper.set_forced_skill_token_sequences(sequences)
    for step in range(8 + 4 * (batch_size - 1)):
        batch = {key: value.expand(batch_size, *value.shape[1:]).clone() for key, value in _batch().items()}
        batch["skill_decoder_state"] = torch.full((batch_size, 8), float(step + 1))
        batch["observation.state"] = torch.full((batch_size, 8), float(step + 10))
        batch["observation.images.image"] = torch.full((batch_size, 3, 8, 8), step / 10.)
        batch["observation.images.wrist_image"] = torch.full((batch_size, 3, 8, 8), step / 10. + .01)
        wrapper.select_action(batch)
    for index in range(batch_size):
        bos = [row for row in expert.received if row[memory.CACHE_ID][index].item() == 0]
        past = [row for row in expert.received if row[memory.CACHE_ID][index].item() == 1]
        assert bos and past
        assert all(not row[memory.VALID][index].item() for row in bos)
        first = past[0]
        assert first[memory.VALID][index].item()
        # The preceding observation, NOT the last chunk-planning observation.
        transition_step = next(row["episode_timestep"] for row in wrapper.get_skill_trace()
                               if row["skill_index"] == 1 and row["batch_index"] == index)
        expected = transition_step - 1
        torch.testing.assert_close(first[memory.IMAGE][index], torch.full((3, 8, 8), expected / 10.))
        torch.testing.assert_close(first[memory.STATE_NORMALIZED][index], torch.full((8,), float(expected + 10)))
        torch.testing.assert_close(first[memory.START_STATE][index], torch.ones(8))
        assert first[memory.CODE][index].item() == 3  # equal-code occurrences still refresh
        for row in past[1:]:
            torch.testing.assert_close(row[memory.IMAGE][index], first[memory.IMAGE][index])
    wrapper.reset()
    assert all(value is None for value in wrapper._preskill_inputs)
    assert all(value is None for value in wrapper._preskill_last_observations)


def test_foveated_predictor_does_not_advance_past_last_gt_focus() -> None:
    class _FoveatedPredictor(_FakeExpert):
        def __init__(self):
            super().__init__()
            self.config.foveated_vision_enabled = True
            self.config.foveation_randomization_enabled = False
            self.config.foveation_mode = "crop"
            self.config.foveation_crop_size = 6
            self.config.foveation_output_size = 8
            self.config.foveation_inner_box_enabled = False
            self.config.foveation_inner_box_size = 2

    class _AlwaysEndingTerminator:
        requires_state = True

        def terminate(self, codes, state, image, wrist, previous_action=None):
            del state, image, wrist, previous_action
            ones = torch.ones_like(codes, dtype=torch.float32)
            return ones, ones

    wrapper = Stage1OraclePolicy(
        _FoveatedPredictor(),
        _AlwaysEndingTerminator(),
        skill_source="external",
        advance_mode="external",
        end_mode="termination",
        end_threshold=0.5,
        progress_threshold=0.95,
        max_skill_length=10,
        n_action_steps=1,
        immediate_replan_on_skill_end=True,
    )
    wrapper.set_reference_skill_token_sequences(
        [[{"token": 3, "gt_length": 5, "focus_uv": [0.0, 0.0]}]]
    )
    batch = _batch()
    batch.update(
        {
            run_eval.RAW_STATE: torch.zeros(1, 8),
            run_eval.RAW_IMAGE: torch.zeros(1, 3, 8, 8),
            run_eval.RAW_WRIST: torch.zeros(1, 3, 8, 8),
            "observation.images.image": torch.zeros(1, 3, 8, 8),
            "observation.images.wrist_image": torch.zeros(1, 3, 8, 8),
        }
    )

    wrapper.select_action(batch)

    assert len(wrapper.get_skill_trace()) == 1
    assert wrapper.get_skill_trace()[0]["focus_uv"] == [0.0, 0.0]
    assert wrapper.get_episode_done() == [True]


def test_predictor_does_not_advance_past_last_gt_end_pose() -> None:
    class _EndPoseExpert(_FakeExpert):
        def __init__(self):
            super().__init__()
            self.config.architecture_label = "arch18_skill"
            self.config.skill_end_pose_mode = "xyz"

    class _AlwaysEndingTerminator:
        requires_state = True

        def terminate(self, codes, state, image, wrist, previous_action=None):
            del state, image, wrist, previous_action
            ones = torch.ones_like(codes, dtype=torch.float32)
            return ones, ones

    wrapper = Stage1OraclePolicy(
        _EndPoseExpert(),
        _AlwaysEndingTerminator(),
        skill_source="external",
        pose_source="gt",
        advance_mode="external",
        end_mode="termination",
        end_threshold=0.5,
        progress_threshold=0.95,
        max_skill_length=10,
        n_action_steps=1,
        immediate_replan_on_skill_end=True,
    )
    wrapper.set_reference_skill_token_sequences(
        [[{
            "token": 3,
            "gt_length": 5,
            "end_state": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
        }]]
    )

    wrapper.select_action(_batch())

    assert len(wrapper.get_skill_trace()) == 1
    assert wrapper.get_skill_trace()[0]["end_pose"] == pytest.approx(
        [0.1, 0.2, 0.3]
    )
    assert wrapper.get_episode_done() == [True]


def test_foveated_predictor_uses_joint_predicted_skill_and_focus_per_boundary() -> None:
    class _JointFoveatedPredictor(_FakeExpert):
        def __init__(self):
            super().__init__()
            self.config.foveated_vision_enabled = True
            self.config.foveation_randomization_enabled = False
            self.config.foveation_mode = "crop"
            self.config.foveation_crop_size = 6
            self.config.foveation_output_size = 8
            self.config.foveation_inner_box_enabled = False
            self.config.foveation_inner_box_size = 2
            self.joint_predictions = [
                (torch.tensor([4]), torch.tensor([[-0.75, 0.25]])),
                (torch.tensor([8]), torch.tensor([[0.5, -0.5]])),
            ]

        def predict_skill_code_and_focus_uv(self, batch):
            del batch
            return self.joint_predictions.pop(0)

    class _AlwaysEndingTerminator:
        requires_state = True

        def terminate(self, codes, state, image, wrist, previous_action=None):
            del state, image, wrist, previous_action
            ones = torch.ones_like(codes, dtype=torch.float32)
            return ones, ones

    expert = _JointFoveatedPredictor()
    wrapper = Stage1OraclePolicy(
        expert,
        _AlwaysEndingTerminator(),
        skill_source="external",
        focus_source="predictor",
        advance_mode="external",
        end_mode="termination",
        end_threshold=0.5,
        progress_threshold=0.95,
        max_skill_length=10,
        n_action_steps=1,
        immediate_replan_on_skill_end=True,
    )
    # References remain useful for reporting, but predictor focus no longer
    # consumes their exact demonstration UV values.
    wrapper.set_reference_skill_token_sequences(
        [[{"token": 3, "gt_length": 5}, {"token": 6, "gt_length": 5}]]
    )
    batch = _batch()
    batch.update(
        {
            run_eval.RAW_STATE: torch.zeros(1, 8),
            run_eval.RAW_IMAGE: torch.zeros(1, 3, 8, 8),
            run_eval.RAW_WRIST: torch.zeros(1, 3, 8, 8),
            "observation.images.image": torch.zeros(1, 3, 8, 8),
            "observation.images.wrist_image": torch.zeros(1, 3, 8, 8),
        }
    )

    wrapper.select_action(batch)

    trace = wrapper.get_skill_trace()
    assert [row["codebook_token"] for row in trace] == [4, 8]
    assert [row["focus_uv"] for row in trace] == [[-0.75, 0.25], [0.5, -0.5]]
    assert [row["focus_source"] for row in trace] == ["predictor", "predictor"]
    assert expert.joint_predictions == []


def test_stage1_eval_json_paths_are_collected_under_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("TASK_TAG", "t0-1")
    panel_root = tmp_path / "eval_run" / "panels" / "00_arch0"

    assert run_eval._panel_cache_path(panel_root) == (
        tmp_path
        / "eval_run"
        / "metrics"
        / "panel_cache"
        / "00_arch0"
        / "eval_info_t0-1.json"
    )
    assert run_eval._legacy_panel_cache_path(panel_root) == (
        panel_root / "eval_info_t0-1.json"
    )


def test_stage2_eval_does_not_reuse_artifacts_after_contract_change(
    tmp_path: Path, monkeypatch
) -> None:
    for key, value in {
        "SKILL_END_MODE": "or",
        "SKILL_END_THRESHOLD": "0.5",
        "SKILL_END_PROGRESS_THRESHOLD": "0.9",
        "INFERENCE_SKILL_MAX_LENGTH": "200",
        "IMMEDIATE_REPLAN_ON_SKILL_END": "true",
    }.items():
        monkeypatch.setenv(key, value)
    panel_root = tmp_path / "eval_run/panels/00_dsbc"
    video = panel_root / "videos/libero_90_0/eval_episode_0.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    cfg = SimpleNamespace(
        eval=SimpleNamespace(
            n_episodes=1,
            max_videos_per_task=1,
            skill_html=False,
        ),
        policy=SimpleNamespace(n_action_steps=5),
        seed=1000,
    )
    spec = {
        "policy_path": "/tmp/stage2",
        "mode": "stage2",
        "stage2_mode": "dsbc",
        "dsbc_noise_output_mode": "shared",
        "skill_source": "gt",
        "advance_mode": "gt",
    }
    cache = run_eval._panel_cache_path(panel_root)
    cache.parent.mkdir(parents=True)
    cache.write_text(
        '{"signature":{"stage2_mode":"likelihood"},"info":{"stale":true}}'
    )

    info, source = run_eval._load_resumed_panel_info(
        panel_root, spec, {"libero_90_0"}, cfg
    )

    assert info is None
    assert source is None


@pytest.mark.parametrize("old_group,new_group", [
    ("skillVLA_stage1", "skillVLA_PT"), ("skillVLA_NewTask_FT", "skillVLA_FT"),
])
def test_panel_resume_keeps_metrics_after_pt_output_rename(tmp_path, monkeypatch, old_group, new_group):
    for key, value in {
        "SKILL_END_MODE": "termination",
        "SKILL_END_THRESHOLD": "0.3",
        "SKILL_END_PROGRESS_THRESHOLD": "0.95",
        "INFERENCE_SKILL_MAX_LENGTH": "200",
    }.items():
        monkeypatch.setenv(key, value)
    cfg = SimpleNamespace(eval=SimpleNamespace(n_episodes=1), policy=SimpleNamespace(n_action_steps=5), seed=1000)
    old_policy = tmp_path / "outputs_filtered" / old_group / "VSA/run/checkpoints/100000/pretrained_model"
    new_policy = tmp_path / "outputs_filtered" / new_group / "VSA/run/checkpoints/100000/pretrained_model"
    new_policy.mkdir(parents=True)
    old_spec = {"policy_path": str(old_policy), "skill_source": "gt", "advance_mode": "gt"}
    panel_root = tmp_path / "eval_run/panels/run"
    cache = run_eval._panel_cache_path(panel_root)
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"signature": run_eval._panel_signature(old_spec, {"task"}, cfg), "info": {"sr": 0.8}}))
    monkeypatch.setattr(run_eval, "_panel_artifacts_complete", lambda *args: True)
    spec = {**old_spec, "policy_path": str(new_policy)}
    assert run_eval._load_resumed_panel_info(panel_root, spec, {"task"}, cfg) == ({"sr": 0.8}, "metrics cache")
    # Path aliases must not hide a changed evaluation contract.
    assert run_eval._load_resumed_panel_info(panel_root, {**spec, "skill_source": "own"}, {"task"}, cfg) == (None, None)
    # If both paths exist, they can be different checkpoints: do not merge them.
    old_policy.mkdir(parents=True)
    assert run_eval._load_resumed_panel_info(panel_root, spec, {"task"}, cfg) == (None, None)


@pytest.mark.parametrize("old_contract", [None, "per_skill_cap_v0", "unversioned_artifacts"])
def test_panel_resume_rejects_old_final_skill_timeout_contract(tmp_path, monkeypatch, old_contract):
    for key, value in {
        "SKILL_END_MODE": "termination", "SKILL_END_THRESHOLD": "0.5",
        "SKILL_END_PROGRESS_THRESHOLD": "0.95", "INFERENCE_SKILL_MAX_LENGTH": "200",
    }.items():
        monkeypatch.setenv(key, value)
    cfg = SimpleNamespace(eval=SimpleNamespace(n_episodes=1), policy=SimpleNamespace(n_action_steps=5), seed=1000)
    spec = {"policy_path": "/model", "skill_source": "gt", "advance_mode": "external"}
    panel_root = tmp_path / "eval_run/panels/run"
    cache = run_eval._panel_cache_path(panel_root)
    cache.parent.mkdir(parents=True)
    monkeypatch.setattr(run_eval, "_panel_artifacts_complete", lambda *args: True)
    signature = run_eval._panel_signature(spec, {"task"}, cfg)
    assert signature["final_skill_completion_contract"] == "success_or_episode_horizon_v1"
    if old_contract != "unversioned_artifacts":
        if old_contract is None:
            signature.pop("final_skill_completion_contract")
        else:
            signature["final_skill_completion_contract"] = old_contract
        cache.write_text(json.dumps({"signature": signature, "info": {"sr": 0.8}}))
    assert run_eval._load_resumed_panel_info(panel_root, spec, {"task"}, cfg) == (None, None)
    cache.write_text(json.dumps({"signature": run_eval._panel_signature(spec, {"task"}, cfg), "info": {"sr": 0.9}}))
    assert run_eval._load_resumed_panel_info(panel_root, spec, {"task"}, cfg) == ({"sr": 0.9}, "metrics cache")


def test_resumed_panel_assets_follow_the_renamed_eval_folder(tmp_path):
    panel = tmp_path / "FT/eval/outputs/eval_run/panels/model"
    new_video = panel / "videos/task/episode.mp4"
    new_video.parent.mkdir(parents=True)
    new_video.touch()
    old_video = tmp_path / "NewTask_FT/eval/outputs/eval_run/panels/model/videos/task/episode.mp4"
    info = {"overall": {"sr": 0.8, "video_paths": [str(old_video)]}, "label": "unchanged"}
    assert run_eval._relocate_panel_info_paths(info, panel) == {
        "overall": {"sr": 0.8, "video_paths": [str(new_video)]}, "label": "unchanged",
    }
    assert info["overall"]["video_paths"] == [str(old_video)]  # Archive is not mutated.
    old_video.parent.mkdir(parents=True)
    old_video.touch()
    assert run_eval._relocate_panel_info_paths(info, panel) == info


def test_video_skill_timeline_tracks_trace_at_render_stride() -> None:
    trace = [
        {"batch_index": 0, "episode_timestep": 0, "codebook_token": 2},
        {"batch_index": 1, "episode_timestep": 0, "codebook_token": 9},
        {"batch_index": 0, "episode_timestep": 3, "codebook_token": 7},
        {"batch_index": 0, "episode_timestep": 8, "codebook_token": 4},
    ]

    assert _skill_ids_from_trace(
        trace,
        batch_index=0,
        n_video_frames=6,
        video_frame_stride=2,
    ) == [2, 2, 7, 7, 4, 4]


def test_video_latent_timeline_tracks_active_skill_at_render_stride() -> None:
    trace = [
        {
            "batch_index": 0,
            "episode_timestep": 0,
            "predicted_latent": [0.1, 0.2],
        },
        {
            "batch_index": 0,
            "episode_timestep": 3,
            "predicted_latent": [-0.7, 0.8],
        },
    ]

    np.testing.assert_allclose(
        _latent_values_from_trace(
            trace,
            field="predicted_latent",
            batch_index=0,
            n_video_frames=3,
            video_frame_stride=2,
        ),
        [[0.1, 0.2], [0.1, 0.2], [-0.7, 0.8]],
    )


def test_video_latent_timeline_tracks_each_action_chunk() -> None:
    trace = [
        {
            "batch_index": 0,
            "episode_timestep": 0,
            # Backward-compatible snapshot is the final chunk value and must
            # not overwrite the temporal points during rendering.
            "predicted_latent": [0.9, 0.9],
            "latent_points": [
                {"episode_timestep": 0, "predicted_latent": [0.1, 0.2]},
                {"episode_timestep": 5, "predicted_latent": [-0.4, 0.6]},
                {"episode_timestep": 10, "predicted_latent": [0.9, 0.9]},
            ],
        }
    ]

    np.testing.assert_allclose(
        _latent_values_from_trace(
            trace,
            field="predicted_latent",
            batch_index=0,
            n_video_frames=7,
            video_frame_stride=2,
        ),
        [
            [0.1, 0.2],
            [0.1, 0.2],
            [0.1, 0.2],
            [-0.4, 0.6],
            [-0.4, 0.6],
            [0.9, 0.9],
            [0.9, 0.9],
        ],
    )


def test_video_latent_panel_draws_predicted_and_oracle_points() -> None:
    frames = np.zeros((2, 120, 160, 3), dtype=np.uint8)
    annotated = _annotate_eval_video(
        frames,
        success=True,
        task_description=None,
        predicted_latents=[[0.0, 0.0], [0.0, 0.0]],
        oracle_latents=[[1.0, -1.0], [1.0, -1.0]],
    )

    top_h = max(18, 120 // 10)
    latent_y0 = top_h + 120
    latent_h = max(72, 120 // 3)
    square_size = max(42, latent_h - 28)
    square_y0 = (latent_h - square_size) // 2
    predicted_xy = (10 + square_size // 2, square_y0 + square_size // 2)
    oracle_xy = (10 + square_size, square_y0 + square_size)

    assert annotated.shape == (2, top_h + 120 + latent_h, 160, 3)
    assert tuple(annotated[0, latent_y0 + predicted_xy[1], predicted_xy[0]]) == (
        52,
        152,
        219,
    )
    assert tuple(annotated[0, latent_y0 + oracle_xy[1], oracle_xy[0]]) == (
        255,
        159,
        67,
    )


def test_video_places_actual_vsa_top_input_beside_rollout() -> None:
    frames = np.zeros((2, 120, 160, 3), dtype=np.uint8)
    vsa_top = np.full((2, 60, 80, 3), 255, dtype=np.uint8)

    annotated = _annotate_eval_video(
        frames,
        success=True,
        task_description=None,
        vsa_top_frames=vsa_top,
    )

    top_h = max(18, 120 // 10)
    assert annotated.shape == (2, top_h + 120, 320, 3)
    assert tuple(annotated[0, top_h + 100, 80]) == (0, 0, 0)
    assert tuple(annotated[0, top_h + 100, 160 + 80]) == (255, 255, 255)


@pytest.mark.parametrize(
    "architecture_label",
    [
        "wristonly_1",
        "wristonly_2",
        "wristonly_lit_1",
        "wristonly_lit_4",
        "wristonly_lit_5",
        "arch18_align_skill",
    ],
)
def test_wrist_only_architectures_black_out_the_vsa_top_panel(
    architecture_label: str,
) -> None:
    assert not run_eval._architecture_uses_vsa_top_view(architecture_label)


@pytest.mark.parametrize(
    "architecture_label", ["both_1", "both_3", "both_4", "both_5", "both_lit_4", "both_lit_4_dedicated", "both_lit_5", "both_lit_6", "both_lit_7", "arch13",
                           *run_eval.DIRECT_PATCH_ARCH_LABELS]
)
def test_both_and_top_view_architectures_keep_the_vsa_top_panel(
    architecture_label: str,
) -> None:
    assert run_eval._architecture_uses_vsa_top_view(architecture_label)


def test_video_places_actual_vsa_wrist_input_after_top_input() -> None:
    frames = np.zeros((2, 120, 160, 3), dtype=np.uint8)
    vsa_top = np.full((2, 60, 80, 3), 127, dtype=np.uint8)
    vsa_wrist = np.full((2, 60, 80, 3), 255, dtype=np.uint8)

    annotated = _annotate_eval_video(
        frames,
        success=True,
        task_description=None,
        vsa_top_frames=vsa_top,
        vsa_wrist_frames=vsa_wrist,
    )

    top_h = max(18, 120 // 10)
    assert annotated.shape == (2, top_h + 120, 480, 3)
    assert tuple(annotated[0, top_h + 100, 80]) == (0, 0, 0)
    assert tuple(annotated[0, top_h + 100, 160 + 80]) == (127, 127, 127)
    assert tuple(annotated[0, top_h + 100, 320 + 80]) == (255, 255, 255)


def test_video_skill_banner_changes_color_with_active_skill() -> None:
    frames = np.zeros((4, 120, 160, 3), dtype=np.uint8)
    annotated = _annotate_eval_video(
        frames,
        success=True,
        task_description="pick up the object",
        skill_ids=[3, 3, 8, 8],
    )

    assert annotated.shape[0] == len(frames)
    skill_banner_y = max(18, 120 // 10) + 120
    assert tuple(annotated[0, skill_banner_y, 0]) == _skill_banner_color(3)
    assert tuple(annotated[1, skill_banner_y, 0]) == _skill_banner_color(3)
    assert tuple(annotated[2, skill_banner_y, 0]) == _skill_banner_color(8)
    assert tuple(annotated[0, -1, 0]) == (20, 20, 20)
    assert _skill_banner_color(3) != _skill_banner_color(8)


def test_video_progress_gauge_tracks_terminator_trace_at_render_stride() -> None:
    trace = [
        {
            "batch_index": 0,
            "episode_timestep": 0,
            "end_probs": [
                {"episode_timestep": 0, "skill_step": 0, "progress": 0.1},
                {"episode_timestep": 1, "skill_step": 1, "progress": 0.2},
                {"episode_timestep": 2, "skill_step": 2, "progress": 0.4},
            ],
        },
        {
            "batch_index": 0,
            "episode_timestep": 3,
            "end_probs": [
                {"episode_timestep": 3, "skill_step": 0, "progress": 0.7},
                {"episode_timestep": 4, "skill_step": 1, "progress": 0.95},
            ],
        },
    ]

    assert _progress_values_from_trace(
        trace,
        batch_index=0,
        n_video_frames=3,
        video_frame_stride=2,
    ) == pytest.approx([0.1, 0.4, 0.95])


def test_video_progress_values_add_right_panel() -> None:
    frames = np.zeros((3, 120, 160, 3), dtype=np.uint8)
    annotated = _annotate_eval_video(
        frames,
        success=True,
        task_description="pick up the object",
        skill_ids=[3, 3, 8],
        progress_values=[0.1, 0.5, 0.95],
        progress_threshold=0.9,
    )

    gauge_width = max(48, 160 // 6)
    assert annotated.shape[2] == 160 + gauge_width


def test_video_termination_gauge_latches_until_skill_transition() -> None:
    trace = [
        {
            "batch_index": 0,
            "episode_timestep": 0,
            "end_probs": [
                {"episode_timestep": 0, "skill_step": 0, "prob": 0.2},
                {"episode_timestep": 1, "skill_step": 1, "prob": 0.7},
                {"episode_timestep": 2, "skill_step": 2, "prob": 0.3},
            ],
        },
        {
            "batch_index": 0,
            "episode_timestep": 3,
            "end_probs": [
                {"episode_timestep": 3, "skill_step": 0, "prob": 0.1},
            ],
        },
    ]

    assert _termination_values_from_trace(
        trace,
        batch_index=0,
        n_video_frames=4,
        video_frame_stride=1,
        end_threshold=0.5,
    ) == pytest.approx([0.2, 0.7, 0.7, 0.1])


def test_video_progress_values_add_gauge_beside_termination_gauge() -> None:
    frames = np.zeros((3, 120, 160, 3), dtype=np.uint8)
    annotated = _annotate_eval_video(
        frames,
        success=True,
        task_description="pick up the object",
        skill_ids=[3, 3, 8],
        progress_values=[0.1, 0.5, 0.95],
        progress_threshold=0.9,
        termination_values=[0.2, 0.7, 0.7],
        end_threshold=0.5,
    )

    gauge_width = max(48, 160 // 6)
    assert annotated.shape[2] == 160 + 2 * gauge_width


def test_policy_config_enforces_retained_arch0_contract(monkeypatch) -> None:
    loaded = SimpleNamespace(
        type="skill_expert",
        architecture="cond_gemma",
        architecture_label="arch0_skill",
        architecture_revision="skillvla_real_v1",
        conditioning_route="state_cond",
    )
    monkeypatch.setattr(
        run_eval.PreTrainedConfig,
        "from_pretrained",
        lambda *args, **kwargs: loaded,
    )

    result = run_eval._policy_config(
        {
            "policy_path": "/tmp/current-stage1",
            "architecture": "cond_gemma",
            "architecture_label": "arch0_skill",
            "architecture_revision": "skillvla_real_v1",
            "conditioning_route": "state_cond",
            "fsq_path": "/tmp/fsq",
            "dino_model_path": "/tmp/dino",
            "tokenizer_path": "/tmp/tokenizer",
        },
        SimpleNamespace(use_amp=False, n_action_steps=5),
        torch.device("cpu"),
    )

    assert result.architecture == "cond_gemma"
    assert result.architecture_label == "arch0_skill"
    assert result.architecture_revision == "skillvla_real_v1"
    assert result.conditioning_route == "state_cond"


def test_policy_config_enforces_retained_arch1_contract(monkeypatch) -> None:
    loaded = SimpleNamespace(
        type="skill_expert",
        architecture="fixed_visual_bottleneck",
        architecture_label="arch1_skill",
        architecture_revision="fixed_visual_bottleneck_v1",
        vision_conditioning_mode="fixed_bottleneck_cross_attention",
        conditioning_route="state_cond",
    )
    monkeypatch.setattr(
        run_eval.PreTrainedConfig,
        "from_pretrained",
        lambda *args, **kwargs: loaded,
    )

    result = run_eval._policy_config(
        {
            "policy_path": "/tmp/current-stage1",
            "architecture": "fixed_visual_bottleneck",
            "architecture_label": "arch1_skill",
            "architecture_revision": "fixed_visual_bottleneck_v1",
            "vision_conditioning_mode": "fixed_bottleneck_cross_attention",
            "conditioning_route": "state_cond",
            "fsq_path": "/tmp/fsq",
            "dino_model_path": "/tmp/dino",
            "tokenizer_path": "/tmp/tokenizer",
        },
        SimpleNamespace(use_amp=False, n_action_steps=5),
        torch.device("cpu"),
    )

    assert result.architecture == "fixed_visual_bottleneck"
    assert result.architecture_label == "arch1_skill"
    assert result.architecture_revision == "fixed_visual_bottleneck_v1"
    assert result.vision_conditioning_mode == "fixed_bottleneck_cross_attention"


def test_policy_config_enforces_arch2_contract(monkeypatch) -> None:
    loaded = SimpleNamespace(
        type="skill_expert",
        architecture="fixed_visual_bottleneck",
        architecture_label="arch2_skill",
        architecture_revision="late_visual_bottleneck_v1",
        vision_conditioning_mode="fixed_bottleneck_cross_attention",
        conditioning_route="state_cond",
    )
    monkeypatch.setattr(
        run_eval.PreTrainedConfig,
        "from_pretrained",
        lambda *args, **kwargs: loaded,
    )

    result = run_eval._policy_config(
        {
            "policy_path": "/tmp/current-stage1",
            "architecture": "fixed_visual_bottleneck",
            "architecture_label": "arch2_skill",
            "architecture_revision": "late_visual_bottleneck_v1",
            "vision_conditioning_mode": "fixed_bottleneck_cross_attention",
            "conditioning_route": "state_cond",
            "fsq_path": "/tmp/fsq",
            "dino_model_path": "/tmp/dino",
            "tokenizer_path": "/tmp/tokenizer",
        },
        SimpleNamespace(use_amp=False, n_action_steps=5),
        torch.device("cpu"),
    )

    assert result.architecture == "fixed_visual_bottleneck"
    assert result.architecture_label == "arch2_skill"
    assert result.architecture_revision == "late_visual_bottleneck_v1"


def test_policy_config_enforces_arch3_contract(monkeypatch) -> None:
    loaded = SimpleNamespace(
        type="skill_expert",
        architecture="layerwise_cond_bottleneck",
        architecture_label="arch3_skill",
        architecture_revision="layerwise_cond_bottleneck_v1",
        vision_conditioning_mode="layerwise_cond_bottleneck_cross_attention",
        visual_bridge_last_n_layers=6,
        conditioning_route="state_cond",
    )
    monkeypatch.setattr(
        run_eval.PreTrainedConfig,
        "from_pretrained",
        lambda *args, **kwargs: loaded,
    )

    result = run_eval._policy_config(
        {
            "policy_path": "/tmp/current-stage1",
            "architecture": "layerwise_cond_bottleneck",
            "architecture_label": "arch3_skill",
            "architecture_revision": "layerwise_cond_bottleneck_v1",
            "vision_conditioning_mode": "layerwise_cond_bottleneck_cross_attention",
            "visual_bridge_last_n_layers": 6,
            "conditioning_route": "state_cond",
            "fsq_path": "/tmp/fsq",
            "dino_model_path": "/tmp/dino",
            "tokenizer_path": "/tmp/tokenizer",
        },
        SimpleNamespace(use_amp=False, n_action_steps=5),
        torch.device("cpu"),
    )

    assert result.architecture == "layerwise_cond_bottleneck"
    assert result.architecture_label == "arch3_skill"
    assert result.architecture_revision == "layerwise_cond_bottleneck_v1"
    assert result.visual_bridge_last_n_layers == 6


def test_policy_config_enforces_arch4_contract(monkeypatch) -> None:
    loaded = SimpleNamespace(
        type="skill_expert",
        architecture="layerwise_cond_bottleneck",
        architecture_label="arch4_skill",
        architecture_revision="layerwise_cond_bottleneck_core_exit_v1",
        vision_conditioning_mode="layerwise_cond_bottleneck_cross_attention",
        visual_bridge_last_n_layers=1,
        conditioning_route="state_cond",
    )
    monkeypatch.setattr(
        run_eval.PreTrainedConfig,
        "from_pretrained",
        lambda *args, **kwargs: loaded,
    )

    result = run_eval._policy_config(
        {
            "policy_path": "/tmp/current-stage1",
            "architecture": "layerwise_cond_bottleneck",
            "architecture_label": "arch4_skill",
            "architecture_revision": "layerwise_cond_bottleneck_core_exit_v1",
            "vision_conditioning_mode": "layerwise_cond_bottleneck_cross_attention",
            "visual_bridge_last_n_layers": 1,
            "conditioning_route": "state_cond",
            "fsq_path": "/tmp/fsq",
            "dino_model_path": "/tmp/dino",
            "tokenizer_path": "/tmp/tokenizer",
        },
        SimpleNamespace(use_amp=False, n_action_steps=5),
        torch.device("cpu"),
    )

    assert result.architecture_label == "arch4_skill"
    assert result.architecture_revision == "layerwise_cond_bottleneck_core_exit_v1"
    assert result.visual_bridge_last_n_layers == 1


@pytest.mark.parametrize("label", run_eval.FULLSKILL_VIS_ARCH_LABELS)
@pytest.mark.parametrize("revision_kind", ["current", "legacy", "invalid", "changed"])
@pytest.mark.parametrize("bridge_layers", [1, 9])
def test_policy_config_validates_fullskill_visual_revisions(monkeypatch, label, revision_kind, bridge_layers):
    current_revision = (
        run_eval.FULLSKILL_VIS_REVISIONS[label] if bridge_layers == 1 else
        run_eval.FULLSKILL_VIS_SPLIT_REVISIONS.get(label, run_eval.FULLSKILL_VIS_REVISIONS[label])
    )
    revision = (
        run_eval.FULLSKILL_VIS_LEGACY_REVISIONS.get(label, current_revision)
        if revision_kind == "legacy" and bridge_layers == 1 else current_revision
    )
    loaded = SimpleNamespace(
        type="skill_expert",
        architecture="layerwise_cond_bottleneck",
        architecture_label=label,
        architecture_revision=revision,
        vision_conditioning_mode="layerwise_cond_bottleneck_cross_attention",
        visual_bridge_last_n_layers=bridge_layers,
        conditioning_route="state_cond",
    )
    spec = {
        "policy_path": "/tmp/fullskill-visual",
        "architecture": loaded.architecture,
        "architecture_label": label,
        "architecture_revision": revision,
        "vision_conditioning_mode": loaded.vision_conditioning_mode,
        "visual_bridge_last_n_layers": bridge_layers,
        "conditioning_route": "state_cond",
        "fsq_path": "/tmp/fsq",
        "dino_model_path": "/tmp/dino",
        "tokenizer_path": "/tmp/tokenizer",
    }
    monkeypatch.setattr(run_eval.PreTrainedConfig, "from_pretrained", lambda *args, **kwargs: loaded)
    if revision_kind in {"invalid", "changed"}:
        loaded.architecture_revision = "unsupported_fullskill_visual_revision"
        if revision_kind == "invalid":
            spec["architecture_revision"] = loaded.architecture_revision
        with pytest.raises(RuntimeError, match="outside the retained Stage-1"):
            run_eval._policy_config(
                spec, SimpleNamespace(use_amp=False, n_action_steps=5), torch.device("cpu"),
            )
        return

    result = run_eval._policy_config(
        spec, SimpleNamespace(use_amp=False, n_action_steps=5), torch.device("cpu"),
    )
    assert result.architecture_label == label
    assert result.architecture_revision == revision
    assert result.visual_bridge_last_n_layers == bridge_layers
    assert result.pretrained_path == Path(spec["policy_path"])
    assert result.compile_model is result.gradient_checkpointing is False


def test_policy_config_rejects_removed_architecture(monkeypatch) -> None:
    loaded = SimpleNamespace(
        type="skill_expert",
        architecture="removed_architecture",
        architecture_label="removed_mode",
        architecture_revision="removed_revision",
        conditioning_route="removed_route",
    )
    monkeypatch.setattr(
        run_eval.PreTrainedConfig,
        "from_pretrained",
        lambda *args, **kwargs: loaded,
    )

    with pytest.raises(RuntimeError, match="outside the retained Stage-1"):
        run_eval._policy_config(
            {
                "policy_path": "/tmp/removed-stage1",
                "architecture": "removed_architecture",
                "architecture_label": "removed_mode",
                "architecture_revision": "removed_revision",
                "conditioning_route": "removed_route",
            },
            SimpleNamespace(use_amp=False, n_action_steps=5),
            torch.device("cpu"),
        )
def test_policy_config_verifies_checkpoint_owned_dsbc_mode(monkeypatch) -> None:
    loaded = SimpleNamespace(
        type="skill_vla_stage2",
        architecture="cond_gemma",
        architecture_label="arch0",
        architecture_revision="skillvla_real_v1",
        conditioning_route="state_cond",
        stage2_mode="dsbc",
        dsbc_noise_output_mode="per_step",
        dsbc_frs_num_steps=8,
        dsbc_anchor_seed=17,
        dsbc_reader="all_layers",
        dsbc_latent_predictor_enabled=True,
        dsbc_latent_predictor_mode="per_chunk_final",
        dsbc_latent_supervision="skill_only",
        dsbc_latent_loss_weight=0.75,
        dsbc_latent_timesteps=3,
    )
    monkeypatch.setattr(
        run_eval.PreTrainedConfig,
        "from_pretrained",
        lambda *args, **kwargs: loaded,
    )
    spec = {
        "policy_path": "/tmp/stage2-dsbc",
        "architecture": "cond_gemma",
        "architecture_label": "arch0",
        "architecture_revision": "skillvla_real_v1",
        "conditioning_route": "state_cond",
        "stage2_mode": "dsbc",
        "dsbc_noise_output_mode": "per_step",
        "dsbc_frs_num_steps": 8,
        "dsbc_anchor_seed": 17,
        "dsbc_reader": "all_layers",
        "dsbc_latent_predictor_enabled": True,
        "dsbc_latent_predictor_mode": "per_chunk_final",
        "dsbc_latent_supervision": "skill_only",
        "dsbc_latent_loss_weight": 0.75,
        "dsbc_latent_timesteps": 3,
        "fsq_path": "/tmp/fsq",
        "dino_model_path": "/tmp/dino",
        "tokenizer_path": "/tmp/tokenizer",
    }

    result = run_eval._policy_config(
        spec,
        SimpleNamespace(use_amp=False, n_action_steps=5),
        torch.device("cpu"),
    )

    assert result.stage2_mode == "dsbc"
    assert result.dsbc_noise_output_mode == "per_step"

    spec["stage2_mode"] = "likelihood"
    with pytest.raises(RuntimeError, match="stage2_mode resolved=likelihood"):
        run_eval._policy_config(
            spec,
            SimpleNamespace(use_amp=False, n_action_steps=5),
            torch.device("cpu"),
        )


def test_oracle_defers_terminator_advance_until_fixed_replan() -> None:
    expert = _FakeExpert()
    wrapper = Stage1OraclePolicy(
        expert,
        _FakeTerminator(),
        advance_mode="terminator",
        end_mode="termination",
        end_threshold=0.5,
        progress_threshold=0.95,
        max_skill_length=0,
        n_action_steps=2,
    )
    wrapper.set_forced_skill_token_sequences(
        [[{"token": 3, "gt_length": 5}, {"token": 7, "gt_length": 5}]]
    )

    assert wrapper.select_action(_batch()).item() == 3
    assert wrapper.select_action(_batch()).item() == 3
    assert wrapper.select_action(_batch()).item() == 7
    assert [call.item() for call in expert.calls] == [3, 7]


def test_oracle_can_interrupt_chunk_and_replan_on_terminator_advance() -> None:
    expert = _FakeExpert()
    wrapper = Stage1OraclePolicy(
        expert,
        _FakeTerminator(),
        advance_mode="terminator",
        end_mode="termination",
        end_threshold=0.5,
        progress_threshold=0.95,
        max_skill_length=0,
        n_action_steps=2,
        immediate_replan_on_skill_end=True,
    )
    wrapper.set_forced_skill_token_sequences(
        [[{"token": 3, "gt_length": 5}, {"token": 7, "gt_length": 5}]]
    )

    assert wrapper.select_action(_batch()).item() == 3
    # The second observation fires. The remaining action for skill 3 is
    # discarded, and the first action replanned for skill 7 is returned now.
    assert wrapper.select_action(_batch()).item() == 7
    assert wrapper.select_action(_batch()).item() == 7
    assert [call.item() for call in expert.calls] == [3, 7]


def test_oracle_marks_a_fired_final_skill_as_episode_done() -> None:
    wrapper = Stage1OraclePolicy(
        _FakeExpert(),
        _FakeTerminator(),
        advance_mode="terminator",
        end_mode="termination",
        end_threshold=0.5,
        progress_threshold=0.95,
        max_skill_length=0,
        n_action_steps=2,
        immediate_replan_on_skill_end=True,
    )
    wrapper.set_forced_skill_token_sequences(
        [[{"token": 3, "gt_length": 5}]]
    )

    wrapper.select_action(_batch())
    assert wrapper.get_episode_done() == [False]
    assert wrapper.get_skill_end_fired() == [False]

    wrapper.select_action(_batch())
    assert wrapper.get_episode_done() == [True]
    assert wrapper.get_skill_end_fired() == [True]


def test_final_skill_boundary_waits_for_success_or_bounded_timeout() -> None:
    policy = SimpleNamespace(
        wait_for_success_after_final_skill=True,
        get_episode_done=lambda: [True, True, True],
        get_skill_end_fired=lambda: [True, True, True],
        get_final_skill_failure_timeout=lambda: [False, False, True],
    )

    done, fired = _policy_completion_vector(
        policy,
        3,
        np.asarray([False, True, False], dtype=bool),
    )

    np.testing.assert_array_equal(fired, [True, True, True])
    # env 0 keeps executing the final skill, env 1 has succeeded, and env 2
    # records a bounded failure after exhausting max_skill_length.
    np.testing.assert_array_equal(done, [False, True, True])


@pytest.mark.parametrize("immediate", [False, True])
def test_oracle_final_skill_boundary_keeps_running_past_skill_cap(immediate: bool) -> None:
    wrapper = Stage1OraclePolicy(
        _FakeExpert(),
        _FakeTerminator(),
        advance_mode="terminator",
        end_mode="termination",
        end_threshold=0.5,
        progress_threshold=0.95,
        max_skill_length=3,
        n_action_steps=2,
        immediate_replan_on_skill_end=immediate,
    )
    wrapper.set_forced_skill_token_sequences(
        [[{"token": 3, "gt_length": 5}]]
    )

    for _ in range(8):
        assert wrapper.select_action(_batch()).item() == 3
        assert wrapper.get_final_skill_failure_timeout() == [False]
        done, _ = _policy_completion_vector(wrapper, 1, np.array([False]))
        assert not done[0]
    # The learned final boundary still latches, but cannot declare failure.
    assert wrapper.get_episode_done() == [True]
    done, _ = _policy_completion_vector(wrapper, 1, np.array([True]))
    assert done[0]
    assert wrapper.get_skill_trace()[0]["length"] == 8


def test_nonfinal_skill_timeout_still_advances_but_final_skill_has_no_cap() -> None:
    wrapper = Stage1OraclePolicy(
        _FakeExpert(), _ConstantTerminator(), advance_mode="external",
        end_mode="termination", end_threshold=0.5, progress_threshold=0.95,
        max_skill_length=3, n_action_steps=2, immediate_replan_on_skill_end=True,
    )
    wrapper.set_forced_skill_token_sequences(
        [[{"token": 3, "gt_length": 5}, {"token": 7, "gt_length": 5}]]
    )
    codes = [wrapper.select_action(_batch()).item() for _ in range(10)]
    assert codes[:2] == [3, 3]
    assert codes[2:] == [7] * 8
    assert [t["codebook_token"] for t in wrapper.get_skill_trace()] == [3, 7]
    assert wrapper.get_episode_done() == [False]
    assert wrapper.get_final_skill_failure_timeout() == [False]
    assert wrapper.get_skill_end_fired() == [False]  # no synthetic boundary after the final cap


def test_open_ended_predictor_still_uses_skill_timeout() -> None:
    wrapper = Stage1OraclePolicy(
        _FakeExpert(), _ConstantTerminator(), skill_source="external",
        advance_mode="external", end_mode="termination", end_threshold=0.5,
        progress_threshold=0.95, max_skill_length=2, n_action_steps=2,
        immediate_replan_on_skill_end=True,
    )
    wrapper.set_reference_skill_token_sequences([[{"token": 3, "gt_length": 5}]])
    assert wrapper.select_action(_batch()).item() == 4
    assert wrapper.select_action(_batch()).item() == 8
    assert wrapper.get_episode_done() == [False]


@pytest.mark.parametrize("probability", [0.0, 1.0])
@pytest.mark.parametrize("success_step", [None, 4])
@pytest.mark.parametrize("tokens", [(3,), (3, 7)])
def test_rollout_enforces_total_episode_budget_not_final_skill_cap(
    monkeypatch, probability: float, success_step: int | None, tokens: tuple[int, ...],
) -> None:
    wrapper = Stage1OraclePolicy(
        _FakeExpert(), _ConstantTerminator(probability), advance_mode="external",
        end_mode="termination", end_threshold=0.5, progress_threshold=0.95,
        max_skill_length=2, n_action_steps=2, immediate_replan_on_skill_end=True,
    )
    wrapper.set_forced_skill_token_sequences([[{"token": code, "gt_length": 2} for code in tokens]])

    class EpisodeBudgetEnv:
        num_envs = 1
        steps = 0

        def reset(self, seed=None):
            self.steps = 0
            return _batch(), {}

        def call(self, name):
            assert name == "_max_episode_steps"
            return [7]

        def step(self, action):
            self.steps += 1
            assert action.shape == (1, 1)
            success = success_step is not None and self.steps >= success_step
            return _batch(), np.array([float(success)]), np.array([False]), np.array([False]), {
                "is_success": np.array([success]),
            }

    monkeypatch.setattr(skillvla_eval, "preprocess_observation", lambda observation: observation)
    monkeypatch.setattr(skillvla_eval, "add_envs_task", lambda env, observation: observation)
    monkeypatch.setattr(skillvla_eval, "check_env_attributes_and_types", lambda env: None)
    monkeypatch.setattr(skillvla_eval, "inside_slurm", lambda: True)
    env = EpisodeBudgetEnv()

    def identity(value):
        return value

    result = skillvla_eval.rollout(env, wrapper, identity, identity, identity, identity)
    # Preserve success + learned-boundary completion; a failed episode must
    # exhaust its total budget even if the final terminator fires prematurely.
    expected = success_step if probability > 0.5 and success_step is not None else 7
    assert env.steps == expected
    assert result["action"].shape[1] == expected
    assert result["done"][0, -1].item()
    assert not result["done"][0, :-1].any()
    assert result["success"].any().item() is (success_step is not None)


def test_oracle_passes_the_previous_executed_action_to_the_terminator() -> None:
    class _RecordingTerminator:
        requires_state = True

        def __init__(self):
            self.previous_actions = []

        def terminate(
            self, codes, state, image, wrist, previous_action=None
        ):
            del state, image, wrist
            self.previous_actions.append(
                None if previous_action is None else previous_action.detach().clone()
            )
            zeros = torch.zeros_like(codes, dtype=torch.float32)
            return zeros, zeros

    terminator = _RecordingTerminator()
    wrapper = Stage1OraclePolicy(
        _FakeExpert(),
        terminator,
        advance_mode="terminator",
        end_mode="termination",
        end_threshold=0.5,
        progress_threshold=0.95,
        max_skill_length=0,
        n_action_steps=2,
    )
    wrapper.set_forced_skill_token_sequences(
        [[{"token": 3, "gt_length": 5}, {"token": 7, "gt_length": 5}]]
    )

    wrapper.select_action(_batch())
    assert terminator.previous_actions == [None]

    executed = torch.tensor([[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 1.0]])
    wrapper.record_executed_action(executed)
    wrapper.select_action(_batch())
    torch.testing.assert_close(terminator.previous_actions[1], executed)

    wrapper.reset()
    wrapper.set_forced_skill_token_sequences(
        [[{"token": 3, "gt_length": 5}, {"token": 7, "gt_length": 5}]]
    )
    wrapper.select_action(_batch())
    assert terminator.previous_actions[-1] is None


def test_episode_exact_gt_codes_are_resolved_per_model_skill_space(
    monkeypatch,
) -> None:
    def _load(dataset_dir, init_states_path, suite_name):
        del init_states_path, suite_name
        token = 4 if str(dataset_dir) == "zero-space" else 19
        return {
            0: [
                {
                    "episode_index": 12,
                    "init_state": np.asarray([1.0, 2.0]),
                    "skills": [{"token": token, "gt_length": 7}],
                }
            ]
        }

    monkeypatch.setattr(run_eval, "load_episode_exact_data", _load)
    init_states = {}
    maps = run_eval._episode_exact_oracle_maps(
        {"libero_90": {0: object()}},
        [
            {"skill_dataset_dir": "zero-space", "eval_init_states_path": "same"},
            {"skill_dataset_dir": "action-space", "eval_init_states_path": "same"},
        ],
        "libero_90",
        1,
        init_states,
    )

    assert maps[0][("libero_90", 0)][0][0]["token"] == 4
    assert maps[1][("libero_90", 0)][0][0]["token"] == 19
    np.testing.assert_array_equal(init_states[("libero_90", 0)], [[1.0, 2.0]])


def test_episode_exact_repeat_cycles_scarce_episodes(monkeypatch) -> None:
    def _load(dataset_dir, init_states_path, suite_name):
        del dataset_dir, init_states_path, suite_name
        return {
            0: [{"episode_index": 5, "init_state": np.asarray([1.0]), "skills": [{"token": 1}]}],
            1: [
                {"episode_index": 7, "init_state": np.asarray([2.0]), "skills": [{"token": 2}]},
                {"episode_index": 8, "init_state": np.asarray([3.0]), "skills": [{"token": 3}]},
            ],
        }

    monkeypatch.setattr(run_eval, "load_episode_exact_data", _load)
    spec = [{"skill_dataset_dir": "d", "eval_init_states_path": "same"}]

    envs, init_states = {"libero_10": {0: object(), 1: object()}}, {}
    maps = run_eval._episode_exact_oracle_maps(envs, spec, "libero_10", 3, init_states, repeat=True)
    assert [skills[0]["token"] for skills in maps[0][("libero_10", 0)]] == [1, 1, 1]
    assert [skills[0]["token"] for skills in maps[0][("libero_10", 1)]] == [2, 3, 2]
    np.testing.assert_array_equal(init_states[("libero_10", 1)], [[2.0], [3.0], [2.0]])

    envs = {"libero_10": {0: object(), 1: object()}}
    maps = run_eval._episode_exact_oracle_maps(envs, spec, "libero_10", 3, {})
    assert maps[0] == {} and envs["libero_10"] == {}  # default: scarce tasks are dropped


def test_episode_exact_repeat_changes_the_resume_signature_only_when_on(monkeypatch) -> None:
    monkeypatch.setenv("SKILL_END_MODE", "termination")
    monkeypatch.setenv("SKILL_END_THRESHOLD", "0.3")
    monkeypatch.setenv("SKILL_END_PROGRESS_THRESHOLD", "0.95")
    monkeypatch.setenv("INFERENCE_SKILL_MAX_LENGTH", "200")
    cfg = SimpleNamespace(
        eval=SimpleNamespace(n_episodes=10), policy=SimpleNamespace(n_action_steps=5), seed=1000
    )
    spec = defaultdict(str, policy_path="p", label="a")  # every other spec field blank
    monkeypatch.delenv("EPISODE_EXACT_REPEAT", raising=False)
    off = run_eval._panel_signature(spec, {"t"}, cfg)
    assert "episode_exact_repeat" not in off
    monkeypatch.setenv("EPISODE_EXACT_REPEAT", "true")
    assert run_eval._panel_signature(spec, {"t"}, cfg) == {**off, "episode_exact_repeat": True}


def test_cond_goal_interventions_and_grasp_diagnostics_invalidate_old_panel_caches(monkeypatch):
    for name, value in (("SKILL_END_MODE", "termination"), ("SKILL_END_THRESHOLD", "0.3"),
                        ("SKILL_END_PROGRESS_THRESHOLD", "0.95"), ("INFERENCE_SKILL_MAX_LENGTH", "200")):
        monkeypatch.setenv(name, value)
    cfg = SimpleNamespace(eval=SimpleNamespace(n_episodes=10, goal_predicate_diagnostics=False),
                          policy=SimpleNamespace(n_action_steps=5), seed=1000)
    spec = defaultdict(str, policy_path="p", label="a")
    normal = run_eval._panel_signature(spec, {"task5"}, cfg)
    assert "cond_goal_intervention" not in normal
    spec.update(cond_goal_mode="offset", cond_goal_offset_m=[.05, 0, 0])
    offset = run_eval._panel_signature(spec, {"task5"}, cfg)
    assert offset != normal
    assert offset["cond_goal_intervention"]["offset_m"] == [.05, 0, 0]
    cfg.eval.goal_predicate_diagnostics = True
    assert run_eval._panel_signature(spec, {"task5"}, cfg)["goal_predicate_diagnostics"] == "bddl_partial_success_grasp_v3"


def test_panel_resume_requires_each_enabled_episode_skill_video_dir(tmp_path: Path) -> None:
    cfg = SimpleNamespace(
        eval=SimpleNamespace(
            n_episodes=2,
            max_videos_per_task=2,
            skill_html=False,
            skill_videos=True,
            presentation_videos=False,
        )
    )
    task_dir = tmp_path / "videos" / "libero_10_0"
    for episode_index in range(2):
        (task_dir / f"eval_episode_{episode_index}.mp4").parent.mkdir(
            parents=True, exist_ok=True
        )
        (task_dir / f"eval_episode_{episode_index}.mp4").write_bytes(b"video")
        skill_dir = task_dir / "skill_videos" / f"eval_episode_{episode_index}"
        skill_dir.mkdir(parents=True)
        (skill_dir / "occurrence_00_skill_00_token_02.mp4").write_bytes(b"clip")

    assert run_eval._panel_artifacts_complete(tmp_path, {"libero_10_0"}, cfg)

    incomplete_root = tmp_path / "incomplete"
    incomplete_task_dir = incomplete_root / "videos" / "libero_10_0"
    incomplete_task_dir.mkdir(parents=True)
    for episode_index in range(2):
        (incomplete_task_dir / f"eval_episode_{episode_index}.mp4").write_bytes(b"video")
    first_skill_dir = incomplete_task_dir / "skill_videos" / "eval_episode_0"
    first_skill_dir.mkdir(parents=True)
    (first_skill_dir / "occurrence_00_skill_00_token_02.mp4").write_bytes(b"clip")

    assert not run_eval._panel_artifacts_complete(
        incomplete_root, {"libero_10_0"}, cfg
    )


def test_episode_exact_per_chunk_oracle_uses_action_replan_stride(
    monkeypatch,
) -> None:
    calls = []

    def _load(dataset_dir, init_states_path, suite_name, **kwargs):
        calls.append((dataset_dir, init_states_path, suite_name, kwargs))
        return {
            0: [
                {
                    "episode_index": 12,
                    "init_state": np.asarray([1.0, 2.0]),
                    "skills": [{"token": 4, "gt_length": 12}],
                }
            ]
        }

    monkeypatch.setattr(run_eval, "load_episode_exact_data", _load)
    run_eval._episode_exact_oracle_maps(
        {"libero_90": {0: object()}},
        [
            {
                "skill_dataset_dir": "latent-space",
                "eval_init_states_path": "same",
                "latent_source": "oracle",
                "oracle_latent_target": "per_chunk",
                "chunk_size": 10,
            }
        ],
        "libero_90",
        1,
        {},
        n_action_steps=5,
    )

    assert calls[0][3] == {
        "action_chunk_size": 10,
        "action_chunk_stride": 5,
        "oracle_latent_target": "per_chunk",
    }


@pytest.mark.parametrize("stage2_mode", ["likelihood", "dsbc"])
def test_stage2_eval_holds_vlm_start_condition_until_the_next_skill(
    stage2_mode: str,
) -> None:
    expert = _FakeStage2Expert(stage2_mode)
    wrapper = Stage1OraclePolicy(
        expert,
        None,
        advance_mode="gt",
        end_mode="or",
        end_threshold=0.5,
        progress_threshold=0.95,
        max_skill_length=0,
        n_action_steps=1,
    )
    wrapper.set_forced_skill_token_sequences(
        [[{"token": 3, "gt_length": 3}, {"token": 7, "gt_length": 3}]]
    )

    assert wrapper.select_action(_stage2_batch(1)).item() == 3
    assert wrapper.select_action(_stage2_batch(2)).item() == 3
    assert wrapper.select_action(_stage2_batch(3)).item() == 7

    assert [call["current_image"].flatten()[0].item() for call in expert.vlm_calls] == [
        1.0,
        2.0,
        3.0,
    ]
    assert [call["start_image"].flatten()[0].item() for call in expert.vlm_calls] == [
        1.0,
        1.0,
        3.0,
    ]
    assert [call["start_wrist"].flatten()[0].item() for call in expert.vlm_calls] == [
        11.0,
        11.0,
        13.0,
    ]
    assert [call["start_tokens"].item() for call in expert.vlm_calls] == [1, 1, 3]
    assert [call["cache_id"].item() for call in expert.vlm_calls] == [0, 0, 1]


def test_stage2_eval_selects_and_injects_one_hindsight_latent_per_skill() -> None:
    expert = _FakeOracleStage2Expert()
    wrapper = Stage1OraclePolicy(
        expert,
        None,
        advance_mode="gt",
        end_mode="max_length",
        end_threshold=0.5,
        progress_threshold=0.95,
        max_skill_length=0,
        n_action_steps=1,
        latent_source="oracle",
        oracle_latent_grid_size=5,
        oracle_latent_timesteps=3,
        oracle_action_q01=[0.0],
        oracle_action_q99=[2.0],
    )
    wrapper.set_forced_skill_token_sequences(
        [
            [
                {
                    "token": 3,
                    "gt_length": 2,
                    "gt_actions": np.asarray([[1.0], [2.0]], dtype=np.float32),
                    "gt_action_valid": np.asarray([True, True]),
                }
            ]
        ]
    )

    assert wrapper.select_action(_stage2_batch(1)).item() == 3
    assert len(expert.oracle_targets) == 1
    target = expert.oracle_targets[0]
    torch.testing.assert_close(target["actions"], torch.tensor([[[0.0], [1.0]]]))
    torch.testing.assert_close(target["valid"], torch.tensor([[True, True]]))
    assert target["grid_size"] == 5
    assert target["timesteps"] == 3
    torch.testing.assert_close(
        expert.mode_overrides[0], torch.tensor([[0.25, -0.5]])
    )
    trace = wrapper.get_skill_trace()[0]
    assert trace["predicted_latent"] == pytest.approx([0.1, 0.2])
    assert trace["oracle_latent"] == pytest.approx([0.25, -0.5])
    assert trace["oracle_latent_score"] == pytest.approx(0.2)
    assert trace["predicted_latent_score"] == pytest.approx(0.7)


def test_stage2_eval_reselects_hindsight_latent_for_every_action_chunk() -> None:
    expert = _FakeOracleStage2Expert()
    wrapper = Stage1OraclePolicy(
        expert,
        None,
        advance_mode="gt",
        end_mode="max_length",
        end_threshold=0.5,
        progress_threshold=0.95,
        max_skill_length=0,
        n_action_steps=1,
        latent_source="oracle",
        oracle_latent_target="per_chunk",
        oracle_latent_grid_size=3,
        oracle_latent_timesteps=2,
        oracle_action_q01=[0.0],
        oracle_action_q99=[2.0],
    )
    wrapper.set_forced_skill_token_sequences(
        [
            [
                {
                    "token": 3,
                    "gt_length": 3,
                    "gt_actions": np.asarray(
                        [
                            [[0.0], [2.0]],
                            [[1.0], [2.0]],
                            [[2.0], [0.0]],
                        ],
                        dtype=np.float32,
                    ),
                    "gt_action_valid": np.ones((3, 2), dtype=bool),
                }
            ]
        ]
    )

    wrapper.select_action(_stage2_batch(1))
    wrapper.select_action(_stage2_batch(2))

    assert len(expert.oracle_targets) == 2
    torch.testing.assert_close(
        expert.oracle_targets[0]["actions"], torch.tensor([[[-1.0], [1.0]]])
    )
    torch.testing.assert_close(
        expert.oracle_targets[1]["actions"], torch.tensor([[[0.0], [1.0]]])
    )
    trace = wrapper.get_skill_trace()[0]
    assert trace["oracle_latent_target"] == "per_chunk"
    assert trace["oracle_window_index"] == 1
    assert [point["episode_timestep"] for point in trace["latent_points"]] == [0, 1]
    assert all("oracle_latent" in point for point in trace["latent_points"])


def test_stage2_eval_full_skill_oracle_reuses_start_context_for_all_windows(
    monkeypatch,
) -> None:
    class _VideoReader:
        def __init__(self, dataset_dir):
            del dataset_dir

        def read(self, episode_index, timestamps):
            del episode_index
            count = len(timestamps)
            return {
                "observation.images.image": torch.full(
                    (count, 3, 4, 4), 7.0
                ),
                "observation.images.wrist_image": torch.full(
                    (count, 3, 4, 4), 8.0
                ),
            }

    monkeypatch.setattr(run_eval, "_OracleSkillVideoReader", _VideoReader)

    class _FullOracleStage2(_FakeOracleStage2Expert):
        def select_hindsight_mode_latent(
            self,
            batch,
            target_actions,
            target_valid,
            *,
            grid_size,
            timesteps,
            aggregate_windows=False,
        ):
            self.oracle_targets.append(
                {
                    "batch": {key: value.detach().clone() for key, value in batch.items()},
                    "actions": target_actions.detach().clone(),
                    "valid": target_valid.detach().clone(),
                    "aggregate_windows": aggregate_windows,
                }
            )
            return torch.tensor([[0.25, -0.5]]), torch.tensor([[0.7, 0.2]])

    expert = _FullOracleStage2()
    expert.config.proprio_grounding = "episode_start_xyz"
    wrapper = Stage1OraclePolicy(
        expert,
        None,
        advance_mode="gt",
        end_mode="max_length",
        end_threshold=0.5,
        progress_threshold=0.95,
        max_skill_length=0,
        n_action_steps=1,
        latent_source="oracle",
        oracle_latent_target="full_skill",
        oracle_action_q01=[0.0],
        oracle_action_q99=[2.0],
        oracle_state_q01=[0.0] * 8,
        oracle_state_q99=[2.0] * 8,
        oracle_dataset_dir="dataset/run/skillvla",
    )
    wrapper.set_forced_skill_token_sequences(
        [
            [
                {
                    "token": 3,
                    "gt_length": 3,
                    "gt_actions": np.asarray(
                        [[[1.0], [2.0]], [[0.0], [0.0]]], dtype=np.float32
                    ),
                    "gt_action_valid": np.asarray(
                        [[True, True], [True, False]]
                    ),
                    "gt_window_states": np.asarray(
                        [[1.0] * 8, [2.0] * 8], dtype=np.float32
                    ),
                    "gt_window_timestamps": np.asarray([0.0, 0.1]),
                    "gt_episode_start_state": np.zeros(8, dtype=np.float32),
                    "gt_episode_index": 9,
                }
            ]
        ]
    )

    assert wrapper.select_action(_stage2_batch(1)).item() == 3
    target = expert.oracle_targets[0]
    assert target["aggregate_windows"] is True
    torch.testing.assert_close(
        target["batch"]["observation.images.image"],
        torch.full((2, 3, 4, 4), 7.0),
    )
    # The latent/noise readers keep the rollout's deployed skill-start VLM
    # context fixed while exact-demo current observations span the skill.
    torch.testing.assert_close(
        target["batch"]["skill_start_image"],
        torch.ones(2, 3, 8, 8),
    )
    torch.testing.assert_close(
        target["batch"]["observation.language.tokens"],
        torch.ones(2, 1, dtype=torch.long),
    )
    torch.testing.assert_close(
        expert.mode_overrides[0], torch.tensor([[0.25, -0.5]])
    )


def test_stage1_eval_selects_and_executes_same_hindsight_latent_noise_pair() -> None:
    expert = _FakeOracleStage1Expert()
    wrapper = Stage1OraclePolicy(
        expert,
        None,
        advance_mode="gt",
        end_mode="max_length",
        end_threshold=0.5,
        progress_threshold=0.95,
        max_skill_length=0,
        n_action_steps=1,
        latent_source="oracle",
        oracle_latent_grid_size=3,
        oracle_latent_timesteps=2,
        oracle_action_q01=[0.0],
        oracle_action_q99=[2.0],
    )
    wrapper.set_forced_skill_token_sequences(
        [
            [
                {
                    "token": 5,
                    "gt_length": 2,
                    "gt_actions": np.asarray([[1.0], [2.0]], dtype=np.float32),
                    "gt_action_valid": np.asarray([True, True]),
                }
            ]
        ]
    )

    assert wrapper.select_action(_batch()).item() == 5
    assert len(expert.oracle_targets) == 1
    torch.testing.assert_close(
        expert.oracle_targets[0]["actions"], torch.tensor([[[0.0], [1.0]]])
    )
    torch.testing.assert_close(
        expert.mode_overrides[0], torch.tensor([[-0.5, 0.75]])
    )
    torch.testing.assert_close(
        expert.noise_overrides[0], torch.full((1, 2, 4), 0.75)
    )
    trace = wrapper.get_skill_trace()[0]
    assert trace["baseline_latent"] == pytest.approx([0.2, -0.1])
    assert trace["oracle_latent"] == pytest.approx([-0.5, 0.75])
    assert trace["oracle_latent_score"] == pytest.approx(0.1)
    assert trace["predicted_latent_score"] == pytest.approx(0.6)


def test_stage1_eval_full_skill_oracle_uses_all_exact_windows(
    monkeypatch,
) -> None:
    class _VideoReader:
        def __init__(self, dataset_dir):
            self.dataset_dir = dataset_dir
            self.calls = []

        def read(self, episode_index, timestamps):
            self.calls.append((episode_index, np.asarray(timestamps).copy()))
            count = len(timestamps)
            return {
                "observation.images.image": torch.ones(count, 3, 4, 4),
                "observation.images.wrist_image": torch.full(
                    (count, 3, 4, 4), 2.0
                ),
            }

    readers = []

    def _reader_factory(dataset_dir):
        reader = _VideoReader(dataset_dir)
        readers.append(reader)
        return reader

    monkeypatch.setattr(run_eval, "_OracleSkillVideoReader", _reader_factory)

    class _FullOracleExpert(_FakeOracleStage1Expert):
        def __init__(self):
            super().__init__()
            self.config.proprio_grounding = "episode_start_xyz"

        def select_hindsight_mode_latent(
            self,
            batch,
            target_actions,
            target_valid,
            *,
            grid_size,
            timesteps,
            aggregate_windows=False,
        ):
            self.oracle_targets.append(
                {
                    "batch": {key: value.detach().clone() for key, value in batch.items()},
                    "actions": target_actions.detach().clone(),
                    "valid": target_valid.detach().clone(),
                    "aggregate_windows": aggregate_windows,
                }
            )
            self._last_hindsight_mode_noise = torch.full((1, 2, 4), 0.25)
            return torch.tensor([[0.5, -0.5]]), torch.tensor([[0.8, 0.3]])

    expert = _FullOracleExpert()
    wrapper = Stage1OraclePolicy(
        expert,
        None,
        advance_mode="gt",
        end_mode="max_length",
        end_threshold=0.5,
        progress_threshold=0.95,
        max_skill_length=0,
        n_action_steps=1,
        latent_source="oracle",
        oracle_latent_target="full_skill",
        oracle_latent_grid_size=3,
        oracle_latent_timesteps=2,
        oracle_action_q01=[0.0],
        oracle_action_q99=[2.0],
        oracle_state_q01=[0.0, -2.0, 0.0, 0.0],
        oracle_state_q99=[2.0, 2.0, 4.0, 1.0],
        oracle_dataset_dir="dataset/run/skillvla",
    )
    wrapper.set_forced_skill_token_sequences(
        [
            [
                {
                    "token": 5,
                    "gt_length": 3,
                    "gt_actions": np.asarray(
                        [[[1.0], [2.0]], [[0.0], [0.0]]], dtype=np.float32
                    ),
                    "gt_action_valid": np.asarray(
                        [[True, True], [True, False]]
                    ),
                    "gt_window_states": np.asarray(
                        [[11.0, 18.0, 31.0, 0.5], [12.0, 20.0, 33.0, 0.7]],
                        dtype=np.float32,
                    ),
                    "gt_window_timestamps": np.asarray([0.0, 0.1]),
                    "gt_episode_start_state": np.asarray(
                        [10.0, 20.0, 30.0, 0.0], dtype=np.float32
                    ),
                    "gt_episode_index": 7,
                }
            ]
        ]
    )

    assert wrapper.select_action(_batch()).item() == 5
    target = expert.oracle_targets[0]
    assert target["aggregate_windows"] is True
    torch.testing.assert_close(
        target["actions"],
        torch.tensor([[[0.0], [1.0]], [[-1.0], [-1.0]]]),
    )
    torch.testing.assert_close(
        target["batch"]["observation.state"],
        torch.tensor([[0.0, -1.0, -0.5, 0.0], [1.0, 0.0, 0.5, 0.4]]),
    )
    assert readers[0].calls[0][0] == 7
    np.testing.assert_allclose(readers[0].calls[0][1], [0.0, 0.1])
    torch.testing.assert_close(
        expert.mode_overrides[0], torch.tensor([[0.5, -0.5]])
    )
    torch.testing.assert_close(
        expert.noise_overrides[0], torch.full((1, 2, 4), 0.25)
    )


def test_stage1_policy_hindsight_selector_scores_main_flow_on_shared_noise() -> None:
    class _TinyFlowModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self._vsa_debug_active = False
            self._gradient_checkpointing = False

        def sample_mode_latent(self, batch_shape, device):
            return torch.zeros((*batch_shape, 2), device=device)

        def sample_noise(self, shape, device):
            return torch.zeros(shape, device=device)

        def _condition_tokens(self, images, *, batch_size, skill_code=None):
            del images, skill_code
            return torch.zeros(batch_size, 1, 1)

        def _predict_velocity_from_condition(
            self, condition, noisy_actions, state, skill, time, mode_latent
        ):
            del condition, state, skill, time
            return mode_latent[:, :1, None].expand_as(noisy_actions)

    policy = SkillExpertPolicy.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.model = _TinyFlowModel()
    policy.config = SimpleNamespace(
        skill_flow_latent_best_of_n_enabled=True,
        skill_flow_latent_dim=2,
        conditioning_route="state_cond",
        max_state_dim=2,
        max_action_dim=2,
        chunk_size=2,
        output_features={"action": SimpleNamespace(shape=(1,))},
    )
    policy._mode_latent_cache = None
    policy._mode_latent_skill_code = None
    policy._skill_code = lambda batch: batch["skill_code"]
    policy._collect_images = lambda batch: []

    selected, scores = policy.select_hindsight_mode_latent(
        {
            "observation.state": torch.zeros(1, 2),
            "skill_code": torch.tensor([1]),
        },
        torch.ones(1, 2, 1),
        torch.ones(1, 2, dtype=torch.bool),
        grid_size=3,
        timesteps=2,
    )

    torch.testing.assert_close(selected, torch.tensor([[-1.0, -1.0]]))
    assert scores[0, 0].item() == pytest.approx(1.0)
    assert scores.min().item() == pytest.approx(0.0)
    torch.testing.assert_close(
        policy._last_hindsight_mode_noise, torch.zeros(1, 2, 2)
    )


def test_stage1_hindsight_selector_aggregates_one_latent_over_skill_windows() -> None:
    class _TinyFlowModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self._vsa_debug_active = False
            self._gradient_checkpointing = False

        def sample_mode_latent(self, batch_shape, device):
            return torch.zeros((*batch_shape, 2), device=device)

        def sample_noise(self, shape, device):
            return torch.zeros(shape, device=device)

        def _condition_tokens(self, images, *, batch_size, skill_code=None):
            del images, skill_code
            return torch.zeros(batch_size, 1, 1)

        def _predict_velocity_from_condition(
            self, condition, noisy_actions, state, skill, time, mode_latent
        ):
            del condition, state, skill, time
            return mode_latent[:, :1, None].expand_as(noisy_actions)

    policy = SkillExpertPolicy.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.model = _TinyFlowModel()
    policy.config = SimpleNamespace(
        skill_flow_latent_best_of_n_enabled=True,
        skill_flow_latent_dim=2,
        conditioning_route="state_cond",
        max_state_dim=2,
        max_action_dim=2,
        chunk_size=2,
        output_features={"action": SimpleNamespace(shape=(1,))},
    )
    policy._mode_latent_cache = None
    policy._mode_latent_skill_code = None
    policy._skill_code = lambda batch: batch["skill_code"]
    policy._collect_images = lambda batch: []

    selected, scores = policy.select_hindsight_mode_latent(
        {
            "observation.state": torch.zeros(2, 2),
            "skill_code": torch.tensor([4, 4]),
        },
        torch.ones(2, 2, 1),
        torch.tensor([[True, True], [True, False]]),
        grid_size=3,
        timesteps=2,
        aggregate_windows=True,
    )

    torch.testing.assert_close(selected, torch.tensor([[-1.0, -1.0]]))
    assert scores.shape == (1, 10)
    assert scores[0, 0].item() == pytest.approx(1.0)
    assert scores.min().item() == pytest.approx(0.0)
    # Only the first window's source is reused for the real first rollout.
    torch.testing.assert_close(
        policy._last_hindsight_mode_noise, torch.zeros(1, 2, 2)
    )


def test_checkpoint_terminator_converts_logits_to_probability() -> None:
    class _Model:
        fsq_term_train = object()

        def terminator_predict(self, codes, state, image, wrist):
            return torch.tensor([0.25]), torch.tensor([0.0])

    adapter = CheckpointTerminator(SimpleNamespace(model=_Model()))
    progress, probability = adapter.terminate(
        torch.tensor([1]),
        torch.zeros(1, 8),
        torch.zeros(1, 3, 8, 8),
        torch.zeros(1, 3, 8, 8),
    )

    assert progress.item() == 0.25
    assert probability.item() == 0.5


@pytest.mark.parametrize("seed_current", [True, False])
def test_masked_history_skill_reset_is_row_local_and_matches_dataset(seed_current):
    from lerobot.policies.skillVLA.newterm_supervision import history_valid_mask
    module = SimpleNamespace(
        proprio_history=True, history_length=20, history_mask_padding=True, history_reset_on_skill=True
    )
    adapter = CheckpointTerminator(SimpleNamespace(model=SimpleNamespace(fsq_term_train=module)))
    first = torch.tensor([[1., 2.], [10., 20.]])
    initial = adapter._history_context(first)
    assert adapter._history_valid_mask.sum(dim=1).tolist() == [1, 1]
    assert (initial[:, :-1] == 0).all()
    boundary = first + 1
    if seed_current:
        adapter._history_context(boundary)
    adapter.reset_skill([0], boundary, seed_current=seed_current)
    next_state = boundary + 1 if seed_current else boundary
    history = adapter._history_context(next_state)
    count = 2 if seed_current else 1
    assert adapter._history_valid_mask.sum(dim=1).tolist() == [count, count + 1]
    torch.testing.assert_close(adapter._history_valid_mask[0], history_valid_mask(count - 1, 0, 20))
    assert (history[0, :-count] == 0).all()
    if seed_current:
        torch.testing.assert_close(history[0, -2], boundary[0])
    torch.testing.assert_close(history[1, -(count + 1)], first[1])
    adapter.reset()
    adapter._history_context(next_state)
    assert adapter._history_valid_mask.sum(dim=1).tolist() == [1, 1]


def test_checkpoint_terminator_rolls_and_resets_proprio_history() -> None:
    class _Terminator:
        context_mode = "proprio"
        proprio_history = True
        history_length = 3

    class _Model:
        fsq_term_train = _Terminator()

        def __init__(self):
            self.states = []

        def terminator_predict(self, codes, state, image, wrist):
            del codes, image, wrist
            self.states.append(state.clone())
            return torch.tensor([0.0]), torch.tensor([0.0])

    model = _Model()
    adapter = CheckpointTerminator(SimpleNamespace(model=model))
    image = torch.zeros(1, 3, 8, 8)
    adapter.terminate(torch.tensor([1]), torch.ones(1, 8), image, image)
    adapter.terminate(torch.tensor([1]), torch.full((1, 8), 2.0), image, image)

    torch.testing.assert_close(model.states[0], torch.ones(1, 3, 8))
    torch.testing.assert_close(
        model.states[1][:, :, 0], torch.tensor([[1.0, 1.0, 2.0]])
    )

    adapter.reset()
    adapter.terminate(torch.tensor([1]), torch.full((1, 8), 3.0), image, image)
    torch.testing.assert_close(model.states[2], torch.full((1, 3, 8), 3.0))


def test_checkpoint_image_only_terminator_does_not_use_state() -> None:
    class _Model:
        fsq_term_train = None
        fsq_image_term_train = object()

    class _Policy:
        model = _Model()

        def image_only_terminator_predict(self, codes, image, wrist):
            del codes, image, wrist
            return torch.tensor([0.75]), torch.tensor([0.0])

    adapter = CheckpointTerminator(_Policy(), variant="image_only")
    progress, probability = adapter.terminate(
        torch.tensor([1]),
        None,
        torch.zeros(1, 3, 8, 8),
        torch.zeros(1, 3, 8, 8),
    )

    assert adapter.requires_state is False
    assert progress.item() == 0.75
    assert probability.item() == 0.5


def test_predictor_source_repredicts_at_a_detected_boundary() -> None:
    expert = _FakeExpert()
    wrapper = Stage1OraclePolicy(
        expert,
        _FakeTerminator(),
        skill_source="predictor",
        advance_mode="terminator",
        end_mode="termination",
        end_threshold=0.5,
        progress_threshold=0.95,
        max_skill_length=0,
        n_action_steps=2,
    )
    wrapper.set_reference_skill_token_sequences(
        [[{"token": 3, "gt_length": 5}, {"token": 7, "gt_length": 5}]]
    )

    assert wrapper.select_action(_batch()).item() == 4
    assert wrapper.select_action(_batch()).item() == 4
    assert wrapper.select_action(_batch()).item() == 8
    assert [call.item() for call in expert.calls] == [4, 8]
    assert [record["skill_source"] for record in wrapper.get_skill_trace()] == [
        "own",
        "own",
    ]


def test_predictor_can_repredict_on_gt_boundaries() -> None:
    expert = _FakeExpert()
    wrapper = Stage1OraclePolicy(
        expert,
        None,
        skill_source="own",
        advance_mode="gt",
        end_mode="or",
        end_threshold=0.5,
        progress_threshold=0.95,
        max_skill_length=0,
        n_action_steps=2,
    )
    wrapper.set_reference_skill_token_sequences(
        [[{"token": 3, "gt_length": 1}, {"token": 7, "gt_length": 2}]]
    )

    assert wrapper.select_action({"observation.state": torch.zeros(1, 8)}).item() == 8
    assert [call.item() for call in expert.calls] == [8]


def test_previous_skill_context_tracks_executed_predictions_per_environment_and_resets():
    class RecordingExpert(_FakeExpert):
        def __init__(self):
            super().__init__()
            self.history = []
            self.predictions = [torch.tensor(x) for x in ([4, 8], [5, 9], [6, 10], [7, 11])]

        def predict_skill_code(self, batch):
            self.history.append(batch["skill_previous_code"].clone())
            return super().predict_skill_code(batch)

    expert = RecordingExpert()
    wrapper = Stage1OraclePolicy(
        expert, None, skill_source="own", advance_mode="gt", end_mode="or",
        end_threshold=.5, progress_threshold=.95, max_skill_length=0, n_action_steps=2,
    )
    references = [[{"token": 3, "gt_length": 5}, {"token": 7, "gt_length": 5}]] * 2
    wrapper.set_reference_skill_token_sequences(references)
    batch = {"observation.state": torch.zeros(2, 8)}
    wrapper.select_action(batch)
    torch.testing.assert_close(expert.history[0], torch.tensor([-1, -1]))
    wrapper._pending_advance = {0}
    wrapper._activate_pending_advances(batch, torch.device("cpu"))
    torch.testing.assert_close(expert.history[1], torch.tensor([4, -1]))
    wrapper._pending_advance = {1}
    wrapper._activate_pending_advances(batch, torch.device("cpu"))
    torch.testing.assert_close(expert.history[2], torch.tensor([4, 8]))
    assert "skill_previous_code" not in batch
    wrapper.reset()
    wrapper.select_action(batch)
    torch.testing.assert_close(expert.history[3], torch.tensor([-1, -1]))


def test_rejected_same_skill_replan_does_not_commit_a_new_previous_skill():
    class RecordingExpert(_FakeExpert):
        def __init__(self):
            super().__init__()
            self.history = []
            self.predictions = [torch.tensor([code]) for code in (4, 4, 8)]

        def predict_skill_code(self, batch):
            self.history.append(batch["skill_previous_code"].clone())
            return super().predict_skill_code(batch)

    expert = RecordingExpert()
    wrapper = Stage1OraclePolicy(
        expert, _FakeTerminator(), skill_source="own", advance_mode="terminator",
        end_mode="termination", end_threshold=.5, progress_threshold=.95,
        max_skill_length=0, n_action_steps=2, reject_same_skill_transition=True,
    )
    wrapper.set_reference_skill_token_sequences(
        [[{"token": 3, "gt_length": 5}, {"token": 7, "gt_length": 5}]]
    )
    batch = _batch()
    wrapper.select_action(batch)
    wrapper._pending_advance = {0}
    wrapper._activate_pending_advances(batch, torch.device("cpu"))
    torch.testing.assert_close(wrapper._predictor_previous_codes, torch.tensor([-1]))
    wrapper._pending_advance = {0}
    wrapper._activate_pending_advances(batch, torch.device("cpu"))
    torch.testing.assert_close(wrapper._predictor_previous_codes, torch.tensor([4]))
    assert [x.item() for x in expert.history] == [-1, 4, 4]


def test_oracle_skill_is_selected_once_and_held_until_the_next_boundary(
    monkeypatch,
) -> None:
    class _VideoReader:
        def __init__(self, dataset_dir):
            del dataset_dir

        def read(self, episode_index, timestamps):
            del episode_index
            count = len(timestamps)
            values = torch.arange(1, count + 1, dtype=torch.float32).view(
                count, 1, 1, 1
            )
            return {
                "observation.images.image": values.expand(count, 3, 2, 2),
                "observation.images.wrist_image": (values + 10).expand(
                    count, 3, 2, 2
                ),
            }

    monkeypatch.setattr(run_eval, "_OracleSkillVideoReader", _VideoReader)
    expert = _FakeOracleSkillStage2Expert()
    wrapper = Stage1OraclePolicy(
        expert,
        None,
        skill_source="oracle",
        advance_mode="gt",
        end_mode="max_length",
        end_threshold=0.5,
        progress_threshold=0.95,
        max_skill_length=0,
        n_action_steps=1,
        oracle_skill_timesteps=3,
        oracle_action_q01=[0.0],
        oracle_action_q99=[2.0],
        oracle_state_q01=[0.0],
        oracle_state_q99=[2.0],
        oracle_dataset_dir="unused",
    )
    wrapper.set_reference_skill_token_sequences(
        [
            [
                {
                    "token": 3,
                    "gt_length": 3,
                    "gt_actions": np.asarray(
                        [[[0.0], [1.0]], [[2.0], [0.0]]], dtype=np.float32
                    ),
                    "gt_action_valid": np.asarray(
                        [[True, True], [True, False]]
                    ),
                    "gt_window_states": np.asarray([[0.0], [2.0]], dtype=np.float32),
                    "gt_window_timestamps": np.asarray([0.0, 0.1]),
                    "gt_episode_start_state": np.asarray([0.0], dtype=np.float32),
                    "gt_episode_index": 4,
                },
                {
                    "token": 7,
                    "gt_length": 1,
                    "gt_actions": np.asarray(
                        [[[2.0], [1.0]]], dtype=np.float32
                    ),
                    "gt_action_valid": np.asarray([[True, True]]),
                    "gt_window_states": np.asarray([[1.0]], dtype=np.float32),
                    "gt_window_timestamps": np.asarray([0.2]),
                    "gt_episode_start_state": np.asarray([0.0], dtype=np.float32),
                    "gt_episode_index": 4,
                },
            ]
        ]
    )

    assert wrapper.select_action(_stage2_batch(1)).item() == 9
    assert wrapper.select_action(_stage2_batch(2)).item() == 9
    assert wrapper.select_action(_stage2_batch(3)).item() == 10

    assert len(expert.oracle_skill_targets) == 2
    first = expert.oracle_skill_targets[0]
    torch.testing.assert_close(
        first["actions"],
        torch.tensor([[[ -1.0], [0.0]], [[1.0], [-1.0]]]),
    )
    assert first["timesteps"] == 3
    assert first["aggregate_windows"] is True
    assert first["start_image"].shape[0] == 2
    assert float(first["start_image"].mean()) == pytest.approx(1.0)
    assert [float(value) for value in first["current_image"].mean((1, 2, 3))] == [
        1.0,
        2.0,
    ]
    torch.testing.assert_close(first["state"], torch.tensor([[-1.0], [1.0]]))
    assert float(expert.oracle_skill_targets[1]["start_image"].mean()) == pytest.approx(
        3.0
    )
    trace = wrapper.get_skill_trace()
    assert [record["codebook_token"] for record in trace] == [9, 10]
    assert [record["oracle_skill_gt_code"] for record in trace] == [3, 7]
    assert all(
        record["oracle_skill_score_metric"]
        == "main_route_full_skill_flow_mse"
        for record in trace
    )


def test_gt_timed_advancement_does_not_call_a_terminator() -> None:
    expert = _FakeExpert()
    wrapper = Stage1OraclePolicy(
        expert,
        None,
        skill_source="gt",
        advance_mode="gt",
        end_mode="or",
        end_threshold=0.5,
        progress_threshold=0.95,
        max_skill_length=0,
        n_action_steps=2,
    )
    wrapper.set_forced_skill_token_sequences(
        [[{"token": 3, "gt_length": 1}, {"token": 7, "gt_length": 2}]]
    )

    assert wrapper.select_action({"observation.state": torch.zeros(1, 8)}).item() == 7
    assert [call.item() for call in expert.calls] == [7]


def test_gt_termination_guard_ignores_early_firing() -> None:
    class _Terminator:
        def terminate(self, *args, **kwargs):
            del args, kwargs
            return torch.tensor([0.0]), torch.tensor([1.0])

    expert = _FakeExpert()
    wrapper = Stage1OraclePolicy(
        expert,
        _Terminator(),
        skill_source="gt",
        advance_mode="external",
        end_mode="termination",
        end_threshold=0.5,
        progress_threshold=0.95,
        max_skill_length=0,
        n_action_steps=1,
        gt_termination_min_fraction=0.5,
    )
    wrapper.set_forced_skill_token_sequences(
        [[{"token": 3, "gt_length": 4}, {"token": 7, "gt_length": 2}]]
    )

    # The terminator fires at every step, but the four-step GT skill cannot
    # advance until step 2 (the first half) has been reached.
    batch = {
        "observation.state": torch.zeros(1, 8),
        run_eval.RAW_STATE: torch.zeros(1, 8),
        run_eval.RAW_IMAGE: torch.zeros(1, 3, 2, 2),
        run_eval.RAW_WRIST: torch.zeros(1, 3, 2, 2),
    }
    for _ in range(3):
        wrapper.select_action(batch)
    assert wrapper.get_skill_end_fired() == [True]
    assert wrapper._cursor == [1]


def test_learned_termination_waits_for_executed_action_dwell() -> None:
    class _AlwaysTerminator:
        def terminate(self, codes, *args, **kwargs):
            del args, kwargs
            probability = torch.ones_like(codes, dtype=torch.float32)
            return torch.zeros_like(probability), probability

    expert = _FakeExpert()
    wrapper = Stage1OraclePolicy(
        expert,
        _AlwaysTerminator(),
        skill_source="external",
        advance_mode="external",
        end_mode="termination",
        end_threshold=0.5,
        progress_threshold=0.95,
        max_skill_length=0,
        n_action_steps=2,
        min_skill_steps=5,
        immediate_replan_on_skill_end=True,
    )
    wrapper.set_reference_skill_token_sequences(
        [[{"token": 4, "gt_length": 5}, {"token": 8, "gt_length": 5}]]
    )
    batch = _batch()

    for _ in range(5):
        action = wrapper.select_action(batch)
        wrapper.record_executed_action(action)

    assert [record["codebook_token"] for record in wrapper.get_skill_trace()] == [4]
    assert wrapper._executed_skill_steps == [5]

    wrapper.select_action(batch)
    assert [record["codebook_token"] for record in wrapper.get_skill_trace()] == [4, 8]
    assert wrapper._executed_skill_steps == [0]


def test_same_predicted_skill_is_replanned_without_new_occurrence() -> None:
    class _AlwaysTerminator:
        def terminate(self, codes, *args, **kwargs):
            del args, kwargs
            probability = torch.ones_like(codes, dtype=torch.float32)
            return torch.zeros_like(probability), probability

    expert = _FakeExpert()
    expert.predictions = [torch.tensor([4]), torch.tensor([4]), torch.tensor([8])]
    wrapper = Stage1OraclePolicy(
        expert,
        _AlwaysTerminator(),
        skill_source="external",
        advance_mode="external",
        end_mode="termination",
        end_threshold=0.5,
        progress_threshold=0.95,
        max_skill_length=0,
        n_action_steps=2,
        reject_same_skill_transition=True,
        immediate_replan_on_skill_end=True,
    )
    wrapper.set_reference_skill_token_sequences(
        [[{"token": 4, "gt_length": 5}, {"token": 8, "gt_length": 5}]]
    )
    batch = _batch()

    action = wrapper.select_action(batch)
    wrapper.record_executed_action(action)
    trace = wrapper.get_skill_trace()
    assert [record["codebook_token"] for record in trace] == [4]
    assert len(trace[0]["same_skill_transition_rejections"]) == 1
    assert wrapper.get_skill_end_fired() == [False]

    wrapper.select_action(batch)
    assert [record["codebook_token"] for record in wrapper.get_skill_trace()] == [4, 8]


@pytest.mark.parametrize("skill_source", ["gt", "own", "external"])
@pytest.mark.parametrize(
    ("advance_mode", "terminator_variant"),
    [
        ("gt", "state_image"),
        ("gt", "image_only"),
        ("own", "state_image"),
        ("own", "image_only"),
        ("external", "state_image"),
        ("external", "image_only"),
        ("original", "state_image"),
    ],
)
def test_stage1_eval_selects_own_external_original_or_gt_skill_modules(
    monkeypatch,
    skill_source: str,
    advance_mode: str,
    terminator_variant: str,
) -> None:
    class _Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SkillExpertConfig(n_action_steps=2, chunk_size=2)
            self.model = SimpleNamespace(
                skill_predictor=object(),
                fsq_term_train=object(),
                fsq_image_term_train=object(),
            )
            self.loaded_predictor = None
            self.loaded_skill_head = None
            self.loaded_terminator = None
            self.loaded_image_terminator = None

        def load_external_skill_predictor(self, checkpoint):
            self.loaded_predictor = checkpoint
            self.model.skill_predictor = object()

        def load_external_skill_head(self, checkpoint):
            self.loaded_skill_head = checkpoint

        def load_external_terminator(self, checkpoint):
            self.loaded_terminator = checkpoint
            self.model.fsq_term_train = object()

        def load_external_image_only_terminator(self, checkpoint):
            self.loaded_image_terminator = checkpoint
            self.model.fsq_image_term_train = object()

        def reset(self):
            return None

        def get_optim_params(self):
            return []

    policies = []
    original_terminator_paths = []

    def make_test_policy(**kwargs):
        del kwargs
        policy = _Policy()
        policies.append(policy)
        return policy

    resolved_config = SimpleNamespace(
        type="skill_expert",
        n_action_steps=2,
        pretrained_path=Path("/tmp/stage1"),
        tokenizer_path="/tmp/tokenizer",
        use_amp=False,
    )
    monkeypatch.setattr(run_eval, "_policy_config", lambda *args: resolved_config)
    monkeypatch.setattr(run_eval, "make_policy", make_test_policy)
    monkeypatch.setattr(
        run_eval,
        "_attach_original_terminator",
        lambda policy, path: original_terminator_paths.append(str(path)),
    )
    processor_calls = []

    def make_test_processors(**kwargs):
        processor_calls.append(kwargs)
        return object(), object()

    monkeypatch.setattr(run_eval, "make_pre_post_processors", make_test_processors)
    monkeypatch.setattr(
        run_eval,
        "_saved_preprocessor_step_names",
        lambda *args: [
            "rename_observations_processor",
            "to_batch_processor",
            "normalizer_processor",
            "tokenizer_processor",
            "skill_vla_terminator_predictor_tokenizer_processor_step",
            "device_processor",
        ],
    )
    runtime_calls = []
    quantile_paths = []
    predictor_quantiles = (np.zeros(8), np.ones(8))

    def read_predictor_quantiles(path):
        quantile_paths.append(str(path))
        return predictor_quantiles

    monkeypatch.setattr(run_eval, "_predictor_state_quantiles", read_predictor_quantiles)
    monkeypatch.setattr(
        run_eval, "_ensure_skill_runtime_steps",
        lambda *args, **kwargs: runtime_calls.append(kwargs),
    )
    monkeypatch.setenv("SKILL_END_MODE", "or")
    monkeypatch.setenv("SKILL_END_THRESHOLD", "0.5")
    monkeypatch.setenv("SKILL_END_PROGRESS_THRESHOLD", "0.9")
    monkeypatch.setenv("INFERENCE_SKILL_MAX_LENGTH", "200")

    context = run_eval._build_context(
        {
            "label": "source-selection",
            "skill_source": skill_source,
            # A Joint checkpoint uses its own Predictor for both outputs; it
            # must not be replaced just because the XYZ source is predicted.
            "pose_source": "predictor" if skill_source == "own" else "gt",
            "advance_mode": advance_mode,
            "terminator_variant": terminator_variant,
            "external_skill_model": "/tmp/external",
            "external_predictor_skill_model": (
                "/tmp/skill-head" if skill_source == "external" else ""
            ),
            "fsq_path": "/tmp/fsq/FSQ.pt",
            "tokenizer_path": "/tmp/tokenizer",
        },
        SimpleNamespace(policy=object(), env=object(), rename_map={}),
        torch.device("cpu"),
    )

    policy = policies[0]
    assert policy.loaded_predictor == (
        "/tmp/external" if skill_source == "external" else None
    )
    assert quantile_paths == ["/tmp/stage1" if skill_source == "own" else "/tmp/external"]
    assert runtime_calls[0]["predictor_state_quantiles"] is predictor_quantiles
    assert policy.loaded_skill_head == (
        "/tmp/skill-head" if skill_source == "external" else None
    )
    assert policy.loaded_terminator == (
        "/tmp/external"
        if advance_mode == "external" and terminator_variant == "state_image"
        else None
    )
    assert policy.loaded_image_terminator == (
        "/tmp/external"
        if advance_mode == "external" and terminator_variant == "image_only"
        else None
    )
    assert original_terminator_paths == (
        ["/tmp/fsq/FSQ.pt"] if advance_mode == "original" else []
    )
    tokenizer_overrides = processor_calls[0]["preprocessor_overrides"]
    assert tokenizer_overrides["tokenizer_processor"]["tokenizer_name"] == "/tmp/tokenizer"
    assert (
        tokenizer_overrides[
            "skill_vla_terminator_predictor_tokenizer_processor_step"
        ]["tokenizer_name"]
        == "/tmp/tokenizer"
    )
    assert context["policy"].skill_source == skill_source
    assert context["policy"].pose_source == (
        "predictor" if skill_source == "own" else "gt"
    )
    assert context["policy"].advance_mode == advance_mode
    assert (policy.model.skill_predictor is None) == (skill_source == "gt")


def test_the_full_skill_oracle_carries_the_skill_start_state_for_arch16_to_arch18() -> None:
    """Arch16--Arch18 build the Expert goal from the skill START too, so a replay must carry it."""
    expert = _FakeExpert()
    expert.config.architecture_label = "arch18_skill"
    expert.config.proprio_grounding = "episode_start_xyz"
    wrapper = Stage1OraclePolicy(
        expert,
        None,
        advance_mode="gt",
        end_mode="max_length",
        end_threshold=0.5,
        progress_threshold=0.95,
        max_skill_length=0,
        n_action_steps=2,
    )
    states = np.array(
        [[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.0, 0.0], [4.0, 5.0, 6.0, 0.4, 0.5, 0.6, 0.0, 0.0]],
        dtype=np.float32,
    )
    wrapper._oracle_action_payload_at = lambda *_: {
        "actions": np.zeros((2, 2, 7), dtype=np.float32),
        "valid": np.ones((2, 2), dtype=bool),
        "states": states,
        "timestamps": np.array([0.0, 0.05]),
        "episode_start_state": np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "episode_index": 0,
    }
    wrapper._oracle_action_q01, wrapper._oracle_action_q99 = torch.zeros(7), torch.ones(7)
    wrapper._oracle_state_q01, wrapper._oracle_state_q99 = torch.zeros(8), torch.ones(8)
    wrapper._oracle_video_reader = SimpleNamespace(
        read=lambda *_: {
            "observation.images.image": torch.zeros(2, 3, 8, 8),
            "observation.images.wrist_image": torch.zeros(2, 3, 8, 8),
        }
    )
    action_batch = {"skill_end_state": torch.tensor([[7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0]])}

    full_batch, _, _ = wrapper._full_skill_oracle_inputs(action_batch, 0, 0, torch.device("cpu"))

    # Raw grounded metres, like the latch in select_action: the skill's first window minus the
    # episode-start xyz, shared by every window of the skill.
    assert full_batch["skill_start_state"].shape == (2, 8)
    torch.testing.assert_close(
        full_batch["skill_start_state"][:, :3],
        torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]),
    )
    # Without the goal the failure names the unsupported combination instead of a bare KeyError.
    with pytest.raises(ValueError, match="skill_source=oracle is unsupported"):
        wrapper._full_skill_oracle_inputs({}, 0, 0, torch.device("cpu"))


def test_goal_noise_is_paired_cached_and_resampled_per_rollout() -> None:
    left = GoalNoisePerturber(std_m=0.005, target="both", seed=17)
    right = GoalNoisePerturber(std_m=0.005, target="both", seed=17)
    left.reset()
    right.reset()

    first = left.noise_xyz(target="end", batch_index=0, skill_order=2)
    np.testing.assert_array_equal(
        first, left.noise_xyz(target="end", batch_index=0, skill_order=2)
    )
    np.testing.assert_array_equal(
        first, right.noise_xyz(target="end", batch_index=0, skill_order=2)
    )
    assert not np.array_equal(
        first, left.noise_xyz(target="start", batch_index=0, skill_order=2)
    )

    left.reset()
    assert not np.array_equal(
        first, left.noise_xyz(target="end", batch_index=0, skill_order=2)
    )


def test_arch18_goal_noise_is_added_in_raw_metres_before_the_policy() -> None:
    class _GoalRecordingExpert(_FakeExpert):
        def __init__(self, architecture_label: str):
            super().__init__()
            self.config.architecture_label = architecture_label
            self.config.skill_end_pose_mode = "xyz"
            self.received = None

        def predict_action_chunk(self, batch):
            self.received = (
                batch["skill_end_state"].detach().clone(),
                batch["skill_start_state"].detach().clone(),
            )
            return super().predict_action_chunk(batch)

    def run(architecture_label: str):
        expert = _GoalRecordingExpert(architecture_label)
        wrapper = Stage1OraclePolicy(
            expert,
            None,
            advance_mode="gt",
            end_mode="max_length",
            end_threshold=0.5,
            progress_threshold=0.95,
            max_skill_length=0,
            n_action_steps=2,
            goal_noise_std_m=0.005,
            goal_noise_target="both",
            goal_noise_seed=23,
        )
        clean_end = torch.tensor([0.10, -0.20, 0.30])
        wrapper.set_forced_skill_token_sequences(
            [[{
                "token": 3,
                "gt_length": 5,
                "end_state": clean_end.tolist() + [0.0, 0.0, 0.0],
            }]]
        )
        batch = _batch()
        clean_start = torch.tensor([0.01, 0.02, -0.03])
        batch["skill_decoder_state"][0, :3] = clean_start
        wrapper.select_action(batch)
        received_end, received_start = expert.received
        trace = wrapper.get_skill_trace()[0]
        return (
            received_end[0, :3] - clean_end,
            received_start[0, :3] - clean_start,
            trace,
        )

    raw_end_noise, raw_start_noise, raw_trace = run("arch18_align_skill")
    norm_end_noise, norm_start_noise, norm_trace = run("arch18_align_norm_skill")

    # The evaluator perturbs the raw batch. Normalization, if any, happens later
    # inside the policy, and model identity never changes the paired noise draw.
    torch.testing.assert_close(raw_end_noise, norm_end_noise)
    torch.testing.assert_close(raw_start_noise, norm_start_noise)
    assert raw_trace["goal_noise_end_xyz_m"] == norm_trace["goal_noise_end_xyz_m"]
    assert raw_trace["goal_noise_start_xyz_m"] == norm_trace["goal_noise_start_xyz_m"]
    for label in run_eval.DIRECT_PATCH_ARCH_LABELS:
        direct_end, direct_start, direct_trace = run(label)
        torch.testing.assert_close(direct_end, raw_end_noise)
        torch.testing.assert_close(direct_start, raw_start_noise)
        assert direct_trace["goal_noise_start_xyz_m"] == raw_trace["goal_noise_start_xyz_m"]


def test_gt_skills_can_be_paired_with_a_predicted_goal() -> None:
    """The spatial head is skill-conditioned, so the GT skill isolates the predicted GOAL."""

    class _PredictingExpert(_FakeExpert):
        def __init__(self):
            super().__init__()
            self.config.architecture_label = "arch18_align_skill"
            self.config.skill_end_pose_mode = "xyz"
            self.asked_with = "unset"
            self.received = None

        def predict_skill_code_and_end_state(self, batch, skill_code=None):
            del batch
            self.asked_with = None if skill_code is None else skill_code.clone()
            # A predictor that disagrees with the GT skill: its code must be discarded here.
            return torch.tensor([9]), torch.tensor([[0.25, -0.5, 0.75]])

        def predict_action_chunk(self, batch):
            self.received = (batch["skill_code"].clone(), batch["skill_end_state"].clone())
            return super().predict_action_chunk(batch)

    expert = _PredictingExpert()
    wrapper = Stage1OraclePolicy(
        expert, None, skill_source="gt", pose_source="predictor",
        advance_mode="gt", end_mode="max_length", end_threshold=0.5,
        progress_threshold=0.95, max_skill_length=0, n_action_steps=2,
    )
    wrapper.set_forced_skill_token_sequences(
        [[{"token": 3, "gt_length": 5, "end_state": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]}]]
    )
    wrapper.select_action(_batch())

    code, pose = expert.received
    torch.testing.assert_close(code, torch.tensor([3]))               # the GT skill drives the VSA
    torch.testing.assert_close(pose, torch.tensor([[0.25, -0.5, 0.75]]))   # ...with the PREDICTED goal
    # The predictor was conditioned on that same GT skill rather than choosing its own.
    torch.testing.assert_close(expert.asked_with, torch.tensor([3]))


def test_an_ordinary_predictor_panel_is_still_asked_for_its_own_skill() -> None:
    """skill_source=external must keep working exactly as before (no skill code forced on it)."""

    class _PredictingExpert(_FakeExpert):
        def __init__(self):
            super().__init__()
            self.config.architecture_label = "arch18_align_skill"
            self.config.skill_end_pose_mode = "xyz"
            self.asked_with = "unset"

        def predict_skill_code_and_end_state(self, batch, skill_code=None):
            del batch
            self.asked_with = skill_code
            return torch.tensor([4]), torch.tensor([[0.25, -0.5, 0.75]])

    expert = _PredictingExpert()
    wrapper = Stage1OraclePolicy(
        expert, None, skill_source="external", pose_source="predictor",
        advance_mode="gt", end_mode="max_length", end_threshold=0.5,
        progress_threshold=0.95, max_skill_length=0, n_action_steps=2,
    )
    wrapper.set_reference_skill_token_sequences([[{"token": 3, "gt_length": 5}]])
    wrapper.select_action(_batch())

    assert expert.asked_with is None
