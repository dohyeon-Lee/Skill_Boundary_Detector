import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


_EVAL_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/stage1_eval/src"
)
sys.path.insert(0, str(_EVAL_SRC))

import run_eval
from run_eval import CheckpointTerminator, Stage1OraclePolicy
from lerobot.policies.skill_expert import modeling_skill_expert
from lerobot.policies.skill_expert.configuration_skill_expert import SkillExpertConfig
from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy
from lerobot.policies.skill_expert.processor_skill_expert import (
    EpisodeStartXYZGroundingProcessorStep,
)
from lerobot.scripts.lerobot_skillvla_eval import (
    _annotate_eval_video,
    _progress_values_from_trace,
    _skill_banner_color,
    _skill_ids_from_trace,
    _termination_values_from_trace,
)
from lerobot.utils.constants import STAGE2_VLM_CACHE_ID
from lerobot.types import TransitionKey
from lerobot.utils.constants import OBS_STATE


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


def test_external_terminator_is_rebuilt_from_its_saved_contract(
    monkeypatch, tmp_path: Path
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
          "terminator_termination_only": true
        }"""
    )

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
                "context": "prev_action",
                "default_arch": "fusion",
                "vision_backbone": "dino",
                "freeze_vision_encoder": False,
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


def test_video_progress_gauge_adds_dynamic_right_panel() -> None:
    frames = np.zeros((3, 120, 160, 3), dtype=np.uint8)
    annotated = _annotate_eval_video(
        frames,
        success=True,
        task_description="pick up the object",
        skill_ids=[3, 3, 8],
        progress_values=[0.1, 0.5, 0.95],
        progress_threshold=0.9,
    )

    assert annotated.shape[2] == 160 + max(48, 160 // 6)
    top_bar_height = max(18, 120 // 10)
    right_panel = annotated[:, top_bar_height : top_bar_height + 120, 160:]
    assert not np.array_equal(right_panel[0], right_panel[1])
    assert not np.array_equal(right_panel[1], right_panel[2])


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


def test_video_progress_and_termination_gauges_are_side_by_side() -> None:
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


@pytest.mark.parametrize(
    ("include_state", "include_skill"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_policy_config_keeps_checkpoint_visual_query_switches(
    monkeypatch, include_state: bool, include_skill: bool
) -> None:
    loaded = SimpleNamespace(
        include_state_in_visual_crossattn=include_state,
        include_skill_in_visual_crossattn=include_skill,
    )
    monkeypatch.setattr(
        run_eval.PreTrainedConfig,
        "from_pretrained",
        lambda *args, **kwargs: loaded,
    )

    result = run_eval._policy_config(
        {
            "policy_path": "/tmp/new-stage1",
            "include_state_in_visual_crossattn": include_state,
            "include_skill_in_visual_crossattn": include_skill,
            "fsq_path": "/tmp/fsq",
            "dino_model_path": "/tmp/dino",
            "terminator_dino_model_path": "/tmp/term-dino",
            "tokenizer_path": "/tmp/tokenizer",
        },
        SimpleNamespace(use_amp=False, n_action_steps=5),
        torch.device("cpu"),
    )

    assert result.include_state_in_visual_crossattn is include_state
    assert result.include_skill_in_visual_crossattn is include_skill
    assert result.visual_perceiver_width == 1024


def test_policy_config_rejects_visual_query_contract_drift(monkeypatch) -> None:
    loaded = SimpleNamespace(
        include_state_in_visual_crossattn=False,
        include_skill_in_visual_crossattn=False,
    )
    monkeypatch.setattr(
        run_eval.PreTrainedConfig,
        "from_pretrained",
        lambda *args, **kwargs: loaded,
    )

    with pytest.raises(RuntimeError, match="contract changed.*include_state"):
        run_eval._policy_config(
            {
                "policy_path": "/tmp/new-stage1",
                "include_state_in_visual_crossattn": True,
            },
            SimpleNamespace(use_amp=False, n_action_steps=5),
            torch.device("cpu"),
        )


def test_policy_config_keeps_checkpoint_vision_mode(monkeypatch) -> None:
    loaded = SimpleNamespace(
        vision_conditioning_mode="in_context_tokens",
        include_state_in_visual_crossattn=True,
        include_skill_in_visual_crossattn=True,
    )
    monkeypatch.setattr(
        run_eval.PreTrainedConfig,
        "from_pretrained",
        lambda *args, **kwargs: loaded,
    )

    result = run_eval._policy_config(
        {
            "policy_path": "/tmp/in-context-stage1",
            "vision_conditioning_mode": "in_context_tokens",
            "include_state_in_visual_crossattn": True,
            "include_skill_in_visual_crossattn": True,
            "fsq_path": "/tmp/fsq",
            "dino_model_path": "/tmp/dino",
            "terminator_dino_model_path": "/tmp/term-dino",
            "tokenizer_path": "/tmp/tokenizer",
        },
        SimpleNamespace(use_amp=False, n_action_steps=5),
        torch.device("cpu"),
    )

    assert result.vision_conditioning_mode == "in_context_tokens"


def test_policy_config_materializes_implicit_skillvla_real_architecture(
    monkeypatch,
) -> None:
    # Loading an old config through the current dataclass supplies the VSA
    # defaults for fields that did not exist on skillVLA_real. The raw-config
    # resolver marks that case explicitly, so eval restores the Cond contract.
    loaded = SimpleNamespace(
        architecture="vsa_perceiver_crossattn",
        architecture_revision="residual_sa18_v2",
        conditioning_route="state_skill_cond",
    )
    monkeypatch.setattr(
        run_eval.PreTrainedConfig,
        "from_pretrained",
        lambda *args, **kwargs: loaded,
    )

    result = run_eval._policy_config(
        {
            "policy_path": "/tmp/skillvla-real-stage1",
            "architecture": "cond_gemma",
            "architecture_revision": "skillvla_real_v1",
            "architecture_inferred": True,
            "conditioning_route": "state_skill_cond",
            "fsq_path": "/tmp/fsq",
            "dino_model_path": "/tmp/dino",
            "terminator_dino_model_path": "/tmp/term-dino",
            "tokenizer_path": "/tmp/tokenizer",
        },
        SimpleNamespace(use_amp=False, n_action_steps=5),
        torch.device("cpu"),
    )

    assert result.architecture == "cond_gemma"
    assert result.architecture_revision == "skillvla_real_v1"
    assert result.conditioning_route == "state_skill_cond"


def test_policy_config_verifies_checkpoint_owned_dsbc_mode(monkeypatch) -> None:
    loaded = SimpleNamespace(
        type="skill_vla_stage2",
        architecture="cond_gemma",
        architecture_label="arch0",
        conditioning_route="state_cond",
        stage2_mode="dsbc",
        dsbc_noise_output_mode="per_step",
        dsbc_frs_num_steps=8,
        dsbc_anchor_seed=17,
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
            self.loaded_terminator = None
            self.loaded_image_terminator = None

        def load_external_skill_predictor(self, checkpoint):
            self.loaded_predictor = checkpoint
            self.model.skill_predictor = object()

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
        use_amp=False,
    )
    monkeypatch.setattr(run_eval, "_policy_config", lambda *args: resolved_config)
    monkeypatch.setattr(run_eval, "make_policy", make_test_policy)
    monkeypatch.setattr(
        run_eval,
        "_attach_original_terminator",
        lambda policy, path: original_terminator_paths.append(str(path)),
    )
    monkeypatch.setattr(
        run_eval, "make_pre_post_processors", lambda **kwargs: (object(), object())
    )
    monkeypatch.setattr(
        run_eval,
        "_saved_preprocessor_step_names",
        lambda *args: [
            "rename_observations_processor",
            "to_batch_processor",
            "normalizer_processor",
            "device_processor",
        ],
    )
    monkeypatch.setattr(run_eval, "_ensure_skill_runtime_steps", lambda *args, **kwargs: None)
    monkeypatch.setenv("SKILL_END_MODE", "or")
    monkeypatch.setenv("SKILL_END_THRESHOLD", "0.5")
    monkeypatch.setenv("SKILL_END_PROGRESS_THRESHOLD", "0.9")
    monkeypatch.setenv("INFERENCE_SKILL_MAX_LENGTH", "200")

    context = run_eval._build_context(
        {
            "label": "source-selection",
            "skill_source": skill_source,
            "advance_mode": advance_mode,
            "terminator_variant": terminator_variant,
            "external_skill_model": "/tmp/external",
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
    assert context["policy"].skill_source == skill_source
    assert context["policy"].advance_mode == advance_mode
    assert (policy.model.skill_predictor is None) == (skill_source == "gt")
