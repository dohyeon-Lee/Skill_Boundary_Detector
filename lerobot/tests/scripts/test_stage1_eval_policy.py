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
from lerobot.policies.skill_expert.configuration_skill_expert import SkillExpertConfig
from lerobot.scripts.lerobot_skillvla_eval import (
    _annotate_eval_video,
    _progress_values_from_trace,
    _skill_banner_color,
    _skill_ids_from_trace,
    _termination_values_from_trace,
)


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


class _FakeTerminator:
    use_wrist = True

    def __init__(self):
        self.step = 0

    def terminate(self, codes, state, image, wrist):
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
@pytest.mark.parametrize("advance_mode", ["gt", "own", "external"])
@pytest.mark.parametrize("terminator_variant", ["state_image", "image_only"])
def test_stage1_eval_selects_own_external_or_gt_skill_modules(
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
    assert context["policy"].skill_source == skill_source
    assert context["policy"].advance_mode == advance_mode
    assert (policy.model.skill_predictor is None) == (skill_source == "gt")
