import sys
from pathlib import Path
from types import SimpleNamespace

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
def test_stage1_eval_selects_own_external_or_gt_skill_modules(
    monkeypatch, skill_source: str, advance_mode: str
) -> None:
    class _Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SkillExpertConfig(n_action_steps=2, chunk_size=2)
            self.model = SimpleNamespace(
                skill_predictor=object(), fsq_term_train=object()
            )
            self.loaded_predictor = None
            self.loaded_terminator = None

        def load_external_skill_predictor(self, checkpoint):
            self.loaded_predictor = checkpoint
            self.model.skill_predictor = object()

        def load_external_terminator(self, checkpoint):
            self.loaded_terminator = checkpoint
            self.model.fsq_term_train = object()

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
        "/tmp/external" if advance_mode == "external" else None
    )
    assert context["policy"].skill_source == skill_source
    assert context["policy"].advance_mode == advance_mode
    assert (policy.model.skill_predictor is None) == (skill_source == "gt")
