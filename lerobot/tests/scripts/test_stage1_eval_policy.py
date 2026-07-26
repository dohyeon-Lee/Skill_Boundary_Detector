import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


_EVAL_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/stage1_eval/src"
)
sys.path.insert(0, str(_EVAL_SRC))

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


def test_oracle_replans_immediately_when_terminator_advances() -> None:
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
    assert wrapper.select_action(_batch()).item() == 8
    assert [call.item() for call in expert.calls] == [4, 8]
    assert [record["skill_source"] for record in wrapper.get_skill_trace()] == [
        "predictor",
        "predictor",
    ]
