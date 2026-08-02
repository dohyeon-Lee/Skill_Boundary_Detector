import sys
from pathlib import Path

import pytest
import torch


_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/stage1_eval/src"
)
sys.path.insert(0, str(_SRC))
from input_influence_eval import (  # noqa: E402
    _counterfactual_batch,
    _different_skill_codes,
    _masked_errors,
)


def test_masked_errors_are_per_action_element() -> None:
    target = torch.zeros(2, 2, 2)
    prediction = torch.tensor(
        [[[1.0, 1.0], [9.0, 9.0]], [[2.0, 0.0], [0.0, 2.0]]]
    )
    padding = torch.tensor([[False, True], [False, False]])

    mse, mae = _masked_errors(prediction, target, padding)

    torch.testing.assert_close(mse, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(mae, torch.tensor([1.0, 1.0]))


def test_different_skill_codes_never_leave_code_unchanged() -> None:
    codes = torch.tensor([2, 2, 5, 5])
    wrong, fallback = _different_skill_codes(codes, vocab_size=27)

    assert torch.all(wrong != codes)
    assert not fallback.any()

    identical = torch.tensor([3, 3, 3])
    wrong, fallback = _different_skill_codes(identical, vocab_size=27)
    assert torch.all(wrong == 4)
    assert fallback.all()


@pytest.mark.parametrize(
    ("condition", "changed"),
    [
        ("state_swap", {"observation.state"}),
        ("top_image_swap", {"observation.images.image"}),
        ("wrist_image_swap", {"observation.images.wrist_image"}),
        (
            "image_swap",
            {"observation.images.image", "observation.images.wrist_image"},
        ),
        ("skill_swap", {"skill_code", "skill_sequence"}),
    ],
)
def test_counterfactual_changes_only_requested_inputs(condition: str, changed: set[str]) -> None:
    batch = {
        "observation.state": torch.arange(6).reshape(3, 2),
        "observation.images.image": torch.arange(3).reshape(3, 1),
        "observation.images.wrist_image": torch.arange(3).reshape(3, 1) + 10,
        "skill_code": torch.tensor([0, 1, 2]),
        "skill_sequence": torch.tensor([[0], [1], [2]]),
        "skill_index": torch.zeros(3, dtype=torch.long),
    }
    variant, _ = _counterfactual_batch(
        batch, condition, batch["skill_code"], vocab_size=27
    )

    for key, original in batch.items():
        if key in changed:
            assert not torch.equal(variant[key], original)
        else:
            assert torch.equal(variant[key], original)
