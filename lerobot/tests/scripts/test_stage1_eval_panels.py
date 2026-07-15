import sys
from pathlib import Path


_EVAL_SRC = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_skillVLA/stage1_eval/src"
sys.path.insert(0, str(_EVAL_SRC))

from stage1_eval_config import _expand_stage1_panels


def test_stage1_entry_expands_to_action_mode_triplet() -> None:
    stage1 = {
        "kind": "stage1",
        "label": "candidate",
        "modes": "",
        "fsq_path": Path("/tmp/FSQ.pt"),
        "policy_path": Path("/tmp/pretrained_model"),
    }

    panels = _expand_stage1_panels([stage1])

    assert [p["label"] for p in panels] == [
        "candidate [A]", "candidate [B]", "candidate [Frozen-FSQ]",
    ]
    assert [p["kind"] for p in panels] == ["stage1", "stage1", "fsq_expert"]
    assert [p["eval_mode"] for p in panels] == ["a", "b", "frozen_expert"]
    assert panels[2]["expert_fsq_path"] == stage1["fsq_path"]
    assert panels[2]["terminator_policy_path"] == stage1["policy_path"]


def test_fsq_only_entry_remains_one_panel() -> None:
    fsq = {"kind": "fsq_expert", "label": "pretrain"}

    assert _expand_stage1_panels([fsq]) == [{
        "kind": "fsq_expert", "label": "pretrain", "eval_mode": "fsq_only",
    }]
