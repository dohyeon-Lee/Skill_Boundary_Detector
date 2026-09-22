from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/stage1_eval/src/merge_eval_chunks.py"
)
SPEC = importlib.util.spec_from_file_location("stage1_merge_eval_chunks", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _merged(labels: list[str]) -> dict:
    return {
        label: {"overall": {"pc_success": 10.0 * index, "n_tasks": 10}, "per_task": []}
        for index, label in enumerate(labels)
    }


def test_checkpoint_sweep_groups_panels_by_model_setting() -> None:
    labels = ["A | ckpt 010000", "B | ckpt 010000", "A | ckpt 005000", "B | ckpt 005000"]
    sweep = MODULE.checkpoint_sweep(labels)
    assert sweep == {
        "A": [(5000, "A | ckpt 005000"), (10000, "A | ckpt 010000")],
        "B": [(5000, "B | ckpt 005000"), (10000, "B | ckpt 010000")],
    }
    # Ordinary comparisons and single-checkpoint evals are not sweeps.
    assert MODULE.checkpoint_sweep(["GT_skill_GT_term", "PRED_skill_PRED_term"]) is None
    assert MODULE.checkpoint_sweep(["A | ckpt 010000", "plain"]) is None
    assert MODULE.checkpoint_sweep(["A | ckpt 010000", "B | ckpt 010000"]) is None


def test_sweep_with_more_panels_than_colors_still_gets_a_chart(tmp_path: Path) -> None:
    labels = [f"{name} | ckpt {step:06d}" for step in range(5000, 35000, 5000) for name in ("A", "B")]
    assert len(labels) == 12 > len(MODULE._PALETTE)
    chart = tmp_path / "checkpoint_success_rates.png"
    assert MODULE.draw_checkpoint_chart(_merged(labels), labels, chart, expected_tasks=10) is True
    assert chart.stat().st_size > 0
    plain = ["A", "B"]
    assert MODULE.draw_checkpoint_chart(_merged(plain), plain, tmp_path / "none.png", expected_tasks=10) is False
