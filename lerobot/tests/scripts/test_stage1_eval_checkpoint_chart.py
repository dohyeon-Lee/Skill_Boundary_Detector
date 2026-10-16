from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

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


def test_ft_checkpoint_sweep_groups_zero_and_non_thousand_steps() -> None:
    labels = ["A_10k", "A_ckpt_002500", "A_0k"]
    assert MODULE.checkpoint_sweep(labels) == {
        "A": [(0, "A_0k"), (2500, "A_ckpt_002500"), (10000, "A_10k")],
    }


def test_sweep_with_more_panels_than_colors_still_gets_a_chart(tmp_path: Path) -> None:
    labels = [f"{name} | ckpt {step:06d}" for step in range(5000, 35000, 5000) for name in ("A", "B")]
    assert len(labels) == 12 > len(MODULE._PALETTE)
    chart = tmp_path / "checkpoint_success_rates.png"
    assert MODULE.draw_checkpoint_chart(_merged(labels), labels, chart, expected_tasks=10) is True
    assert chart.stat().st_size > 0
    plain = ["A", "B"]
    assert MODULE.draw_checkpoint_chart(_merged(plain), plain, tmp_path / "none.png", expected_tasks=10) is False


def _merged_with_tasks(labels: list[str]) -> dict:
    return {
        label: {
            "overall": {"pc_success": 10.0 * index, "n_tasks": 2},
            "per_task": [
                {"task_group": "libero_90", "task_id": task, "pc_success": 50.0} for task in (0, 1)
            ],
        }
        for index, label in enumerate(labels)
    }


def test_more_panels_than_colors_still_get_a_task_chart(tmp_path: Path) -> None:
    """Regression: ten architectures in one comparison used to skip the chart entirely."""
    labels = [f"arch{index}" for index in range(6, 16)]
    assert len(labels) == 10 > len(MODULE._PALETTE)
    chart = tmp_path / "task_success_rates.png"
    assert MODULE.draw_chart(_merged_with_tasks(labels), labels, chart, expected_tasks=2) is True
    assert chart.stat().st_size > 0

    too_many = [f"arch{index}" for index in range(len(MODULE._PALETTE) * len(MODULE._HATCHES) + 1)]
    assert MODULE.draw_chart(_merged_with_tasks(too_many), too_many, tmp_path / "big.png", expected_tasks=2) is False


def test_a_sweep_also_gets_one_task_chart_per_checkpoint(tmp_path: Path) -> None:
    """Lines show the trend; the per-checkpoint bars show which task each model fails."""
    labels = [f"{name} | ckpt {step:06d}" for step in (5000, 10000) for name in ("A", "B")]
    written = MODULE.sweep_task_charts(
        _merged_with_tasks(labels), labels, tmp_path, expected_tasks=2
    )
    assert [path.name for path in written] == [
        "task_success_rates_ckpt005000.png",
        "task_success_rates_ckpt010000.png",
    ]
    assert all(path.stat().st_size > 0 for path in written)
    # Ordinary comparisons keep just the one chart.
    assert MODULE.sweep_task_charts(_merged_with_tasks(["A", "B"]), ["A", "B"], tmp_path, expected_tasks=2) == []


def test_task_checkpoint_rates_keep_missing_results_as_gaps() -> None:
    labels = ["A | ckpt 002000", "A | ckpt 010000", "A | ckpt 020000"]
    merged = _merged_with_tasks(labels)
    merged[labels[0]]["per_task"][0]["pc_success"] = 20.0
    merged[labels[1]]["per_task"] = [merged[labels[1]]["per_task"][1]]
    merged[labels[2]]["per_task"][0]["pc_success"] = 80.0
    points = MODULE.checkpoint_sweep(labels)["A"]
    rates = MODULE.checkpoint_task_rates(merged, points)
    first = rates[("libero_90", 0)]
    assert first[0] == 20.0 and math.isnan(first[1]) and first[2] == 80.0
    assert rates[("libero_90", 1)] == [50.0, 50.0, 50.0]


def test_task_checkpoint_lines_are_separate_per_model_and_sorted(tmp_path: Path, monkeypatch) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.axes import Axes

    labels = [f"{name} | ckpt {step:06d}" for step in (10000, 2000) for name in ("A", "B")]
    merged = _merged_with_tasks(labels)
    # Distinct model values make accidental mixing of the two models visible.
    for label in labels:
        for task in merged[label]["per_task"]:
            task["pc_success"] = (20 if label.startswith("A") else 70) + task["task_id"] * 10
    calls = []
    original = Axes.plot

    def plot(axis, steps, values, **kwargs):
        calls.append((list(steps), list(values), kwargs["label"]))
        return original(axis, steps, values, **kwargs)

    monkeypatch.setattr(Axes, "plot", plot)
    paths = MODULE.draw_checkpoint_task_charts(merged, labels, tmp_path, expected_tasks=2)
    assert [path.name for path in paths] == ["00_A.png", "01_B.png"]
    assert all(path.parent.name == "checkpoint_task_success_rates" and path.stat().st_size > 0 for path in paths)
    assert calls == [
        ([2000, 10000], [20.0, 20.0], "Task 0"),
        ([2000, 10000], [30.0, 30.0], "Task 1"),
        ([2000, 10000], [70.0, 70.0], "Task 0"),
        ([2000, 10000], [80.0, 80.0], "Task 1"),
    ]
    assert MODULE.draw_checkpoint_task_charts(_merged_with_tasks(["A"]), ["A"], tmp_path, expected_tasks=2) == []


def test_merge_automatically_writes_overall_and_model_task_line_charts(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    labels = [f"{name}_{step}k" for step in (10, 20) for name in ("A", "B")]
    for task_id in (0, 1):
        chunks = {
            label: {"per_task": [{
                "task_group": "libero_90", "task_id": task_id,
                "metrics": {"successes": [True, task_id == 0]},
            }]}
            for label in labels
        }
        (metrics / f"eval_info_t{task_id}-{task_id}.json").write_text(json.dumps(chunks))
    merged, _ = MODULE.run_merge(tmp_path, expected_tasks=2, labels=labels)
    assert all(info["overall"]["pc_success"] == 75.0 for info in merged.values())
    assert (tmp_path / "checkpoint_success_rates.png").stat().st_size > 0
    assert sorted(path.name for path in (tmp_path / "checkpoint_task_success_rates").glob("*.png")) == [
        "00_A.png", "01_B.png",
    ]


@pytest.mark.parametrize("component_kind", ["separate", "joint"])
@pytest.mark.parametrize("model_count", [1, 2])
def test_ft_component_sweeps_automatically_plot_each_models_task_lines(
    tmp_path: Path, monkeypatch, component_kind: str, model_count: int,
) -> None:
    """Exercise FT YAML expansion through the same merge entry point as eval.sbatch."""
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    from stage1_eval_config import _model_entries

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.axes import Axes

    models = []
    for index in range(model_count):
        run = f"run_{index}"
        models.append({
            "label": f"ft_{index}", "vsa_dir": run,
            "vsa_checkpoint": [10000, 2500, 0],
            "predictor_dir": run if component_kind == "joint" else "gt",
            "predictor_checkpoint": "002000" if component_kind == "joint" else "",
            "terminator_dir": run if component_kind == "joint" else "term_run",
            "terminator_checkpoint": "010000",
        })
    entries = _model_entries({"models": models})
    labels = [entry["label"] for entry in entries]
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    for task_id in (0, 1):
        chunks = {}
        for entry in entries:
            count = entry["checkpoint_index"] + entry["model_index"] + task_id
            chunks[entry["label"]] = {"per_task": [{
                "task_group": "libero_10", "task_id": task_id,
                "metrics": {"successes": [trial < count for trial in range(10)]},
            }]}
        (metrics / f"eval_info_t{task_id}-{task_id}.json").write_text(json.dumps(chunks))

    task_lines = []
    original_plot = Axes.plot

    def plot(axis, steps, values, **kwargs):
        if kwargs.get("label", "").startswith("Task "):
            task_lines.append((list(steps), list(values), kwargs["label"]))
        return original_plot(axis, steps, values, **kwargs)

    monkeypatch.setattr(Axes, "plot", plot)
    MODULE.run_merge(tmp_path, expected_tasks=2, labels=labels)

    assert (tmp_path / "checkpoint_success_rates.png").stat().st_size > 0
    paths = sorted((tmp_path / "checkpoint_task_success_rates").glob("*.png"))
    assert [path.name for path in paths] == [f"{index:02d}_ft_{index}.png" for index in range(model_count)]
    assert all(path.stat().st_size > 0 for path in paths)
    assert task_lines == [
        ([0, 2500, 10000], [10.0 * (step + index + task_id) for step in (2, 1, 0)], f"Task {task_id}")
        for index in range(model_count)
        for task_id in (0, 1)
    ]
