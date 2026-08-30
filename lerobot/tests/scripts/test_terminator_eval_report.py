from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/terminator_eval/src/html_report.py"
)
SPEC = importlib.util.spec_from_file_location("terminator_eval_html_report", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_four_dimensional_fsq_uses_separate_cube_slices(tmp_path: Path) -> None:
    payload = {
        "levels": [3, 3, 3, 3],
        "model_label": "FSQ3333",
        "target_task": "libero_90",
        "task_ids": [0],
        "selected_episode_count": 1,
        "occurrence_count": 1,
        "time_shift_offset": 15,
        "terminator_models": [],
        "skills": [{"token": 54, "coord": [0, 0, 0, 2], "occurrences": []}],
    }

    html = MODULE.write_html_report(tmp_path, payload).read_text(encoding="utf-8")

    assert 'id="cube-grid"' in html
    assert "extraSliceCoordinates(levels)" in html
    assert "c.slice(3).some" in html
    assert "dim ${index+4} = ${value}" in html
    assert '"levels":[3,3,3,3]' in html
