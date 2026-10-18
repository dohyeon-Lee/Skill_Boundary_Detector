import json
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_skillVLA/PT/eval/src"
sys.path.insert(0, str(_SRC))
from backfill_skill_html_outcomes import annotate_html, backfill  # noqa: E402


def _html(indices=(0, 1)):
    data = {"task_group": "libero_10", "task_id": 4,
            "episodes": [{"episode_index": index, "skills": [{"token": 8}]} for index in indices]}
    return f'<html><head></head><body><div id="app"></div><script>const DATA = {json.dumps(data)};</script></body></html>'


def test_outcomes_follow_episode_ids_and_preserve_original_payload():
    original = _html((2, 0))
    updated, count = annotate_html(original, [True, True, False], task_group="libero_10", task_id=4)
    data, _ = json.JSONDecoder().raw_decode(updated.split("const DATA = ", 1)[1])
    assert count == 2
    assert [(ep["episode_index"], ep["success"]) for ep in data["episodes"]] == [(2, False), (0, True)]
    assert all(ep["skills"] == [{"token": 8}] for ep in data["episodes"])
    assert 'id="episode-outcome-backfill-script"' in updated
    assert annotate_html(updated, [True, True, False], task_group="libero_10", task_id=4)[0] == updated


@pytest.mark.parametrize("indices,successes", [((2,), [True]), ((0, 0), [True]), ((0,), ["False"])])
def test_invalid_episode_mapping_is_rejected(indices, successes):
    with pytest.raises(ValueError):
        annotate_html(_html(indices), successes, task_group="libero_10", task_id=4)


def _metrics(root, html_path):
    path = root / "metrics/eval_info_merged.json"
    path.parent.mkdir()
    source = json.dumps({"panel": {"per_task": [{"task_group": "libero_10", "task_id": 4,
        "metrics": {"successes": [True, False], "skill_html_paths": [str(html_path)]}}]}})
    path.write_text(source)
    return path, source


def test_dry_run_apply_backup_and_idempotence(tmp_path):
    target = tmp_path / "skill_trace.html"
    original = _html()
    target.write_text(original)
    metrics_path, metrics_source = _metrics(tmp_path, target)
    assert backfill(tmp_path) == {"html_files": 1, "episodes": 2, "changed": 1}
    assert target.read_text() == original
    assert backfill(tmp_path, apply=True)["changed"] == 1
    assert target.with_name(target.name + ".before-outcome-badges.bak").read_text() == original
    assert backfill(tmp_path, apply=True)["changed"] == 0
    assert metrics_path.read_text() == metrics_source


def test_outside_output_is_not_modified(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    outside = tmp_path / "outside.html"
    outside.write_text(_html())
    _metrics(output, outside)
    with pytest.raises(ValueError, match="outside"):
        backfill(output, apply=True)
    assert outside.read_text() == _html()
