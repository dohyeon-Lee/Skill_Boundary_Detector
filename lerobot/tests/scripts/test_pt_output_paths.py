"""PT/FT output renames: new writes and historical checkpoint references (no GPUs)."""

from pathlib import Path
import runpy
import sys

import pytest


CONFIGS = Path(__file__).resolve().parents[2] / "examples/libero/configs"
sys.path.insert(0, str(CONFIGS / "train_skills/src"))
from train_skills_config import resolve_path, resolve_run_checkpoint, stage1_run_dir  # noqa: E402


@pytest.mark.parametrize("root", ["outputs", "outputs_filtered"])
@pytest.mark.parametrize("old_group,new_group", [
    ("skillVLA_stage1", "skillVLA_PT"), ("skillVLA_NewTask_FT", "skillVLA_FT"),
])
@pytest.mark.parametrize("component", ["VSA", "Predictor", "Terminator", "Joint"])
@pytest.mark.parametrize("form", ["relative", "absolute", "other_server"])
def test_saved_old_paths_resolve_after_rename(tmp_path, root, old_group, new_group, component, form):
    suffix = Path(root) / old_group / component / "run/checkpoints/010000/pretrained_model"
    target = tmp_path / root / new_group / component / "run/checkpoints/010000/pretrained_model"
    target.mkdir(parents=True)
    old = {"relative": suffix, "absolute": tmp_path / suffix, "other_server": Path("/other/server") / suffix}[form]
    assert resolve_path(tmp_path, old) == str(target)


@pytest.mark.parametrize("group", ["skillVLA_PT", "skillVLA_stage1", "skillVLA_FT", "skillVLA_NewTask_FT"])
@pytest.mark.parametrize("component", ["VSA", "Predictor", "Terminator"])
def test_run_selectors_and_last_work_in_both_layouts(tmp_path, group, component):
    run = tmp_path / group / component / "run"
    policy = run / "checkpoints/010000/pretrained_model"
    policy.mkdir(parents=True)
    (policy / "config.json").write_text("{}")
    (run / "checkpoints/last").symlink_to("010000")
    assert stage1_run_dir(tmp_path, "run", component) == run
    assert resolve_run_checkpoint(tmp_path, component, {"run": "run", "checkpoint": "last"}) == policy


def test_flat_vsa_layout_moves_without_changing_run_names(tmp_path):
    run = tmp_path / "outputs/skillVLA_PT/old_run"
    run.mkdir(parents=True)
    assert stage1_run_dir(tmp_path / "outputs", "old_run", "VSA") == run
    assert resolve_path(tmp_path, "outputs/skillVLA_stage1/old_run") == str(run)


@pytest.mark.parametrize("old_group,new_group", [
    ("skillVLA_stage1", "skillVLA_PT"), ("skillVLA_NewTask_FT", "skillVLA_FT"),
])
def test_existing_explicit_path_is_not_overridden(tmp_path, old_group, new_group):
    for group in (old_group, new_group):
        (tmp_path / "outputs" / group / "VSA/run").mkdir(parents=True)
    old = tmp_path / "outputs" / old_group / "VSA/run"
    assert resolve_path(tmp_path, old) == str(old)
    assert stage1_run_dir(tmp_path / "outputs", "run", "VSA") == tmp_path / "outputs" / new_group / "VSA/run"


def test_missing_unrelated_path_and_blank_remain_unchanged(tmp_path):
    missing = "dataset/skillVLA_stage1/run"
    assert resolve_path(tmp_path, missing) == str(tmp_path / missing)
    assert resolve_path(tmp_path, "") == ""


@pytest.mark.parametrize("form", ["relative", "absolute", "other_server"])
@pytest.mark.parametrize("old_group,new_group", [
    ("skillVLA_stage1", "skillVLA_PT"), ("skillVLA_NewTask_FT", "skillVLA_FT"),
])
def test_eval_relocates_saved_pt_paths(tmp_path, form, old_group, new_group):
    relocate = runpy.run_path(str(CONFIGS / "train_skillVLA/PT/eval/src/stage1_eval_config.py"))["_relocate_project_path"]
    suffix = Path("outputs_filtered") / old_group / "Predictor/run/checkpoints/100000/pretrained_model"
    target = tmp_path / "outputs_filtered" / new_group / "Predictor/run/checkpoints/100000/pretrained_model"
    target.mkdir(parents=True)
    old = {"relative": suffix, "absolute": tmp_path / suffix, "other_server": Path("/other/server") / suffix}[form]
    assert relocate(tmp_path, old) == target


@pytest.mark.parametrize("group", ["skillVLA_PT", "skillVLA_FT"])
def test_sync_scanner_finds_the_renamed_group(tmp_path, group):
    scan = runpy.run_path(str(CONFIGS / "src/hf_checkpoints.py"))["scan"]
    run = tmp_path / "outputs_filtered" / group / "VSA/run"
    step = run / "checkpoints/010000"
    step.mkdir(parents=True)
    (run / "checkpoints/last").symlink_to("010000")
    found, pulled = scan(tmp_path, ["outputs_filtered"])
    assert found == {f"outputs_filtered/{group}/VSA/run/checkpoints/010000": step}
    assert not pulled
