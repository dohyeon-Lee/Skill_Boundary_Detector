"""GT-skill probe of FSQ co-trained terminators: joining, scoring, reporting."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "examples/libero/configs/train_skills/skill_eval/src"
sys.path.insert(0, str(SRC))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CONFIG = _load("fsq_terminator_eval_config_test", "fsq_terminator_eval_config.py")
REPORT = _load("fsq_terminator_eval_report_test", "fsq_terminator_eval_report.py")


def _record(task, episode, skill, *, token, termination):
    return {
        "task_id": task,
        "episode_id": episode,
        "skill_index": skill,
        "token": token,
        "frame_start": 0,
        "frame_end": len(termination),
        "length": len(termination),
        "gt_end": len(termination) - 1,
        "pred_end": 0,
        "fired": True,
        "timing": 0,
        "termination": termination,
        "progress": [0.0] * len(termination),
    }


def _manifest(label, records, **extra):
    return {
        "completed": True,
        "label": label,
        "run_name": f"run_{label}",
        "epoch_tag": "epoch0100",
        "terminator_kind": "state_image",
        "termination_only": False,
        "fsq_levels": [3, 3, 3],
        "codebook_size": 27,
        "end_threshold": 0.5,
        "summary": {},
        "records": records,
        **extra,
    }


# ── config ────────────────────────────────────────────────────────────────────


def _probe_run(tmp_path: Path, name: str, *, epochs: list[int], original: bool = False) -> None:
    run_dir = tmp_path / "outputs" / "FSQ" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    for epoch in epochs:
        (run_dir / f"FSQ_epoch{epoch:04d}.pt").write_bytes(b"x")
    meta = "fsq_original_meta.json" if original else "fsq_meta.json"
    (run_dir / meta).write_text(
        json.dumps(
            {
                "fsq_dataset_root": "FSQ_dataset",
                "target_dataset": "ds",
                "fsq_inputs_name": "FSQ_inputs",
                "skillset_seg_name": "seg",
                "skillset_name": "skillset",
            }
        )
    )
    dataset_root = tmp_path / "dataset"
    (
        dataset_root / "FSQ_dataset" / "ds" / "FSQ_inputs" / "seg" / "skillset" / "skills"
    ).mkdir(parents=True, exist_ok=True)
    (dataset_root / "ds" / "videos").mkdir(parents=True, exist_ok=True)


def _probe_config(tmp_path: Path, models: list[dict]) -> dict:
    return {
        "project_root": str(tmp_path),
        "dataset_root": "dataset",
        "outputs_root": "outputs",
        "terminator_models": models,
        "target_task": "libero_90",
        "task_ids": [0],
        "episodes_per_task": 1,
        "output_name": "probe",
    }


def test_config_resolves_each_listed_checkpoint(tmp_path: Path) -> None:
    _probe_run(tmp_path, "a", epochs=[100])
    _probe_run(tmp_path, "b", epochs=[200])
    config = _probe_config(
        tmp_path,
        [
            {"label": "A", "run_name": "a", "checkpoint": "100"},
            {"label": "B", "run_name": "b", "checkpoint": "200"},
        ],
    )
    settings = CONFIG.build_settings(config)
    assert settings["fsq_model_labels"] == "A B"
    assert settings["fsq_model_count"] == 2
    # Without an override the first entry is the one this task runs.
    assert settings["fsq_model_label"] == "A"
    assert CONFIG.build_settings(config, model_override="B")["fsq_model_label"] == "B"


def test_config_rejects_fsq_original_runs(tmp_path: Path) -> None:
    """FSQ-original checkpoints have no terminator module to probe at all."""
    _probe_run(tmp_path, "legacy", epochs=[100], original=True)
    config = _probe_config(tmp_path, [{"label": "L", "run_name": "legacy", "checkpoint": "100"}])
    with pytest.raises(ValueError, match="FSQ-original"):
        CONFIG.build_settings(config)


def test_config_rejects_duplicate_labels(tmp_path: Path) -> None:
    _probe_run(tmp_path, "a", epochs=[100, 200])
    config = _probe_config(
        tmp_path,
        [
            {"label": "same", "run_name": "a", "checkpoint": "100"},
            {"label": "same", "run_name": "a", "checkpoint": "200"},
        ],
    )
    with pytest.raises(ValueError, match="unique"):
        CONFIG.build_settings(config)


def test_task_id_space_is_validated_and_defaults_to_the_skillset(tmp_path: Path) -> None:
    """The skillset and the episode-exact tools number tasks differently."""
    _probe_run(tmp_path, "a", epochs=[100])
    config = _probe_config(tmp_path, [{"label": "A", "run_name": "a", "checkpoint": "100"}])
    assert CONFIG.build_settings(config)["task_id_space"] == "dataset"

    config["task_id_space"] = "suite"
    assert CONFIG.build_settings(config)["task_id_space"] == "suite"

    config["task_id_space"] = "libero"
    with pytest.raises(ValueError, match="task_id_space"):
        CONFIG.build_settings(config)


# ── report ────────────────────────────────────────────────────────────────────


def test_align_skills_keeps_only_what_every_model_scored() -> None:
    """Comparing different subsets would silently answer different questions."""
    shared = _record(0, 1, 0, token=3, termination=[0.1, 0.9])
    manifests = {
        "A": _manifest("A", [shared, _record(0, 2, 0, token=4, termination=[0.2, 0.3])]),
        "B": _manifest("B", [_record(0, 1, 0, token=7, termination=[0.4, 0.6])]),
    }
    skills = REPORT.align_skills(manifests, ["A", "B"])
    assert len(skills) == 1
    assert skills[0]["episode_id"] == 1
    # Each model keeps its own code for the same skill.
    assert skills[0]["models"]["A"]["token"] == 3
    assert skills[0]["models"]["B"]["token"] == 7


def test_timing_histogram_clips_into_a_window_and_counts_overflow() -> None:
    records = [
        {"timing": t} for t in (-30, -2, 0, 0, 1, 40)
    ]
    hist = REPORT.timing_histogram(records, limit=3)
    assert hist["bins"][0] == 2
    assert hist["bins"][1] == 1
    assert hist["bins"][-2] == 1
    assert hist["under"] == 1 and hist["over"] == 1


def test_display_selection_groups_by_the_first_model_codebook() -> None:
    skills = [
        {"models": {"A": {"token": 1}, "B": {"token": 9}}},
        {"models": {"A": {"token": 1}, "B": {"token": 2}}},
        {"models": {"A": {"token": 5}, "B": {"token": 9}}},
    ]
    display = REPORT.select_display_skills(
        skills, "A", max_entries=0, max_samples=5, seed=0
    )
    assert sorted(display) == [1, 5]
    assert sorted(display[1]) == [0, 1]
    assert display[5] == [2]


def test_display_selection_caps_entries_and_samples() -> None:
    skills = [{"models": {"A": {"token": t}}} for t in (1, 1, 1, 2)]
    display = REPORT.select_display_skills(
        skills, "A", max_entries=1, max_samples=2, seed=0
    )
    # Entry 1 has the most skills, so the cap keeps it and drops entry 2.
    assert sorted(display) == [1]
    assert len(display[1]) == 2


def test_maybe_build_waits_for_every_model(tmp_path: Path) -> None:
    collection = tmp_path / "probe"
    models = collection / "models"
    (models / "A" / "metrics").mkdir(parents=True)
    (models / "A" / "metrics" / "manifest.json").write_text(
        json.dumps(_manifest("A", [_record(0, 1, 0, token=1, termination=[0.1, 0.9])]))
    )
    options = {
        "max_entries": 0,
        "max_samples": 1,
        "seed": 0,
        "fps": 10,
        "frame_stride": 1,
        "render_video": False,
    }
    assert REPORT.maybe_build(collection, ["A", "B"], **options) is None
    assert not (collection / "index.html").exists()


def test_report_builds_once_every_model_is_present(tmp_path: Path) -> None:
    collection = tmp_path / "probe"
    for label, token in (("A", 1), ("B", 4)):
        directory = collection / "models" / label / "metrics"
        directory.mkdir(parents=True)
        directory.joinpath("manifest.json").write_text(
            json.dumps(
                _manifest(
                    label,
                    [_record(0, 1, 0, token=token, termination=[0.1, 0.4, 0.95])],
                    dataset_dir=str(tmp_path / "ds"),
                    skills_dir=str(tmp_path / "skills"),
                )
            )
        )
    options = {
        "max_entries": 0,
        "max_samples": 2,
        "seed": 0,
        "fps": 10,
        "frame_stride": 1,
        "render_video": False,
    }
    path = REPORT.maybe_build(collection, ["A", "B"], **options)
    assert path is not None and path.is_file()
    page = path.read_text(encoding="utf-8")
    assert "FSQ terminator probe" in page
    # Both models reach the page, each with its own code for the shared skill.
    assert '"token": 1' in page.replace(" ", " ") or '"token":1' in page
    payload = json.loads((collection / "metrics" / "compare.json").read_text())
    assert payload["labels"] == ["A", "B"]
    assert payload["grouping_label"] == "A"
    assert len(payload["skills"]) == 1
