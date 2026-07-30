import json
import sys
from pathlib import Path

import pytest


CONFIG_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skills/skill_eval/src"
)
sys.path.insert(0, str(CONFIG_SRC))

from eval_config import build_settings  # noqa: E402


def _artifact(tmp_path: Path) -> tuple[Path, Path]:
    run_name = "demo_state_obs20_std_episodemean_100p_fsq333"
    model_dir = tmp_path / "outputs/FSQ" / run_name
    model_dir.mkdir(parents=True)
    (model_dir / "FSQ_epoch0025.pt").touch()
    (model_dir / "FSQ_epoch0100.pt").touch()

    skillset = (
        tmp_path
        / "dataset/FSQ_dataset/demo_full_full/FSQ_inputs"
        / "seg_demo_full_full_state_obs20_ck100000_std_episodemean_100p"
        / "skillset"
    )
    (skillset / "skills").mkdir(parents=True)
    (tmp_path / "dataset/demo_full_full/videos").mkdir(parents=True)
    (tmp_path / "models/dinov3-vitl16").mkdir(parents=True)
    (model_dir / "fsq_meta.json").write_text(
        json.dumps(
            {
                "fsq_dataset_root": "FSQ_dataset",
                "target_dataset": "demo_full_full",
                "fsq_inputs_name": "FSQ_inputs",
                "skillset_seg_name": (
                    "seg_demo_full_full_state_obs20_ck100000_"
                    "std_episodemean_100p"
                ),
                "skillset_name": "skillset",
            }
        )
    )
    return model_dir, skillset


def _config(tmp_path: Path, run_name: str, checkpoint: str) -> Path:
    path = tmp_path / "fsq_eval_config.yaml"
    path.write_text(
        "\n".join(
            [
                f"project_root: {tmp_path}",
                "dataset_root: dataset",
                "outputs_root: outputs",
                f"fsq_eval_run_name: {run_name}",
                f"fsq_eval_checkpoint: {checkpoint}",
            ]
        )
    )
    return path


def test_minimal_fsq_eval_resolves_metadata_and_explicit_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir, skillset = _artifact(tmp_path)
    monkeypatch.setenv("EVAL_RUN_FSQ", "true")
    monkeypatch.setenv("EVAL_RUN_DP", "false")

    settings = build_settings(_config(tmp_path, model_dir.name, "25"))

    assert settings["fsq_eval_model_path"] == str(model_dir / "FSQ_epoch0025.pt")
    assert settings["fsq_eval_epoch_tag"] == "epoch0025"
    assert settings["fsq_eval_skillset_dir"] == str(skillset)
    assert settings["fsq_eval_dataset_dir"] == str(tmp_path / "dataset/demo_full_full")


def test_fsq_eval_last_selects_highest_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir, _ = _artifact(tmp_path)
    monkeypatch.setenv("EVAL_RUN_FSQ", "true")
    monkeypatch.setenv("EVAL_RUN_DP", "false")

    settings = build_settings(_config(tmp_path, model_dir.name, "last"))

    assert settings["fsq_eval_model_path"] == str(model_dir / "FSQ_epoch0100.pt")
    assert settings["fsq_eval_epoch_tag"] == "epoch0100"


def test_fsq_eval_requires_complete_training_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir, _ = _artifact(tmp_path)
    (model_dir / "fsq_meta.json").write_text(json.dumps({"target_dataset": "demo_full_full"}))
    monkeypatch.setenv("EVAL_RUN_FSQ", "true")
    monkeypatch.setenv("EVAL_RUN_DP", "false")

    with pytest.raises(ValueError, match="fsq_meta.fsq_dataset_root"):
        build_settings(_config(tmp_path, model_dir.name, "last"))
