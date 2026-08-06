from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/terminator_eval/src/terminator_eval_config.py"
)
SPEC = importlib.util.spec_from_file_location("terminator_eval_config", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_fsq_initial_entry_needs_only_label_and_variant(tmp_path: Path) -> None:
    models = MODULE._resolve_display_models(
        {
            "terminator_models": [
                {"label": "FSQ_INIT", "variant": "fsq_initial"}
            ]
        },
        tmp_path,
        tmp_path / "outputs",
    )

    assert models == [
        {"label": "FSQ_INIT", "variant": "fsq_initial", "path": ""}
    ]


@pytest.mark.parametrize("field", ["path", "model_dir", "checkpoint"])
def test_fsq_initial_rejects_checkpoint_fields(tmp_path: Path, field: str) -> None:
    with pytest.raises(ValueError, match="accepts no checkpoint path"):
        MODULE._resolve_display_models(
            {
                "terminator_models": [
                    {
                        "label": "FSQ_INIT",
                        "variant": "fsq_initial",
                        field: "unexpected",
                    }
                ]
            },
            tmp_path,
            tmp_path / "outputs",
        )


def test_fsq_initial_validation_requires_raw_fsq_file(tmp_path: Path) -> None:
    model = {
        "label": "FSQ_INIT",
        "variant": "fsq_initial",
        "path": str(tmp_path / "FSQ.pt"),
    }
    with pytest.raises(FileNotFoundError, match="Raw FSQ terminator"):
        MODULE._validate_display_model(model, target_policy={})

    Path(model["path"]).touch()
    MODULE._validate_display_model(model, target_policy={})


def test_build_settings_binds_fsq_initial_to_selected_policy_fsq(
    tmp_path: Path, monkeypatch
) -> None:
    fsq_path = tmp_path / "dataset/run/FSQ.pt"
    fsq_path.parent.mkdir(parents=True)
    fsq_path.touch()
    eval_init_states = tmp_path / "dataset/eval_init_states.npz"
    eval_init_states.touch()
    skill_latents = tmp_path / "dataset/run/skill_latents.npz"
    skill_latents.touch()
    original_dataset = tmp_path / "libero_original_dataset/libero_90"
    original_dataset.mkdir(parents=True)

    contract = {
        "policy": {"chunk_size": 10, "skill_fsq_levels": [3, 3, 3]},
        "has_terminator": False,
        "fsq_path": fsq_path,
        "eval_init_states_path": eval_init_states,
        "skill_latents_path": skill_latents,
        "dino_model_path": tmp_path / "models/dino",
        "tokenizer_path": tmp_path / "models/tokenizer",
        "skill_dataset_dir": tmp_path / "dataset/run/skillvla",
        "architecture_label": "test_arch",
    }
    monkeypatch.setattr(MODULE, "_checkpoint_contract", lambda *_args: contract)

    settings = MODULE.build_settings(
        {
            "project_root": str(tmp_path),
            "outputs_root": "outputs",
            "model": {
                "model_dir": "action_model",
                "checkpoint": "010000",
                "label": "ACTION",
                "terminator_source": "external",
            },
            "external_skill_model": {"variant": "fsq_initial"},
            "terminator_models": [
                {"label": "FSQ_INIT", "variant": "fsq_initial"}
            ],
            "episode_exact": True,
            "target_task": "libero_90",
            "task_ids": [0],
            "episodes_per_task": 1,
            "eval_num_gpus": 1,
            "n_action_steps": 5,
            "original_dataset_dir": str(original_dataset),
        }
    )

    models = json.loads(settings["terminator_models_json"])
    assert models == [
        {
            "label": "FSQ_INIT",
            "variant": "fsq_initial",
            "path": str(fsq_path),
        }
    ]
    spec = json.loads(settings["spec_json"])
    assert spec["advance_mode"] == "external"
    assert spec["external_skill_model"] == str(fsq_path)
    assert spec["external_skill_model_variant"] == "fsq_initial"


def test_external_fsq_initial_rejects_checkpoint_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="accepts no checkpoint fields"):
        MODULE._resolve_external_model(
            {
                "external_skill_model": {
                    "variant": "fsq_initial",
                    "checkpoint": "010000",
                }
            },
            tmp_path,
            tmp_path / "outputs",
            fsq_path=tmp_path / "FSQ.pt",
        )
