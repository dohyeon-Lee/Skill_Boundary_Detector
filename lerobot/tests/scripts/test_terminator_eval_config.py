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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("state_only", "state_only"),
        ("state_only_terminator", "state_only"),
        ("state_rnn", "state_rnn"),
        ("state_rnn_terminator", "state_rnn"),
    ],
)
def test_state_terminator_variants_are_normalized(value: str, expected: str) -> None:
    assert MODULE._normalize_display_variant(value) == expected


@pytest.mark.parametrize(
    ("variant", "train_field"),
    [
        ("state_only", "train_state_only_terminator"),
        ("state_rnn", "train_state_rnn_terminator"),
    ],
)
def test_state_terminator_validation_uses_matching_checkpoint_field(
    tmp_path: Path,
    variant: str,
    train_field: str,
) -> None:
    checkpoint = tmp_path / variant
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "skill_aux",
                train_field: True,
                "skill_fsq_levels": [3, 3, 3],
            }
        )
    )
    (checkpoint / "model.safetensors").touch()

    MODULE._validate_display_model(
        {"variant": variant, "path": str(checkpoint)},
        target_policy={"skill_fsq_levels": [3, 3, 3]},
    )


def test_state_image_validation_rejects_different_skill_code_space(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "state_image"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "skill_aux",
                "train_terminator": True,
                "skill_fsq_levels": [3, 3, 3],
                "skill_code_space_id": "norm_action_space",
            }
        )
    )
    (checkpoint / "model.safetensors").touch()

    with pytest.raises(ValueError, match="skill-code space mismatch"):
        MODULE._validate_display_model(
            {"variant": "state_image", "path": str(checkpoint)},
            target_policy={
                "skill_fsq_levels": [3, 3, 3],
                "fsq_path": str(tmp_path / "zero_space/FSQ.pt"),
            },
        )


@pytest.mark.parametrize("variant", ["state_only", "state_rnn"])
def test_external_main_accepts_state_terminator_variants(
    tmp_path: Path,
    variant: str,
) -> None:
    checkpoint, resolved_variant = MODULE._resolve_external_model(
        {
            "external_skill_model": {
                "variant": variant,
                "group": "skillVLA_terminator",
                "model_dir": "state_models",
                "checkpoint": "030000",
            }
        },
        tmp_path,
        tmp_path / "outputs",
        fsq_path=tmp_path / "FSQ.pt",
    )

    assert resolved_variant == variant
    assert checkpoint == (
        tmp_path
        / "outputs/skillVLA_terminator/state_models/checkpoints/030000/pretrained_model"
    )


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


def test_build_settings_original_uses_selected_policy_fsq_without_external(
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
        # The Stage-1 checkpoint need not contain its own terminator because
        # original loads the co-trained terminator from the source FSQ.pt.
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
                "terminator_source": "original",
            },
            "terminator_models": [
                {"label": "FSQ_INIT", "variant": "fsq_initial"}
            ],
            "episode_exact": True,
            "target_task": "libero_90",
            "task_ids": [0],
            "episodes_per_task": 1,
            "eval_num_gpus": 1,
            "n_action_steps": 5,
            "terminator": {"max_skill_length_scale": 1.5},
            "original_dataset_dir": str(original_dataset),
        }
    )

    spec = json.loads(settings["spec_json"])
    assert spec["advance_mode"] == "original"
    assert spec["fsq_path"] == str(fsq_path)
    assert spec["external_skill_model"] == ""
    assert spec["external_skill_model_variant"] == "checkpoint"
    assert settings["eval_num_gpus"] == 1
    assert settings["eval_max_workers_per_gpu"] == 4
    assert settings["eval_work_unit_count"] == 1
    assert settings["skill_max_length_mode"] == "gt_scale"
    assert settings["skill_max_length_scale"] == pytest.approx(1.5)
    assert settings["inference_skill_max_length"] == 1


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
