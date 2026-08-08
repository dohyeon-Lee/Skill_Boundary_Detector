from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/stage1_skill_eval/src/stage1_skill_eval_config.py"
)
SPEC = importlib.util.spec_from_file_location("stage1_skill_eval_config", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _contract(tmp_path: Path, *, architecture_label: str) -> dict:
    dataset = tmp_path / "dataset/run/skillvla"
    latent = tmp_path / "dataset/run/skill_latents.npz"
    init_states = tmp_path / "dataset/eval_init_states.npz"
    fsq = tmp_path / "dataset/run/FSQ.pt"
    dataset.mkdir(parents=True, exist_ok=True)
    for path in (latent, init_states, fsq):
        path.touch(exist_ok=True)
    return {
        "policy": {"chunk_size": 10, "skill_fsq_levels": [3, 3, 3]},
        "skill_dataset_dir": dataset,
        "skill_latents_path": latent,
        "eval_init_states_path": init_states,
        "fsq_path": fsq,
        "dino_model_path": tmp_path / "models/dino",
        "tokenizer_path": tmp_path / "models/tokenizer",
        "architecture_label": architecture_label,
    }


def _config(tmp_path: Path) -> dict:
    original = tmp_path / "libero_original_dataset/libero_90"
    original.mkdir(parents=True)
    return {
        "project_root": str(tmp_path),
        "outputs_root": "outputs",
        "model_defaults": {"checkpoint": "010000"},
        "models": [
            {"model_dir": "policy_a", "label": "A"},
            {"model_dir": "policy_b", "label": "B"},
        ],
        "main_terminator": {
            "label": "FSQ_INIT",
            "variant": "fsq_initial",
            "end_mode": "or",
            "end_threshold": 0.4,
            "progress_threshold": 0.8,
            "max_skill_length": 123,
            "finish_action_chunk_on_end": True,
        },
        "terminator_model": {
            "label": "TERM",
            "variant": "state_image",
            "group": "skillVLA_terminator",
            "model_dir": "shared_term",
            "checkpoint": "020000",
            "end_mode": "termination",
            "end_threshold": 0.9,
            "progress_threshold": 0.7,
        },
        "episode_exact": True,
        "target_task": "libero_90",
        "task_ids": [0],
        "episodes_per_task": 1,
        "eval_num_gpus": 1,
        "n_action_steps": 5,
        "original_dataset_dir": str(original),
    }


def test_multiple_policies_share_one_external_terminator(
    tmp_path: Path, monkeypatch
) -> None:
    contracts = iter(
        [
            _contract(tmp_path, architecture_label="arch_a"),
            _contract(tmp_path, architecture_label="arch_b"),
        ]
    )
    validations: list[tuple[Path, str]] = []
    monkeypatch.setattr(MODULE, "_checkpoint_contract", lambda *_args: next(contracts))
    monkeypatch.setattr(
        MODULE,
        "_validate_external_terminator",
        lambda path, *, target_policy, variant: validations.append((path, variant)),
    )

    settings = MODULE.build_settings(_config(tmp_path))
    models = json.loads(settings["models_json"])

    expected_term = (
        tmp_path
        / "outputs/skillVLA_terminator/shared_term/checkpoints/020000/pretrained_model"
    )
    assert [model["label"] for model in models] == ["A", "B"]
    expected_fsq = str(tmp_path / "dataset/run/FSQ.pt")
    assert {model["external_skill_model"] for model in models} == {expected_fsq}
    assert {model["external_skill_model_variant"] for model in models} == {
        "fsq_initial"
    }
    assert {
        model["terminator_models"][0]["path"] for model in models
    } == {str(expected_term)}
    assert {
        model["terminator_models"][0]["end_threshold"] for model in models
    } == {0.9}
    assert {model["advance_mode"] for model in models} == {"external"}
    assert validations == [(expected_term, "state_image")] * 2
    assert settings["model_count"] == 2
    assert settings["terminator_model_label"] == "TERM"
    assert settings["main_terminator_label"] == "FSQ_INIT"
    assert settings["main_terminator_variant"] == "fsq_initial"
    assert settings["main_terminator_path"] == Path(expected_fsq)
    assert settings["skill_end_threshold"] == 0.4
    assert settings["skill_end_progress_threshold"] == 0.8
    assert settings["inference_skill_max_length"] == 123
    assert settings["terminator_model_end_threshold"] == 0.9
    assert settings["terminator_model_progress_threshold"] == 0.7


def test_gpu_count_is_capped_by_policy_episode_pairs(
    tmp_path: Path, monkeypatch
) -> None:
    contracts = iter(
        [
            _contract(tmp_path, architecture_label="arch_a"),
            _contract(tmp_path, architecture_label="arch_b"),
        ]
    )
    monkeypatch.setattr(MODULE, "_checkpoint_contract", lambda *_args: next(contracts))
    monkeypatch.setattr(
        MODULE,
        "_validate_external_terminator",
        lambda *_args, **_kwargs: None,
    )
    config = _config(tmp_path)
    config["eval_num_gpus"] = 10

    settings = MODULE.build_settings(config)

    assert settings["eval_num_gpus"] == 2


def test_main_terminator_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="fsq_initial.*terminator_model"):
        MODULE._resolve_main_terminator(
            {"main_terminator": {"variant": "checkpoint"}}
        )


def test_main_terminator_can_share_trained_display_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    contracts = iter(
        [
            _contract(tmp_path, architecture_label="arch_a"),
            _contract(tmp_path, architecture_label="arch_b"),
        ]
    )
    monkeypatch.setattr(MODULE, "_checkpoint_contract", lambda *_args: next(contracts))
    monkeypatch.setattr(
        MODULE,
        "_validate_external_terminator",
        lambda *_args, **_kwargs: None,
    )
    config = _config(tmp_path)
    config["main_terminator"] = {
        **config["main_terminator"],
        "label": "TRAINED_MAIN",
        "variant": "terminator_model",
        "end_mode": "and",
        "end_threshold": 0.25,
        "progress_threshold": 0.6,
    }

    settings = MODULE.build_settings(config)
    models = json.loads(settings["models_json"])
    expected_term = (
        tmp_path
        / "outputs/skillVLA_terminator/shared_term/checkpoints/020000/pretrained_model"
    )

    assert {model["external_skill_model"] for model in models} == {
        str(expected_term)
    }
    assert {model["external_skill_model_variant"] for model in models} == {
        "state_image"
    }
    assert settings["main_terminator_label"] == "TRAINED_MAIN"
    assert settings["main_terminator_variant"] == "state_image"
    assert settings["main_terminator_path"] == expected_term
    assert settings["skill_end_mode"] == "and"
    assert settings["skill_end_threshold"] == 0.25
    assert settings["skill_end_progress_threshold"] == 0.6
    # Display-only rules remain independent despite sharing the weights.
    assert settings["terminator_model_end_mode"] == "termination"
    assert settings["terminator_model_end_threshold"] == 0.9
    assert settings["terminator_model_progress_threshold"] == 0.7


def test_short_main_config_fully_inherits_terminator_model(
    tmp_path: Path, monkeypatch
) -> None:
    contracts = iter(
        [
            _contract(tmp_path, architecture_label="arch_a"),
            _contract(tmp_path, architecture_label="arch_b"),
        ]
    )
    monkeypatch.setattr(MODULE, "_checkpoint_contract", lambda *_args: next(contracts))
    monkeypatch.setattr(
        MODULE,
        "_validate_external_terminator",
        lambda *_args, **_kwargs: None,
    )
    config = _config(tmp_path)
    config["main_terminator"] = {
        "max_skill_length_scale": 1.1,
        "finish_action_chunk_on_end": False,
    }

    settings = MODULE.build_settings(config)
    expected_term = (
        tmp_path
        / "outputs/skillVLA_terminator/shared_term/checkpoints/020000/pretrained_model"
    )

    assert settings["main_terminator_label"] == "MAIN_TERM"
    assert settings["main_terminator_variant"] == "state_image"
    assert settings["main_terminator_path"] == expected_term
    assert settings["skill_end_mode"] == settings["terminator_model_end_mode"]
    assert (
        settings["skill_end_threshold"]
        == settings["terminator_model_end_threshold"]
    )
    assert (
        settings["skill_end_progress_threshold"]
        == settings["terminator_model_progress_threshold"]
    )
    assert settings["skill_max_length_mode"] == "gt_scale"
    assert settings["skill_max_length_scale"] == 1.1
    assert settings["finish_action_chunk_on_end"] is False


def test_main_terminator_rejects_fixed_and_scaled_max_length_together() -> None:
    with pytest.raises(ValueError, match="cannot set both"):
        MODULE._resolve_main_terminator(
            {
                "main_terminator": {
                    "max_skill_length": 150,
                    "max_skill_length_scale": 1.1,
                }
            }
        )


def test_trained_main_explicit_variant_must_match_display_variant() -> None:
    with pytest.raises(ValueError, match="must be terminator_model or match"):
        MODULE._resolve_main_terminator(
            {"main_terminator": {"variant": "image_only"}},
            terminator_variant="state_image",
        )


def test_policies_with_different_exact_datasets_are_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    first = _contract(tmp_path / "first", architecture_label="arch_a")
    second = _contract(tmp_path / "second", architecture_label="arch_b")
    contracts = iter([first, second])
    monkeypatch.setattr(MODULE, "_checkpoint_contract", lambda *_args: next(contracts))
    monkeypatch.setattr(MODULE, "_validate_external_terminator", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="same skill_dataset_dir"):
        MODULE.build_settings(_config(tmp_path))
