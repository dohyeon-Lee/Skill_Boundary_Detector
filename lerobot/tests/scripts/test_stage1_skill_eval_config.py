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
    terminator_model = {
        "label": "TERM",
        "model_dir": "shared_term",
        "checkpoint": "020000",
        "end_threshold": 0.9,
    }
    return {
        "project_root": str(tmp_path),
        "outputs_root": "outputs",
        "model_defaults": {"checkpoint": "010000"},
        "models": [
            {
                "model_dir": "policy_a",
                "label": "A",
                "terminator_model": dict(terminator_model),
            },
            {
                "model_dir": "policy_b",
                "label": "B",
                "terminator_model": dict(terminator_model),
            },
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
    assert settings["terminator_model_progress_threshold"] == 0.95


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
    assert settings["eval_max_workers_per_gpu"] == 4
    assert settings["eval_work_unit_count"] == 2


@pytest.mark.parametrize("workers", [0, 5])
def test_eval_workers_per_gpu_is_bounded(
    tmp_path: Path, monkeypatch, workers: int
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
    config["eval_max_workers_per_gpu"] = workers

    with pytest.raises(ValueError, match="between 1 and 4"):
        MODULE.build_settings(config)


def test_same_space_models_can_select_individual_checkpoints(
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
    config["models"][0]["checkpoint"] = "030000"
    config["models"][1]["checkpoint"] = "080000"

    settings = MODULE.build_settings(config)
    models = json.loads(settings["models_json"])

    assert [model["checkpoint"] for model in models] == ["030000", "080000"]
    assert models[0]["policy_path"].endswith(
        "policy_a/checkpoints/030000/pretrained_model"
    )
    assert models[1]["policy_path"].endswith(
        "policy_b/checkpoints/080000/pretrained_model"
    )


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
    assert settings["terminator_model_progress_threshold"] == 0.95


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


def test_policies_with_different_fsq_spaces_are_resolved_independently(
    tmp_path: Path, monkeypatch
) -> None:
    first = _contract(tmp_path / "first", architecture_label="arch_a")
    second = _contract(tmp_path / "second", architecture_label="arch_b")
    second["eval_init_states_path"] = first["eval_init_states_path"]
    contracts = iter([first, second])
    monkeypatch.setattr(MODULE, "_checkpoint_contract", lambda *_args: next(contracts))
    monkeypatch.setattr(MODULE, "_validate_external_terminator", lambda *_args, **_kwargs: None)

    settings = MODULE.build_settings(_config(tmp_path))
    models = json.loads(settings["models_json"])

    assert models[0]["fsq_path"] != models[1]["fsq_path"]
    assert models[0]["skill_latents_path"] != models[1]["skill_latents_path"]
    assert models[0]["fsq_levels"] == models[1]["fsq_levels"] == [3, 3, 3]


def test_each_skill_space_can_override_its_terminator(
    tmp_path: Path, monkeypatch
) -> None:
    first = _contract(tmp_path / "first", architecture_label="arch_a")
    second = _contract(tmp_path / "second", architecture_label="arch_b")
    second["eval_init_states_path"] = first["eval_init_states_path"]
    contracts = iter([first, second])
    monkeypatch.setattr(MODULE, "_checkpoint_contract", lambda *_args: next(contracts))
    monkeypatch.setattr(MODULE, "_validate_external_terminator", lambda *_args, **_kwargs: None)
    config = _config(tmp_path)
    config["models"][1]["terminator_model"] = {
        "label": "TERM_B",
        "model_dir": "term_b",
        "checkpoint": "030000",
    }

    models = json.loads(MODULE.build_settings(config)["models_json"])

    assert models[0]["terminator_models"][0]["label"] == "TERM"
    assert models[1]["terminator_models"][0]["label"] == "TERM_B"
    assert models[1]["terminator_models"][0]["variant"] == "state_image"
    assert "term_b/checkpoints/030000" in models[1]["terminator_models"][0]["path"]


def test_external_mode_requires_a_terminator_source(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        MODULE,
        "_checkpoint_contract",
        lambda *_args: _contract(tmp_path, architecture_label="arch"),
    )
    monkeypatch.setattr(MODULE, "_validate_external_terminator", lambda *_args, **_kwargs: None)
    config = _config(tmp_path)
    config["models"][1].pop("terminator_model")

    with pytest.raises(ValueError, match=r"models\[1\].*external_terminator_model"):
        MODULE.build_settings(config)


def test_stage1_eval_style_external_terminator_uses_shared_defaults(
    tmp_path: Path, monkeypatch
) -> None:
    contracts = iter(
        [
            _contract(tmp_path, architecture_label="arch_a"),
            _contract(tmp_path, architecture_label="arch_b"),
        ]
    )
    validations: list[Path] = []
    monkeypatch.setattr(MODULE, "_checkpoint_contract", lambda *_args: next(contracts))
    monkeypatch.setattr(
        MODULE,
        "_validate_external_terminator",
        lambda path, **_kwargs: validations.append(path),
    )
    config = _config(tmp_path)
    config["model_defaults"].update(
        {
            "advance_mode": "external",
            "terminator_variant": "state_image",
            "external_terminator_checkpoint": "020000",
        }
    )
    for model in config["models"]:
        model.pop("terminator_model")
        model["external_terminator_model"] = "shared_term"
    config["terminator"] = {
        "end_mode": "termination",
        "end_threshold": 0.9,
        "progress_threshold": 0.95,
    }
    config["main_terminator"] = {
        "max_skill_length_scale": 1.5,
        "finish_action_chunk_on_end": False,
    }

    settings = MODULE.build_settings(config)
    models = json.loads(settings["models_json"])
    expected = (
        tmp_path
        / "outputs/skillVLA_terminator/shared_term/checkpoints/020000/pretrained_model"
    )

    assert {model["checkpoint"] for model in models} == {"010000"}
    assert {model["skill_source"] for model in models} == {"gt"}
    assert {model["advance_mode"] for model in models} == {"external"}
    assert {model["external_skill_model"] for model in models} == {str(expected)}
    assert {
        model["terminator_models"][0]["end_threshold"] for model in models
    } == {0.9}
    assert validations == [expected, expected]


def test_original_selector_uses_each_policys_fsq_terminator(
    tmp_path: Path, monkeypatch
) -> None:
    first = _contract(tmp_path / "first", architecture_label="arch_a")
    second = _contract(tmp_path / "second", architecture_label="arch_b")
    second["eval_init_states_path"] = first["eval_init_states_path"]
    contracts = iter([first, second])
    validations: list[Path] = []
    monkeypatch.setattr(MODULE, "_checkpoint_contract", lambda *_args: next(contracts))
    monkeypatch.setattr(
        MODULE,
        "_validate_external_terminator",
        lambda path, **_kwargs: validations.append(path),
    )
    config = _config(tmp_path)
    for model in config["models"]:
        model.pop("terminator_model")
        model["external_terminator_model"] = "original"
    config["terminator"] = {
        "end_mode": "termination",
        "end_threshold": 0.7,
        "progress_threshold": 0.95,
    }
    config["main_terminator"] = {
        "max_skill_length_scale": 1.5,
        "finish_action_chunk_on_end": False,
    }

    settings = MODULE.build_settings(config)
    models = json.loads(settings["models_json"])

    assert {model["advance_mode"] for model in models} == {"original"}
    assert [model["external_skill_model"] for model in models] == [
        str(first["fsq_path"]),
        str(second["fsq_path"]),
    ]
    assert {model["external_skill_model_variant"] for model in models} == {
        "fsq_initial"
    }
    assert {
        model["terminator_models"][0]["label"] for model in models
    } == {"original"}
    assert {
        model["main_terminator"]["end_threshold"] for model in models
    } == {0.7}
    assert validations == []


@pytest.mark.parametrize(
    ("scope", "field"),
    [
        ("defaults", "skill_source"),
        ("defaults", "external_predictor_checkpoint"),
        ("model", "external_predictor_model"),
    ],
)
def test_predictor_fields_are_rejected(
    tmp_path: Path, scope: str, field: str
) -> None:
    config = _config(tmp_path)
    target = config["model_defaults"] if scope == "defaults" else config["models"][0]
    target[field] = "unused"

    with pytest.raises(ValueError, match="always replays GT skill occurrences"):
        MODULE.build_settings(config)


def test_top_level_terminator_default_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["terminator_model"] = config["models"][0].pop("terminator_model")

    with pytest.raises(ValueError, match="Top-level terminator_model was removed"):
        MODULE.build_settings(config)


@pytest.mark.parametrize("obsolete", ["end_mode", "progress_threshold"])
def test_terminator_model_rejects_obsolete_end_options(
    tmp_path: Path, monkeypatch, obsolete: str
) -> None:
    monkeypatch.setattr(
        MODULE,
        "_checkpoint_contract",
        lambda *_args: _contract(tmp_path, architecture_label="arch"),
    )
    config = _config(tmp_path)
    config["models"][0]["terminator_model"][obsolete] = (
        "termination" if obsolete == "end_mode" else 0.95
    )

    with pytest.raises(ValueError, match="termination mode is fixed"):
        MODULE.build_settings(config)


@pytest.mark.parametrize(
    ("fixed_field", "value"),
    [("variant", "state_image"), ("group", "skillVLA_terminator")],
)
def test_terminator_model_rejects_repeated_fixed_options(
    tmp_path: Path, monkeypatch, fixed_field: str, value: str
) -> None:
    monkeypatch.setattr(
        MODULE,
        "_checkpoint_contract",
        lambda *_args: _contract(tmp_path, architecture_label="arch"),
    )
    config = _config(tmp_path)
    config["models"][0]["terminator_model"][fixed_field] = value

    with pytest.raises(ValueError, match="variant=state_image"):
        MODULE.build_settings(config)


def test_different_exact_episode_maps_are_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    contracts = iter(
        [
            _contract(tmp_path / "first", architecture_label="arch_a"),
            _contract(tmp_path / "second", architecture_label="arch_b"),
        ]
    )
    monkeypatch.setattr(MODULE, "_checkpoint_contract", lambda *_args: next(contracts))
    monkeypatch.setattr(MODULE, "_validate_external_terminator", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="same exact source episodes"):
        MODULE.build_settings(_config(tmp_path))
