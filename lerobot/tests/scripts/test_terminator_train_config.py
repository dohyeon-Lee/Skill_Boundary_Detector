from __future__ import annotations

import importlib.util
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/PT/Terminator/src/terminator_train_config.py"
)
SPEC = importlib.util.spec_from_file_location("terminator_train_config", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _config(
    tmp_path: Path,
    *,
    mode: str = "pt",
    terminator: bool = True,
    predictor: bool = False,
    predictor_checkpoint: str = "",
    terminator_checkpoint: str = "",
    dataset_source: str = "source",
) -> dict:
    run = "FSQ345_test"
    dataset_dir = (
        tmp_path / "dataset/skillvla_dataset" / dataset_source / run / "skillvla"
    )
    (dataset_dir / "meta").mkdir(parents=True)
    (dataset_dir / "meta/info.json").write_text(
        json.dumps(
            {
                "skill_fsq_levels": [3, 4, 5],
                "skill_pmax": 20,
                "skill_jitter_early_start_pmax": 20,
                "skill_jitter_late_start_pmax": 20,
                "skill_jitter_early_end_pmax": 20,
                "skill_jitter_late_end_pmax": 20,
                "skill_jitter_distribution": "half_normal",
                "features": {
                    "observation.state": {"shape": [8]},
                    "action": {"shape": [7]},
                },
            }
        )
    )
    (dataset_dir.parent / "FSQ.pt").touch()

    pi_base = tmp_path / "models/pi05_base"
    pi_base.mkdir(parents=True)
    (pi_base / "model.safetensors").touch()
    tokenizer = tmp_path / "models/tokenizer"
    tokenizer.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (tokenizer / name).write_text("{}")

    return {
        "project_root": str(tmp_path),
        "dataset_root": "dataset",
        "outputs_root": "outputs",
        "mode": mode,
        "dataset": {
            "skillvla_root": "skillvla_dataset",
            "source": dataset_source,
            "run": run,
        },
        "run": {"suffix": "test"},
        "warm_start": {
            "pi_base": "models/pi05_base",
            "tokenizer": "models/tokenizer",
            "predictor_checkpoint": predictor_checkpoint,
            "terminator_checkpoint": terminator_checkpoint,
        },
        "fsq_terminator": {
            "termination": terminator,
            "context": "prev_action",
            "cameras": "both",
            "default_arch": "fusion",
            "vision_backbone": "resnet",
            "freeze_vision_encoder": True,
        },
        "skill_predictor": {
            "train": predictor,
            "all_layers": True,
            "lora": {
                "enabled": True,
                "targets": "q,k,v,o",
                "rank": 8,
                "alpha": 16.0,
                "dropout": 0.0,
            },
            "reader": {
                "tokens": 4,
                "depth": 2,
                "heads": 8,
                "deadzone_frac": 0.8,
            },
            "token_access": {"image": True, "language": True},
        },
        "termination_loss": {"target_sigma": 2.0, "positive_weight": 1.0},
        "training": {
            "dataloader": {"batch_size": 2, "workers": 0, "gpus": 1},
            "optimizer": {
                "base_lr": 2.5e-5,
                "terminator_lr_scale": 1.0,
                "predictor_lr_scale": 1.0,
                "predictor_lora_lr_scale": 10.0,
                "grad_clip_norm": 1.0,
            },
            "schedule": {
                "steps": 10,
                "lr_mode": "warmup_constant",
                "warmup_steps": 1,
                "lr_decay_steps": 5,
                "log_every": 1,
                "save_every": 5,
            },
        },
        "logging": {"wandb": {"enable": True, "project": "VLA_auxiliary"}},
    }


def test_predictor_transition_jitter_is_resolved_at_training_time(tmp_path: Path) -> None:
    config = _config(tmp_path, terminator=False, predictor=True)
    config["transition_randomization"] = {
        "early_start_pmax": 15,
        "late_start_pmax": 15,
        "early_end_pmax": 15,
        "late_end_pmax": 15,
        "distribution": "uniform",
    }

    settings = MODULE.build_settings(config)

    assert settings["transition_jitter_pmax"] == 15
    assert settings["transition_jitter_distribution"] == "uniform"
    assert "_jit" not in settings["run_name"]


def _write_auxiliary_checkpoint(
    tmp_path: Path,
    *,
    name: str,
    predictor: bool = True,
    terminator: bool = True,
    code_space_id: str = "FSQ345_test",
    terminator_context: str = "prev_action",
    terminator_cameras: str = "both",
    terminator_arch: str = "fusion",
    terminator_vision_backbone: str = "resnet",
    terminator_freeze_vision_encoder: bool = True,
    training_batch_size: int = 2,
    dataset_source_lineage: list[str] | None = None,
    run_suffix_lineage: list[str] | None = None,
) -> str:
    checkpoint = tmp_path / "checkpoints" / name
    checkpoint.mkdir(parents=True)
    source = {
        "type": "skill_aux",
        "train_skill_predictor": predictor,
        "train_terminator": terminator,
        "skill_fsq_levels": [3, 4, 5],
        "skill_vocab_size": 60,
        "skill_code_space_id": code_space_id,
        "training_batch_size": training_batch_size,
        "dataset_source_lineage": dataset_source_lineage or ["pt_source"],
        "run_suffix_lineage": run_suffix_lineage or [],
        "fsq_path": str(tmp_path / f"dataset/source/{code_space_id}/FSQ.pt"),
        "terminator_context": terminator_context,
        "terminator_cameras": terminator_cameras,
        "terminator_arch": terminator_arch,
        "terminator_vision_backbone": terminator_vision_backbone,
        "terminator_freeze_vision_encoder": terminator_freeze_vision_encoder,
        "terminator_termination_only": True,
        "terminator_end_target_sigma": 1.5,
        "terminator_end_pos_weight": 2.0,
        "skill_predictor_vlm_variant": "gemma_2b",
        "skill_predictor_image_size": 224,
        "skill_predictor_reader_tokens": 4,
        "skill_predictor_reader_depth": 2,
        "skill_predictor_reader_heads": 8,
        "skill_predictor_all_layers": True,
        "skill_predictor_detach_vlm": False,
        "skill_predictor_lora": True,
        "skill_predictor_lora_targets": "q,k,v,o",
        "skill_predictor_lora_rank": 8,
        "skill_predictor_lora_alpha": 16.0,
        "skill_predictor_lora_dropout": 0.0,
        "skill_predictor_deadzone_frac": 0.8,
        "skill_predictor_attend_image": True,
        "skill_predictor_attend_language": True,
        "tokenizer_path": str(tmp_path / "models/tokenizer"),
        "tokenizer_max_length": 200,
    }
    (checkpoint / "config.json").write_text(json.dumps(source))
    (checkpoint / "model.safetensors").touch()
    return str(checkpoint.relative_to(tmp_path))


def _write_relabeled_variant(config: dict, suffix: str = "relabeled_85k") -> Path:
    project = Path(config["project_root"])
    source = config["dataset"]["source"]
    source_run = config["dataset"]["run"]
    source_info = (
        project
        / config["dataset_root"]
        / config["dataset"]["skillvla_root"]
        / source
        / source_run
        / "skillvla/meta/info.json"
    )
    run = source_info.parents[2].parent / f"{source_run}_{suffix}"
    info_path = run / "skillvla/meta/info.json"
    info_path.parent.mkdir(parents=True)
    info = json.loads(source_info.read_text())
    info["skill_code_space_id"] = source_run
    info_path.write_text(json.dumps(info))
    (run / "FSQ.pt").touch()
    (run / "relabel_provenance.json").write_text(
        json.dumps({"source_run": source_run})
    )
    return run


def test_exact_checkpoint_steps_are_resolved(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["training"]["schedule"]["save_steps"] = [0, 1, 3, 10]

    settings = MODULE.build_settings(config)

    assert settings["steps"] == 10
    assert settings["save_steps"] == [0, 1, 3, 10]


def test_checkpoint_archive_is_opt_in_and_does_not_change_training_identity(tmp_path: Path) -> None:
    config = _config(tmp_path)
    original = MODULE.build_settings(config)
    assert original["checkpoint_archive"] == ""
    assert original["keep_checkpoints"] == -1
    config["training"].update(checkpoint_archive="scratch2", keep_checkpoints=2)
    settings = MODULE.build_settings(config)
    assert settings["checkpoint_archive"] == "scratch2"
    assert settings["keep_checkpoints"] == 2
    for key in ("run_name", "output_dir", "lr", "batch_size", "steps"):
        assert settings[key] == original[key]


@pytest.mark.parametrize("keep,destination", [(0, "scratch2"), (-2, "scratch2"),
                                               (True, "scratch2"), (2, ""), (2, "elsewhere")])
def test_checkpoint_archive_rejects_unsafe_retention(tmp_path: Path, keep, destination) -> None:
    config = _config(tmp_path)
    config["training"].update(checkpoint_archive=destination, keep_checkpoints=keep)
    with pytest.raises(ValueError, match="checkpoint"):
        MODULE.build_settings(config)


@pytest.mark.parametrize("save_steps", [[1, 1], [3, 2], [-1, 10], [0, 11]])
def test_exact_checkpoint_steps_are_validated(tmp_path: Path, save_steps: list[int]) -> None:
    config = _config(tmp_path)
    config["training"]["schedule"]["save_steps"] = save_steps

    with pytest.raises(ValueError, match="save_steps"):
        MODULE.build_settings(config)


@pytest.mark.parametrize(
    ("terminator", "predictor", "training_mode"),
    [
        (True, False, "terminator"),
        (False, True, "predictor"),
    ],
)
def test_pt_target_combinations(tmp_path, terminator, predictor, training_mode):
    config = _config(tmp_path, terminator=terminator, predictor=predictor)
    config["run"]["common_suffix"] = "shared"
    settings = MODULE.build_settings(config)

    assert settings["initialization_mode"] == "pt"
    named_mode = (
        training_mode.replace("terminator", "terminator_prev_both")
        if terminator
        else training_mode
    )
    assert settings["training_mode"] == named_mode
    assert settings["train_terminator"] is terminator
    assert settings["train_skill_predictor"] is predictor
    assert settings["wandb_project"] == "VLA_auxiliary"
    assert settings["output_dir"].parent.name == "skillVLA_terminator"
    expected_suffix = "_test_shared" if predictor else "_test"
    assert settings["run_name"] == f"bs2_FSQ345_test_source_{named_mode}{expected_suffix}"
    assert ("shared" in settings["run_suffix_lineage"]) is predictor


def test_pt_rejects_predictor_and_terminator_in_one_job(tmp_path):
    with pytest.raises(ValueError, match="separate jobs"):
        MODULE.build_settings(
            _config(tmp_path, terminator=True, predictor=True)
        )


@pytest.mark.parametrize(
    ("architecture", "distribution"),
    [
        ("term14", "half_normal"),
        ("term15", "uniform"),
    ],
)
def test_registered_token_terminator_contracts_are_preserved(
    tmp_path: Path, architecture: str, distribution: str
) -> None:
    config = _config(tmp_path)
    config["fsq_terminator"] = {"architecture": architecture}

    settings = MODULE.build_settings(config)

    assert "proprio_conditioning" not in MODULE.TERMINATOR_ARCHITECTURES[architecture]
    assert "proprio_noise" not in MODULE.TERMINATOR_ARCHITECTURES[architecture]
    assert settings["training_mode"] == f"terminator_{architecture}"
    assert settings["terminator_architecture_label"] == architecture
    assert settings["terminator_start_proprio"] is True
    assert settings["terminator_proprio_conditioning"] == "tokens"
    assert settings["terminator_proprio_noise_magnitude"] == 0.0
    assert settings["terminator_start_randomization"] is True
    assert settings["terminator_start_randomization_early_frames"] == 15
    assert settings["terminator_start_randomization_late_frames"] == 10
    assert settings["terminator_start_randomization_distribution"] == distribution
    assert settings["terminator_start_randomization_probability"] == pytest.approx(1.0)
    assert settings["terminator_start_randomization_shift_current_observation"] is True
    assert settings["terminator_cameras"] == "top"
    assert settings["terminator_agent_patch_align_weight"] == 0.0
    assert settings["terminator_wrist_patch_align_weight"] == 0.0


@pytest.mark.parametrize(
    ("architecture", "conditioning", "distribution"),
    [
        ("term18_norm", "tokens", "half_normal"),
        ("term18_uni", "tokens", "uniform"),
        ("term19_norm", "adarms", "half_normal"),
        ("term19_uni", "adarms", "uniform"),
        ("term20_norm", "tokens_delta_adarms", "half_normal"),
        ("term20_uni", "tokens_delta_adarms", "uniform"),
    ],
)
def test_registered_proprio_noise_terminator_variants(
    tmp_path: Path,
    architecture: str,
    conditioning: str,
    distribution: str,
) -> None:
    config = _config(tmp_path)
    config["fsq_terminator"] = {"architecture": architecture}

    settings = MODULE.build_settings(config)

    assert settings["training_mode"] == f"terminator_{architecture}"
    assert settings["terminator_start_proprio"] is True
    assert settings["terminator_proprio_conditioning"] == conditioning
    assert settings["terminator_proprio_noise_magnitude"] == pytest.approx(0.02)
    assert settings["terminator_proprio_noise_distribution"] == "uniform"
    assert settings["terminator_proprio_noise_exclude_last_n"] == 2
    assert settings["terminator_proprio_noise_clamp"] is True
    assert settings["terminator_start_randomization_distribution"] == distribution
    assert settings["terminator_agent_patch_align_weight"] == 0.0
    assert settings["terminator_wrist_patch_align_weight"] == 0.0


@pytest.mark.parametrize(
    ("architecture", "conditioning", "distribution"),
    [
        ("term16_norm", "adarms", "half_normal"),
        ("term16_uni", "adarms", "uniform"),
        ("term17_norm", "tokens_delta_adarms", "half_normal"),
        ("term17_uni", "tokens_delta_adarms", "uniform"),
    ],
)
def test_registered_adarms_terminator_distribution_variants(
    tmp_path: Path,
    architecture: str,
    conditioning: str,
    distribution: str,
) -> None:
    config = _config(tmp_path)
    config["fsq_terminator"] = {"architecture": architecture}

    settings = MODULE.build_settings(config)

    assert "proprio_noise" not in MODULE.TERMINATOR_ARCHITECTURES[architecture]
    assert settings["training_mode"] == f"terminator_{architecture}"
    assert settings["terminator_architecture_label"] == architecture
    assert settings["terminator_start_proprio"] is True
    assert settings["terminator_proprio_conditioning"] == conditioning
    assert settings["terminator_proprio_noise_magnitude"] == 0.0
    assert settings["terminator_start_randomization"] is True
    assert settings["terminator_start_randomization_early_frames"] == 15
    assert settings["terminator_start_randomization_late_frames"] == 10
    assert settings["terminator_start_randomization_distribution"] == distribution
    assert settings["terminator_start_randomization_shift_current_observation"] is True
    assert settings["terminator_cameras"] == "top"


@pytest.mark.parametrize(
    ("architecture", "conditioning", "skill_skip"),
    [
        ("term21", "tokens_delta_adarms", False),
        ("term22", "tokens", False),
        ("term22_skip", "tokens", True),
    ],
)
def test_registered_mild_mixed_start_randomization_variants(
    tmp_path: Path, architecture: str, conditioning: str, skill_skip: bool
) -> None:
    config = _config(tmp_path)
    config["fsq_terminator"] = {"architecture": architecture}
    config["termination_loss"]["positive_weight"] = 2.0

    settings = MODULE.build_settings(config)

    assert settings["training_mode"] == f"terminator_{architecture}"
    assert settings["terminator_architecture_label"] == architecture
    assert settings["terminator_skill_skip"] is skill_skip
    assert settings["terminator_start_proprio"] is True
    assert settings["terminator_proprio_conditioning"] == conditioning
    assert settings["terminator_proprio_noise_magnitude"] == 0.0
    assert settings["terminator_start_randomization"] is True
    assert settings["terminator_start_randomization_early_frames"] == 5
    assert settings["terminator_start_randomization_late_frames"] == 5
    assert settings["terminator_start_randomization_distribution"] == "half_normal"
    assert settings["terminator_start_randomization_probability"] == pytest.approx(0.5)
    assert settings[
        "terminator_start_randomization_shift_current_observation"
    ] is False
    assert settings["terminator_end_pos_weight"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("architecture", "start_proprio", "start_randomization"),
    [
        ("term23", True, True),
        ("term24", False, False),
    ],
)
def test_registered_current_proprio_noise_variants(
    tmp_path: Path,
    architecture: str,
    start_proprio: bool,
    start_randomization: bool,
) -> None:
    config = _config(tmp_path)
    config["fsq_terminator"] = {"architecture": architecture}

    settings = MODULE.build_settings(config)

    assert settings["training_mode"] == f"terminator_{architecture}"
    assert settings["terminator_architecture_label"] == architecture
    assert settings["terminator_start_proprio"] is start_proprio
    assert settings["terminator_proprio_conditioning"] == "tokens"
    assert settings["terminator_proprio_noise_magnitude"] == pytest.approx(0.02)
    assert settings["terminator_proprio_noise_distribution"] == "uniform"
    assert settings["terminator_proprio_noise_exclude_last_n"] == 2
    assert settings["terminator_proprio_noise_clamp"] is True
    assert settings["terminator_start_randomization"] is start_randomization
    if architecture == "term23":
        assert settings["terminator_start_randomization_early_frames"] == 5
        assert settings["terminator_start_randomization_late_frames"] == 5
        assert settings["terminator_start_randomization_distribution"] == "half_normal"
        assert settings["terminator_start_randomization_probability"] == pytest.approx(0.5)
        assert settings[
            "terminator_start_randomization_shift_current_observation"
        ] is False


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
def test_start_randomization_probability_is_validated(
    tmp_path: Path, probability: float
) -> None:
    config = _config(tmp_path)
    config["fsq_terminator"] = {
        "termination": True,
        "context": "proprio",
        "cameras": "top",
        "default_arch": "fusion",
        "start_proprio": True,
        "start_randomization": {"enabled": True, "probability": probability},
    }

    with pytest.raises(ValueError, match="probability"):
        MODULE.build_settings(config)


@pytest.mark.parametrize(
    "base_architecture",
    [
        "term14",
        "term15",
        "term16_norm",
        "term16_uni",
        "term17_norm",
        "term17_uni",
        "term18_norm",
        "term18_uni",
        "term19_norm",
        "term19_uni",
        "term20_norm",
        "term20_uni",
    ],
)
def test_registered_new_variants_randomize_only_the_start_anchor(
    tmp_path: Path, base_architecture: str
) -> None:
    architecture = f"{base_architecture}_new"
    expected_profile = copy.deepcopy(
        MODULE.TERMINATOR_ARCHITECTURES[base_architecture]
    )
    expected_profile["start_randomization"]["shift_current_observation"] = False

    assert MODULE.TERMINATOR_ARCHITECTURES[architecture] == expected_profile
    assert MODULE.TERMINATOR_ARCHITECTURES[base_architecture][
        "start_randomization"
    ]["shift_current_observation"] is True

    config = _config(tmp_path)
    config["fsq_terminator"] = {"architecture": architecture}
    settings = MODULE.build_settings(config)

    assert settings["training_mode"] == f"terminator_{architecture}"
    assert settings["terminator_architecture_label"] == architecture
    assert settings["terminator_start_randomization"] is True
    assert settings[
        "terminator_start_randomization_shift_current_observation"
    ] is False


def test_registered_architecture_can_be_selected_by_cli_override(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["fsq_terminator"] = {"architecture": "term14"}

    settings = MODULE.build_settings(config, architecture_override="term8")

    assert settings["training_mode"] == "terminator_term8"
    assert settings["terminator_architecture_label"] == "term8"
    assert settings["terminator_start_proprio"] is False
    assert settings["terminator_start_randomization"] is False


@pytest.mark.parametrize("variant,context,history", [
    ("Newterm1", "none", False), ("Newterm2", "proprio", False),
    ("Newterm3", "proprio", True),
    ("Newterm4", "proprio", False),
    ("Newterm5", "proprio", False), ("Newterm6", "proprio", False),
    ("Newterm7", "proprio", False),
    ("Newterm8", "proprio", False), ("Newterm9", "proprio", False),
])
def test_newterm_matches_notion_contract_and_supervision(tmp_path, variant, context, history):
    config = _config(tmp_path)
    config["fsq_terminator"] = {"architecture": variant}
    config["termination_loss"] = {"target_sigma": 1.0, "positive_weight": 2.0}
    if variant in {"Newterm5", "Newterm6", "Newterm7", "Newterm9"}:
        info_path = tmp_path / "dataset/skillvla_dataset/source/FSQ345_test/skillvla/meta/info.json"
        info = json.loads(info_path.read_text())
        info["skill_focus_uv_path"] = str(info_path.parent.parent.parent / "skill_focus_uv.npz")
        info_path.write_text(json.dumps(info))
    settings = MODULE.build_settings(config)
    assert settings["training_mode"] == f"terminator_{variant}"
    assert settings["terminator_cameras"] == "top"
    assert settings["terminator_context"] == context
    assert settings["terminator_skill_skip"] is True
    assert settings["terminator_proprio_history"] is history
    assert settings["terminator_start_proprio"] is (variant in {"Newterm4", "Newterm6", "Newterm7", "Newterm8"})
    assert settings["terminator_current_proprio"] is (variant not in {"Newterm7", "Newterm8", "Newterm9"})
    assert settings["terminator_history_length"] == 20
    assert settings["terminator_freeze_vision_encoder"] is True
    assert settings["terminator_vision_backbone"] == "dino"
    assert settings["terminator_post_end_steps"] == 10
    assert settings["terminator_end_target_sigma"] == 1.0
    assert settings["terminator_end_pos_weight"] == 2.0
    assert not settings["terminator_start_randomization"]
    assert settings["terminator_goal_xyz"] is (variant in {"Newterm5", "Newterm6", "Newterm7", "Newterm9"})
    assert settings["terminator_goal_noise_max_m"] == (0.01 if variant in {"Newterm5", "Newterm6", "Newterm7", "Newterm9"} else 0.0)
    assert f"terminator_{variant}" in settings["run_name"]
    if variant in {"Newterm8", "Newterm9"}:
        # FT restores the saved component contract rather than the current YAML.
        restored = MODULE._checkpoint_terminator_contract(settings, tmp_path)
        for field in (
            "terminator_current_proprio", "terminator_start_proprio",
            "terminator_goal_xyz", "terminator_goal_noise_max_m",
            "terminator_skill_skip", "terminator_post_end_steps",
        ):
            assert restored[field] == settings[field]
        overridden = MODULE.build_settings(config, architecture_override=variant.lower())
        assert overridden["run_name"] == settings["run_name"]
    config["termination_loss"]["post_end_steps"] = 0
    assert MODULE.build_settings(config)["terminator_post_end_steps"] == 0


def test_omitting_current_proprio_requires_a_start_or_goal_anchor(tmp_path):
    config = _config(tmp_path)
    config["fsq_terminator"] = copy.deepcopy(MODULE.TERMINATOR_ARCHITECTURES["newterm8"])
    config["fsq_terminator"]["start_proprio"] = False
    with pytest.raises(ValueError, match="Omitting current proprio"):
        MODULE.build_settings(config)


def test_dataloader_pipeline_settings_are_forwarded(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["training"]["dataloader"].update(
        {
            "workers": 24,
            "prefetch_factor": 2,
            "pin_memory": True,
            "persistent_workers": True,
        }
    )

    settings = MODULE.build_settings(config)

    assert settings["num_workers"] == 24
    assert settings["dataloader_prefetch_factor"] == 2
    assert settings["dataloader_pin_memory"] is True
    assert settings["dataloader_persistent_workers"] is True


def test_newterm3_masked_is_distinct_from_legacy_history(tmp_path):
    config = _config(tmp_path)
    config["termination_loss"] = {}
    config["fsq_terminator"] = {"architecture": "Newterm3_masked"}
    settings = MODULE.build_settings(config)
    assert settings["terminator_history_mask_padding"] is True
    assert settings["terminator_history_reset_on_skill"] is True
    assert settings["terminator_history_length"] == 20
    assert settings["terminator_post_end_steps"] == 10
    assert settings["terminator_end_target_sigma"] == 1.0
    assert settings["terminator_end_pos_weight"] == 2.0
    assert "terminator_Newterm3_masked" in settings["run_name"]
    config["fsq_terminator"] = {"architecture": "Newterm3"}
    original = MODULE.build_settings(config)
    assert original["terminator_history_mask_padding"] is False
    assert original["terminator_history_reset_on_skill"] is False
    assert original["run_name"] != settings["run_name"]


def test_persistent_workers_requires_worker_processes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["training"]["dataloader"].update(
        {"workers": 0, "persistent_workers": True}
    )

    with pytest.raises(ValueError, match="persistent_workers requires workers > 0"):
        MODULE.build_settings(config)


def test_progress_and_transformer_history_are_independent_switches(tmp_path):
    config = _config(tmp_path)
    config["fsq_terminator"].update(
        {
            "progress": True,
            "context": "proprio",
            "cameras": "top",
            "proprio_history": {
                "enabled": True,
                "length": 12,
                "dim": 64,
                "layers": 2,
                "heads": 4,
            },
        }
    )

    settings = MODULE.build_settings(config)

    assert settings["terminator_termination_only"] is False
    assert settings["terminator_proprio_history"] is True
    assert settings["terminator_history_length"] == 12
    assert settings["training_mode"] == "terminator_prop_top_progress_hist12"

    config["fsq_terminator"]["progress"] = False
    config["fsq_terminator"]["proprio_history"]["enabled"] = False
    settings = MODULE.build_settings(config)
    assert settings["terminator_termination_only"] is True
    assert settings["terminator_proprio_history"] is False
    assert settings["training_mode"] == "terminator_prop_top"


def test_progress_can_train_a_detached_readout(tmp_path):
    config = _config(tmp_path)
    config["fsq_terminator"].update(
        {
            "progress": {"enabled": True, "detach_backbone": True},
            "context": "proprio",
            "cameras": "top",
        }
    )

    settings = MODULE.build_settings(config)

    assert settings["terminator_termination_only"] is False
    assert settings["terminator_progress_detach_backbone"] is True
    assert settings["training_mode"] == "terminator_prop_top_progress_detached"


def test_progress_detach_requires_progress_training(tmp_path):
    config = _config(tmp_path)
    config["fsq_terminator"]["progress"] = {
        "enabled": False,
        "detach_backbone": True,
    }

    with pytest.raises(ValueError, match="requires progress.enabled=true"):
        MODULE.build_settings(config)


def test_train_launcher_forwards_progress_detach_backbone():
    launcher = SCRIPT.with_name("train.sbatch").read_text()
    assert (
        '--policy.terminator_progress_detach_backbone='
        '"${TERMINATOR_PROGRESS_DETACH_BACKBONE}"'
    ) in launcher


def test_goal_xyz_terminator_contract_and_run_name(tmp_path):
    config = _config(tmp_path, terminator=True, predictor=False)
    config["fsq_terminator"].update(
        {
            "context": "proprio",
            "goal_xyz": True,
            "goal_noise_max_m": 0.01,
            "skill_skip": False,
            "agent_patch_align_weight": 0.1,
            "wrist_patch_align_weight": 0.1,
            "patch_align_target_sigma": 0.7,
        }
    )
    info_path = (
        tmp_path / "dataset/skillvla_dataset/source/FSQ345_test/skillvla/meta/info.json"
    )
    info = json.loads(info_path.read_text())
    info["skill_focus_uv_path"] = str(info_path.parents[2] / "skill_focus_uv.npz")
    info_path.write_text(json.dumps(info))

    settings = MODULE.build_settings(config)

    assert settings["terminator_goal_xyz"] is True
    assert settings["terminator_goal_noise_max_m"] == 0.01
    assert settings["terminator_skill_skip"] is False
    assert settings["training_mode"] == "terminator_prop_both_goalxyz_noskip"


@pytest.mark.parametrize(
    ("mode", "goal_xyz", "expected_name"),
    [
        (
            "goal_token",
            True,
            "terminator_prop_top_goalxyz_progress_detached_chunkendpose_noskip",
        ),
        (
            "learned_token",
            False,
            "terminator_prop_top_progress_detached_chunkendpose_learned_noskip",
        ),
    ],
)
def test_lit_style_terminator_chunk_end_contract(
    tmp_path, mode, goal_xyz, expected_name
):
    config = _config(tmp_path, terminator=True, predictor=False)
    config["fsq_terminator"].update(
        {
            "context": "proprio",
            "cameras": "top",
            "vision_backbone": "dino",
            "progress": {"enabled": True, "detach_backbone": True},
            "goal_xyz": goal_xyz,
            "goal_noise_max_m": 0.01 if goal_xyz else 0.0,
            "skill_skip": False,
            "chunk_end_pose": {
                "mode": mode,
                "loss_weight": 0.3,
                "horizon": 10,
            },
        }
    )
    meta = (
        tmp_path / "dataset/skillvla_dataset/source/FSQ345_test/skillvla/meta"
    )
    (meta / "stats.json").write_text(
        json.dumps(
            {
                "observation.state": {
                    "q01": [0.0] * 8,
                    "q99": [1.0] * 8,
                }
            }
        )
    )
    if goal_xyz:
        info_path = meta / "info.json"
        info = json.loads(info_path.read_text())
        info["skill_focus_uv_path"] = str(meta.parent.parent / "skill_focus_uv.npz")
        info_path.write_text(json.dumps(info))

    settings = MODULE.build_settings(config)

    assert settings["training_mode"] == expected_name
    assert settings["terminator_chunk_end_pose_mode"] == mode
    assert settings["terminator_chunk_end_state_horizon"] == 10
    assert settings["terminator_chunk_end_state_q01"] == "[0,0,0,0,0,0,0,0]"
    assert settings["terminator_chunk_end_state_q99"] == "[1,1,1,1,1,1,1,1]"


@pytest.mark.parametrize(
    ("target", "name_suffix", "end_mode"),
    [("uv", "_uv", "off"), ("xyz", "_xyz", "xyz"), ("full_state", "_state", "full_state")],
)
def test_predictor_spatial_target_probe_uses_existing_dataset_state(
    tmp_path, target, name_suffix, end_mode
):
    config = _config(tmp_path, terminator=False, predictor=True)
    config["skill_predictor"]["spatial_target"] = target
    config["skill_predictor"]["spatial_loss_weight"] = 1.0
    info_path = (
        tmp_path / "dataset/skillvla_dataset/source/FSQ345_test/skillvla/meta/info.json"
    )
    info = json.loads(info_path.read_text())
    info["skill_focus_uv_path"] = str(info_path.parents[2] / "skill_focus_uv.npz")
    info_path.write_text(json.dumps(info))

    settings = MODULE.build_settings(config)

    assert settings["training_mode"] == f"predictor{name_suffix}"
    assert settings["skill_predictor_focus_uv_enabled"] is (target == "uv")
    assert settings["skill_predictor_end_state_mode"] == end_mode
    assert settings["skill_predictor_end_state_dim"] == 8
    assert settings["skill_predictor_end_state_loss_weight"] == 1.0


def test_new_predictor_mode2_is_parallel_xyz_with_occurrence_sampling(tmp_path):
    config = _config(tmp_path, terminator=False, predictor=True)
    del config["fsq_terminator"]
    config["skill_predictor"].update(mode="mode2", spatial_target="xyz")
    config["skill_predictor"]["xyz_skill_condition"] = {"mode": "scheduled"}
    info_path = tmp_path / "dataset/skillvla_dataset/source/FSQ345_test/skillvla/meta/info.json"
    info = json.loads(info_path.read_text())
    info["skill_focus_uv_path"] = str(info_path.parents[2] / "skill_focus_uv.npz")
    info_path.write_text(json.dumps(info))

    settings = MODULE.build_settings(config)

    assert settings["train_terminator"] is False
    assert settings["skill_predictor_mode"] == "mode2"
    assert settings["skill_predictor_sampling_mode"] == "mode1"
    assert settings["skill_predictor_boundary_fraction"] == 0.7
    assert settings["skill_predictor_boundary_window"] == 10
    assert settings["skill_predictor_end_state_skill_source"] == "gt"
    assert settings["training_mode"] == "predictor_xyz_mode2"


def test_predictor_xyz_scheduled_skill_conditioning(tmp_path):
    config = _config(tmp_path, terminator=False, predictor=True)
    config["skill_predictor"]["spatial_target"] = "xyz"
    config["skill_predictor"]["xyz_skill_condition"] = {
        "mode": "scheduled",
        "schedule_start_step": 10,
        "schedule_end_step": 80,
        "max_predicted_probability": 0.75,
    }
    info_path = (
        tmp_path / "dataset/skillvla_dataset/source/FSQ345_test/skillvla/meta/info.json"
    )
    info = json.loads(info_path.read_text())
    info["skill_focus_uv_path"] = str(info_path.parents[2] / "skill_focus_uv.npz")
    info_path.write_text(json.dumps(info))

    settings = MODULE.build_settings(config)

    assert settings["skill_predictor_end_state_skill_source"] == "scheduled"
    assert settings["skill_predictor_end_state_schedule_start_step"] == 10
    assert settings["skill_predictor_end_state_schedule_end_step"] == 80
    assert settings["skill_predictor_end_state_schedule_max_probability"] == 0.75
    assert settings["training_mode"] == "predictor_xyz_schedskill"


@pytest.mark.parametrize("mode", [
    "mode1_preskill", "mode2_preskill", "mode3_preskill",
    "mode3_preskill_wristX", "mode3_preskill_wristX_proprioX",
    "mode3_preskill_proprioX",
])
@pytest.mark.parametrize("freeze_vlm", [False, True])
def test_preskill_predictor_modes_have_distinct_names_and_unchanged_sampling(tmp_path, mode, freeze_vlm):
    config = _config(tmp_path, terminator=False, predictor=True)
    config["skill_predictor"].update(mode=mode, spatial_target="xyz", freeze_vlm=freeze_vlm)
    config["skill_predictor"]["lora"] = {"enabled": False}
    info_path = tmp_path / "dataset/skillvla_dataset/source/FSQ345_test/skillvla/meta/info.json"
    info = json.loads(info_path.read_text())
    info["skill_focus_uv_path"] = str(info_path.parents[2] / "skill_focus_uv.npz")
    info_path.write_text(json.dumps(info))
    settings = MODULE.build_settings(config)
    assert settings["skill_predictor_mode"] == mode
    assert settings["skill_predictor_sampling_mode"] == "mode1"
    expected = "predictor_xyz" + ("" if freeze_vlm else "_fullvlm") + f"_{mode}"
    assert settings["training_mode"] == expected
    assert expected in settings["run_name"]


@pytest.mark.parametrize(
    ("field", "value"),
    [("mode", "other"), ("boundary_fraction", 1.5), ("boundary_window", 0)],
)
def test_predictor_mode2_rejects_invalid_sampling(tmp_path, field, value):
    config = _config(tmp_path, terminator=False, predictor=True)
    config["skill_predictor"]["sampling"] = {"mode": "mode1", field: value}
    with pytest.raises(ValueError, match="skill_predictor.sampling"):
        MODULE.build_settings(config)


def test_old_predictor_sampling_mode2_is_not_reinterpreted_as_new_architecture(tmp_path):
    config = _config(tmp_path, terminator=False, predictor=True)
    config["skill_predictor"]["sampling"] = {"mode": "mode2"}
    with pytest.raises(ValueError, match="removed"):
        MODULE.build_settings(config)


@pytest.mark.parametrize(
    "saved_contract",
    [{"skill_predictor_sampling_mode": "mode2"}, {"skill_predictor_mode": "mode1"}],
)
def test_predictor_mode2_cannot_resume_an_incompatible_run(tmp_path, saved_contract):
    config = _config(tmp_path, terminator=False, predictor=True)
    config["skill_predictor"].update(mode="mode2", spatial_target="xyz")
    info_path = tmp_path / "dataset/skillvla_dataset/source/FSQ345_test/skillvla/meta/info.json"
    info = json.loads(info_path.read_text())
    info["skill_focus_uv_path"] = str(info_path.parents[2] / "skill_focus_uv.npz")
    info_path.write_text(json.dumps(info))
    settings = MODULE.build_settings(config)
    saved = settings["output_dir"] / "checkpoints/last/pretrained_model/config.json"
    saved.parent.mkdir(parents=True)
    saved.write_text(json.dumps(saved_contract))
    with pytest.raises(ValueError, match="Cannot resume a different Predictor"):
        MODULE.build_settings(config)
    saved.write_text(json.dumps({"skill_predictor_mode": "mode2"}))
    assert MODULE.build_settings(config)["output_dir"] == settings["output_dir"]


def test_all_targets_false_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Enable fsq_terminator"):
        MODULE.build_settings(
            _config(tmp_path, terminator=False, predictor=False)
        )


@pytest.mark.parametrize(
    "legacy_key",
    [
        "terminator",
        "image_only_terminator",
        "wrist_only_terminator",
        "state_only_terminator",
        "state_rnn_terminator",
    ],
)
def test_legacy_terminator_sections_are_rejected(tmp_path, legacy_key):
    config = _config(tmp_path)
    config[legacy_key] = {"train": True}
    with pytest.raises(ValueError, match="Legacy terminator sections were removed"):
        MODULE.build_settings(config)


def test_fsq_terminator_contract_is_exported(tmp_path):
    settings = MODULE.build_settings(_config(tmp_path))

    assert settings["terminator_context"] == "prev_action"
    assert settings["terminator_cameras"] == "both"
    assert settings["terminator_arch"] == "fusion"
    assert settings["terminator_vision_backbone"] == "resnet"
    assert settings["terminator_freeze_vision_encoder"] is True
    assert settings["terminator_termination_only"] is True


def test_context_free_single_camera_contract_has_distinct_name(tmp_path):
    config = _config(tmp_path, terminator=True, predictor=False)
    config["fsq_terminator"]["context"] = "none"
    config["fsq_terminator"]["cameras"] = "top"

    settings = MODULE.build_settings(config)

    assert settings["terminator_context"] == "none"
    assert settings["terminator_cameras"] == "top"
    assert settings["training_mode"] == "terminator_none_top"
    assert settings["run_name"] == (
        "bs2_FSQ345_test_source_terminator_none_top_test"
    )


def test_auxiliary_training_uses_dataset_logical_code_space(tmp_path):
    config = _config(tmp_path)
    info_path = (
        tmp_path
        / "dataset/skillvla_dataset/source/FSQ345_test/skillvla/meta/info.json"
    )
    info = json.loads(info_path.read_text())
    info["skill_code_space_id"] = "FSQ345_original_taxonomy"
    info_path.write_text(json.dumps(info))

    settings = MODULE.build_settings(config)

    assert settings["skill_code_space_id"] == "FSQ345_original_taxonomy"


def test_terminator_only_selects_explicit_relabeled_dataset(tmp_path):
    config = _config(tmp_path, terminator=True, predictor=False)
    relabeled_run = _write_relabeled_variant(config)
    config["dataset"]["relabeled"] = "relabeled_85k"

    settings = MODULE.build_settings(config)

    assert settings["skillvla_dataset_dir"] == relabeled_run / "skillvla"
    assert settings["fsq_path"] == relabeled_run / "FSQ.pt"
    assert settings["skill_code_space_id"] == "FSQ345_test"
    assert settings["dataset_relabeled"] is True
    assert settings["dataset_relabel_ignored_for_predictor"] is False


def test_predictor_training_ignores_relabeled_dataset(tmp_path):
    config = _config(tmp_path, terminator=False, predictor=True)
    _write_relabeled_variant(config)
    config["dataset"]["relabeled"] = "relabeled_85k"

    settings = MODULE.build_settings(config)

    original_run = (
        tmp_path / "dataset/skillvla_dataset/source/FSQ345_test"
    )
    assert settings["skillvla_dataset_dir"] == original_run / "skillvla"
    assert settings["fsq_path"] == original_run / "FSQ.pt"
    assert settings["dataset_relabeled"] is False
    assert settings["dataset_relabel_ignored_for_predictor"] is True


def test_pt_rejects_component_checkpoint(tmp_path):
    checkpoint = _write_auxiliary_checkpoint(tmp_path, name="predictor_pt")
    config = _config(tmp_path, predictor_checkpoint=checkpoint)

    with pytest.raises(ValueError, match="mode=pt must leave"):
        MODULE.build_settings(config)


def test_fsq_override_is_rejected(tmp_path):
    config = _config(tmp_path)
    config["warm_start"]["fsq"] = "models/some_other_fsq.pt"

    with pytest.raises(ValueError, match="FSQ checkpoint is always"):
        MODULE.build_settings(config)


def test_ft_requires_at_least_one_component_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="mode=ft requires"):
        MODULE.build_settings(_config(tmp_path, mode="ft"))


def test_ft_predictor_only_is_inferred_from_checkpoint_path(tmp_path):
    checkpoint = _write_auxiliary_checkpoint(
        tmp_path,
        name="predictor_pt",
        predictor=True,
        terminator=False,
        training_batch_size=8,
        dataset_source_lineage=["libero_90_full_firsthalf_scene"],
        run_suffix_lineage=["pttag"],
    )
    settings = MODULE.build_settings(
        _config(
            tmp_path,
            mode="ft",
            predictor=False,
            terminator=True,
            predictor_checkpoint=checkpoint,
            dataset_source="libero_10_full_1",
        )
    )

    assert settings["initialization_mode"] == "ft"
    assert settings["train_skill_predictor"] is True
    assert settings["train_terminator"] is False
    assert settings["predictor_checkpoint_path"] == tmp_path / checkpoint
    assert settings["terminator_checkpoint_path"] == ""
    assert settings["batch_size"] == 8
    assert settings["training_batch_size"] == 8
    assert settings["run_name"] == (
        "bs8_FSQ345_test_libero_90_full_firsthalf_scene_"
        "libero_10_full_1_predictor_pttag_test"
    )
    assert settings["run_suffix_lineage"] == '["pttag", "test"]'


@pytest.mark.parametrize("mode", [
    "mode3_preskill", "mode3_preskill_wristX", "mode3_preskill_wristX_proprioX", "mode3_preskill_proprioX",
])
def test_ft_inherits_relative_xyz_and_input_ablation_mode(tmp_path, mode):
    checkpoint = _write_auxiliary_checkpoint(tmp_path, name="predictor_mode3", predictor=True, terminator=False)
    config_path = tmp_path / checkpoint / "config.json"
    source = json.loads(config_path.read_text())
    source.update(skill_predictor_mode=mode, skill_predictor_end_state_mode="xyz")
    config_path.write_text(json.dumps(source))
    settings = MODULE.build_settings(_config(
        tmp_path, mode="ft", predictor=False, terminator=True,
        predictor_checkpoint=checkpoint, dataset_source="libero_10_full_1",
    ))
    assert settings["skill_predictor_mode"] == mode
    assert mode in settings["run_name"]
    assert settings["skill_predictor_end_state_mode"] == "xyz"
    assert settings["skill_predictor_sampling_mode"] == "mode1"


def test_ft_terminator_only_inherits_checkpoint_contract(tmp_path):
    checkpoint = _write_auxiliary_checkpoint(
        tmp_path,
        name="terminator_pt",
        predictor=False,
        terminator=True,
        terminator_context="proprio",
        terminator_arch="small",
        terminator_vision_backbone="dino",
        terminator_freeze_vision_encoder=False,
    )
    settings = MODULE.build_settings(
        _config(
            tmp_path,
            mode="ft",
            predictor=True,
            terminator=False,
            terminator_checkpoint=checkpoint,
        )
    )

    assert settings["train_skill_predictor"] is False
    assert settings["train_terminator"] is True
    assert settings["terminator_context"] == "proprio"
    assert settings["terminator_arch"] == "small"
    assert settings["terminator_vision_backbone"] == "dino"
    assert settings["terminator_freeze_vision_encoder"] is False
    assert settings["terminator_end_target_sigma"] == 1.5
    assert settings["terminator_end_pos_weight"] == 2.0
    assert settings["terminator_cameras"] == "both"
    assert settings["run_name"] == (
        "bs2_FSQ345_test_pt_source_source_terminator_prop_both_test"
    )


def test_ft_rejects_predictor_and_terminator_checkpoints_together(tmp_path):
    predictor_checkpoint = _write_auxiliary_checkpoint(
        tmp_path,
        name="predictor_pt",
        predictor=True,
        terminator=False,
        dataset_source_lineage=["predictor_pt_source"],
    )
    terminator_checkpoint = _write_auxiliary_checkpoint(
        tmp_path,
        name="terminator_pt",
        predictor=False,
        terminator=True,
        dataset_source_lineage=["terminator_pt_source"],
    )
    with pytest.raises(ValueError, match="separate jobs"):
        MODULE.build_settings(
            _config(
                tmp_path,
                mode="ft",
                predictor_checkpoint=predictor_checkpoint,
                terminator_checkpoint=terminator_checkpoint,
            )
        )


def test_ft_rejects_different_component_pt_batch_sizes(tmp_path):
    """The joint request is rejected before component batch contracts matter."""
    predictor_checkpoint = _write_auxiliary_checkpoint(
        tmp_path,
        name="predictor_pt_batch",
        predictor=True,
        terminator=False,
        training_batch_size=8,
    )
    terminator_checkpoint = _write_auxiliary_checkpoint(
        tmp_path,
        name="terminator_pt_batch",
        predictor=False,
        terminator=True,
        training_batch_size=16,
    )
    config = _config(
        tmp_path,
        mode="ft",
        predictor_checkpoint=predictor_checkpoint,
        terminator_checkpoint=terminator_checkpoint,
    )

    with pytest.raises(ValueError, match="separate jobs"):
        MODULE.build_settings(config)


def test_ft_rejects_checkpoint_without_requested_component(tmp_path):
    checkpoint = _write_auxiliary_checkpoint(
        tmp_path, name="terminator_pt", predictor=False, terminator=True
    )
    config = _config(
        tmp_path, mode="ft", predictor_checkpoint=checkpoint
    )

    with pytest.raises(ValueError, match="no trained predictor"):
        MODULE.build_settings(config)


def test_ft_rejects_different_code_space(tmp_path):
    checkpoint = _write_auxiliary_checkpoint(
        tmp_path,
        name="predictor_pt",
        predictor=True,
        terminator=False,
        code_space_id="FSQ345_other",
    )
    config = _config(tmp_path, mode="ft", predictor_checkpoint=checkpoint)

    with pytest.raises(ValueError, match="code-space mismatch"):
        MODULE.build_settings(config)


def test_ft_ignores_pt_only_model_sections(tmp_path):
    checkpoint = _write_auxiliary_checkpoint(
        tmp_path, name="predictor_pt", predictor=True, terminator=False
    )
    config = _config(tmp_path, mode="ft", predictor_checkpoint=checkpoint)
    config["fsq_terminator"] = {"this_would_be_invalid_in_pt": True}
    config["skill_predictor"] = {"train": False, "reader": {"tokens": -123}}

    settings = MODULE.build_settings(config)

    assert settings["training_mode"] == "predictor"
    assert settings["skill_predictor_reader_tokens"] == 4


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("context", "state", "context must be"),
        ("default_arch", "gru", "default_arch must be"),
        ("vision_backbone", "vit", "vision_backbone must be"),
    ],
)
def test_invalid_terminator_contract_is_rejected(tmp_path, key, value, error):
    config = _config(tmp_path)
    config["fsq_terminator"][key] = value

    with pytest.raises(ValueError, match=error):
        MODULE.build_settings(config)


def test_full_vlm_co_training_forces_lora_off(tmp_path):
    config = _config(tmp_path, terminator=False, predictor=True)
    config["skill_predictor"]["freeze_vlm"] = False

    settings = MODULE.build_settings(config)

    assert settings["skill_predictor_freeze_vlm"] is False
    assert settings["skill_predictor_lora"] is False
    assert settings["skill_predictor_detach_vlm"] is False
    assert "predictor_fullvlm" in settings["run_name"]


def test_focus_uv_predictor_requires_focus_artifact(tmp_path):
    config = _config(tmp_path, terminator=False, predictor=True)
    config["skill_predictor"]["focus_uv"] = {
        "enabled": True,
        "loss_weight": 0.25,
    }

    with pytest.raises(FileNotFoundError, match="skill_focus_uv_path"):
        MODULE.build_settings(config)

    info_path = (
        tmp_path
        / "dataset/skillvla_dataset/source/FSQ345_test/skillvla/meta/info.json"
    )
    info = json.loads(info_path.read_text())
    info["skill_focus_uv_path"] = str(info_path.parents[2] / "skill_focus_uv.npz")
    info_path.write_text(json.dumps(info))
    settings = MODULE.build_settings(config)
    assert settings["skill_predictor_focus_uv_enabled"] is True
    assert "predictor_uv" in settings["run_name"]


@pytest.mark.parametrize("target", ["xyz", "full_state"])
@pytest.mark.parametrize("mode", ["mode1", "mode2", "mode1_preskill", "mode2_preskill"])
def test_xyz_predictor_requires_no_uv_artifact(tmp_path, target, mode):
    config = _config(tmp_path, terminator=False, predictor=True)
    config["skill_predictor"].update(spatial_target=target, mode=mode)
    settings = MODULE.build_settings(config)
    assert settings["skill_predictor_end_state_mode"] == target
    assert settings["skill_predictor_focus_uv_enabled"] is False


def test_goal_xyz_terminator_config_requires_no_uv_artifact(tmp_path):
    config = _config(tmp_path, terminator=True, predictor=False)
    config["fsq_terminator"].update(goal_xyz=True, context="proprio")
    settings = MODULE.build_settings(config)
    assert settings["terminator_goal_xyz"] is True


def test_top_patch_terminator_still_requires_uv_artifact(tmp_path):
    config = _config(tmp_path, terminator=True, predictor=False)
    config["fsq_terminator"].update(
        goal_xyz=True, context="proprio", agent_patch_align_weight=0.1,
    )
    with pytest.raises(FileNotFoundError, match="skill_focus_uv_path"):
        MODULE.build_settings(config)


@pytest.mark.parametrize(
    ("component", "predictor", "terminator"),
    [("Predictor", True, False), ("Terminator", False, True)],
)
def test_stage1_auxiliary_output_component(tmp_path, component, predictor, terminator):
    config = _config(tmp_path, predictor=predictor, terminator=terminator)
    config["stage1_component"] = component
    settings = MODULE.build_settings(config)
    assert settings["output_dir"] == (
        tmp_path / "outputs" / "skillVLA_PT" / component / settings["run_name"]
    )


def test_stage1_auxiliary_component_rejects_wrong_target(tmp_path):
    config = _config(tmp_path, predictor=False, terminator=True)
    config["stage1_component"] = "Predictor"
    with pytest.raises(ValueError, match="train only the predictor"):
        MODULE.build_settings(config)


def _rewrite_checkpoint_fields(tmp_path: Path, checkpoint: str, **fields) -> None:
    path = tmp_path / checkpoint / "config.json"
    source = json.loads(path.read_text())
    source.update(fields)
    path.write_text(json.dumps(source))


def test_ft_predictor_can_re_choose_how_the_inherited_vlm_adapts(tmp_path):
    """The predictor is inherited whole; only the way its VLM adapts is the FT run's to pick."""
    checkpoint = _write_auxiliary_checkpoint(
        tmp_path, name="predictor_pt", predictor=True, terminator=False
    )
    # A "fullvlm" source: the VLM co-trained and carries no adapter.
    _rewrite_checkpoint_fields(
        tmp_path, checkpoint, skill_predictor_freeze_vlm=False, skill_predictor_lora=False
    )

    base = _config(
        tmp_path,
        mode="ft",
        predictor=False,
        terminator=True,
        predictor_checkpoint=checkpoint,
        dataset_source="libero_10_full_1",
    )

    def build(**predictor_ft):
        config = copy.deepcopy(base)
        if predictor_ft:
            config["predictor_ft"] = predictor_ft
        return MODULE.build_settings(config)

    # Saying nothing keeps the checkpoint's own choice, so existing FT runs are untouched.
    inherited = build()
    assert inherited["skill_predictor_freeze_vlm"] is False
    assert inherited["skill_predictor_lora"] is False
    assert "_fullvlm" in inherited["run_name"]

    # Freeze the VLM: the reader and the skill/xyz heads keep training, and the name says so.
    frozen = build(freeze_vlm=True, lora={"enabled": False})
    assert frozen["skill_predictor_freeze_vlm"] is True
    assert frozen["skill_predictor_lora"] is False
    assert frozen["skill_predictor_detach_vlm"] is True
    assert "_frozenvlm" in frozen["run_name"] and "_fullvlm" not in frozen["run_name"]

    # Freeze it behind LoRA. The adapter is new, so its shape comes from this run, not the source.
    lora = build(freeze_vlm=True, lora={"enabled": True, "rank": 16})
    assert lora["skill_predictor_freeze_vlm"] is True
    assert lora["skill_predictor_lora"] is True
    assert lora["skill_predictor_detach_vlm"] is False
    assert lora["skill_predictor_lora_rank"] == 16
    assert lora["skill_predictor_lora_targets"] == "q,k,v,o"
    assert "_lora" in lora["run_name"]

    # Nothing else may be re-chosen: the inherited weights would stop fitting.
    assert lora["skill_predictor_reader_tokens"] == inherited["skill_predictor_reader_tokens"]
    assert lora["skill_predictor_end_state_mode"] == inherited["skill_predictor_end_state_mode"]

    with pytest.raises(ValueError, match="cannot co-train"):
        build(freeze_vlm=False, lora={"enabled": True})


def test_ft_refuses_to_drop_the_lora_a_checkpoint_was_trained_with(tmp_path):
    """Its projections are stored wrapped, so removing the adapter would orphan them."""
    checkpoint = _write_auxiliary_checkpoint(
        tmp_path, name="predictor_lora_pt", predictor=True, terminator=False
    )
    _rewrite_checkpoint_fields(
        tmp_path, checkpoint, skill_predictor_freeze_vlm=True, skill_predictor_lora=True
    )
    config = _config(
        tmp_path,
        mode="ft",
        predictor=False,
        terminator=True,
        predictor_checkpoint=checkpoint,
        dataset_source="libero_10_full_1",
    )
    config["predictor_ft"] = {"freeze_vlm": True, "lora": {"enabled": False}}

    with pytest.raises(ValueError, match="cannot drop the LoRA"):
        MODULE.build_settings(config)


@pytest.mark.parametrize(
    "unfreeze,freeze,vision,detach,suffix",
    [
        ("all", False, False, False, "_fullvlm"),
        ("none", True, False, True, "_frozenvlm"),
        ("vision_encoder", True, True, False, "_visionencoder"),
    ],
)
def test_ft_unfreeze_selector_resolves_training_contract_and_name(
    tmp_path, unfreeze, freeze, vision, detach, suffix,
):
    checkpoint = _write_auxiliary_checkpoint(
        tmp_path, name="predictor_pt", predictor=True, terminator=False
    )
    _rewrite_checkpoint_fields(tmp_path, checkpoint, skill_predictor_lora=False)
    config = _config(tmp_path, mode="ft", predictor_checkpoint=checkpoint)
    config["predictor_ft"] = {"unfreeze": unfreeze, "lora": {"enabled": False}}
    settings = MODULE.build_settings(config)
    assert settings["skill_predictor_freeze_vlm"] is freeze
    assert settings["skill_predictor_train_vision_encoder"] is vision
    assert settings["skill_predictor_detach_vlm"] is detach
    assert suffix in settings["run_name"]


@pytest.mark.parametrize("unfreeze", ["none", "vision_encoder"])
def test_ft_unfreeze_keeps_lora_as_an_independent_option(tmp_path, unfreeze):
    checkpoint = _write_auxiliary_checkpoint(
        tmp_path, name="predictor_pt", predictor=True, terminator=False
    )
    _rewrite_checkpoint_fields(tmp_path, checkpoint, skill_predictor_lora=False)
    config = _config(tmp_path, mode="ft", predictor_checkpoint=checkpoint)
    config["predictor_ft"] = {"unfreeze": unfreeze, "lora": {"enabled": True}}
    settings = MODULE.build_settings(config)
    assert settings["skill_predictor_lora"] is True
    assert settings["skill_predictor_detach_vlm"] is False
    assert settings["skill_predictor_train_vision_encoder"] is (unfreeze == "vision_encoder")
    assert "_lora" in settings["run_name"]


@pytest.mark.parametrize(
    "selection,error",
    [
        ({"unfreeze": "invalid"}, "unfreeze must be"),
        ({"unfreeze": "all", "lora": {"enabled": True}}, "cannot co-train"),
        ({"unfreeze": "none", "freeze_vlm": True}, "not both"),
    ],
)
def test_ft_unfreeze_rejects_ambiguous_or_invalid_options(tmp_path, selection, error):
    checkpoint = _write_auxiliary_checkpoint(
        tmp_path, name="predictor_pt", predictor=True, terminator=False
    )
    config = _config(tmp_path, mode="ft", predictor_checkpoint=checkpoint)
    config["predictor_ft"] = selection
    with pytest.raises(ValueError, match=error):
        MODULE.build_settings(config)


@pytest.mark.parametrize("predictor", [True, False])
def test_scalar_launch_preserves_existing_training_settings(tmp_path, predictor):
    config = _config(tmp_path, predictor=predictor, terminator=not predictor)
    single = MODULE.build_settings(config)
    launched = MODULE.build_launch_settings(config)
    assert launched == {**single, "predictor_parallel_count": 1, "predictor_parallel_runs": "[]"}


def test_predictor_mode_list_expands_to_distinct_single_gpu_runs(tmp_path):
    config = _config(tmp_path, predictor=True, terminator=False)
    config["stage1_component"] = "Predictor"
    modes = ["mode3_preskill", "mode3_preskill_proprioX"]
    config["skill_predictor"].update(mode=modes, spatial_target="xyz", freeze_vlm=False)
    original = copy.deepcopy(config)
    settings = MODULE.build_launch_settings(config)
    manifest = json.loads(settings["predictor_parallel_runs"])
    assert config == original
    assert settings["predictor_parallel_count"] == 2
    assert settings["num_gpus"] == 1
    assert settings["train_gres"] == "gpu:1"
    assert settings["batch_size"] == 2  # not split between models
    assert [run["mode"] for run in manifest] == modes
    for run in manifest:
        scalar = copy.deepcopy(config)
        scalar["skill_predictor"]["mode"] = run["mode"]
        expected = MODULE.build_settings(scalar)
        assert run["output_dir"] == str(expected["output_dir"])
        assert run["run_name"] == expected["run_name"]
        assert expected["skill_predictor_mode"] in run["run_name"]
        worker = MODULE.build_launch_settings(config, predictor_mode_override=run["mode"])
        assert worker["predictor_parallel_count"] == 1  # never recurse into another group
        assert worker["output_dir"] == expected["output_dir"]
        assert worker["skill_predictor_mode"] == run["mode"]


@pytest.mark.parametrize("modes", [[], ["mode1", 3], [["mode1"]]])
def test_predictor_mode_list_requires_nonempty_strings(tmp_path, modes):
    config = _config(tmp_path, predictor=True, terminator=False)
    config["skill_predictor"]["mode"] = modes
    with pytest.raises(ValueError, match="non-empty list of strings"):
        MODULE.build_launch_settings(config)


def test_predictor_mode_list_rejects_normalized_duplicates(tmp_path):
    config = _config(tmp_path, predictor=True, terminator=False)
    config["skill_predictor"]["mode"] = ["mode1_preskill", "MODE1_PRESKILL"]
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.build_launch_settings(config)


@pytest.mark.parametrize("field,value", [("gpus", 2), ("gres", "gpu:2"), ("gres", "gpu:b200:2")])
def test_predictor_mode_list_requires_one_allocated_gpu(tmp_path, field, value):
    config = _config(tmp_path, predictor=True, terminator=False)
    config["skill_predictor"]["mode"] = ["mode1", "mode1_preskill"]
    if field == "gpus":
        config["training"]["dataloader"]["gpus"] = value
    else:
        config["slurm"] = {"gres": value}
    with pytest.raises(ValueError, match="shares ONE GPU"):
        MODULE.build_launch_settings(config)


def test_predictor_mode_list_accepts_typed_gpu_and_single_item(tmp_path):
    config = _config(tmp_path, predictor=True, terminator=False)
    config["skill_predictor"]["mode"] = ["mode1"]
    config["slurm"] = {"gres": "gpu:b200:1"}
    settings = MODULE.build_launch_settings(config)
    assert settings["predictor_parallel_count"] == 1
    assert settings["train_gres"] == "gpu:b200:1"


def test_predictor_mode_list_validates_every_run_before_submission(tmp_path):
    config = _config(tmp_path, predictor=True, terminator=False)
    config["skill_predictor"]["mode"] = ["mode1", "mode3_preskill"]
    with pytest.raises(ValueError, match="spatial_target"):
        MODULE.build_launch_settings(config)  # second mode needs XYZ, not the default "off"


def test_predictor_worker_selector_cannot_invent_a_mode(tmp_path):
    config = _config(tmp_path, predictor=True, terminator=False)
    with pytest.raises(ValueError, match="only valid"):
        MODULE.build_launch_settings(config, predictor_mode_override="mode1")
    config["skill_predictor"]["mode"] = ["mode1"]
    with pytest.raises(ValueError, match="not selected"):
        MODULE.build_launch_settings(config, predictor_mode_override="mode1_preskill")


def test_ft_launch_keeps_checkpoint_mode_and_rejects_pt_mode_lists(tmp_path):
    checkpoint = _write_auxiliary_checkpoint(
        tmp_path, name="preskill_pt", predictor=True, terminator=False
    )
    _rewrite_checkpoint_fields(tmp_path, checkpoint, skill_predictor_mode="mode1_preskill")
    config = _config(
        tmp_path, mode="ft", predictor_checkpoint=checkpoint, dataset_source="libero_10_full_1"
    )
    settings = MODULE.build_launch_settings(config)
    assert settings["skill_predictor_mode"] == "mode1_preskill"
    assert settings["predictor_parallel_count"] == 1
    config["skill_predictor"]["mode"] = ["mode1", "mode1_preskill"]
    with pytest.raises(ValueError, match="PT-only"):
        MODULE.build_launch_settings(config)


@pytest.mark.parametrize("modes", ["mode1", ["mode1", "mode1_preskill"]])
def test_submitter_requests_exactly_one_allocation_for_scalar_or_group(tmp_path, modes):
    import yaml

    config = _config(tmp_path, predictor=True, terminator=False)
    config["stage1_component"] = "Predictor"
    config["skill_predictor"]["mode"] = modes
    path = tmp_path / "predictor.yaml"
    path.write_text(yaml.safe_dump(config))
    # The real submitter snapshots the config but uses our fake sbatch, never Slurm.
    shared = SCRIPT.parents[4] / "src"
    (tmp_path / "src").mkdir()
    for name in ("snapshot_config.sh", "submit_job.sh"):
        (tmp_path / "src" / name).symlink_to(shared / name)
    (tmp_path / "global_config.yaml").write_text("{}\n")
    marker = tmp_path / "sbatch_calls"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$FAKE_SBATCH_LOG"\n')
    fake_sbatch.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{Path(sys.executable).parent}:{os.environ['PATH']}",
        "SKILL_AUX_TRAIN_CONFIG": str(path),
        "SKILL_AUX_SUBMIT_DIR": str(tmp_path),
        "SBD_SCHEDULER": "slurm",
        "FAKE_SBATCH_LOG": str(marker),
    }
    env.pop("SKILL_AUX_DRY_RUN", None)
    result = subprocess.run(
        ["bash", str(SCRIPT.with_name("submit_auxiliary.sh"))],
        env=env, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    calls = marker.read_text().splitlines()
    assert len(calls) == 1
    assert "--gres=gpu:1" in calls[0] and "--array" not in calls[0]
    assert ("independent Predictors on the SAME GPU" in result.stdout) == isinstance(modes, list)
    snapshots = list((tmp_path / ".config_snapshots").glob("*/predictor.yaml"))
    assert len(snapshots) == 1
    assert yaml.safe_load(snapshots[0].read_text())["skill_predictor"]["mode"] == modes


@pytest.mark.parametrize("parallel", [False, True])
def test_batch_script_routes_modes_and_resumes_each_model_with_one_staging_pass(tmp_path, parallel):
    """Exercise the real bash orchestration with tiny fake trainers, no CUDA/Slurm."""
    import yaml

    config = _config(tmp_path, predictor=True, terminator=False)
    config["stage1_component"] = "Predictor"
    modes = ["mode3_preskill", "mode3_preskill_proprioX"] if parallel else ["mode3_preskill"]
    config["skill_predictor"].update(
        mode=modes if parallel else modes[0], spatial_target="xyz", freeze_vlm=False
    )
    path = tmp_path / "predictor.yaml"
    path.write_text(yaml.safe_dump(config))
    venv_bin = tmp_path / ".venv/bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(sys.executable)
    # Only the GPU diagnostics are mocked. Config resolution, group supervision,
    # independent model commands and automatic resume all use the real scripts.
    (venv_bin / "activate").write_text(
        f'export PATH="{venv_bin}:$PATH"\n'
        f'python() {{ if [ "$1" = - ]; then return 0; fi; "{sys.executable}" "$@"; }}\n'
    )
    smi = venv_bin / "nvidia-smi"
    smi.write_text("#!/usr/bin/env bash\nexit 0\n")
    smi.chmod(0o755)
    shared = tmp_path / "lerobot/examples/libero/configs/src"
    shared.mkdir(parents=True)
    (shared / "gpu_guard.sh").write_text(
        'require_cuda_or_requeue() { echo preflight >> "$TEST_EVENTS"; }\n'
        'handle_inline_cuda_guard_exit() { return 0; }\n'
    )
    (shared / "runtime_env.sh").write_text("")
    staged = tmp_path / "stage/skillvla"
    staged.mkdir(parents=True)
    (shared / "stage_skillvla_dataset.sh").write_text(
        'stage_skillvla_dataset_on_node() {\n'
        '  echo staged >> "$TEST_EVENTS"\n'
        f'  STAGED_SKILLVLA_DATASET_DIR="{staged}"\n'
        '}\n'
    )
    trainer = venv_bin / "lerobot-train"
    trainer.write_text(f'''#!{sys.executable}
import json, os, sys
from pathlib import Path
args = dict(arg[2:].split("=", 1) for arg in sys.argv[1:])
output = Path(args["output_dir"])
output.mkdir(parents=True, exist_ok=True)
record = {{"args": args, "cuda": os.environ["CUDA_VISIBLE_DEVICES"]}}
with (output / "calls.jsonl").open("a") as stream:
    stream.write(json.dumps(record) + "\\n")
checkpoint = output / "checkpoints/last/pretrained_model"
checkpoint.mkdir(parents=True, exist_ok=True)
(checkpoint / "train_config.json").write_text("{{}}")
''')
    trainer.chmod(0o755)
    env = {
        **os.environ,
        "SKILL_AUX_TRAIN_CONFIG": str(path),
        "SKILL_AUX_SRC_DIR": str(SCRIPT.parent),
        "SLURM_SUBMIT_DIR": str(tmp_path),
        "CUDA_VISIBLE_DEVICES": "GPU-same-device",
        "TEST_EVENTS": str(tmp_path / "events"),
    }
    for key in ("SKILL_AUX_GROUP_WORKER", "SKILL_AUX_PREDICTOR_MODE_OVERRIDE"):
        env.pop(key, None)
    for _attempt in range(2):
        result = subprocess.run(
            ["bash", str(SCRIPT.with_name("train.sbatch"))],
            env=env, cwd=tmp_path, text=True, capture_output=True, timeout=30,
        )
        assert result.returncode == 0, (result.stdout, result.stderr)
    assert (tmp_path / "events").read_text().splitlines().count("staged") == 2
    assert (tmp_path / "events").read_text().splitlines().count("preflight") == 2
    for mode in modes:
        scalar = copy.deepcopy(config)
        scalar["skill_predictor"]["mode"] = mode
        output = MODULE.build_settings(scalar)["output_dir"]
        records = [json.loads(line) for line in (output / "calls.jsonl").read_text().splitlines()]
        assert len(records) == 2
        assert {record["cuda"] for record in records} == {"GPU-same-device"}
        assert {record["args"]["policy.skill_predictor_mode"] for record in records} == {mode}
        assert {record["args"]["dataset.root"] for record in records} == {str(staged)}
        assert "resume" not in records[0]["args"]
        assert records[1]["args"]["resume"] == "true"
        assert records[1]["args"]["config_path"] == str(output / "checkpoints/last/pretrained_model/train_config.json")
