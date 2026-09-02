import json
import sys
from pathlib import Path

import pytest


_EVAL_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/stage2_eval/src"
)
sys.path.insert(0, str(_EVAL_SRC))

from stage2_eval_config import build_settings

_PREDICTOR_FIELDS = {
    "skill_vocab_size": 27,
    "skill_fsq_levels": [3, 3, 3],
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
    "tokenizer_max_length": 200,
}


def _write_checkpoint(
    directory: Path,
    config: dict,
    *,
    dataset_root: str | None = None,
    with_processors: bool = False,
) -> None:
    directory.mkdir(parents=True)
    (directory / "model.safetensors").touch()
    (directory / "config.json").write_text(json.dumps(config))
    if with_processors:
        (directory / "policy_preprocessor.json").touch()
        (directory / "policy_postprocessor.json").touch()
    if dataset_root is not None:
        (directory / "train_config.json").write_text(
            json.dumps({"dataset": {"root": dataset_root}})
        )


def _checkpoint_tree(tmp_path: Path, *, policy_type: str = "skill_vla_stage2") -> dict:
    project = tmp_path / "project"
    run = (
        project
        / "dataset_filtered/skillvla_dataset/libero_90_full_full/FSQ333_run"
    )
    skill_dataset = run / "skillvla"
    (skill_dataset / "meta").mkdir(parents=True)
    (skill_dataset / "meta/info.json").write_text("{}")
    fsq = run / "FSQ.pt"
    fsq.touch()
    for directory in ("models/dino", "models/tokenizer"):
        (project / directory).mkdir(parents=True)

    stage1_path = (
        project
        / "outputs_filtered/skillVLA_stage1/stage1_prior/checkpoints/100000/pretrained_model"
    )
    _write_checkpoint(
        stage1_path,
        {
            "type": "skill_expert",
            "architecture": "cond_gemma",
            "architecture_label": "arch0",
            "architecture_revision": "skillvla_real_v1",
            "conditioning_route": "state_cond",
            "action_loss_mode": "flow",
            "num_visual_latents_per_camera": 32,
            "visual_perceiver_width": 1024,
            "train_skill_predictor": False,
            "train_terminator": False,
            "fsq_path": str(fsq),
            "dino_model_path": str(project / "models/dino"),
            "tokenizer_path": str(project / "models/tokenizer"),
            "n_action_steps": 10,
            "chunk_size": 10,
            "skill_vocab_size": 27,
            "skill_fsq_levels": [3, 3, 3],
        },
        dataset_root=str(skill_dataset),
        with_processors=True,
    )

    predictor_path = (
        project
        / "outputs_filtered/skillVLA_terminator/predictor_run/checkpoints/050000/pretrained_model"
    )
    _write_checkpoint(
        predictor_path,
        {
            "type": "skill_aux",
            "train_skill_predictor": True,
            "tokenizer_path": str(project / "models/tokenizer"),
            "fsq_path": str(fsq),
            **_PREDICTOR_FIELDS,
        },
    )

    terminator_path = (
        project
        / "outputs_filtered/skillVLA_terminator/terminator_run/checkpoints/020000/pretrained_model"
    )
    _write_checkpoint(
        terminator_path,
        {
            "type": "skill_aux",
            "train_terminator": True,
            "skill_fsq_levels": [3, 3, 3],
            "fsq_path": str(fsq),
        },
    )

    model_dir = "stage1_prior_100000_gt_batchON"
    policy_path = (
        project
        / "outputs_filtered/skillVLA_stage2"
        / model_dir
        / "checkpoints/last/pretrained_model"
    )
    _write_checkpoint(
        policy_path,
        {
            "type": policy_type,
            "architecture": "cond_gemma",
            "architecture_label": "arch0",
            "architecture_revision": "skillvla_real_v1",
            "conditioning_route": "state_cond",
            "action_loss_mode": "flow",
            "num_visual_latents_per_camera": 32,
            "visual_perceiver_width": 1024,
            "train_skill_predictor": True,
            "train_terminator": False,
            "stage1_checkpoint_path": str(stage1_path),
            "fsq_path": str(fsq),
            "dino_model_path": str(project / "models/dino"),
            "tokenizer_path": str(project / "models/tokenizer"),
            "n_action_steps": 10,
            "chunk_size": 10,
            **_PREDICTOR_FIELDS,
        },
        dataset_root=str(skill_dataset),
        with_processors=True,
    )
    return {
        "project_root": str(project),
        "outputs_root": "outputs_filtered",
        "model_dir": model_dir,
        "checkpoint": "last",
        "skill_source": "predictor",
        "external_predictor_model": str(predictor_path),
        "external_terminator_model": str(terminator_path),
        "output_name": "smoke",
        "target_task": "libero_90",
        "task_ids": [0, 1],
        "oracle": {"episode_exact": False, "advance_mode": "terminator"},
        "terminator": {"end_mode": "or"},
        "logging": {"wandb": {"enable": False}},
    }


def _stage2_config_path(config: dict) -> Path:
    return (
        Path(config["project_root"])
        / "outputs_filtered/skillVLA_stage2"
        / config["model_dir"]
        / "checkpoints"
        / config["checkpoint"]
        / "pretrained_model/config.json"
    )


def test_stage2_eval_expands_into_stage2_and_prior_panels(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    settings = build_settings(config)
    panels = json.loads(settings["models_json"])

    assert settings["model_count"] == 2
    assert settings["grid_columns"] == 2
    assert [panel["mode"] for panel in panels] == ["stage2", "prior"]
    assert [panel["label"] for panel in panels] == [
        "model1-predictor-stage2",
        "model1-predictor-prior",
    ]
    stage2_panel, prior_panel = panels
    # No eval selector is needed. Checkpoints created before stage2_mode was
    # introduced are unambiguously interpreted as likelihood.
    assert stage2_panel["stage2_mode"] == "likelihood"
    assert stage2_panel["dsbc_noise_output_mode"] == "shared"
    assert "stage2_mode" not in prior_panel
    assert stage2_panel["policy_path"].endswith(
        "skillVLA_stage2/stage1_prior_100000_gt_batchON/checkpoints/last/pretrained_model"
    )
    assert prior_panel["policy_path"].endswith(
        "skillVLA_stage1/stage1_prior/checkpoints/100000/pretrained_model"
    )
    # Both panels select skills from the shared external predictor and advance
    # with the shared external terminator.
    for panel in panels:
        assert panel["skill_source"] == "external"
        assert panel["advance_mode"] == "external"
        assert panel["external_predictor_model"] == config["external_predictor_model"]
        assert panel["external_terminator_model"] == config["external_terminator_model"]
    # The prior panel evaluates on the Stage-2 checkpoint's dataset for one
    # shared oracle map.
    assert prior_panel["skill_dataset_dir"] == stage2_panel["skill_dataset_dir"]
    assert prior_panel["architecture"] == "cond_gemma"
    assert settings["eval_out_dir"] == _EVAL_SRC.parent / "outputs/smoke"


def test_stage2_eval_external_predictor_owns_module_contract_and_tokenizer(
    tmp_path: Path,
) -> None:
    config = _checkpoint_tree(tmp_path)
    project = Path(config["project_root"])
    # Stage-2 training used only a pristine pi0.5 VLM placeholder. The eval
    # predictor legitimately adds its own all-layer reader and skill LoRA.
    stage2_config = _stage2_config_path(config)
    stage2_policy = json.loads(stage2_config.read_text())
    stage2_policy.update(
        {
            "skill_predictor_all_layers": False,
            "skill_predictor_detach_vlm": True,
            "skill_predictor_lora": False,
            "skill_predictor_deadzone_frac": 0.0,
        }
    )
    stage2_config.write_text(json.dumps(stage2_policy))

    predictor_path = Path(config["external_predictor_model"])
    predictor_config = predictor_path / "config.json"
    predictor_policy = json.loads(predictor_config.read_text())
    external_tokenizer = project / "models/external_predictor_tokenizer"
    external_tokenizer.mkdir()
    predictor_policy["tokenizer_path"] = str(external_tokenizer)
    predictor_config.write_text(json.dumps(predictor_policy))

    panels = json.loads(build_settings(config)["models_json"])

    assert {panel["tokenizer_path"] for panel in panels} == {
        str(external_tokenizer)
    }
    assert all(panel["skill_source"] == "external" for panel in panels)


def test_stage2_eval_automatically_reads_dsbc_mode_from_checkpoint(
    tmp_path: Path,
) -> None:
    config = _checkpoint_tree(tmp_path)
    checkpoint_config = _stage2_config_path(config)
    policy = json.loads(checkpoint_config.read_text())
    policy.update(
        {
            "stage2_mode": "dsbc",
            "dsbc_noise_output_mode": "per_step",
            "dsbc_frs_num_steps": 8,
            "dsbc_anchor_seed": 17,
        }
    )
    checkpoint_config.write_text(json.dumps(policy))

    panels = json.loads(build_settings(config)["models_json"])
    stage2_panel, prior_panel = panels

    assert stage2_panel["stage2_mode"] == "dsbc"
    assert stage2_panel["dsbc_noise_output_mode"] == "per_step"
    assert stage2_panel["dsbc_frs_num_steps"] == 8
    assert stage2_panel["dsbc_anchor_seed"] == 17
    assert "stage2_mode" not in prior_panel


def test_stage2_eval_rejects_invalid_checkpoint_mode(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    checkpoint_config = _stage2_config_path(config)
    policy = json.loads(checkpoint_config.read_text())
    policy["stage2_mode"] = "manual_override"
    checkpoint_config.write_text(json.dumps(policy))

    with pytest.raises(ValueError, match="Invalid Stage-2 mode"):
        build_settings(config)


def test_stage2_eval_dedupes_shared_prior_panels(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    project = Path(config["project_root"])
    first_dir = config.pop("model_dir")
    first_path = (
        project
        / "outputs_filtered/skillVLA_stage2"
        / first_dir
        / "checkpoints/last/pretrained_model"
    )
    second_dir = "second_run_100000_gt_batchON"
    second_path = (
        project
        / "outputs_filtered/skillVLA_stage2"
        / second_dir
        / "checkpoints/last/pretrained_model"
    )
    second_path.mkdir(parents=True)
    for name in (
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "config.json",
        "train_config.json",
    ):
        content = (first_path / name).read_bytes()
        (second_path / name).write_bytes(content)
    config["models"] = [
        {"model_dir": first_dir, "label": "m1"},
        {"model_dir": second_dir, "label": "m2"},
    ]

    settings = build_settings(config)
    panels = json.loads(settings["models_json"])

    # Both models share one frozen Stage-1 prior, so it is evaluated once.
    assert settings["model_count"] == 3
    assert [panel["label"] for panel in panels] == ["m1-stage2", "prior", "m2-stage2"]


def test_stage2_eval_gt_mode_needs_no_predictor(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    config["skill_source"] = "gt"
    del config["external_predictor_model"]

    settings = build_settings(config)
    panels = json.loads(settings["models_json"])

    assert [panel["skill_source"] for panel in panels] == ["gt", "gt"]
    assert [panel["advance_mode"] for panel in panels] == ["external", "external"]


def test_stage2_eval_uses_explicit_oracle_skill_dataset(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    project = Path(config["project_root"])
    oracle_dataset = (
        project
        / "dataset_filtered/skillvla_dataset/libero_10_full_5/FSQ333_run/skillvla"
    )
    (oracle_dataset / "meta").mkdir(parents=True)
    (oracle_dataset / "meta/info.json").write_text("{}")
    (oracle_dataset.parent / "skill_latents.npz").touch()
    config["skill_source"] = "gt"
    config["oracle"]["skill_dataset_dir"] = str(
        oracle_dataset.relative_to(project)
    )

    settings = build_settings(config)
    panels = json.loads(settings["models_json"])

    assert settings["skill_dataset_dir"] == oracle_dataset
    assert settings["skill_latents_path"] == oracle_dataset.parent / "skill_latents.npz"
    assert settings["raw_dataset_dir"] == project / "dataset_filtered/libero_10_full_5"
    assert all(panel["skill_dataset_dir"] == str(oracle_dataset) for panel in panels)
    assert all(panel["skill_source"] == "gt" for panel in panels)


def test_stage2_eval_rejects_missing_oracle_skill_dataset(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    config["oracle"]["skill_dataset_dir"] = "dataset_filtered/missing/skillvla"

    with pytest.raises(FileNotFoundError, match="oracle SkillVLA dataset"):
        build_settings(config)


def test_stage2_eval_empty_oracle_dataset_uses_checkpoint_dataset(
    tmp_path: Path,
) -> None:
    config = _checkpoint_tree(tmp_path)
    config["oracle"]["skill_dataset_dir"] = ""

    settings = build_settings(config)
    expected = (
        Path(config["project_root"])
        / "dataset_filtered/skillvla_dataset/libero_90_full_full/FSQ333_run/skillvla"
    )

    assert settings["skill_dataset_dir"] == expected


def test_stage2_eval_recovers_node_local_dataset_from_portable_lineage(
    tmp_path: Path,
) -> None:
    config = _checkpoint_tree(tmp_path)
    checkpoint = _stage2_config_path(config).parent
    (checkpoint / "train_config.json").write_text(
        json.dumps(
            {
                "dataset": {
                    "root": "/tmp/stage2-job/skillvla",
                    "repo_id": "dohyeon/libero_90_full_full",
                }
            }
        )
    )

    settings = build_settings(config)

    assert settings["skill_dataset_dir"] == (
        Path(config["project_root"])
        / "dataset_filtered/skillvla_dataset/libero_90_full_full/FSQ333_run/skillvla"
    )


def test_stage2_eval_preserves_relabeled_physical_lineage(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    project = Path(config["project_root"])
    original_run = (
        project
        / "dataset_filtered/skillvla_dataset/libero_90_full_full/FSQ333_run"
    )
    relabeled_run = original_run.with_name("FSQ333_run_relabeled_85k")
    original_run.rename(relabeled_run)
    info_path = relabeled_run / "skillvla/meta/info.json"
    info = json.loads(info_path.read_text())
    info["skill_code_space_id"] = "FSQ333_run"
    info_path.write_text(json.dumps(info))

    stage2_path = _stage2_config_path(config).parent
    policy = json.loads((stage2_path / "config.json").read_text())
    policy["fsq_path"] = str(relabeled_run / "FSQ.pt")
    policy["skill_code_space_id"] = "FSQ333_run"
    (stage2_path / "config.json").write_text(json.dumps(policy))
    (stage2_path / "train_config.json").write_text(
        json.dumps(
            {
                "dataset": {
                    "root": "/tmp/expired-stage2/skillvla",
                    "repo_id": "dohyeon/libero_90_full_full",
                }
            }
        )
    )
    stage1_path = (
        project
        / "outputs_filtered/skillVLA_stage1/stage1_prior/checkpoints/100000/pretrained_model"
    )
    stage1_policy = json.loads((stage1_path / "config.json").read_text())
    stage1_policy["fsq_path"] = str(relabeled_run / "FSQ.pt")
    stage1_policy["skill_code_space_id"] = "FSQ333_run"
    (stage1_path / "config.json").write_text(json.dumps(stage1_policy))
    (stage1_path / "train_config.json").write_text(
        json.dumps({"dataset": {"root": "/tmp/expired-stage1/skillvla"}})
    )

    settings = build_settings(config)

    assert settings["skill_dataset_dir"] == relabeled_run / "skillvla"
    assert settings["skill_latents_path"] == relabeled_run / "skill_latents.npz"


def test_stage2_eval_supports_new_grounded_skill_architecture(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    project = Path(config["project_root"])
    dataset_info = (
        project
        / "dataset_filtered/skillvla_dataset/libero_90_full_full/FSQ333_run/skillvla/meta/info.json"
    )
    dataset_info.write_text(json.dumps({"proprio_grounding": "episode_start_xyz"}))

    stage2_config = _stage2_config_path(config)
    stage2_policy = json.loads(stage2_config.read_text())
    stage2_policy.update(
        {
            "architecture_label": "arch0_2_skill_chunk",
            "architecture_revision": "cond_expert_state_adarms_v1",
            "proprio_grounding": "episode_start_xyz",
        }
    )
    stage2_config.write_text(json.dumps(stage2_policy))

    stage1_config = Path(stage2_policy["stage1_checkpoint_path"]) / "config.json"
    stage1_policy = json.loads(stage1_config.read_text())
    stage1_policy.update(
        {
            "architecture_label": "arch0_2_skill_chunk",
            "architecture_revision": "cond_expert_state_adarms_v1",
            "proprio_grounding": "episode_start_xyz",
        }
    )
    stage1_config.write_text(json.dumps(stage1_policy))

    panels = json.loads(build_settings(config)["models_json"])

    assert {panel["architecture_label"] for panel in panels} == {
        "arch0_2_skill_chunk"
    }
    assert {panel["proprio_grounding"] for panel in panels} == {
        "episode_start_xyz"
    }


def test_stage2_eval_rejects_grounding_mismatched_oracle_dataset(
    tmp_path: Path,
) -> None:
    config = _checkpoint_tree(tmp_path)
    project = Path(config["project_root"])
    stage2_config = _stage2_config_path(config)
    stage2_policy = json.loads(stage2_config.read_text())
    stage2_policy["proprio_grounding"] = "episode_start_xyz"
    stage2_config.write_text(json.dumps(stage2_policy))
    default_info = (
        project
        / "dataset_filtered/skillvla_dataset/libero_90_full_full/FSQ333_run/skillvla/meta/info.json"
    )
    default_info.write_text(json.dumps({"proprio_grounding": "episode_start_xyz"}))

    oracle_dataset = (
        project
        / "dataset_filtered/skillvla_dataset/libero_10_full_5/FSQ333_run/skillvla"
    )
    (oracle_dataset / "meta").mkdir(parents=True)
    (oracle_dataset / "meta/info.json").write_text("{}")
    config["oracle"]["skill_dataset_dir"] = str(oracle_dataset)

    with pytest.raises(ValueError, match="checkpoint/oracle proprio grounding mismatch"):
        build_settings(config)


def test_stage2_eval_single_mode_selection(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    config["modes"] = ["stage2"]

    settings = build_settings(config)
    panels = json.loads(settings["models_json"])

    assert settings["model_count"] == 1
    assert panels[0]["mode"] == "stage2"

    config["modes"] = ["everything"]
    with pytest.raises(ValueError, match="modes only accepts"):
        build_settings(config)


def test_stage2_eval_maps_compact_langgap_task_ids(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    source_dir = (
        Path(config["project_root"])
        / "dataset_filtered/skillvla_dataset/libero_90_full_full"
    )
    diagnostics = source_dir / "eval_init_states.diagnostics.json"
    diagnostics.write_text(
        json.dumps(
            {
                "matched": [
                    {
                        "suite_name": "langgap_ext",
                        "dataset_task_id": 0,
                        "suite_task_id": 41,
                    },
                    {
                        "suite_name": "langgap_ext",
                        "dataset_task_id": 1,
                        "suite_task_id": 44,
                    },
                ]
            }
        )
    )
    config["target_task"] = "langgap_ext"
    config["task_ids"] = [0, 1]

    settings = build_settings(config)

    assert settings["dataset_task_ids"] == "[0,1]"
    assert settings["task_ids"] == "[41,44]"
    assert settings["eval_expected_tasks"] == 2


def test_stage2_only_does_not_require_recorded_stage1_checkpoint(
    tmp_path: Path,
) -> None:
    config = _checkpoint_tree(tmp_path)
    config["modes"] = ["stage2"]
    stage2_config = json.loads(_stage2_config_path(config).read_text())
    stage1_path = Path(stage2_config["stage1_checkpoint_path"])
    (stage1_path / "config.json").unlink()

    panels = json.loads(build_settings(config)["models_json"])

    assert [panel["mode"] for panel in panels] == ["stage2"]


def test_prior_mode_still_requires_recorded_stage1_checkpoint(
    tmp_path: Path,
) -> None:
    config = _checkpoint_tree(tmp_path)
    stage2_config = json.loads(_stage2_config_path(config).read_text())
    stage1_path = Path(stage2_config["stage1_checkpoint_path"])
    (stage1_path / "config.json").unlink()

    with pytest.raises(FileNotFoundError, match="Stage-1 prior recorded"):
        build_settings(config)


def test_stage2_eval_supports_an_alternate_outputs_subdir(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    project = Path(config["project_root"])
    source = (
        project
        / "outputs_filtered/skillVLA_stage2"
        / config["model_dir"]
        / "checkpoints/last/pretrained_model"
    )
    target = (
        project
        / "outputs_filtered/skillVLA_FT"
        / config["model_dir"]
        / "checkpoints/last/pretrained_model"
    )
    target.mkdir(parents=True)
    for path in source.iterdir():
        (target / path.name).write_bytes(path.read_bytes())
    config["outputs_subdir"] = "skillVLA_FT"

    panels = json.loads(build_settings(config)["models_json"])

    stage2_panel = next(panel for panel in panels if panel["mode"] == "stage2")
    prior_panel = next(panel for panel in panels if panel["mode"] == "prior")
    assert "/skillVLA_FT/" in stage2_panel["policy_path"]
    assert "/skillVLA_stage1/" in prior_panel["policy_path"]


def test_stage2_eval_supports_per_model_outputs_root(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    project = Path(config["project_root"])
    source = (
        project
        / "outputs_filtered/skillVLA_stage2"
        / config["model_dir"]
        / "checkpoints/last/pretrained_model"
    )
    target = (
        project
        / "outputs/skillVLA_stage2"
        / config["model_dir"]
        / "checkpoints/last/pretrained_model"
    )
    target.mkdir(parents=True)
    for path in source.iterdir():
        (target / path.name).write_bytes(path.read_bytes())
    config["models"] = [
        {
            "model_dir": config.pop("model_dir"),
            "outputs_root": "outputs",
            "label": "alternate-root",
        }
    ]

    panels = json.loads(build_settings(config)["models_json"])

    stage2_panel = next(panel for panel in panels if panel["mode"] == "stage2")
    assert "/outputs/skillVLA_stage2/" in stage2_panel["policy_path"]
    # Auxiliary folders remain rooted at the snapshotted global outputs_root.
    assert "/outputs_filtered/skillVLA_terminator/" in stage2_panel[
        "external_terminator_model"
    ]


def test_stage2_eval_requires_external_terminator(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    del config["external_terminator_model"]

    with pytest.raises(ValueError, match="external_terminator_model"):
        build_settings(config)


def test_stage2_eval_resolves_concise_overlay_run_and_checkpoint(
    tmp_path: Path,
) -> None:
    config = _checkpoint_tree(tmp_path)
    project = Path(config["project_root"])
    expected_predictor = Path(config["external_predictor_model"])
    expected_terminator = Path(config["external_terminator_model"])
    config.update(
        {
            "external_predictor_model": "predictor_run",
            "external_predictor_checkpoint": "050000",
            "external_terminator_model": "terminator_run",
            "external_terminator_checkpoint": "020000",
        }
    )

    panels = json.loads(build_settings(config)["models_json"])

    assert all(
        panel["external_predictor_model"] == str(expected_predictor)
        for panel in panels
    )
    assert all(
        panel["external_terminator_model"] == str(expected_terminator)
        for panel in panels
    )
    assert expected_predictor.is_relative_to(project)
    assert expected_terminator.is_relative_to(project)


def test_stage2_eval_last_selects_latest_numeric_overlay_checkpoint(
    tmp_path: Path,
) -> None:
    config = _checkpoint_tree(tmp_path)
    source = Path(config["external_predictor_model"])
    latest = source.parents[2] / "checkpoints/100000/pretrained_model"
    _write_checkpoint(latest, json.loads((source / "config.json").read_text()))
    config.update(
        {
            "external_predictor_model": "predictor_run",
            "external_predictor_checkpoint": "last",
        }
    )

    panels = json.loads(build_settings(config)["models_json"])

    assert all(
        panel["external_predictor_model"] == str(latest) for panel in panels
    )


def test_stage2_eval_supports_original_fsq_terminator(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    config["external_terminator_model"] = "original"

    panels = json.loads(build_settings(config)["models_json"])

    assert [panel["advance_mode"] for panel in panels] == [
        "original",
        "original",
    ]
    assert all(not panel["external_terminator_model"] for panel in panels)


def test_stage2_eval_infers_sources_from_model_selectors(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    config.pop("skill_source")
    config["oracle"].pop("advance_mode")
    config["external_predictor_model"] = "gt"
    config["external_terminator_model"] = "original"

    panels = json.loads(build_settings(config)["models_json"])

    assert all(panel["skill_source"] == "gt" for panel in panels)
    assert all(panel["advance_mode"] == "original" for panel in panels)
    assert all(not panel["external_predictor_model"] for panel in panels)
    assert all(not panel["external_terminator_model"] for panel in panels)


def test_stage2_eval_folder_selectors_imply_external_sources(
    tmp_path: Path,
) -> None:
    config = _checkpoint_tree(tmp_path)
    config.pop("skill_source")
    config["oracle"].pop("advance_mode")

    panels = json.loads(build_settings(config)["models_json"])

    assert all(panel["skill_source"] == "external" for panel in panels)
    assert all(panel["advance_mode"] == "external" for panel in panels)


def test_stage2_eval_gt_terminator_selector_uses_gt_boundaries(
    tmp_path: Path,
) -> None:
    config = _checkpoint_tree(tmp_path)
    config.pop("skill_source")
    config["oracle"].pop("advance_mode")
    config["external_predictor_model"] = "gt"
    config["external_terminator_model"] = "gt"

    panels = json.loads(build_settings(config)["models_json"])

    assert all(panel["skill_source"] == "gt" for panel in panels)
    assert all(panel["advance_mode"] == "gt" for panel in panels)


def test_stage2_eval_prior_panel_requires_external_predictor(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    del config["external_predictor_model"]

    with pytest.raises(ValueError, match="external_predictor_model"):
        build_settings(config)


def test_stage2_eval_rejects_fsq_mismatched_terminator(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    project = Path(config["project_root"])
    other_fsq = project / "dataset_filtered/skillvla_dataset/libero_90_full_full/FSQ333_other/FSQ.pt"
    other_fsq.parent.mkdir(parents=True)
    other_fsq.write_bytes(b"different-fsq-checkpoint")
    terminator_config = Path(config["external_terminator_model"]) / "config.json"
    terminator = json.loads(terminator_config.read_text())
    terminator["fsq_path"] = str(other_fsq)
    terminator_config.write_text(json.dumps(terminator))

    with pytest.raises(ValueError, match="skill-code space mismatch"):
        build_settings(config)


def test_stage2_eval_accepts_aliased_code_space_for_identical_fsq(
    tmp_path: Path,
) -> None:
    config = _checkpoint_tree(tmp_path)
    terminator_config = Path(config["external_terminator_model"]) / "config.json"
    terminator = json.loads(terminator_config.read_text())
    terminator["skill_code_space_id"] = "FSQ333_same_weights_different_suffix"
    terminator_config.write_text(json.dumps(terminator))

    panels = json.loads(build_settings(config)["models_json"])

    assert len(panels) == 2


def test_stage2_eval_accepts_aliased_predictor_for_identical_fsq(
    tmp_path: Path,
) -> None:
    config = _checkpoint_tree(tmp_path)
    predictor_config = Path(config["external_predictor_model"]) / "config.json"
    predictor = json.loads(predictor_config.read_text())
    predictor["skill_code_space_id"] = "FSQ333_same_weights_different_suffix"
    predictor_config.write_text(json.dumps(predictor))

    panels = json.loads(build_settings(config)["models_json"])

    assert len(panels) == 2


def test_stage2_eval_rejects_aliased_predictor_for_different_fsq(
    tmp_path: Path,
) -> None:
    config = _checkpoint_tree(tmp_path)
    project = Path(config["project_root"])
    other_fsq = (
        project
        / "dataset_filtered/skillvla_dataset/libero_90_full_full/FSQ333_other/FSQ.pt"
    )
    other_fsq.parent.mkdir(parents=True)
    other_fsq.write_bytes(b"different-fsq-checkpoint")
    predictor_config = Path(config["external_predictor_model"]) / "config.json"
    predictor = json.loads(predictor_config.read_text())
    predictor["fsq_path"] = str(other_fsq)
    predictor_config.write_text(json.dumps(predictor))

    with pytest.raises(ValueError, match="predictor skill-code space mismatch"):
        build_settings(config)


def test_stage2_eval_exports_replan_and_terminator_variant(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    config["terminator"]["immediate_replan_on_skill_end"] = True
    config["terminator"]["variant"] = "state_image"

    settings = build_settings(config)
    panels = json.loads(settings["models_json"])

    assert settings["immediate_replan_on_skill_end"] is True
    assert settings["terminator_variant"] == "state_image"
    assert all(panel["terminator_variant"] == "state_image" for panel in panels)


def test_stage2_eval_exports_gt_termination_guard(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    config["terminator"] = {"gt_termination_min_fraction": 0.5}

    settings = build_settings(config)

    assert settings["gt_termination_min_fraction"] == 0.5


def test_stage2_eval_image_only_variant_requires_matching_source(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    config["terminator"]["variant"] = "image_only"

    with pytest.raises(ValueError, match="image_only"):
        build_settings(config)

    terminator_config = Path(config["external_terminator_model"]) / "config.json"
    terminator = json.loads(terminator_config.read_text())
    terminator["train_image_only_terminator"] = True
    terminator_config.write_text(json.dumps(terminator))

    settings = build_settings(config)
    panels = json.loads(settings["models_json"])
    assert all(panel["terminator_variant"] == "image_only" for panel in panels)

    config["terminator"]["variant"] = "sideways"
    with pytest.raises(ValueError, match="state_image|image_only"):
        build_settings(config)


def test_stage2_eval_rejects_legacy_policy_type(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path, policy_type="skill_vla")

    with pytest.raises(ValueError, match="skill_vla_stage2"):
        build_settings(config)


def test_stage2_episode_exact_requires_init_state_map(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    config["oracle"]["episode_exact"] = True

    with pytest.raises(FileNotFoundError, match="episode_exact=true"):
        build_settings(config)
