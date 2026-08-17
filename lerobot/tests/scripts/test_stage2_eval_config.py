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


def test_stage2_eval_requires_external_terminator(tmp_path: Path) -> None:
    config = _checkpoint_tree(tmp_path)
    del config["external_terminator_model"]

    with pytest.raises(ValueError, match="external_terminator_model"):
        build_settings(config)


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
    other_fsq.touch()
    terminator_config = Path(config["external_terminator_model"]) / "config.json"
    terminator = json.loads(terminator_config.read_text())
    terminator["fsq_path"] = str(other_fsq)
    terminator_config.write_text(json.dumps(terminator))

    with pytest.raises(ValueError, match="FSQ run does not match"):
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
