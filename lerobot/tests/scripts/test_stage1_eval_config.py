import json
import sys
from pathlib import Path

import pytest


_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/stage1_eval/src"
)
sys.path.insert(0, str(_SRC))
from stage1_eval_config import (  # noqa: E402
    _apply_gt_dataset,
    _checkpoint_contract,
    _effective_latent_source,
    _gt_dataset_override,
    _model_entries,
    _resolve_video_grid_columns,
)


def test_video_grid_columns_default_to_model_count() -> None:
    assert _resolve_video_grid_columns({}, model_count=3) == 3


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        (0, 3),
        (2, 2),
        (4, 4),
    ],
)
def test_video_grid_columns_support_auto_and_explicit_values(
    columns: int, expected: int
) -> None:
    assert _resolve_video_grid_columns(
        {"video": {"grid_columns": columns}}, model_count=3
    ) == expected


def test_video_grid_columns_reject_negative_values() -> None:
    with pytest.raises(ValueError, match="grid_columns"):
        _resolve_video_grid_columns(
            {"video": {"grid_columns": -1}}, model_count=3
        )


def _checkpoint(
    tmp_path: Path,
    *,
    label: str = "arch0",
    architecture: str = "cond_gemma",
    revision: str = "skillvla_real_v1",
) -> tuple[Path, Path]:
    project = tmp_path / "project"
    policy_path = project / "outputs/skillVLA_stage1/run/checkpoints/000100/pretrained_model"
    policy_path.mkdir(parents=True)
    dataset = project / "dataset/skillvla_dataset/source/run/skillvla"
    (dataset / "meta").mkdir(parents=True)
    info = {"proprio_grounding": "none"}
    if label.startswith(("arch8_1", "arch8_2", "arch9_1", "arch9_2", "arch10_1", "arch10_2", "arch11_1", "arch11_2", "arch12_1", "arch12_2")):
        focus_path = dataset.parent / "skill_focus_uv.npz"
        focus_path.touch()
        info["skill_focus_uv_path"] = str(focus_path)
    (dataset / "meta/info.json").write_text(json.dumps(info))
    dino = project / "models/dinov3-vitl16"
    dino.mkdir(parents=True)
    skill_flow_enabled = label not in {"arch0", "arch1", "arch2", "arch3", "arch4", "arch5", "arch6", "arch7", "arch8_1", "arch8_2", "arch9_1", "arch9_2", "arch10_1", "arch10_2", "arch11_1", "arch11_2", "arch12_1", "arch12_2"}
    policy = {
        "type": "skill_expert",
        "architecture": architecture,
        "architecture_label": label,
        "architecture_revision": revision,
        "vision_conditioning_mode": (
            "layerwise_cond_bottleneck_cross_attention"
            if label.startswith(("arch3", "arch4", "arch5", "arch6", "arch7", "arch8_1", "arch8_2", "arch9_1", "arch9_2", "arch10_1", "arch10_2", "arch11_1", "arch11_2", "arch12_1", "arch12_2"))
            else (
                "fixed_bottleneck_cross_attention"
                if label == "arch1" or label.startswith(("arch1_", "arch2"))
                else "interleaved_cross_attention"
            )
        ),
        "conditioning_route": "state_cond",
        "action_loss_mode": "flow",
        "skill_flow_enabled": skill_flow_enabled,
        "skill_flow_target": (
            "extended_chunk" if label.endswith("_skill_chunk") else "canonical"
        ),
        "skill_flow_state_conditioned": False,
        "proprio_grounding": "none",
        "dino_model_path": "models/dinov3-vitl16",
        "train_skill_predictor": False,
        "training_skill_source": "gt",
        "train_terminator": False,
        "fsq_path": "dataset/skillvla_dataset/source/run/FSQ.pt",
    }
    (policy_path / "config.json").write_text(json.dumps(policy))
    (policy_path / "train_config.json").write_text(
        json.dumps(
            {"dataset": {"root": "dataset/skillvla_dataset/source/run/skillvla"}}
        )
    )
    for name in (
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        (policy_path / name).touch()
    return project, policy_path


@pytest.mark.parametrize(
    ("label", "architecture", "revision"),
    [
        ("arch0", "cond_gemma", "skillvla_real_v1"),
        ("arch0_skill", "cond_gemma", "skillvla_real_v1"),
        ("arch0_skill_chunk", "cond_gemma", "skillvla_real_v1"),
        ("arch1", "fixed_visual_bottleneck", "fixed_visual_bottleneck_v1"),
        ("arch1_skill", "fixed_visual_bottleneck", "fixed_visual_bottleneck_v1"),
        (
            "arch1_skill_chunk",
            "fixed_visual_bottleneck",
            "fixed_visual_bottleneck_v1",
        ),
        ("arch2", "fixed_visual_bottleneck", "late_visual_bottleneck_v1"),
        (
            "arch2_skill",
            "fixed_visual_bottleneck",
            "late_visual_bottleneck_v1",
        ),
        (
            "arch2_skill_chunk",
            "fixed_visual_bottleneck",
            "late_visual_bottleneck_v1",
        ),
        ("arch3", "layerwise_cond_bottleneck", "layerwise_cond_bottleneck_v1"),
        (
            "arch3_skill",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_v1",
        ),
        (
            "arch3_skill_chunk",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_v1",
        ),
        (
            "arch4",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_core_exit_v1",
        ),
        (
            "arch4_skill",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_core_exit_v1",
        ),
        (
            "arch4_skill_chunk",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_core_exit_v1",
        ),
        (
            "arch5",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_core_exit_uv_v1",
        ),
        (
            "arch5_skill",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_core_exit_uv_v1",
        ),
        (
            "arch5_skill_chunk",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_core_exit_uv_v1",
        ),
        (
            "arch6",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_core_exit_latent_uv_v1",
        ),
        (
            "arch6_skill",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_core_exit_latent_uv_v1",
        ),
        (
            "arch6_skill_chunk",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_core_exit_latent_uv_v1",
        ),
        (
            "arch7",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_core_exit_latent_xyz_v1",
        ),
        (
            "arch7_skill",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_core_exit_latent_xyz_v1",
        ),
        (
            "arch7_skill_chunk",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_core_exit_latent_xyz_v1",
        ),
        (
            "arch8_1",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_uv_cond_xyz_v1",
        ),
        (
            "arch8_1_skill",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_uv_cond_xyz_v1",
        ),
        (
            "arch8_1_skill_chunk",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_uv_cond_xyz_v1",
        ),
        (
            "arch8_2",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_uv_cond_xyz_termination_v1",
        ),
        (
            "arch8_2_skill",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_uv_cond_xyz_termination_v1",
        ),
        (
            "arch8_2_skill_chunk",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_uv_cond_xyz_termination_v1",
        ),
        (
            "arch8_2_skill",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_uv_cond_xyz_cond_termination_v2",
        ),
        (
            "arch9_1",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_wrist_skill_end_pose_v1",
        ),
        (
            "arch9_1_skill",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_wrist_skill_end_pose_v1",
        ),
        (
            "arch9_1_skill_chunk",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_wrist_skill_end_pose_v1",
        ),
        (
            "arch9_2",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_wrist_skill_end_pose_termination_v1",
        ),
        (
            "arch9_2_skill",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_wrist_skill_end_pose_termination_v1",
        ),
        (
            "arch9_2_skill_chunk",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_wrist_skill_end_pose_termination_v1",
        ),
        (
            "arch9_2_skill",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_wrist_skill_end_pose_cond_termination_v2",
        ),
        (
            "arch10_1_skill",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_wrist_proprio_expert_end_pose_v1",
        ),
        (
            "arch10_2_skill",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_wrist_proprio_expert_end_pose_cond_termination_v1",
        ),
        (
            "arch11_1_skill",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_end_pose_v1",
        ),
        (
            "arch11_2_skill",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_end_pose_cond_termination_v1",
        ),
        (
            "arch12_1_skill",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_v1",
        ),
        (
            "arch12_2_skill",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_cond_termination_v1",
        ),
    ],
)
def test_checkpoint_contract_accepts_retained_modes(
    tmp_path: Path, label: str, architecture: str, revision: str
) -> None:
    project, policy_path = _checkpoint(
        tmp_path,
        label=label,
        architecture=architecture,
        revision=revision,
    )
    contract = _checkpoint_contract(policy_path, project)

    assert contract["architecture"] == architecture
    assert contract["architecture_revision"] == revision
    assert contract["architecture_label"] == label
    assert contract["conditioning_route"] == "state_cond"
    assert contract["vision_conditioning_mode"] == (
        "layerwise_cond_bottleneck_cross_attention"
        if label.startswith(("arch3", "arch4", "arch5", "arch6", "arch7", "arch8_1", "arch8_2", "arch9_1", "arch9_2", "arch10_1", "arch10_2", "arch11_1", "arch11_2", "arch12_1", "arch12_2"))
        else (
            "fixed_bottleneck_cross_attention"
            if label == "arch1" or label.startswith(("arch1_", "arch2"))
            else "interleaved_cross_attention"
        )
    )


def test_checkpoint_contract_rejects_removed_modes(tmp_path: Path) -> None:
    project, policy_path = _checkpoint(tmp_path, label="removed_mode")
    with pytest.raises(ValueError, match="supports only arch0"):
        _checkpoint_contract(policy_path, project)


def test_checkpoint_contract_rejects_old_model_implementation(tmp_path: Path) -> None:
    project, policy_path = _checkpoint(
        tmp_path,
        label="removed_mode",
        architecture="removed_architecture",
        revision="removed_revision",
    )
    with pytest.raises(ValueError, match="supports only arch0"):
        _checkpoint_contract(policy_path, project)


def test_checkpoint_contract_rejects_arch1_with_arch0_implementation(
    tmp_path: Path,
) -> None:
    project, policy_path = _checkpoint(tmp_path, label="arch1")
    with pytest.raises(ValueError, match="requires architecture='fixed_visual_bottleneck'"):
        _checkpoint_contract(policy_path, project)


def test_checkpoint_label_must_match_auxiliary_objective(tmp_path: Path) -> None:
    project, policy_path = _checkpoint(tmp_path, label="arch0_skill")
    config_path = policy_path / "config.json"
    config = json.loads(config_path.read_text())
    config["skill_flow_enabled"] = False
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="skill_flow_enabled=True"):
        _checkpoint_contract(policy_path, project)


def test_foveated_checkpoint_requires_portable_gt_focus_artifact(
    tmp_path: Path,
) -> None:
    project, policy_path = _checkpoint(
        tmp_path,
        label="arch1",
        architecture="fixed_visual_bottleneck",
        revision="fixed_visual_bottleneck_v1",
    )
    config_path = policy_path / "config.json"
    config = json.loads(config_path.read_text())
    config["foveated_vision_enabled"] = True
    config_path.write_text(json.dumps(config))
    dataset = project / "dataset/skillvla_dataset/source/run/skillvla"
    info_path = dataset / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["skill_focus_uv_path"] = "/old/server/run/skill_focus_uv.npz"
    info_path.write_text(json.dumps(info))

    with pytest.raises(FileNotFoundError, match="focus artifact"):
        _checkpoint_contract(policy_path, project)

    portable = dataset.parent / "skill_focus_uv.npz"
    portable.touch()
    contract = _checkpoint_contract(policy_path, project)

    assert contract["foveated_vision_enabled"] is True
    assert contract["focus_uv_path"] == portable


def test_model_defaults_are_inherited_and_model_values_override_them() -> None:
    entries = _model_entries(
        {
            "terminator": {"end_threshold": 0.3},
            "model_defaults": {
                "checkpoint": "015000",
                "skill_source": "gt",
                "focus_source": "gt",
                "advance_mode": "external",
                "external_skill_model": "outputs/shared/ckpt",
            },
            "models": [
                {"model_dir": "first", "label": "first"},
                {
                    "model_dir": "second",
                    "label": "second",
                    "checkpoint": "030000",
                    "skill_source": "own",
                    "focus_source": "predictor",
                    "advance_mode": "gt",
                    "end_threshold": 0.5,
                },
            ],
        }
    )

    assert entries[0]["checkpoint"] == "015000"
    assert entries[0]["skill_source"] == "gt"
    assert entries[0]["focus_source"] == "gt"
    assert entries[0]["advance_mode"] == "external"
    assert entries[0]["end_threshold"] == 0.3
    assert entries[1]["checkpoint"] == "030000"
    assert entries[1]["skill_source"] == "own"
    assert entries[1]["focus_source"] == "predictor"
    assert entries[1]["advance_mode"] == "gt"
    assert entries[1]["end_threshold"] == 0.5
    assert "previous_checkpoint" not in entries[0]


def test_model_defaults_end_threshold_overrides_global_terminator() -> None:
    entries = _model_entries(
        {
            "terminator": {"end_threshold": 0.3},
            "model_defaults": {"end_threshold": 0.4},
            "models": [{"model_dir": "first"}],
        }
    )

    assert entries[0]["end_threshold"] == 0.4


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_model_end_threshold_rejects_values_outside_probability_range(
    threshold: float,
) -> None:
    with pytest.raises(ValueError, match="models\\[\\].end_threshold"):
        _model_entries(
            {
                "models": [
                    {"model_dir": "first", "end_threshold": threshold}
                ]
            }
        )


def test_previous_historical_selector_is_removed() -> None:
    with pytest.raises(ValueError, match="Unknown model_defaults"):
        _model_entries(
            {
                "model_defaults": {"previous": True},
                "models": [{"model_dir": "old", "label": "old"}],
            }
        )


def test_oracle_latent_is_ignored_for_latent_free_checkpoint() -> None:
    assert (
        _effective_latent_source(
            "oracle", {"skill_flow_latent_best_of_n_enabled": False}
        )
        == "random"
    )
    assert (
        _effective_latent_source(
            "oracle", {"skill_flow_latent_best_of_n_enabled": True}
        )
        == "oracle"
    )


def _gt_run(project: Path, *, fsq=("fsq_run", "2000"), grounding="none", levels=(3, 3, 3), focus=False) -> Path:
    run_dir = project / "dataset/skillvla_dataset/libero_10_full_1/FSQ333_ft"
    (run_dir / "skillvla/meta").mkdir(parents=True)
    (run_dir / "skillvla/meta/info.json").write_text(
        json.dumps({"skill_fsq_levels": list(levels), "proprio_grounding": grounding})
    )
    if fsq is not None:
        (run_dir / "fsq_source.json").write_text(
            json.dumps({"source_fsq_run_name": fsq[0], "source_fsq_checkpoint": fsq[1]})
        )
    if focus:
        (run_dir / "skill_focus_uv.npz").touch()
    return run_dir


def _contract_with_fsq_source(tmp_path: Path, **kwargs):
    project, policy_path = _checkpoint(tmp_path, **kwargs)
    (project / "dataset/skillvla_dataset/source/run/fsq_source.json").write_text(
        json.dumps({"source_fsq_run_name": "fsq_run", "source_fsq_checkpoint": "2000"})
    )
    config_path = policy_path / "config.json"
    config = json.loads(config_path.read_text())
    config["skill_fsq_levels"] = [3, 3, 3]
    config_path.write_text(json.dumps(config))
    return project, policy_path, _checkpoint_contract(policy_path, project)


def test_gt_dataset_is_optional_and_blank_keeps_the_training_dataset(tmp_path: Path) -> None:
    base = {"project_root": str(tmp_path), "dataset_root": "dataset"}
    assert _gt_dataset_override(base, tmp_path) is None
    assert _gt_dataset_override({**base, "gt_dataset": {"source": "", "run": ""}}, tmp_path) is None
    assert _gt_dataset_override(
        {**base, "gt_dataset": {"source": "libero_10_full_1", "run": "FSQ333_ft"}}, tmp_path
    ) == tmp_path / "dataset/skillvla_dataset/libero_10_full_1/FSQ333_ft"
    with pytest.raises(ValueError, match="both source and run"):
        _gt_dataset_override({**base, "gt_dataset": {"source": "only"}}, tmp_path)
    with pytest.raises(ValueError, match="must be a mapping"):
        _gt_dataset_override({**base, "gt_dataset": {"source": "a", "run": "b", "typo": 1}}, tmp_path)


def test_gt_dataset_moves_only_the_demonstration_inputs(tmp_path: Path) -> None:
    project, policy_path, contract = _contract_with_fsq_source(tmp_path)
    run_dir = _gt_run(project)

    moved = _apply_gt_dataset(contract, run_dir, policy_path=policy_path)

    assert moved["skill_dataset_dir"] == run_dir / "skillvla"
    assert moved["eval_init_states_path"] == run_dir.parent / "eval_init_states.npz"
    assert moved["skill_latents_path"] == run_dir / "skill_latents.npz"
    assert moved["raw_dataset_dir"] == project / "dataset/libero_10_full_1"
    for kept in ("policy", "fsq_path", "dino_model_path", "tokenizer_path", "proprio_grounding"):
        assert moved[kept] == contract[kept]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fsq": ("other_fsq", "2000")}, "FSQ model"),
        ({"grounding": "episode_start_xyz"}, "proprio_grounding"),
        ({"levels": (3, 3, 3, 3)}, "skill_fsq_levels"),
    ],
)
def test_gt_dataset_rejects_incompatible_runs(tmp_path: Path, kwargs: dict, message: str) -> None:
    project, policy_path, contract = _contract_with_fsq_source(tmp_path)
    with pytest.raises(ValueError, match=message):
        _apply_gt_dataset(contract, _gt_run(project, **kwargs), policy_path=policy_path)


def test_gt_dataset_fsq_verification_can_be_waived(tmp_path: Path) -> None:
    project, policy_path, contract = _contract_with_fsq_source(tmp_path)
    run_dir = _gt_run(project, fsq=None)
    with pytest.raises(FileNotFoundError, match="fsq_source.json"):
        _apply_gt_dataset(contract, run_dir, policy_path=policy_path)
    assert _apply_gt_dataset(contract, run_dir, policy_path=policy_path, verify_fsq_source=False)


def test_gt_dataset_must_provide_focus_uv_when_the_checkpoint_uses_it(tmp_path: Path) -> None:
    project, policy_path, contract = _contract_with_fsq_source(
        tmp_path,
        label="arch8_1",
        architecture="layerwise_cond_bottleneck",
        revision="layerwise_cond_bottleneck_uv_cond_xyz_v1",
    )
    with pytest.raises(FileNotFoundError, match="skill_focus_uv.npz"):
        _apply_gt_dataset(contract, _gt_run(project), policy_path=policy_path)
    (project / "dataset/skillvla_dataset/libero_10_full_1/FSQ333_ft/skill_focus_uv.npz").touch()
    run_dir = project / "dataset/skillvla_dataset/libero_10_full_1/FSQ333_ft"
    assert _apply_gt_dataset(contract, run_dir, policy_path=policy_path)["focus_uv_path"] == (
        run_dir / "skill_focus_uv.npz"
    )


def test_run_lookup_also_searches_newtask_ft_outputs(tmp_path: Path) -> None:
    sys.path.insert(0, str(_SRC.parents[2] / "train_skills" / "src"))
    from train_skills_config import stage1_run_dir, stage1_run_dirs

    outputs = tmp_path / "outputs"
    for component in ("VSA", "Predictor", "Terminator"):
        dirs = stage1_run_dirs(outputs, "run", component)
        assert dirs[0] == outputs / "skillVLA_stage1" / component / "run"
        assert dirs[1] == outputs / "skillVLA_NewTask_FT" / component / "run"
    # Nothing exists → the legacy path is still what diagnostics report.
    assert stage1_run_dir(outputs, "run", "VSA") == outputs / "skillVLA_stage1/run"
    ft_run = outputs / "skillVLA_NewTask_FT/Predictor/ft_run"
    ft_run.mkdir(parents=True)
    assert stage1_run_dir(outputs, "ft_run", "Predictor") == ft_run


def test_eval_outputs_follow_the_work_dir(tmp_path: Path, monkeypatch) -> None:
    import stage1_eval_config as module

    source = Path(module.__file__).read_text()
    assert 'os.environ.get("STAGE1_EVAL_WORK_DIR"' in source
    shipped = _SRC.parents[1] / "NewTask_FT/eval/ft_eval_config.yaml"
    config = module.load_config(shipped)
    assert config["target_task"] == "libero_10" and config["gt_dataset"]["source"]
    assert module._model_entries(config)
