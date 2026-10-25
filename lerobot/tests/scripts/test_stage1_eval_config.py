import json
import sys
from pathlib import Path

import pytest


_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/PT/eval/src"
)
sys.path.insert(0, str(_SRC))
from stage1_eval_config import (  # noqa: E402
    FULLSKILL_VIS_REVISIONS,
    FULLSKILL_VIS_LEGACY_REVISIONS,
    FULLSKILL_VIS_SPLIT_REVISIONS,
    BSPLINE_REVISIONS,
    BSPLINE_LEGACY_REVISIONS,
    _apply_gt_dataset,
    _relocate_project_path,
    _checkpoint_contract,
    _effective_latent_source,
    _gt_dataset_override,
    _model_entries,
    _resolve_joint_zero_predictor_checkpoint,
    _resolve_vsa_eval_checkpoint,
    _resolve_video_grid_columns,
    load_config,
)


def test_video_grid_columns_default_to_model_count() -> None:
    assert _resolve_video_grid_columns({}, model_count=3) == 3


def test_cond_goal_settings_survive_model_and_checkpoint_expansion():
    entries = _model_entries({
        "model_defaults": {"checkpoint": ["010000", "050000"]},
        "models": [
            {"model_dir": "both3", "label": "normal"},
            {"model_dir": "both3", "label": "off", "cond_goal_mode": "off"},
            {"model_dir": "both3", "label": "offset", "cond_goal_mode": "offset",
             "cond_goal_offset_m": [.05, 0, 0]},
        ],
    })
    assert len(entries) == 6
    assert [entry["cond_goal_mode"] for entry in entries] == ["normal", "off", "offset"] * 2
    assert entries[2]["cond_goal_offset_m"] == [.05, 0, 0]
    assert entries[5]["cond_goal_offset_m"] == [.05, 0, 0]


def test_task01_cond_xyz_diagnostic_keeps_paired_inputs_and_two_gpu_plan():
    import yaml
    path = (Path(__file__).resolve().parents[2]
            / "examples/libero/configs/train_skillVLA/FT/eval/both3_task01_cond_xyz_diagnostic.yaml")
    config = yaml.safe_load(path.read_text())
    assert config["task_ids"] == [0, 1]
    assert config["eval_num_gpus"] == 2 and config["eval_max_workers_per_gpu"] == 1
    assert config["partial_success"] and config["n_episodes"] == 10
    models = config["models"]
    assert len(models) == 3
    baseline, _, shifted = models
    for model in models:
        for key in ("vsa_checkpoint", "predictor_dir", "terminator_dir", "terminator_checkpoint", "gt_dataset"):
            assert model[key] == baseline[key]
    assert shifted["vsa_dir"] == baseline["vsa_dir"]
    assert shifted["cond_goal_mode"] == "offset" and shifted["cond_goal_offset_m"] == [.05, 0, 0]
    assert all(model.get("cond_goal_mode") != "off" for model in models)


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


@pytest.mark.parametrize("variant", range(1, 7))
@pytest.mark.parametrize("family", ["residual", "direct"])
@pytest.mark.parametrize("dgd", [False, True])
@pytest.mark.parametrize("bridge_layers", [1, 9])
def test_window_checkpoint_contract(tmp_path, variant, family, dgd, bridge_layers):
    project, checkpoint = _checkpoint(tmp_path, label="both_3", architecture="layerwise_cond_bottleneck",
        revision="layerwise_cond_bottleneck_both_cond_skill_end_pose_expert_skill_start_end_bridge_proprio_dual_patch_dedicated_align_v1")
    # _checkpoint returns (project, policy_path), like the original tests below.
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    label = f"{family}_patch_{variant}" + ("_dGD" if dgd else "")
    config.update(architecture_label=label,
        visual_bridge_last_n_layers=bridge_layers,
        architecture_revision=f"layerwise_cond_bottleneck_{family}_patch_v1",
        skill_flow_target="residual_window", skill_flow_state_conditioned=True,
        skill_flow_max_length=4 * config.get("chunk_size", 10))
    path.write_text(json.dumps(config))
    contract = _checkpoint_contract(checkpoint, project)
    assert contract["architecture_label"] == label
    assert contract["visual_bridge_last_n_layers"] == bridge_layers
    if label.startswith("uv_"):
        assert contract["needs_focus_uv"] and not contract["needs_end_pose"]


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
    if label in BSPLINE_REVISIONS:
        focus_path = dataset.parent / "skill_focus_uv.npz"
        focus_path.touch()
        info["skill_focus_uv_path"] = str(focus_path)
    if label.startswith(("arch8_1", "arch8_2", "arch9_1", "arch9_2", "arch10_1", "arch10_2", "arch11_1", "arch11_2", "arch12_1", "arch12_2", "wristonly_", "both_", "fullskill_")):
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
            if label.startswith(("arch3", "arch4", "arch5", "arch6", "arch7", "arch8_1", "arch8_2", "arch9_1", "arch9_2", "arch10_1", "arch10_2", "arch11_1", "arch11_2", "arch12_1", "arch12_2", "wristonly_", "both_", "fullskill_"))
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
    if label in BSPLINE_REVISIONS:
        policy["vision_conditioning_mode"] = "layerwise_cond_bottleneck_cross_attention"
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
        (
            "wristonly_1",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_wrist_patch_align_v1",
        ),
        (
            "both_1",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_both_cond_skill_end_pose_expert_skill_dual_patch_align_v1",
        ),
        (
            "wristonly_2",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_wrist_patch_dedicated_align_v1",
        ),
        (
            "both_2",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_both_cond_skill_end_pose_expert_skill_dual_patch_dedicated_align_v1",
        ),
        (
            "both_3",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_both_cond_skill_end_pose_expert_skill_start_end_bridge_proprio_dual_patch_dedicated_align_v1",
        ),
        ("fullskill_1", "layerwise_cond_bottleneck", "layerwise_cond_bottleneck_fullskill_1_chunk_end_dual_patch_v1"),
        ("fullskill_2", "layerwise_cond_bottleneck", "layerwise_cond_bottleneck_fullskill_2_no_visual_aux_v1"),
        *[(label, "layerwise_cond_bottleneck", revision) for label, revision in FULLSKILL_VIS_REVISIONS.items()],
        *[(label, "layerwise_cond_bottleneck", revision) for label, revision in FULLSKILL_VIS_LEGACY_REVISIONS.items() if label != "fullskill_vis_1"],
        (
            "both_4",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_both_cond_skill_expert_skill_start_end_bridge_proprio_dual_patch_dedicated_align_v1",
        ),
        (
            "both_5",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_both_cond_skill_end_pose_expert_skill_start_end_bridge_proprio_dual_patch_align_v1",
        ),
        (
            "wristonly_lit_1",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_wrist_cond_skill_expert_skill_start_end_bridge_proprio_chunk_end_pose_v1",
        ),
        (
            "both_lit_1",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_both_cond_skill_expert_skill_start_end_bridge_proprio_chunk_end_pose_v1",
        ),
        (
            "wristonly_lit_2",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_wrist_cond_skill_expert_skill_start_end_chunk_end_pose_v1",
        ),
        (
            "both_lit_2",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_both_cond_skill_expert_skill_start_end_chunk_end_pose_v1",
        ),
        (
            "wristonly_lit_3",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_wrist_cond_skill_expert_skill_end_chunk_end_pose_v1",
        ),
        (
            "both_lit_3",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_both_cond_skill_expert_skill_end_chunk_end_pose_v1",
        ),
        (
            "wristonly_lit_4",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_start_end_bridge_proprio_chunk_end_pose_v1",
        ),
        (
            "both_lit_4",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_both_cond_skill_end_pose_expert_skill_start_end_bridge_proprio_chunk_end_pose_v1",
        ),
        (
            "wristonly_lit_5",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_start_end_chunk_end_pose_v1",
        ),
        (
            "both_lit_5",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_both_cond_skill_end_pose_expert_skill_start_end_chunk_end_pose_v1",
        ),
        (
            "both_lit_6",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_both_cond_proprio_expert_skill_start_end_chunk_end_pose_v1",
        ),
        (
            "both_lit_7",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_both_cond_skill_end_pose_expert_skill_start_end_dual_patch_dedicated_align_v1",
        ),
        (
            "both_lit_4_dedicated",
            "layerwise_cond_bottleneck",
            "layerwise_cond_bottleneck_both_cond_skill_end_pose_expert_skill_start_end_bridge_proprio_dedicated_chunk_end_pose_v1",
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
        if label.startswith(("arch3", "arch4", "arch5", "arch6", "arch7", "arch8_1", "arch8_2", "arch9_1", "arch9_2", "arch10_1", "arch10_2", "arch11_1", "arch11_2", "arch12_1", "arch12_2", "wristonly_", "both_", "fullskill_"))
        else (
            "fixed_bottleneck_cross_attention"
            if label == "arch1" or label.startswith(("arch1_", "arch2"))
            else "interleaved_cross_attention"
        )
    )


@pytest.mark.parametrize("label,revision", [*BSPLINE_REVISIONS.items(), *BSPLINE_LEGACY_REVISIONS.items()])
@pytest.mark.parametrize("bridge_layers", [1, 9, 17])
def test_bspline_checkpoint_contract_accepts_plan_without_canonical_flow(tmp_path, label, revision, bridge_layers):
    project, path = _checkpoint(tmp_path, label=label, architecture="layerwise_cond_bottleneck", revision=revision)
    config = json.loads((path / "config.json").read_text())
    config.update(skill_flow_enabled=False, skill_flow_max_length=0, visual_bridge_last_n_layers=bridge_layers,
                  bspline_tokens=30, bspline_loss_weight=1.)
    (path / "config.json").write_text(json.dumps(config))
    if revision == BSPLINE_LEGACY_REVISIONS.get(label) and bridge_layers != 9:
        with pytest.raises(ValueError, match="Legacy Bspline"):
            _checkpoint_contract(path, project)
        return
    contract = _checkpoint_contract(path, project)
    assert contract["architecture_label"] == label
    assert contract["visual_bridge_last_n_layers"] == bridge_layers


_UV_CURRENT_LABELS = [
    "uv_fullskill_bspline_3_lastlayer_lastvis_current" + suffix
    for suffix in ("", "_preskill", "_diffaux", "_preskill_diffaux")
]


def _goal_terminator_eval_config(tmp_path, label, *, goal_xyz=True, advance_mode="external"):
    import stage1_eval_config as module

    uv = label.startswith("uv_")
    project, path = _checkpoint(
        tmp_path, label=label,
        architecture="layerwise_cond_bottleneck" if uv else "cond_gemma",
        revision=BSPLINE_REVISIONS[label] if uv else "skillvla_real_v1",
    )
    policy = json.loads((path / "config.json").read_text())
    policy.update(
        skill_flow_enabled=False, skill_flow_max_length=0, bspline_tokens=30,
        skill_fsq_levels=[3, 3, 3], skill_code_space_id="test-space",
        train_terminator=advance_mode == "own", terminator_goal_xyz=goal_xyz,
    )
    (path / "config.json").write_text(json.dumps(policy))
    (project / policy["fsq_path"]).touch()
    (project / "dataset/skillvla_dataset/source/eval_init_states.npz").touch()
    term = project / "external_term/pretrained_model"
    term.mkdir(parents=True)
    (term / "config.json").write_text(json.dumps({
        "type": "skill_aux", "train_terminator": True,
        "skill_fsq_levels": [3, 3, 3], "skill_code_space_id": "test-space",
        "terminator_goal_xyz": goal_xyz,
    }))
    (term / "model.safetensors").touch()
    return module, {
        "project_root": str(project), "outputs_root": "outputs",
        "models": [{
            "model_dir": "run", "checkpoint": "000100", "label": "test",
            "skill_source": "gt", "focus_source": "gt", "pose_source": "gt",
            "advance_mode": advance_mode, "external_terminator_model": str(term),
        }],
        "oracle": {"episode_exact": True}, "output_name": "goal_test",
    }


@pytest.mark.parametrize("advance", ["external", "own", "gt"])
def test_uv_terminator_needs_focus_without_changing_xyz_vsa_contract(tmp_path, advance):
    module, config = _goal_terminator_eval_config(tmp_path, "both_3", goal_xyz=False, advance_mode=advance)
    entry = config["models"][0]
    external = Path(entry["external_terminator_model"]) / "config.json"
    term = json.loads(external.read_text())
    term["terminator_goal_uv"] = True
    external.write_text(json.dumps(term))
    policy_path = Path(config["project_root"]) / "outputs/skillVLA_stage1/run/checkpoints/000100/pretrained_model/config.json"
    policy = json.loads(policy_path.read_text())
    policy["terminator_goal_uv"] = True
    policy.update(
        architecture="layerwise_cond_bottleneck",
        architecture_revision="layerwise_cond_bottleneck_both_cond_skill_end_pose_expert_skill_start_end_bridge_proprio_dual_patch_dedicated_align_v1",
        skill_flow_enabled=True, skill_flow_max_length=10,
    )
    policy_path.write_text(json.dumps(policy))
    panel, = json.loads(module.build_settings(config)["models_json"])
    assert panel["terminator_needs_goal_uv"] is (advance != "gt")
    assert not panel["needs_focus_uv"]  # the XYZ VSA itself must not acquire UV inputs
    assert panel["needs_end_pose"]
    if advance != "gt":
        config["oracle"]["episode_exact"] = False
        with pytest.raises(ValueError, match="episode_exact"):
            module.build_settings(config)


@pytest.mark.parametrize("label", _UV_CURRENT_LABELS)
@pytest.mark.parametrize("goal_xyz", [False, True])
@pytest.mark.parametrize("advance_mode", ["external", "own", "gt"])
def test_uv_resolver_keeps_terminator_xyz_separate_from_vsa(tmp_path, label, goal_xyz, advance_mode):
    module, config = _goal_terminator_eval_config(
        tmp_path, label, goal_xyz=goal_xyz, advance_mode=advance_mode,
    )
    panel, = json.loads(module.build_settings(config)["models_json"])
    assert panel["needs_focus_uv"] and not panel["needs_end_pose"]
    assert panel["terminator_needs_goal_xyz"] is (goal_xyz and advance_mode != "gt")
    assert panel["pose_source"] == panel["focus_source"] == "gt"


def test_terminator_only_gt_goal_requires_exact_episode_alignment(tmp_path):
    module, config = _goal_terminator_eval_config(tmp_path, "arch0")
    config["oracle"]["episode_exact"] = False
    with pytest.raises(ValueError, match="GT end-pose.*episode_exact"):
        module.build_settings(config)


def test_uv_terminator_goal_rejects_a_predictor_without_xyz(tmp_path, monkeypatch):
    module, config = _goal_terminator_eval_config(tmp_path, _UV_CURRENT_LABELS[0])
    row = config["models"][0]
    row["pose_source"] = "predictor"
    row["external_predictor_model"] = row["external_terminator_model"]
    monkeypatch.setattr(module, "_external_predictor_contract", lambda *args, **kwargs: {
        "tokenizer_path": Path(config["project_root"]), "end_state_mode": "off",
        "focus_uv_enabled": True,
    })
    with pytest.raises(ValueError, match="end-state head"):
        module.build_settings(config)


@pytest.mark.parametrize("label", FULLSKILL_VIS_REVISIONS)
@pytest.mark.parametrize("bridge_layers", [1, 9, 17])
def test_fullskill_visual_checkpoint_contract_preserves_split(tmp_path, label, bridge_layers):
    revision = (
        FULLSKILL_VIS_REVISIONS[label] if bridge_layers == 1 else
        FULLSKILL_VIS_SPLIT_REVISIONS.get(label, FULLSKILL_VIS_REVISIONS[label])
    )
    project, path = _checkpoint(tmp_path, label=label, architecture="layerwise_cond_bottleneck", revision=revision)
    config = json.loads((path / "config.json").read_text())
    config["visual_bridge_last_n_layers"] = bridge_layers
    (path / "config.json").write_text(json.dumps(config))
    contract = _checkpoint_contract(path, project)
    assert contract["architecture_label"] == label
    assert contract["architecture_revision"] == revision
    assert contract["visual_bridge_last_n_layers"] == bridge_layers
    if label in FULLSKILL_VIS_LEGACY_REVISIONS and bridge_layers != 1:
        config["architecture_revision"] = FULLSKILL_VIS_REVISIONS[label]
        (path / "config.json").write_text(json.dumps(config))
        with pytest.raises(ValueError, match="Legacy Fullskill visual"):
            _checkpoint_contract(path, project)


def test_checkpoint_contract_ignores_existing_node_local_dataset_copy(
    tmp_path: Path,
) -> None:
    project, policy_path = _checkpoint(tmp_path)
    portable_dataset = project / "dataset/skillvla_dataset/source/run/skillvla"
    staged_dataset = tmp_path / "dohyeon_skillvla_42997" / "skillvla"
    (staged_dataset / "meta").mkdir(parents=True)
    (staged_dataset / "meta/info.json").write_text(
        (portable_dataset / "meta/info.json").read_text()
    )
    train_config_path = policy_path / "train_config.json"
    train_config = json.loads(train_config_path.read_text())
    train_config["dataset"]["root"] = str(staged_dataset)
    train_config_path.write_text(json.dumps(train_config))

    contract = _checkpoint_contract(policy_path, project)

    assert contract["skill_dataset_dir"] == portable_dataset
    assert contract["raw_dataset_dir"] == project / "dataset/source"


def test_checkpoint_contract_rejects_removed_modes(tmp_path: Path) -> None:
    project, policy_path = _checkpoint(tmp_path, label="removed_mode")
    with pytest.raises(ValueError, match="explicitly blocked"):
        _checkpoint_contract(policy_path, project)


def test_checkpoint_contract_rejects_old_model_implementation(tmp_path: Path) -> None:
    project, policy_path = _checkpoint(
        tmp_path,
        label="removed_mode",
        architecture="removed_architecture",
        revision="removed_revision",
    )
    with pytest.raises(ValueError, match="explicitly blocked"):
        _checkpoint_contract(policy_path, project)


def test_checkpoint_contract_does_not_whitelist_architecture_labels(
    tmp_path: Path,
) -> None:
    project, policy_path = _checkpoint(tmp_path, label="experimental_cond")

    contract = _checkpoint_contract(policy_path, project)

    assert contract["architecture_label"] == "experimental_cond"


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


def test_component_ui_maps_three_independent_runs_and_checkpoints() -> None:
    entry = _model_entries(
        {
            "models": [
                {
                    "label": "composed",
                    "vsa_dir": "vsa_run",
                    "vsa_checkpoint": "005000",
                    "predictor_dir": "predictor_run",
                    "predictor_checkpoint": "002000",
                    "terminator_dir": "terminator_run",
                    "terminator_checkpoint": "010000",
                }
            ],
        }
    )[0]

    assert entry["model_dir"] == "vsa_run"
    assert entry["checkpoint"] == "005000"
    assert entry["skill_source"] == "external"
    assert entry["pose_source"] == "predictor"
    assert entry["advance_mode"] == "external"
    assert entry["predictor_diagnostics"] is False
    assert entry["external_predictor_model_value"] == "predictor_run"
    assert entry["external_predictor_checkpoint"] == "002000"
    assert entry["external_terminator_model_value"] == "terminator_run"
    assert entry["external_terminator_checkpoint"] == "010000"


def test_component_ui_joint_predictor_follows_one_vsa_checkpoint_list() -> None:
    entries = _model_entries({"models": [{
        "label": "joint", "vsa_dir": "joint_run",
        "vsa_checkpoint": [0, 1000, "002000"],
        "predictor_dir": "joint", "predictor_checkpoint": "",
        "terminator_dir": "external_term", "terminator_checkpoint": "010000",
    }]})
    assert [row["checkpoint"] for row in entries] == ["000000", "001000", "002000"]
    assert [row["label"] for row in entries] == ["joint_0k", "joint_1k", "joint_2k"]
    assert all(row["skill_source"] == "own" and row["joint_predictor"] for row in entries)
    assert all(row["pose_source"] == "predictor" for row in entries)
    assert all(row["external_predictor_model_value"] == "" for row in entries)
    assert all(row["external_terminator_checkpoint"] == "010000" for row in entries)


def test_joint_predictor_selector_rejects_an_independent_checkpoint() -> None:
    with pytest.raises(ValueError, match="blank when predictor_dir=joint"):
        _model_entries({"models": [{
            "vsa_dir": "joint_run", "vsa_checkpoint": "001000",
            "predictor_dir": "joint", "predictor_checkpoint": "002000",
            "terminator_dir": "gt", "terminator_checkpoint": "",
        }]})


@pytest.mark.parametrize("train_config", [False, True])
def test_joint_zero_predictor_relocates_saved_warm_start(tmp_path, train_config):
    outputs = tmp_path / "downloaded_outputs"
    reference = outputs / "skillVLA_FT/Joint/joint_run/checkpoints/001000/pretrained_model"
    predictor = outputs / "skillVLA_stage1/Predictor/pred_run/checkpoints/100000/pretrained_model"
    reference.mkdir(parents=True)
    predictor.mkdir(parents=True)
    (predictor / "config.json").write_text("{}")
    policy = {
        "newtask_joint_enabled": True,
        "skill_predictor_checkpoint_path": "/remote/outputs/skillVLA_stage1/pred_run/checkpoints/100000/pretrained_model",
    }
    if train_config:
        (reference / "config.json").write_text(json.dumps({
            "newtask_joint_enabled": True, "skill_predictor_checkpoint_path": str(reference),
        }))
        (reference / "train_config.json").write_text(json.dumps({"policy": policy}))
    else:
        (reference / "config.json").write_text(json.dumps(policy))
    assert _resolve_joint_zero_predictor_checkpoint(tmp_path, outputs, reference) == predictor


def test_joint_zero_predictor_reports_missing_source_and_provenance(tmp_path):
    reference = tmp_path / "outputs/skillVLA_FT/Joint/run/checkpoints/001000/pretrained_model"
    reference.mkdir(parents=True)
    path = reference / "config.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="requires a Joint checkpoint"):
        _resolve_joint_zero_predictor_checkpoint(tmp_path, tmp_path / "outputs", reference)
    path.write_text(json.dumps({"newtask_joint_enabled": True}))
    with pytest.raises(ValueError, match="original skill_predictor_checkpoint_path"):
        _resolve_joint_zero_predictor_checkpoint(tmp_path, tmp_path / "outputs", reference)
    path.write_text(json.dumps({
        "newtask_joint_enabled": True,
        "skill_predictor_checkpoint_path": str(tmp_path / "outputs/skillVLA_stage1/Predictor/pred/checkpoints/100000/pretrained_model"),
    }))
    with pytest.raises(FileNotFoundError, match="Download that checkpoint first"):
        _resolve_joint_zero_predictor_checkpoint(tmp_path, tmp_path / "outputs", reference)


@pytest.mark.parametrize("explicit_gt", [False, True])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_joint_sweep_uses_embedded_predictor_except_virtual_zero(tmp_path, explicit_gt, diagnostics):
    import stage1_eval_config as module

    project, source, _ = _contract_with_fsq_source(tmp_path)
    gt = _gt_run(project)
    (gt.parent / "eval_init_states.npz").touch()
    tokenizer = project / "models/tokenizer"
    tokenizer.mkdir(parents=True)
    source_policy = json.loads((source / "config.json").read_text())
    predictor = project / "outputs/skillVLA_stage1/Predictor/pred_run/checkpoints/100000/pretrained_model"
    predictor.mkdir(parents=True)
    predictor_policy = {
        **source_policy, "type": "skill_aux", "train_skill_predictor": True,
        "tokenizer_path": str(tokenizer), "skill_predictor_end_state_mode": "full_state",
    }
    (predictor / "config.json").write_text(json.dumps(predictor_policy))
    (predictor / "model.safetensors").touch()
    joint = project / "outputs/skillVLA_FT/Joint/joint_run"
    for step in ("001000", "002000"):
        reference = joint / "checkpoints" / step / "pretrained_model"
        reference.mkdir(parents=True)
        (reference / "config.json").write_text(json.dumps({
            **predictor_policy, "type": "skill_expert", "newtask_ft_enabled": True,
            "newtask_joint_enabled": True, "pretrained_path": str(source),
            "skill_predictor_checkpoint_path": str(predictor),
        }))
        (reference / "train_config.json").write_text(json.dumps({"dataset": {"root": str(gt / "skillvla")}}))
        for name in ("model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
            (reference / name).touch()
    terminator = project / "outputs/skillVLA_FT/Terminator/term_run/checkpoints/010000/pretrained_model"
    terminator.mkdir(parents=True)
    (terminator / "config.json").write_text(json.dumps({
        "type": "skill_aux", "train_terminator": True, "skill_fsq_levels": [3, 3, 3],
    }))
    (terminator / "model.safetensors").touch()
    row = {
        "label": "joint", "vsa_dir": "joint_run", "vsa_checkpoint": [0, 1000, 2000],
        "predictor_dir": "joint", "predictor_checkpoint": "", "predictor_diagnostics": diagnostics,
        "terminator_dir": "term_run", "terminator_checkpoint": "010000",
    }
    if explicit_gt:
        row["gt_dataset"] = {"source": gt.parent.name, "run": gt.name}
    config = {
        "project_root": str(project), "outputs_root": "outputs", "dataset_root": "dataset",
        "models": [row], "oracle": {"episode_exact": True}, "output_name": "joint_sweep",
    }
    panels = json.loads(module.build_settings(config)["models_json"])
    expected_sources = ["external"] + ["external" if diagnostics else "own"] * 2
    expected_predictors = [str(predictor)] + (
        [p["policy_path"] for p in panels[1:]] if diagnostics else ["", ""]
    )
    assert [p["skill_source"] for p in panels] == expected_sources
    assert [p["external_predictor_model"] for p in panels] == expected_predictors
    assert panels[0]["policy_path"] == str(source)
    assert [Path(p["policy_path"]).parent.name for p in panels[1:]] == ["001000", "002000"]
    assert all(p["external_terminator_model"] == str(terminator) for p in panels)
    assert all(p["pose_source"] == "predictor" for p in panels)
    assert all(p["skill_dataset_dir"] == str(gt / "skillvla") for p in panels)

    # A real Joint step-zero file takes precedence over virtual warm-start lookup.
    real_zero = joint / "checkpoints/000000/pretrained_model"
    real_zero.mkdir(parents=True)
    reference = joint / "checkpoints/001000/pretrained_model"
    for filename in ("config.json", "train_config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
        (real_zero / filename).write_bytes((reference / filename).read_bytes())
    panels = json.loads(module.build_settings(config)["models_json"])
    assert panels[0]["skill_source"] == ("external" if diagnostics else "own")
    assert panels[0]["external_predictor_model"] == (str(real_zero) if diagnostics else "")
    assert panels[0]["policy_path"] == str(real_zero)

    nonjoint_policy = json.loads((reference / "config.json").read_text())
    nonjoint_policy["newtask_joint_enabled"] = False
    (reference / "config.json").write_text(json.dumps(nonjoint_policy))
    with pytest.raises(ValueError, match="predictor_dir=joint requires a Joint checkpoint"):
        module.build_settings(config)


def test_stage1_checkpoint_lists_accept_numeric_steps_and_padded_strings() -> None:
    entries = _model_entries({
        "model_defaults": {"checkpoint": [20000, 2000, 10000]},
        "models": [
            {"label": "A", "model_dir": "run_a"},
            {"label": "B", "model_dir": "run_b", "checkpoint": ["020000", "002000", "010000"]},
        ],
    })
    assert [row["checkpoint"] for row in entries] == [
        "020000", "020000", "002000", "002000", "010000", "010000",
    ]
    assert [row["label"] for row in entries] == [
        "A | ckpt 020000", "B | ckpt 020000", "A | ckpt 002000", "B | ckpt 002000",
        "A | ckpt 010000", "B | ckpt 010000",
    ]


def test_checkpoint_lists_reject_duplicate_numeric_aliases() -> None:
    with pytest.raises(ValueError, match="duplicate checkpoints"):
        _model_entries({
            "model_defaults": {"checkpoint": [10000, "010000"]},
            "models": [{"model_dir": "run"}],
        })


def test_ft_zero_step_reads_saved_warm_start_and_relocates_output_root(tmp_path: Path) -> None:
    outputs = tmp_path / "downloaded_outputs"
    ft = outputs / "skillVLA_FT/VSA/ft_run"
    reference = ft / "checkpoints/010000/pretrained_model"
    reference.mkdir(parents=True)
    source = outputs / "skillVLA_stage1/VSA/source_run/checkpoints/100000/pretrained_model"
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}")
    (reference / "config.json").write_text(json.dumps({
        "newtask_ft_enabled": True,
        "pretrained_path": "/other/server/outputs/skillVLA_stage1/source_run/checkpoints/100000/pretrained_model",
    }))
    resolved, provenance = _resolve_vsa_eval_checkpoint(tmp_path, outputs, ft, "000000")
    assert resolved == source
    assert provenance == reference
    # Regular checkpoints keep loading their FT weights.
    assert _resolve_vsa_eval_checkpoint(tmp_path, outputs, ft, "010000") == (reference, None)


def test_ft_zero_step_supports_last_only_and_train_config_provenance(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    ft = outputs / "skillVLA_FT/VSA/ft_run"
    reference = ft / "checkpoints/last/pretrained_model"
    reference.mkdir(parents=True)
    source = outputs / "skillVLA_stage1/VSA/source_run/checkpoints/100000/pretrained_model"
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}")
    # A resume may overwrite config.pretrained_path with its own last checkpoint.
    (reference / "config.json").write_text(json.dumps({
        "newtask_ft_enabled": True, "pretrained_path": str(reference),
    }))
    (reference / "train_config.json").write_text(json.dumps({
        "policy": {"newtask_ft_enabled": True, "pretrained_path": str(source)},
    }))
    assert _resolve_vsa_eval_checkpoint(tmp_path, outputs, ft, "000000") == (source, reference)


def test_ft_zero_step_reports_missing_source_and_missing_provenance(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    ft = outputs / "skillVLA_FT/VSA/ft_run"
    reference = ft / "checkpoints/010000/pretrained_model"
    reference.mkdir(parents=True)
    (reference / "config.json").write_text(json.dumps({"newtask_ft_enabled": True}))
    with pytest.raises(ValueError, match="no original policy.pretrained_path"):
        _resolve_vsa_eval_checkpoint(tmp_path, outputs, ft, "000000")
    (reference / "config.json").write_text(json.dumps({
        "newtask_ft_enabled": True,
        "pretrained_path": str(outputs / "skillVLA_stage1/VSA/source_run/checkpoints/100000/pretrained_model"),
    }))
    with pytest.raises(FileNotFoundError, match="Download that checkpoint first"):
        _resolve_vsa_eval_checkpoint(tmp_path, outputs, ft, "000000")


def test_real_stage1_zero_checkpoint_keeps_its_original_meaning(tmp_path: Path) -> None:
    run = tmp_path / "outputs/skillVLA_stage1/VSA/run"
    source = run / "checkpoints/000000/pretrained_model"
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}")
    assert _resolve_vsa_eval_checkpoint(tmp_path, tmp_path / "outputs", run, "000000") == (source, None)


@pytest.mark.parametrize("explicit_gt", [False, True])
def test_ft_zero_step_grid_uses_source_weights_and_new_task_gt(tmp_path: Path, explicit_gt: bool) -> None:
    import stage1_eval_config as module

    project, source, _ = _contract_with_fsq_source(tmp_path)
    gt = _gt_run(project)
    (gt.parent / "eval_init_states.npz").touch()
    reference = project / "outputs/skillVLA_FT/VSA/ft_run/checkpoints/001000/pretrained_model"
    reference.mkdir(parents=True)
    for name in ("model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
        (reference / name).touch()
    source_policy = json.loads((source / "config.json").read_text())
    (reference / "config.json").write_text(json.dumps({
        **source_policy, "newtask_ft_enabled": True, "pretrained_path": str(source),
        "fsq_path": str(gt / "FSQ.pt"),
    }))
    (reference / "train_config.json").write_text(json.dumps({"dataset": {"root": str(gt / "skillvla")}}))
    row = {
        "label": "ft", "vsa_dir": "ft_run", "vsa_checkpoint": ["000000", "001000"],
        "predictor_dir": "gt", "predictor_checkpoint": "",
        "terminator_dir": "gt", "terminator_checkpoint": "",
    }
    if explicit_gt:
        row["gt_dataset"] = {"source": gt.parent.name, "run": gt.name}
    settings = module.build_settings({
        "project_root": str(project), "outputs_root": "outputs", "dataset_root": "dataset",
        "models": [row], "oracle": {"episode_exact": True},
        "output_name": "zero_shot_grid",
    })
    panels = json.loads(settings["models_json"])
    assert [panel["label"] for panel in panels] == ["ft_0k", "ft_1k"]
    assert [panel["policy_path"] for panel in panels] == [str(source), str(reference)]
    assert all(panel["skill_dataset_dir"] == str(gt / "skillvla") for panel in panels)
    assert all(panel["eval_init_states_path"] == str(gt.parent / "eval_init_states.npz") for panel in panels)
    # Source normalization/FSQ still belong to the original VSA at step zero.
    assert panels[0]["fsq_path"] == str(project / source_policy["fsq_path"])


def test_component_ui_accepts_multiple_systems_with_different_checkpoints() -> None:
    def system(label: str, step: str) -> dict:
        return {
            "label": label,
            "vsa_dir": "joint_run",
            "vsa_checkpoint": step,
            "predictor_dir": "joint_run",
            "predictor_checkpoint": step,
            "terminator_dir": "joint_run",
            "terminator_checkpoint": step,
        }

    entries = _model_entries(
        {"models": [system("joint_5k", "005000"), system("joint_10k", "010000")]}
    )

    assert [entry["checkpoint"] for entry in entries] == ["005000", "010000"]
    assert [entry["external_predictor_checkpoint"] for entry in entries] == [
        "005000",
        "010000",
    ]
    assert [entry["external_terminator_checkpoint"] for entry in entries] == [
        "005000",
        "010000",
    ]


def test_model_entries_preserve_optional_per_model_gt_datasets() -> None:
    def system(label: str, gt_dataset=None) -> dict:
        row = {
            "label": label,
            "vsa_dir": "joint_run",
            "vsa_checkpoint": "005000",
            "predictor_dir": "gt",
            "predictor_checkpoint": "",
            "terminator_dir": "gt",
            "terminator_checkpoint": "",
        }
        if gt_dataset is not None:
            row["gt_dataset"] = gt_dataset
        return row

    selected = {"source": "libero_10_full_1", "run": "new_fsq_labels"}
    entries = _model_entries(
        {"models": [system("own_dataset", selected), system("inherit_default")]}
    )

    assert entries[0]["gt_dataset_value"] == selected
    assert entries[1]["gt_dataset_value"] is None


def test_component_ui_requires_all_three_runs_and_checkpoints() -> None:
    with pytest.raises(ValueError, match="all six selectors"):
        _model_entries(
            {
                "models": [
                    {
                        "label": "incomplete",
                        "vsa_dir": "vsa_run",
                        "vsa_checkpoint": "005000",
                    }
                ],
            }
        )


def test_component_ui_accepts_gt_predictor_and_terminator_sentinels() -> None:
    entry = _model_entries(
        {
            "models": [
                {
                    "label": "all_gt",
                    "vsa_dir": "vsa_run",
                    "vsa_checkpoint": "005000",
                    "predictor_dir": "gt",
                    "predictor_checkpoint": "",
                    "terminator_dir": "gt",
                    "terminator_checkpoint": "",
                }
            ],
        }
    )[0]

    assert entry["skill_source"] == "gt"
    assert entry["pose_source"] == "gt"
    assert entry["advance_mode"] == "gt"
    assert entry["external_predictor_model_value"] == ""
    assert entry["external_terminator_model_value"] == ""


def test_component_ui_accepts_original_fsq_terminator_sentinel() -> None:
    entry = _model_entries(
        {
            "models": [
                {
                    "label": "fsq_terminator",
                    "vsa_dir": "vsa_run",
                    "vsa_checkpoint": "005000",
                    "predictor_dir": "gt",
                    "predictor_checkpoint": "",
                    "terminator_dir": "original",
                    "terminator_checkpoint": "",
                }
            ],
        }
    )[0]

    assert entry["advance_mode"] == "original"
    assert entry["external_terminator_model_value"] == ""
    assert entry["external_terminator_checkpoint"] == "last"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("predictor_checkpoint", "001000", "blank when predictor_dir=gt"),
        ("terminator_checkpoint", "001000", "blank when terminator_dir=gt"),
    ],
)
def test_gt_component_sentinel_rejects_a_checkpoint(
    field: str, value: str, message: str
) -> None:
    row = {
        "label": "all_gt",
        "vsa_dir": "vsa_run",
        "vsa_checkpoint": "005000",
        "predictor_dir": "gt",
        "predictor_checkpoint": "",
        "terminator_dir": "gt",
        "terminator_checkpoint": "",
    }
    row[field] = value
    with pytest.raises(ValueError, match=message):
        _model_entries({"models": [row]})


def test_original_fsq_terminator_sentinel_rejects_a_checkpoint() -> None:
    row = {
        "label": "fsq_terminator",
        "vsa_dir": "vsa_run",
        "vsa_checkpoint": "005000",
        "predictor_dir": "gt",
        "predictor_checkpoint": "",
        "terminator_dir": "original",
        "terminator_checkpoint": "001000",
    }
    with pytest.raises(ValueError, match="blank when terminator_dir=original"):
        _model_entries({"models": [row]})


def test_model_defaults_end_threshold_overrides_global_terminator() -> None:
    entries = _model_entries(
        {
            "terminator": {"end_threshold": 0.3},
            "model_defaults": {"end_threshold": 0.4},
            "models": [{"model_dir": "first"}],
        }
    )

    assert entries[0]["end_threshold"] == 0.4


def test_model_end_mode_overrides_global_and_blank_inherits() -> None:
    entries = _model_entries(
        {
            "terminator": {"end_mode": "termination"},
            "models": [
                {"model_dir": "first", "end_mode": "progress"},
                {"model_dir": "second", "end_mode": ""},
                {"model_dir": "third"},
            ],
        }
    )

    assert [entry["end_mode"] for entry in entries] == [
        "progress",
        "termination",
        "termination",
    ]


def test_model_defaults_end_mode_overrides_global_terminator() -> None:
    entries = _model_entries(
        {
            "terminator": {"end_mode": "termination"},
            "model_defaults": {"end_mode": "or"},
            "models": [{"model_dir": "first"}],
        }
    )

    assert entries[0]["end_mode"] == "or"


def test_model_end_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="models\\[\\].end_mode"):
        _model_entries(
            {
                "terminator": {"end_mode": "termination"},
                "models": [{"model_dir": "first", "end_mode": "sometimes"}],
            }
        )


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


@pytest.mark.parametrize("workers", [1, 5, 6, 10, 0, 11])
def test_eval_worker_limit_matches_shared_gpu_packing(tmp_path: Path, workers: int) -> None:
    import stage1_eval_config as module

    project, _, _ = _contract_with_fsq_source(tmp_path)
    config = {
        "project_root": str(project), "outputs_root": "outputs", "dataset_root": "dataset",
        "models": [{"vsa_dir": "run", "vsa_checkpoint": "000100",
                    "predictor_dir": "gt", "predictor_checkpoint": "",
                    "terminator_dir": "gt", "terminator_checkpoint": ""}],
        "eval_max_workers_per_gpu": workers, "eval_num_gpus": 10,
        "task_ids": list(range(10)), "output_name": "worker_limit",
    }
    assert module.MAX_SUPPORTED_WORKERS_PER_GPU == 10
    if workers in (0, 11):
        with pytest.raises(ValueError, match="between 1 and 10"):
            module.build_settings(config)
    else:
        assert module.build_settings(config)["eval_max_workers_per_gpu"] == workers


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
    sys.path.insert(0, str(_SRC.parents[3] / "train_skills" / "src"))
    from train_skills_config import stage1_run_dir, stage1_run_dirs

    outputs = tmp_path / "outputs"
    for component in ("VSA", "Predictor", "Terminator"):
        dirs = stage1_run_dirs(outputs, "run", component)
        assert dirs[0] == outputs / "skillVLA_PT" / component / "run"
        assert dirs[1] == outputs / "skillVLA_FT" / component / "run"
        assert dirs[2] == outputs / "skillVLA_FT/Joint/run"
    # Nothing exists → the legacy path is still what diagnostics report.
    assert stage1_run_dir(outputs, "run", "VSA") == outputs / "skillVLA_stage1/run"
    ft_run = outputs / "skillVLA_FT/Predictor/ft_run"
    ft_run.mkdir(parents=True)
    assert stage1_run_dir(outputs, "ft_run", "Predictor") == ft_run
    joint_run = outputs / "skillVLA_FT/Joint/joint_run"
    joint_run.mkdir(parents=True)
    for component in ("VSA", "Predictor", "Terminator"):
        assert stage1_run_dir(outputs, "joint_run", component) == joint_run


def test_ft_eval_yaml_hides_legacy_latent_and_diagnostic_controls() -> None:
    config = load_config(
        _SRC.parents[2] / "FT/eval/ft_eval_config.yaml"
    )

    assert "model_defaults" not in config
    assert "terminator_variant" not in config
    assert "predictor_diagnostics" not in config
    assert not {
        "latent_source",
        "oracle_latent_target",
        "oracle_latent_grid_size",
        "oracle_latent_timesteps",
    } & config.keys()
    entries = _model_entries(config)
    assert all(entry["terminator_variant"] == "state_image" for entry in entries)
    assert all(entry["predictor_diagnostics"] is False for entry in entries)
    assert all(entry["latent_source"] == "random" for entry in entries)


def test_eval_outputs_follow_the_work_dir(tmp_path: Path, monkeypatch) -> None:
    import stage1_eval_config as module

    source = Path(module.__file__).read_text()
    assert 'os.environ.get("STAGE1_EVAL_WORK_DIR"' in source
    shipped = _SRC.parents[2] / "FT/eval/ft_eval_config.yaml"
    config = module.load_config(shipped)
    assert config["target_task"] == "libero_10"
    assert module._model_entries(config)


def test_a_checkpoint_path_from_another_machine_relocates_even_when_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pod records /root/workspace/...; this account cannot even stat it, which is not a failure."""
    recorded = "/root/workspace/Skill_Boundary_Detector/dataset_filtered/skillvla/FSQ.pt"

    def refuse(self):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "exists", refuse)
    assert _relocate_project_path(tmp_path, recorded) == (
        tmp_path / "dataset_filtered/skillvla/FSQ.pt"
    )
    # A relative path never needed the check, and an unanchored one is returned unchanged.
    assert _relocate_project_path(tmp_path, "models/dino") == tmp_path / "models/dino"
    assert _relocate_project_path(tmp_path, "/root/elsewhere/thing.pt") == Path(
        "/root/elsewhere/thing.pt"
    )


def test_a_predictor_checkpoint_list_becomes_one_panel_each() -> None:
    """Sweeping the predictor against a fixed policy: predictor columns x policy checkpoint rows."""
    entries = _model_entries(
        {
            "terminator": {"end_threshold": 0.3},
            "model_defaults": {
                "checkpoint": ["005000", "010000"],
                "skill_source": "external",
                "external_skill_model": "outputs/shared/predictor",
                "external_predictor_checkpoint": ["002000", "010000"],
            },
            "models": [{"model_dir": "policy", "label": "ft"}],
        }
    )

    assert [
        (entry["label"], entry["checkpoint"], entry["external_predictor_checkpoint"])
        for entry in entries
    ] == [
        ("ft | pred 002000 | ckpt 005000", "005000", "002000"),
        ("ft | pred 010000 | ckpt 005000", "005000", "010000"),
        ("ft | pred 002000 | ckpt 010000", "010000", "002000"),
        ("ft | pred 010000 | ckpt 010000", "010000", "010000"),
    ]
    # The predictor is the model axis, so the two sweeps stay distinguishable downstream.
    assert {entry["model_index"] for entry in entries} == {0, 1}
    assert {entry["checkpoint_index"] for entry in entries} == {0, 1}


def _component_predictor_sweep(**overrides) -> dict:
    return {
        "label": "both3", "vsa_dir": "vsa", "vsa_checkpoint": "050000",
        "predictor_dir": "predictor", "predictor_checkpoint": [0, 500, 1000, 2000],
        "terminator_dir": "terminator", "terminator_checkpoint": "010000",
        **overrides,
    }


def test_component_predictor_list_sweeps_fixed_vsa() -> None:
    entries = _model_entries({"models": [_component_predictor_sweep()]})
    assert [entry["external_predictor_checkpoint"] for entry in entries] == [
        "000000", "000500", "001000", "002000",
    ]
    assert {entry["checkpoint"] for entry in entries} == {"050000"}
    assert {entry["external_terminator_checkpoint"] for entry in entries} == {"010000"}
    assert len({entry["label"] for entry in entries}) == 4
    assert all(entry["skill_source"] == "external" and entry["pose_source"] == "predictor"
               for entry in entries)


def test_component_predictor_and_vsa_lists_form_cartesian_grid() -> None:
    row = _component_predictor_sweep(vsa_checkpoint=[10000, 50000], predictor_checkpoint=[500, 2000])
    entries = _model_entries({"models": [row]})
    assert {(entry["checkpoint"], entry["external_predictor_checkpoint"]) for entry in entries} == {
        ("010000", "000500"), ("010000", "002000"),
        ("050000", "000500"), ("050000", "002000"),
    }


@pytest.mark.parametrize("selector", ["gt", "joint"])
def test_predictor_sentinels_reject_nonempty_lists(selector) -> None:
    with pytest.raises(ValueError, match="must be blank"):
        _model_entries({"models": [_component_predictor_sweep(predictor_dir=selector)]})


@pytest.mark.parametrize("steps", [None, "", []])
def test_external_predictor_still_requires_checkpoint(steps) -> None:
    with pytest.raises(ValueError, match="is required"):
        _model_entries({"models": [_component_predictor_sweep(predictor_checkpoint=steps)]})


def test_component_predictor_list_rejects_normalized_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate checkpoints"):
        _model_entries({"models": [_component_predictor_sweep(predictor_checkpoint=[500, "000500"])]})


def test_a_single_predictor_checkpoint_leaves_the_label_alone() -> None:
    """The scalar form is unchanged: no panel split and no ``| pred`` suffix."""
    entries = _model_entries(
        {
            "terminator": {"end_threshold": 0.3},
            "model_defaults": {
                "checkpoint": "005000",
                "skill_source": "external",
                "external_skill_model": "outputs/shared/predictor",
                "external_predictor_checkpoint": "010000",
            },
            "models": [{"model_dir": "policy", "label": "ft"}],
        }
    )

    assert len(entries) == 1
    assert entries[0]["label"] == "ft"
    assert entries[0]["external_predictor_checkpoint"] == "010000"


def test_skill_head_checkpoint_sweep_keeps_complete_predictor_fixed() -> None:
    entries = _model_entries(
        {
            "terminator": {"end_threshold": 0.3},
            "model_defaults": {
                "checkpoint": "005000",
                "skill_source": "external",
                "external_predictor_model": "predictor_run",
                "external_predictor_checkpoint": "002000",
            },
            "models": [
                {
                    "model_dir": "policy",
                    "label": "xyz2k",
                    "external_predictor_skill_checkpoint": [
                        "000000",
                        "000100",
                        "002000",
                    ],
                }
            ],
        }
    )

    assert [
        (
            entry["label"],
            entry["external_predictor_checkpoint"],
            entry["external_predictor_skill_checkpoint"],
        )
        for entry in entries
    ] == [
        ("xyz2k | skill 000000", "002000", "000000"),
        ("xyz2k | skill 000100", "002000", "000100"),
        ("xyz2k | skill 002000", "002000", "002000"),
    ]
    assert {entry["model_index"] for entry in entries} == {0, 1, 2}


@pytest.mark.parametrize("label", __import__("lerobot.configs.uv_fullskill", fromlist=["UV_BSPLINE_GT_XYZ_LABELS"]).UV_BSPLINE_GT_XYZ_LABELS)
def test_gt_xyz_checkpoint_requires_both_uv_and_endpoint(tmp_path, label):
    from lerobot.configs.uv_fullskill import UV_BSPLINE_REVISIONS
    project, path = _checkpoint(tmp_path, label=label, architecture="layerwise_cond_bottleneck", revision=UV_BSPLINE_REVISIONS[label])
    config = json.loads((path / "config.json").read_text())
    config.update(skill_flow_enabled=False, skill_flow_max_length=0, visual_bridge_last_n_layers=9, bspline_tokens=30, bspline_loss_weight=1.)
    (path / "config.json").write_text(json.dumps(config))
    contract = _checkpoint_contract(path, project)
    assert contract["needs_focus_uv"] and contract["needs_end_pose"]
