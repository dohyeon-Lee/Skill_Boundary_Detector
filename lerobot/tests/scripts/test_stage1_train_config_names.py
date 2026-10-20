import json
import subprocess
import sys
from pathlib import Path

import pytest


_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/PT/src"
)
sys.path.insert(0, str(_SRC))
from stage1_train_config import build_settings, FULLSKILL_VIS_REVISIONS, FULLSKILL_VIS_LEGACY_REVISIONS, FULLSKILL_VIS_SPLIT_REVISIONS, BSPLINE_REVISIONS, BSPLINE_LEGACY_REVISIONS  # noqa: E402
from train_skills_config import load_stage1_component_config, stage1_run_dir  # noqa: E402


def _config(tmp_path: Path, architecture: str = "arch0") -> dict:
    project = tmp_path / "project"
    run = "FSQ333_test"
    dataset = project / "dataset/skillvla_dataset/source" / run / "skillvla"
    (dataset / "meta").mkdir(parents=True)
    # Real datasets carry quantile stats; the _align_norm labels read the xyz bounds from here.
    (dataset / "meta/stats.json").write_text(
        json.dumps(
            {
                "observation.state": {
                    "q01": [-0.16, -0.28, -0.26, -3.0, -3.0, -3.0, -0.04, -0.04],
                    "q99": [0.29, 0.32, 0.09, 3.0, 3.0, 3.0, 0.04, 0.04],
                }
            }
        )
    )
    (dataset / "meta/info.json").write_text(
        json.dumps(
            {
                "skill_fsq_levels": [3, 3, 3],
                "skill_code_space_id": "FSQ333_test",
                "skill_observed_max_length": 120,
                "skill_pmax": 15,
                "skill_jitter_early_start_pmax": 10,
                "skill_jitter_late_start_pmax": 5,
                "skill_jitter_early_end_pmax": 10,
                "skill_jitter_late_end_pmax": 5,
                "skill_jitter_distribution": "half_normal",
                "skill_focus_uv_path": (
                    str(dataset.parent / "skill_focus_uv.npz")
                    if architecture.startswith(("arch5", "arch6", "arch7", "arch8_1", "arch8_2", "arch9_1", "arch9_2", "arch10_1", "arch10_2", "arch11_1", "arch11_2", "arch12_1", "arch12_2", "arch13", "arch14", "arch15", "arch16", "arch17", "arch18", "arch19", "arch20", "wristonly_1", "wristonly_2", "both_1", "both_2", "both_3", "both_4", "both_5", "wristonly_lit_", "both_lit_")) else ""
                ),
                "skill_focus_uv_normalization": (
                    "minus_one_to_one" if architecture.startswith(("arch5", "arch6", "arch8_1", "arch8_2", "arch9_1", "arch9_2", "arch10_1", "arch10_2", "arch11_1", "arch11_2", "arch12_1", "arch12_2", "arch13", "arch14", "arch15", "arch16", "arch17", "arch18", "arch19", "arch20", "wristonly_1", "wristonly_2", "both_1", "both_2", "both_3", "both_4", "both_5", "wristonly_lit_", "both_lit_")) else ""
                ),
                "features": {
                    "observation.state": {"shape": [8]},
                    "action": {"shape": [7]},
                },
            }
        )
    )
    if architecture.startswith(("arch5", "arch6", "arch7", "arch8_1", "arch8_2", "arch9_1", "arch9_2", "arch10_1", "arch10_2", "arch11_1", "arch11_2", "arch12_1", "arch12_2", "arch13", "arch14", "arch15", "arch16", "arch17", "arch18", "arch19", "arch20", "wristonly_1", "wristonly_2", "both_1", "both_2", "both_3", "both_4", "both_5", "wristonly_lit_", "both_lit_")):
        (dataset.parent / "skill_focus_uv.npz").touch()
    pi_base = project / "models/pi05_base"
    dino = project / "models/dino"
    pi_base.mkdir(parents=True)
    dino.mkdir(parents=True)
    (pi_base / "model.safetensors").touch()
    return {
        "project_root": str(project),
        "dataset_root": "dataset",
        "outputs_root": "outputs",
        "dataset": {
            "skillvla_root": "skillvla_dataset",
            "source": "source",
            "run": run,
        },
        "warm_start": {"pi_base": "models/pi05_base"},
        "vision": {"dino_model": "models/dino"},
        "architecture": {
            "name": architecture,
            "expert_variant": "gemma_300m",
            "max_state_dim": 32,
            "max_action_dim": 32,
            "chunk_size": 10,
            **(
                {"visual_bottleneck_tokens": 100}
                if architecture in {"wristonly_2", "both_2", "both_3", "both_4"}
                else {}
            ),
        },
        "skill_flow": {
            "weight": 1.0,
            "chunk_multiplier": 3,
            "latent_best_of_n": {"enabled": False},
        },
        "training": {"optimizer": {"dino_lr_scale": 0.1}},
    }


@pytest.mark.parametrize("label", [
    "arch7", "arch9_1", "arch18_align", "wristonly_1", "wristonly_lit_4",
    "both_lit_4", "both_lit_4_dedicated", "fullskill_2",
])
def test_xyz_only_architectures_need_no_focus_artifact(tmp_path, label):
    config = _config(tmp_path, label)
    if "lit" in label or label == "fullskill_2":
        config["architecture"]["visual_bottleneck_tokens"] = 100
    meta = Path(config["project_root"]) / "dataset/skillvla_dataset/source/FSQ333_test/skillvla/meta/info.json"
    info = json.loads(meta.read_text())
    info.pop("skill_focus_uv_path", None)
    info.pop("skill_focus_uv_normalization", None)
    meta.write_text(json.dumps(info))
    assert build_settings(config)["architecture_label"] == label


@pytest.mark.parametrize("label", [
    "arch5", "arch8_1", "arch13", "both_3", "fullskill_1",
    "direct_patch_2", "residual_patch_2", "fullskill_bspline_5_firstlayer",
])
def test_real_uv_targets_still_require_focus_artifact(tmp_path, label):
    config = _config(tmp_path, "both_3")
    config["architecture"]["name"] = label
    meta = Path(config["project_root"]) / "dataset/skillvla_dataset/source/FSQ333_test/skillvla/meta/info.json"
    info = json.loads(meta.read_text())
    info.pop("skill_focus_uv_path", None)
    meta.write_text(json.dumps(info))
    with pytest.raises(FileNotFoundError, match="skill_focus_uv"):
        build_settings(config)


@pytest.mark.parametrize("variant", range(1, 7))
@pytest.mark.parametrize("family", ["residual", "direct"])
@pytest.mark.parametrize("dgd", [False, True])
@pytest.mark.parametrize("bridge_layers", [1, 9, 17])
def test_window_patch_names_and_fixed_horizon(tmp_path, variant, family, dgd, bridge_layers):
    config = _config(tmp_path, "both_3")
    suffix = "_dGD" if dgd else ""
    config["architecture"]["name"] = f"{family.title()}_patch_{variant}" + suffix
    config["architecture"]["visual_bridge_last_n_layers"] = bridge_layers
    config["stage1_component"] = "VSA"
    settings = build_settings(config)
    assert settings["architecture_label"] == f"{family}_patch_{variant}" + suffix
    assert settings["architecture_revision"] == f"layerwise_cond_bottleneck_{family}_patch_v1"
    assert settings["skill_flow_target"] == "residual_window"
    assert settings["skill_flow_state_conditioned"] is True
    assert settings["skill_flow_max_length"] == 4 * settings["chunk_size"]
    assert f"{family}_patch_{variant}" + suffix in settings["pt_run_name"]
    assert settings["visual_bridge_last_n_layers"] == bridge_layers
    if bridge_layers == 1:
        assert "_layer1" not in settings["pt_run_name"]
    else:
        assert f"_layer{bridge_layers}" in settings["pt_run_name"]
        assert "_vlast" not in settings["pt_run_name"]
    assert settings["mask_actions_after_skill_end"] is False  # existing setting is not changed


@pytest.mark.parametrize("bridge_layers", [0, 18, 19])
def test_window_patch_rejects_missing_base_or_terminal_layers(tmp_path, bridge_layers):
    config = _config(tmp_path, "both_3")
    config["architecture"].update(name="direct_patch_1", visual_bridge_last_n_layers=bridge_layers)
    with pytest.raises(ValueError, match="visual_bridge_last_n_layers"):
        build_settings(config)


def test_new_stage1_vsa_output_keeps_legacy_runs_separate(tmp_path: Path) -> None:
    config = _config(tmp_path)
    legacy = build_settings(config)
    config["stage1_component"] = "VSA"
    current = build_settings(config)
    assert current["pt_output_dir"] == legacy["pt_output_dir"].parent / "VSA" / legacy["pt_run_name"]


@pytest.mark.parametrize("label", ["both_3", "direct_patch_2", "residual_patch_2_dGD", *FULLSKILL_VIS_REVISIONS])
def test_sdpa_is_opt_in_and_uses_a_separate_output(tmp_path, label):
    config = _config(tmp_path, "both_3")
    config["architecture"]["name"] = label
    eager = build_settings(config)
    assert eager["attention_backend"] == "eager"
    config["architecture"]["attention_backend"] = " SDPA "
    sdpa = build_settings(config)
    assert sdpa["cond_attention_backend"] == "sdpa"
    assert sdpa["attention_backend"] == sdpa["expert_attention_backend"] == "sdpa"
    assert sdpa["pt_run_name"] == eager["pt_run_name"] + "_sdpa"
    assert sdpa["pt_output_dir"] != eager["pt_output_dir"]
    assert sdpa["architecture_revision"] == eager["architecture_revision"]


@pytest.mark.parametrize("backend", ["flash_attention_2", "auto", "", None])
def test_sdpa_bootstrap_rejects_invalid_backends(tmp_path, backend):
    config = _config(tmp_path, "both_3")
    config["architecture"]["attention_backend"] = backend
    with pytest.raises(ValueError, match="attention_backend"):
        build_settings(config)


@pytest.mark.parametrize("label", ["arch0", "arch1", "arch2"])
def test_sdpa_bootstrap_rejects_architectures_without_layerwise_cond(tmp_path, label):
    config = _config(tmp_path, label)
    config["architecture"]["attention_backend"] = "sdpa"
    with pytest.raises(ValueError, match="layerwise Cond"):
        build_settings(config)


def test_legacy_cond_only_snapshot_keeps_its_expert_eager_and_cannot_resume_as_unified(tmp_path):
    config = _config(tmp_path, "both_3")
    config["architecture"]["cond_attention_backend"] = "sdpa"
    legacy = build_settings(config)
    assert legacy["attention_backend"] == "null"
    assert legacy["expert_attention_backend"] == "eager"
    saved = legacy["pt_output_dir"] / "checkpoints/last/pretrained_model/config.json"
    saved.parent.mkdir(parents=True)
    saved.write_text(json.dumps({"cond_attention_backend": "sdpa"}))
    assert build_settings(config)["pt_run_name"] == legacy["pt_run_name"]
    del config["architecture"]["cond_attention_backend"]
    config["architecture"]["attention_backend"] = "sdpa"
    with pytest.raises(ValueError, match="different attention backends"):
        build_settings(config)
    config["run"] = {"suffix": "unified"}
    assert build_settings(config)["expert_attention_backend"] == "sdpa"


def test_unified_and_legacy_attention_keys_cannot_be_mixed(tmp_path):
    config = _config(tmp_path, "both_3")
    config["architecture"].update(attention_backend="sdpa", cond_attention_backend="eager")
    with pytest.raises(ValueError, match="not both"):
        build_settings(config)


@pytest.mark.parametrize("family", ["Direct", "Residual"])
@pytest.mark.parametrize("variant", range(1, 7))
def test_dgd_cli_override_accepts_case_insensitive_names(tmp_path, monkeypatch, capsys, family, variant):
    import stage1_train_config as bootstrap

    config = _config(tmp_path, "both_3")
    monkeypatch.setattr(bootstrap, "load_stage1_component_config", lambda path: config)
    monkeypatch.setattr(sys, "argv", ["stage1_train_config.py", "--architecture", f"{family}_patch_{variant}_dgd"])
    bootstrap.main()
    assert f"architecture_label: {family.lower()}_patch_{variant}_dGD" in capsys.readouterr().out


def test_stage1_transition_jitter_can_be_narrowed_and_change_distribution(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["transition_randomization"] = {
        "enabled": True,
        "early_start_pmax": 8,
        "late_start_pmax": 4,
        "early_end_pmax": 7,
        "late_end_pmax": 3,
        "distribution": "uniform",
    }

    settings = build_settings(config)

    assert settings["transition_jitter_pmax"] == 8
    assert settings["transition_jitter_early_start_pmax"] == 8
    assert settings["transition_jitter_late_start_pmax"] == 4
    assert settings["transition_jitter_early_end_pmax"] == 7
    assert settings["transition_jitter_late_end_pmax"] == 3
    assert settings["transition_jitter_distribution"] == "uniform"
    assert "_jit" not in settings["pt_run_name"]


def test_stage1_transition_jitter_rejects_dataset_contract_overflow(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["transition_randomization"] = {"early_start_pmax": 11}

    with pytest.raises(ValueError, match="dataset contract"):
        build_settings(config)


def test_stage1_percentage_jitter_uses_built_gripper_contract(tmp_path: Path) -> None:
    config = _config(tmp_path)
    info_path = (
        Path(config["project_root"])
        / "dataset/skillvla_dataset/source/FSQ333_test/skillvla/meta/info.json"
    )
    info = json.loads(info_path.read_text())
    info["skill_jitter_max_percent"] = 35.0
    info["skill_jitter_gripper_validity"] = True
    info_path.write_text(json.dumps(info))
    config["transition_randomization"] = {
        "enabled": True,
        "percent": 15,
        "distribution": "uniform",
        "preserve_gripper": True,
    }

    settings = build_settings(config)

    assert settings["transition_jitter_pmax"] == 0
    assert settings["transition_jitter_pmax_percent"] == 15
    assert settings["transition_jitter_preserve_gripper"] is True
    assert settings["transition_jitter_distribution"] == "uniform"


def test_stage1_common_suffix_follows_component_suffix(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["run"] = {"suffix": "vsa", "common_suffix": "shared"}

    settings = build_settings(config)

    assert settings["pt_run_name"].endswith("_vsa_shared")


def test_stage1_common_yaml_merges_nested_component_overrides(tmp_path: Path) -> None:
    component_dir = tmp_path / "PT" / "VSA"
    component_dir.mkdir(parents=True)
    (tmp_path / "global_config.yaml").write_text("outputs_root: outputs\n")
    (component_dir.parent / "stage1_common_config.yaml").write_text(
        "dataset:\n  source: libero_90\n  run: FSQ333_base\n"
    )
    component = component_dir / "vsa_train_config.yaml"
    component.write_text("stage1_common: true\ndataset:\n  run: FSQ333_fov\n")
    loaded = load_stage1_component_config(component)
    assert loaded["dataset"] == {"source": "libero_90", "run": "FSQ333_fov"}
    assert loaded["outputs_root"] == "outputs"


def test_stage1_run_lookup_keeps_old_runs_and_prefers_new(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    legacy = outputs / "skillVLA_stage1" / "same_name"
    current = outputs / "skillVLA_PT" / "VSA" / "same_name"
    legacy.mkdir(parents=True)
    assert stage1_run_dir(outputs, "same_name", "VSA") == legacy
    current.mkdir(parents=True)
    assert stage1_run_dir(outputs, "same_name", "VSA") == current


@pytest.mark.parametrize(
    ("label", "enabled", "target", "length"),
    [
        ("arch0", False, "canonical", 0),
        ("arch0_skill", True, "canonical", 120),
        ("arch0_skill_chunk", True, "extended_chunk", 30),
        ("arch1", False, "canonical", 0),
        ("arch1_skill", True, "canonical", 120),
        ("arch1_skill_chunk", True, "extended_chunk", 30),
        ("arch2", False, "canonical", 0),
        ("arch2_skill", True, "canonical", 120),
        ("arch2_skill_chunk", True, "extended_chunk", 30),
        ("arch3", False, "canonical", 0),
        ("arch3_skill", True, "canonical", 120),
        ("arch3_skill_chunk", True, "extended_chunk", 30),
        ("arch4", False, "canonical", 0),
        ("arch4_skill", True, "canonical", 120),
        ("arch4_skill_chunk", True, "extended_chunk", 30),
        ("arch5", False, "canonical", 0),
        ("arch5_skill", True, "canonical", 120),
        ("arch5_skill_chunk", True, "extended_chunk", 30),
        ("arch6", False, "canonical", 0),
        ("arch6_skill", True, "canonical", 120),
        ("arch6_skill_chunk", True, "extended_chunk", 30),
        ("arch7", False, "canonical", 0),
        ("arch7_skill", True, "canonical", 120),
        ("arch7_skill_chunk", True, "extended_chunk", 30),
        ("arch8_1", False, "canonical", 0),
        ("arch8_1_skill", True, "canonical", 120),
        ("arch8_1_skill_chunk", True, "extended_chunk", 30),
        ("arch8_2", False, "canonical", 0),
        ("arch8_2_skill", True, "canonical", 120),
        ("arch8_2_skill_chunk", True, "extended_chunk", 30),
        ("arch9_1", False, "canonical", 0),
        ("arch9_1_skill", True, "canonical", 120),
        ("arch9_1_skill_chunk", True, "extended_chunk", 30),
        ("arch9_2", False, "canonical", 0),
        ("arch9_2_skill", True, "canonical", 120),
        ("arch9_2_skill_chunk", True, "extended_chunk", 30),
        ("arch10_1", False, "canonical", 0),
        ("arch10_1_skill", True, "canonical", 120),
        ("arch10_1_skill_chunk", True, "extended_chunk", 30),
        ("arch10_2", False, "canonical", 0),
        ("arch10_2_skill", True, "canonical", 120),
        ("arch10_2_skill_chunk", True, "extended_chunk", 30),
        ("arch11_1", False, "canonical", 0),
        ("arch11_1_skill", True, "canonical", 120),
        ("arch11_1_skill_chunk", True, "extended_chunk", 30),
        ("arch11_2", False, "canonical", 0),
        ("arch11_2_skill", True, "canonical", 120),
        ("arch11_2_skill_chunk", True, "extended_chunk", 30),
        ("arch12_1", False, "canonical", 0),
        ("arch12_1_skill", True, "canonical", 120),
        ("arch12_1_skill_chunk", True, "extended_chunk", 30),
        ("arch12_2", False, "canonical", 0),
        ("arch12_2_skill", True, "canonical", 120),
        ("arch12_2_skill_chunk", True, "extended_chunk", 30),
        ("arch13", False, "canonical", 0),
        ("arch13_skill", True, "canonical", 120),
        ("arch13_skill_chunk", True, "extended_chunk", 30),
        ("arch14", False, "canonical", 0),
        ("arch14_skill", True, "canonical", 120),
        ("arch14_skill_chunk", True, "extended_chunk", 30),
        ("arch15", False, "canonical", 0),
        ("arch15_skill", True, "canonical", 120),
        ("arch15_skill_chunk", True, "extended_chunk", 30),
        ("arch16", False, "canonical", 0),
        ("arch16_skill", True, "canonical", 120),
        ("arch16_skill_chunk", True, "extended_chunk", 30),
        ("arch16_align", False, "canonical", 0),
        ("arch16_align_skill", True, "canonical", 120),
        ("arch16_align_skill_chunk", True, "extended_chunk", 30),
        ("arch17", False, "canonical", 0),
        ("arch17_skill", True, "canonical", 120),
        ("arch17_skill_chunk", True, "extended_chunk", 30),
        ("arch17_align", False, "canonical", 0),
        ("arch17_align_skill", True, "canonical", 120),
        ("arch17_align_skill_chunk", True, "extended_chunk", 30),
        ("arch18", False, "canonical", 0),
        ("arch18_skill", True, "canonical", 120),
        ("arch18_skill_chunk", True, "extended_chunk", 30),
        ("arch18_align", False, "canonical", 0),
        ("arch18_align_skill", True, "canonical", 120),
        ("arch18_align_skill_chunk", True, "extended_chunk", 30),
        ("arch18_align_norm", False, "canonical", 0),
        ("arch18_align_norm_skill", True, "canonical", 120),
        ("arch18_align_norm_skill_chunk", True, "extended_chunk", 30),
        ("arch19", False, "canonical", 0),
        ("arch19_skill", True, "canonical", 120),
        ("arch19_skill_chunk", True, "extended_chunk", 30),
        ("arch20", False, "canonical", 0),
        ("arch20_skill", True, "canonical", 120),
        ("arch20_skill_chunk", True, "extended_chunk", 30),
        ("wristonly_1", True, "canonical", 120),
        ("wristonly_2", True, "canonical", 120),
        ("both_1", True, "canonical", 120),
        ("both_2", True, "canonical", 120),
    ],
)
def test_stage1_resolves_retained_arch0_and_arch1_modes(
    tmp_path: Path,
    label: str,
    enabled: bool,
    target: str,
    length: int,
) -> None:
    settings = build_settings(_config(tmp_path, label))

    is_arch1 = label == "arch1" or label.startswith("arch1_")
    is_arch2 = label == "arch2" or label.startswith("arch2_")
    is_arch3 = label.startswith("arch3")
    is_arch4 = label.startswith("arch4")
    is_arch5 = label.startswith("arch5")
    is_arch6 = label.startswith("arch6")
    is_arch7 = label.startswith("arch7")
    is_arch8_1 = label.startswith("arch8_1")
    is_arch8_2 = label.startswith("arch8_2")
    is_arch9_1 = label.startswith("arch9_1")
    is_arch9_2 = label.startswith("arch9_2")
    is_arch10_1 = label.startswith("arch10_1")
    is_arch10_2 = label.startswith("arch10_2")
    is_arch11_1 = label.startswith("arch11_1")
    is_arch11_2 = label.startswith("arch11_2")
    is_arch12_1 = label.startswith("arch12_1")
    is_arch12_2 = label.startswith("arch12_2")
    is_arch18 = label.startswith("arch18")
    is_arch17 = label.startswith("arch17")
    is_arch16 = label.startswith("arch16")
    is_align = label.startswith(("arch16_align", "arch17_align", "arch18_align"))
    is_goal_norm = label.startswith("arch18_align_norm")
    is_arch15 = label.startswith("arch15")
    is_arch19 = label.startswith("arch19")
    is_arch20 = label.startswith("arch20")
    is_wristonly = label in {"wristonly_1", "wristonly_2"}
    is_both = label in {"both_1", "both_2"}
    is_arch14 = label.startswith(("arch14", "arch15", "arch19"))
    is_arch13 = label.startswith(("arch13", "arch14", "arch15", "arch19", "arch20"))
    is_layerwise = is_arch3 or is_arch4 or is_arch5 or is_arch6 or is_arch7 or is_arch8_1 or is_arch8_2 or is_arch9_1 or is_arch9_2 or is_arch10_1 or is_arch10_2 or is_arch11_1 or is_arch11_2 or is_arch12_1 or is_arch12_2 or is_arch13 or is_arch16 or is_arch17 or is_arch18 or is_wristonly or is_both
    is_visual_bottleneck = is_arch1 or is_arch2
    assert settings["architecture"] == (
        "layerwise_cond_bottleneck"
        if is_layerwise
        else ("fixed_visual_bottleneck" if is_visual_bottleneck else "cond_gemma")
    )
    expected_revision = "skillvla_real_v1"
    for enabled_flag, revision in (
        (is_arch1, "fixed_visual_bottleneck_v1"),
        (is_arch2, "late_visual_bottleneck_v1"),
        (is_arch3, "layerwise_cond_bottleneck_v1"),
        (is_arch4, "layerwise_cond_bottleneck_core_exit_v1"),
        (is_arch5, "layerwise_cond_bottleneck_core_exit_uv_v1"),
        (is_arch6, "layerwise_cond_bottleneck_core_exit_latent_uv_v1"),
        (is_arch7, "layerwise_cond_bottleneck_core_exit_latent_xyz_v1"),
        (is_arch8_1, "layerwise_cond_bottleneck_uv_cond_xyz_v1"),
        (is_arch8_2, "layerwise_cond_bottleneck_uv_cond_xyz_cond_termination_v2"),
        (is_arch9_1, "layerwise_cond_bottleneck_wrist_skill_end_pose_v1"),
        (is_arch9_2, "layerwise_cond_bottleneck_wrist_skill_end_pose_cond_termination_v2"),
        (is_arch10_1, "layerwise_cond_bottleneck_wrist_proprio_expert_end_pose_v1"),
        (is_arch10_2, "layerwise_cond_bottleneck_wrist_proprio_expert_end_pose_cond_termination_v1"),
        (is_arch11_1, "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_end_pose_v1"),
        (is_arch11_2, "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_end_pose_cond_termination_v1"),
        (is_arch12_1, "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_v1"),
        (is_arch12_2, "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_cond_termination_v1"),
        (is_arch13, "layerwise_cond_bottleneck_xyz_cond_uv_v1"),
        (is_arch14, "layerwise_cond_bottleneck_xyz_cond_uv_expert_end_pose_v1"),
        (is_arch15, "layerwise_cond_bottleneck_xyz_skill_cond_uv_expert_end_pose_v1"),
        (is_arch16, "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_delta_v1"),
        (is_arch17, "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_delta_bridge_proprio_v1"),
        (is_arch18, "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_start_end_bridge_proprio_v1"),
        (is_arch19, "layerwise_cond_bottleneck_xyz_skill_cond_uv_expert_skill_delta_v1"),
        (is_arch20, "layerwise_cond_bottleneck_xyz_skill_cond_uv_v1"),
        (is_align and is_arch16, "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_delta_align_v1"),
        (is_align and is_arch17, "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_delta_bridge_proprio_align_v1"),
        (is_align and is_arch18, "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_start_end_bridge_proprio_align_v1"),
        (is_goal_norm, "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_start_end_bridge_proprio_align_norm_v1"),
        (label == "wristonly_1", "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_wrist_patch_align_v1"),
        (label == "both_1", "layerwise_cond_bottleneck_both_cond_skill_end_pose_expert_skill_dual_patch_align_v1"),
        (label == "wristonly_2", "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_wrist_patch_dedicated_align_v1"),
        (label == "both_2", "layerwise_cond_bottleneck_both_cond_skill_end_pose_expert_skill_dual_patch_dedicated_align_v1"),
    ):
        if enabled_flag:
            expected_revision = revision
    assert settings["architecture_revision"] == expected_revision
    assert settings["vision_conditioning_mode"] == (
        "layerwise_cond_bottleneck_cross_attention"
        if is_layerwise
        else (
            "fixed_bottleneck_cross_attention"
            if is_visual_bottleneck
            else "interleaved_cross_attention"
        )
    )
    assert settings["architecture_label"] == label
    assert settings["conditioning_route"] == "state_cond"
    assert settings["cond_encoder_variant"] == "gemma_300m"
    assert settings["skill_flow_enabled"] is enabled
    assert settings["skill_flow_target"] == target
    assert settings["skill_flow_state_conditioned"] is False
    assert settings["skill_flow_max_length"] == length
    assert settings["skill_flow_checkpoint_layers"] == 2
    assert settings["visual_bridge_last_n_layers"] == 1
    assert settings["pt_run_name"].endswith(f"_{label}")


def test_stage1_accepts_skill_flow_checkpoint_layers_override(tmp_path: Path) -> None:
    config = _config(tmp_path, "arch4_skill")
    config["skill_flow"]["checkpoint_layers"] = 0

    settings = build_settings(config)

    assert settings["skill_flow_checkpoint_layers"] == 0


def test_arch7_uses_existing_focus_index_and_xyz_loss_weight(tmp_path: Path) -> None:
    config = _config(tmp_path, "arch7_skill")
    config["architecture"]["spatial_loss_weight"] = 0.1
    settings = build_settings(config)
    assert settings["cond_end_xyz_loss_weight"] == 0.1
    assert settings["cond_focus_uv_loss_weight"] == 1.0
    assert "_arch7_skill_xyz0p1" in settings["pt_run_name"]


@pytest.mark.parametrize(
    ("label", "revision"),
    [
        (
            "wristonly_lit_4",
            "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_start_end_bridge_proprio_chunk_end_pose_v1",
        ),
        (
            "both_lit_4",
            "layerwise_cond_bottleneck_both_cond_skill_end_pose_expert_skill_start_end_bridge_proprio_chunk_end_pose_v1",
        ),
        (
            "wristonly_lit_5",
            "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_start_end_chunk_end_pose_v1",
        ),
        (
            "both_lit_5",
            "layerwise_cond_bottleneck_both_cond_skill_end_pose_expert_skill_start_end_chunk_end_pose_v1",
        ),
        (
            "both_lit_6",
            "layerwise_cond_bottleneck_both_cond_proprio_expert_skill_start_end_chunk_end_pose_v1",
        ),
        (
            "both_lit_4_dedicated",
            "layerwise_cond_bottleneck_both_cond_skill_end_pose_expert_skill_start_end_bridge_proprio_dedicated_chunk_end_pose_v1",
        ),
    ],
)
def test_lit4_variants_lit5_and_lit6_resolve_training_contract(
    tmp_path: Path, label: str, revision: str
) -> None:
    config = _config(tmp_path, label)
    config["architecture"]["visual_bottleneck_tokens"] = 100
    settings = build_settings(config)

    assert settings["architecture"] == "layerwise_cond_bottleneck"
    assert settings["architecture_revision"] == revision
    assert settings["skill_flow_enabled"] is True
    assert settings["chunk_end_state_loss_weight"] == 0.3
    assert settings["chunk_end_state_dim"] == 8
    assert settings["chunk_end_state_q01"]
    assert settings["chunk_end_state_q99"]


def test_both_lit7_replaces_chunk_end_decoder_with_dual_patch_queries(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "both_lit_7")
    config["architecture"].update(
        visual_bottleneck_tokens=100,
        spatial_loss_weight=0.1,
    )
    settings = build_settings(config)

    assert settings["architecture"] == "layerwise_cond_bottleneck"
    assert settings["architecture_revision"] == (
        "layerwise_cond_bottleneck_both_cond_skill_end_pose_"
        "expert_skill_start_end_dual_patch_dedicated_align_v1"
    )
    assert settings["skill_flow_enabled"] is True
    assert settings["visual_bottleneck_tokens"] == 100
    assert settings["wrist_patch_align_loss_weight"] == pytest.approx(0.1)
    assert "both_lit_7" in settings["pt_run_name"]


@pytest.mark.parametrize("label,revision", BSPLINE_REVISIONS.items())
@pytest.mark.parametrize("bridge_layers", [1, 9, 17])
def test_bspline_models_follow_bridge_split_and_replace_canonical_flow(tmp_path, label, revision, bridge_layers):
    config = _config(tmp_path, "both_lit_7")
    config["architecture"].update(name=label, visual_bottleneck_tokens=100, visual_bridge_last_n_layers=bridge_layers)
    config["bspline"] = {"tokens": 30, "weight": 1.0}
    settings = build_settings(config)
    assert settings["architecture_revision"] == revision
    assert settings["visual_bridge_last_n_layers"] == bridge_layers
    assert not settings["skill_flow_enabled"]
    assert settings["skill_flow_max_length"] == 0
    assert settings["bspline_tokens"] == 30 and settings["bspline_loss_weight"] == 1.
    assert label in settings["pt_run_name"]
    if bridge_layers != 1:
        assert f"_vlast{bridge_layers}" in settings["pt_run_name"]


@pytest.mark.parametrize("label", BSPLINE_REVISIONS)
def test_bspline_submission_cli_accepts_registered_models_without_submitting(label):
    script = (_SRC.parent / "VSA/submit_train.sh").read_text()
    # Run only the argument dispatch, never snapshot creation or job submission.
    dispatch = script.split('case "${ARCHITECTURE_OVERRIDE}" in', 1)[1].split("esac", 1)[0]
    result = subprocess.run(
        ["bash", "-c", 'ARCHITECTURE_OVERRIDE="$1"\ncase "${ARCHITECTURE_OVERRIDE}" in' + dispatch + "esac", "_", label],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_bspline_resume_preserves_saved_split_and_allows_legacy_9_9(tmp_path):
    label = "fullskill_bspline_2_firstlayer"
    config = _config(tmp_path, "both_lit_7")
    config["architecture"].update(name=label, visual_bottleneck_tokens=100, visual_bridge_last_n_layers=9)
    settings = build_settings(config)
    saved = settings["pt_output_dir"] / "checkpoints/last/pretrained_model/config.json"
    saved.parent.mkdir(parents=True)
    saved.write_text(json.dumps({"architecture_revision": BSPLINE_REVISIONS[label], "visual_bridge_last_n_layers": 1}))
    with pytest.raises(ValueError, match="Cannot auto-resume Bspline"):
        build_settings(config)
    saved.write_text(json.dumps({"architecture_revision": BSPLINE_LEGACY_REVISIONS[label], "visual_bridge_last_n_layers": 9}))
    assert build_settings(config)["visual_bridge_last_n_layers"] == 9


@pytest.mark.parametrize("label,revision", FULLSKILL_VIS_REVISIONS.items())
def test_fullskill_visual_models_have_separate_start_vision_and_own_run(tmp_path, label, revision):
    config = _config(tmp_path, "both_lit_7")
    config["architecture"].update(name=label, visual_bottleneck_tokens=100)
    settings = build_settings(config)
    assert settings["architecture_label"] == label
    assert settings["architecture_revision"] == revision
    assert settings["skill_flow_target"] == "canonical"
    assert settings["skill_flow_enabled"]
    assert not settings["skill_flow_state_conditioned"]
    assert settings["visual_bridge_last_n_layers"] == 1
    assert settings["pt_run_name"].endswith(f"_{label}")
    config["architecture"]["visual_bridge_last_n_layers"] = 9
    split_settings = build_settings(config)
    assert split_settings["visual_bridge_last_n_layers"] == 9
    assert split_settings["architecture_revision"] == FULLSKILL_VIS_SPLIT_REVISIONS.get(label, revision)
    assert split_settings["skill_flow_target"] == "canonical"
    assert split_settings["pt_run_name"].endswith(f"_{label}_vlast9")
    assert split_settings["pt_output_dir"] != settings["pt_output_dir"]
    config["architecture"]["visual_bridge_last_n_layers"] = 18
    with pytest.raises(ValueError, match="Fullskill visual"):
        build_settings(config)


@pytest.mark.parametrize("label", [label for label in FULLSKILL_VIS_LEGACY_REVISIONS if label != "fullskill_vis_1"])
def test_symmetric_start_model_cannot_silently_auto_resume_v1(tmp_path, label):
    config = _config(tmp_path, "both_lit_7")
    config["architecture"].update(name=label, visual_bottleneck_tokens=100)
    settings = build_settings(config)
    saved = settings["pt_output_dir"] / "checkpoints/last/pretrained_model/config.json"
    saved.parent.mkdir(parents=True)
    saved.write_text(json.dumps({"architecture_revision": FULLSKILL_VIS_LEGACY_REVISIONS[label]}))
    with pytest.raises(ValueError, match="Cannot auto-resume"):
        build_settings(config)
    saved.write_text(json.dumps({"architecture_revision": FULLSKILL_VIS_REVISIONS[label]}))
    assert build_settings(config)["architecture_revision"] == FULLSKILL_VIS_REVISIONS[label]


@pytest.mark.parametrize("label", FULLSKILL_VIS_REVISIONS)
def test_fullskill_visual_split_cannot_silently_auto_resume_another_split(tmp_path, label):
    config = _config(tmp_path, "both_lit_7")
    config["architecture"].update(name=label, visual_bottleneck_tokens=100, visual_bridge_last_n_layers=9)
    settings = build_settings(config)
    saved = settings["pt_output_dir"] / "checkpoints/last/pretrained_model/config.json"
    saved.parent.mkdir(parents=True)
    saved.write_text(json.dumps({"architecture_revision": settings["architecture_revision"], "visual_bridge_last_n_layers": 1}))
    with pytest.raises(ValueError, match="different Expert split"):
        build_settings(config)
    saved.write_text(json.dumps({"architecture_revision": settings["architecture_revision"], "visual_bridge_last_n_layers": 9}))
    assert build_settings(config)["architecture_revision"] == settings["architecture_revision"]


@pytest.mark.parametrize(
    ("label", "revision"),
    [
        (
            "both_3",
            "layerwise_cond_bottleneck_both_cond_skill_end_pose_expert_skill_start_end_bridge_proprio_dual_patch_dedicated_align_v1",
        ),
        (
            "both_4",
            "layerwise_cond_bottleneck_both_cond_skill_expert_skill_start_end_bridge_proprio_dual_patch_dedicated_align_v1",
        ),
    ],
)
def test_both3_and_both4_resolve_dedicated_dual_patch_contract(
    tmp_path: Path, label: str, revision: str
) -> None:
    config = _config(tmp_path, label)
    config["architecture"]["spatial_loss_weight"] = 0.1
    settings = build_settings(config)

    assert settings["architecture"] == "layerwise_cond_bottleneck"
    assert settings["architecture_revision"] == revision
    assert settings["skill_flow_enabled"] is True
    assert settings["visual_bottleneck_tokens"] == 100
    assert settings["wrist_patch_align_loss_weight"] == pytest.approx(0.1)
    assert label in settings["pt_run_name"]
    assert "vtok" not in settings["pt_run_name"]


@pytest.mark.parametrize("variant", [1, 2])
def test_fullskill_reuses_both3_canonical_flow_and_has_its_own_run(tmp_path: Path, variant: int) -> None:
    config = _config(tmp_path, "both_3")
    both3 = build_settings(config)
    config["architecture"]["name"] = f"Fullskill_{variant}"
    fullskill = build_settings(config)
    assert fullskill["architecture_label"] == f"fullskill_{variant}"
    assert fullskill["architecture_revision"] == ("layerwise_cond_bottleneck_fullskill_1_chunk_end_dual_patch_v1"
                                                 if variant == 1 else "layerwise_cond_bottleneck_fullskill_2_no_visual_aux_v1")
    assert f"_fullskill_{variant}" in fullskill["pt_run_name"]
    assert fullskill["pt_output_dir"] != both3["pt_output_dir"]
    for name in ("skill_flow_enabled", "skill_flow_target", "skill_flow_state_conditioned",
                 "skill_flow_max_length", "visual_bottleneck_tokens",
                 "visual_bridge_last_n_layers", "mask_actions_after_skill_end"):
        assert fullskill[name] == both3[name]
    assert fullskill["wrist_patch_align_loss_weight"] == (both3["wrist_patch_align_loss_weight"] if variant == 1 else 0.0)


def test_both5_resolves_shared_dual_patch_contract(tmp_path: Path) -> None:
    config = _config(tmp_path, "both_5")
    config["architecture"].update(
        visual_bottleneck_tokens=100,
        spatial_loss_weight=0.1,
    )
    settings = build_settings(config)

    assert settings["architecture"] == "layerwise_cond_bottleneck"
    assert settings["architecture_revision"] == (
        "layerwise_cond_bottleneck_both_cond_skill_end_pose_"
        "expert_skill_start_end_bridge_proprio_dual_patch_align_v1"
    )
    assert settings["skill_flow_enabled"] is True
    assert settings["visual_bottleneck_tokens"] == 100
    assert settings["wrist_patch_align_loss_weight"] == pytest.approx(0.1)
    assert "both_5" in settings["pt_run_name"]
    assert "vtok" not in settings["pt_run_name"]


def test_arch8_1_uses_original_view_and_existing_focus_uv(tmp_path: Path) -> None:
    config = _config(tmp_path, "arch8_1_skill")
    config["architecture"]["spatial_loss_weight"] = 0.1
    settings = build_settings(config)
    assert settings["foveated_vision_enabled"] is False
    assert settings["cond_end_xyz_loss_weight"] == 0.1
    assert "_arch8_1_skill_xyz0p1" in settings["pt_run_name"]
    config["vision"]["foveation"] = {"enabled": True}
    with pytest.raises(ValueError, match="original agent view"):
        build_settings(config)


def test_arch8_2_exports_termination_objective_and_rejects_foveation(tmp_path: Path) -> None:
    config = _config(tmp_path, "arch8_2_skill")
    config["termination_loss"] = {"loss_weight": 0.5, "target_sigma": 2.0, "positive_weight": 2.0}
    settings = build_settings(config)
    assert settings["bottleneck_termination_loss_weight"] == 0.5
    assert settings["bottleneck_termination_target_sigma"] == 2.0
    assert settings["bottleneck_termination_positive_weight"] == 2.0
    assert "_arch8_2_skill_term0p5" in settings["pt_run_name"]
    config["vision"]["foveation"] = {"enabled": True}
    with pytest.raises(ValueError, match="original agent view"):
        build_settings(config)


def test_arch13_reverses_arch8_spatial_contract(tmp_path: Path) -> None:
    config = _config(tmp_path, "arch13_skill")
    config["architecture"]["spatial_loss_weight"] = 0.1
    # The shared VSA YAML keeps this field populated while switching families.
    config["architecture"]["end_pose_mode"] = "xyz"
    settings = build_settings(config)
    assert settings["architecture_revision"] == "layerwise_cond_bottleneck_xyz_cond_uv_v1"
    assert settings["foveated_vision_enabled"] is False
    assert settings["cond_focus_uv_loss_weight"] == 0.1
    assert settings["cond_end_xyz_loss_weight"] == 1.0
    assert "_arch13_skill_uv0p1" in settings["pt_run_name"]
    config["vision"]["foveation"] = {"enabled": True}
    with pytest.raises(ValueError, match="original agent view"):
        build_settings(config)


def test_arch14_is_arch13_plus_a_configurable_expert_end_pose(tmp_path: Path) -> None:
    config = _config(tmp_path, "arch14_skill")
    config["architecture"]["spatial_loss_weight"] = 0.1
    settings = build_settings(config)
    assert settings["architecture_revision"] == "layerwise_cond_bottleneck_xyz_cond_uv_expert_end_pose_v1"
    assert settings["architecture"] == "layerwise_cond_bottleneck"
    assert settings["cond_focus_uv_loss_weight"] == 0.1          # Arch13's UV auxiliary
    assert settings["skill_end_pose_mode"] == "xyz"
    assert "_arch14_skill_uv0p1" in settings["pt_run_name"]
    assert "endxyz" not in settings["pt_run_name"]

    config["architecture"]["end_pose_mode"] = "pose"            # Arch11-style pose option
    posed = build_settings(config)
    assert posed["skill_end_pose_mode"] == "pose" and "_endpose" in posed["pt_run_name"]

    config["vision"]["foveation"] = {"enabled": True}           # still Arch13's agent-view rule
    with pytest.raises(ValueError, match="original agent view"):
        build_settings(config)

    arch13 = _config(tmp_path / "arch13", "arch13_skill")
    arch13["architecture"]["end_pose_mode"] = "pose"
    with pytest.raises(ValueError, match="Arch13 fixes"):
        build_settings(arch13)


def test_arch9_1_names_end_pose_mode_and_rejects_foveation(tmp_path: Path) -> None:
    config = _config(tmp_path, "arch9_1_skill")
    config["architecture"]["end_pose_mode"] = "pose"
    settings = build_settings(config)
    assert settings["skill_end_pose_mode"] == "pose"
    assert settings["pt_run_name"].endswith("_arch9_1_skill_endpose")
    config["vision"]["foveation"] = {"enabled": True}
    with pytest.raises(ValueError, match="wrist-only"):
        build_settings(config)


def test_arch9_2_exports_termination_objective_and_end_pose(tmp_path: Path) -> None:
    config = _config(tmp_path, "arch9_2_skill")
    config["architecture"]["end_pose_mode"] = "pose"
    config["termination_loss"] = {"loss_weight": 0.5, "target_sigma": 3.0, "positive_weight": 4.0}
    settings = build_settings(config)
    assert settings["architecture_revision"] == "layerwise_cond_bottleneck_wrist_skill_end_pose_cond_termination_v2"
    assert settings["skill_end_pose_mode"] == "pose"
    assert settings["bottleneck_termination_loss_weight"] == 0.5
    assert settings["bottleneck_termination_target_sigma"] == 3.0
    assert settings["bottleneck_termination_positive_weight"] == 4.0
    assert "_arch9_2_skill_term0p5_ts3_pw4_endpose" in settings["pt_run_name"]


@pytest.mark.parametrize("label", ["arch9_1_skill", "arch9_2_skill"])
def test_arch9_ignores_leftover_spatial_loss_weight(tmp_path: Path, label: str) -> None:
    config = _config(tmp_path, label)
    baseline = build_settings(config)
    config["architecture"]["spatial_loss_weight"] = 0.1
    settings = build_settings(config)
    assert settings["cond_focus_uv_loss_weight"] == 1.0
    assert settings["cond_end_xyz_loss_weight"] == 1.0
    assert settings["pt_run_name"] == baseline["pt_run_name"]


def test_arch3_layerwise_interface_is_configurable_and_named(tmp_path: Path) -> None:
    config = _config(tmp_path, "arch3_skill")
    config["architecture"]["visual_bottleneck_tokens"] = 24
    config["architecture"]["visual_bridge_last_n_layers"] = 6

    settings = build_settings(config)

    assert settings["visual_bottleneck_tokens"] == 24
    assert settings["visual_bridge_last_n_layers"] == 6
    assert "_arch3_skill_vlast6" in settings["pt_run_name"]


def test_arch4_core_exit_uses_arch3_interface_and_distinct_name(tmp_path: Path) -> None:
    config = _config(tmp_path, "arch4_skill")
    config["architecture"]["visual_bottleneck_tokens"] = 24
    config["architecture"]["visual_bridge_last_n_layers"] = 6

    settings = build_settings(config)

    assert settings["visual_bottleneck_tokens"] == 24
    assert settings["visual_bridge_last_n_layers"] == 6
    assert "_arch4_skill_vlast6" in settings["pt_run_name"]


def test_arch4_rejects_no_motion_core(tmp_path: Path) -> None:
    config = _config(tmp_path, "arch4_skill")
    config["architecture"]["visual_bridge_last_n_layers"] = 18

    with pytest.raises(ValueError, match="visual_bridge_last_n_layers <= 17"):
        build_settings(config)


@pytest.mark.parametrize("arch", ["arch5", "arch6"])
def test_uv_aligned_architecture_requires_artifact_and_names_weight(
    tmp_path: Path, arch: str
) -> None:
    config = _config(tmp_path, f"{arch}_skill")
    config["architecture"]["focus_uv_loss_weight"] = 0.25
    settings = build_settings(config)
    assert settings["cond_focus_uv_loss_weight"] == 0.25
    assert f"_{arch}_skill_uv0p25" in settings["pt_run_name"]
    (Path(settings["skillvla_dataset_dir"]).parent / "skill_focus_uv.npz").unlink()
    with pytest.raises(FileNotFoundError, match="require skill_focus_uv"):
        build_settings(config)


def test_arch1_fixed_visual_interface_is_exported(tmp_path: Path) -> None:
    settings = build_settings(_config(tmp_path, "arch1"))

    assert settings["visual_bottleneck_tokens"] == 4
    assert settings["visual_bottleneck_width"] == 256
    assert settings["visual_bottleneck_heads"] == 4
    assert settings["visual_bridge_heads"] == 8
    assert settings["visual_bridge_gate_init"] == pytest.approx(0.01)


def test_arch1_visual_token_count_is_configurable_but_hidden_from_name(tmp_path: Path) -> None:
    config = _config(tmp_path, "arch1_skill")
    config["architecture"]["visual_bottleneck_tokens"] = 8

    settings = build_settings(config)

    assert settings["visual_bottleneck_tokens"] == 8
    assert settings["pt_run_name"].endswith("_arch1_skill")
    assert "vtok" not in settings["pt_run_name"]


@pytest.mark.parametrize("tokens", [0, 3, -2])
def test_arch1_visual_token_count_must_split_across_cameras(
    tmp_path: Path, tokens: int
) -> None:
    config = _config(tmp_path, "arch1")
    config["architecture"]["visual_bottleneck_tokens"] = tokens

    with pytest.raises(ValueError, match="positive even"):
        build_settings(config)


def test_arch0_ignores_visual_token_count(tmp_path: Path) -> None:
    config = _config(tmp_path, "arch0")
    config["architecture"]["visual_bottleneck_tokens"] = 8

    settings = build_settings(config)

    assert settings["visual_bottleneck_tokens"] == 4
    assert "vtok" not in settings["pt_run_name"]


def test_arch2_bridge_depth_override_is_validated_and_named(tmp_path: Path) -> None:
    config = _config(tmp_path, "arch2_skill")
    config["architecture"]["visual_bridge_last_n_layers"] = 4

    settings = build_settings(config)

    assert settings["visual_bridge_last_n_layers"] == 4
    assert "_arch2_skill_vlast4" in settings["pt_run_name"]


def test_removed_architecture_names_fail_at_resolution(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="architecture.name"):
        build_settings(_config(tmp_path, "removed_mode"))


def test_removed_architecture_knobs_are_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["architecture"]["vsa"] = {"visual_latents_per_camera": 32}

    with pytest.raises(ValueError, match="unsupported architecture keys"):
        build_settings(config)


def test_skill_chunk_latent_probe_and_suffix(tmp_path: Path) -> None:
    config = _config(tmp_path, "arch0_skill_chunk")
    config["mask_actions_after_skill_end"] = True
    config["skill_flow"]["latent_best_of_n"] = {
        "enabled": True,
        "candidates": 5,
        "top_k": 1,
        "timesteps": 2,
        "ranking": "main",
        "fp32": True,
    }
    settings = build_settings(config)

    assert settings["skill_flow_latent_best_of_n_enabled"] is True
    assert settings["skill_flow_latent_ranking_route"] == "main"
    assert settings["skill_flow_latent_fp32"] is True
    assert settings["pt_run_name"].endswith(
        "_arch0_skill_chunk_zbest5k1m2_rank_zfp32"
    )


def test_arch0_rejects_latent_probe(tmp_path: Path) -> None:
    config = _config(tmp_path, "arch0")
    config["skill_flow"]["latent_best_of_n"] = {"enabled": True}
    with pytest.raises(ValueError, match="supported only"):
        build_settings(config)


def test_foveated_training_requires_and_exports_focus_contract(tmp_path: Path) -> None:
    config = _config(tmp_path, "arch0")
    config["vision"]["foveation"] = {
        "enabled": True,
        "shape": "circle",
        "sharp_size": 80,
        "feather": 12,
        "blur_radius": 7.0,
        "randomization": {
            "enabled": True,
            "color": {
                "enabled": True,
                "brightness": [0.7, 1.3],
                "contrast": [0.8, 1.2],
                "saturation": [0.9, 1.1],
                "hue": [-0.2, 0.2],
            },
            "crop": {"enabled": True, "offset_px": [-16, 16]},
            "blur": {"enabled": True, "blur_radius": [0.5, 3.0]},
        },
    }
    project = Path(config["project_root"])
    run_dir = project / "dataset/skillvla_dataset/source/FSQ333_test"
    info_path = run_dir / "skillvla/meta/info.json"
    info = json.loads(info_path.read_text())
    focus_path = run_dir / "skill_focus_uv.npz"
    focus_path.touch()
    info.update(
        {
            "skill_focus_uv_path": str(focus_path),
            "skill_focus_uv_camera": "agentview",
            "skill_focus_uv_normalization": "minus_one_to_one",
        }
    )
    info_path.write_text(json.dumps(info))

    settings = build_settings(config)

    assert settings["foveated_vision_enabled"] is True
    assert settings["foveation_randomization_enabled"] is True
    assert settings["foveation_shape"] == "circle"
    assert settings["foveation_color_enabled"] is True
    assert settings["foveation_crop_enabled"] is True
    assert settings["foveation_input_blur_enabled"] is True
    assert settings["foveation_hue_min"] == pytest.approx(-0.2)
    assert settings["foveation_input_blur_max_radius"] == pytest.approx(3.0)
    assert settings["pt_run_name"].endswith("_arch0_partial_fov_rand")

    config["vision"]["foveation"]["peripheral_mode"] = "black"
    black_settings = build_settings(config)
    assert black_settings["foveation_peripheral_mode"] == "black"
    assert black_settings["pt_run_name"].endswith("_arch0_partial_fov_black_rand")


def test_crop_inner_blur_training_contract_and_suffix(tmp_path: Path) -> None:
    config = _config(tmp_path, "arch0_skill")
    config["vision"]["foveation"] = {
        "enabled": True,
        "mode": "crop",
        "crop_size": 128,
        "output_size": 224,
        "inner_box": {
            "enabled": True,
            "mode": "blur",
            "size": 32,
            "line_width": 3,
        },
        "shape": "square",
        "sharp_size": 96,
        "feather": 20,
        "blur_radius": 8.0,
        "randomization": {
            "enabled": True,
            "crop": {
                "enabled": True,
                "offset_px": [-40, 40],
                "inner_box_offset_px": [-4, 4],
            },
        },
    }
    project = Path(config["project_root"])
    run_dir = project / "dataset/skillvla_dataset/source/FSQ333_test"
    info_path = run_dir / "skillvla/meta/info.json"
    info = json.loads(info_path.read_text())
    focus_path = run_dir / "skill_focus_uv.npz"
    focus_path.touch()
    info.update(
        {
            "skill_focus_uv_path": str(focus_path),
            "skill_focus_uv_camera": "agentview",
            "skill_focus_uv_normalization": "minus_one_to_one",
        }
    )
    info_path.write_text(json.dumps(info))

    settings = build_settings(config)

    assert settings["foveation_mode"] == "crop"
    assert settings["foveation_crop_size"] == 128
    assert settings["foveation_output_size"] == 224
    assert settings["foveation_inner_box_enabled"] is True
    assert settings["foveation_inner_box_mode"] == "blur"
    assert settings["foveation_inner_box_size"] == 32
    assert settings["foveation_inner_box_offset_min_px"] == -4
    assert settings["foveation_inner_box_offset_max_px"] == 4
    assert settings["pt_run_name"].endswith("_arch0_skill_crop_fov_rand")


def test_foveated_training_rejects_dataset_without_focus_artifact(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "arch0")
    config["vision"]["foveation"] = {"enabled": True}

    with pytest.raises(FileNotFoundError, match="skill_focus_uv.npz"):
        build_settings(config)


def test_foveation_and_randomization_master_switches_are_independent(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "arch0_skill")
    config["vision"]["foveation"] = {
        "enabled": False,
        "randomization": {
            "enabled": False,
            "color": {"enabled": True},
            "crop": {"enabled": True},
            "blur": {"enabled": True},
        },
    }
    baseline = build_settings(config)
    assert baseline["foveated_vision_enabled"] is False
    assert baseline["foveation_randomization_enabled"] is False
    assert baseline["pt_run_name"].endswith("_arch0_skill")

    config["vision"]["foveation"]["randomization"]["enabled"] = True
    randomized = build_settings(config)
    assert randomized["foveated_vision_enabled"] is False
    assert randomized["foveation_randomization_enabled"] is True
    assert randomized["pt_run_name"].endswith("_arch0_skill_rand")


@pytest.mark.parametrize("predictor_mode", [None, "mode2", "mode1_preskill", "mode2_preskill"])
def test_stage1_can_use_but_never_train_an_external_frozen_predictor(
    tmp_path: Path, predictor_mode,
) -> None:
    config = _config(tmp_path, "arch0")
    project = Path(config["project_root"])
    predictor = project / "outputs/predictor/checkpoints/000100/pretrained_model"
    predictor.mkdir(parents=True)
    predictor_config = {
        "type": "skill_aux",
        "train_skill_predictor": True,
        "skill_fsq_levels": [3, 3, 3],
        "skill_vocab_size": 27,
        "skill_predictor_vlm_variant": "gemma_2b",
        "skill_predictor_image_size": 224,
        "skill_predictor_reader_tokens": 6,
        "skill_predictor_reader_depth": 3,
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
    if predictor_mode is not None:
        predictor_config.update(
            skill_predictor_mode=predictor_mode,
            skill_predictor_end_state_mode="xyz",
            skill_predictor_freeze_vlm=False,
        )
    (predictor / "config.json").write_text(json.dumps(predictor_config))
    (predictor / "model.safetensors").touch()
    tokenizer = project / "models/tokenizer"
    tokenizer.mkdir(parents=True)
    for filename in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (tokenizer / filename).write_text("{}")

    config["warm_start"].update(
        {"predictor_checkpoint": str(predictor), "tokenizer": str(tokenizer)}
    )
    config["action_conditioning"] = {"training_skill_source": "predictor"}
    settings = build_settings(config)

    assert settings["skill_predictor_checkpoint_path"] == predictor
    assert settings["skill_predictor_reader_tokens"] == 6
    assert settings["skill_predictor_lora"] is True
    assert settings["skill_predictor_mode"] == (predictor_mode or "mode1")
    assert settings["skill_predictor_end_state_mode"] == ("xyz" if predictor_mode else "off")
    assert settings["skill_predictor_freeze_vlm"] is (predictor_mode is None)
    assert "train_skill_predictor" not in settings
    assert settings["pt_run_name"].endswith("_arch0_pretrained_predictor")


def test_arch15_shares_arch14_rules_with_its_own_revision(tmp_path: Path) -> None:
    config = _config(tmp_path, "arch15_skill")
    config["architecture"]["spatial_loss_weight"] = 0.1
    config["architecture"]["end_pose_mode"] = "pose"
    settings = build_settings(config)
    assert settings["architecture_revision"] == "layerwise_cond_bottleneck_xyz_skill_cond_uv_expert_end_pose_v1"
    assert settings["skill_end_pose_mode"] == "pose" and settings["cond_focus_uv_loss_weight"] == 0.1
    assert "_arch15_skill_uv0p1_endpose" in settings["pt_run_name"]
    config["vision"]["foveation"] = {"enabled": True}
    with pytest.raises(ValueError, match="original agent view"):
        build_settings(config)


def test_arch16_arch17_follow_the_wrist_rules_with_a_displacement_goal(tmp_path: Path) -> None:
    for label, revision in (("arch16_align_skill", "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_delta_align_v1"), ("arch17_align_skill", "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_delta_bridge_proprio_align_v1"), ("arch18_align_skill", "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_start_end_bridge_proprio_align_v1"), ("arch16_skill", "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_delta_v1"), ("arch17_skill", "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_delta_bridge_proprio_v1"), ("arch18_skill", "layerwise_cond_bottleneck_wrist_cond_skill_end_pose_expert_skill_start_end_bridge_proprio_v1")):
        config = _config(tmp_path / label, label)
        config["architecture"]["spatial_loss_weight"] = 0.1          # ignored, like Arch9--Arch12
        settings = build_settings(config)
        assert settings["architecture_revision"] == revision
        assert settings["cond_focus_uv_loss_weight"] == 1.0 and settings["skill_end_pose_mode"] == "xyz"
        assert settings["pt_run_name"].endswith(f"_{label}")
        config["architecture"]["end_pose_mode"] = "pose"
        with pytest.raises(ValueError, match="end_pose_mode: xyz"):
            build_settings(config)
        config["architecture"]["end_pose_mode"] = "xyz"
        config["vision"]["foveation"] = {"enabled": True}
        with pytest.raises(ValueError, match="wrist-only"):
            build_settings(config)


def test_the_align_labels_spend_spatial_loss_weight_on_the_wrist_patch_head(tmp_path: Path) -> None:
    """Arch16--Arch18 ignore spatial_loss_weight; their _align variants use it for the patch head."""
    base = build_settings(_config(tmp_path / "arch18_skill", "arch18_skill"))
    assert base["wrist_patch_align_loss_weight"] == 0.1                  # unused by the base label

    config = _config(tmp_path / "arch18_align_skill", "arch18_align_skill")
    assert build_settings(config)["wrist_patch_align_loss_weight"] == 0.1        # the default
    config["architecture"]["spatial_loss_weight"] = 0.25
    settings = build_settings(config)
    assert settings["wrist_patch_align_loss_weight"] == 0.25
    assert settings["cond_focus_uv_loss_weight"] == 1.0                  # the UV aux stays off

    ignored = _config(tmp_path / "arch18_plain", "arch18_skill")
    ignored["architecture"]["spatial_loss_weight"] = 0.25
    assert build_settings(ignored)["wrist_patch_align_loss_weight"] == 0.1

    config["architecture"]["spatial_loss_weight"] = 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        build_settings(config)


def test_the_align_norm_label_carries_the_datasets_proprio_quantiles(tmp_path: Path) -> None:
    """The policy rescales the goal itself, so the bounds have to travel inside its config."""
    settings = build_settings(_config(tmp_path / "norm", "arch18_align_norm_skill"))
    assert settings["goal_xyz_q01"] == "[-0.16,-0.28,-0.26]"          # xyz only, from meta/stats.json
    assert settings["goal_xyz_q99"] == "[0.29,0.32,0.09]"

    # Evaluation reads them back from the checkpoint, which is what makes a predictor-supplied
    # goal (also raw metres) land on the same scale without touching the predictor.
    plain = build_settings(_config(tmp_path / "plain", "arch18_align_skill"))
    assert plain["goal_xyz_q01"] == settings["goal_xyz_q01"]          # carried, simply unused

    stripped = _config(tmp_path / "missing", "arch18_align_norm_skill")
    stats = (tmp_path / "missing" / "project/dataset/skillvla_dataset/source/FSQ333_test"
             / "skillvla/meta/stats.json")
    stats.unlink()
    with pytest.raises(FileNotFoundError, match="q01/q99"):
        build_settings(stripped)


def test_arch19_arch20_follow_the_arch13_rules_with_their_own_expert_goal(tmp_path: Path) -> None:
    for label, revision, suffix in (
        ("arch19_skill", "layerwise_cond_bottleneck_xyz_skill_cond_uv_expert_skill_delta_v1", "_arch19_skill_uv0p1"),
        ("arch20_skill", "layerwise_cond_bottleneck_xyz_skill_cond_uv_v1", "_arch20_skill_uv0p1"),
    ):
        config = _config(tmp_path / label, label)
        config["architecture"]["spatial_loss_weight"] = 0.1          # UV aux, like Arch13--Arch15
        settings = build_settings(config)
        assert settings["architecture_revision"] == revision
        assert settings["architecture"] == "layerwise_cond_bottleneck"
        assert settings["cond_focus_uv_loss_weight"] == 0.1 and settings["skill_end_pose_mode"] == "xyz"
        assert suffix in settings["pt_run_name"]
        config["architecture"]["end_pose_mode"] = "pose"
        with pytest.raises(ValueError, match="xyz"):
            build_settings(config)
        config["architecture"]["end_pose_mode"] = "xyz"
        config["vision"]["foveation"] = {"enabled": True}
        with pytest.raises(ValueError, match="original agent view"):
            build_settings(config)
