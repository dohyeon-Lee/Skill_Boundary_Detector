from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

import pytest

CONFIGS = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_skillVLA"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


VSA = _load("newtask_ft_vsa_config", CONFIGS / "FT/VSA/src/newtask_ft_vsa_config.py")
AUX = _load("terminator_train_config_newtask", CONFIGS / "PT/Terminator/src/terminator_train_config.py")
JOINT = _load(
    "newtask_ft_joint_config",
    CONFIGS / "FT/Joint/src/newtask_ft_joint_config.py",
)


def _dataset(root: Path, source: str, run: str, *, fsq=("fsq_run", "2000"), focus_uv=True, levels=(3, 3, 3)) -> Path:
    run_dir = root / "dataset/skillvla_dataset" / source / run
    (run_dir / "skillvla/meta").mkdir(parents=True)
    info = {
        "skill_fsq_levels": list(levels),
        "skill_pmax": 10,
        "skill_jitter_late_start_pmax": 5,
        "skill_observed_max_length": 171,
        "proprio_grounding": "episode_start_xyz",
        "features": {"observation.state": {"shape": [8]}, "action": {"shape": [7]}},
    }
    if focus_uv:
        (run_dir / "skill_focus_uv.npz").touch()
        info["skill_focus_uv_path"] = str(run_dir / "skill_focus_uv.npz")
        info["skill_focus_uv_normalization"] = "minus_one_to_one"
    (run_dir / "skillvla/meta/info.json").write_text(json.dumps(info))
    (run_dir / "FSQ.pt").touch()
    if fsq is not None:
        (run_dir / "fsq_source.json").write_text(
            json.dumps({"source_fsq_run_name": fsq[0], "source_fsq_checkpoint": fsq[1]})
        )
    return run_dir


def _checkpoint(root: Path, group: str, run: str, config: dict) -> Path:
    path = root / "outputs" / group / run / "checkpoints/050000/pretrained_model"
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config))
    (path / "model.safetensors").touch()
    (path / "policy_preprocessor.json").write_text("{}")
    return path


def _vsa_setup(tmp_path: Path, *, label="arch13_skill", **dataset_kwargs) -> dict:
    pt_run = _dataset(tmp_path, "pt_source", "FSQ333_pt")
    _dataset(tmp_path, "new_task", "FSQ333_ft", **dataset_kwargs)
    (tmp_path / "models/dino").mkdir(parents=True)
    checkpoint = _checkpoint(
        tmp_path,
        "skillVLA_stage1/VSA",
        "bs32_pt_run",
        {
            "type": "skill_expert",
            "architecture_label": label,
            "skill_fsq_levels": [3, 3, 3],
            "skill_vocab_size": 27,
            "skill_code_space_id": "FSQ333_pt",
            "proprio_grounding": "episode_start_xyz",
            "fsq_path": str(pt_run / "FSQ.pt"),
            "dino_model_path": "/other/machine/models/dino",
            "tokenizer_path": "/other/machine/models/tokenizer",
            "visual_bridge_last_n_layers": 1,
            "skill_flow_enabled": label.endswith(("_skill", "_skill_chunk")),
            "skill_flow_max_length": 186,
            "skill_flow_target": "canonical",
            "input_features": {"observation.state": {"shape": [8]}},
            "output_features": {"action": {"shape": [7]}},
        },
    )
    return {
        "project_root": str(tmp_path),
        "dataset_root": "dataset",
        "outputs_root": "outputs",
        "stage1_component": "VSA",
        "newtask_ft": True,
        "dataset": {"skillvla_root": "skillvla_dataset", "source": "new_task", "run": "FSQ333_ft"},
        "warm_start": {"vsa_checkpoint": str(checkpoint.relative_to(tmp_path))},
        "run": {"suffix": "t1"},
    }


def test_vsa_inherits_checkpoint_and_uses_new_dataset(tmp_path: Path) -> None:
    settings = VSA.build_settings(_vsa_setup(tmp_path))
    assert settings["architecture_label"] == "arch13_skill"
    assert settings["freeze_vision_encoder"] is True
    assert settings["transition_jitter_pmax"] == 10
    assert settings["transition_jitter_late_start_pmax"] == 5
    assert settings["dino_model_path"] == tmp_path / "models/dino"
    assert settings["fsq_path"] == tmp_path / "dataset/skillvla_dataset/new_task/FSQ333_ft/FSQ.pt"
    assert settings["run_name"] == "bs32_pt_run_50k_new_task_ft_bs32_t1"
    assert settings["output_dir"].parent == tmp_path / "outputs/skillVLA_FT/VSA"


def test_vsa_appends_common_suffix_after_component_suffix(tmp_path: Path) -> None:
    config = _vsa_setup(tmp_path)
    config["run"]["common_suffix"] = "shared"

    settings = VSA.build_settings(config)

    assert settings["run_name"].endswith("_t1_shared")


@pytest.mark.parametrize("label", [
    "both_3", "fullskill_bspline_3_firstlayer", "fullskill_bspline_3_lastlayer",
    "fullskill_bspline_3_lastlayer_lastvis", "fullskill_bspline_3_lastlayer_lastvis_first",
    "fullskill_bspline_3_lastlayer_lastvis_last",
    "fullskill_bspline_3_firstlayer_lastvis", "fullskill_bspline_3_firstlayer_lastvis_first",
    "fullskill_bspline_3_firstlayer_lastvis_last",
    "fullskill_bspline_3_lastlayer_lastvis_current",
    "fullskill_bspline_3_lastlayer_lastvis_current_preskill",
    "fullskill_bspline_3_lastlayer_lastvis_current_none",
    "uv_fullskill_bspline_3_lastlayer_lastvis",
    "uv_fullskill_bspline_3_lastlayer_lastvis_current",
    "uv_fullskill_bspline_3_lastlayer_lastvis_current_preskill",
    "uv_fullskill_bspline_3_lastlayer_lastvis_current_none",
    "uv_fullskill_bspline_3_lastlayer_lastvis_diffaux",
    "uv_fullskill_bspline_3_lastlayer_lastvis_current_diffaux",
    "uv_fullskill_bspline_3_lastlayer_lastvis_current_preskill_diffaux",
    "uv_fullskill_bspline_3_lastlayer_lastvis_current_none_diffaux",
])
@pytest.mark.parametrize("destination,keep", [("scratch2", 1), ("scratch", 2), ("scratch2", -1)])
def test_vsa_archive_settings_are_opt_in_and_preserve_training_contract(tmp_path, label, destination, keep):
    config = _vsa_setup(tmp_path, label=label)
    config["training"] = {"schedule": {"steps": 10000, "save_every": 1000}}
    baseline = VSA.build_settings(config)
    assert baseline["checkpoint_archive"] == ""
    assert baseline["keep_checkpoints"] == -1

    config["training"].update(checkpoint_archive=destination, keep_checkpoints=keep)
    settings = VSA.build_settings(config)

    assert settings["checkpoint_archive"] == destination
    assert settings["keep_checkpoints"] == keep
    assert settings["steps"] == 10000 and settings["save_freq"] == 1000
    assert {key for key in settings if settings[key] != baseline[key]} == {
        key for key, value in {"checkpoint_archive": destination, "keep_checkpoints": keep}.items()
        if value != baseline[key]
    }


@pytest.mark.parametrize("keep,destination", [
    (0, "scratch2"), (-2, "scratch2"), (True, "scratch2"), (1.5, "scratch2"),
    ("1", "scratch2"), (None, "scratch2"), (1, ""), (1, "local"),
])
def test_vsa_rejects_unsafe_checkpoint_retention(tmp_path, keep, destination):
    config = _vsa_setup(tmp_path)
    config["training"] = {"checkpoint_archive": destination, "keep_checkpoints": keep}
    with pytest.raises(ValueError, match="checkpoint"):
        VSA.build_settings(config)


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("archive_enabled", [False, True])
def test_vsa_launcher_passes_archive_settings_on_fresh_and_resumed_runs(resume, archive_enabled):
    script = (CONFIGS / "FT/VSA/src/train.sbatch").read_text()
    match = re.search(r"(?ms)^TRAIN_ARGS\+=\(\n.*?^\)", script)
    assert match is not None
    # The common append must be outside/after the fresh-vs-resume conditional.
    assert match.start() > script.rindex("\nfi\n")
    assert match.end() < script.index("PYTORCH_ALLOC_CONF=")
    env = dict(os.environ)
    env.pop("CHECKPOINT_ARCHIVE", None)
    env.pop("KEEP_CHECKPOINTS", None)
    if archive_enabled:
        env.update(CHECKPOINT_ARCHIVE="scratch2", KEEP_CHECKPOINTS="1")
    initial = "--resume=true" if resume else "--policy.path=/source/pretrained_model"
    result = subprocess.run(
        ["bash", "-c", f"set -eu\nTRAIN_ARGS=({initial})\n{match.group()}\nprintf '%s\\0' \"${{TRAIN_ARGS[@]}}\""],
        env=env, capture_output=True, text=True, check=True,
    )
    assert result.stdout.split("\0")[:-1] == [
        initial,
        "--checkpoint_archive=" + ("scratch2" if archive_enabled else ""),
        "--keep_checkpoints=" + ("1" if archive_enabled else "-1"),
    ]


@pytest.mark.parametrize("variant", range(1, 7))
@pytest.mark.parametrize("family", ["direct", "residual"])
@pytest.mark.parametrize("dgd", [False, True])
@pytest.mark.parametrize("bridge_layers", [1, 9])
def test_direct_patch_ft_inherits_window_checkpoint(tmp_path, variant, family, dgd, bridge_layers):
    label = f"{family}_patch_{variant}" + ("_dGD" if dgd else "")
    config = _vsa_setup(tmp_path, label=label)
    checkpoint = tmp_path / config["warm_start"]["vsa_checkpoint"]
    path = checkpoint / "config.json"
    source = json.loads(path.read_text())
    source.update(architecture_revision=f"layerwise_cond_bottleneck_{family}_patch_v1",
                  visual_bridge_last_n_layers=bridge_layers,
                  skill_flow_enabled=True, skill_flow_target="residual_window",
                  skill_flow_state_conditioned=True, skill_flow_max_length=40,
                  chunk_size=10, n_action_steps=10)
    path.write_text(json.dumps(source))
    settings = VSA.build_settings(config)
    assert settings["architecture_label"] == label
    assert settings["trainable_expert_layers"] == bridge_layers
    assert settings["unfreeze_action_head"] is False
    assert settings["unfreeze_skill_layers"] is False
    assert settings["full_unfreeze"] is False
    assert settings["skill_flow_active"] is False  # default FT freezes the base/head
    assert json.loads(path.read_text())["skill_flow_max_length"] == 40


@pytest.mark.parametrize("label", ["arch0_skill", "arch3_skill", "arch1_skill_chunk"])
def test_vsa_rejects_shared_route_architectures(tmp_path: Path, label: str) -> None:
    with pytest.raises(ValueError, match="shared-route Arch0--Arch3"):
        VSA.build_settings(_vsa_setup(tmp_path, label=label))


@pytest.mark.parametrize("label", ["arch4_skill", "arch10_1_skill", "arch12_2_skill", "arch8_1_skill", "arch14_skill", "arch15_skill", "arch16_skill", "arch17_skill", "arch18_skill", "arch16_align_skill", "arch17_align_skill", "arch18_align_skill", "arch18_align_norm_skill", "arch19_skill", "arch20_skill"])
def test_vsa_accepts_core_exit_architectures(tmp_path: Path, label: str) -> None:
    assert VSA.build_settings(_vsa_setup(tmp_path, label=label))["architecture_label"] == label


@pytest.mark.parametrize(
    "label",
    ["wristonly_1", "both_2", "wristonly_lit_3", "both_lit_5", "future_core_exit_v1"],
)
def test_vsa_does_not_require_new_core_exit_names_in_a_second_allow_list(
    tmp_path: Path, label: str
) -> None:
    assert VSA.build_settings(_vsa_setup(tmp_path, label=label))["architecture_label"] == label


def test_vsa_rejects_different_fsq_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="different FSQ model"):
        VSA.build_settings(_vsa_setup(tmp_path, fsq=("other_fsq", "2000")))


def test_vsa_requires_focus_uv_for_spatial_architectures(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="skill_focus_uv"):
        VSA.build_settings(_vsa_setup(tmp_path, focus_uv=False))
    assert VSA.build_settings(_vsa_setup(tmp_path / "b", label="arch4_skill", focus_uv=False))


@pytest.mark.parametrize("label", ["arch7_skill", "arch18_align", "both_lit_4_dedicated", "fullskill_2"])
def test_vsa_xyz_only_ft_accepts_dataset_without_uv(tmp_path, label):
    settings = VSA.build_settings(_vsa_setup(tmp_path, label=label, focus_uv=False))
    assert settings["architecture_label"] == label


@pytest.mark.parametrize("label", ["both_3", "direct_patch_2", "fullskill_1"])
def test_vsa_top_patch_ft_still_requires_uv(tmp_path, label):
    with pytest.raises(FileNotFoundError, match="skill_focus_uv"):
        VSA.build_settings(_vsa_setup(tmp_path, label=label, focus_uv=False))


def test_vsa_rejects_model_sections_in_yaml(tmp_path: Path) -> None:
    config = _vsa_setup(tmp_path)
    config["architecture"] = {"name": "arch13_skill"}
    with pytest.raises(ValueError, match="inherits the model contract"):
        VSA.build_settings(config)


def _aux_setup(tmp_path: Path, component: str, **dataset_kwargs) -> dict:
    pt_run = _dataset(tmp_path, "pt_source", "FSQ333_pt")
    _dataset(tmp_path, "new_task", "FSQ333_ft", **dataset_kwargs)
    tokenizer = tmp_path / "models/tokenizer"
    tokenizer.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (tokenizer / name).write_text("{}")
    common = {
        "type": "skill_aux",
        "skill_fsq_levels": [3, 3, 3],
        "skill_vocab_size": 27,
        "skill_code_space_id": "FSQ333_pt",
        "fsq_path": str(pt_run / "FSQ.pt"),
        "training_batch_size": 16,
        "dataset_source_lineage": ["pt_source"],
        "run_suffix_lineage": [],
        "tokenizer_path": "models/tokenizer",
    }
    predictor = _checkpoint(
        tmp_path, "skillVLA_stage1/Predictor", "p",
        {**common, "train_skill_predictor": True, **AUX._predictor_contract({})},
    )
    terminator = _checkpoint(
        tmp_path, "skillVLA_stage1/Terminator", "t",
        {
            **common,
            "train_terminator": True,
            "terminator_context": "none",
            "terminator_arch": "fusion",
            "terminator_vision_backbone": "dino",
            "terminator_freeze_vision_encoder": True,
            "terminator_termination_only": True,
        },
    )
    return {
        "project_root": str(tmp_path),
        "dataset_root": "dataset",
        "outputs_root": "outputs",
        "stage1_component": component,
        "newtask_ft": True,
        "mode": "ft",
        "dataset": {"skillvla_root": "skillvla_dataset", "source": "new_task", "run": "FSQ333_ft"},
        "warm_start": {
            "vsa_checkpoint": "unused",
            "predictor_checkpoint": str(predictor),
            "terminator_checkpoint": str(terminator),
        },
    }


@pytest.mark.parametrize(
    ("component", "trained", "skipped"),
    [("Predictor", "train_skill_predictor", "train_terminator"),
     ("Terminator", "train_terminator", "train_skill_predictor")],
)
def test_aux_component_uses_only_its_checkpoint(tmp_path: Path, component, trained, skipped) -> None:
    settings = AUX.build_settings(_aux_setup(tmp_path, component))
    assert settings[trained] is True and settings[skipped] is False
    assert settings["initialization_mode"] == "ft"
    # The new run tag differs, but the shared FSQ model keeps the code space.
    assert settings["skill_code_space_id"] == "FSQ333_pt"
    assert json.loads(settings["dataset_source_lineage"]) == ["pt_source", "new_task"]
    assert settings["output_dir"].parent == tmp_path / "outputs/skillVLA_FT" / component


def test_aux_rejects_different_fsq_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="different FSQ model"):
        AUX.build_settings(_aux_setup(tmp_path, "Predictor", fsq=("other", "1")))


def test_aux_can_skip_fsq_verification(tmp_path: Path) -> None:
    config = _aux_setup(tmp_path, "Terminator", fsq=None)
    with pytest.raises(FileNotFoundError, match="fsq_source.json"):
        AUX.build_settings(config)
    config["fsq"] = {"verify_source": False}
    assert AUX.build_settings(config)["skill_code_space_id"] == "FSQ333_pt"


def test_shipped_yaml_layers_resolve_common_file() -> None:
    from_loader = VSA.load_stage1_component_config(CONFIGS / "FT/VSA/vsa_ft_config.yaml")
    assert from_loader["newtask_ft"] is True
    assert from_loader["dataset"]["source"]
    assert "vsa_checkpoint" in from_loader["warm_start"]
    for name in ("Predictor/predictor_ft_config.yaml", "Terminator/terminator_ft_config.yaml"):
        merged = VSA.load_stage1_component_config(CONFIGS / "FT" / name)
        assert merged["mode"] == "ft" and merged["newtask_ft"] is True
        assert "stage1_common" not in merged and "newtask_ft_common" not in merged
    joint = VSA.load_stage1_component_config(
        CONFIGS / "FT/Joint/joint_ft_config.yaml"
    )
    assert joint["stage1_component"] == "Joint"
    assert joint["loss"]["xyz_to_skill"] is False


def test_vsa_rebases_fsq_path_recorded_on_another_cluster(tmp_path: Path) -> None:
    config = _vsa_setup(tmp_path)
    checkpoint = tmp_path / config["warm_start"]["vsa_checkpoint"] / "config.json"
    saved = json.loads(checkpoint.read_text())
    saved["fsq_path"] = "/data1/other/dataset_filtered/skillvla_dataset/pt_source/FSQ333_pt/FSQ.pt"
    checkpoint.write_text(json.dumps(saved))
    assert VSA.build_settings(config)["architecture_label"] == "arch13_skill"


@pytest.mark.parametrize(("step", "label"), [("170000", "170k"), ("085000", "85k"), ("001500", "001500"), ("best", "best")])
def test_vsa_step_label(step: str, label: str) -> None:
    assert VSA._step_label(step) == label


def _joint_setup(tmp_path: Path, *, label="arch13_skill", terminator_contract=None) -> dict:
    config = _vsa_setup(tmp_path, label=label)
    pt_run = tmp_path / "dataset/skillvla_dataset/pt_source/FSQ333_pt"
    predictor_contract = AUX._predictor_contract(
        {
            "skill_predictor": {
                "spatial_target": "xyz",
                "freeze_vlm": True,
                "lora": {"enabled": False},
            }
        }
    )
    _checkpoint(
        tmp_path,
        "skillVLA_stage1/Predictor",
        "predictor_xyz",
        {
            "type": "skill_aux",
            "train_skill_predictor": True,
            "skill_fsq_levels": [3, 3, 3],
            "skill_vocab_size": 27,
            "skill_code_space_id": "FSQ333_pt",
            "fsq_path": str(pt_run / "FSQ.pt"),
            "tokenizer_max_length": 200,
            **predictor_contract,
        },
    )
    _checkpoint(
        tmp_path,
        "skillVLA_stage1/Terminator",
        "terminator_base",
        {
            "type": "skill_aux",
            "train_terminator": True,
            "skill_fsq_levels": [3, 3, 3],
            "skill_vocab_size": 27,
            "skill_code_space_id": "FSQ333_pt",
            "fsq_path": str(pt_run / "FSQ.pt"),
            "terminator_context": "proprio",
            "terminator_cameras": "top",
            "terminator_arch": "fusion",
            "terminator_vision_backbone": "dino",
            "terminator_freeze_vision_encoder": True,
            "terminator_termination_only": True,
            **(terminator_contract or {}),
        },
    )
    config.update(
        {
            "stage1_component": "Joint",
            "warm_start": {
                "vsa_checkpoint": {"run": "bs32_pt_run", "checkpoint": "050000"},
                "predictor_checkpoint": {"run": "predictor_xyz", "checkpoint": "050000"},
                "terminator_checkpoint": {"run": "terminator_base", "checkpoint": "050000"},
            },
            "loss": {"route_timesteps": 2, "xyz_to_skill": False},
            "training": {
                "dataloader": {
                    "batch_size": 16,
                    "workers": 24,
                    "gpus": 1,
                    "prefetch_factor": 2,
                    "pin_memory": True,
                    "persistent_workers": True,
                },
                "gradient_checkpointing": False,
            },
            "slurm": {"cpus": 32, "memory": "160G"},
        }
    )
    return config


def test_joint_resolves_two_warm_starts_and_uses_its_own_output_group(tmp_path: Path) -> None:
    config = _joint_setup(tmp_path)
    settings = JOINT.build_settings(config)
    assert settings["predictor_checkpoint_path"].name == "pretrained_model"
    assert settings["skill_predictor_end_state_mode"] == "xyz"
    assert settings["skill_predictor_freeze_vlm"] is True
    assert settings["joint_xyz_to_skill"] is False
    assert settings["newtask_joint_terminator_enabled"] is True
    assert settings["train_terminator"] is True
    assert settings["terminator_checkpoint_path"].name == "pretrained_model"
    assert settings["batch_size"] == 16
    assert settings["num_workers"] == 24
    assert settings["dataloader_prefetch_factor"] == 2
    assert settings["dataloader_pin_memory"] is True
    assert settings["dataloader_persistent_workers"] is True
    assert settings["gradient_checkpointing"] is False
    assert settings["train_cpus_per_task"] == 32
    assert settings["train_mem"] == "160G"
    assert settings["output_dir"].parent == tmp_path / "outputs/skillVLA_FT/Joint"
    assert "dsbc_enabled" not in settings
    config["predictor"] = {"freeze_vlm": False, "vlm_lr_scale": 0.05}
    unfrozen = JOINT.build_settings(config)
    assert unfrozen["skill_predictor_freeze_vlm"] is False
    assert unfrozen["skill_predictor_detach_vlm"] is False
    assert unfrozen["skill_predictor_vlm_lr_scale"] == pytest.approx(0.05)
    config["terminator"] = {"enabled": False}
    config["warm_start"].pop("terminator_checkpoint")
    without_terminator = JOINT.build_settings(config)
    assert without_terminator["newtask_joint_terminator_enabled"] is False
    assert without_terminator["train_terminator"] is False
    assert "_term" not in without_terminator["run_name"]


@pytest.mark.parametrize("predictor_mode", [None, "mode2", "mode1_preskill", "mode2_preskill"])
def test_joint_inherits_predictor_architecture_with_legacy_default(tmp_path, predictor_mode):
    config = _joint_setup(tmp_path)
    path = (
        tmp_path / "outputs/skillVLA_stage1/Predictor/predictor_xyz"
        / "checkpoints/050000/pretrained_model/config.json"
    )
    source = json.loads(path.read_text())
    if predictor_mode is None:
        source.pop("skill_predictor_mode", None)
    else:
        source["skill_predictor_mode"] = predictor_mode
    path.write_text(json.dumps(source))
    settings = JOINT.build_settings(config)
    assert settings["skill_predictor_mode"] == (predictor_mode or "mode1")
    assert settings["skill_predictor_end_state_mode"] == "xyz"


@pytest.mark.parametrize("variant", range(1, 7))
@pytest.mark.parametrize("family", ["direct", "residual"])
@pytest.mark.parametrize("dgd", [False, True])
@pytest.mark.parametrize("bridge_layers", [1, 9])
def test_joint_accepts_window_variants_and_preserves_newterm7_contract(
    tmp_path, variant, family, dgd, bridge_layers,
):
    label = f"{family}_patch_{variant}" + ("_dGD" if dgd else "")
    config = _joint_setup(tmp_path, label=label, terminator_contract={
        "terminator_architecture_label": "newterm7", "terminator_post_end_steps": 10,
        "terminator_start_proprio": True, "terminator_current_proprio": False,
        "terminator_goal_xyz": True, "terminator_end_target_sigma": 1.0,
        "terminator_end_pos_weight": 2.0,
    })
    path = tmp_path / "outputs/skillVLA_stage1/VSA/bs32_pt_run/checkpoints/050000/pretrained_model/config.json"
    source = json.loads(path.read_text())
    source.update(skill_flow_enabled=True, skill_flow_target="residual_window",
                  skill_flow_state_conditioned=True, chunk_size=10, skill_flow_max_length=40,
                  visual_bridge_last_n_layers=bridge_layers)
    path.write_text(json.dumps(source))
    settings = JOINT.build_settings(config)
    assert settings["architecture_label"] == label
    assert settings["trainable_expert_layers"] == bridge_layers
    assert settings["joint_route_target"] == "residual_window"
    assert settings["unfreeze_action_head"] is False
    assert settings["unfreeze_skill_layers"] is False
    assert settings["skill_flow_active"] is False
    assert settings["freeze_vision_encoder"] is True
    assert "dsbc_enabled" not in settings
    assert settings["terminator_post_end_steps"] == 10
    assert settings["terminator_current_proprio"] is False
    assert settings["terminator_end_target_sigma"] == 1.0
    assert settings["terminator_end_pos_weight"] == 2.0


@pytest.mark.parametrize("label", ["both_3", "both_lit_4", "both_lit_5"])
def test_joint_still_accepts_existing_canonical_models(tmp_path, label):
    config = _joint_setup(tmp_path, label=label)
    path = tmp_path / "outputs/skillVLA_stage1/VSA/bs32_pt_run/checkpoints/050000/pretrained_model/config.json"
    source = json.loads(path.read_text())
    source["skill_flow_enabled"] = True
    path.write_text(json.dumps(source))
    settings = JOINT.build_settings(config)
    assert settings["joint_route_target"] == "canonical"
    assert settings["unfreeze_action_head"] is False
    assert settings["skill_flow_active"] is False


def test_joint_hides_adaptation_and_keeps_old_default_snapshots_readable(tmp_path):
    import yaml
    visible = yaml.safe_load((CONFIGS / "FT/Joint/joint_ft_config.yaml").read_text())
    assert "adaptation" not in visible
    config = _joint_setup(tmp_path)
    baseline = JOINT.build_settings(config)
    config["adaptation"] = {"train_dino": False, "dino_lr_scale": 1.0, "dsbc": False}
    legacy = JOINT.build_settings(config)
    assert baseline["run_name"] == legacy["run_name"]
    assert legacy["freeze_vision_encoder"] is True
    assert "dsbc_enabled" not in legacy


@pytest.mark.parametrize("key,value", [
    ("train_dino", True), ("dino_lr_scale", 0.5), ("dsbc", True),
    ("skill_flow_loss", True), ("unfreeze_action_head", True),
])
def test_joint_rejects_removed_adaptation_choices(tmp_path, key, value):
    config = _joint_setup(tmp_path)
    config["adaptation"] = {key: value}
    with pytest.raises(ValueError, match="not supported in Joint|is fixed to"):
        JOINT.build_settings(config)


def test_aux_checkpoint_under_the_wrong_group_is_relocated(tmp_path: Path) -> None:
    config = _aux_setup(tmp_path, "Terminator")
    real = Path(config["warm_start"]["terminator_checkpoint"])
    legacy_run = tmp_path / "outputs/skillVLA_terminator/t"
    legacy_run.parent.mkdir(parents=True)
    real.parents[2].rename(legacy_run)  # the run really lives in the legacy group
    settings = AUX.build_settings(config)  # yaml still says skillVLA_stage1/Terminator/t
    assert settings["terminator_checkpoint_path"] == legacy_run / "checkpoints/050000/pretrained_model"


def test_warm_start_accepts_run_and_checkpoint_selectors(tmp_path: Path) -> None:
    vsa = _vsa_setup(tmp_path / "vsa")
    vsa["warm_start"] = {"vsa_checkpoint": {"run": "bs32_pt_run", "checkpoint": "last"}}
    settings = VSA.build_settings(vsa)
    assert settings["vsa_checkpoint_path"].parts[-4:] == ("bs32_pt_run", "checkpoints", "050000", "pretrained_model")
    assert "_50k_" in settings["run_name"]  # "last" resolved to the concrete step

    aux = _aux_setup(tmp_path / "aux", "Terminator")
    legacy_run = tmp_path / "aux/outputs/skillVLA_terminator/t"
    legacy_run.parent.mkdir(parents=True)
    Path(aux["warm_start"]["terminator_checkpoint"]).parents[2].rename(legacy_run)
    aux["warm_start"] = {
        "vsa_checkpoint": {"run": "ignored", "checkpoint": "1"},
        "predictor_checkpoint": {"run": "", "checkpoint": ""},
        "terminator_checkpoint": {"run": "t", "checkpoint": "050000"},
    }
    resolved = AUX.build_settings(aux)
    assert resolved["terminator_checkpoint_path"] == legacy_run / "checkpoints/050000/pretrained_model"

    aux["warm_start"]["terminator_checkpoint"] = {"run": "missing", "checkpoint": "050000"}
    with pytest.raises(FileNotFoundError, match="not found in"):
        AUX.build_settings(aux)
    aux["warm_start"]["terminator_checkpoint"] = {"run": "a/b", "checkpoint": "1"}
    with pytest.raises(ValueError, match="names, not paths"):
        AUX.build_settings(aux)


def test_vsa_removed_choices_are_fixed_but_old_default_snapshots_still_load(tmp_path: Path) -> None:
    config = _vsa_setup(tmp_path)
    base = VSA.build_settings(config)
    assert base["freeze_vision_encoder"] is True
    assert base["unfreeze_action_head"] is False
    assert base["full_unfreeze"] is False
    assert base["dsbc_enabled"] is False
    config["adaptation"] = {
        "train_dino": False,
        "dino_lr_scale": 1.0,
        "unfreeze_action_head": False,
        "full_unfreeze": False,
        "skill_flow_loss": False,
        "dsbc": False,
    }
    assert VSA.build_settings(config)["run_name"] == base["run_name"]
    for key, value in (
        ("train_dino", True),
        ("dino_lr_scale", 0.5),
        ("unfreeze_action_head", True),
        ("full_unfreeze", True),
        ("dsbc", True),
    ):
        config["adaptation"] = {key: value}
        with pytest.raises(ValueError, match="no longer a standalone VSA FT choice"):
            VSA.build_settings(config)
    config["adaptation"] = {"normalization": "dataset"}
    with pytest.raises(ValueError, match="Unsupported adaptation settings"):
        VSA.build_settings(config)


def test_vsa_skill_flow_variants_need_a_dataset_that_fits_the_horizon(tmp_path: Path) -> None:
    # A checkpoint without a skill-flow route cannot enable the sole switch.
    plain = _vsa_setup(tmp_path / "plain", label="arch13")
    plain["adaptation"] = {"skill_flow_loss": True}
    with pytest.raises(ValueError, match="skill_flow_enabled=true"):
        VSA.build_settings(plain)

    config = _vsa_setup(tmp_path / "long")
    config["adaptation"] = {"skill_flow_loss": True}
    info = tmp_path / "long/dataset/skillvla_dataset/new_task/FSQ333_ft/skillvla/meta/info.json"
    meta = json.loads(info.read_text()); meta["skill_observed_max_length"] = 300
    info.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="does not fit"):
        VSA.build_settings(config)


def test_vsa_skill_flow_loss_unfreezes_the_expert_route(tmp_path: Path) -> None:
    config = _vsa_setup(tmp_path)
    base = VSA.build_settings(config)
    assert base["skill_flow_active"] is False and "_noskill" not in base["run_name"]
    config["adaptation"] = {"skill_flow_loss": False}
    assert VSA.build_settings(config)["run_name"] == base["run_name"]
    config["adaptation"] = {"skill_flow_loss": True}
    on = VSA.build_settings(config)
    assert on["skill_flow_active"] is True
    assert on["unfreeze_action_head"] is True
    assert on["unfreeze_skill_layers"] is True
    assert on["full_unfreeze"] is False
    assert on["run_name"].endswith("_skillflow_t1")


@pytest.mark.parametrize("label", ["both_3", "direct_patch_2", "residual_patch_2_dGD"])
@pytest.mark.parametrize("bridge_layers", [1, 9])
def test_vsa_can_freeze_all_expert_layers_without_changing_other_ft_choices(tmp_path, label, bridge_layers):
    config = _vsa_setup(tmp_path, label=label)
    checkpoint = tmp_path / config["warm_start"]["vsa_checkpoint"] / "config.json"
    source = json.loads(checkpoint.read_text())
    source["visual_bridge_last_n_layers"] = bridge_layers
    checkpoint.write_text(json.dumps(source))
    baseline = VSA.build_settings(config)
    assert baseline["freeze_expert"] is False
    config["adaptation"] = {"freeze_expert": False}
    assert VSA.build_settings(config) == baseline
    config["adaptation"] = {"freeze_expert": True, "skill_flow_loss": False}
    frozen = VSA.build_settings(config)
    assert frozen["freeze_expert"] is True
    assert frozen["trainable_expert_layers"] == 0
    assert frozen["skill_flow_active"] is False
    assert frozen["unfreeze_action_head"] is False
    assert frozen["unfreeze_skill_layers"] is False
    assert frozen["run_name"].endswith("_frozenexpert_t1")
    changed = {key for key in baseline if frozen[key] != baseline[key]}
    assert changed == {"freeze_expert", "trainable_expert_layers", "run_name", "output_dir"}
    assert json.loads(checkpoint.read_text()) == source


@pytest.mark.parametrize("label", [
    "fullskill_vis_2_lastlayer", "fullskill_vis_2_lastlayer_lastvis",
    "fullskill_bspline_3_lastlayer", "fullskill_bspline_3_lastlayer_lastvis_current",
    "uv_fullskill_bspline_3_lastlayer_lastvis_current_diffaux",
])
def test_vsa_frozen_expert_can_train_visual_inputs_with_skill_auxiliary_loss(tmp_path, label):
    config = _vsa_setup(tmp_path, label=label)
    path = tmp_path / config["warm_start"]["vsa_checkpoint"] / "config.json"
    source = json.loads(path.read_text())
    bspline = "bspline" in label
    source.update(skill_flow_enabled=not bspline, skill_flow_max_length=0 if bspline else 186)
    path.write_text(json.dumps(source))
    config["adaptation"] = {"freeze_expert": True, "skill_flow_loss": False}
    off = VSA.build_settings(config)
    config["adaptation"] = {"freeze_expert": True, "skill_flow_loss": True}
    on = VSA.build_settings(config)
    assert on["freeze_expert"] and on["skill_flow_active"]
    assert not on["unfreeze_skill_layers"] and not on["unfreeze_action_head"]
    assert on["trainable_expert_layers"] == 0
    kind = "bspline" if bspline else "flow"
    assert on["skill_loss_kind"] == kind
    assert on["run_name"].endswith(f"_frozenexpert_skill{kind}_t1")
    assert on["output_dir"] != off["output_dir"]
    assert on["vsa_checkpoint_path"] == off["vsa_checkpoint_path"]
    assert json.loads(path.read_text()) == source


def test_frozen_vis_skill_loss_still_requires_a_sufficient_flow_horizon(tmp_path):
    config = _vsa_setup(tmp_path)
    config["adaptation"] = {"freeze_expert": True, "skill_flow_loss": True}
    info = tmp_path / "dataset/skillvla_dataset/new_task/FSQ333_ft/skillvla/meta/info.json"
    meta = json.loads(info.read_text())
    meta["skill_observed_max_length"] = 300
    info.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="does not fit"):
        VSA.build_settings(config)
