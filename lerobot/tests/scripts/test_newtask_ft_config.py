from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

CONFIGS = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_skillVLA"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


VSA = _load("newtask_ft_vsa_config", CONFIGS / "NewTask_FT/VSA/src/newtask_ft_vsa_config.py")
AUX = _load("terminator_train_config_newtask", CONFIGS / "terminator/src/terminator_train_config.py")


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
            "skill_code_space_id": "FSQ333_pt",
            "proprio_grounding": "episode_start_xyz",
            "fsq_path": str(pt_run / "FSQ.pt"),
            "dino_model_path": "/other/machine/models/dino",
            "tokenizer_path": "/other/machine/models/tokenizer",
            "visual_bridge_last_n_layers": 1,
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
    assert settings["output_dir"].parent == tmp_path / "outputs/skillVLA_NewTask_FT/VSA"


@pytest.mark.parametrize("label", ["arch0_skill", "arch3_skill", "arch1_skill_chunk"])
def test_vsa_rejects_shared_route_architectures(tmp_path: Path, label: str) -> None:
    with pytest.raises(ValueError, match="Arch4--Arch16"):
        VSA.build_settings(_vsa_setup(tmp_path, label=label))


@pytest.mark.parametrize("label", ["arch4_skill", "arch10_1_skill", "arch12_2_skill", "arch8_1_skill", "arch14_skill", "arch15_skill", "arch16_skill"])
def test_vsa_accepts_core_exit_architectures(tmp_path: Path, label: str) -> None:
    assert VSA.build_settings(_vsa_setup(tmp_path, label=label))["architecture_label"] == label


def test_vsa_rejects_different_fsq_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="different FSQ model"):
        VSA.build_settings(_vsa_setup(tmp_path, fsq=("other_fsq", "2000")))


def test_vsa_requires_focus_uv_for_spatial_architectures(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="skill_focus_uv"):
        VSA.build_settings(_vsa_setup(tmp_path, focus_uv=False))
    assert VSA.build_settings(_vsa_setup(tmp_path / "b", label="arch4_skill", focus_uv=False))


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
    assert settings["output_dir"].parent == tmp_path / "outputs/skillVLA_NewTask_FT" / component


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
    from_loader = VSA.load_stage1_component_config(CONFIGS / "NewTask_FT/VSA/vsa_ft_config.yaml")
    assert from_loader["newtask_ft"] is True
    assert from_loader["dataset"]["source"]
    assert "vsa_checkpoint" in from_loader["warm_start"]
    for name in ("Predictor/predictor_ft_config.yaml", "Terminator/terminator_ft_config.yaml"):
        merged = VSA.load_stage1_component_config(CONFIGS / "NewTask_FT" / name)
        assert merged["mode"] == "ft" and merged["newtask_ft"] is True
        assert "stage1_common" not in merged and "newtask_ft_common" not in merged


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
