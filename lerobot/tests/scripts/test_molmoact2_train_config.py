"""Boundary checks for the MolmoAct2 adapter (no torch, network, or GPU required)."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path

import pytest

CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_molmoact2"
SPEC = importlib.util.spec_from_file_location("molmoact2_train_config", CONFIG_DIR / "PT/src/molmoact2_config.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def config(tmp_path: Path, stage: str = "pt") -> dict:
    return {
        "stage": stage, "project_root": str(tmp_path), "dataset_root": "dataset", "outputs_root": "outputs",
        "train_partition": ["gpu_a", "gpu_b"], "train_qos": "base_qos", "train_exclude_nodes": ["node18"],
        "warm_start": {"run_name": "our_pt", "checkpoint": "010000"},
        "training": {"dataloader": {"batch_size": 16, "micro_batch_size": 2, "gpus": 2, "workers": 4}},
        "slurm": {"gres": "gpu:2", "cpus": 16},
    }


def make_training_checkpoint(path: Path, grounding: str = "none") -> None:
    for name in ("config.yaml", "model_and_optim/.metadata", "train/rank0.pt"):
        file = path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps({"model": {"robot_processor": {"metadata_by_tag": {
            "libero": {"proprio_grounding": grounding},
        }}}}) if name == "config.yaml" else "fixture")


def make_inputs(s: dict) -> None:
    python = Path(s["python"])
    python.parent.mkdir(parents=True, exist_ok=True)
    python.touch()
    trainer = Path(s["source"]) / "launch_scripts/train_lerobot.py"
    trainer.parent.mkdir(parents=True, exist_ok=True)
    trainer.touch()
    info = Path(s["dataset_dir"]) / "meta/info.json"
    info.parent.mkdir(parents=True, exist_ok=True)
    info.write_text(json.dumps({"features": {
        "observation.images.image": {"shape": [256, 256, 3]},
        "observation.images.wrist_image": {"shape": [256, 256, 3]},
        "observation.state": {"shape": [8]}, "action": {"shape": [7]},
    }}))


def test_shipped_yaml_routes_to_official_training_and_isolated_environment():
    for stage in ("PT", "FT"):
        s = MODULE.build_settings(MODULE.load_config(CONFIG_DIR / stage / f"molmoact2_{stage}_config.yaml"))
        assert s["python"].endswith("/.venv_molmoact2/bin/python")
        assert s["source"].endswith("/third_party/molmoact2/experiments")
        assert f"/molmoact2_{stage}/" in s["output_dir"]
        assert s["keep_checkpoints"] == 2
        assert s["protect_checkpoints"] == [30000, 50000, 100000]
        argv = MODULE.trainer_args(s)
        assert "--ft_vlm=true" in argv and "--ft_action_expert=true" in argv
        assert "--action_format=both" in argv and "--num_flow_timesteps=8" in argv
        assert "--discrete_action_tokenizer=allenai/MolmoAct2-FAST-Tokenizer" in argv
        assert "--action_expert_detach_vlm=false" in argv
        assert "--save_num_checkpoints_to_keep=2" in argv
        assert not any("protect_checkpoints" in arg for arg in argv)  # adapter-only, not an unknown native flag
        assert "--max_action_dim=32" in argv and "--state_format=discrete" in argv
        assert s["lr_mode"] == "warmup_constant"
        assert all(value == 2.5e-5 for value in s["optimizer"].values())
        assert "--scheduler.alpha_f=1.0" in argv
        assert "--scheduler.warmup_min_lr=0.0" in argv
        assert f"--scheduler.llm_t_warmup={1001 if stage == 'PT' else 501}" in argv
        if stage == "PT":
            assert s["proprio_grounding"] == "episode_start_xyz"


def test_global_batch_and_smoke_are_distinct(tmp_path):
    cfg = config(tmp_path)
    normal = MODULE.build_settings(cfg)
    smoke = MODULE.build_settings(cfg, smoke=True)
    assert normal["batch_size"] == 16 and normal["micro_batch_size"] == 2 and normal["gpus"] == 2
    assert smoke["batch_size"] == smoke["micro_batch_size"] == smoke["gpus"] == 1
    assert smoke["workers"] == 0 and smoke["steps"] == smoke["save_every"] == 20
    assert smoke["wandb_enable"] is False
    assert smoke["output_dir"] == normal["output_dir"] + "_smoke"
    assert smoke["tuning"] == normal["tuning"] and smoke["action_format"] == normal["action_format"]
    assert normal["partition"] == "gpu_a,gpu_b" and normal["qos"] == "base_qos"
    cfg["training"]["protect_checkpoints"] = [30000, 50000, 100000]
    assert MODULE.build_settings(cfg, smoke=True)["protect_checkpoints"] == []


def test_retention_settings_normalize_without_changing_run_identity(tmp_path):
    cfg = config(tmp_path)
    default = MODULE.build_settings(cfg)
    assert default["keep_checkpoints"] == -1 and default["protect_checkpoints"] == []
    cfg["training"].update(keep_checkpoints=2, protect_checkpoints=[100000, 30000, 50000, 30000])
    settings = MODULE.build_settings(cfg)
    assert settings["protect_checkpoints"] == [30000, 50000, 100000]
    assert settings["run_name"] == default["run_name"]


@pytest.mark.parametrize("protected", ["30000", None, [True], [0], [-1], [30000.0], ["30000"]])
def test_invalid_protected_steps_are_rejected(tmp_path, protected):
    cfg = config(tmp_path)
    cfg["training"]["protect_checkpoints"] = protected
    with pytest.raises(ValueError, match="protect_checkpoints"):
        MODULE.build_settings(cfg)


def test_protected_steps_must_be_saved_by_native_schedule(tmp_path):
    cfg = config(tmp_path)
    cfg["training"].update(protect_checkpoints=[30001])
    with pytest.raises(ValueError, match="multiple of save_every"):
        MODULE.build_settings(cfg)
    cfg["training"]["schedule"] = {"steps": 30001}
    assert MODULE.build_settings(cfg)["protect_checkpoints"] == [30001]  # native final checkpoint


def test_unspecified_supervision_and_sampling_follow_the_paper_ft_recipe(tmp_path):
    settings = MODULE.build_settings(config(tmp_path))
    assert settings["action_format"] == "both"
    assert settings["num_flow_timesteps"] == 8
    assert settings["tuning"] == "full"


def test_paper_recipe_is_distinct_from_original_cli_fallbacks(tmp_path):
    original = CONFIG_DIR.parents[4] / "third_party/molmoact2/experiments/launch_scripts/train_lerobot.py"
    if not original.is_file():
        pytest.skip("Original MolmoAct2 checkout is not installed")
    defaults = {}
    for node in ast.walk(ast.parse(original.read_text())):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)):
            flag = node.args[0].value
            if flag in {"--action_format", "--num_flow_timesteps"}:
                defaults[flag[2:]] = next(ast.literal_eval(kw.value) for kw in node.keywords if kw.arg == "default")
    settings = MODULE.build_settings(config(tmp_path))
    # The CLI fallback is not the authors' chosen fine-tuning recipe. Our local
    # PT and FT both adapt the released model, so both use the paper FT recipe.
    assert defaults == {"action_format": "continuous", "num_flow_timesteps": 1}
    assert {key: settings[key] for key in defaults} == {"action_format": "both", "num_flow_timesteps": 8}


def test_flow_only_and_single_sample_ablation_remain_explicitly_available(tmp_path):
    cfg = config(tmp_path)
    cfg["training"].update(action_format="continuous", num_flow_timesteps=1)
    args = MODULE.trainer_args(MODULE.build_settings(cfg))
    assert "--action_format=continuous" in args and "--num_flow_timesteps=1" in args
    assert not any(arg.startswith("--discrete_action_tokenizer=") for arg in args)


@pytest.mark.parametrize("change,match", [
    ({"training": {"dataloader": {"batch_size": 3, "gpus": 2}}}, "divisible"),
    ({"training": {"dataloader": {"gpus": 1}}}, "GPU count"),
    ({"training": {"dataloader": {"workers": 17}}}, "exceeds"),
    ({"training": {"tuning": "action_expert", "action_format": "both"}}, "requires action_format"),
    ({"training": {"dataloader": {"batch_sze": 1}}}, "Unsupported"),
    ({"dataset": {"name": "../wrong"}}, "single folder"),
    ({"training": {"gradient_checkpointing": "false"}}, "true or false"),
])
def test_invalid_recipe_fails_before_submission(tmp_path, change, match):
    cfg = config(tmp_path)
    for key, value in change.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, dict):
                    cfg.setdefault(key, {}).setdefault(subkey, {}).update(subvalue)
                else:
                    cfg.setdefault(key, {})[subkey] = subvalue
        else:
            cfg[key] = value
    with pytest.raises(ValueError, match=match):
        MODULE.build_settings(cfg)


def test_missing_download_shards_fail_before_submission(tmp_path):
    s = MODULE.build_settings(config(tmp_path))
    make_inputs(s)
    model = Path(s["initial_checkpoint"])
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "a.safetensors", "b": "b.safetensors"}}))
    (model / "a.safetensors").write_bytes(b"fixture")
    MODULE.validate_inputs(s, model=False)
    with pytest.raises(FileNotFoundError, match="b.safetensors"):
        MODULE.validate_inputs(s)
    (model / "b.safetensors").write_bytes(b"fixture")
    MODULE.validate_inputs(s)


def test_ft_selects_our_pt_step_and_last_ignores_unsharded_and_incomplete(tmp_path):
    cfg = config(tmp_path, stage="ft")
    s = MODULE.build_settings(cfg)
    assert s["initial_checkpoint"].endswith("/outputs/molmoact2_PT/our_pt/step10000")
    pt = tmp_path / "outputs/molmoact2_PT/our_pt"
    make_training_checkpoint(pt / "step10000")
    make_training_checkpoint(pt / "step20000")
    (pt / "step30000").mkdir()
    (pt / "step99999-unsharded").mkdir()
    cfg["warm_start"]["checkpoint"] = "last"
    latest = MODULE.build_settings(cfg)
    assert latest["initial_checkpoint"].endswith("/step20000")
    assert "step20000" in latest["run_name"]
    frozen = copy.deepcopy(cfg)
    frozen["_initial_checkpoint"] = latest["initial_checkpoint"]
    make_training_checkpoint(pt / "step40000")
    assert MODULE.build_settings(frozen)["initial_checkpoint"] == latest["initial_checkpoint"]


def test_output_with_incomplete_latest_checkpoint_is_not_overwritten(tmp_path):
    s = MODULE.build_settings(config(tmp_path, stage="ft"))
    make_inputs(s)
    make_training_checkpoint(Path(s["initial_checkpoint"]))
    output = Path(s["output_dir"])
    make_training_checkpoint(output / "step100")
    MODULE.validate_inputs(s)
    (output / "step200").mkdir()
    with pytest.raises(ValueError, match="Latest checkpoint is incomplete"):
        MODULE.validate_inputs(s)
    assert (output / "step100/train/rank0.pt").exists()


def test_ae_only_and_disabled_wandb_pass_through(tmp_path):
    cfg = config(tmp_path)
    cfg["training"].update({"tuning": "action_expert", "action_format": "continuous"})
    cfg["logging"] = {"wandb": {"enable": False}}
    args = MODULE.trainer_args(MODULE.build_settings(cfg))
    assert "--ft_vlm=false" in args and "--ft_embedding=none" in args
    assert "--wandb=null" in args
    assert not any(arg.startswith("--discrete_action_tokenizer=") for arg in args)


def test_ft_inherits_grounding_from_self_contained_checkpoint(tmp_path):
    cfg = config(tmp_path, stage="ft")
    checkpoint = Path(MODULE.build_settings(cfg)["initial_checkpoint"])
    make_training_checkpoint(checkpoint, grounding="episode_start_xyz")
    assert MODULE.build_settings(cfg)["proprio_grounding"] == "episode_start_xyz"
    make_training_checkpoint(checkpoint, grounding="none")
    assert MODULE.build_settings(cfg)["proprio_grounding"] == "none"
    cfg["pt_proprio_grounding"] = True
    with pytest.raises(ValueError, match="FT inherits"):
        MODULE.build_settings(cfg)


def test_grounding_cannot_change_on_resume(tmp_path):
    cfg = config(tmp_path, stage="ft")
    initial = Path(MODULE.build_settings(cfg)["initial_checkpoint"])
    make_training_checkpoint(initial, grounding="episode_start_xyz")
    settings = MODULE.build_settings(cfg)
    make_inputs(settings)
    make_training_checkpoint(Path(settings["output_dir"]) / "step100", grounding="none")
    with pytest.raises(ValueError, match="Cannot change proprio grounding"):
        MODULE.validate_inputs(settings)


def test_common_base_lr_allows_explicit_component_override(tmp_path):
    cfg = config(tmp_path)
    cfg["training"]["optimizer"] = {"base_lr": 2.5e-5, "action_expert_lr": 5e-5}
    lr = MODULE.build_settings(cfg)["optimizer"]
    assert lr["llm_lr"] == lr["vit_lr"] == lr["connector_lr"] == 2.5e-5
    assert lr["action_expert_lr"] == 5e-5
