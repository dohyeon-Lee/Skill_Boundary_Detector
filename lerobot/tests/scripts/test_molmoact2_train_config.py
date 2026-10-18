"""Submission and checkpoint guards: no torch/network/GPU required."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_molmoact2"
sys.path.insert(0, str(CONFIG_DIR / "PT/src"))
import molmoact2_config as MODULE


def config(tmp_path, stage="pt"):
    return {
        "stage": stage, "project_root": str(tmp_path), "dataset_root": "dataset", "outputs_root": "outputs",
        "train_partition": ["gpu_a", "gpu_b"], "train_qos": "base_qos", "train_exclude_nodes": ["node18"],
        "warm_start": {"run_name": "our_pt_lerobot", "checkpoint": "010000"},
        "training": {"dataloader": {"batch_size": 16, "micro_batch_size": 2, "gpus": 2, "workers": 4}},
        "slurm": {"gres": "gpu:2", "cpus": 16},
    }


def make_training_checkpoint(path, grounding="none"):
    for name in ("pretrained_model/config.json", "pretrained_model/train_config.json",
                 "pretrained_model/model.safetensors", "pretrained_model/sbd_molmoact2.json",
                 "pretrained_model/policy_preprocessor.json", "pretrained_model/policy_postprocessor.json",
                 "training_state/training_step.json", "training_state/optimizer_state.safetensors",
                 "training_state/optimizer_param_groups.json", "training_state/scheduler_state.json",
                 "training_state/rng_state.safetensors"):
        file = path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps({"proprio_grounding": grounding}) if name.endswith("sbd_molmoact2.json") else "{}")


def make_inputs(s):
    for name in (s["python"], str(Path(s["source"]) / "src/lerobot/scripts/lerobot_train.py")):
        p = Path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    info = Path(s["dataset_dir"]) / "meta/info.json"
    info.parent.mkdir(parents=True, exist_ok=True)
    info.write_text(json.dumps({"features": {
        "observation.images.image": {"shape": [256, 256, 3]},
        "observation.images.wrist_image": {"shape": [256, 256, 3]},
        "observation.state": {"shape": [8]}, "action": {"shape": [7]},
    }}))


def test_shipped_yamls_preserve_user_recipe_and_isolate_environment():
    for stage, fmt, flow in (("PT", "continuous", 1), ("FT", "both", 8)):
        s = MODULE.build_settings(MODULE.load_config(CONFIG_DIR / stage / f"molmoact2_{stage}_config.yaml"))
        assert s["python"].endswith("/.venv_molmoact2_lerobot/bin/python")
        assert s["source"].endswith("/third_party/lerobot_molmoact2")
        assert s["run_name"].endswith("_lerobot")
        assert s["action_format"] == fmt and s["num_flow_timesteps"] == flow
        assert s["keep_checkpoints"] == 2 and s["protect_checkpoints"] == [30000, 50000, 100000]
        assert s["lr_mode"] == "warmup_constant"
        assert all(v == 2.5e-5 for v in s["optimizer"].values())
        args = MODULE.trainer_args(s)
        assert '--policy.dtype="bfloat16"' in args and '--policy.train_mode_vlm="fft"' in args
        assert "--policy.enable_knowledge_insulation=false" in args
        assert "--steps=1600000" in args
        if stage == "PT":
            assert s["proprio_grounding"] == "episode_start_xyz"


def test_global_batch_and_micro_clock(tmp_path):
    s = MODULE.build_settings(config(tmp_path))
    assert s["accumulation_steps"] == 4
    args = MODULE.trainer_args(s)
    assert "--batch_size=2" in args and "--accelerator.gradient_accumulation.steps=4" in args
    assert "--steps=400000" in args and "--save_freq=40000" in args
    assert "our_pt" not in s["run_name"]


def test_smoke_preserves_model_but_is_separate(tmp_path):
    c = config(tmp_path)
    normal, smoke = MODULE.build_settings(c), MODULE.build_settings(c, smoke=True)
    assert smoke["output_dir"] == normal["output_dir"] + "_smoke"
    assert smoke["batch_size"] == smoke["micro_batch_size"] == smoke["gpus"] == smoke["accumulation_steps"] == 1
    assert smoke["workers"] == 0 and smoke["steps"] == smoke["save_every"] == 20
    assert not smoke["wandb_enable"] and smoke["protect_checkpoints"] == []
    assert smoke["tuning"] == normal["tuning"]


@pytest.mark.parametrize("protected", ["30000", None, [True], [0], [-1], [30000.0], ["30000"]])
def test_invalid_protection(tmp_path, protected):
    c = config(tmp_path)
    c["training"]["protect_checkpoints"] = protected
    with pytest.raises(ValueError, match="protect_checkpoints"):
        MODULE.build_settings(c)


def test_protection_is_not_an_extra_save_schedule(tmp_path):
    c = config(tmp_path)
    c["training"]["protect_checkpoints"] = [30001]
    with pytest.raises(ValueError, match="multiple of save_every"):
        MODULE.build_settings(c)
    c["training"]["schedule"] = {"steps": 30001}
    assert MODULE.build_settings(c)["protect_checkpoints"] == [30001]


@pytest.mark.parametrize("change,match", [
    ({"batch_size": 3}, "divisible"), ({"workers": 17}, "exceeds"),
    ({"batch_sze": 1}, "Unsupported"), ({"gpus": 1}, "GPU count"),
])
def test_invalid_dataloader(tmp_path, change, match):
    c = config(tmp_path)
    c["training"]["dataloader"].update(change)
    with pytest.raises(ValueError, match=match):
        MODULE.build_settings(c)


def test_missing_shards(tmp_path):
    s = MODULE.build_settings(config(tmp_path))
    make_inputs(s)
    model = Path(s["initial_checkpoint"])
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "a.safetensors", "b": "b.safetensors"}}))
    (model / "a.safetensors").write_text("fixture")
    MODULE.validate_inputs(s, model=False)
    with pytest.raises(FileNotFoundError, match="b.safetensors"):
        MODULE.validate_inputs(s)
    (model / "b.safetensors").write_text("fixture")
    MODULE.validate_inputs(s)


def test_ft_latest_and_frozen_initial(tmp_path):
    c = config(tmp_path, "ft")
    initial = Path(MODULE.build_settings(c)["initial_checkpoint"])
    assert initial.name == "010000" and initial.parent.name == "checkpoints"
    make_training_checkpoint(initial, "episode_start_xyz")
    make_training_checkpoint(initial.with_name("020000"), "episode_start_xyz")
    initial.with_name("030000").mkdir()  # not resumable, not a valid warm start
    c["warm_start"]["checkpoint"] = "last"
    s = MODULE.build_settings(c)
    assert Path(s["initial_checkpoint"]).name == "020000"
    assert s["proprio_grounding"] == "episode_start_xyz"
    c["_initial_checkpoint"] = s["initial_checkpoint"]
    make_training_checkpoint(initial.with_name("040000"))
    assert MODULE.build_settings(c)["initial_checkpoint"] == s["initial_checkpoint"]


def test_auto_resume_and_incomplete_latest_guard(tmp_path):
    c = config(tmp_path, "ft")
    s = MODULE.build_settings(c)
    make_inputs(s)
    make_training_checkpoint(Path(s["initial_checkpoint"]))
    checkpoint = Path(s["output_dir"]) / "checkpoints/010000"
    make_training_checkpoint(checkpoint)
    MODULE.validate_inputs(s)
    args = MODULE.trainer_args(s)
    assert "--resume=true" in args
    assert f"--config_path={checkpoint / 'pretrained_model/train_config.json'}" in args
    assert not any(a.startswith("--policy.path=") for a in args)
    checkpoint.with_name("020000").mkdir()
    with pytest.raises(ValueError, match="Latest checkpoint is incomplete"):
        MODULE.validate_inputs(s)


def test_ft_grounding_cannot_be_guessed_or_changed(tmp_path):
    c = config(tmp_path, "ft")
    initial = Path(MODULE.build_settings(c)["initial_checkpoint"])
    make_training_checkpoint(initial, "episode_start_xyz")
    s = MODULE.build_settings(c)
    make_inputs(s)
    make_training_checkpoint(Path(s["output_dir"]) / "checkpoints/010000", "none")
    with pytest.raises(ValueError, match="Cannot change proprio grounding"):
        MODULE.validate_inputs(s)
    (initial / "pretrained_model/sbd_molmoact2.json").unlink()
    with pytest.raises(ValueError, match="Missing grounding provenance"):
        MODULE.build_settings(c)
