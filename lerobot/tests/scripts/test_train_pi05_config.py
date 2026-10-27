from __future__ import annotations

import importlib.util
import json

import pytest
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_pi05/PT/src/train_pi05_config.py"
)
SPEC = importlib.util.spec_from_file_location("train_pi05_config", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _config(tmp_path: Path, **overrides) -> dict:
    tokenizer = tmp_path / "models/tokenizer"
    tokenizer.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (tokenizer / name).write_text("{}")
    return {
        "project_root": str(tmp_path),
        "outputs_root": "outputs",
        "dataset_root": "dataset",
        "pi05_tokenizer": "models/tokenizer",
        "pt_dataset": "libero_90_full_full",
        "pt_batch_size": 16,
        "pt_exp": "pro",
        "pt_freeze_language_model": True,  # probes must not leak into the name
        "pt_lr_mode": "warmup_constant",
        "ft_dataset": "libero_10_full_2",
        "ft_batch_size": 8,
        "ft_exp": "",
        "ft_pretrained_checkpoint": "020000",
        **overrides,
    }


def _write_pt_checkpoint(tmp_path: Path, run: str, policy: dict) -> None:
    path = tmp_path / "outputs/pi05_PT" / run / "checkpoints/020000/pretrained_model"
    path.mkdir(parents=True)
    (path / "train_config.json").write_text(json.dumps({"policy": policy}))


def test_run_names_carry_only_batch_dataset_and_exp(tmp_path: Path) -> None:
    settings = MODULE.build_settings(_config(tmp_path))
    assert settings["pt_run_name"] == "bs16_libero_90_full_full_pro"
    assert settings["pt_output_dir"] == tmp_path / "outputs/pi05_PT/bs16_libero_90_full_full_pro"
    assert settings["ft_run_name"] == "bs8_libero_10_full_2_PT20k"
    assert MODULE.build_settings(_config(tmp_path, pt_exp=""))["pt_run_name"] == "bs16_libero_90_full_full"
    assert not any("lora" in key for key in settings)


def test_grounding_is_a_boolean_toggle(tmp_path: Path) -> None:
    assert MODULE.build_settings(_config(tmp_path))["pt_proprio_grounding"] == "none"
    on = MODULE.build_settings(_config(tmp_path, pt_proprio_grounding=True))
    assert on["pt_proprio_grounding"] == "episode_start_xyz"
    off = MODULE.build_settings(_config(tmp_path, pt_proprio_grounding=False))
    assert off["pt_proprio_grounding"] == "none"


def test_ft_inherits_grounding_from_the_pt_checkpoint(tmp_path: Path) -> None:
    _write_pt_checkpoint(tmp_path, "grounded", {"proprio_grounding": "episode_start_xyz"})
    _write_pt_checkpoint(tmp_path, "legacy", {"freeze_vision_encoder": False})
    grounded = MODULE.build_settings(_config(tmp_path, ft_pretrained_run_name="grounded"))
    # A yaml toggle cannot override what the loaded PT was trained with.
    legacy = MODULE.build_settings(
        _config(tmp_path, ft_pretrained_run_name="legacy", ft_proprio_grounding=True)
    )
    assert grounded["ft_proprio_grounding"] == "episode_start_xyz"
    assert legacy["ft_proprio_grounding"] == "none"


def test_gradient_checkpointing_is_configurable_and_defaults_on(tmp_path: Path) -> None:
    default = MODULE.build_settings(_config(tmp_path))
    assert default["pt_gradient_checkpointing"] is True and default["ft_gradient_checkpointing"] is True
    off = MODULE.build_settings(
        _config(tmp_path, pt_gradient_checkpointing=False, ft_gradient_checkpointing=False)
    )
    assert off["pt_gradient_checkpointing"] is False and off["ft_gradient_checkpointing"] is False
    assert off["pt_run_name"] == default["pt_run_name"]


def test_stage1_style_nested_blocks_feed_the_declared_stage(tmp_path: Path) -> None:
    base = {k: v for k, v in _config(tmp_path).items() if k not in {"pt_batch_size"}}
    settings = MODULE.build_settings({
        **base,
        "stage": "pt",
        "training": {
            "dataloader": {"batch_size": 8, "workers": 6, "gpus": 2},
            "optimizer": {"base_lr": 1e-5},
            "gradient_checkpointing": False,
            "schedule": {"steps": 1234, "lr_mode": "cosine_decay", "warmup_steps": 10,
                         "lr_decay_steps": 500, "decay_lr": 1e-6, "log_every": 50, "save_every": 617},
        },
        "logging": {"wandb": {"enable": False, "project": "VLA_stage1"}},
        "slurm": {"gres": "gpu:2", "cpus": 12, "memory": "96G", "time": "24:00:00"},
    })
    assert settings["pt_run_name"] == "bs8_libero_90_full_full_pro"
    assert settings["pt_lr"] == pytest.approx(2e-5)  # base_lr x gpus
    expected = {
        "pt_num_workers": 6, "pt_num_gpus": 2, "pt_gradient_checkpointing": False, "pt_steps": 1234,
        "pt_lr_mode": "cosine_decay", "pt_warmup_steps": 10, "pt_decay_steps": 500, "pt_log_freq": 50,
        "pt_save_freq": 617, "pt_wandb_enable": False, "pt_wandb_project": "VLA_stage1",
        "pt_gres": "gpu:2", "pt_cpus_per_task": 12, "pt_mem": "96G", "pt_time": "24:00:00",
    }
    assert {key: settings[key] for key in expected} == expected
    # The FT side of the same yaml is untouched by a stage: pt block.
    assert settings["ft_batch_size"] == 8 and settings["ft_steps"] == 5000


def test_nested_blocks_reject_typos_and_missing_stage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stage: pt"):
        MODULE.build_settings({**_config(tmp_path), "training": {"dataloader": {"batch_size": 8}}})
    with pytest.raises(ValueError, match="training.optimizer.muon"):
        MODULE.build_settings(
            {**_config(tmp_path), "stage": "pt", "training": {"optimizer": {"muon": False}}}
        )
    with pytest.raises(ValueError, match="not configurable for pi05 FT"):
        MODULE.build_settings(
            {**_config(tmp_path), "stage": "ft", "training": {"dataloader": {"gpus": 2}}}
        )


def test_shipped_yamls_resolve() -> None:
    root = SCRIPT.parents[2]
    pt = MODULE.build_settings(MODULE.load_config(root / "PT/pi05_PT_config.yaml"))
    ft = MODULE.build_settings(MODULE.load_config(root / "FT/pi05_FT_config.yaml"))
    assert pt["pt_run_name"].startswith(f"bs{pt['pt_batch_size']}_{pt['pt_dataset']}")
    assert ft["ft_run_name"].startswith(f"bs{ft['ft_batch_size']}_{ft['ft_dataset']}_PT")
    import yaml
    groot = yaml.safe_load((root.parent / "train_groot/FT/groot_FT_config.yaml").read_text())["training"]
    assert ft["ft_steps"] == groot["schedule"]["steps"] == 10000
    assert ft["ft_save_freq"] == groot["schedule"]["save_every"] == 1000
    assert ft["ft_keep_checkpoints"] == groot["keep_checkpoints"] == 2
    assert ft["ft_protect_checkpoints"] == groot["protect_checkpoints"] == [1000, 3000, 5000, 10000]


def test_ft_retention_is_opt_in_and_normalizes_protected_steps(tmp_path):
    assert MODULE.build_settings(_config(tmp_path))["ft_keep_checkpoints"] == -1
    settings = MODULE.build_settings({**_config(tmp_path), "stage": "ft", "training": {
        "keep_checkpoints": 2, "protect_checkpoints": [5000, 1000, 5000],
    }})
    assert settings["ft_keep_checkpoints"] == 2
    assert settings["ft_protect_checkpoints"] == [1000, 5000]


@pytest.mark.parametrize("keep", [0, -2, True, 1.5, "2"])
def test_ft_retention_rejects_invalid_keep(tmp_path, keep):
    with pytest.raises(ValueError, match="keep_checkpoints"):
        MODULE.build_settings({**_config(tmp_path), "stage": "ft", "training": {"keep_checkpoints": keep}})


@pytest.mark.parametrize("protected", ["1000", [0], [-1], [True], [1.5], ["1000"]])
def test_ft_retention_rejects_invalid_protected_steps(tmp_path, protected):
    with pytest.raises(ValueError, match="protect_checkpoints"):
        MODULE.build_settings({**_config(tmp_path), "stage": "ft", "training": {"protect_checkpoints": protected}})


def test_ft_has_pt_schedule_knobs_and_inherits_the_pt_chunk_size(tmp_path: Path) -> None:
    _write_pt_checkpoint(tmp_path, "pt_run", {"chunk_size": 25, "proprio_grounding": "episode_start_xyz"})
    base = {k: v for k, v in _config(tmp_path, ft_pretrained_run_name="pt_run").items() if k != "ft_batch_size"}
    settings = MODULE.build_settings({
        **base,
        "stage": "ft",
        "training": {
            "dataloader": {"batch_size": 8, "workers": 2},
            "optimizer": {"base_lr": 1e-5},
            "schedule": {"steps": 3000, "lr_mode": "warmup_constant", "warmup_steps": 200,
                         "lr_decay_steps": 3000, "decay_lr": 1e-6, "log_every": 50, "save_every": 1000},
        },
    })
    expected = {"ft_lr_mode": "warmup_constant", "ft_warmup_steps": 200, "ft_decay_steps": 3000,
                "ft_steps": 3000, "ft_log_freq": 50, "ft_save_freq": 1000, "ft_batch_size": 8,
                "ft_chunk_size": 25, "ft_proprio_grounding": "episode_start_xyz"}
    assert {key: settings[key] for key in expected} == expected
    # Flat yamls written before the schedule keys existed keep the policy default.
    assert MODULE.build_settings(_config(tmp_path))["ft_lr_mode"] == "cosine_decay"
    with pytest.raises(ValueError, match="lr_mode"):
        MODULE.build_settings({**base, "stage": "ft", "training": {"schedule": {"lr_mode": "linear"}}})


def test_step_label_formats_checkpoint_steps() -> None:
    assert MODULE.step_label("030000") == "30k"
    assert MODULE.step_label("100000") == "100k"
    assert MODULE.step_label("001500") == "001500"   # not a whole thousand: left as-is
    assert MODULE.step_label("last") == "last"


@pytest.mark.parametrize("mode", ["ft", "pt"])
def test_ft_stats_probe_selects_mode_and_separates_run(tmp_path, mode):
    c = _config(tmp_path)
    default = MODULE.build_settings(c)
    selected = MODULE.build_settings({**c, "stage": "ft", "training": {"normalization_stats": mode}})
    assert selected["ft_normalization_stats"] == mode
    assert selected["ft_run_name"] == default["ft_run_name"] + ("_ptstats" if mode == "pt" else "")


@pytest.mark.parametrize("mode", ["unknown", None, True])
def test_invalid_ft_stats_probe(tmp_path, mode):
    with pytest.raises(ValueError, match="normalization_stats"):
        MODULE.build_settings({**_config(tmp_path), "stage": "ft", "training": {"normalization_stats": mode}})
