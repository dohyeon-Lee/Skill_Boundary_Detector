"""Architecture choices must route to distinct runs and matching training flags."""
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "lerobot/skillVLA/abc/configs/train_skills/DP/src/train_dp_config.py"
SPEC = importlib.util.spec_from_file_location("abc_dp_architecture_config", PATH)
RESOLVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESOLVER)


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.delenv("DP_TASKS", raising=False)
    monkeypatch.delenv("DP_DATASET", raising=False)
    dataset = tmp_path / "dataset_ABC/abc_sim_all"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "native_dataset.json").write_text(json.dumps({
        "format": RESOLVER.ABC_NATIVE_FORMAT,
        "task_sha256": {"grab_clutter": "fixture"},
    }))
    feature = {"shape": [14], "names": RESOLVER.YAM_JOINT_NAMES}
    (dataset / "meta/info.json").write_text(json.dumps({
        "fps": 30, "features": {"observation.state": feature, "action": feature},
    }))
    return {
        "project_root": str(tmp_path), "dp_tasks": "grab_clutter",
        "dp_action_mode": "delta_joint", "dp_base_config": "base.yaml",
        "dp_n_obs_steps": 60, "dp_future_action_horizon": 72,
    }


def test_architectures_use_distinct_paths_and_matching_training_settings(config):
    names = set()
    for encoder in ["flat", "GRU_iid", "GRU_epi", "transformer", "transformer_cls"]:
        for size, dims in [("small", [128, 256, 512]), ("base", [512, 1024, 2048])]:
            settings = RESOLVER.build_settings({
                **config, "dp_history_encoder": encoder, "dp_unet_size": size,
            })
            is_gru = encoder.lower().startswith("gru")
            assert settings["dp_history_encoder"] == ("gru" if is_gru else encoder)
            assert settings["dp_history_encoder_dim"] == 128
            assert json.loads(settings["dp_down_dims_arg"]) == dims
            history_suffix = "" if encoder == "flat" else "_" + encoder.lower()
            assert settings["dp_history_gru_stateful"] == is_gru
            assert settings["dp_history_gru_layers"] == 2
            assert settings["dp_history_gru_training"] == ("epi" if encoder == "GRU_epi" else "iid")
            assert settings["dp_dataloader_persistent_workers"] == (encoder == "GRU_epi")
            size_suffix = "_small" if size == "small" else ""
            expected = (
                "abc_sim_all_task-grab_clutter_delta-joint_future_only_"
                f"obs60_future72{history_suffix}_constLR{size_suffix}"
            )
            assert settings["dp_run_name"] == expected
            assert settings["dp_output_dir"].name == expected
            assert settings["dp_final_checkpoint"].relative_to(settings["dp_output_dir"]).as_posix() == (
                "checkpoints/100000/pretrained_model"
            )
            names.add(settings["dp_output_dir"])
    assert len(names) == 10


def test_old_gru_spelling_selects_iid_variant(config):
    old = RESOLVER.build_settings({**config, "dp_history_encoder": "gru"})
    explicit = RESOLVER.build_settings({**config, "dp_history_encoder": "GRU_iid"})
    assert old["dp_run_name"] == explicit["dp_run_name"]


def test_epi_without_workers_disables_worker_persistence(config):
    settings = RESOLVER.build_settings({
        **config, "dp_history_encoder": "GRU_epi", "dp_num_workers": 0,
    })
    assert settings["dp_dataloader_persistent_workers"] is False


@pytest.mark.parametrize("overrides,key", [
    ({"dp_history_encoder": "rnn"}, "dp_history_encoder"),
    ({"dp_unet_size": "large"}, "dp_unet_size"),
    ({"dp_history_encoder_dim": 256}, "dp_history_encoder_dim"),
])
def test_unsupported_architectures_fail_before_submission(config, overrides, key):
    with pytest.raises(ValueError, match=key):
        RESOLVER.build_settings({**config, **overrides})
