"""Cross-domain pi05 FT must keep the demo selection, source stats and exact XML."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from lerobot.utils.mimicgen_init_states import apply_coffee_init_states, coffee_records

SCRIPT = Path(__file__).resolve().parents[2] / "skillVLA/mimicgen/configs/train_pi05/FT/src/config.py"
spec = importlib.util.spec_from_file_location("coffee_ft_test", SCRIPT)
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)


def fixture_config(tmp_path, monkeypatch):
    monkeypatch.setattr(ft.common, "CONFIGS", tmp_path)
    (tmp_path / "comparison.yaml").write_text(yaml.safe_dump({
        "dataset": {"source": "coffee", "n_episodes": 2},
    }))
    dataset = tmp_path / "data/coffee"
    (dataset / "meta/episodes").mkdir(parents=True)
    pd.DataFrame({"episode_index": [8, 2, 5]}).to_parquet(dataset / "meta/episodes/file.parquet")
    inputs = {"observation.state": {"shape": [8]}, "observation.images.image": {"shape": [256, 256, 3]}}
    policy = {"type": "pi05", "chunk_size": 10, "proprio_grounding": "episode_start_xyz",
              "freeze_vision_encoder": False, "freeze_language_model": False,
              "input_features": inputs, "output_features": {"action": {"shape": [7]}}}
    features = {**inputs, "action": {"shape": [7]}}
    (dataset / "meta/info.json").write_text(json.dumps({"features": features}))
    checkpoint = tmp_path / "libero_outputs/pi05_PT/libero/checkpoints/100000/pretrained_model"
    checkpoint.mkdir(parents=True)
    for name, value in {"config.json": policy, "train_config.json": {"policy": policy},
                        "policy_preprocessor.json": {}, "policy_postprocessor.json": {}}.items():
        (checkpoint / name).write_text(json.dumps(value))
    (checkpoint / "model.safetensors").write_bytes(b"fixture")
    tokenizer = tmp_path / "models/paligemma-3b-pt-224-tokenizer"
    tokenizer.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (tokenizer / name).write_text("{}")
    cfg = {"project_root": str(tmp_path), "dataset_root": "data", "outputs_root": "coffee_outputs",
           "server": "local", "comparison_config": "comparison.yaml", "stage": "ft",
           "warm_start": {"outputs_root": "libero_outputs", "run": "libero", "checkpoint": 100000},
           "adaptation": {"freeze_expert": True},
           "training": {"normalization_stats": "pt", "schedule": {"steps": 20000}}}
    return cfg, checkpoint


def test_ft_uses_sorted_demo_ids_and_separate_pt_and_ft_roots(tmp_path, monkeypatch):
    cfg, checkpoint = fixture_config(tmp_path, monkeypatch)
    settings = ft.build_settings(cfg)
    assert settings["ft_dataset_episodes"] == [2, 5]
    assert settings["ft_pretrained_model_path"] == checkpoint
    assert settings["ft_output_dir"].is_relative_to(tmp_path / "coffee_outputs/pi05_FT")
    assert settings["ft_freeze_expert"] and not settings["ft_freeze_language_model"]
    assert settings["ft_normalization_stats"] == "pt"
    assert settings["ft_chunk_size"] == 10
    assert settings["ft_proprio_grounding"] == "episode_start_xyz"


def test_frozen_selection_survives_later_skillvla_config_edits(tmp_path, monkeypatch):
    cfg, _ = fixture_config(tmp_path, monkeypatch)
    prepared = {**ft.prepare(cfg), "_resolved_pi05": True}
    frozen = tmp_path / "frozen.yaml"
    frozen.write_text(yaml.safe_dump(prepared))
    (tmp_path / "comparison.yaml").write_text("dataset: {source: changed, n_episodes: all}")
    settings = ft.build_settings(ft.common.load_config(frozen))
    assert settings["ft_dataset_episodes"] == [2, 5]
    assert settings["ft_dataset"] == "coffee"


def test_incompatible_action_schema_is_rejected(tmp_path, monkeypatch):
    cfg, _ = fixture_config(tmp_path, monkeypatch)
    info = tmp_path / "data/coffee/meta/info.json"
    content = json.loads(info.read_text())
    content["features"]["action"]["shape"] = [14]
    info.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="action shape"):
        ft.build_settings(cfg)


def test_coffee_exact_reset_keeps_xml_and_episode_order(tmp_path):
    bank = tmp_path / "bank.npz"
    np.savez(bank, environment="Coffee_D0", episode_index=[8, 2], init_states=[[8., 0.], [2., 0.]],
             model_xml=['<mujoco model="eight"/>', '<mujoco model="two"/>'])
    with pytest.raises(ValueError, match="Enable repeat"):
        coffee_records(bank, 3)
    base = SimpleNamespace(episode_index=1)
    vector = SimpleNamespace(envs=[SimpleNamespace(unwrapped=base)])
    apply_coffee_init_states({"Coffee_D0": {0: vector}}, bank, 3, repeat=True)
    assert base.init_states and base.init_state_id == 1
    assert [row["state"][0] for row in base._init_states] == [2., 8., 2.]
    assert 'model="two"' in base._init_states[0]["model_xml"]
    with pytest.raises(ValueError, match="only Coffee_D0"):
        apply_coffee_init_states({"libero_90": {0: vector}}, bank, 3, repeat=True)


@pytest.mark.parametrize("stage", ["FT", "FT/eval"])
def test_yonsei_dispatch_uses_coffee_pi05_resolver(stage):
    root = SCRIPT.parents[7]
    remote = ft.common.module("coffee_pi05_remote", root / "tools/remote/yonsei.py")
    name = "submit_eval.sh" if stage.endswith("eval") else "submit_train.sh"
    script = root / f"lerobot/skillVLA/mimicgen/configs/train_pi05/{stage}/{name}"
    config, resolver, extra = remote.config_and_resolver(script, [], {}, root)
    assert resolver == script.parent / "src/config.py"
    assert config.name == ("pi05_eval_config.yaml" if stage.endswith("eval") else "pi05_FT_config.yaml")
    assert not extra


def test_yonsei_progress_reads_coffee_report_src_folder(tmp_path):
    root = SCRIPT.parents[7]
    probe = ft.common.module("coffee_progress_test", root / "tools/remote/probe.py")
    video = tmp_path / "report/src/videos/Coffee_D0_0"
    video.mkdir(parents=True)
    (video / "eval_progress.json").write_text(json.dumps({"successes": [True, False]}))
    result = probe.progress(tmp_path, {"kind": "mimicgen_pi05", "output": "report",
                                      "tasks": [0], "labels": ["pi05"], "episodes": 5})
    assert result["completed"] == 2 and result["total"] == 5
    assert result["success_pct"] == 50.
