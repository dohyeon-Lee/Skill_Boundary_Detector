from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_pi05/pi05_eval"
SPEC = importlib.util.spec_from_file_location("pi05_eval_config", EVAL_DIR / "src/pi05_eval_config.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _checkpoint(root: Path, group: str, run: str, step: str, **policy) -> Path:
    path = root / "outputs" / group / run / "checkpoints" / step / "pretrained_model"
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"type": "pi05", "chunk_size": 10, **policy}))
    return path


def _config(tmp_path: Path, **overrides) -> dict:
    tokenizer = tmp_path / "models/tokenizer"
    tokenizer.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (tokenizer / name).write_text("{}")
    return {
        "project_root": str(tmp_path),
        "outputs_root": "outputs",
        "pi05_tokenizer": "models/tokenizer",
        "model_defaults": {"stage": "PT", "checkpoint": "last"},
        "models": [{"model_dir": "run_a"}],
        "target_task": "libero_10",
        "task_ids": [0, 1, 2],
        "episode_offset": 25,
        **overrides,
    }


def test_single_panel_resolves_last_checkpoint_and_default_name(tmp_path: Path) -> None:
    _checkpoint(tmp_path, "pi05_PT", "run_a", "010000")
    newest = _checkpoint(tmp_path, "pi05_PT", "run_a", "020000", proprio_grounding="episode_start_xyz")
    settings = MODULE.build_settings(_config(tmp_path))
    (panel,) = json.loads(settings["models_json"])
    assert panel["policy_path"] == str(newest) and panel["checkpoint"] == "020000"
    assert panel["proprio_grounding"] == "episode_start_xyz"
    assert settings["eval_out_dir"].name == "run_a_020000_libero_10_offset25"
    assert settings["task_ids"] == "[0,1,2]" and settings["eval_expected_tasks"] == 3


def test_panels_mix_stages_and_inherit_defaults(tmp_path: Path) -> None:
    _checkpoint(tmp_path, "pi05_PT", "run_a", "020000")
    _checkpoint(tmp_path, "pi05_FT", "run_b", "005000")
    settings = MODULE.build_settings(_config(
        tmp_path,
        models=[
            {"model_dir": "run_a", "label": "pt"},
            {"model_dir": "run_b", "stage": "FT", "checkpoint": "005000", "label": "ft"},
        ],
        video={"enable": False, "max_per_task": 5},
    ))
    assert json.loads(settings["models_labels"]) == ["pt", "ft"]
    assert settings["panel_count"] == 2
    assert settings["max_videos_per_task"] == 0  # video.enable=false renders nothing
    assert settings["eval_out_dir"].name == "compare_pt_vs_ft_libero_10_offset25_graph"
    named = MODULE.build_settings(_config(tmp_path, output_name="my_eval"))
    assert named["eval_out_dir"].name == "my_eval" and named["wandb_run_name"] == "my_eval"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"models": []}, "non-empty list"),
        ({"models": [{"model_dir": "run_a", "skill_source": "gt"}]}, "supports only"),
        ({"models": [{"model_dir": "run_a", "label": "x"}, {"model_dir": "run_a", "label": "x"}]}, "Duplicate panel label"),
        ({"task_ids": [1, 1]}, "unique task ids"),
        ({"n_action_steps": 11}, "chunk_size=10"),
        ({"eval_graph_only": True}, "Unsupported pi05 eval config keys"),
    ],
)
def test_invalid_configs_are_rejected(tmp_path: Path, overrides: dict, message: str) -> None:
    _checkpoint(tmp_path, "pi05_PT", "run_a", "020000")
    with pytest.raises(ValueError, match=message):
        MODULE.build_settings(_config(tmp_path, **overrides))


def test_non_pi05_checkpoint_is_rejected(tmp_path: Path) -> None:
    _checkpoint(tmp_path, "pi05_PT", "run_a", "020000", type="skill_expert")
    with pytest.raises(ValueError, match="Not a pi05 checkpoint"):
        MODULE.build_settings(_config(tmp_path))


def test_merge_accepts_packed_worker_tags(tmp_path: Path) -> None:
    for tag, task_id, successes in (("w000_t0-0", 0, [True, False]), ("w001_t1-1", 1, [True, True])):
        (tmp_path / f"eval_info_{tag}.json").write_text(json.dumps({
            "per_task": [{"task_group": "libero_10", "task_id": task_id, "metrics": {"successes": successes}}]
        }))
    subprocess.run(
        [sys.executable, str(EVAL_DIR / "src/merge_eval_chunks.py"), f"--out_dir={tmp_path}"], check=True
    )
    merged = json.loads((tmp_path / "eval_info.json").read_text())
    assert [t["task_id"] for t in merged["per_task"]] == [0, 1]
    assert merged["overall"]["pc_success"] == pytest.approx(75.0)


def test_shipped_yaml_has_only_supported_keys() -> None:
    config = MODULE.load_config(EVAL_DIR / "pi05_eval_config.yaml")
    assert not set(config) - MODULE._TOP_LEVEL_KEYS - MODULE._GLOBAL_KEYS
    assert MODULE._model_entries(config)


def test_episode_exact_points_at_the_stage1_init_state_map(tmp_path: Path) -> None:
    _checkpoint(tmp_path, "pi05_PT", "run_a", "020000")
    base = _config(tmp_path, episode_offset=0, oracle={"episode_exact": True, "dataset_source": "libero_10_full_1"})
    with pytest.raises(FileNotFoundError, match="eval_init_states.npz"):
        MODULE.build_settings(base)
    init_states = tmp_path / "dataset/skillvla_dataset/libero_10_full_1/eval_init_states.npz"
    init_states.parent.mkdir(parents=True)
    init_states.touch()
    settings = MODULE.build_settings(base)
    assert settings["episode_exact"] is True and settings["eval_init_states_path"] == init_states
    assert settings["eval_out_dir"].name == "run_a_020000_libero_10_exact"
    with pytest.raises(ValueError, match="must be 0"):
        MODULE.build_settings({**base, "episode_offset": 25})
    with pytest.raises(ValueError, match="dataset_source"):
        MODULE.build_settings({**base, "oracle": {"episode_exact": True}})
    off = MODULE.build_settings(_config(tmp_path))
    assert off["episode_exact"] is False and off["eval_init_states_path"] == ""
