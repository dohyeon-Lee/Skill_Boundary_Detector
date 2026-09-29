"""Standalone Stage-1 wrist alignment-map evaluator helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/stage1/eval/src"
)
sys.path.insert(0, str(_SRC))

from eval_config import load_eval_config  # noqa: E402
from visualize_attention import (  # noqa: E402
    evenly_spaced,
    normalize_and_pad_state,
    save_action_panel,
    save_panel,
    write_gallery,
)


def _fake_config(tmp_path: Path) -> Path:
    checkpoint = (
        tmp_path
        / "outputs/skillVLA_stage1/VSA/test_run/checkpoints/100/pretrained_model"
    )
    dataset = tmp_path / "datasets/skillvla_dataset/test_source/test_space/skillvla"
    dino = tmp_path / "models/dino"
    checkpoint.mkdir(parents=True)
    (dataset / "meta").mkdir(parents=True)
    dino.mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "skill_expert",
                "architecture_label": "arch18_align_skill",
                "visual_bottleneck_tokens": 100,
                "skill_code_space_id": "test_space",
                "dino_model_path": "/old/server/models/dino",
            }
        )
    )
    (checkpoint / "model.safetensors").touch()
    (dataset / "meta" / "info.json").write_text(
        json.dumps({"repo_id": "skillvla/test"})
    )
    (dataset / "meta" / "stats.json").write_text("{}")
    (dino / "config.json").write_text("{}")
    (tmp_path / "global_config.yaml").write_text(
        yaml.safe_dump(
            {
                "project_root": str(tmp_path),
                "outputs_root": "outputs",
                "dataset_root": "datasets",
                "train_partition": ["gpu_a", "gpu_b"],
                "train_qos": "base_qos",
                "train_nodelist": "",
                "train_exclude_nodes": ["node_bad"],
            }
        )
    )
    (tmp_path / "stage1_common_config.yaml").write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "skillvla_root": "skillvla_dataset",
                    "source": "test_source",
                    "run": "newer_mutable_run",
                }
            }
        )
    )
    path = tmp_path / "eval.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "checkpoint": {
                    "architecture": "arch18_align_skill",
                    "step": 100,
                },
                "samples": {
                    "episodes": [2],
                    "skills_per_episode": 2,
                    "frames_per_skill": 3,
                    "top_queries": 5,
                },
            }
        )
    )
    return path


def test_config_resolves_paths_and_alignment_contract(tmp_path: Path) -> None:
    config = load_eval_config(_fake_config(tmp_path))
    assert config.architecture_label == "arch18_align_skill"
    assert config.visual_bottleneck_tokens == 100
    assert config.episode_ids == (2,)
    assert config.dataset_dir == (
        tmp_path / "datasets/skillvla_dataset/test_source/test_space/skillvla"
    )
    assert config.dino_model_path == tmp_path / "models" / "dino"
    assert config.slurm["partition"] == "gpu_a,gpu_b"
    assert config.output_dir.name == "arch18_align_skill_100"
    assert config.action_maps is True
    assert config.action_probe_time == 0.5


def test_config_rejects_more_panels_than_queries(tmp_path: Path) -> None:
    path = _fake_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["samples"]["top_queries"] = 101
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="exceeds bottleneck tokens"):
        load_eval_config(path)


def test_config_can_disable_action_maps_with_one_boolean(tmp_path: Path) -> None:
    path = _fake_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["action_maps"] = False
    path.write_text(yaml.safe_dump(raw))
    assert load_eval_config(path).action_maps is False


def test_config_maps_stage1_eval_task_id_to_dataset_task_by_exact_language(
    tmp_path: Path,
) -> None:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = _fake_config(tmp_path)
    dataset_meta = (
        tmp_path
        / "datasets/skillvla_dataset/test_source/test_space/skillvla/meta"
    )
    pq.write_table(
        pa.table(
            {
                "task_index": [70],
                "task": ["close the top drawer of the cabinet"],
            }
        ),
        dataset_meta / "tasks.parquet",
    )
    np.savez(
        dataset_meta.parents[2] / "eval_init_states.npz",
        episode_index=np.asarray([123], dtype=np.int32),
        scene_file=np.asarray(
            ["KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_demo.hdf5"]
        ),
    )
    raw = yaml.safe_load(path.read_text())
    raw["samples"].update(
        {"task_suite": "libero_90", "tasks": [0], "episodes": []}
    )
    path.write_text(yaml.safe_dump(raw))

    config = load_eval_config(path)
    assert config.task_suite == "libero_90"
    assert config.task_ids == (0,)
    assert config.dataset_task_ids == (70,)
    assert config.task_names == ("close the top drawer of the cabinet",)
    assert config.task_episode_ids == ((123,),)


def test_state_normalization_matches_quantile_contract_and_pads() -> None:
    state = torch.tensor([[0.0, 2.0]])
    stats = {"q01": [-1.0, 0.0], "q99": [1.0, 4.0]}
    normalized = normalize_and_pad_state(state, stats, 4)
    torch.testing.assert_close(normalized, torch.tensor([[0.0, 0.0, 0.0, 0.0]]))
    assert evenly_spaced(list(range(10)), 3) == [0, 4, 9]


def test_panel_ranks_queries_by_weighted_spatial_contribution(tmp_path: Path) -> None:
    pooled = torch.zeros(4)
    per_query = torch.tensor(
        [[0.0, 5.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 2.0]]
    )
    weights = torch.tensor([0.1, 0.1, 0.8])
    weighted = weights[:, None] * per_query
    pooled.copy_(weighted.sum(dim=0))
    centered = weighted - weighted.mean(dim=-1, keepdim=True)
    contribution = centered.square().mean(dim=-1).sqrt()
    output = tmp_path / "panel.png"
    top = save_panel(
        output,
        torch.zeros(3, 16, 16).permute(1, 2, 0).numpy(),
        pooled,
        per_query,
        weights,
        contribution,
        top_n=2,
        alpha=0.5,
        target_cell=1,
    )
    assert output.is_file()
    assert top == [2, 0]


def test_action_panel_draws_each_timestep_for_both_diagnostics(tmp_path: Path) -> None:
    output = tmp_path / "action.png"
    save_action_panel(
        output,
        torch.zeros(16, 16, 3).numpy(),
        torch.arange(12, dtype=torch.float32).reshape(3, 4),
        torch.arange(12, 0, -1, dtype=torch.float32).reshape(3, 4),
        alpha=0.5,
        target_cell=1,
    )
    assert output.is_file()


def test_gallery_groups_images_by_task_and_skill(tmp_path: Path) -> None:
    manifest = {
        "architecture": "arch18_align_skill",
        "selected_tasks": [
            {"task_id": 0, "task": "task zero"},
            {"task_id": 1, "task": "task one"},
        ],
        "samples": [
            {
                "panel": "images/a.png",
                "action_panel": "images/a_action.png",
                "task_id": 0,
                "task": "task zero",
                "episode": 10,
                "skill_index": 2,
                "skill_code": 7,
                "frame": 3,
                "target_valid": True,
                "target_cell": 1,
                "predicted_cell": 1,
                "cell_distance": 0,
                "top_query_indices": [4],
            },
            {
                "panel": "images/b.png",
                "task_id": 1,
                "task": "task one",
                "episode": 20,
                "skill_index": 0,
                "skill_code": 5,
                "frame": 1,
                "target_valid": True,
                "target_cell": 2,
                "predicted_cell": 3,
                "cell_distance": 1,
                "top_query_indices": [8],
            },
        ],
    }
    page = write_gallery(tmp_path, manifest).read_text()
    assert page.count('class="task-button"') == 2
    assert 'data-task="0" data-skill-index="2"' in page
    assert "function selectTask(button)" in page
    assert "function selectView(value)" in page
    assert "images/a_action.png" in page
    assert "Skill ${index} · code ${code}" in page
