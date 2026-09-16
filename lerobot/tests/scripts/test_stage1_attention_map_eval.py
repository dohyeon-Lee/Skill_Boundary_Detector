import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/stage1_eval/src"
)
sys.path.insert(0, str(_SRC))

from attention_map_eval import AttentionMapCapture, composed_camera_attention  # noqa: E402
from merge_attention_maps import merge as merge_attention_maps  # noqa: E402
from lerobot.policies.skill_expert.layerwise_cond_bottleneck import (  # noqa: E402
    LayerwiseCondBottleneckSkillExpert,
)


def test_arch3_attention_composition_splits_both_cameras() -> None:
    # Two bridge queries vote for the first latent, which reads top patch 1.
    # Four patches per camera => 2x2 maps after each CLS position.
    bridge = torch.zeros(1, 2, 2, 2)
    bridge[..., 0] = 1
    reader = torch.zeros(1, 2, 2, 10)
    reader[..., 0, 1] = 1
    reader[..., 1, 7] = 1
    top, wrist, mass = composed_camera_attention(bridge, reader)
    assert top.shape == wrist.shape == (2, 2)
    assert top[0, 0] == pytest.approx(1)
    assert wrist.sum() == pytest.approx(0)
    assert mass["top_mass"] == pytest.approx(1)
    assert mass["wrist_mass"] == pytest.approx(0)


def test_arch3_attention_rejects_non_square_patch_layout() -> None:
    bridge = torch.ones(1, 1, 1, 2)
    reader = torch.ones(1, 1, 2, 12)
    with pytest.raises(ValueError, match="not square"):
        composed_camera_attention(bridge, reader)


def test_attention_capture_writes_both_camera_maps_and_restores_modules(tmp_path) -> None:
    class Reader(nn.Module):
        def __init__(self):
            super().__init__()
            self.cross_attention = nn.MultiheadAttention(8, 2, batch_first=True)

    class SmallArch3(LayerwiseCondBottleneckSkillExpert):
        def __init__(self):
            nn.Module.__init__(self)
            self.config = SimpleNamespace(visual_bridge_last_n_layers=2)
            self.gemma_expert = SimpleNamespace(
                model=SimpleNamespace(config=SimpleNamespace(num_hidden_layers=2))
            )
            self.layerwise_condition_readers = nn.ModuleList([Reader(), Reader()])
            self.visual_bridge_attention = nn.MultiheadAttention(8, 2, batch_first=True)
            self.visual_bridge_gates = nn.Parameter(torch.zeros(2))

    model = SmallArch3()

    class Policy:
        def __init__(self):
            self.model = model

        def predict_action_chunk(self, batch):
            query = torch.rand(1, 2, 8)
            cond = torch.rand(1, 10, 8)
            for reader in model.layerwise_condition_readers:
                reader.cross_attention(query, cond, cond, need_weights=False)
            for _ in range(2):
                for _ in range(2):
                    model.visual_bridge_attention(
                        torch.rand(1, 3, 8), query, query, need_weights=False
                    )
            return torch.zeros(1, 3, 7)

    wrapper = SimpleNamespace(policy=Policy(), _episode_step=0)
    capture = AttentionMapCapture(
        wrapper,
        tmp_path,
        {
            "max_chunks_per_task": 1,
            "every_n_chunks": 1,
            "expert_layers": "last",
            "denoise_step": "last",
        },
    )
    original = model.visual_bridge_attention.forward
    capture.start_task("libero_90", 0)
    capture.attach()
    batch = {
        "skill_code": torch.tensor([3]),
        "observation.images.image": torch.rand(1, 3, 16, 16),
        "observation.images.wrist_image": torch.rand(1, 3, 16, 16),
    }
    wrapper.policy.predict_action_chunk(batch)
    capture.detach()
    capture.finish()
    assert model.visual_bridge_attention.forward == original
    assert (tmp_path / "libero_90_task00" / "index.html").is_file()
    assert len(list(tmp_path.rglob("*_top.png"))) == 1
    assert len(list(tmp_path.rglob("*_wrist.png"))) == 1
    assert capture.records[0]["layer"] == 2
    assert capture.records[0]["denoise_step"] == 2


def test_parallel_attention_merge_keeps_disjoint_task_reports(tmp_path) -> None:
    folder = tmp_path / "panels" / "00_arch3" / "attention_maps"
    for task_id in (0, 1):
        task = folder / f"libero_90_task{task_id:02d}"
        task.mkdir(parents=True)
        (task / "attention_maps.json").write_text(
            f'[{{"task": "libero_90_task{task_id:02d}"}}]', encoding="utf-8"
        )
        (task / "index.html").write_text("complete", encoding="utf-8")
        counts = merge_attention_maps(tmp_path, expected_tasks=2)
        assert counts == {"00_arch3": task_id + 1}
    root_page = (tmp_path / "attention_maps.html").read_text()
    panel_page = (folder / "index.html").read_text()
    assert "2/2 tasks" in root_page
    assert "libero_90_task00" in panel_page
    assert "libero_90_task01" in panel_page
    assert len(json.loads((folder / "attention_maps.json").read_text())) == 2
