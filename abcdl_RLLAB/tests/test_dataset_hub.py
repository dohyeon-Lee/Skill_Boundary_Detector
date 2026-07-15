"""Tests for LeRobot-style HuggingFace wiring in AbcdlDataset — network-free."""

from __future__ import annotations

import shutil

import abcdl.dataset as dsmod
from abcdl.dataset import AbcdlDataset


def test_repo_id_autopull_then_load(monkeypatch, tmp_abcdl_episode, tmp_path):
    # A repo_id (owner/name, not an existing path) triggers hf.pull, whose returned
    # dir is then loaded as a normal abcdl root. Point pull at a real local episode root.
    root = tmp_path / "pulled"
    root.mkdir()
    shutil.copytree(tmp_abcdl_episode, root / "episode_0000")
    seen = {}
    monkeypatch.setattr(dsmod.hf, "pull", lambda **k: seen.update(k) or str(root))
    ds = AbcdlDataset("jellyho/yam_pick", version="v1")
    assert seen["repo_id"] == "jellyho/yam_pick"
    assert seen["fmt"] == "abcdl" and seen["version"] == "v1"
    assert ds.num_episodes == 1 and len(ds) == 6


def test_push_to_hub_calls_hf_push(monkeypatch, tmp_abcdl_episode, tmp_path):
    seen = {}
    monkeypatch.setattr(dsmod.hf, "push", lambda **k: seen.update(k) or k["fmt"])
    # Build a proper root (parent dir with an episode subdir) from the fixture.
    root = tmp_path / "ds"
    root.mkdir()
    shutil.copytree(tmp_abcdl_episode, root / "episode_0000")
    ds = AbcdlDataset(str(root))  # existing local dir → root
    out = ds.push_to_hub("jellyho/yam_pick", "v2", fmt="abcdl")
    assert seen["repo_id"] == "jellyho/yam_pick"
    assert seen["fmt"] == "abcdl" and seen["version"] == "v2"
    assert seen["local_dir"] == ds.root
