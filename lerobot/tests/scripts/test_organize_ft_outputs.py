"""Directory organization must not delete media or change success metrics."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_skillVLA/FT/eval"
sys.path.insert(0, str(EVAL / "src"))
import organize_outputs as organizer


def test_grouping():
    assert organizer.destination("ft_denoising_last_4models_frozenexpert_dino_7k_fullvlm_preskill_2k_noneboth20k_10eps_th03").as_posix() == "denoising_last/expert_frozen_dino_trainable/pred_preskill/7k"
    assert organizer.destination("ft_vanila_4models_frozenexpert_dino_1k_fullvlm_preskill_2k_noneboth20k_10eps_th03").parts[0] == "vanilla"
    assert "gt_broadcast" in organizer.destination("ft_denoising_4models_5k_GTskill_GTUV_broadcast_noneboth20k_10eps_th03").parts
    with pytest.raises(ValueError):
        organizer.destination("unrecognized")


def test_move_repair_browse_and_idempotency(tmp_path):
    original = "ft_denoising_last_4models_frozenexpert_dino_7k_fullvlm_preskill_2k_noneboth20k_10eps_th03"
    source = tmp_path / original
    video = source / "panels/00_model/videos/libero_10_0/eval_episode_0.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"unchanged media")
    inode = video.stat().st_ino
    metrics = source / "metrics/eval_info.json"
    metrics.parent.mkdir()
    metrics.write_text(json.dumps({"successes": [True, False], "pc_success": 50., "video_paths": [str(video)]}))
    archive = tmp_path / "Prev/old"
    archive.mkdir(parents=True)
    assert organizer.organize(tmp_path)["moves"] == 1
    assert source.is_dir() and not (tmp_path / "output_locations.json").exists()
    result = organizer.organize(tmp_path, apply=True)
    assert result["moves"] == 1 and result["videos"] == 1
    target = tmp_path / organizer.destination(original)
    moved = target / video.relative_to(source)
    assert not source.exists() and moved.read_bytes() == b"unchanged media"
    assert moved.stat().st_ino == inode
    saved = json.loads((target / "metrics/eval_info.json").read_text())
    assert saved == {"successes": [True, False], "pc_success": 50., "video_paths": [str(moved)]}
    report = (target / "index.html").read_text()
    assert 'preload="none"' in report and 'src="panels/00_model/videos/libero_10_0/eval_episode_0.mp4"' in report
    assert (tmp_path / "index.html").is_file() and archive.is_dir()
    assert organizer.organize(tmp_path, apply=True)["moves"] == 0
    spec = importlib.util.spec_from_file_location("stage1_config_organize_test", EVAL.parents[1] / "PT/eval/src/stage1_eval_config.py")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    assert config._organized_output_dir(tmp_path, original) == target
    assert config._organized_output_dir(tmp_path, "new_run") == tmp_path / "new_run"


def test_collision_never_overwrites_results(tmp_path):
    name = "ft_learnable_4models_frozenexpert_dino_1k_fullvlm_preskill_2k_noneboth20k_10eps_th03"
    source = tmp_path / name
    source.mkdir()
    target = tmp_path / organizer.destination(name)
    target.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        organizer.organize(tmp_path, apply=True)
    assert source.is_dir() and target.is_dir()
