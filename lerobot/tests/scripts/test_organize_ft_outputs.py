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


@pytest.mark.parametrize("name,expected", [
    ("ft_denoising_wovis_pi05vlm_2models_frozenexpert_10k_fullvlm_preskill_2k_noneboth20k_10eps_th03",
     "pi05vlm/denoising_wovis/expert_frozen_vision_trainable/pred_preskill/10k"),
    ("ft_denoising_wovis_pi05vlm_bridgehead2_5k_fullvlm_preskill_2k_noneboth20k_10eps_th03",
     "pi05vlm/denoising_wovis/bridge_and_head_trainable/pred_preskill/5k"),
    ("ft_dual_bottleneck_pi05vlm_9_9_10k_20k_fullvlm_preskill_2k_noneboth20k_10eps_th03",
     "pi05vlm/dual_bottleneck_9_9/expert_frozen_vision_trainable/pred_preskill/10k_20k"),
    ("ft_dual_bottleneck_pi05vlm_9_9_5k_vanila_9_9_30k_fullvlm_preskill_2k_noneboth20k_10eps_th03",
     "pi05vlm/comparisons/dual_bottleneck_5k_vs_vanilla_30k/pred_preskill"),
    ("ft_vanila_last_pi05vlm_9_9_fromPT100k_frozenexpert_5k_10k_fullvlm_preskill_2k_noneboth20k_10eps_th03",
     "pi05vlm/vanilla_9_9/expert_frozen_vision_trainable/pred_preskill/5k_10k"),
    ("ft_learnable_last_2models_frozenexpert_dino_3k_fullvlm_preskill_2k_noneboth20k_10eps_th03",
     "learnable_last/expert_frozen_dino_trainable/pred_preskill/3k"),
    ("ft_vanila_nouv_last_frozenexpert_dino_5k_fullvlm_preskill_2k_noneboth20k_10eps_th03_shared_init_range2_bank07525bd7",
     "vanilla/expert_frozen_dino_trainable/pred_preskill/random_range2/5k"),
])
def test_new_result_families_do_not_collide(name, expected):
    assert organizer.destination(name).as_posix() == expected


def test_summary_presentation_and_absolute_trace_links(tmp_path):
    name = "ft_learnable_last_2models_frozenexpert_dino_3k_fullvlm_preskill_2k_noneboth20k_10eps_th03"
    source = tmp_path / name
    video = source / "panels/00_model/videos/libero_10_0/presentation/eval_episode_0.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"presentation")
    trace = source / "panels/00_model/skill_html/libero_10_task00/skill_trace.html"
    trace.parent.mkdir(parents=True)
    trace.write_text(f'<a href="{video}">video</a>')
    metrics = source / "metrics/eval_info_merged.json"
    metrics.parent.mkdir()
    metrics.write_text(json.dumps({"model": {"overall": {"pc_success": 50}, "per_task": [
        {"task_id": 0, "metrics": {"successes": [True, False]}}
    ]}}))
    organizer.organize(tmp_path, apply=True)
    target = tmp_path / organizer.destination(name)
    assert str(target / video.relative_to(source)) in (target / trace.relative_to(source)).read_text()
    report = (target / "index.html").read_text()
    assert "presentation/eval_episode_0.mp4" in report
    assert "50%" in report and "2 episodes" in report
    assert "50%" in (tmp_path / "index.html").read_text()


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
