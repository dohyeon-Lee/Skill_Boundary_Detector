import json
import sys
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_skillVLA/FT/eval"
sys.path.insert(0, str(EVAL / "src"))
from split_model_outputs import model_destination, split_outputs


def test_split_cross_model_metrics_media_and_resume(tmp_path):
    original = "ft_dual_bottleneck_pi05vlm_9_9_5k_vanila_9_9_30k_fullvlm_preskill_2k_noneboth20k_10eps_th03"
    relative = "pi05vlm/comparisons/dual_bottleneck_5k_vs_vanilla_30k/pred_preskill"
    source = tmp_path / relative
    data = {}
    labels = ["dual_bottleneck_pi05vlm_9_9_5k", "vanila_last_pi05vlm_9_9_30k"]
    inodes = {}
    for i, label in enumerate(labels):
        panel = source / "panels" / f"{i:02d}_{label}"
        video = panel / "videos/libero_10_0/eval_episode_0.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(label.encode())
        inodes[label] = video.stat().st_ino
        data[label] = {"overall": {"pc_success": 50}, "per_task": [
            {"task_id": 0, "metrics": {"successes": [True, False], "pc_success": 50,
                                       "video_paths": [str(video)]}}]}
    metrics = source / "metrics"
    metrics.mkdir()
    (metrics / "eval_info_merged.json").write_text(json.dumps(data))
    (metrics / "eval_info_w000.json").write_text(json.dumps(data))
    (tmp_path / "output_locations.json").write_text(json.dumps({"version": 1, "locations": {original: relative}}))
    assert split_outputs(tmp_path)["new_models"] == 2
    assert source.exists()
    result = split_outputs(tmp_path, apply=True, charts=False)
    assert result["models"] == 2
    records = json.loads((tmp_path / "model_results.json").read_text())["results"]
    archive = tmp_path / ".evaluation_runs" / original
    assert json.loads((tmp_path / "output_locations.json").read_text())["locations"][original] == str(archive.relative_to(tmp_path))
    for record in records:
        folder = tmp_path / record["target"]
        label = record["label"]
        saved = json.loads((folder / "metrics/eval_info_merged.json").read_text())
        assert list(saved) == [label]
        assert saved[label]["overall"] == data[label]["overall"]
        assert saved[label]["per_task"][0]["metrics"]["successes"] == [True, False]
        video = folder / "panels" / record["panel"] / "videos/libero_10_0/eval_episode_0.mp4"
        assert video.stat().st_ino == inodes[label]
        assert saved[label]["per_task"][0]["metrics"]["video_paths"] == [str(video)]
        alias = archive / "panels" / record["panel"]
        assert alias.is_symlink() and alias.resolve() == video.parents[2]
        assert not (folder / "panels" / record["panel"]).is_symlink()
        assert (folder / "index.html").exists()
    assert any("vanilla_9_9" in r["target"] and r["target"].endswith("/30k") for r in records)
    assert split_outputs(tmp_path, apply=True, charts=False)["new_models"] == 0


def test_models_with_and_without_uv_stay_separate():
    base = "pi05vlm/denoising_wovis/expert_frozen_vision_trainable/pred_preskill/5k"
    with_uv = model_destination("original", base, "denoising_none_wovis_last_pi05vlm", "5k")
    no_uv = model_destination("original", base, "denoising_none_wovis_nouv_last_pi05vlm", "5k")
    assert with_uv != no_uv and "with_uv" in with_uv.parts and "no_uv" in no_uv.parts


def test_collision_is_detected_before_moving(tmp_path):
    original = "ft_vanila_2models_frozenexpert_dino_5k_test"
    relative = "vanilla/expert_frozen_dino_trainable/pred_preskill/5k"
    source = tmp_path / relative
    (source / "panels/00_vanila").mkdir(parents=True)
    (source / "metrics").mkdir()
    (source / "metrics/eval_info_merged.json").write_text(json.dumps({"vanila": {}}))
    (tmp_path / "output_locations.json").write_text(json.dumps({"version": 1, "locations": {original: relative}}))
    target = tmp_path / model_destination(original, relative, "vanila", "5k")
    target.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        split_outputs(tmp_path, apply=True, charts=False)
    assert source.exists()
