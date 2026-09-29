from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "examples/libero/configs/train_skills/skill_eval/eef_projection/src"
sys.path.insert(0, str(SRC))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CONFIG = _load("skill_attention_preview_config_test", "config.py")
RUNNER = _load("run_skill_attention_preview_test", "preview.py")


def test_fixed_camera_projection_uses_libero_display_flip() -> None:
    xml = '<mujoco><worldbody><camera name="agentview" pos="0 0 0" quat="1 0 0 0"/></worldbody></mujoco>'
    transform = RUNNER.camera_transform_from_recorded_xml(
        xml, camera_name="agentview", height=256, width=256
    )
    result = RUNNER._agent_projection(
        np.asarray([0.0, 0.0, -1.0]),
        transform,
        grid=16,
        height=256,
        width=256,
    )
    assert result["visible"] is True
    assert result["valid"] is True
    assert np.allclose(result["pixel"], [127.0, 128.0])
    assert result["cell"] == [8, 7]


def test_fixed_camera_projection_does_not_clip_invalid_target_into_a_patch() -> None:
    xml = '<mujoco><worldbody><camera name="agentview" pos="0 0 0" quat="1 0 0 0"/></worldbody></mujoco>'
    transform = RUNNER.camera_transform_from_recorded_xml(
        xml, camera_name="agentview", height=256, width=256
    )
    result = RUNNER._agent_projection(
        np.asarray([2.0, 0.0, -1.0]),
        transform,
        grid=14,
        height=256,
        width=256,
    )
    assert result["visible"] is False
    assert result["valid"] is False
    assert result["cell"] is None
    assert result["reason"] == "outside image"


def test_sampled_frames_always_include_start_and_endpoint() -> None:
    assert RUNNER._frame_indices(3, 11, 5) == [3, 5, 7, 9, 11]
    assert RUNNER._frame_indices(3, 4, 5) == [3, 4]
    assert RUNNER._frame_indices(3, 3, 5) == [3]


def test_attention_drawing_keeps_source_size_and_marks_target_patch() -> None:
    image = np.full((56, 56, 3), 50, dtype=np.uint8)
    output = RUNNER._draw_attention_target(
        image,
        target_pixel=(30.0, 18.0),
        eef_pixel=(28.0, 42.0),
        target_visible=True,
        grid=14,
        sigma=0.7,
    )
    assert output.shape == image.shape
    assert not np.array_equal(output, image)
    # Red target crosshair and blue EEF marker are both present.
    assert bool(((output[..., 0] > 220) & (output[..., 1] < 80)).any())
    assert bool(((output[..., 2] > 220) & (output[..., 0] < 80)).any())


def test_wrist_endpoint_is_visible_but_masked_when_it_is_the_eef_patch() -> None:
    states = np.asarray([[0.0, 0.0, 0.8, 0.0, 0.0, 0.0]], dtype=np.float64)
    records, eef_pixels = RUNNER._wrist_projections(
        states,
        states[0, :3],
        grid=14,
        height=256,
        width=256,
    )
    assert records[0]["visible"] is True
    assert records[0]["valid"] is False
    assert records[0]["reason"] == "masked: same patch as current EEF"
    assert eef_pixels[0] is not None


def _fake_run(tmp_path: Path) -> Path:
    run = tmp_path / "dataset" / "skillvla_dataset" / "ds" / "run"
    (run / "skillvla" / "meta").mkdir(parents=True)
    (run / "skillvla" / "meta" / "info.json").write_text(
        json.dumps({"skill_fsq_levels": [3, 3, 3]})
    )
    (run / "skill_latents.npz").write_bytes(b"assignments")
    (run.parent / "eval_init_states.npz").write_bytes(b"exact")
    (tmp_path / "libero_original_dataset" / "libero_90").mkdir(parents=True)
    return run


def test_config_resolves_dual_camera_attention_contract(tmp_path: Path) -> None:
    run = _fake_run(tmp_path)
    settings = CONFIG.build_settings(
        {
            "project_root": str(tmp_path),
            "dataset_root": "dataset",
            "source_dataset": "ds",
            "skillvla_run": "run",
            "target_task": "libero_90",
            "original_dataset_dir": "libero_original_dataset/libero_90",
            "task_ids": [3],
            "episodes_per_task": 1,
            "output_name": "attention",
            "cameras": {
                "agent": {
                    "name": "agentview",
                    "video_key": "observation.images.image",
                },
                "wrist": {"video_key": "observation.images.wrist_image"},
            },
            "attention": {
                "patch_grid": 16,
                "soft_sigma": 0.5,
                "frames_per_skill": 7,
            },
            "input_xyz_noise": {
                "std_m": [0.0, 0.01, 0.02],
                "samples_per_level": 24,
            },
            "slurm": {
                "partition": "dell_cpu",
                "qos": "cpu_qos",
                "gres": "",
            },
            "train_partition": ["debug"],
            "train_qos": "base_qos",
        }
    )
    assert settings["skill_latents_path"] == str(run / "skill_latents.npz")
    assert settings["agent_camera"] == "agentview"
    assert settings["agent_video_key"] == "observation.images.image"
    assert settings["wrist_video_key"] == "observation.images.wrist_image"
    assert settings["patch_grid"] == 16
    assert settings["soft_sigma"] == 0.5
    assert settings["frames_per_skill"] == 7
    assert settings["noise_std_m"] == "[0.0,0.01,0.02]"
    assert settings["noise_samples"] == 24
    assert settings["preview_partition"] == "dell_cpu"
    assert settings["preview_qos"] == "cpu_qos"
    assert settings["preview_gres"] == ""
    assert Path(settings["preview_output_dir"]).name == "attention"
    assert "foveation" not in json.dumps(settings).lower()


def test_preview_html_exposes_both_cameras_and_no_foveation_controls() -> None:
    report = RUNNER._report_html(
        {
            "title": "Skill attention targets: agent + wrist",
            "records": [],
            "noise_levels": [{"key": "sigma_0mm", "std_mm": 0.0}],
            "noise_stats": {
                "sigma_0mm": {
                    "agent": {"p95_shift_px": 0.0, "patch_change_fraction": 0.0},
                    "wrist": {"p95_shift_px": 0.0, "patch_change_fraction": 0.0},
                }
            },
            "stats": {
                "agent_valid_fraction": 0.0,
                "wrist_visible_fraction": 0.0,
                "wrist_valid_fraction": 0.0,
            },
        }
    )
    assert "agent + wrist" in report
    assert "cameraRow(r,'agent','agent')" in report
    assert "cameraRow(r,'wrist','wrist')" in report
    assert 'id="task"' in report
    assert 'id="token"' in report
    assert 'id="noise"' in report
    assert "foveation controls" not in report.lower()


def test_noise_draws_are_stable_and_scale_across_levels() -> None:
    base = RUNNER._standard_noise(seed=7, uid="skill", samples=8)
    repeated = RUNNER._standard_noise(seed=7, uid="skill", samples=8)
    assert np.array_equal(base, repeated)
    assert np.allclose(base * 0.02, 2.0 * (base * 0.01))
