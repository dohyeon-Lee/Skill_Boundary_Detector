from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "examples/libero/configs/train_skills/skill_eval/src"
sys.path.insert(0, str(SRC))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CONFIG = _load("foveated_skill_preview_config_test", "foveated_skill_preview_config.py")
RUNNER = _load("run_foveated_skill_preview_test", "run_foveated_skill_preview.py")


def test_fixed_camera_projection_uses_libero_display_flip() -> None:
    xml = '<mujoco><worldbody><camera name="agentview" pos="0 0 0" quat="1 0 0 0"/></worldbody></mujoco>'
    transform = RUNNER._camera_transform_from_recorded_xml(
        xml, camera_name="agentview", height=256, width=256
    )
    # Identity MuJoCo camera looks down -Z after robosuite's axis correction.
    assert RUNNER._project_eef(
        np.asarray([0.0, 0.0, -1.0]), transform, height=256, width=256
    ) == (127, 128)


def test_projection_clamps_valid_point_to_visible_image_edge() -> None:
    xml = '<mujoco><worldbody><camera name="agentview" pos="0 0 0" quat="1 0 0 0"/></worldbody></mujoco>'
    transform = RUNNER._camera_transform_from_recorded_xml(
        xml, camera_name="agentview", height=256, width=256
    )
    # The projected point is beyond the right edge before LIBERO's display
    # flip, so the visible foveation center must be clamped to the left edge.
    assert RUNNER._project_eef(
        np.asarray([2.0, 0.0, -1.0]), transform, height=256, width=256
    ) == (0, 128)


def test_foveation_keeps_center_exact_and_blurs_periphery() -> None:
    yy, xx = np.mgrid[:64, :64]
    checker = (((xx + yy) % 2) * 255).astype(np.uint8)
    image = np.repeat(checker[..., None], 3, axis=2)
    output = RUNNER.foveate_image(
        image,
        center_xy=(32, 32),
        shape="square",
        sharp_size=16,
        feather=4,
        blur_radius=5.0,
    )
    assert np.array_equal(output[32, 32], image[32, 32])
    assert not np.array_equal(output[0, 0], image[0, 0])
    assert 80 <= int(output[0, 0, 0]) <= 175


def test_crop_at_image_edge_shifts_window_without_black_padding() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[..., 0] = np.arange(8, dtype=np.uint8)[None, :] + 20
    image[..., 1] = np.arange(8, dtype=np.uint8)[:, None] + 40
    image[..., 2] = 100
    output = RUNNER.crop_focus_image(
        image,
        center_xy=(0, 0),
        crop_size=4,
        output_size=4,
        inner_box_enabled=False,
    )
    assert RUNNER._crop_bounds(
        center_xy=(0, 0), crop_size=4, height=8, width=8
    ) == (0, 0, 4, 4)
    assert np.array_equal(output, image[:4, :4])
    assert np.all(output[..., 2] == 100)


def test_crop_window_shifts_inward_at_bottom_right() -> None:
    assert RUNNER._crop_bounds(
        center_xy=(7, 7), crop_size=4, height=8, width=8
    ) == (4, 4, 8, 8)


def test_crop_draws_inner_red_box_after_resize() -> None:
    image = np.full((8, 8, 3), 80, dtype=np.uint8)
    output = RUNNER.crop_focus_image(
        image,
        center_xy=(4, 4),
        crop_size=8,
        output_size=8,
        inner_box_enabled=True,
        inner_box_center_xy=(4, 4),
        inner_box_size=4,
        inner_box_line_width=1,
    )
    assert np.array_equal(output[2, 2], np.asarray([255, 0, 0], dtype=np.uint8))
    assert np.array_equal(output[4, 4], np.asarray([80, 80, 80], dtype=np.uint8))


def test_crop_can_use_inner_blur_instead_of_red_box() -> None:
    yy, xx = np.mgrid[:32, :32]
    checker = (((xx + yy) % 2) * 255).astype(np.uint8)
    image = np.repeat(checker[..., None], 3, axis=2)
    output = RUNNER.crop_focus_image(
        image,
        center_xy=(16, 16),
        crop_size=32,
        output_size=32,
        inner_box_enabled=True,
        inner_box_mode="blur",
        inner_box_center_xy=(16, 16),
        inner_box_size=8,
        inner_blur_shape="square",
        inner_blur_feather=0,
        inner_blur_radius=3.0,
    )
    assert np.array_equal(output[16, 16], image[16, 16])
    assert not np.array_equal(output[0, 0], image[0, 0])
    assert not np.array_equal(output[0, 0], np.asarray([255, 0, 0], dtype=np.uint8))


def test_inner_box_is_kept_inside_outer_crop() -> None:
    assert RUNNER._box_bounds_inside(
        center_xy=(0, 0),
        box_size=4,
        outer_bounds=(0, 0, 8, 8),
    ) == (0, 0, 4, 4)


def test_config_resolves_existing_skillvla_run_without_fsq_checkpoint(tmp_path: Path) -> None:
    run = tmp_path / "dataset" / "skillvla_dataset" / "ds" / "run"
    (run / "skillvla" / "meta").mkdir(parents=True)
    (run / "skillvla" / "meta" / "info.json").write_text(
        json.dumps({"skill_fsq_levels": [3, 3, 3]})
    )
    (run / "skill_latents.npz").write_bytes(b"assignments")
    (run.parent / "eval_init_states.npz").write_bytes(b"exact")
    (tmp_path / "libero_original_dataset" / "libero_90").mkdir(parents=True)
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
            "output_name": "preview",
            "foveation": {
                "mode": "crop",
                "crop_size": 80,
                "output_size": 224,
                "inner_box": {
                    "enabled": True,
                    "mode": "blur",
                    "size": 24,
                    "line_width": 2,
                },
            },
            "train_partition": ["debug"],
            "train_qos": "base_qos",
        }
    )
    assert settings["skill_latents_path"] == str(run / "skill_latents.npz")
    assert settings["foveation_mode"] == "crop"
    assert settings["foveation_crop_size"] == 80
    assert settings["foveation_output_size"] == 224
    assert settings["foveation_inner_box_enabled"] == 1
    assert settings["foveation_inner_box_mode"] == "blur"
    assert settings["foveation_inner_box_size"] == 24
    assert settings["foveation_inner_box_line_width"] == 2
    assert Path(settings["preview_output_dir"]).name == "preview_crop"
    assert settings["foveation_sharp_size"] == 96
    assert settings["random_color_enabled"] == 1
    assert settings["random_crop_offset_px"] == "[-24,24]"
    assert settings["random_crop_inner_box_offset_px"] == "[-4,4]"
    assert settings["random_blur_radius"] == "[0.0,4.0]"
    assert "FSQ_epoch" not in json.dumps(settings)


def test_output_mode_suffix_is_replaced_instead_of_duplicated(tmp_path: Path) -> None:
    run = tmp_path / "dataset" / "skillvla_dataset" / "ds" / "run"
    (run / "skillvla" / "meta").mkdir(parents=True)
    (run / "skillvla" / "meta" / "info.json").write_text(
        json.dumps({"skill_fsq_levels": [3, 3, 3]})
    )
    (run / "skill_latents.npz").write_bytes(b"assignments")
    (run.parent / "eval_init_states.npz").write_bytes(b"exact")
    (tmp_path / "libero_original_dataset" / "libero_90").mkdir(parents=True)
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
            "output_name": "preview_crop",
            "foveation": {"mode": "partial_fov"},
        }
    )
    assert Path(settings["preview_output_dir"]).name == "preview_partial_fov"


def test_preview_html_has_code_selection_but_no_fsq_metrics() -> None:
    report = RUNNER._report_html(
        {
            "title": "preview",
            "levels": [3, 3, 3],
            "records": [],
            "foveation": {"sharp_size": 96, "feather": 20, "blur_radius": 18.0},
        }
    )
    assert 'id="cube"' in report
    assert 'data-view="task"' in report
    assert "function selectTask" in report
    assert 'data-variant="color"' in report
    assert 'data-variant="crop"' in report
    assert 'data-variant="blur"' in report
    assert "reconstruction loss" not in report.lower()
    assert "termination timing" not in report.lower()


def test_input_blur_is_applied_before_foveation() -> None:
    yy, xx = np.mgrid[:64, :64]
    checker = (((xx + yy) % 2) * 255).astype(np.uint8)
    image = np.repeat(checker[..., None], 3, axis=2)
    blurred_source = RUNNER.input_blur_image(image, radius=3.0)
    output = RUNNER.foveate_image(
        blurred_source,
        center_xy=(32, 32),
        shape="square",
        sharp_size=16,
        feather=0,
        blur_radius=5.0,
    )
    # The center comes from the already-augmented source, not the raw frame.
    assert np.array_equal(output[32, 32], blurred_source[32, 32])
    assert not np.array_equal(output[32, 32], image[32, 32])


def test_crop_position_randomization_is_clamped() -> None:
    assert RUNNER._jitter_center(
        (4, 60), dx=-20, dy=20, height=64, width=64
    ) == (0, 63)
