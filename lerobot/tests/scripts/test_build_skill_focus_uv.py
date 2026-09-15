import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/build_data/src"
)
sys.path.insert(0, str(_SRC))
import build_skill_focus_uv as focus_builder  # noqa: E402


def test_build_records_explicit_start_and_backward_compatible_end_uv(
    tmp_path: Path, monkeypatch
) -> None:
    skill_dataset = tmp_path / "skillvla"
    info_path = skill_dataset / "meta/info.json"
    info_path.parent.mkdir(parents=True)
    info_path.write_text(
        json.dumps(
            {
                "features": {
                    "observation.images.image": {
                        "shape": [3, 8, 8],
                        "names": ["channels", "height", "width"],
                    }
                }
            }
        )
    )
    latents = tmp_path / "skill_latents.npz"
    np.savez_compressed(
        latents,
        episode_id=np.asarray([0, 0]),
        task_id=np.asarray([3, 3]),
        skill_index=np.asarray([0, 1]),
        frame_start=np.asarray([0, 2]),
        frame_end=np.asarray([2, 4]),
    )
    states = np.zeros((5, 7), dtype=np.float32)
    states[:, 0] = np.arange(5, dtype=np.float32) / 10.0
    states[:, 1] = np.arange(5, dtype=np.float32) / 20.0

    class FakeDataset:
        proprio_grounding = "none"

        def __init__(self, **kwargs):
            del kwargs

        def load_aligned_episode(self, episode_id: int):
            assert episode_id == 0
            return SimpleNamespace(
                model_xml="<mujoco/>",
                filtered_states=states,
                episode_start_xyz=np.zeros(3, dtype=np.float32),
            )

    monkeypatch.setattr(
        focus_builder, "_skill_evaluation_dataset_cls", lambda: FakeDataset
    )
    monkeypatch.setattr(
        focus_builder,
        "camera_transform_from_recorded_xml",
        lambda *args, **kwargs: np.eye(4),
    )

    def fake_project(xyz, transform, *, height, width):
        del transform, height, width
        xy = np.asarray(xyz[:2], dtype=np.float32)
        return SimpleNamespace(
            normalized_xy=xy,
            pixel_xy=np.rint(xy * 10).astype(np.int32),
            raw_xy=xy * 10,
            valid=True,
            clipped=False,
        )

    monkeypatch.setattr(focus_builder, "project_eef", fake_project)
    output = tmp_path / "skill_focus_uv.npz"
    focus_builder.build(
        argparse.Namespace(
            skill_dataset_dir=skill_dataset,
            skill_latents_path=latents,
            eval_init_states_path=tmp_path / "unused_init.npz",
            original_dataset_dir=tmp_path / "unused_original",
            suite="libero_10",
            camera="agentview",
            output=output,
        )
    )

    with np.load(output, allow_pickle=False) as built:
        assert int(built["schema_version"]) == 2
        np.testing.assert_allclose(
            built["start_focus_uv"], [[0.0, 0.0], [0.2, 0.1]]
        )
        np.testing.assert_allclose(built["focus_uv"], [[0.2, 0.1], [0.4, 0.2]])
        # Contiguous skills intentionally share boundary coordinates.
        np.testing.assert_allclose(built["focus_uv"][0], built["start_focus_uv"][1])
    assert focus_builder.artifact_is_current(output)

    info = json.loads(info_path.read_text())
    assert info["skill_focus_uv_schema_version"] == 2
    assert info["skill_focus_uv_start"] == "frame_start"
