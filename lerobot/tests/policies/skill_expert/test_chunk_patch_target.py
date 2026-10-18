from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.policies.skillVLA.chunk_patch_target import cache_top_focus, load_top_camera_transforms, project_top_points
from lerobot.policies.skillVLA.dataset_skillVLA import SkillVLADataset
from lerobot.policies.skillVLA.focus_projection import camera_transform_from_recorded_xml, project_eef

XML = '<mujoco><worldbody><camera name="agentview" pos="0 0 1" quat="1 0 0 0" fovy="45"/></worldbody></mujoco>'


def test_top_projection_matches_existing_builder_and_masks_invalid_points():
    matrix = camera_transform_from_recorded_xml(XML, camera_name="agentview", height=256, width=256)
    points = np.array([[0., 0., 0.], [.1, .05, .1], [10., 0., 0.], [0., 0., 2.], [np.nan, 0., 0.]])
    uv, valid = project_top_points(points, matrix, height=256, width=256)
    for i in range(3):
        expected = project_eef(points[i], matrix, height=256, width=256)
        np.testing.assert_allclose(uv[i], expected.normalized_xy, atol=1e-6)
        assert valid[i] == expected.valid
    assert valid.tolist() == [True, True, False, False, False]
    assert np.isfinite(uv).all()


def test_embedded_grounded_camera_needs_no_original_dataset(tmp_path):
    path = tmp_path / "skill_focus_uv.npz"
    matrix = np.eye(4)
    np.savez(path, image_height=256, image_width=256,
             camera_episode_id=np.array([7]), camera_grounded_transform=np.array([matrix]))
    matrices, height, width = load_top_camera_transforms(path, {7: np.zeros(3)})
    np.testing.assert_array_equal(matrices[7], matrix)
    assert (height, width) == (256, 256)
    with pytest.raises(ValueError, match="missing episodes"):
        load_top_camera_transforms(path, {8: np.zeros(3)})


def test_per_frame_focus_cache_respects_selected_episode_index_mapping(tmp_path):
    path = tmp_path / "skill_focus_uv.npz"
    matrix = camera_transform_from_recorded_xml(XML, camera_name="agentview", height=256, width=256)
    np.savez(path, image_height=256, image_width=256, camera_episode_id=[7], camera_grounded_transform=[matrix])
    states = torch.zeros(4, 8)
    states[:, 0] = torch.arange(4) * .05
    dataset = SimpleNamespace(
        episodes=[7], reader=SimpleNamespace(_absolute_to_relative_idx={10+i:i for i in range(4)}),
        meta=SimpleNamespace(info={"proprio_grounding":"episode_start_xyz"},
                             episodes={7:{"dataset_from_index":10,"length":4}}),
    )
    uv, valid = cache_top_focus(dataset, path, states)
    expected, expected_valid = project_top_points(states[:, :3].numpy(), matrix, height=256, width=256)
    np.testing.assert_allclose(uv.numpy(), expected)
    np.testing.assert_array_equal(valid.numpy(), expected_valid)
    assert not torch.equal(uv[0], uv[1])


def test_legacy_artifact_reads_recorded_xml_and_restores_grounding(tmp_path):
    import h5py

    run = tmp_path / "dataset_filtered/skillvla_dataset/source/run"
    run.mkdir(parents=True)
    path = run / "skill_focus_uv.npz"
    np.savez(path, image_height=256, image_width=256, camera="agentview")
    np.savez(run.parent / "eval_init_states.npz", episode_index=[7], scene_file=["test_demo.hdf5"], demo=["demo_0"])
    original = tmp_path / "libero_original_dataset/libero_90"
    original.mkdir(parents=True)
    with h5py.File(original / "test_demo.hdf5", "w") as source:
        source.create_group("data/demo_0").attrs["model_file"] = XML
    offset = np.array([.05, .1, .2])
    matrices, height, width = load_top_camera_transforms(path, {7: offset})
    point = np.array([[0., 0., 0.]])
    actual, _ = project_top_points(point, matrices[7], height=height, width=width)
    world = camera_transform_from_recorded_xml(XML, camera_name="agentview", height=256, width=256)
    expected, _ = project_top_points(point + offset, world, height=height, width=width)
    np.testing.assert_allclose(actual, expected)


def test_legacy_grounding_uses_action_aligned_origin_not_original_frame_zero(tmp_path):
    import h5py

    run = tmp_path / "dataset_filtered/skillvla_dataset/source/run"
    run.mkdir(parents=True)
    path = run / "skill_focus_uv.npz"
    np.savez(path, image_height=256, image_width=256, camera="agentview")
    np.savez(run.parent / "eval_init_states.npz", episode_index=[7], scene_file=["demo.hdf5"], demo=["demo_0"])
    original = tmp_path / "libero_original_dataset/libero_90"
    original.mkdir(parents=True)
    with h5py.File(original / "demo.hdf5", "w") as source:
        recorded = source.create_group("data/demo_0")
        recorded.attrs["model_file"] = XML
        recorded.create_dataset("actions", data=np.array([[0.] * 7, [.5] * 7]))
        recorded.create_dataset("obs/ee_states", data=np.array([[10., 10., 10.], [.1, .1, .1]]))
    matrices, h, w = load_top_camera_transforms(path, {7: np.zeros(3)}, first_actions={7: np.full(7, .5)})
    actual, _ = project_top_points(np.zeros((1, 3)), matrices[7], height=h, width=w)
    world = camera_transform_from_recorded_xml(XML, camera_name="agentview", height=h, width=w)
    expected, _ = project_top_points(np.array([[.1, .1, .1]]), world, height=h, width=w)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("frame,remaining,expected_index,valid", [
    (0, 19, 4, True), (1, 18, 5, True), (6, 1, 8, True),
    (18, 5, 19, False), (19, 0, 19, False),
])
def test_chunk_endpoint_and_patch_move_each_timestep_and_respect_boundaries(frame, remaining, expected_index, valid):
    cache = torch.arange(20 * 8).reshape(20, 8).float()
    uv = torch.arange(40).reshape(20, 2).float()
    dataset = SimpleNamespace(
        _chunk_end_state_cache=cache, _chunk_end_state_horizon=4,
        _chunk_end_state_respects_skill_end=True, _include_chunk_end_patch_target=True,
        _chunk_end_focus_cache=uv, _chunk_end_focus_valid_cache=torch.ones(20, dtype=torch.bool),
    )
    item = {"skill_effective_de": torch.tensor(remaining)}
    SkillVLADataset._add_chunk_end_targets(dataset, item, item_index=frame, frame_index=frame, episode_length=20)
    torch.testing.assert_close(item["skill_chunk_end_state"], cache[expected_index])
    torch.testing.assert_close(item["skill_chunk_end_focus_uv"], uv[expected_index])
    assert item["skill_chunk_end_state_valid"].item() == valid
    assert item["skill_chunk_end_focus_valid"].item() == valid


def test_chunk_endpoint_uses_unpadded_actions_and_does_not_cross_episode():
    dataset = SimpleNamespace(
        _chunk_end_state_cache=torch.arange(20 * 8).reshape(20, 8).float(),
        _chunk_end_state_horizon=4, _chunk_end_state_respects_skill_end=False,
        _include_chunk_end_patch_target=False,
    )
    item = {"action_is_pad": torch.tensor([False, False, True, True])}
    SkillVLADataset._add_chunk_end_targets(dataset, item, item_index=0, frame_index=0, episode_length=20)
    torch.testing.assert_close(item["skill_chunk_end_state"], dataset._chunk_end_state_cache[2])
    item["action_is_pad"] = torch.tensor([False, True, False, True])
    with pytest.raises(ValueError, match="valid prefix"):
        SkillVLADataset._add_chunk_end_targets(dataset, item, item_index=0, frame_index=0, episode_length=20)
