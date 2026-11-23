import numpy as np
import pytest

from lerobot.policies.skillVLA.abc_native_skillvla_dataset import _ABCSpatialStore


def test_retained_skill_rank_can_differ_from_source_index(tmp_path):
    path = tmp_path / "spatial.npz"
    np.savez(
        path,
        schema_name="abc_bimanual_skill_spatial/v1",
        image_height=256,
        image_width=256,
        episode_id=[18161],
        frame_offsets=[0, 500],
        skill_episode_id=[18161, 18161],
        skill_index=[1, 3],  # Source skills 0 and 2 were filtered out.
        frame_start=[14, 200],
        skill_end_xyz=np.zeros((2, 2, 3)),
        top_projection=np.zeros((1, 3, 4)),
        wrist_projection=np.zeros((500, 2, 3, 4)),
    )
    store = _ABCSpatialStore(path)
    assert store._flat_skill(18161, 0, 14) == 0
    assert store._flat_skill(18161, 1, 200) == 1
    with pytest.raises(ValueError, match="spatial/annotation mismatch"):
        store._flat_skill(18161, 0, 15)
