import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SRC = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_skillVLA/build_data/eval/src/visualize_skills.py"
spec = importlib.util.spec_from_file_location("boundary_visualization", SRC)
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


def sample():
    prefix = "videos/observation.images.image"
    metadata = pd.DataFrame([{
        "episode_index": 42, "length": 8, "tasks": ["pick <cup>"],
        f"{prefix}/chunk_index": 2, f"{prefix}/file_index": 3,
        f"{prefix}/from_timestamp": 5.0,
    }]).set_index("episode_index")
    latents = {"episode_id": np.array([42, 42]), "skill_index": np.array([1, 0]),
               "frame_start": np.array([4, 0]), "frame_end": np.array([8, 4]),
               "tokens": np.array([7, 3])}
    info = {"fps": 20, "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}
    return info, metadata, latents


def test_video_offsets_and_training_endpoint_convention(tmp_path):
    info, meta, latents = sample()
    episodes, requests = viewer.image_requests(tmp_path, info, meta, latents, [42])
    video = tmp_path / "videos/observation.images.image/chunk-002/file-003.mp4"
    assert requests == {video: {100, 104, 107}}
    first, last = episodes[0]["skills"]
    assert first["end_image_frame"] == last["frame_start"] == 4
    assert last["frame_end"] == 8 and last["end_image_frame"] == 7
    images = {(str(video), i): "data:image/jpeg;base64,AA==" for i in requests[video]}
    page = viewer.render(tmp_path, episodes, images)
    assert "pick &lt;cup&gt;" in page and "pick <cup>" not in page
    assert page.count('<figure>') == 4


def test_invalid_boundary_is_not_silently_clamped(tmp_path):
    info, meta, latents = sample()
    latents["frame_end"][0] = 9
    with pytest.raises(ValueError, match="invalid skill boundary"):
        viewer.image_requests(tmp_path, info, meta, latents, [42])
