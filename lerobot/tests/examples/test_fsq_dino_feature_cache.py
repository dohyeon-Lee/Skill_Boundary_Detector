from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "lerobot/examples/libero"))

from fsq_dino_feature_cache import (  # noqa: E402
    CACHE_FORMAT_NAME,
    CACHE_FORMAT_VERSION,
    DINOFeatureCache,
    source_fingerprint,
)


def test_dino_feature_cache_reads_bfloat16_tokens_by_timestamp(tmp_path: Path) -> None:
    raw = tmp_path / "dataset"
    video = raw / "videos/observation.images.image/chunk-000/file-000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fake-video-for-fingerprint")
    (raw / "meta").mkdir()
    (raw / "meta/info.json").write_text("{}\n")

    model = tmp_path / "dinov3"
    model.mkdir()
    (model / "config.json").write_text('{"hidden_size": 4}\n')
    (model / "model.safetensors").write_bytes(b"fake-weights")
    fingerprint = source_fingerprint(raw, model, 224)
    cache = tmp_path / "cache" / fingerprint
    features_path = cache / "features/observation.images.image/chunk-000/file-000.npy"
    pts_path = cache / "pts/observation.images.image/chunk-000/file-000.npy"
    features_path.parent.mkdir(parents=True)
    pts_path.parent.mkdir(parents=True)

    expected = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4).to(
        torch.bfloat16
    )
    np.save(features_path, expected.view(torch.uint16).numpy())
    np.save(pts_path, np.asarray([0.0, 0.1, 0.2], dtype=np.float64))
    relative = video.relative_to(raw).as_posix()
    manifest = {
        "format": CACHE_FORMAT_NAME,
        "format_version": CACHE_FORMAT_VERSION,
        "source_fingerprint": fingerprint,
        "videos": {
            relative: {
                "features": features_path.relative_to(cache).as_posix(),
                "pts": pts_path.relative_to(cache).as_posix(),
                "shape": [3, 2, 4],
                "dtype": "bfloat16_bits_uint16",
            }
        },
    }
    (cache / "manifest.json").write_text(json.dumps(manifest))
    (cache / "_SUCCESS").write_text(fingerprint + "\n")

    reader = DINOFeatureCache(
        cache,
        raw,
        model_path=model,
        image_size=224,
    )
    actual = reader.get_features(video, [0.19, 0.01], tolerance_s=0.05)

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected[[2, 0]], rtol=0, atol=0)
