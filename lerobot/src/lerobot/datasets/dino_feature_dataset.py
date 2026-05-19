from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


def _safe_key(image_key: str) -> str:
    return image_key.replace("/", "_").replace(".", "_")


class DinoFrameFeatureDataset(torch.utils.data.Dataset):
    """Attach precomputed frame-level DINO token features to a LeRobotDataset sample.

    The wrapped dataset still provides state/action windows. This wrapper uses the
    scalar current episode/frame index to gather the same observation history from
    per-episode DINO feature shards.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        *,
        feature_dir: str | Path,
        image_keys: list[str],
        output_key: str,
        observation_delta_indices: list[int],
        cache_size: int = 8,
    ):
        self.dataset = dataset
        self.feature_dir = Path(feature_dir)
        self.image_keys = list(image_keys)
        self.output_key = output_key
        self.observation_delta_indices = list(observation_delta_indices)
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()

        if not self.feature_dir.is_dir():
            raise FileNotFoundError(f"DINO feature dir not found: {self.feature_dir}")

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name: str):
        return getattr(self.dataset, name)

    def _path(self, image_key: str, episode_id: int) -> Path:
        return self.feature_dir / _safe_key(image_key) / f"episode_{episode_id:07d}.npz"

    def _load_features(self, image_key: str, episode_id: int) -> np.ndarray:
        cache_key = (image_key, episode_id)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        path = self._path(image_key, episode_id)
        if not path.is_file():
            raise FileNotFoundError(f"Missing DINO feature shard: {path}")
        features = np.load(str(path))["features"]
        self._cache[cache_key] = features
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return features

    @staticmethod
    def _scalar_int(value) -> int:
        if isinstance(value, torch.Tensor):
            return int(value.reshape(-1)[0].item())
        return int(value)

    def __getitem__(self, idx) -> dict:
        item = self.dataset[idx]
        episode_id = self._scalar_int(item["episode_index"])
        frame_index = self._scalar_int(item["frame_index"])

        camera_tokens = []
        for image_key in self.image_keys:
            features = self._load_features(image_key, episode_id)
            max_idx = features.shape[0] - 1
            indices = [max(0, min(max_idx, frame_index + delta)) for delta in self.observation_delta_indices]
            tokens = torch.from_numpy(features[indices])
            camera_tokens.append(tokens)

        # (S, N, T, F), where S=history, N=cameras, T=CLS+patch tokens.
        item[self.output_key] = torch.stack(camera_tokens, dim=1)
        return item
