from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


class SkillVLADinoTokenDataset(torch.utils.data.Dataset):
    """Attach FSQ decoder DINO tokens to each LeRobot frame.

    The token file is the FSQ precompute output:
      features: (N_total_skill_frames, 65, feat_dim)
      offsets:  (N_skills + 1,)
      episode_id, frame_start, length: (N_skills,)

    Frames that are not covered by any skill range receive zero tokens. The
    wrapper writes the current-frame token to ``output_key``; SkillVLA uses
    ``skill_decoder_image`` so raw VLM images can remain loaded separately.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        *,
        tokens_path: str | Path,
        output_key: str = "skill_decoder_image",
        cache_path: str | Path | None = None,
        build_cache: bool = True,
    ):
        self.dataset = dataset
        self.tokens_path = Path(tokens_path)
        self.output_key = output_key
        self.cache_path = Path(cache_path) if cache_path else self.tokens_path.with_suffix(".features.npy")

        if not self.tokens_path.is_file():
            raise FileNotFoundError(f"SkillVLA DINO token npz not found: {self.tokens_path}")

        meta = np.load(str(self.tokens_path), mmap_mode="r", allow_pickle=False)
        try:
            required = {"offsets", "episode_id", "frame_start", "length", "n_tokens", "feat_dim"}
            missing = required - set(meta.files)
            if missing:
                raise ValueError(f"{self.tokens_path} is missing keys: {sorted(missing)}")

            self.n_tokens = int(np.asarray(meta["n_tokens"]).reshape(-1)[0])
            self.feat_dim = int(np.asarray(meta["feat_dim"]).reshape(-1)[0])
            offsets = meta["offsets"].astype(np.int64)
            episode_ids = meta["episode_id"].astype(np.int64)
            frame_starts = meta["frame_start"].astype(np.int64)
            lengths = meta["length"].astype(np.int64)
        finally:
            meta.close()

        self._frame_to_row: dict[tuple[int, int], int] = {}
        # Every frame → its own skill's START frame, so the reconstructor branch can
        # fetch the skill-start image/state for any step (start = frame - skill_ds).
        self._frame_to_skill_start: dict[tuple[int, int], int] = {}

        for skill_i, (ep, fs, length) in enumerate(zip(episode_ids, frame_starts, lengths, strict=True)):
            start = int(offsets[skill_i])
            for j in range(int(length)):
                self._frame_to_row[(int(ep), int(fs) + j)] = start + j
                self._frame_to_skill_start[(int(ep), int(fs) + j)] = int(fs)

        self.features = self._open_or_build_feature_cache(build_cache=build_cache)
        if self.features.shape[1:] != (self.n_tokens, self.feat_dim):
            raise ValueError(
                f"Feature shape mismatch: got {self.features.shape[1:]}, "
                f"expected {(self.n_tokens, self.feat_dim)}"
            )
        self._zero = np.zeros((self.n_tokens, self.feat_dim), dtype=self.features.dtype)

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name: str):
        return getattr(self.dataset, name)

    def _open_or_build_feature_cache(self, *, build_cache: bool) -> np.ndarray:
        if self.cache_path.is_file():
            try:
                return np.load(str(self.cache_path), mmap_mode="r")
            except ValueError as e:
                # "mmap length is greater than file size" = 잘린 캐시 — 과거 비원자적 빌드가 도중에
                # 죽었거나, 옛 코드가 지금 쓰는 중인 파일. 지우고 아래에서 원자적으로 재빌드.
                if not build_cache:
                    raise
                print(f"[SkillVLA DINO] truncated cache ({e}) → rebuilding: {self.cache_path}")
                self.cache_path.unlink(missing_ok=True)

        if not build_cache:
            raise FileNotFoundError(
                f"Feature cache not found: {self.cache_path}. "
                "Create it once from the FSQ token npz or set build_cache=True."
            )

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[SkillVLA DINO] Building mmap cache: {self.cache_path}")
        print("[SkillVLA DINO] This reads the large .npz once; later workers use the .npy mmap.")
        # ATOMIC build (per-pid tmp → os.replace): 동시 제출된 잡이 "쓰다 만" 캐시를 mmap하다 죽는
        # 경합 방지. 두 잡이 동시에 빌드하면 중복 작업일 뿐 결과는 안전. (tmp 이름이 .npy로 끝나야
        # np.save가 확장자를 덧붙이지 않음.)
        tmp = self.cache_path.with_name(f".tmp{os.getpid()}.{self.cache_path.name}")
        raw = np.load(str(self.tokens_path), mmap_mode="r", allow_pickle=False)
        try:
            features = raw["features"]
            np.save(str(tmp), features)
            del features
            os.replace(str(tmp), str(self.cache_path))
        finally:
            raw.close()
            tmp.unlink(missing_ok=True)
        return np.load(str(self.cache_path), mmap_mode="r")

    @staticmethod
    def _scalar_int(value) -> int:
        if isinstance(value, torch.Tensor):
            return int(value.reshape(-1)[0].item())
        return int(value)

    def _feat_at(self, episode_id: int, frame: int) -> np.ndarray:
        row = self._frame_to_row.get((episode_id, frame))
        return self._zero if row is None else self.features[row]

    def __getitem__(self, idx) -> dict:
        item = self.dataset[idx]
        episode_id = self._scalar_int(item["episode_index"])
        frame_index = self._scalar_int(item["frame_index"])
        item[self.output_key] = torch.from_numpy(
            np.asarray(self._feat_at(episode_id, frame_index), dtype=np.float32)
        )

        # Skill-START frame image (DINO) + raw state, for the reconstructor branch.
        # The start frame is in the same episode, so its global index is idx - skill_ds.
        start_frame = self._frame_to_skill_start.get((episode_id, frame_index), frame_index)
        item["skill_decoder_start_image"] = torch.from_numpy(
            np.asarray(self._feat_at(episode_id, start_frame), dtype=np.float32)
        )
        start_idx = int(idx) - (frame_index - start_frame)
        start_state = self.dataset.hf_dataset[start_idx]["observation.state"]
        item["skill_decoder_start_state"] = torch.as_tensor(start_state, dtype=torch.float32)
        return item
