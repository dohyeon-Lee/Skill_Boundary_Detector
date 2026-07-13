"""Online (frozen) DINO tokenizer — the live replacement for the DINO PRECOMPUTE pipeline.

`precompute_frame_dino_features.py`가 디스크에 구웠던 skill-decoder 토큰(프레임당 65×384: CLS +
8×8 풀링 패치)을 학습/추론 루프 안에서 그때그때 계산한다. 계약(전처리·풀링·dtype 캐스팅)은
precompute의 ``encode_frames``를 **정확히** 재현한다 — 기존 FSQ.pt / terminator 가중치가 이 토큰
분포 위에서 학습되었으므로, 여기서 한 비트라도 다르면 기존 체크포인트와 호환이 깨진다:

    uint8 HWC (또는 float CHW [0,1]) → /255 → interpolate(224×224, bilinear, align_corners=False)
    → ImageNet mean/std 정규화 → DINOv3 ViT-S/16 forward (CUDA에선 fp16 autocast — precompute와 동일)
    → last_hidden_state에서 CLS(token 0) + 마지막 196(=14×14) 패치 → AdaptiveAvgPool2d(8×8)=64
    → concat [CLS, pooled] = (B, 65, 384) float32

검증: configs/train_skillVLA/build_data/tools/verify_online_dino.py 가 기존 per-episode
precompute 산출물(_work/dino/pg8/{cam}/episode_*.npz)과 같은 프레임에서 수치 대조한다.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class OnlineDino(nn.Module):
    """Frozen DINOv3 → (B, 1 + patch_grid², feat_dim) skill-decoder tokens.

    입력은 두 형태를 받는다(소비자 편의):
      - uint8 (B, H, W, 3)  — raw 비디오 디코드 그대로 (precompute와 동일 경로)
      - float (B, 3, H, W)  — LeRobot 배치 이미지 ([0,1]; 정책 배치에서 바로 투입)
    해상도는 내부에서 224로 bilinear 리사이즈하므로 아무 H,W나 가능 (precompute도 동일하게 리사이즈).
    """

    def __init__(self, model_path: str, image_size: int = 224, patch_grid: int = 8,
                 n_patch_raw: int = 196):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel  # noqa: PLC0415 (heavy import, lazy)

        self.image_size = int(image_size)
        self.patch_grid = int(patch_grid)
        self.n_patch_raw = int(n_patch_raw)
        sqrt_raw = int(n_patch_raw ** 0.5)
        if sqrt_raw * sqrt_raw != n_patch_raw:
            raise ValueError(f"n_patch_raw must be a square number, got {n_patch_raw}")
        self._sqrt_raw = sqrt_raw

        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        self.model.requires_grad_(False)

        # 정규화 상수는 precompute와 동일하게 프로세서에서 (없으면 ImageNet 폴백)
        try:
            proc = AutoImageProcessor.from_pretrained(model_path)
            mean = getattr(proc, "image_mean", [0.485, 0.456, 0.406])
            std = getattr(proc, "image_std", [0.229, 0.224, 0.225])
        except Exception:  # noqa: BLE001
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.register_buffer("_mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer("_std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1),
                             persistent=False)
        self._pool = nn.AdaptiveAvgPool2d((self.patch_grid, self.patch_grid))

    @property
    def feat_dim(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def n_tokens(self) -> int:
        return 1 + self.patch_grid * self.patch_grid

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        """images: uint8 (B,H,W,3) 또는 float (B,3,H,W) [0,1] → (B, n_tokens, feat_dim) float32."""
        if images.ndim != 4:
            raise ValueError(f"expected 4D image batch, got {tuple(images.shape)}")
        if images.dtype == torch.uint8:                      # (B,H,W,3) raw decode
            x = images.permute(0, 3, 1, 2).float() / 255.0
        else:                                                # (B,3,H,W) [0,1] LeRobot 배치
            x = images.float()
            if x.shape[1] != 3 and x.shape[-1] == 3:         # 실수로 HWC float가 와도 수용
                x = x.permute(0, 3, 1, 2)
        dev = self._mean.device
        x = x.to(dev)
        x = F.interpolate(x, size=(self.image_size, self.image_size),
                          mode="bilinear", align_corners=False)
        x = (x - self._mean) / self._std

        # precompute와 동일: CUDA에서 fp16 autocast (CPU는 fp32 그대로)
        device_type = "cuda" if dev.type == "cuda" else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.float16,
                            enabled=(device_type == "cuda")):
            out = self.model(pixel_values=x)

        hidden = out.last_hidden_state.float()               # (B, 1+regs+196, F)
        bsz, _, feat_dim = hidden.shape
        cls_tok = hidden[:, :1, :]
        patch_tok = hidden[:, -self.n_patch_raw:, :]         # 마지막 196 = 14×14 (registers 제외)
        patch_map = patch_tok.reshape(bsz, self._sqrt_raw, self._sqrt_raw, feat_dim).permute(0, 3, 1, 2)
        pooled = self._pool(patch_map).permute(0, 2, 3, 1).reshape(
            bsz, self.patch_grid * self.patch_grid, feat_dim)
        return torch.cat([cls_tok, pooled], dim=1)           # (B, 65, F)


# ── 데이터셋 단위 warm-pass 인코딩 (FSQ 학습의 "in-RAM precompute" — 디스크 산출물 없음) ────────────
#
# FSQ 학습은 스킬 전체 프레임의 토큰을 1000 에폭 동안 반복 소비하므로 per-batch 라이브는 낭비
# (같은 프레임 ×1000 재계산). 대신 학습 시작 시 mp4를 파일 단위로 한 번씩 순차 디코드하며 GPU로
# 인코딩해 RAM에 올린다 — 기존 디스크 precompute와 산출물이 같지만(에피소드당 (T,65,384) fp16)
# 빌드 단계·디스크 파일이 없다. LeRobot v3 규약(여러 에피소드가 한 mp4를 공유, from_timestamp
# 슬라이스)은 precompute_frame_dino_features.py와 동일하게 처리.

def _read_video_frames(path) -> "np.ndarray":
    import numpy as np
    try:
        from torchvision.io import read_video
        frames, _, _ = read_video(str(path), output_format="THWC", pts_unit="sec")
        return frames.numpy().astype(np.uint8)[..., :3]
    except Exception:  # noqa: BLE001 — torchvision 미지원 코덱 등 → pyav 폴백 (precompute와 동일)
        import av
        frames = []
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24")[..., :3])
        if not frames:
            raise ValueError(f"No frames in {path}")
        return np.stack(frames).astype(np.uint8)


def encode_episode_dino(dataset_dir, episode_ids, image_key: str, dino: OnlineDino,
                        batch_size: int = 256, out_dtype: str = "float16",
                        log_prefix: str = "[OnlineDino]") -> dict:
    """에피소드별 프레임 전체를 DINO 토큰으로 → {episode_id: (T, n_tokens, F) np.<out_dtype>}.

    mp4 파일당 1회 디코드(공유 파일은 에피소드들을 from_timestamp로 슬라이스), GPU 배치 인코딩.
    호출측(train_FSQ)이 스킬 frame_start/frame_end로 잘라 쓴다."""
    import json
    import time
    from pathlib import Path

    import numpy as np
    import pandas as pd

    dataset_dir = Path(dataset_dir)
    meta_files = sorted((dataset_dir / "meta" / "episodes").rglob("file-*.parquet"))
    if not meta_files:
        raise FileNotFoundError(f"No episode parquet files under {dataset_dir / 'meta' / 'episodes'}")
    meta = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)
    fps = float(json.loads((dataset_dir / "meta" / "info.json").read_text()).get("fps", 20.0))
    ep_index = meta.set_index("episode_index")

    want = sorted(set(int(e) for e in episode_ids))
    # (chunk, file) mp4 → [(ep_id, frame_start_in_file, length)] — precompute의 file_map 규약 그대로
    file_map: dict[tuple, list] = {}
    for ep_id in want:
        row = ep_index.loc[ep_id]
        ck = int(row[f"videos/{image_key}/chunk_index"])
        fi = int(row[f"videos/{image_key}/file_index"])
        fs = round(float(row[f"videos/{image_key}/from_timestamp"]) * fps)
        file_map.setdefault((ck, fi), []).append((ep_id, fs, int(row["length"])))

    np_dtype = np.dtype(out_dtype)
    out: dict[int, np.ndarray] = {}
    t0, done_frames = time.perf_counter(), 0
    for n_file, ((ck, fi), eps) in enumerate(sorted(file_map.items()), 1):
        path = dataset_dir / "videos" / image_key / f"chunk-{ck:03d}" / f"file-{fi:03d}.mp4"
        frames = _read_video_frames(path)                      # (T_file, H, W, 3) uint8
        for ep_id, fs, length in eps:
            if fs + length > len(frames):
                # precompute와 동일 규약: 멀티-에피소드 mp4 경계의 pts 반올림으로 마지막 프레임이
                # 모자랄 수 있음 → WARN+절단 (소비자 fit_feature_length가 마지막 토큰 반복으로 패딩).
                print(f"{log_prefix} [WARN] {path.name}: ep{ep_id} slice [{fs}:{fs+length}] > "
                      f"{len(frames)} frames — truncating (consumer pads by repeat)", flush=True)
            ep_frames = frames[fs: min(fs + length, len(frames))]
            toks = []
            for s in range(0, length, batch_size):
                x = torch.from_numpy(ep_frames[s: s + batch_size].copy())
                toks.append(dino(x).cpu().numpy().astype(np_dtype))
            out[ep_id] = np.concatenate(toks, axis=0)          # (T, n_tokens, F)
            done_frames += length
        if n_file % 10 == 0 or n_file == len(file_map):
            dt = time.perf_counter() - t0
            print(f"{log_prefix} {image_key}: files {n_file}/{len(file_map)}  "
                  f"episodes {len(out)}/{len(want)}  frames {done_frames}  "
                  f"({done_frames / max(dt, 1e-9):.0f} f/s)", flush=True)
    return out


def fit_feature_length(features, length: int):
    """Trim or pad-by-repeat the temporal axis to `length` (구 precompute_dino_features에서 이관 —
    per-skill DINO 클립 길이를 skill length에 맞춤). numpy 배열 in/out."""
    import numpy as np  # noqa: PLC0415
    if len(features) == length:
        return features
    if len(features) > length:
        return features[:length]
    pad = np.repeat(features[-1:], length - len(features), axis=0)
    return np.concatenate([features, pad], axis=0)
