#!/usr/bin/env python3
"""Build the STAGE-3 / FT-replay "transition pack" from a built skillvla dataset.

Why: the VLM only ever runs at skill TRANSITIONS (predict once per segment, then cached), so the
skill-prediction stages (stage 3, FT's SKILL regime, FT replay) only need each segment's start
neighborhood — (start ± pmax) frames of both cameras + the language + the FSQ code. Frame-uniform
sampling over the full videos re-decodes the same pairs thousands of times; this pack materializes
the unique pairs once.

Output: {skillvla_dir}/../transitions.npz  (next to FSQ.pt / ISS / dino.npz — the run-dir artifacts):
  jpeg_3rd,  off_3rd   : one flat uint8 buffer of JPEG bytes + offsets (N*(2p+1)+1) — 3rd-person cam
  jpeg_wrist, off_wrist: same for the wrist cam
  skill_code (N)       : the segment's FSQ code
  task_index (N), tasks (unique strings)  : language (tokenize at load time → packs merge freely)
  episode_id (N), seg_rank (N), frame_start (N), frame_end (N)  : provenance — future terminator-replay
                         extension joins these against dino.npz / ds,de without re-building
  pmax, fps            : window half-size (inherited from the ISS npz) + dataset fps

Images are stored at the dataset's native resolution (decode → JPEG re-encode); the loader returns
float [0,1] CHW exactly like SkillVLADataset, so the policy preprocessing path is unchanged.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.skillVLA.dataset_skillVLA import CAM_3RD, CAM_WRIST, SkillVLADataset


def _jpeg_bytes(img_chw: torch.Tensor, quality: int) -> bytes:
    """float [0,1] CHW → JPEG bytes (PIL; keeps the builder free of torchvision-version quirks)."""
    from PIL import Image  # noqa: PLC0415

    arr = (img_chw.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def build(skillvla_dir: Path, out: Path, quality: int, limit_segments: int = 0) -> None:
    # SkillVLADataset restricts episodes to the ISS-covered set and owns the video reader — reuse it.
    ds = SkillVLADataset(f"local/{skillvla_dir.parent.parent.name}", root=str(skillvla_dir))
    reader = ds._ensure_reader()  # noqa: SLF001
    pmax = ds._pmax  # noqa: SLF001
    fps = ds.fps
    iss = ds._iss  # noqa: SLF001
    win = 2 * pmax + 1

    # Segment table straight from the ISS npz grouping: (episode, rank) → frame_start; codes/lengths
    # come from the parquet's per-frame sequence columns (identical across an episode → read ONE row).
    hf = ds.hf_dataset.with_format(None)          # plain python values; video cols are paths (no decode)
    ep_first_row: dict[int, int] = {}
    for i, ep in enumerate(hf["episode_index"]):
        e = int(ep if np.isscalar(ep) else np.asarray(ep).reshape(-1)[0])
        if e not in ep_first_row:
            ep_first_row[e] = i

    segs: list[tuple[int, int, int, int, int, int]] = []   # (ep, rank, start_f, end_f, code, task_idx)
    for ep, ranks in sorted(iss.by_ep.items()):
        row = hf[ep_first_row[ep]]
        ss = np.asarray(row["skill_sequence"]).reshape(-1)
        ifs = np.asarray(row["skill_initial_frame"]).reshape(-1)
        lens = np.asarray(row["skill_length_sequence"]).reshape(-1)
        task_idx = int(np.asarray(row["task_index"]).reshape(-1)[0])
        for rank, flat in enumerate(ranks):
            start_f = int(iss.frame_start[flat])
            if start_f != int(ifs[rank]):
                raise ValueError(f"ISS/IFS mismatch ep={ep} rank={rank}: {start_f} != {int(ifs[rank])}")
            segs.append((ep, rank, start_f, start_f + int(lens[rank]) - 1, int(ss[rank]), task_idx))

    if limit_segments > 0:
        segs = segs[:limit_segments]                     # debug/smoke builds only
    print(f"[transition-pack] {len(segs)} segments × window {win} (±{pmax}) × 2 cams — building …")
    j3, o3, jw, ow = bytearray(), [0], bytearray(), [0]
    code_a, task_a, ep_a, rank_a, fs_a, fe_a = [], [], [], [], [], []
    for n, (ep, rank, start_f, end_f, code, task_idx) in enumerate(segs):
        ep_len = int(ds.meta.episodes[ep]["length"])
        ts = [int(np.clip(start_f + o, 0, ep_len - 1)) / fps for o in range(-pmax, pmax + 1)]
        imgs = reader._query_videos({CAM_3RD: ts, CAM_WRIST: ts}, ep)  # noqa: SLF001
        for cam, (buf, offs) in ((CAM_3RD, (j3, o3)), (CAM_WRIST, (jw, ow))):
            frames = imgs[cam] if imgs[cam].dim() == 4 else imgs[cam].unsqueeze(0)
            for f in range(win):
                buf.extend(_jpeg_bytes(frames[min(f, frames.shape[0] - 1)], quality))
                offs.append(len(buf))
        code_a.append(code); task_a.append(task_idx); ep_a.append(ep)
        rank_a.append(rank); fs_a.append(start_f); fe_a.append(end_f)
        if (n + 1) % 2000 == 0:
            print(f"  {n + 1}/{len(segs)}  (buffers: {len(j3) / 1e9:.2f} + {len(jw) / 1e9:.2f} GB)")

    tasks = [ds.meta.tasks.iloc[i].name if hasattr(ds.meta.tasks, "iloc") else str(ds.meta.tasks[i])
             for i in range(len(ds.meta.tasks))]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".npz.tmp.npz")
    np.savez_compressed(
        tmp,
        jpeg_3rd=np.frombuffer(bytes(j3), dtype=np.uint8), off_3rd=np.asarray(o3, dtype=np.int64),
        jpeg_wrist=np.frombuffer(bytes(jw), dtype=np.uint8), off_wrist=np.asarray(ow, dtype=np.int64),
        skill_code=np.asarray(code_a, dtype=np.int64), task_index=np.asarray(task_a, dtype=np.int64),
        episode_id=np.asarray(ep_a, dtype=np.int64), seg_rank=np.asarray(rank_a, dtype=np.int64),
        frame_start=np.asarray(fs_a, dtype=np.int64), frame_end=np.asarray(fe_a, dtype=np.int64),
        tasks=np.asarray(tasks, dtype=np.str_), pmax=np.int64(pmax), fps=np.float64(fps),
    )
    tmp.rename(out)
    print(f"[transition-pack] wrote {out}  ({out.stat().st_size / 1e9:.2f} GB, {len(segs)} segments)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skillvla_dir", type=Path, required=True,
                    help=".../skillvla_dataset/{source}/{run_tag}/skillvla (the built dataset root)")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: {run_dir}/transitions.npz (next to FSQ.pt / ISS)")
    ap.add_argument("--jpeg_quality", type=int, default=92)
    ap.add_argument("--limit_segments", type=int, default=0, help="debug: build only the first N segments")
    args = ap.parse_args()
    out = args.out or (args.skillvla_dir.resolve().parent / "transitions.npz")
    build(args.skillvla_dir.resolve(), out, args.jpeg_quality, args.limit_segments)


if __name__ == "__main__":
    main()
