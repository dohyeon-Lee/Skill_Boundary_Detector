#!/usr/bin/env python3
"""FSQ decoder z-swap test — does the reconstructor actually use the skill latent z?

Mirrors the FSQ training decode path for the motion branch EXACTLY:
the reconstructor sees [z, skill-START state, skill-START 3rd-person image, per-step GT
progress] (start-frame inputs fixed for the whole skill — no per-step images), and the
target is the GT K-step action chunk at every step. We decode each sampled skill with its
TRUE z and with z swapped from random other skills (same start context, same progress):

  mse_true ≈ mse_swap & win_rate ≈ 0.5 → the decoder ignores z → nothing constrains the
  encoder's z layout → the diffuse/inconsistent codes we measured are explained.

Reads only the skill-start DINO token rows straight out of the (uncompressed) 32 GB npz
via the zip member's byte offset, so no full load is needed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from FSQ import SplineFSQAE  # noqa: E402

FSQ_KEYS = {
    "action_dim", "state_dim", "n_control", "spline_degree", "hidden_dim", "fsq_levels",
    "num_layers", "dropout", "max_length", "action_min", "action_max", "delta_min", "delta_max",
    "feat_dim", "n_tokens", "image_encoder_layers", "image_encoder_heads", "terminator_use_wrist",
    "image_model_name", "image_size", "patch_grid", "n_patch_raw", "image_token_dim", "chunk_size",
    "reconstructor_mode",
}


def npz_member_memmap(npz_path: Path, member: str) -> np.memmap:
    """Memory-map one (uncompressed) member of an npz without reading the archive."""
    zf = zipfile.ZipFile(npz_path)
    zinfo = zf.getinfo(member)
    if zinfo.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"{member} is compressed; cannot memory-map.")
    with open(npz_path, "rb") as f:
        f.seek(zinfo.header_offset)
        lh = struct.unpack("<IHHHHHIIIHH", f.read(30))
        data_start = zinfo.header_offset + 30 + lh[9] + lh[10]  # + name len + extra len
        f.seek(data_start)
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)  # noqa: SLF001
        arr_offset = f.tell()
    if fortran:
        raise ValueError("Fortran-ordered member not supported.")
    return np.memmap(npz_path, dtype=dtype, mode="r", offset=arr_offset, shape=shape)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", required=True,
                    help="skillvla run dir (e.g. dataset/skillvla_dataset/libero_90_full_full/FSQ865_dino8_700)")
    ap.add_argument("--skills_dir", required=True,
                    help="FSQ_inputs skillset skills dir (per-skill npz with actions/states)")
    ap.add_argument("--dino_npz", required=True, help="FSQ_inputs dino_tokens npz (3rd-person)")
    ap.add_argument("--n_segments", type=int, default=400)
    ap.add_argument("--n_swap", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    run_dir = Path(args.run_dir)

    # ── FSQ model (CPU; reconstruction branch only, no raw-image model needed) ──
    ckpt = torch.load(run_dir / "FSQ.pt", map_location="cpu", weights_only=False)
    cfg = dataclasses.asdict(ckpt["cfg"])
    fsq = SplineFSQAE(**{k: v for k, v in cfg.items() if k in FSQ_KEYS})
    fsq.load_state_dict(ckpt["model_state"])
    fsq.eval()
    K, A = fsq.chunk_size, fsq.action_dim
    print(f"FSQ loaded: levels={cfg['fsq_levels']} chunk={K} action_dim={A}")

    # ── per-skill GT (actions/states) + alignment metadata ──
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_FSQ import _make_episode_future_targets  # noqa: E402

    files = sorted(Path(args.skills_dir).rglob("*.npz"))
    dec_states, dec_targets, meta = [], [], []
    for f in files:
        d = np.load(str(f))
        dec_states.append(d["states"].astype(np.float32))
        dec_targets.append(d["actions"].astype(np.float32))
        meta.append({"episode_id": int(d["episode_id"]), "skill_index": int(d["skill_index"]),
                     "frame_start": int(d["frame_start"]), "frame_end": int(d["frame_end"]),
                     "length": len(d["actions"])})
    dec_targets = _make_episode_future_targets(dec_targets, meta)
    print(f"skills loaded: {len(meta)}")

    # ── z per skill from skill_latents.npz, aligned by (episode, frame_start) ──
    lat = np.load(run_dir / "skill_latents.npz")
    zmap = {(int(e), int(s)): lat["latents"][i]
            for i, (e, s) in enumerate(zip(lat["episode_id"], lat["frame_start"]))}

    # ── start-frame DINO tokens straight from the npz (no full read) ──
    feats = npz_member_memmap(Path(args.dino_npz), "features.npy")
    small = np.load(args.dino_npz)
    offsets = small["offsets"].astype(np.int64)
    assert len(offsets) == len(meta) + 1, "dino npz / skills dir mismatch"
    for i in (0, len(meta) // 2, len(meta) - 1):  # spot-check alignment
        assert int(small["episode_id"][i]) == meta[i]["episode_id"], f"alignment broken at {i}"

    # ── z-swap decode ──
    idxs = rng.choice(len(meta), size=min(args.n_segments, len(meta)), replace=False)
    res = {"mse_true": [], "mse_swap": [], "out_delta": []}
    skipped = 0
    with torch.no_grad():
        for i in idxs:
            m = meta[i]
            key = (m["episode_id"], m["frame_start"])
            if key not in zmap:
                skipped += 1
                continue
            T = m["length"]
            states = torch.from_numpy(dec_states[i][:T]).unsqueeze(0)                     # (1,T,S)
            start_tok = torch.from_numpy(np.asarray(feats[offsets[i]], dtype=np.float32))
            start_tok = start_tok.unsqueeze(0).unsqueeze(0)                               # (1,1,N,F)
            # GT chunk target at every step t: future actions [t:t+K], padded at episode end
            fut = dec_targets[i]
            gt = np.zeros((T, K, A), dtype=np.float32)
            mask = np.zeros((T, K), dtype=bool)
            for t in range(T):
                n = min(K, len(fut) - t)
                gt[t, :n], mask[t, :n] = fut[t:t + n], True
            gt_t = torch.from_numpy(gt)
            mask_t = torch.from_numpy(mask)[..., None].float()

            # progress = GT per-skill progress (linspace 0→1 over the skill), as in training
            fi = torch.linspace(0.0, 1.0, T).unsqueeze(0)                                 # (1,T)
            prog_tok = fsq.motion_prog_proj(fi.unsqueeze(-1))

            def recon(z_vec: np.ndarray) -> torch.Tensor:
                z = torch.from_numpy(np.asarray(z_vec, dtype=np.float32)).unsqueeze(0)    # (1,D)
                z_tok = fsq.dec_z_proj(z.unsqueeze(1).expand(1, T, -1))                   # (1,T,H)
                return fsq._reconstruct_chunk(z_tok, states, start_tok, prog_tok)[0]      # noqa: SLF001 (T,K,A)

            pred_true = recon(zmap[key])
            mse_true = (((pred_true - gt_t) ** 2) * mask_t).sum() / mask_t.sum() / A
            res["mse_true"].append(float(mse_true))

            for _ in range(args.n_swap):
                j = int(rng.integers(len(meta)))
                kj = (meta[j]["episode_id"], meta[j]["frame_start"])
                if kj not in zmap or kj == key:
                    continue
                pred_s = recon(zmap[kj])
                mse_s = (((pred_s - gt_t) ** 2) * mask_t).sum() / mask_t.sum() / A
                res["mse_swap"].append(float(mse_s))
                res["out_delta"].append(float((pred_s - pred_true).abs().mean()))

    mt, ms = np.array(res["mse_true"]), np.array(res["mse_swap"])
    od = np.array(res["out_delta"])
    n_sw = args.n_swap
    wins = []
    k = 0
    for v in mt:  # pair each segment's true mse with its own swaps
        cnt = min(n_sw, len(ms) - k)
        wins.extend(v < ms[k:k + cnt])
        k += cnt
    gt_scale = float(np.mean([np.abs(t).mean() for t in dec_targets[:500]]))

    print(f"\n==== FSQ DECODER Z-SWAP ({len(mt)} segments, {len(ms)} swaps, {skipped} skipped) ====")
    print(f"mse_true   : {mt.mean():.5f}")
    print(f"mse_swap   : {ms.mean():.5f}")
    print(f"output |Δ| from z swap: {od.mean():.5f}  (GT action scale |a|≈{gt_scale:.3f})")
    print(f"true-z win rate: {np.mean(wins):.3f}  (0.5 = decoder ignores z)")
    out = run_dir / "fsq_zswap.json"
    json.dump({"mse_true": float(mt.mean()), "mse_swap": float(ms.mean()),
               "out_delta": float(od.mean()), "win_rate": float(np.mean(wins)),
               "gt_scale": gt_scale, "n_segments": int(len(mt))}, open(out, "w"))
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
