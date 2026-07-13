#!/usr/bin/env python3
"""Unified SkillVLA-data evaluation, run off the FINAL artifacts only.

Inputs (no skillset / per-episode DINO needed):
  raw video    : {dataset_root}/{source_dataset}      (LeRobot videos + meta)
  skillvla/    : {run_dir}/skillvla                    (skill columns: ds, boundary, sequence, ...)
  FSQ.pt       : {run_dir}/FSQ.pt                       (only for fsq_recon)

Evals (toggle via flags), each writing under {out_dir}/{name}/:
  dino       : raw frames vs 8x8 DINO patch PCA-RGB over time
  skillset   : skill-boundary split — start/end frames per skill, laid horizontally
  fsq_patch  : per-skill init/final frames + DINO PCA + random sample patches
  fsq_recon  : FSQ reconstruction/termination/progress — same interactive HTML as
               train_skills/skill_eval (reuses fsq_eval.py).

Skills are reconstructed from the skillvla dataset columns:
  skill_ds==0 marks a skill start, skill_boundary==1 marks a skill end,
  skill_sequence[skill_index] gives the FSQ token.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# examples/libero on path (fsq_eval, codebook_visualizer, train_FSQ helpers)
LIBERO_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(LIBERO_DIR))

from codebook_visualizer import (  # noqa: E402
    _clip_frame_or_blank,
    _episode_row,
    _load_episodes_meta,
    _read_episode_clip,
    _resolve_image_key,
    _video_path,
)
from skillset_boundary_viz import (  # noqa: E402
    fig_to_b64 as _fig_to_b64,
    load_boundary_curve,
    render_skillset_card,
    save_gallery as _save_gallery,
)


# ── shared loaders ──────────────────────────────────────────────────────────────

def load_skillvla_episodes(skillvla_dir: Path) -> pd.DataFrame:
    files = sorted((skillvla_dir / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet under {skillvla_dir / 'data'}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)


def reconstruct_skills(ep_df: pd.DataFrame) -> list[tuple[int, int, int]]:
    """Per-episode [(frame_start, frame_end, token), ...] from skillvla columns."""
    fr = ep_df["frame_index"].to_numpy()
    ds = ep_df["skill_ds"].to_numpy()
    bn = ep_df["skill_boundary"].to_numpy()
    si = ep_df["skill_index"].to_numpy()
    seq = np.asarray(ep_df["skill_sequence"].iloc[0]).reshape(-1)
    starts = sorted(fr[ds == 0].tolist())
    ends = sorted(fr[bn == 1].tolist())
    skills = []
    for fs, fe1 in zip(starts, ends):
        s_si = int(si[fr == fs][0])
        skills.append((int(fs), int(fe1) + 1, int(seq[s_si])))
    return skills


class DinoNpz:
    """Frame DINO computed ONLINE from the raw dataset (디스크 dino.npz 제거) — 동일 인터페이스
    (episode_clip). 첫 접근 시 frozen DINO(OnlineDino)를 로드하고, 에피소드별로 mp4를 디코드해
    토큰을 계산·캐시한다 (train_FSQ의 warm-pass와 같은 계약)."""

    def __init__(self, dataset_dir: Path, image_key: str, model_path: str):
        sys.path.insert(0, str(LIBERO_DIR.parent / "src"))  # lerobot (OnlineDino)
        from lerobot.utils.online_dino import OnlineDino, _read_video_frames  # noqa: PLC0415
        import torch  # noqa: PLC0415

        self._dataset_dir = Path(dataset_dir)
        self._image_key = image_key
        self._torch = torch
        self._read_frames = _read_video_frames
        self._dino = OnlineDino(model_path)
        if torch.cuda.is_available():
            self._dino = self._dino.to("cuda")
        self.n_tokens = int(self._dino.n_tokens)
        self.feat_dim = int(self._dino.feat_dim)
        meta_files = sorted((self._dataset_dir / "meta" / "episodes").rglob("file-*.parquet"))
        self._meta = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True).set_index(
            "episode_index")
        self._fps = float(json.loads((self._dataset_dir / "meta" / "info.json").read_text()).get("fps", 20.0))
        self._cache: dict[int, np.ndarray] = {}

    def episode_clip(self, ep_id: int) -> np.ndarray:
        """(T_ep, n_tokens, feat_dim) for one episode (float32) — ONLINE 계산 + 캐시."""
        ep_id = int(ep_id)
        if ep_id in self._cache:
            return self._cache[ep_id]
        row = self._meta.loc[ep_id]
        ck = int(row[f"videos/{self._image_key}/chunk_index"])
        fi = int(row[f"videos/{self._image_key}/file_index"])
        fs = round(float(row[f"videos/{self._image_key}/from_timestamp"]) * self._fps)
        length = int(row["length"])
        path = self._dataset_dir / "videos" / self._image_key / f"chunk-{ck:03d}" / f"file-{fi:03d}.mp4"
        frames = self._read_frames(path)[fs: fs + length]     # (T,H,W,3) uint8
        toks = []
        with self._torch.no_grad():
            for s in range(0, len(frames), 256):
                x = self._torch.from_numpy(frames[s: s + 256].copy())
                toks.append(self._dino(x).cpu().numpy().astype(np.float32))
        clip = np.concatenate(toks, axis=0)
        self._cache[ep_id] = clip
        return clip


def patch_pca_rgb_clip(clip_patches: np.ndarray, grid: int, size: int = 96) -> np.ndarray:
    """(T, P, F) episode patch tokens → (T, size, size, 3) uint8.

    A SINGLE 3-component PCA is fit over the WHOLE episode (all frames stacked) and
    reused for every frame, so patch colours are temporally consistent — i.e. the
    PCA basis is per-EPISODE, not per-frame. (Per-frame PCA would re-fit each frame
    and the same region's colour would flicker over time.)"""
    x = np.asarray(clip_patches, dtype=np.float32)            # (T, P, F)
    T, P, F = x.shape
    flat = x.reshape(T * P, F)
    flat = flat - flat.mean(0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(flat, full_matrices=False)
        comp = flat @ vt[:3].T                                # (T*P, 3) — shared basis
    except np.linalg.LinAlgError:
        comp = flat[:, :3]
    lo, hi = comp.min(0), comp.max(0)                         # shared normalization
    rgb = ((comp - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8).reshape(T, grid, grid, 3)
    from PIL import Image
    out = np.empty((T, size, size, 3), dtype=np.uint8)
    for t in range(T):
        out[t] = np.asarray(Image.fromarray(rgb[t]).resize((size, size), Image.NEAREST))
    return out


def select_episodes(df: pd.DataFrame, task_ids, n_episodes: int) -> list[tuple]:
    """Ordered [(task_label, [ep, ...]), ...].

    task_ids empty/None → first n_episodes episodes overall (task_label=None).
    Otherwise → n_episodes episodes per listed task (so several tasks are covered)."""
    if not task_ids:
        return [(None, sorted(df["episode_index"].unique())[:n_episodes])]
    groups = []
    for t in task_ids:
        sub = df[df["task_index"] == int(t)]
        groups.append((int(t), sorted(sub["episode_index"].unique())[:n_episodes]))
    return groups


def _cap(task_label, ep) -> str:
    return f"episode {ep}" if task_label is None else f"task{int(task_label):02d} · episode {ep}"


# _fig_to_b64 / _save_gallery are imported from skillset_boundary_viz (shared with train_skills/skill_eval).


# ── eval 1: DINO sanity ──────────────────────────────────────────────────────────

def eval_dino(df, dino: DinoNpz, frames_src, out_dir: Path, n_episodes: int,
              task_ids=None, n_cols: int = 8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = int(round(((dino.n_tokens - 1) ** 0.5)))
    cards = []
    for task_label, eps in select_episodes(df, task_ids, n_episodes):
        for ep in eps:
            clip = dino.episode_clip(ep)                          # (T, n_tokens, F)
            raw = frames_src(int(ep))                             # (T_raw, H, W, 3) or None
            T = len(clip)
            pca = patch_pca_rgb_clip(clip[:, 1:], grid)           # (T, 96, 96, 3) — per-episode basis
            idxs = np.linspace(0, T - 1, min(n_cols, T)).astype(int)
            fig, axes = plt.subplots(2, len(idxs), figsize=(1.6 * len(idxs), 3.4), squeeze=False)
            for c, t in enumerate(idxs):
                frame = _clip_frame_or_blank(raw, t, 96) if raw is not None and len(raw) else np.full((96, 96, 3), 80, np.uint8)
                axes[0][c].imshow(frame); axes[0][c].axis("off"); axes[0][c].set_title(f"t{t}", fontsize=7)
                axes[1][c].imshow(pca[t]); axes[1][c].axis("off")
            axes[0][0].set_ylabel("raw", fontsize=8); axes[1][0].set_ylabel("PCA", fontsize=8)
            fig.suptitle(_cap(task_label, ep), fontsize=9)
            cards.append((task_label, _cap(task_label, ep), _fig_to_b64(fig)))
    _save_gallery(out_dir, "DINO patch sanity", cards)


# ── eval 2: skillset split ───────────────────────────────────────────────────────

def eval_skillset(df, frames_src, out_dir: Path, n_episodes: int,
                  task_ids=None, thumb: int = 110, curves_dir=None):
    # Rendering (boxed frames + multimodality curve + gallery) is shared with
    # train_skills/skill_eval via skillset_boundary_viz; here the per-episode skills
    # come from the skillvla dataset parquet (reconstruct_skills).
    cards = []
    for task_label, eps in select_episodes(df, task_ids, n_episodes):
        for ep in eps:
            ep_df = df[df["episode_index"] == ep]
            skills = reconstruct_skills(ep_df)
            raw = frames_src(int(ep))
            curve = load_boundary_curve(curves_dir, ep)
            b64 = render_skillset_card(skills, raw, curve, thumb=thumb)
            cards.append((task_label, f"{_cap(task_label, ep)} — {len(skills)} skills", b64))
    _save_gallery(out_dir, "Skill boundary split", cards)


# ── eval 3: FSQ patch ────────────────────────────────────────────────────────────

def eval_fsq_patch(df, dino: DinoNpz, frames_src, out_dir: Path, n_episodes: int,
                   n_samples: int = 3, seed: int = 42, task_ids=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = int(round(((dino.n_tokens - 1) ** 0.5)))
    rng = np.random.default_rng(seed)
    cards = []
    for task_label, eps in select_episodes(df, task_ids, n_episodes):
        for ep in eps:
            ep_df = df[df["episode_index"] == ep]
            skills = reconstruct_skills(ep_df)
            clip = dino.episode_clip(ep)
            raw = frames_src(int(ep))
            rows = len(skills)
            if rows == 0:
                continue
            pca = patch_pca_rgb_clip(clip[:, 1:], grid)           # per-episode basis (shared across frames)
            ncol = 2 + n_samples
            fig, axes = plt.subplots(rows, ncol, figsize=(1.5 * ncol, 1.5 * rows), squeeze=False)
            for r, (fs, fe, tok) in enumerate(skills):
                fi = _clip_frame_or_blank(raw, fs, 96) if raw is not None and len(raw) else np.full((96,96,3),80,np.uint8)
                ff = _clip_frame_or_blank(raw, max(0, fe-1), 96) if raw is not None and len(raw) else np.full((96,96,3),80,np.uint8)
                axes[r][0].imshow(fi); axes[r][0].axis("off"); axes[r][0].set_ylabel(f"sk{r}\ntok{tok}", fontsize=7)
                axes[r][1].imshow(ff); axes[r][1].axis("off")
                cand = list(range(fs, min(fe, len(clip))))
                pick = rng.choice(cand, size=min(n_samples, len(cand)), replace=False) if cand else []
                for c, t in enumerate(sorted(pick)):
                    axes[r][2 + c].imshow(pca[t]); axes[r][2 + c].axis("off")
                for c in range(len(pick), n_samples):
                    axes[r][2 + c].axis("off")
            axes[0][0].set_title("init", fontsize=7); axes[0][1].set_title("final", fontsize=7)
            fig.suptitle(f"{_cap(task_label, ep)} — patch PCA", fontsize=9)
            cards.append((task_label, _cap(task_label, ep), _fig_to_b64(fig)))
    _save_gallery(out_dir, "FSQ patch visualization", cards)


# ── eval 4: FSQ reconstruction (reuse fsq_eval HTML) ─────────────────────────────

def eval_fsq_recon(df, frames_src, fsq_model_path: Path, out_dir: Path,
                   image_key: str, dataset_dir: Path, *,
                   n_action_steps, n_samples, max_entries, end_threshold, thumb, seed, batch_size, device):
    import torch  # noqa: F401
    import fsq_eval as FE
    from decoder_eval import load_model
    from train_FSQ import attach_episode_offsets

    model, cfg = load_model(str(fsq_model_path), device)
    levels = list(cfg.fsq_levels)
    codebook_size = int(model.fsq.codebook_size)

    # Reconstruct per-skill motion inputs. Camera frames are read live below.
    segments, dec_states, dec_targets, metadata = [], [], [], []
    for ep in sorted(df["episode_index"].unique()):
        ep_df = df[df["episode_index"] == ep].sort_values("frame_index").reset_index(drop=True)
        skills = reconstruct_skills(ep_df)
        states = np.stack(ep_df["observation.state"].to_numpy()).astype(np.float32)
        actions = np.stack(ep_df["action"].to_numpy()).astype(np.float32)
        dstate = np.stack(ep_df["skill_decoder_state"].to_numpy()).astype(np.float32)
        for fs, fe, _tok in skills:
            segments.append(states[fs:fe].astype(np.float32))         # encoder traj = full state (8D)
            dec_states.append(dstate[fs:fe])
            dec_targets.append(actions[fs:fe].astype(np.float32))
            metadata.append({"episode_id": int(ep), "task_id": -1, "skill_index": len(metadata),
                             "frame_start": int(fs), "frame_end": int(fe), "length": int(fe - fs)})
    lengths = [m["length"] for m in metadata]
    attach_episode_offsets(str(dataset_dir), metadata)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    raw_dataset = LeRobotDataset(
        repo_id=f"local/{dataset_dir.name}", root=dataset_dir,
        video_keys_to_load=["observation.images.image", "observation.images.wrist_image"],
    )

    latents, tokens = FE.batched_encode(model, segments, lengths, device, batch_size)  # action-only encoder
    deltas, progresses, term_probs = FE.batched_decode(
        model, latents, dec_states, metadata, raw_dataset, lengths, device, batch_size)

    action_dim = dec_targets[0].shape[-1]
    groups = FE._dim_groups(action_dim)
    dim_labels = [f"d{i}" for i in range(action_dim - 1)] + ["grip"]
    per_skill = [FE.skill_metrics(deltas[i], progresses[i], term_probs[i], dec_targets[i], lengths[i],
                                  end_threshold, groups) for i in range(len(metadata))]

    counts = [0] * codebook_size
    for t in tokens:
        counts[int(t)] += 1
    active = [c for c in counts if c > 0]
    enc_stats = {
        "utilization_pct": 100.0 * len(active) / codebook_size,
        "skills_per_entry_mean": float(np.mean(active)) if active else 0.0,
        "skills_per_entry_std": float(np.std(active)) if active else 0.0,
        "skills_per_entry_max": int(max(counts)) if counts else 0,
    }
    keys = ["chunk_mse", "mse_xyz", "mse_rpy", "mse_grip", "timing_abs", "prog_err"]
    dec_means = {k: float(np.mean([s[k] for s in per_skill])) for k in keys}
    dec_means["early_rate"] = float(np.mean([s["timing"] < 0 for s in per_skill]))
    dec_means["late_rate"] = float(np.mean([s["timing"] > 0 for s in per_skill]))

    by_entry: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(tokens):
        by_entry[int(t)].append(i)
    entry_data = {
        tok: {k: float(np.mean([per_skill[i][k] for i in ids])) for k in
              ["chunk_mse", "mse_xyz", "mse_rpy", "mse_grip", "timing_abs", "timing", "prog_err", "length"]}
        for tok, ids in by_entry.items()
    }

    active_tokens = sorted(by_entry)
    plot_tokens = (set(sorted(active_tokens, key=lambda t: (-counts[t], t))[:max_entries])
                   if max_entries > 0 else set(active_tokens))
    rng = np.random.default_rng(seed)
    sample_by_tok, sample_ids = {}, []
    for tok in plot_tokens:
        pool = by_entry[tok]
        n = min(n_samples, len(pool))
        chosen = [pool[j] for j in rng.choice(len(pool), size=n, replace=False)] if n > 0 else []
        sample_by_tok[tok] = chosen
        sample_ids.extend(chosen)

    frames = FE.load_sample_frames(metadata, sample_ids, dataset_dir, image_key, thumb) if n_samples > 0 else {}
    samples = {}
    for tok in sample_by_tok:
        imgs = []
        for i in sample_by_tok[tok]:
            T = lengths[i]
            blank = np.full((thumb,) * 2 + (3,), 80, np.uint8)
            s_img, e_img = frames.get(i, (blank, blank))
            imgs.append(FE.make_sample_plot(s_img, e_img, deltas[i], progresses[i], term_probs[i],
                                            dec_targets[i], T, dim_labels, n_action_steps, end_threshold))
        samples[tok] = imgs

    summary = (
        f"codebook {len(active)}/{codebook_size} ({enc_stats['utilization_pct']:.1f}%) | "
        f"chunk MSE={dec_means['chunk_mse']:.3e} | term|err|={dec_means['timing_abs']:.2f} "
        f"early={dec_means['early_rate']:.0%} late={dec_means['late_rate']:.0%} | "
        f"progress|err|={dec_means['prog_err']:.3f}"
    )
    title = f"{fsq_model_path.parent.name} ({fsq_model_path.stem})"
    html = FE.build_html(title, summary, levels, counts, entry_data, samples, codebook_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fsq_eval.html").write_text(html, encoding="utf-8")
    print(f"[eval] fsq_recon → {out_dir / 'fsq_eval.html'}  ({summary})")


# ── main ────────────────────────────────────────────────────────────────────────

def make_frames_loader(dataset_dir: Path, image_key: str):
    episodes_meta = _load_episodes_meta(dataset_dir)
    key = _resolve_image_key(episodes_meta, image_key)

    def load(ep_id: int):
        try:
            row = _episode_row(episodes_meta, ep_id)
            from_ts = float(row[f"videos/{key}/from_timestamp"])
            to_ts = float(row[f"videos/{key}/to_timestamp"])
            return _read_episode_clip(_video_path(dataset_dir, episodes_meta, ep_id, key),
                                      from_ts, to_ts, int(row["length"]))
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] frames ep{ep_id}: {exc}")
            return None
    return load


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skillvla_dir", required=True)
    p.add_argument("--dino_model_path", default="models/dinov3-vits16",
                   help="ONLINE DINO 모델 경로 (dino.npz 제거 — raw mp4에서 라이브 인코딩)")
    p.add_argument("--fsq_model", required=True)
    p.add_argument("--dataset_dir", required=True, help="raw LeRobot dataset (videos + meta)")
    p.add_argument("--image_key", default="observation.images.image")
    p.add_argument("--out_dir", required=True, help="{run_dir}/eval")
    p.add_argument("--boundary_curves_dir", default=None,
                   help="{skillset_dir}/curves with per-episode multimodality curves "
                        "(build_skill_dataset --dump_curves). Absent → skillset eval shows frames only.")
    p.add_argument("--run_dino", action="store_true")
    p.add_argument("--run_skillset", action="store_true")
    p.add_argument("--run_fsq_patch", action="store_true")
    p.add_argument("--run_fsq_recon", action="store_true")
    p.add_argument("--n_episodes", type=int, default=12,
                   help="episodes shown per visual eval; per task when --task_ids is given")
    p.add_argument("--task_ids", type=int, nargs="*", default=None,
                   help="restrict dino/skillset/fsq_patch to these tasks (n_episodes each); "
                        "empty = first n_episodes overall")
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--max_entries", type=int, default=0)
    p.add_argument("--n_action_steps", type=int, default=5)
    p.add_argument("--end_threshold", type=float, default=0.5)
    p.add_argument("--thumb_size", type=int, default=160)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    df = load_skillvla_episodes(Path(args.skillvla_dir))
    frames_src = make_frames_loader(Path(args.dataset_dir), args.image_key)
    dino = None
    if args.run_dino or args.run_fsq_patch:
        dino = DinoNpz(Path(args.dataset_dir), args.image_key, args.dino_model_path)

    if args.run_dino:
        eval_dino(df, dino, frames_src, out / "dino", args.n_episodes, task_ids=args.task_ids)
    if args.run_skillset:
        eval_skillset(df, frames_src, out / "skillset", args.n_episodes, task_ids=args.task_ids,
                      curves_dir=args.boundary_curves_dir)
    if args.run_fsq_patch:
        eval_fsq_patch(df, dino, frames_src, out / "fsq_patch", args.n_episodes,
                       seed=args.seed, task_ids=args.task_ids)
    if args.run_fsq_recon:
        import torch
        device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
        eval_fsq_recon(df, frames_src, Path(args.fsq_model), out / "fsq_recon",
                       args.image_key, Path(args.dataset_dir),
                       n_action_steps=args.n_action_steps,
                       n_samples=args.n_samples, max_entries=args.max_entries,
                       end_threshold=args.end_threshold, thumb=args.thumb_size, seed=args.seed,
                       batch_size=args.batch_size, device=device)
    print(f"[eval] done → {out}")


if __name__ == "__main__":
    main()
