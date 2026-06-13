#!/usr/bin/env python3
"""Unified SkillVLA-data evaluation, run off the FINAL artifacts only.

Inputs (no skillset / per-episode DINO needed):
  raw video    : {dataset_root}/{source_dataset}      (LeRobot videos + meta)
  skillvla/    : {run_dir}/skillvla                    (skill columns: ds, boundary, sequence, ...)
  dino.npz     : {run_dir}/dino.npz                    (merged frame DINO tokens)
  FSQ.pt       : {run_dir}/FSQ.pt                       (only for fsq_recon)

Evals (toggle via flags), each writing under {out_dir}/{name}/:
  dino       : raw frames vs 8x8 DINO patch PCA-RGB over time
  skillset   : skill-boundary split — start/end frames per skill, laid horizontally
  fsq_patch  : per-skill init/final frames + DINO PCA + random sample patches
  fsq_recon  : FSQ reconstruction/termination/progress — same interactive HTML as
               train_skills/FSQ_eval (reuses fsq_eval.py).

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
    """Frame DINO from the merged dino.npz, indexed by (episode_id, frame)."""

    def __init__(self, npz_path: Path):
        raw = np.load(str(npz_path), mmap_mode="r", allow_pickle=False)
        self.features = raw["features"]                       # (N_total, n_tokens, F) mmap
        self.offsets = raw["offsets"].astype(np.int64)
        self.episode_id = raw["episode_id"].astype(np.int64)
        self.frame_start = raw["frame_start"].astype(np.int64)
        self.length = raw["length"].astype(np.int64)
        self.n_tokens = int(np.asarray(raw["n_tokens"]).reshape(-1)[0])
        self.feat_dim = int(np.asarray(raw["feat_dim"]).reshape(-1)[0])
        self._ep_to_i = {int(e): i for i, e in enumerate(self.episode_id)}

    def episode_clip(self, ep_id: int) -> np.ndarray:
        """(T_ep, n_tokens, feat_dim) for one episode (float32)."""
        i = self._ep_to_i[int(ep_id)]
        return np.asarray(self.features[int(self.offsets[i]):int(self.offsets[i + 1])]).astype(np.float32)


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


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", dpi=110, bbox_inches="tight", pil_kwargs={"quality": 88})
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _save_gallery(out_dir: Path, title: str, cards: list[tuple]) -> None:
    """Write an HTML gallery to out_dir/index.html.

    cards: [(task_label, caption, jpeg_b64), ...]. When task_label is not None, a
    sticky section header is inserted whenever it changes, so episodes are grouped
    and visually separated per task. (task_label None → flat gallery as before.)"""
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks, prev = [], "\0"
    for task_label, cap, b in cards:
        if task_label is not None and task_label != prev:
            blocks.append(f"<h2 class='task'>task {int(task_label):02d}</h2>")
        prev = task_label
        blocks.append(
            f"<div class='card'><div class='cap'>{cap}</div>"
            f"<img src='data:image/jpeg;base64,{b}'></div>"
        )
    body = "".join(blocks)
    html = (
        "<!DOCTYPE html><html><head><meta charset='UTF-8'><title>" + title + "</title><style>"
        "body{font-family:sans-serif;background:#f5f5f5;margin:0;padding:12px;}"
        "h1{font-size:18px;}"
        "h2.task{font-size:15px;color:#fff;background:#3367d6;margin:18px 0 4px;"
        "padding:5px 10px;border-radius:4px;position:sticky;top:0;z-index:2;}"
        ".grid{display:flex;flex-direction:column;gap:14px;}"
        ".card{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px;overflow-x:auto;}"
        ".cap{font-size:12px;color:#555;margin-bottom:4px;}"
        ".card img{display:block;max-width:none;}"
        "</style></head><body><h1>" + title + "</h1><div class='grid'>" + body + "</div></body></html>"
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"[eval] {title} → {out_dir / 'index.html'}")


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
                  task_ids=None, thumb: int = 110):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cards = []
    for task_label, eps in select_episodes(df, task_ids, n_episodes):
        for ep in eps:
            ep_df = df[df["episode_index"] == ep]
            skills = reconstruct_skills(ep_df)
            raw = frames_src(int(ep))
            ncol = max(1, 2 * len(skills))
            fig, axes = plt.subplots(1, ncol, figsize=(1.3 * ncol, 1.7), squeeze=False)
            for k, (fs, fe, tok) in enumerate(skills):
                s = _clip_frame_or_blank(raw, fs, thumb) if raw is not None and len(raw) else np.full((thumb,)*2+(3,), 80, np.uint8)
                e = _clip_frame_or_blank(raw, max(0, fe - 1), thumb) if raw is not None and len(raw) else np.full((thumb,)*2+(3,), 80, np.uint8)
                axes[0][2*k].imshow(s);   axes[0][2*k].axis("off");   axes[0][2*k].set_title(f"sk{k} tok{tok}\nf{fs}", fontsize=6)
                axes[0][2*k+1].imshow(e); axes[0][2*k+1].axis("off"); axes[0][2*k+1].set_title(f"→f{fe}", fontsize=6)
            fig.suptitle(f"{_cap(task_label, ep)} — {len(skills)} skills", fontsize=9)
            cards.append((task_label, _cap(task_label, ep), _fig_to_b64(fig)))
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

def eval_fsq_recon(df, dino: DinoNpz, frames_src, fsq_model_path: Path, out_dir: Path,
                   image_key: str, dataset_dir: Path, *, eef_dims, gripper_action_dim,
                   n_action_steps, n_samples, max_entries, end_threshold, thumb, seed, batch_size, device):
    import torch  # noqa: F401
    import fsq_eval as FE
    from decoder_eval import load_model
    from train_FSQ import _make_episode_future_targets

    model, cfg = load_model(str(fsq_model_path), device)
    levels = list(cfg.fsq_levels)
    codebook_size = int(model.fsq.codebook_size)

    # reconstruct per-skill inputs from skillvla columns + dino.npz
    # NOTE: this diagnostic only has the 3rd-person dino.npz, so the same clip stands in for the
    # wrist camera. Action reconstruction (3rd-person only) is exact; the terminator's
    # progress/termination curves are approximate (real wrist tokens not loaded here).
    segments, dec_states, dec_targets_cur, dec_tokens, metadata = [], [], [], [], []
    for ep in sorted(df["episode_index"].unique()):
        ep_df = df[df["episode_index"] == ep].sort_values("frame_index").reset_index(drop=True)
        skills = reconstruct_skills(ep_df)
        states = np.stack(ep_df["observation.state"].to_numpy()).astype(np.float32)
        actions = np.stack(ep_df["action"].to_numpy()).astype(np.float32)
        dstate = np.stack(ep_df["skill_decoder_state"].to_numpy()).astype(np.float32)
        clip = dino.episode_clip(int(ep))
        gi = (actions.shape[-1] + gripper_action_dim) % actions.shape[-1]
        for fs, fe, _tok in skills:
            pose = states[fs:fe, eef_dims].astype(np.float32)
            grip = actions[fs:fe, gi:gi + 1].astype(np.float32)
            segments.append(np.concatenate([pose, grip], axis=-1))     # encoder traj (abs)
            dec_states.append(dstate[fs:fe])
            dec_targets_cur.append(actions[fs:fe].astype(np.float32))
            dec_tokens.append(clip[fs:fe])
            metadata.append({"episode_id": int(ep), "task_id": -1, "skill_index": len(metadata),
                             "frame_start": int(fs), "frame_end": int(fe), "length": int(fe - fs)})
    dec_targets = _make_episode_future_targets(dec_targets_cur, metadata)
    lengths = [m["length"] for m in metadata]

    latents, tokens = FE.batched_encode(model, segments, lengths, device, batch_size)  # action-only encoder
    # dec_tokens stands in for the wrist camera here (see NOTE above).
    deltas, progresses, term_probs = FE.batched_decode(
        model, latents, dec_states, dec_tokens, dec_tokens, lengths, device, batch_size)

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
    p.add_argument("--dino_npz", required=True)
    p.add_argument("--fsq_model", required=True)
    p.add_argument("--dataset_dir", required=True, help="raw LeRobot dataset (videos + meta)")
    p.add_argument("--image_key", default="observation.images.image")
    p.add_argument("--out_dir", required=True, help="{run_dir}/eval")
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
    p.add_argument("--eef_dims", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--gripper_action_dim", type=int, default=-1)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    df = load_skillvla_episodes(Path(args.skillvla_dir))
    frames_src = make_frames_loader(Path(args.dataset_dir), args.image_key)
    dino = None
    if args.run_dino or args.run_fsq_patch or args.run_fsq_recon:
        dino = DinoNpz(Path(args.dino_npz))

    if args.run_dino:
        eval_dino(df, dino, frames_src, out / "dino", args.n_episodes, task_ids=args.task_ids)
    if args.run_skillset:
        eval_skillset(df, frames_src, out / "skillset", args.n_episodes, task_ids=args.task_ids)
    if args.run_fsq_patch:
        eval_fsq_patch(df, dino, frames_src, out / "fsq_patch", args.n_episodes,
                       seed=args.seed, task_ids=args.task_ids)
    if args.run_fsq_recon:
        import torch
        device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
        eval_fsq_recon(df, dino, frames_src, Path(args.fsq_model), out / "fsq_recon",
                       args.image_key, Path(args.dataset_dir), eef_dims=args.eef_dims,
                       gripper_action_dim=args.gripper_action_dim, n_action_steps=args.n_action_steps,
                       n_samples=args.n_samples, max_entries=args.max_entries,
                       end_threshold=args.end_threshold, thumb=args.thumb_size, seed=args.seed,
                       batch_size=args.batch_size, device=device)
    print(f"[eval] done → {out}")


if __name__ == "__main__":
    main()
