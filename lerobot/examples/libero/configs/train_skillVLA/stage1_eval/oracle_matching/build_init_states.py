#!/usr/bin/env python3
"""Build the per-episode LIBERO init-state map for Stage-1 oracle eval.

Stage-1 eval injects each episode's GT skill sequence, so the sim env scene must EXACTLY match the
dataset episode that sequence came from. The lerobot-converted dataset has no scene info and is filtered
(failed demos removed + no-noop trimmed), so it does NOT line up 1:1 with the original LIBERO demos.

We recover the link by CONTENT MATCHING: each filtered episode's action trajectory is a contiguous slice
of exactly one original demo (verified: action error = 0). For each episode we find its (scene file, demo)
and pull that demo's `attrs["init_state"]` (the MuJoCo reset state). The env is later reset to it so the
scene reproduces the episode exactly.

This mapping depends ONLY on the source dataset + the original HDF5s — it is INDEPENDENT of the FSQ/skill
encoding — so it is written once at the skillvla-dataset parent and shared by every FSQ_xx run under it.

  episode_index → { init_state (ragged, scene-dependent dim), scene_file, demo, match_err }

Output: {skillvla_root}/{source}/eval_init_states.npz  (default).
"""

from __future__ import annotations

import argparse
import glob
import re
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ACTION_MATCH_THR = 1e-3   # action error below this = a match (observed exact match ≈ 0)
N_GRIPPER = 2             # (only for reference; matching is on the 7-dim action)


def norm_lang(s: str) -> str:
    return str(s).replace("_", " ").strip().lower()


def parse_hdf5_language(fname: str) -> str | None:
    """Recover language from both scene-prefixed and 10-task-suite HDF5 names."""
    m = re.match(r"^(.*SCENE\d+)_(.*)_demo\.hdf5$", fname)
    if m:
        return norm_lang(m.group(2))
    suffix = "_demo.hdf5"
    return norm_lang(fname[: -len(suffix)]) if fname.endswith(suffix) else None


def load_episode_meta(lerobot_dir: Path) -> pd.DataFrame:
    fs = sorted(glob.glob(str(lerobot_dir / "meta/episodes/**/*.parquet"), recursive=True))
    if not fs:
        raise FileNotFoundError(f"No episodes parquet under {lerobot_dir}/meta/episodes")
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def episode_language(row) -> str:
    t = row["tasks"]
    if isinstance(t, (list, np.ndarray)):
        t = t[0]
    return norm_lang(t)


def preload_demos(orig_dir: Path) -> dict[str, list[tuple[str, str, np.ndarray, np.ndarray]]]:
    """language → [(file, demo_name, actions (T,7) float32, init_state (D,) float32), ...]."""
    by_lang: dict[str, list] = defaultdict(list)
    files = sorted(glob.glob(str(orig_dir / "*.hdf5")))
    for f in files:
        lang = parse_hdf5_language(Path(f).name)
        with h5py.File(f, "r") as h:
            for dn, demo in h["data"].items():
                actions = np.asarray(demo["actions"], dtype=np.float32)
                init = np.asarray(demo.attrs["init_state"], dtype=np.float32)
                by_lang[lang].append((Path(f).name, dn, actions, init))
    return by_lang


def _subseq_residual(ep_actions: np.ndarray, dacts: np.ndarray) -> tuple[int, float]:
    """Greedily align ep_actions as an in-order SUBSEQUENCE of dacts (no-noop trims interior frames,
    so an episode is not always a contiguous slice). Returns (#matched, mean residual over matches)."""
    j, res = 0, []
    for t in range(dacts.shape[0]):
        if j >= ep_actions.shape[0]:
            break
        e = float(np.abs(ep_actions[j] - dacts[t]).mean())
        if e <= ACTION_MATCH_THR:
            res.append(e); j += 1
    return j, (float(np.mean(res)) if res else float("inf"))


def match_episode(ep_actions: np.ndarray, candidates: list) -> tuple[float, str, str, np.ndarray, str]:
    """Identify the original demo this episode came from. The contiguous (offset 0, then slide) error
    picks the demo unambiguously (the 2nd-best demo is far off); when no-noop trimmed interior frames
    the contiguous error stays small-but-nonzero, so we then verify the winner is a full subsequence.
    Returns (err, file, demo, init_state, method) with method in {exact, subseq, fail}."""
    L = ep_actions.shape[0]
    best = None  # (contiguous_err, fname, dn, init, dacts)
    for fname, dn, dacts, init in candidates:
        T = dacts.shape[0]
        if T < L:
            continue
        err = float(np.abs(ep_actions - dacts[:L]).mean())               # offset 0 (no-noop trims the TAIL)
        if err > ACTION_MATCH_THR:                                       # fallback: slide
            err = min(float(np.abs(ep_actions - dacts[k:k + L]).mean()) for k in range(T - L + 1))
        if best is None or err < best[0]:
            best = (err, fname, dn, init, dacts)
    if best is None:
        return (float("inf"), "", "", np.empty(0, np.float32), "fail")
    err, fname, dn, init, dacts = best
    if err <= ACTION_MATCH_THR:
        return (err, fname, dn, init, "exact")
    n, res = _subseq_residual(ep_actions, dacts)                          # interior-trim case
    if n == L and res <= ACTION_MATCH_THR:
        return (res, fname, dn, init, "subseq")
    return (err, fname, dn, init, "fail")


def main() -> None:
    ap = argparse.ArgumentParser()
    root = Path("/data2/dohyeon/SBD")
    ap.add_argument("--lerobot_dataset", type=Path, default=root / "dataset_filtered/libero_90_full_full")
    ap.add_argument("--orig_dataset", type=Path, default=root / "libero_original_dataset/libero_90")
    ap.add_argument("--out", type=Path,
                    default=root / "dataset_filtered/skillvla_dataset/libero_90_full_full/eval_init_states.npz")
    args = ap.parse_args()

    print(f"lerobot : {args.lerobot_dataset}")
    print(f"original: {args.orig_dataset}")
    ep_df = load_episode_meta(args.lerobot_dataset)
    print(f"episodes: {len(ep_df)}")
    by_lang = preload_demos(args.orig_dataset)
    print(f"original demos: {sum(len(v) for v in by_lang.values())} over {len(by_lang)} languages")

    # Group episodes by their data parquet shard so each shard is read once.
    by_shard: dict[tuple[int, int], list] = defaultdict(list)
    for _, row in ep_df.iterrows():
        by_shard[(int(row["data/chunk_index"]), int(row["data/file_index"]))].append(row)

    eps, inits, files, demos, errs, methods, fails = [], [], [], [], [], [], []
    for (ci, fi), rows in sorted(by_shard.items()):
        shard = pd.read_parquet(args.lerobot_dataset / f"data/chunk-{ci:03d}/file-{fi:03d}.parquet")
        for row in rows:
            ep = int(row["episode_index"])
            lang = episode_language(row)
            cands = by_lang.get(lang, [])
            if not cands:
                fails.append((ep, f"no HDF5 for language '{lang[:40]}'")); continue
            acts = np.stack(shard[shard.episode_index == ep]["action"].values).astype(np.float32)
            err, fname, dn, init, method = match_episode(acts, cands)
            if method == "fail":
                fails.append((ep, f"no match (best err={round(err, 5)})")); continue
            eps.append(ep); inits.append(init); files.append(fname); demos.append(dn)
            errs.append(err); methods.append(method)
        print(f"  shard {ci:03d}/{fi:03d}: matched {len(eps)} / {len(ep_df)} so far")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    methods = np.asarray(methods)
    np.savez(
        str(args.out),
        episode_index=np.asarray(eps, dtype=np.int32),
        init_states=np.array(inits, dtype=object),       # ragged (scene-dependent dim) → object array
        scene_file=np.asarray(files), demo=np.asarray(demos),
        match_err=np.asarray(errs, dtype=np.float32), match_method=methods,
    )
    print(f"\nWrote {args.out}")
    print(f"  matched: {len(eps)} / {len(ep_df)}  "
          f"(exact={int((methods == 'exact').sum())}, subseq={int((methods == 'subseq').sum())}, "
          f"max err={max(errs) if errs else float('nan'):.2e})")
    if fails:
        print(f"  FAILED: {len(fails)} episodes — first few: {fails[:5]}")


if __name__ == "__main__":
    main()
