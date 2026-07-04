#!/usr/bin/env python3
"""Rename OLD-scheme Stage-2 run folders to the NEW naming scheme.

NEW scheme (see stage2_train_config.py):
    {STAGE-1 정보}__{conn}_{P}p_{A}[_{B}][_{exp}]
      STAGE-1 정보 = {run_tag}_[{s1_vis_tag}_]{stage1_checkpoint}[_{mode}][_<loss tags>]
      conn = (attend_language, attend_image, vlm_cond, vlm_expert) as t/f (cond_expert excluded)
      {P}p = vlm_dropout_p percent (0.5→50p, 0→0p)
      A/B  = freeze_vlm_vsa / freeze_vsa as t/f in (expert, cond, llm, vlm_vision) order; B omitted at p=0

Everything except ``exp`` is derived from the run's LAST checkpoint config.json (single source of
truth); ``exp`` is recovered from the old folder name (the text after the trailing _frzA... tag).
Dirs already containing "__" are skipped. DRY-RUN by default — pass --apply to actually rename.
⚠ Run ONLY after the training job finished (renaming a live run breaks its checkpoint saving/resume).
"""

from __future__ import annotations

import argparse
import getpass
import json
import re
import subprocess
import time
from pathlib import Path

_REGIME_KEYS = ("expert", "cond", "llm", "vlm_vision")


def _last_ckpt_config(run_dir: Path) -> dict | None:
    cks = sorted((run_dir / "checkpoints").glob("*/pretrained_model/config.json")) if (run_dir / "checkpoints").is_dir() else []
    return json.loads(cks[-1].read_text()) if cks else None


def _stage1_bits(cfg: dict) -> tuple[str, str, str, str, list[str]] | None:
    """stage1_checkpoint_path → (run_tag, s1_vis_tag, step, mode_tag, loss_tags). Mirrors the emitter."""
    p = str(cfg.get("stage1_checkpoint_path") or "")
    m = re.search(r"/skillVLA_stage1/([^/]+)/checkpoints/([^/]+)/", p)
    if not m:
        return None
    s1_run, step = m.group(1), m.group(2)
    rt = re.search(r"(FSQ\d+_dino\d+.*?)(?:_((?:dino|siglip)(?:_(?:freeze|unfreeze))?))?_batch\d+", s1_run)
    if not rt:
        return None
    run_tag, s1_vis = rt.group(1), rt.group(2) or ""
    mode_tag = ("state_skill" if "_state_skill" in s1_run else "state" if "_state" in s1_run else "")
    loss_tags = []
    for pat in (r"_(cum_(?:ep|all))(?:_|$)", r"_(ac_w|ac_x|ac)(?:_|$)", r"_(weighted)(?:_|$)"):
        lm = re.search(pat, s1_run)
        if lm:
            loss_tags.append(lm.group(1))
    return run_tag, s1_vis, step, mode_tag, loss_tags


def _grp(cfg: dict, prefix: str, k: str) -> bool:
    # accept the legacy field name (freeze_*_vlm) for runs saved before the llm rename
    if k == "llm":
        return bool(cfg.get(f"freeze_{prefix}_llm", cfg.get(f"freeze_{prefix}_vlm", False)))
    return bool(cfg.get(f"freeze_{prefix}_{k}", False))


def new_name(run_dir: Path) -> str | None:
    cfg = _last_ckpt_config(run_dir)
    if cfg is None:
        return None
    if "num_reader_tokens" not in cfg:      # pre-redesign (skill_query era) run → not this scheme's target
        return None
    bits = _stage1_bits(cfg)
    if bits is None:
        return None
    run_tag, s1_vis, step, mode_tag, loss_tags = bits
    conn = "".join("t" if bool(cfg.get(k, d)) else "f" for k, d in
                   (("attend_language", False), ("attend_image", True), ("vlm_cond", True), ("vlm_expert", False)))
    p = float(cfg.get("vlm_dropout_p", 0.0))
    a = "".join("t" if _grp(cfg, "vlm_vsa", k) else "f" for k in _REGIME_KEYS)
    b = "".join("t" if _grp(cfg, "vsa", k) else "f" for k in _REGIME_KEYS)
    regime = f"{conn}_{int(round(p * 100))}p_{a}" + (f"_{b}" if p > 0 else "")
    # exp: the old-name text after the trailing _frzA... tag (e.g. "..._frzAe_connect" → "connect")
    exp = ""
    m = re.search(r"_frzA[ecvV]*(?:B[ecvV]*)?(?:_(.+))?$", run_dir.name)
    if m and m.group(1):
        exp = m.group(1)
    parts = [run_tag] + ([s1_vis] if s1_vis else []) + [step] + ([mode_tag] if mode_tag else []) + loss_tags
    return "_".join(parts) + "__" + regime + (f"_{exp}" if exp else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help=".../outputs_filtered/skillVLA_stage2")
    ap.add_argument("--apply", action="store_true", help="actually rename (default: dry-run)")
    ap.add_argument("--force", action="store_true", help="rename even while stage2 jobs are in squeue")
    args = ap.parse_args()

    if args.apply and not args.force:
        try:  # refuse --apply while this user still has stage2 jobs queued/running (renaming a live run breaks it)
            jobs = subprocess.run(["squeue", "-u", getpass.getuser(), "-h", "-o", "%j"],
                                  capture_output=True, text=True, timeout=10).stdout.split()
            if any(j.startswith("stage2") for j in jobs):
                ap.error("stage2 jobs are still in squeue — wait for them to finish (or pass --force).")
        except FileNotFoundError:
            pass  # no slurm on this host → nothing to guard

    for run_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        if "__" in run_dir.name:            # already new-scheme
            continue
        target = new_name(run_dir)
        if target is None:
            print(f"SKIP (no ckpt config / unparsable stage1): {run_dir.name}")
            continue
        # freshly-written checkpoint = the job is probably still running → refuse
        last = run_dir / "checkpoints" / "last"
        if last.exists() and time.time() - last.stat().st_mtime < 15 * 60:
            print(f"SKIP (checkpoint <15min old — still training?): {run_dir.name}")
            continue
        dst = run_dir.parent / target
        if dst.exists():
            print(f"SKIP (target exists): {run_dir.name} → {target}")
            continue
        print(f"{'RENAME' if args.apply else 'would rename'}:\n  {run_dir.name}\n  → {target}")
        if args.apply:
            run_dir.rename(dst)


if __name__ == "__main__":
    main()
