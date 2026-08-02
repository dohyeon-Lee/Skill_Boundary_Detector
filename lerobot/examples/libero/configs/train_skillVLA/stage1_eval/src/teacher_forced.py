#!/usr/bin/env python3
"""Offline teacher-forced eval for SkillVLA Stage-1 (skill_expert) — no simulator.

Feeds GT dataset frames (obs + state + GT skill code) straight to the policy and measures:
  1) chunk accuracy : MSE / MAE of the predicted action chunk vs the GT actions
  2) skill sensitivity: how much the predicted chunk changes when ONLY the skill code is
     swapped (same obs, same flow-matching noise) + whether the TRUE code beats swapped
     codes at predicting the GT chunk (win rate ~0.5 ⇒ the model ignores the skill input).

Distinguishes "training problem (skill/obs ignored)" from "sim-eval pipeline problem":
good chunk accuracy here + drawer-only behavior in sim ⇒ the sim eval is feeding bad inputs.

Per-model metric computer (one model+checkpoint per call): --policy_path, --dataset_dir,
--output_dir. Driven by TF_compare/ (multi-model comparison → outputs/summary.md).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("teacher_forced")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy_path", required=True, help="Stage-1 checkpoint pretrained_model dir")
    p.add_argument("--dataset_dir", required=True, help="skillvla dataset dir (the one the model trained on)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_frames", type=int, default=512, help="random frames to evaluate")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_swap", type=int, default=4, help="swapped skill codes per frame")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def masked_err(pred: torch.Tensor, target: torch.Tensor, pad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B,K,A) prediction/target + (B,K) pad → per-sample MSE (B,) and MAE (B,) over valid steps."""
    valid = (~pad).float().unsqueeze(-1)
    diff = (pred - target) * valid
    denom = valid.sum(dim=(1, 2)).clamp(min=1)
    return (diff**2).sum(dim=(1, 2)) / denom, diff.abs().sum(dim=(1, 2)) / denom


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    info = json.loads((Path(args.dataset_dir) / "meta" / "info.json").read_text())
    fps = float(info["fps"])

    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path
    chunk = int(policy_cfg.chunk_size)
    vocab = int(policy_cfg.skill_vocab_size)

    ds = LeRobotDataset(
        repo_id="local/skillvla_teacher_forced",
        root=args.dataset_dir,
        delta_timestamps={"action": [i / fps for i in range(chunk)]},
    )
    log.info("Dataset: %d frames, %d episodes (fps=%s, chunk=%d)", len(ds), ds.meta.total_episodes, fps, chunk)

    policy = make_policy(cfg=policy_cfg, ds_meta=ds.meta)
    policy.eval().to(device)
    pre, post = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    log.info("Policy loaded: vocab=%d (%s)", vocab, args.policy_path)

    frame_ids = rng.choice(len(ds), size=min(args.n_frames, len(ds)), replace=False)
    loader = DataLoader(Subset(ds, frame_ids.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=4)

    max_adim = int(policy_cfg.max_action_dim)
    per_frame: dict[str, list] = defaultdict(list)

    for bi, batch in enumerate(loader):
        gt_raw = batch["action"].clone()                      # (B,K,A) raw dataset actions
        pad = batch.get("action_is_pad", torch.zeros(gt_raw.shape[:2], dtype=torch.bool)).bool()
        batch = pre(batch)
        bsize = gt_raw.shape[0]
        true_code = policy._skill_code(batch).clone()         # (B,) GT FSQ code for each frame

        # One fixed flow-matching noise per batch → output differences are due to the skill code only.
        noise = torch.randn(bsize, chunk, max_adim, dtype=torch.float32, device=device)

        pred_true = post(policy.predict_action_chunk(batch, noise=noise)).cpu()
        mse_t, mae_t = masked_err(pred_true, gt_raw, pad)

        # Determinism check (same code + same noise must reproduce the chunk exactly).
        if bi == 0:
            redo = post(policy.predict_action_chunk(batch, noise=noise)).cpu()
            log.info("Determinism check |Δ| = %.3e (should be ~0)", (redo - pred_true).abs().max().item())

        swap_mse, swap_delta = [], []
        for s in range(1, args.n_swap + 1):
            # Swap codes = other frames' GT codes (realistic, in-distribution codes).
            codes = true_code.roll(shifts=s, dims=0)
            same = codes == true_code
            swapped_batch = dict(batch)
            # Current Stage-1 checkpoints prioritize skill_code over the legacy
            # skill_sequence/index pair, so all three must agree for a real swap.
            swapped_batch["skill_code"] = codes
            swapped_batch["skill_sequence"] = codes.view(bsize, 1)
            swapped_batch["skill_index"] = torch.zeros(
                bsize, dtype=torch.long, device=codes.device
            )
            pred_s = post(policy.predict_action_chunk(swapped_batch, noise=noise)).cpu()
            mse_s, _ = masked_err(pred_s, gt_raw, pad)
            delta = (pred_s - pred_true).norm(dim=-1).mean(dim=1)  # (B,) mean per-step L2 vs true-code chunk
            mse_s[same.cpu()], delta[same.cpu()] = float("nan"), float("nan")
            swap_mse.append(mse_s)
            swap_delta.append(delta)
        # Restore GT codes for clarity (next loop iteration rebuilds the batch anyway).

        per_frame["code"].extend(true_code.cpu().tolist())
        per_frame["episode_index"].extend(batch["episode_index"].cpu().tolist())
        per_frame["mse_true"].extend(mse_t.tolist())
        per_frame["mae_true"].extend(mae_t.tolist())
        per_frame["mse_swap"].extend(torch.stack(swap_mse, 1).tolist())   # (B, n_swap)
        per_frame["swap_delta"].extend(torch.stack(swap_delta, 1).tolist())
        per_frame["gt_step_norm"].extend(gt_raw.norm(dim=-1).mean(dim=1).tolist())
        log.info("batch %d/%d done", bi + 1, len(loader))

    mse_true = np.array(per_frame["mse_true"])
    mae_true = np.array(per_frame["mae_true"])
    mse_swap = np.array(per_frame["mse_swap"])                # (N, n_swap), NaN where swap==true
    swap_delta = np.array(per_frame["swap_delta"])
    gt_norm = np.array(per_frame["gt_step_norm"])

    # Win rate: how often the TRUE code predicts the GT chunk better than a swapped code.
    valid = ~np.isnan(mse_swap)
    wins = (mse_true[:, None] < mse_swap)[valid]
    win_rate = float(wins.mean()) if wins.size else float("nan")

    summary = {
        "policy_path": str(args.policy_path),
        "dataset_dir": str(args.dataset_dir),
        "n_frames": int(mse_true.size),
        "n_swap": args.n_swap,
        "chunk_mse_true": float(mse_true.mean()),
        "chunk_mae_true": float(mae_true.mean()),
        "chunk_mse_swapped": float(np.nanmean(mse_swap)),
        "skill_swap_delta": float(np.nanmean(swap_delta)),   # mean per-step L2 change from swapping the code
        "gt_action_step_norm": float(gt_norm.mean()),        # scale reference for swap_delta
        "true_code_win_rate": win_rate,                      # ~0.5 ⇒ skill ignored; →1.0 ⇒ skill used
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "teacher_forced.json", "w") as f:
        json.dump({"summary": summary, "per_frame": {k: v for k, v in per_frame.items()}}, f)

    log.info("==== TEACHER-FORCED SUMMARY ====")
    for k, v in summary.items():
        log.info("  %-22s : %s", k, v)
    log.info("Reading: mse_true << mse_swapped & win_rate >> 0.5 ⇒ model USES the skill (suspect the sim eval). "
             "mse_true ≈ mse_swapped & win_rate ≈ 0.5 ⇒ model IGNORES the skill (training problem). "
             "mse_true large vs gt scale ⇒ the model did not fit the data at all.")
    log.info("Saved → %s", out / "teacher_forced.json")


if __name__ == "__main__":
    main()
