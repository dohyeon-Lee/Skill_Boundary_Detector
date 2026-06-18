#!/usr/bin/env python3
"""INPUT-influence probe — how much each conditioning input (STATE / SKILL / PROGRESS) actually drives
the predicted action chunk, compared across models (e.g. the Stage-1 state_cond_mode variants vs pi05).

OFFLINE, no simulator. For each model: feed GT dataset frames, fix the flow-matching noise, predict the
action chunk, then RE-predict with ONE input perturbed *before preprocessing* (so each model's own
encoder path runs — pi05 re-tokenizes the prompt, the skill_expert re-encodes its state/skill/progress)
and measure how much the predicted chunk moves:

  state_shuffle / progress_shuffle : swap in another frame's value (in-distribution)   ← primary, fair
  state_gauss@σ / progress_gauss@σ : add σ·std noise (sensitivity curve)
  skill_shuffle                    : swap the GT FSQ code (discrete → shuffle only)    ← skill models only
  image_zero                       : zero the images                                   ← vision-dominance ref

SKILL/PROGRESS exist only for skill models; pi05 (has_skill=False) skips them (shown as — in the report).
Metric = mean per-step L2 change of the predicted chunk vs unperturbed (action units) ÷ the GT action
step norm. All models run on the SAME skillvla dataset (same frames). Reading: input Δ ≈ 0 and << that
model's image_zero ⇒ the input is effectively a dead input for that model.
"""

from __future__ import annotations

import argparse
import gc
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
from lerobot.utils.constants import OBS_STATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("input_probe")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", required=True,
                   help='JSON list of models to compare: [{"label":..,"policy_path":..}, ...]')
    p.add_argument("--dataset_dir", required=True, help="shared skillvla dataset dir (same frames for all)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_frames", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--gauss_sigmas", default="0.5,1.0,2.0")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def chunk_delta(pred: torch.Tensor, ref: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    valid = (~pad).float()
    d = (pred - ref).norm(dim=-1)                                   # (B, K) per-step L2
    return (d * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)   # (B,)


def _clone(batch: dict) -> dict:
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}


def _img_keys(batch: dict) -> list[str]:
    return [k for k in batch if k.startswith("observation.images") and torch.is_tensor(batch[k])]


@torch.no_grad()
def probe_one(label: str, policy_path: str, dataset_dir: str, args, device, frame_ids=None) -> dict:
    """Load one policy, perturb the RAW state (+ references) before preprocessing, and measure the
    action-chunk change. Returns {table, gt_norm, frame_ids} (frame_ids reused across models)."""
    sigmas = [float(s) for s in str(args.gauss_sigmas).split(",") if s.strip()]
    info = json.loads((Path(dataset_dir) / "meta" / "info.json").read_text())
    fps = float(info["fps"])

    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.pretrained_path = policy_path
    chunk, max_adim = int(cfg.chunk_size), int(cfg.max_action_dim)
    ds = LeRobotDataset(repo_id=f"local/{label}_input_probe", root=dataset_dir,
                        delta_timestamps={"action": [i / fps for i in range(chunk)]})
    policy = make_policy(cfg=cfg, ds_meta=ds.meta)
    policy.eval().to(device)
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}})
    has_skill = hasattr(policy, "_skill_code")
    log.info("[%s] type=%s chunk=%d has_skill=%s (%s)", label, getattr(cfg, "type", "?"), chunk, has_skill, policy_path)

    if frame_ids is None:
        frame_ids = np.random.default_rng(args.seed).choice(len(ds), size=min(args.n_frames, len(ds)), replace=False)
    loader = DataLoader(Subset(ds, frame_ids.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=4)

    per_frame: dict[str, list] = defaultdict(list)
    for bi, batch in enumerate(loader):
        gt_raw = batch["action"].clone()
        pad = batch.get("action_is_pad", torch.zeros(gt_raw.shape[:2], dtype=torch.bool)).bool()
        bsize = gt_raw.shape[0]
        noise = torch.randn(bsize, chunk, max_adim, dtype=torch.float32, device=device)

        pred_ref = post(policy.predict_action_chunk(pre(_clone(batch)), noise=noise)).cpu()

        # Perturbations applied to the RAW batch (before pre): each model's own state path then runs.
        raw_state = batch[OBS_STATE]
        std = raw_state.float().std(dim=0, keepdim=True).clamp(min=1e-6)
        perts: dict = {
            "state_shuffle": lambda b: {**_clone(b), OBS_STATE: b[OBS_STATE].roll(shifts=1, dims=0)},
        }
        for sg in sigmas:
            perts[f"state_gauss@{sg:g}"] = (
                lambda b, s=sg: {**_clone(b), OBS_STATE: b[OBS_STATE] + s * std * torch.randn_like(b[OBS_STATE].float())})
        ik = _img_keys(batch)
        if ik:
            def _imgzero(b):
                nb = _clone(b)
                for k in ik:
                    nb[k] = torch.zeros_like(nb[k])
                return nb
            perts["image_zero"] = _imgzero
        if has_skill and "skill_sequence" in batch:
            def _skillshuffle(b):
                nb = _clone(b)
                code = policy._skill_code(pre(_clone(b))).clone()
                nb["skill_sequence"] = code.roll(shifts=1, dims=0).view(bsize, 1)
                nb["skill_index"] = torch.zeros(bsize, dtype=torch.long)
                return nb
            perts["skill_shuffle"] = _skillshuffle    # discrete FSQ code → shuffle only (no gauss)
        if has_skill and "skill_ds" in batch and "skill_de" in batch:
            ds = batch["skill_ds"].float().view(-1)
            de = batch["skill_de"].float().view(-1)
            gt_prog = (ds / (ds + de).clamp(min=1.0)).clamp(0.0, 1.0)   # = policy._skill_progress (GT)
            pstd = gt_prog.std().clamp(min=1e-6)
            perts["progress_shuffle"] = (
                lambda b: {**_clone(b), "skill_progress": gt_prog.roll(shifts=1, dims=0).view(-1, 1)})
            for sg in sigmas:
                perts[f"progress_gauss@{sg:g}"] = (
                    lambda b, s=sg: {**_clone(b), "skill_progress":
                                     (gt_prog + s * pstd * torch.randn_like(gt_prog)).clamp(0.0, 1.0).view(-1, 1)})

        for name, fn in perts.items():
            pred = post(policy.predict_action_chunk(pre(fn(batch)), noise=noise)).cpu()
            per_frame[name].extend(chunk_delta(pred, pred_ref, pad).tolist())
        per_frame["gt_step_norm"].extend(gt_raw.norm(dim=-1).mean(dim=1).tolist())
        log.info("[%s] batch %d/%d", label, bi + 1, len(loader))

    gt_norm = float(np.mean(per_frame.pop("gt_step_norm")))
    table = {n: {"chunk_delta": float(np.mean(v)), "rel_to_gt_norm": float(np.mean(v)) / gt_norm}
             for n, v in per_frame.items()}
    del policy
    gc.collect()
    torch.cuda.empty_cache()
    return {"label": label, "policy_path": policy_path, "gt_action_step_norm": gt_norm,
            "perturbations": table, "frame_ids": frame_ids}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    targets = [(t["label"], t["policy_path"]) for t in json.loads(args.targets) if t.get("policy_path")]
    if not targets:
        raise ValueError("--targets is empty (expect a JSON list of {label, policy_path}).")

    results, frame_ids = {}, None
    for label, path in targets:
        r = probe_one(label, path, args.dataset_dir, args, device, frame_ids=frame_ids)
        frame_ids = r.pop("frame_ids")            # reuse the SAME frames for the next model
        results[label] = r

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "input_probe.json").write_text(json.dumps(results, indent=2))

    # Side-by-side: Δ/gt_norm per perturbation for each model (each line built as one string).
    labels = [l for l, _ in targets]
    names = sorted({n for r in results.values() for n in r["perturbations"]})
    log.info("==== INPUT PROBE COMPARISON (chunk Δ ÷ gt_action_step_norm) ====")
    log.info("  " + f"{'perturbation':<16}" + "".join(f"{l:>12}" for l in labels))
    for n in names:
        cells = "".join(
            f"{results[l]['perturbations'].get(n, {}).get('rel_to_gt_norm', float('nan')):12.3f}" for l in labels)
        log.info("  " + f"{n:<16}" + cells)
    for l in labels:
        log.info("  [%s] gt_action_step_norm = %.4f", l, results[l]["gt_action_step_norm"])
    log.info("Reading: compare *_shuffle (state/skill/progress) ÷gt across models. An input whose Δ ≈ 0 "
             "and << that model's image_zero is effectively a dead input for the model.")
    log.info("Saved → %s", out / "input_probe.json")


if __name__ == "__main__":
    main()
