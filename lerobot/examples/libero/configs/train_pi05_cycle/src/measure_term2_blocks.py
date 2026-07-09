#!/usr/bin/env python
"""Per-block term-2 (inter-group gradient alignment) map for a cycle-PT checkpoint.

term-2 = Σ_{i≠j} ⟨g_i, g_j⟩ = ‖Σ_g g_g‖² − Σ_g ‖g_g‖²  (identity), computed per parameter and
reduced per block. Sign: >0 tasks AGREE here (shared structure), <0 tasks CONFLICT (task-specific).

g_g = gradient of group g's fixed probe batch (fixed flow-matching noise, fork_rng) → the SAME
ruler used everywhere in this project. Streams one group at a time, accumulating only
S1 = Σ_g g_g (sum) and S2 = Σ_g g_g² (sum-of-squares), so it never holds all G gradients at once.

Reported per block: raw term2, and a normalized alignment ratio
    align = ‖S1‖² / (G · Σ‖g‖²) ∈ [1/G, 1]     (1 = perfect agreement, 1/G = full cancellation)
and the fraction of params whose per-param term2 is positive.

Blocks: vision_tower / language_model / gemma_expert(action) / projectors / flow_head.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors

RENAME_MAP = {"observation.images.image2": "observation.images.wrist_image"}


def block_of(name: str) -> str:
    if "vision_tower" in name:
        return "vision_tower"
    if "gemma_expert" in name:
        return "action_expert"
    if "language_model" in name or "paligemma.lm_head" in name:
        return "language_model"
    if "multi_modal_projector" in name or "action_in_proj" in name or "action_out_proj" in name:
        return "projectors"
    if "time_mlp" in name:
        return "flow_head"
    return "other"


def build_probe_batches(dataset, groups, batch_size, n_batches, seed):
    rng = np.random.default_rng(seed)
    ep_from = [int(v) for v in dataset.meta.episodes["dataset_from_index"]]
    ep_to = [int(v) for v in dataset.meta.episodes["dataset_to_index"]]
    ep_task = []
    for t in dataset.meta.episodes["tasks"]:
        if isinstance(t, (list, tuple, np.ndarray)):
            t = t[0]
        ep_task.append(str(t))
    probe = {}
    for g in groups:
        gtasks = set(g["tasks"])
        ep_ids = [i for i, t in enumerate(ep_task) if t in gtasks]
        frames = np.concatenate([np.arange(ep_from[i], ep_to[i]) for i in ep_ids])
        idxs = rng.choice(frames, size=min(n_batches * batch_size, len(frames)), replace=False)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, idxs.tolist()), batch_size=batch_size, num_workers=0, shuffle=False
        )
        probe[g["group_id"]] = list(loader)
    return probe


def group_gradient(policy, preprocessor, batches, accelerator, seed):
    """Mean gradient over a group's probe batches (fixed noise via fork_rng), as a name→cpu-fp32 dict."""
    policy.train()
    policy.zero_grad(set_to_none=True)
    devices = [accelerator.device] if accelerator.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if accelerator.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        for batch in batches:
            b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
            b = preprocessor(b)
            with accelerator.autocast():
                loss, _ = policy.forward(b)
            accelerator.backward(loss / len(batches))
    grads = {n: p.grad.detach().float().cpu() for n, p in policy.named_parameters() if p.grad is not None}
    policy.zero_grad(set_to_none=True)
    return grads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)   # .../pretrained_model
    ap.add_argument("--groups_json", type=Path, required=True)
    ap.add_argument("--dataset_repo_id", default="lerobot/libero_90_full_full")
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_batches", type=int, default=2)
    ap.add_argument("--probe_seed", type=int, default=12345)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    acc = Accelerator()
    groups = json.loads(args.groups_json.read_text())

    cfg = PreTrainedConfig.from_pretrained(str(args.checkpoint))
    cfg.pretrained_path = args.checkpoint
    cfg.device = acc.device.type
    # build the dataset the SAME way training does — with the policy's delta_timestamps
    # (action chunk + obs windows), else the forward gets wrong-length token sequences.
    ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    delta_ts = resolve_delta_timestamps(cfg, ds_meta)
    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root, delta_timestamps=delta_ts)

    policy = make_policy(cfg=cfg, ds_meta=dataset.meta, rename_map=RENAME_MAP)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": acc.device.type},
                                "rename_observations_processor": {"rename_map": RENAME_MAP}},
    )
    policy = acc.prepare(policy)

    probe = build_probe_batches(dataset, groups, args.batch_size, args.n_batches, args.probe_seed)

    # streaming accumulation: S1 = Σ_g g_g, S2 = Σ_g g_g²  (per parameter)
    S1, S2 = {}, {}
    G = len(groups)
    for gi, g in enumerate(groups):
        grads = group_gradient(policy, preprocessor, probe[g["group_id"]], acc, args.probe_seed)
        for n, t in grads.items():
            if n not in S1:
                S1[n] = torch.zeros_like(t)
                S2[n] = torch.zeros_like(t)
            S1[n] += t
            S2[n] += t * t
        print(f"  group {g['group_id']} gradient done ({gi + 1}/{G})", flush=True)

    # per-block reduction
    blk = defaultdict(lambda: {"term2": 0.0, "sum_sq": 0.0, "n_pos": 0, "n_tot": 0, "s1_sq": 0.0})
    per_param = {}
    for n in S1:
        s1sq = (S1[n] ** 2)          # (Σg)²  per param
        s2 = S2[n]                    # Σg²   per param
        t2 = s1sq - s2                # term2 per param
        b = block_of(n)
        blk[b]["term2"] += float(t2.sum())
        blk[b]["sum_sq"] += float(s2.sum())
        blk[b]["s1_sq"] += float(s1sq.sum())
        blk[b]["n_pos"] += int((t2 > 0).sum())
        blk[b]["n_tot"] += t2.numel()
        per_param[n] = {"block": b, "term2": float(t2.sum()), "align": float(s1sq.sum() / (G * s2.sum() + 1e-12))}

    summary = {}
    for b, d in blk.items():
        summary[b] = {
            "term2_raw": round(d["term2"], 3),
            "align_ratio": round(d["s1_sq"] / (G * d["sum_sq"] + 1e-12), 4),   # 1=agree, 1/G=cancel
            "frac_param_positive": round(d["n_pos"] / max(1, d["n_tot"]), 4),
            "n_params": d["n_tot"],
        }
    # overall
    tot = {k: sum(blk[b][k] for b in blk) for k in ["term2", "sum_sq", "s1_sq", "n_pos", "n_tot"]}
    summary["_overall"] = {
        "align_ratio": round(tot["s1_sq"] / (G * tot["sum_sq"] + 1e-12), 4),
        "frac_param_positive": round(tot["n_pos"] / max(1, tot["n_tot"]), 4),
        "n_groups": G, "align_baseline_random": round(1.0 / G, 4),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "term2_blocks.json").write_text(json.dumps(summary, indent=2))
    (args.out_dir / "term2_per_param.json").write_text(json.dumps(per_param, indent=2))

    order = ["vision_tower", "language_model", "action_expert", "projectors", "flow_head", "other"]
    print(f"\n{'block':<16}{'align':>8}{'frac+':>8}{'n_params':>10}   (align: 1=shared, {1/G:.3f}=random)")
    for b in order:
        if b in summary:
            s = summary[b]
            print(f"{b:<16}{s['align_ratio']:>8.3f}{s['frac_param_positive']:>8.3f}{s['n_params']:>10}")
    o = summary["_overall"]
    print(f"{'OVERALL':<16}{o['align_ratio']:>8.3f}{o['frac_param_positive']:>8.3f}   (random baseline align={o['align_baseline_random']})")


if __name__ == "__main__":
    main()
