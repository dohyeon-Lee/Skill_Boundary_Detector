#!/usr/bin/env python
"""Per-block flatness (sharpness) of a cycle-PT checkpoint — forgetting-relevant version.

For each block, add filter-normalized random perturbation to ONLY that block's parameters and
measure how much the OLD-task probe loss rises: ΔL/L. Low = flat/robust (old knowledge survives
being shoved in that block), high = sharp/fragile. This is the "robust to being pushed" property
we think explains cyclic's reduced FT-forgetting — measured directly, per block.

Perturbation: per parameter tensor, δθ = ε · ‖θ‖ · (g/‖g‖), g ~ N(0,I)  (filter normalization, so
ε is a relative magnitude comparable across blocks of wildly different scale). Averaged over
`n_seeds` random draws. Forward-only (no backward). Probe loss uses fixed frames + fixed
flow-matching noise (fork_rng) — the same ruler as everywhere else.

Compare cyc vs iid: the block(s) where cyclic has LOWER ΔL/L = where the curriculum bought
robustness.
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

from measure_term2_blocks import RENAME_MAP, block_of, build_probe_batches


@torch.no_grad()
def probe_loss(policy, preprocessor, probe, acc, seed) -> float:
    """Mean probe loss over ALL groups' batches (old-task fidelity), fixed noise via fork_rng."""
    policy.eval()
    devices = [acc.device] if acc.device.type == "cuda" else []
    vals = []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if acc.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        for batches in probe.values():
            for batch in batches:
                b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
                b = preprocessor(b)
                with acc.autocast():
                    loss, _ = policy.forward(b)
                vals.append(loss.item())
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--groups_json", type=Path, required=True)
    ap.add_argument("--dataset_repo_id", default="lerobot/libero_90_full_full")
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_batches", type=int, default=2)
    ap.add_argument("--probe_seed", type=int, default=12345)
    ap.add_argument("--epsilons", default="0.02,0.05")   # relative perturbation magnitudes
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    acc = Accelerator()
    groups = json.loads(args.groups_json.read_text())
    epsilons = [float(e) for e in args.epsilons.split(",")]

    cfg = PreTrainedConfig.from_pretrained(str(args.checkpoint))
    cfg.pretrained_path = args.checkpoint
    cfg.device = acc.device.type
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

    # params grouped by block
    named = [(n, p) for n, p in policy.named_parameters()]
    block_params = defaultdict(list)
    for n, p in named:
        block_params[block_of(n)].append((n, p))

    base = probe_loss(policy, preprocessor, probe, acc, args.probe_seed)
    print(f"baseline probe loss = {base:.4f}", flush=True)

    gen = torch.Generator(device="cpu")
    results = {}
    for blk, plist in block_params.items():
        results[blk] = {}
        for eps in epsilons:
            rises = []
            for s in range(args.n_seeds):
                gen.manual_seed(1000 * s + int(eps * 1e4))
                saved = []
                with torch.no_grad():  # in-place edits on leaf params that require grad
                    for n, p in plist:
                        saved.append(p.detach().clone())
                        g = torch.randn(p.shape, generator=gen, dtype=torch.float32).to(p.device, p.dtype)
                        gn = g.norm()
                        if gn > 0:
                            p.add_(g * (eps * p.detach().norm() / gn))   # filter-normalized δθ
                L = probe_loss(policy, preprocessor, probe, acc, args.probe_seed)
                with torch.no_grad():
                    for (n, p), orig in zip(plist, saved):
                        p.copy_(orig)                                     # restore
                rises.append((L - base) / max(base, 1e-8))
            results[blk][f"eps{eps}"] = round(float(np.mean(rises)), 4)
            print(f"  {blk:<16} eps={eps}  ΔL/L = {np.mean(rises):+.4f}  (±{np.std(rises):.4f})", flush=True)

    out = {"baseline_loss": round(base, 4), "epsilons": epsilons, "n_seeds": args.n_seeds,
           "per_block_rel_loss_rise": results}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "flatness_blocks.json").write_text(json.dumps(out, indent=2))

    order = ["vision_tower", "language_model", "action_expert", "projectors", "flow_head", "other"]
    print(f"\n{'block':<16}" + "".join(f"{'ΔL/L@' + str(e):>14}" for e in epsilons) + "   (low = flat/robust)")
    for b in order:
        if b in results:
            print(f"{b:<16}" + "".join(f"{results[b][f'eps{e}']:>14.4f}" for e in epsilons))
    print(f"DONE -> {args.out_dir}")


if __name__ == "__main__":
    main()
