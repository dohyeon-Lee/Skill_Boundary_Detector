#!/usr/bin/env python
"""Block-cyclic PT for pi05 — continual-learning curriculum mini-experiments (PT only).

Standard lerobot training samples batches iid over the whole dataset. This script instead
trains in a *block-cyclic task curriculum*:

  - The dataset's tasks are split into `n_groups` fixed groups (seeded, saved to groups.json).
  - Training proceeds in *phases*: `phase_steps` optimizer steps on ONE group's data only,
    then the next group, ... Group order is reshuffled every cycle.
  - (optional) Δ-feedback  — `delta_lambda` > 0: when revisiting group g, its loss is scaled by
        w_g = 1 + delta_lambda * max(0, (probe_g_now − probe_g_own_last) / probe_g_own_last)
    where probe losses are measured on FIXED per-group probe batches with FIXED flow-matching
    noise (fork_rng), so the delta measures parameter change only, not sampling noise.
  - (optional) Reptile — `reptile_beta` < 1: parameters are snapshotted at cycle start (anchor,
    CPU) and after the full cycle interpolated:  θ ← anchor + β·(θ − anchor).
    Cycle-level anchor (not per-phase) so inter-group cross terms live inside one inner loop.

  With delta_lambda=0 and reptile_beta=1 this is pure block-cyclic training; the iid baseline
  is the regular configs/train_pi05 pipeline.

Diagnostics logged to wandb at every phase boundary (probe forward passes only, cheap):
  - probe/g{j}_loss     : fixed-probe loss of every group j  → interference matrix / recovery
                          curves are reconstructable offline from these + cycle/active_group.
  - probe/g{j}_forget   : relative loss regression of j vs. the end of j's own last phase.
  - cycle/*             : cycle index, active group, active Δ-weight.
  - probe/grad_cos_g{X} : (optional, `probe_grad_group` ≥ 0) cosine of group X's probe gradient
                          between consecutive boundaries — Taylor-validity ("rotation") signal.
                          Costs one probe backward per boundary + ~6 GB CPU for the stored grad.

Single-GPU only (mini experiments; asserted). No resume support — runs start fresh.

Example:
  accelerate launch --num_processes=1 lerobot_train_cycle.py \
      --dataset.repo_id=lerobot/libero_90_full_full --dataset.root=... \
      --policy.type=pi05 --policy.pretrained_path=... --output_dir=... \
      --steps=20000 --batch_size=16 \
      --n_groups=8 --phase_steps=500 --delta_lambda=0.5 --reptile_beta=0.5
"""

import json
import logging
import math
import time
from dataclasses import dataclass
from pprint import pformat

import numpy as np
import torch
from accelerate import Accelerator
from termcolor import colored
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import format_big_number, init_logging, inside_slurm


def clone_batch(batch: dict) -> dict:
    """Probe batches are reused for the whole run — hand the pipeline a fresh copy so an
    in-place processor can never corrupt the stored originals."""
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}


@dataclass
class CycleTrainPipelineConfig(TrainPipelineConfig):
    # ── block-cyclic curriculum ─────────────────────────────────────────────
    n_groups: int = 8          # tasks are split into this many fixed groups
    phase_steps: int = 500     # optimizer steps per group phase (the "k" dial)
    n_cycles: int = 0          # >0: OVERRIDES phase_steps = steps // (n_groups × n_cycles)
    group_seed: int = 0        # seeds task→group split AND per-cycle order shuffle
    # ── Δ-feedback (0 disables) ─────────────────────────────────────────────
    delta_lambda: float = 0.0
    delta_max_weight: float = 3.0   # cap on w_g
    # ── Reptile (1.0 disables) ──────────────────────────────────────────────
    reptile_beta: float = 1.0       # θ ← anchor + β(θ − anchor) at cycle end
    # β-schedule: ≥0 → β anneals reptile_beta → reptile_beta_end over cycles (cosine).
    # "commit anneal": sensing stays at full inner-LR scale while the anchor settles —
    # the CL-native alternative to LR decay. -1 disables (constant β).
    reptile_beta_end: float = -1.0
    # ── probe (forgetting measurement) ──────────────────────────────────────
    probe_batches_per_group: int = 2   # fixed batches per group (batch_size each)
    probe_seed: int = 12345            # frame selection + flow-matching noise seed
    # ── optional heavy diagnostic ───────────────────────────────────────────
    probe_grad_group: int = -1  # group id to track probe-gradient rotation for (-1 = off)
    # ── iid baseline mode ───────────────────────────────────────────────────
    iid_baseline: bool = False  # true: PLAIN iid training (global shuffle) with the SAME probe/
                                # logging instrumentation — the fair comparison baseline.
                                # phase_steps then only sets the probe cadence; Δ/Reptile ignored.


# ════════════════════════════════════════════════════════════════════════════
# Group construction
# ════════════════════════════════════════════════════════════════════════════


def build_groups(dataset, n_groups: int, group_seed: int) -> list[dict]:
    """Split tasks into n_groups FRAME-BALANCED groups (greedy bin-packing).

    Tasks stay atomic (a task is never split across groups). Heaviest task first → assigned
    to the currently lightest group, so per-group frame counts come out nearly equal. This
    keeps fixed phase_steps fair data-wise (matches iid's frames-proportional step allocation)
    while keeping the perturbation size k identical across groups. group_seed shuffles
    equal-weight ties so different seeds give different (still balanced) splits.
    """
    eps = dataset.meta.episodes
    ep_from = [int(v) for v in eps["dataset_from_index"]]
    ep_to = [int(v) for v in eps["dataset_to_index"]]
    ep_task = []
    for t in eps["tasks"]:
        if isinstance(t, (list, tuple, np.ndarray)):
            t = t[0]
        ep_task.append(str(t))

    tasks = sorted(set(ep_task))
    task_frames = dict.fromkeys(tasks, 0)
    for i, t in enumerate(ep_task):
        task_frames[t] += ep_to[i] - ep_from[i]

    rng = np.random.default_rng(group_seed)
    order = sorted(rng.permutation(tasks).tolist(), key=lambda t: -task_frames[t])
    loads = [0] * n_groups
    group_of_task = {}
    for t in order:
        g = int(np.argmin(loads))
        group_of_task[t] = g
        loads[g] += task_frames[t]

    groups = []
    for gid in range(n_groups):
        gtasks = {t for t, g in group_of_task.items() if g == gid}
        ep_ids = [i for i, t in enumerate(ep_task) if t in gtasks]
        frames = np.concatenate([np.arange(ep_from[i], ep_to[i]) for i in ep_ids])
        groups.append(
            {
                "group_id": gid,
                "tasks": sorted(gtasks),
                "episode_indices": ep_ids,
                "frame_indices": frames,
            }
        )
    return groups


class GroupCursor:
    """Per-group epoch cursor: each phase consumes the group's shuffled permutation from where
    the last visit stopped, reshuffling only when exhausted — so revisits finish the unfinished
    epoch and every frame is consumed exactly evenly (matches iid's epoch-permutation hygiene)."""

    def __init__(self, frame_indices: np.ndarray, seed):
        self.frames = np.asarray(frame_indices)
        self.rng = np.random.default_rng(seed)
        self.perm = self.rng.permutation(self.frames)
        self.pos = 0
        self.consumed = 0  # total frames handed out — epochs = consumed / len(frames)

    @property
    def epochs(self) -> float:
        return self.consumed / len(self.frames)

    def take(self, n: int) -> np.ndarray:
        self.consumed += n
        out = []
        while n > 0:
            chunk = self.perm[self.pos : self.pos + n]
            out.append(chunk)
            self.pos += len(chunk)
            n -= len(chunk)
            if self.pos >= len(self.perm):
                self.perm = self.rng.permutation(self.frames)
                self.pos = 0
        return np.concatenate(out)


def make_phase_loader(dataset, frame_indices: np.ndarray, cfg):
    """Fresh DataLoader over an EXACT, already-shuffled index list from the group's epoch
    cursor (recreated per phase; worker spin-up is negligible vs. a 100s-of-steps phase)."""
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        sampler=[int(i) for i in frame_indices],  # consumed in cursor order, no re-shuffle
        pin_memory=False,
        drop_last=False,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
    )


# ════════════════════════════════════════════════════════════════════════════
# Probe: fixed batches + fixed noise → deterministic forgetting measurement
# ════════════════════════════════════════════════════════════════════════════


def build_probe_batches(dataset, groups, cfg) -> dict[int, list[dict]]:
    """Pick fixed probe frames per group once; keep raw CPU batches for the whole run."""
    rng = np.random.default_rng(cfg.probe_seed)
    probe = {}
    for g in groups:
        n = cfg.probe_batches_per_group * cfg.batch_size
        idxs = rng.choice(g["frame_indices"], size=min(n, len(g["frame_indices"])), replace=False)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, idxs.tolist()),
            batch_size=cfg.batch_size,
            num_workers=0,
            shuffle=False,
        )
        probe[g["group_id"]] = list(loader)
    return probe


@torch.no_grad()
def measure_probe(policy, preprocessor, probe_batches, accelerator, probe_seed) -> dict[int, float]:
    """Mean probe loss per group. fork_rng + fixed seed pins the flow-matching noise/timestep,
    so two measurements differ only through the parameters ("same ruler")."""
    policy.eval()
    devices = [accelerator.device] if accelerator.device.type == "cuda" else []
    losses = {}
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(probe_seed)
        if accelerator.device.type == "cuda":
            torch.cuda.manual_seed_all(probe_seed)
        for gid, batches in probe_batches.items():
            vals = []
            for batch in batches:
                batch = preprocessor(clone_batch(batch))
                with accelerator.autocast():
                    loss, _ = policy.forward(batch)
                vals.append(loss.item())
            losses[gid] = float(np.mean(vals))
    policy.train()
    return losses


def measure_probe_grad(policy, preprocessor, batches, accelerator, probe_seed) -> list[torch.Tensor]:
    """Probe gradient of ONE group (bf16 CPU copy) for the rotation diagnostic."""
    policy.train()
    policy.zero_grad(set_to_none=True)
    devices = [accelerator.device] if accelerator.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(probe_seed)
        if accelerator.device.type == "cuda":
            torch.cuda.manual_seed_all(probe_seed)
        for batch in batches:
            batch = preprocessor(clone_batch(batch))
            with accelerator.autocast():
                loss, _ = policy.forward(batch)
            accelerator.backward(loss / len(batches))
    grads = [
        (p.grad.detach().to("cpu", dtype=torch.bfloat16) if p.grad is not None else None)
        for p in policy.parameters()
    ]
    policy.zero_grad(set_to_none=True)
    return grads


def grad_cosine(a: list, b: list) -> float:
    dot, na, nb = 0.0, 0.0, 0.0
    for ga, gb in zip(a, b):
        if ga is None or gb is None:
            continue
        ga32, gb32 = ga.float(), gb.float()
        dot += torch.sum(ga32 * gb32).item()
        na += torch.sum(ga32 * ga32).item()
        nb += torch.sum(gb32 * gb32).item()
    return dot / (np.sqrt(na) * np.sqrt(nb) + 1e-12)


# ════════════════════════════════════════════════════════════════════════════
# Reptile: cycle-level anchor + interpolation
# ════════════════════════════════════════════════════════════════════════════


def snapshot_params(policy) -> dict[str, torch.Tensor]:
    return {k: v.detach().to("cpu", copy=True) for k, v in policy.state_dict().items()}


@torch.no_grad()
def reptile_interpolate(policy, anchor: dict[str, torch.Tensor], beta: float, device):
    """θ ← anchor + β·(θ − anchor), streamed per-tensor from the CPU anchor."""
    for k, v in policy.state_dict().items():
        if not torch.is_floating_point(v):
            continue
        a = anchor[k].to(device=device, dtype=v.dtype, non_blocking=True)
        v.mul_(beta).add_(a, alpha=1.0 - beta)


# ════════════════════════════════════════════════════════════════════════════
# One optimizer step (lerobot's update_policy minus RA-BC, plus loss scaling)
# ════════════════════════════════════════════════════════════════════════════


def update_policy_scaled(
    train_metrics, policy, batch, optimizer, grad_clip_norm, accelerator, lr_scheduler, loss_scale: float
):
    start_time = time.perf_counter()
    policy.train()
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)
    accelerator.backward(loss * loss_scale)  # Δ-feedback enters ONLY as a detached scalar gain
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float("inf"), error_if_nonfinite=False)
    optimizer.step()
    optimizer.zero_grad()
    if lr_scheduler is not None:
        lr_scheduler.step()
    train_metrics.loss = loss.item()  # log the UNscaled loss (comparable across conditions)
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════


@parser.wrap()
def train(cfg: CycleTrainPipelineConfig):
    cfg.validate()

    accelerator = Accelerator(step_scheduler_with_optimizer=False)
    if accelerator.num_processes != 1:
        raise RuntimeError("lerobot_train_cycle.py is single-GPU only (mini experiments).")
    init_logging(accelerator=accelerator)
    logging.info(pformat(cfg.to_dict()))

    wandb_logger = WandBLogger(cfg) if cfg.wandb.enable and cfg.wandb.project else None
    if wandb_logger is None:
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    logging.info("Creating policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)

    processor_kwargs = {"dataset_stats": dataset.meta.stats}
    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        tokenizer_path = getattr(policy.config, "tokenizer_path", None)
        if tokenizer_path:
            processor_kwargs["preprocessor_overrides"]["tokenizer_processor"] = {
                "tokenizer_name": tokenizer_path,
            }
        # CRITICAL: without this the checkpoint keeps pi05_base's unnormalizer stats and
        # eval de-normalizes actions with the WRONG scale → garbage motions (found 2026-07-05).
        processor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy, pretrained_path=cfg.policy.pretrained_path, **processor_kwargs
    )

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    policy, optimizer, lr_scheduler = accelerator.prepare(policy, optimizer, lr_scheduler)

    # ── curriculum setup ────────────────────────────────────────────────────
    groups = build_groups(dataset, cfg.n_groups, cfg.group_seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.output_dir / "groups.json", "w") as f:
        json.dump(
            [{k: v for k, v in g.items() if k != "frame_indices"} | {"n_frames": len(g["frame_indices"])}
             for g in groups],
            f, indent=2,
        )
    for g in groups:
        logging.info(f"group {g['group_id']}: {len(g['tasks'])} tasks, "
                     f"{len(g['episode_indices'])} eps, {len(g['frame_indices'])} frames")

    logging.info("Building fixed probe batches")
    probe_batches = build_probe_batches(dataset, groups, cfg)
    cursors = {
        g["group_id"]: GroupCursor(g["frame_indices"], seed=(cfg.group_seed, g["group_id"]))
        for g in groups
    }
    global_cursor = None
    if cfg.iid_baseline:
        all_frames = np.concatenate([g["frame_indices"] for g in groups])
        global_cursor = GroupCursor(all_frames, seed=(cfg.group_seed, 999))
        if cfg.delta_lambda > 0 or cfg.reptile_beta < 1.0:
            logging.warning("iid_baseline=true → delta_lambda / reptile_beta are IGNORED (plain training).")

    if cfg.n_cycles > 0:
        cfg.phase_steps = max(1, cfg.steps // (cfg.n_groups * cfg.n_cycles))
        logging.info(f"n_cycles={cfg.n_cycles} → phase_steps auto-computed = {cfg.phase_steps}")
    steps_per_cycle = cfg.n_groups * cfg.phase_steps
    logging.info(
        f"{cfg.steps=} | {cfg.n_groups=} × {cfg.phase_steps=} → {steps_per_cycle} steps/cycle "
        f"→ ~{cfg.steps / steps_per_cycle:.1f} cycles | delta_lambda={cfg.delta_lambda} "
        f"reptile_beta={cfg.reptile_beta}"
    )
    if cfg.steps / steps_per_cycle < 2:
        logging.warning(
            f"Fewer than 2 cycles ({cfg.steps / steps_per_cycle:.2f}) — groups get (almost) no "
            "revisits, so the whole cyclic mechanism (recovery/Δ/Reptile averaging) cannot act. "
            "Lower phase_steps or set n_cycles."
        )

    total_cycles = max(1, math.ceil(cfg.steps / steps_per_cycle))

    def beta_for(cycle_i: int) -> float:
        """Constant β, or cosine anneal reptile_beta → reptile_beta_end across cycles."""
        if cfg.reptile_beta_end < 0:
            return cfg.reptile_beta
        t = min(1.0, cycle_i / max(1, total_cycles - 1))
        return cfg.reptile_beta_end + (cfg.reptile_beta - cfg.reptile_beta_end) * 0.5 * (1 + math.cos(math.pi * t))

    if cfg.reptile_beta_end >= 0:
        logging.info(f"Reptile β schedule: {cfg.reptile_beta} → {cfg.reptile_beta_end} over {total_cycles} cycles "
                     f"(first cycles: {[round(beta_for(i), 3) for i in range(min(5, total_cycles))]})")
    num_learnable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logging.info(f"num_learnable_params={format_big_number(num_learnable)}")

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics,
        initial_step=0, accelerator=accelerator,
    )
    progbar = tqdm(total=cfg.steps, desc="Training", unit="step", disable=inside_slurm())

    def wandb_log_section(section: str, d: dict, step: int):
        """WandBLogger.log_dict hard-codes its mode whitelist, so log through the raw wandb run —
        probe/cycle/epoch get their own sidebar sections instead of all piling into train/."""
        if wandb_logger:
            wandb_logger._wandb.log({f"{section}/{k}": v for k, v in d.items()}, step=step)

    def log_probe(probe_vals, own_last, step, cycle_idx, active_group, w_active, tag=""):
        # separate sidebar sections: probe_loss/ vs probe_forget/
        forgets = {j: (probe_vals[j] - own_last[j]) / max(own_last[j], 1e-8) for j in probe_vals}
        wandb_log_section("probe_loss", {f"g{j}": v for j, v in probe_vals.items()}, step)
        wandb_log_section("probe_forget", {f"g{j}": v for j, v in forgets.items()}, step)
        # legacy duplicate under probe/ — keeps new runs overlayable with pre-split runs
        wandb_log_section("probe", {f"g{j}_loss": v for j, v in probe_vals.items()}
                          | {f"g{j}_forget": v for j, v in forgets.items()}, step)
        wandb_log_section("cycle", {"index": cycle_idx, "active_group": active_group, "w_active": w_active}, step)
        if cfg.iid_baseline:
            wandb_log_section("epoch", {"global": global_cursor.epochs}, step)
        else:
            wandb_log_section("epoch", {f"g{j}": cursors[j].epochs for j in probe_vals}, step)
        logging.info(
            f"[probe{tag}] step={step} cyc={cycle_idx} g={active_group} w={w_active:.3f} "
            + " ".join(f"g{j}={v:.4f}" for j, v in sorted(probe_vals.items()))
        )

    # initial probe = reference point
    probe_vals = measure_probe(policy, preprocessor, probe_batches, accelerator, cfg.probe_seed)
    own_last = dict(probe_vals)  # probe loss at the end of each group's own last phase
    prev_probe_grad = None
    log_probe(probe_vals, own_last, 0, 0, -1, 1.0, tag="@init")

    step, cycle_idx = 0, 0
    while step < cfg.steps:
        order_rng = np.random.default_rng(cfg.group_seed + 1 + cycle_idx)
        order = order_rng.permutation(cfg.n_groups)  # fresh shuffle each cycle (symmetrizes cross terms)
        steps_in_cycle = 0

        beta = beta_for(cycle_idx)
        anchor = None
        if beta < 1.0 and not cfg.iid_baseline:
            anchor = snapshot_params(accelerator.unwrap_model(policy))

        for gid in order.tolist():
            if step >= cfg.steps:
                break
            g = groups[gid]

            # Δ-feedback weight for this phase (detached scalar; 1.0 when disabled or improved)
            w = 1.0
            if cfg.delta_lambda > 0 and not cfg.iid_baseline:
                rel_forget = (probe_vals[gid] - own_last[gid]) / max(own_last[gid], 1e-8)
                w = min(1.0 + cfg.delta_lambda * max(0.0, rel_forget), cfg.delta_max_weight)

            n_phase = min(cfg.phase_steps, cfg.steps - step)
            cursor = global_cursor if cfg.iid_baseline else cursors[gid]
            idxs = cursor.take(n_phase * cfg.batch_size)
            loader = make_phase_loader(dataset, idxs, cfg)
            dl_iter = iter(loader)
            for _ in range(n_phase):
                t0 = time.perf_counter()
                batch = next(dl_iter)
                batch = preprocessor(batch)
                train_tracker.dataloading_s = time.perf_counter() - t0
                train_tracker, _ = update_policy_scaled(
                    train_tracker, policy, batch, optimizer,
                    cfg.optimizer.grad_clip_norm, accelerator, lr_scheduler, loss_scale=w,
                )
                step += 1
                steps_in_cycle += 1
                progbar.update(1)
                train_tracker.step()

                if cfg.log_freq > 0 and step % cfg.log_freq == 0:
                    logging.info(train_tracker)
                    if wandb_logger:
                        d = train_tracker.to_dict()
                        d = {k: v for k, v in d.items()
                             if k in {"loss", "grad_norm", "lr", "epochs", "dataloading_s", "update_s"}}
                        wandb_logger.log_dict(d, step)  # train/* (epochs = GLOBAL data passes)
                        wandb_log_section("cycle", {
                            "position": cycle_idx + steps_in_cycle / steps_per_cycle,  # fractional
                            "active_group": gid,
                        }, step)
                        epoch_d = ({"global": global_cursor.epochs} if cfg.iid_baseline
                                   else {"active_group": cursors[gid].epochs})
                        wandb_log_section("epoch", epoch_d, step)
                    train_tracker.reset_averages()

                if cfg.save_checkpoint and (step % cfg.save_freq == 0 or step == cfg.steps):
                    checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                    logging.info(f"Checkpoint policy after step {step}")
                    save_checkpoint(
                        checkpoint_dir=checkpoint_dir, step=step, cfg=cfg,
                        policy=accelerator.unwrap_model(policy), optimizer=optimizer,
                        scheduler=lr_scheduler, preprocessor=preprocessor, postprocessor=postprocessor,
                    )
                    update_last_checkpoint(checkpoint_dir)
            del dl_iter, loader

            # ── phase boundary: probe everyone, THEN update own-last ──
            # (logging first makes the active group's "forget" = net drift since its own previous
            # phase end — i.e. recovery completeness — instead of a trivial 0)
            probe_vals = measure_probe(policy, preprocessor, probe_batches, accelerator, cfg.probe_seed)
            log_probe(probe_vals, own_last, step, cycle_idx, -1 if cfg.iid_baseline else gid, w)
            if cfg.iid_baseline:
                own_last = dict(probe_vals)  # iid: every boundary "visits" all groups equally
            else:
                own_last[gid] = probe_vals[gid]

            if cfg.probe_grad_group >= 0:
                cur = measure_probe_grad(
                    policy, preprocessor, probe_batches[cfg.probe_grad_group], accelerator, cfg.probe_seed
                )
                if prev_probe_grad is not None:
                    cos = grad_cosine(prev_probe_grad, cur)
                    wandb_log_section("grad_cos", {f"g{cfg.probe_grad_group}": cos}, step)
                    wandb_log_section("probe", {f"grad_cos_g{cfg.probe_grad_group}": cos}, step)  # legacy
                prev_probe_grad = cur

        # ── cycle end: Reptile pull-back toward the anchor ──
        if anchor is not None:
            reptile_interpolate(accelerator.unwrap_model(policy), anchor, beta, device)
            del anchor
            wandb_log_section("cycle", {"reptile_beta": beta}, step)
            probe_vals = measure_probe(policy, preprocessor, probe_batches, accelerator, cfg.probe_seed)
            log_probe(probe_vals, own_last, step, cycle_idx, -2, 1.0, tag="@post-reptile")
        cycle_idx += 1

    progbar.close()
    logging.info("End of cyclic training")
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
