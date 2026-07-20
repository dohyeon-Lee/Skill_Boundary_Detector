#!/usr/bin/env python
"""Replay-free FT on new tasks + live PT-group forgetting probes — the hypothesis stress test.

Plain iid finetuning (no curriculum, no replay) of a cycle-PT checkpoint on a NEW-task dataset
(e.g., libero_10), while measuring probe losses on the ORIGINAL PT dataset's groups every
`probe_every` steps. Probes reuse the PT run's groups.json + the same probe_seed, so the frames
AND flow-matching noise match the PT-time probes exactly → loss values are directly comparable.

wandb: train/* (FT loss on new tasks) + probe/g{j}_loss, probe/g{j}_forget
(forget = relative regression vs FT step 0 = the PT-end state). The probe/g{j}_forget curves
ARE the forgetting-during-FT trajectories compared across PT conditions.

NOTE (deliberate deviation from the standard FT recipe): the PT checkpoint's processors
(normalizer/unnormalizer stats) are KEPT — re-fitting stats to the FT dataset would itself
damage old tasks and confound parameter-space forgetting with normalization drift.

Single-GPU, no resume (mirrors lerobot_train_cycle.py).
"""

import dataclasses
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
from accelerate import Accelerator
from termcolor import colored
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import get_step_checkpoint_dir, save_checkpoint, update_last_checkpoint
from lerobot.utils.utils import format_big_number, init_logging, inside_slurm

from lerobot_train_cycle import (  # noqa: E402  (same src/ dir)
    GroupCursor,
    build_probe_batches,
    clone_batch,
    measure_probe,
    update_policy,
)


@dataclass
class FTProbeTrainPipelineConfig(TrainPipelineConfig):
    # ── PT-side probe (forgetting measurement during FT) ───────────────────
    probe_dataset_repo_id: str = "lerobot/libero_90_full_full"
    probe_dataset_root: str = ""     # PT dataset dir
    probe_groups_json: str = ""      # the SOURCE PT run's groups.json
    probe_every: int = 250           # steps between probe measurements
    probe_batches_per_group: int = 2
    probe_seed: int = 12345          # MUST match PT runs → identical probe frames + noise


def rebuild_groups_from_json(groups_json: Path, probe_ds) -> list[dict]:
    """Reconstruct per-group frame indices on the probe dataset from the PT run's task lists
    (exact same grouping as PT time; frame indices derived from episode metadata)."""
    groups_spec = json.loads(groups_json.read_text())
    eps = probe_ds.meta.episodes
    ep_from = [int(v) for v in eps["dataset_from_index"]]
    ep_to = [int(v) for v in eps["dataset_to_index"]]
    ep_task = []
    for t in eps["tasks"]:
        if isinstance(t, (list, tuple, np.ndarray)):
            t = t[0]
        ep_task.append(str(t))

    groups = []
    for g in groups_spec:
        gtasks = set(g["tasks"])
        ep_ids = [i for i, t in enumerate(ep_task) if t in gtasks]
        frames = np.concatenate([np.arange(ep_from[i], ep_to[i]) for i in ep_ids])
        groups.append({"group_id": g["group_id"], "tasks": g["tasks"],
                       "episode_indices": ep_ids, "frame_indices": frames})
    return groups


@parser.wrap()
def train(cfg: FTProbeTrainPipelineConfig):
    cfg.validate()
    if not cfg.probe_dataset_root or not cfg.probe_groups_json:
        raise ValueError("probe_dataset_root and probe_groups_json are required (PT-side probes).")

    accelerator = Accelerator(step_scheduler_with_optimizer=False)
    if accelerator.num_processes != 1:
        raise RuntimeError("lerobot_train_ft_probe.py is single-GPU only.")
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

    logging.info("Creating FT dataset (new tasks)")
    dataset = make_dataset(cfg)

    logging.info("Creating probe dataset (PT tasks)")
    probe_cfg = dataclasses.replace(
        cfg, dataset=DatasetConfig(repo_id=cfg.probe_dataset_repo_id, root=cfg.probe_dataset_root)
    )
    probe_ds = make_dataset(probe_cfg)

    logging.info("Creating policy from PT checkpoint")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)

    # KEEP the PT checkpoint's processors (stats) — see module docstring.
    preprocessor_overrides = {
        "device_processor": {"device": device.type},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    tokenizer_path = getattr(policy.config, "tokenizer_path", None)
    if tokenizer_path:
        preprocessor_overrides["tokenizer_processor"] = {"tokenizer_name": tokenizer_path}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    policy, optimizer, lr_scheduler = accelerator.prepare(policy, optimizer, lr_scheduler)

    # ── probes: same groups, same frames, same noise as PT time ────────────
    groups = rebuild_groups_from_json(Path(cfg.probe_groups_json), probe_ds)
    for g in groups:
        logging.info(f"probe group {g['group_id']}: {len(g['tasks'])} tasks, {len(g['frame_indices'])} frames")
    probe_batches = build_probe_batches(probe_ds, groups, cfg)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg.probe_groups_json, cfg.output_dir / "groups.json")  # → cycle_eval group agg

    # FT sampling: global epoch cursor over the whole new-task dataset (plain iid)
    all_frames = np.arange(dataset.num_frames)
    cursor = GroupCursor(all_frames, seed=(cfg.seed or 0, 777))

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
    num_learnable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logging.info(f"num_learnable_params={format_big_number(num_learnable)} | probe_every={cfg.probe_every}")

    def wandb_log_section(section: str, d: dict, step: int):
        if wandb_logger:
            wandb_logger._wandb.log({f"{section}/{k}": v for k, v in d.items()}, step=step)

    def measure_and_log(step: int):
        vals = measure_probe(policy, preprocessor, probe_batches, accelerator, cfg.probe_seed)
        wandb_log_section("probe_loss", {f"g{j}": v for j, v in vals.items()}, step)
        legacy = {f"g{j}_loss": v for j, v in vals.items()}
        if baseline:
            forgets = {j: (vals[j] - baseline[j]) / max(baseline[j], 1e-8) for j in vals}
            wandb_log_section("probe_forget", {f"g{j}": v for j, v in forgets.items()}, step)
            legacy |= {f"g{j}_forget": v for j, v in forgets.items()}
        # legacy duplicate under probe/ — keeps runs overlayable with pre-split PT runs
        wandb_log_section("probe", legacy, step)
        logging.info(f"[probe] step={step} " + " ".join(f"g{j}={v:.4f}" for j, v in sorted(vals.items())))
        return vals

    baseline: dict = {}
    baseline = measure_and_log(0)  # FT step 0 = the PT-end state (forgetting reference)

    progbar = tqdm(total=cfg.steps, desc="FT", unit="step", disable=inside_slurm())
    step = 0
    while step < cfg.steps:
        n_chunk = min(cfg.probe_every, cfg.steps - step)
        idxs = cursor.take(n_chunk * cfg.batch_size)
        loader = torch.utils.data.DataLoader(
            dataset, num_workers=cfg.num_workers, batch_size=cfg.batch_size,
            sampler=[int(i) for i in idxs], pin_memory=False,
            prefetch_factor=4 if cfg.num_workers > 0 else None,
        )
        dl_iter = iter(loader)
        for _ in range(n_chunk):
            t0 = time.perf_counter()
            batch = next(dl_iter)
            batch = preprocessor(batch)
            train_tracker.dataloading_s = time.perf_counter() - t0
            train_tracker, _ = update_policy(
                train_tracker, policy, batch, optimizer,
                cfg.optimizer.grad_clip_norm, accelerator, lr_scheduler,
            )
            step += 1
            progbar.update(1)
            train_tracker.step()

            if cfg.log_freq > 0 and step % cfg.log_freq == 0:
                logging.info(train_tracker)
                if wandb_logger:
                    d = train_tracker.to_dict()
                    d = {k: v for k, v in d.items()
                         if k in {"loss", "grad_norm", "lr", "epochs", "dataloading_s", "update_s"}}
                    wandb_logger.log_dict(d, step)
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
        measure_and_log(step)

    progbar.close()
    logging.info("End of FT (+forgetting probes)")
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
