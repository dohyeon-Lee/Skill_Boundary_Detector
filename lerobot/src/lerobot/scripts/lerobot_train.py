#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import dataclasses
import json
import logging
import math
import numbers
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    rabc_weights_provider=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
        rabc_weights_provider: Optional RABCWeights instance for sample weighting.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Get RA-BC weights if enabled
    rabc_batch_weights = None
    rabc_batch_stats = None
    if rabc_weights_provider is not None:
        rabc_batch_weights, rabc_batch_stats = rabc_weights_provider.compute_batch_weights(batch)

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        # Use per-sample loss when RA-BC is enabled for proper weighting
        if rabc_batch_weights is not None:
            # Get per-sample losses
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")

            # Apply RA-BC weights: L_RA-BC = Σ(w_i * l_i) / (Σw_i + ε)
            # rabc_batch_weights is already normalized to sum to batch_size
            epsilon = 1e-6
            loss = (per_sample_loss * rabc_batch_weights).sum() / (rabc_batch_weights.sum() + epsilon)
            # Log raw mean weight (before normalization) - this is the meaningful metric
            output_dict["rabc_mean_weight"] = rabc_batch_stats["raw_mean_weight"]
            output_dict["rabc_num_zero_weight"] = rabc_batch_stats["num_zero_weight"]
            output_dict["rabc_num_full_weight"] = rabc_batch_stats["num_full_weight"]
        else:
            loss, output_dict = policy.forward(batch)

        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


_WINDOWED_POLICY_METRIC_KEYS = {"action_loss", "action_weighted_loss"}
_WINDOWED_POLICY_METRIC_PREFIXES = ("regime/", "terminator/", "wrong_language/")


class _WindowedPolicyMetrics:
    """Mean selected scalar policy diagnostics over one WandB logging window.

    Regime losses are emitted only on their matching batches, so each key keeps
    its own count instead of treating the other regime as a zero-loss update.
    """

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    @staticmethod
    def _tracks(key: str) -> bool:
        return key in _WINDOWED_POLICY_METRIC_KEYS or key.startswith(_WINDOWED_POLICY_METRIC_PREFIXES)

    def update(self, metrics: dict | None) -> None:
        if not metrics:
            return
        for key, value in metrics.items():
            if not self._tracks(key) or not isinstance(value, numbers.Real):
                continue
            value = float(value)
            if not math.isfinite(value):
                continue
            self._sums[key] = self._sums.get(key, 0.0) + value
            self._counts[key] = self._counts.get(key, 0) + 1

    def averages(self) -> dict[str, float]:
        return {key: total / self._counts[key] for key, total in self._sums.items()}

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()


def build_pt_probe_batches(cfg: TrainPipelineConfig) -> list[dict]:
    """PT-forgetting probe (FT): draw probe_batches fixed batches ONCE from the ORIGINAL PT dataset
    (probe_dataset_root) and keep them as raw CPU batches for the whole run — re-measuring their loss
    with pinned noise makes two measurements differ only through the parameters ("same ruler").

    NOTE: probe batches run through the FT run's own preprocessor (FT-dataset normalizer stats) —
    consistent within the run, which is all the forget curve needs; the values are NOT comparable to
    the parent Stage-2 run's train/loss."""
    import numpy as np

    from lerobot.configs.default import DatasetConfig

    probe_policy = copy.deepcopy(cfg.policy)
    # (dino-token 래퍼 억제 코드 은퇴 — terminator는 배치 이미지를 ONLINE 토큰화하므로 probe 데이터셋
    #  래핑 자체가 없음. probe의 terminator loss는 measure_pt_probe의 train_terminator 토글이 차단.)
    probe_cfg = dataclasses.replace(
        cfg,
        dataset=DatasetConfig(
            repo_id=cfg.probe_dataset_repo_id or cfg.dataset.repo_id, root=cfg.probe_dataset_root
        ),
        policy=probe_policy,
    )
    probe_ds = make_dataset(probe_cfg)
    rng = np.random.default_rng(cfg.probe_seed)
    n = min(cfg.probe_batches * cfg.batch_size, probe_ds.num_frames)
    idxs = rng.choice(probe_ds.num_frames, size=n, replace=False)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(probe_ds, [int(i) for i in idxs]),
        batch_size=cfg.batch_size, num_workers=0, shuffle=False,
    )
    return list(loader)


@torch.no_grad()
def measure_pt_probe(policy, preprocessor, probe_batches, accelerator, cfg) -> dict[str, float]:
    """Mean probe loss over the fixed PT batches. fork_rng + probe_seed pins the flow-matching
    noise/timestep. For skill_vla with probe_vsa, a second pass forces the B/VSA regime
    (cond/action→VLM severed via model._probe_force_drop) → *_vsa keys isolate the VSA path."""
    unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    was_training = policy.training
    policy.eval()
    # Probes measure the policy loss only — force-skip the terminator branch. ONLINE DINO 전환 후
    # probe 배치에도 이미지+skill_ds/de가 있어 terminator가 돌 수 있으므로 이 토글이 유일한 가드.
    term_prev = getattr(unwrapped.config, "train_terminator", False)
    if term_prev:
        unwrapped.config.train_terminator = False

    regimes = [("", None)]
    if cfg.probe_vsa and getattr(cfg.policy, "type", None) == "skill_vla":
        regimes.append(("_vsa", True))
    devices = [accelerator.device] if accelerator.device.type == "cuda" else []
    vals: dict[str, float] = {}
    try:
        for suffix, force in regimes:
            if force is not None:
                unwrapped.model._probe_force_drop = force
            accum: dict[str, list[float]] = {}
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(cfg.probe_seed)
                if accelerator.device.type == "cuda":
                    torch.cuda.manual_seed_all(cfg.probe_seed)
                for batch in probe_batches:
                    # fresh copy — an in-place processor must never corrupt the stored originals
                    b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
                    b = preprocessor(b)
                    with accelerator.autocast():
                        loss, out = policy.forward(b)
                    accum.setdefault("loss", []).append(float(loss.item()))
                    for src, dst in (("loss_flow", "flow"), ("loss_skill", "skill")):
                        if out and src in out:
                            accum.setdefault(dst, []).append(float(out[src]))
            if force is not None:
                unwrapped.model._probe_force_drop = None
            for k, v in accum.items():
                vals[f"{k}{suffix}"] = float(sum(v) / len(v))
            # Policies that don't report a separate loss_flow (e.g. the pi05 baseline, whose total
            # loss IS the flow/action loss) get `flow` aliased to `loss`, so probe/flow and
            # probe_forget/flow overlay across pipelines in wandb.
            if f"flow{suffix}" not in vals and f"loss{suffix}" in vals:
                vals[f"flow{suffix}"] = vals[f"loss{suffix}"]
    finally:
        if hasattr(unwrapped, "model"):
            unwrapped.model._probe_force_drop = None
        if term_prev:
            unwrapped.config.train_terminator = term_prev
        if was_training:
            policy.train()
    return vals


def snapshot_component_init(policy, accelerator) -> dict | None:
    """Per-component drift baseline: CPU clones of each component's params at FT/PT start, keyed by name,
    with the group's ‖θ_init‖ precomputed. Returns None if the policy has no named_component_params()."""
    unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    if not hasattr(unwrapped, "named_component_params"):
        return None
    groups = unwrapped.named_component_params()
    init, init_norm_sq = {}, {}
    for g, plist in groups.items():
        s = 0.0
        for pn, p in plist:
            c = p.detach().to("cpu", copy=True)
            init[pn] = c
            s += float(c.float().pow(2).sum())
        init_norm_sq[g] = s
    return {"init": init, "init_norm_sq": init_norm_sq}


@torch.no_grad()
def measure_component_drift(policy, accelerator, snap: dict) -> tuple[dict, dict]:
    """‖θ_now − θ_init‖ per component group (abs) and ÷‖θ_init‖ (rel). Iterates params once, moving the
    CPU init to the param's device per-tensor (transient) → no extra GPU copy of the whole model."""
    unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    groups = unwrapped.named_component_params()
    init, init_norm_sq = snap["init"], snap["init_norm_sq"]
    abs_drift, rel_drift = {}, {}
    for g, plist in groups.items():
        s = 0.0
        for pn, p in plist:
            if pn not in init:
                continue
            d = p.detach().float() - init[pn].to(p.device, dtype=torch.float32)
            s += float(d.pow(2).sum())
        norm = s ** 0.5
        abs_drift[g] = norm
        denom = init_norm_sq[g] ** 0.5
        rel_drift[g] = norm / denom if denom > 0 else 0.0
    return abs_drift, rel_drift


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # Accelerate auto-detects the device based on the available hardware and ignores the policy.device setting.
        # Force the device to be CPU when policy.device is set to CPU.
        force_cpu = cfg.policy.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    if cfg.peft is not None:
        logging.info("Using PEFT! Wrapping model.")
        # Convert CLI peft config to dict for overrides
        peft_cli_overrides = dataclasses.asdict(cfg.peft)
        policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    processor_dataset_stats = dataset.meta.stats
    if cfg.policy.pretrained_path is not None and getattr(policy.config, "use_relative_actions", False):
        from lerobot.policies.diffusion.processor_diffusion import with_diffusion_relative_action_stats

        processor_dataset_stats = with_diffusion_relative_action_stats(
            policy.config, processor_dataset_stats
        )
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = processor_dataset_stats

    # For SARM, always provide dataset_meta for progress normalization
    if cfg.policy.type == "sarm":
        processor_kwargs["dataset_meta"] = dataset.meta

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": processor_dataset_stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        if cfg.policy.type == "skill_vla":
            # SkillVLA's state-tokenizer step needs observation.state q01/q99 to discretize the
            # skill-start state. These aren't carried by the saved processor, so (like the normalizer
            # above) re-inject them from the dataset on resume — otherwise the step gets None and
            # crashes. Salvages pre-fix checkpoints too (whose saved step config is empty).
            state_stats = dataset.meta.stats.get("observation.state", {}) or {}
            processor_kwargs["preprocessor_overrides"]["skill_vla_prepare_state_tokenizer_processor_step"] = {
                "state_q01": state_stats.get("q01"),
                "state_q99": state_stats.get("q99"),
            }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": processor_dataset_stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Load precomputed SARM progress for RA-BC if enabled
    # Generate progress using: src/lerobot/policies/sarm/compute_rabc_weights.py
    rabc_weights = None
    if cfg.use_rabc:
        from lerobot.utils.rabc import RABCWeights

        # Get chunk_size from policy config
        chunk_size = getattr(policy.config, "chunk_size", None)
        if chunk_size is None:
            raise ValueError("Chunk size is not found in policy config")

        head_mode = getattr(cfg, "rabc_head_mode", "sparse")
        logging.info(f"Loading SARM progress for RA-BC from {cfg.rabc_progress_path}")
        logging.info(f"Using chunk_size={chunk_size} from policy config, head_mode={head_mode}")
        rabc_weights = RABCWeights(
            progress_path=cfg.rabc_progress_path,
            chunk_size=chunk_size,
            head_mode=head_mode,
            kappa=getattr(cfg, "rabc_kappa", 0.01),
            epsilon=getattr(cfg, "rabc_epsilon", 1e-6),
            device=device,
        )

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=False,  # disabled: pinned-memory alloc under concurrent jobs triggered CUDA "unknown error"; ~free for compute-bound VLA training
        drop_last=False,
        prefetch_factor=4 if cfg.num_workers > 0 else None,  # deeper buffer hides per-batch video-decode latency
    )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    # ── PT-forgetting probe (FT): fixed PT-dataset batches re-measured every probe_every steps ──
    probe_batches = None
    probe_baseline: dict[str, float] = {}
    if cfg.probe_dataset_root and is_main_process:
        logging.info(f"Building PT-forgetting probe batches from {cfg.probe_dataset_root}")
        probe_batches = build_pt_probe_batches(cfg)

        def run_probe(at_step: int) -> dict[str, float]:
            vals = measure_pt_probe(policy, preprocessor, probe_batches, accelerator, cfg)
            # relative regression vs the step-0 baseline → its OWN wandb section (probe_forget/*):
            # absolute losses and forget ratios have different scales/semantics — separate panels.
            forgets = {k: (v - probe_baseline[k]) / max(abs(probe_baseline[k]), 1e-8)
                       for k, v in vals.items() if k in probe_baseline}
            logging.info(f"[probe] step={at_step} "
                         + " ".join(f"{k}={v:.4f}" for k, v in sorted(vals.items()))
                         + ("  | forget " + " ".join(f"{k}={v:+.3f}" for k, v in sorted(forgets.items()))
                            if forgets else ""))
            if wandb_logger:
                wandb_logger.log_dict(vals, at_step, mode="probe")
                if forgets:
                    wandb_logger.log_dict(forgets, at_step, mode="probe_forget")
            return vals

        # Baseline = the warm-start state (step 0). Persisted so a resumed run keeps the SAME zero
        # (re-baselining mid-run would flatten the forget curve).
        baseline_path = cfg.output_dir / "probe_baseline.json"
        if baseline_path.is_file():
            probe_baseline = json.loads(baseline_path.read_text())
        first_vals = run_probe(step)
        if not probe_baseline:
            probe_baseline = first_vals
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(json.dumps(probe_baseline, indent=1))

    policy.train()

    # Per-component update tracking (skill_vla): snapshot the start state ONCE so drift is measured
    # against the FT/PT warm-start (resume snapshots the resumed weights → drift resets; acceptable).
    drift_snap = None
    if cfg.track_param_drift and is_main_process:
        drift_snap = snapshot_component_init(policy, accelerator)
        if drift_snap is None:
            logging.info("track_param_drift set but the policy has no named_component_params() — skipping.")
        else:
            logging.info(f"Tracking per-component drift for: {list(drift_snap['init_norm_sq'])}")

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Keep global batch size for logging; MetricsTracker handles world size internally.
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )
    # Stage-1's A/B losses are stochastic per batch. Keep other policies' existing final-batch
    # logging semantics unchanged.
    windowed_policy_metrics = (
        _WindowedPolicyMetrics() if getattr(cfg.policy, "model_type", None) == "skill_expert" else None
    )

    if is_main_process:
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            rabc_weights_provider=rabc_weights,
        )
        if is_main_process and windowed_policy_metrics is not None:
            windowed_policy_metrics.update(output_dict)

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        if is_main_process:
            progbar.update(1)
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                if windowed_policy_metrics is not None:
                    # Stage-1 action/regime diagnostics are per-forward values. Replace the final batch's
                    # value with its average over this full log window; A/B-only keys divide by their own
                    # observed count, never by all batches in the window.
                    wandb_log_dict.update(windowed_policy_metrics.averages())
                # Log RA-BC statistics if enabled
                if rabc_weights is not None:
                    rabc_stats = rabc_weights.get_stats()
                    wandb_log_dict.update(
                        {
                            "rabc_delta_mean": rabc_stats["delta_mean"],
                            "rabc_delta_std": rabc_stats["delta_std"],
                            "rabc_num_frames": rabc_stats["num_frames"],
                        }
                    )
                _wandb_keep = {
                    "loss", "grad_norm", "lr", "epochs", "episodes",
                    "dataloading_s",
                    "loss_skill_decoder", "loss_flow", "loss_skill_predictor",
                    "loss_skill_decoder_action_loss", "loss_skill_decoder_end_loss",
                    "predicted_latent_prob",
                    "loss_skill", "skill_acc",  # Stage-2 SkillVLA: skill regression loss + skill-code accuracy
                    # (co-trained FSQ terminator logs via "terminator/*" → routed to train_terminator/* below)
                    "action_loss",              # Stage-1 skill_expert: PLAIN (unweighted) action MSE — always (comparison)
                    "action_weighted_loss",     # Stage-1 skill_expert: per-sample-weighted action MSE (action_weight only)
                }
                wandb_log_dict = {k: v for k, v in wandb_log_dict.items()
                                  if k in _wandb_keep or k.startswith(
                                      ("terminator/", "regime/", "distill/", "wrong_language/"))}
                # A policy that reports its own action_loss (Stage-1 skill_expert) replaces the generic
                # backprop-scalar "loss" with it → drop the redundant generic "loss".
                if "action_loss" in wandb_log_dict:
                    wandb_log_dict.pop("loss", None)
                # Route "terminator/*" → train_terminator/* and "regime/*" (CFG A/B per-regime losses)
                # → train_regime/*, each a SEPARATE wandb panel; the rest → train/*.
                term_metrics = {k[len("terminator/"):]: v for k, v in wandb_log_dict.items()
                                if k.startswith("terminator/")}
                regime_metrics = {k[len("regime/"):]: v for k, v in wandb_log_dict.items()
                                  if k.startswith("regime/")}
                distill_metrics = {k[len("distill/"):]: v for k, v in wandb_log_dict.items()
                                   if k.startswith("distill/")}
                wrong_language_metrics = {
                    k[len("wrong_language/"):]: v for k, v in wandb_log_dict.items()
                    if k.startswith("wrong_language/")
                }
                main_metrics = {k: v for k, v in wandb_log_dict.items()
                                if not k.startswith(
                                    ("terminator/", "regime/", "distill/", "wrong_language/"))}
                wandb_logger.log_dict(main_metrics, step)
                if term_metrics:
                    wandb_logger.log_dict(term_metrics, step, mode="train_terminator")
                if regime_metrics:
                    wandb_logger.log_dict(regime_metrics, step, mode="train_regime")
                if distill_metrics:
                    wandb_logger.log_dict(distill_metrics, step, mode="train_distill")
                if wrong_language_metrics:
                    wandb_logger.log_dict(
                        wrong_language_metrics, step, mode="train_wrong_language")
            train_tracker.reset_averages()
            if windowed_policy_metrics is not None:
                windowed_policy_metrics.reset()

            if drift_snap is not None and wandb_logger:
                abs_d, rel_d = measure_component_drift(policy, accelerator, drift_snap)
                wandb_logger.log_dict(abs_d, step, mode="param_drift")
                wandb_logger.log_dict(rel_d, step, mode="param_drift_rel")

        if probe_batches is not None and cfg.probe_every > 0 and step % cfg.probe_every == 0:
            run_probe(step)

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if is_main_process:
        progbar.close()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            if cfg.policy.use_peft:
                unwrapped_policy.push_model_to_hub(cfg, peft_model=unwrapped_policy)
            else:
                unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
