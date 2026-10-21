"""Process-local hooks; GR00T architecture, loss and optimizer remain upstream."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.checkpoint import checkpoint
from lerobot.optim.schedulers import LRSchedulerConfig


@LRSchedulerConfig.register_subclass("sbd_groot_update_clock")
@dataclass
class UpdateClockSchedulerConfig(LRSchedulerConfig):
    num_warmup_steps: int = 1000
    accumulation_steps: int = 1
    logical_steps: int = 100000
    mode: str = "warmup_constant"

    def factor(self, micro_step: int) -> float:
        update = micro_step // self.accumulation_steps
        if update < self.num_warmup_steps:
            return (update + 1) / (self.num_warmup_steps + 1)
        if self.mode == "warmup_constant":
            return 1.
        progress = min(1., max(0., (update - self.num_warmup_steps) /
                                  max(1, self.logical_steps - self.num_warmup_steps)))
        return .1 + .9 * .5 * (1 + math.cos(math.pi * progress))

    def build(self, optimizer, num_training_steps):
        return LambdaLR(optimizer, self.factor)


def checkpoint_action_blocks(policy) -> int:
    """Preserve RNG and numerical graph; native DiT flags alone are not read in forward."""
    head = policy._groot_model.action_head
    count = 0
    for branch in (head.model, getattr(head, "vl_self_attention", None)):
        for block in getattr(branch, "transformer_blocks", []):
            original = block.forward
            def wrapped(*args, _forward=original, _block=block, **kwargs):
                if not _block.training or not torch.is_grad_enabled():
                    return _forward(*args, **kwargs)
                return checkpoint(_forward, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)
            block.forward = wrapped
            count += 1
    backbone = getattr(policy._groot_model.backbone, "model", None)
    if backbone is not None and any(p.requires_grad for p in backbone.parameters()):
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        backbone.config.use_cache = False
    return count


def install(native, settings, provenance):
    from checkpoint_retention import prune_checkpoints
    from grounding_adapter import ground_dataset
    from groot_config import latest_checkpoint
    from processor_adapter import make_dataset_processors
    from lerobot.policies.groot.configuration_groot import GrootConfig

    accum = settings["accumulation_steps"]
    original_factory, original_engine = native.make_train_eval_datasets, native.make_accelerator
    original_last, original_processors, original_policy = (
        native.update_last_checkpoint, native.make_pre_post_processors, native.make_policy)
    resuming = latest_checkpoint(Path(settings["output_dir"])) is not None

    def make_datasets(cfg):
        train, val = original_factory(cfg)
        return (ground_dataset(train, settings["proprio_grounding"]),
                ground_dataset(val, settings["proprio_grounding"]))

    def engine(cfg):
        accelerator = original_engine(cfg)
        accelerator.wait_for_everyone()
        provenance(settings)
        accelerator.wait_for_everyone()
        return accelerator

    def make_policy(*args, **kwargs):
        policy = original_policy(*args, **kwargs)
        if settings["checkpointing"]:
            count = checkpoint_action_blocks(policy)
            if not count:
                raise RuntimeError("Requested gradient checkpointing, but no GR00T action blocks were found.")
        return policy

    def processors(*args, **kwargs):
        cfg = kwargs["policy_cfg"]
        meta = getattr(cfg, "_runtime_dataset_meta", None)
        if meta is None:
            raise RuntimeError("GR00T dataset metadata must be attached before processor construction.")
        if resuming:
            # Preserve saved stats AND restore training crop/dropout mode; the
            # upstream trainer otherwise omits dataset_meta for saved pipelines.
            kwargs["dataset_meta"] = meta
            return original_processors(*args, **kwargs)
        return make_dataset_processors(cfg, kwargs["dataset_stats"], meta, horizon=settings["horizon"])

    def checkpoint_dir(output, total, micro_step):
        if micro_step % accum:
            raise ValueError("Cannot checkpoint part-way through an accumulated update.")
        return Path(output) / "checkpoints" / f"{micro_step // accum:0{max(6, len(str(settings['steps'])))}d}"

    def update_last(path):
        payload = {"backend": "hf_lerobot_groot_n1_7", "proprio_grounding": settings["proprio_grounding"],
                   "optimizer_step": int(path.name), "accumulation_steps": accum,
                   "native_micro_step": int(path.name) * accum,
                   "action_space": "delta_eef", "state_dim": 8, "action_dim": 7}
        (path / "pretrained_model/sbd_groot.json").write_text(json.dumps(payload, indent=2))
        original_last(path)
        prune_checkpoints(Path(settings["output_dir"]), settings["keep_checkpoints"], settings["protect_checkpoints"])

    GrootConfig.get_scheduler_preset = lambda self: UpdateClockSchedulerConfig(
        num_warmup_steps=settings["warmup_steps"], accumulation_steps=accum,
        logical_steps=settings["steps"], mode=settings["lr_mode"])
    native.make_train_eval_datasets, native.make_accelerator = make_datasets, engine
    native.make_policy, native.make_pre_post_processors = make_policy, processors
    native.get_step_checkpoint_dir, native.update_last_checkpoint = checkpoint_dir, update_last

    class UpdateClockWandBLogger(native.WandBLogger):
        def log_dict(self, values, step, *args, **kwargs):
            values = dict(values)
            values.update(micro_step=step, optimizer_step=step // accum)
            if "steps" in values:
                values["steps"] = step // accum
            for key in ("update_s", "dataloading_s", "preprocessing_s", "step_s"):
                if key in values:
                    values[key] *= accum
            return super().log_dict(values, step // accum, *args, **kwargs)
    native.WandBLogger = UpdateClockWandBLogger
