"""Process-local data/schedule/checkpoint hooks around unmodified LeRobot.

Model, losses, forward/backward, BF16 Kahan update math, state serialization,
accumulation and sample-exact resume stay upstream. YAML steps, checkpoint
names and W&B axes count optimizer updates rather than micro-batches.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from torch.optim.lr_scheduler import LambdaLR
from lerobot.optim.schedulers import LRSchedulerConfig


@LRSchedulerConfig.register_subclass("sbd_molmo_update_clock")
@dataclass
class UpdateClockSchedulerConfig(LRSchedulerConfig):
    num_warmup_steps: int = 1000
    accumulation_steps: int = 1
    logical_steps: int = 100000
    mode: str = "warmup_constant"

    def factor(self, micro_step: int) -> float:
        update = micro_step // self.accumulation_steps
        warmup = self.num_warmup_steps
        if update < warmup:
            return (update + 1) / (warmup + 1)
        if self.mode == "warmup_constant":
            return 1.0
        progress = min(1.0, max(0.0, (update - warmup) / max(1, self.logical_steps - warmup)))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    def build(self, optimizer, num_training_steps):
        return LambdaLR(optimizer, self.factor)


def install(native, settings, provenance):
    from grounding_adapter import ground_dataset
    from checkpoint_retention import prune_checkpoints
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

    accum = settings["accumulation_steps"]
    original_factory = native.make_train_eval_datasets
    original_engine = native.make_accelerator
    original_update_last = native.update_last_checkpoint
    original_processors = native.make_pre_post_processors
    from molmoact2_config import latest_checkpoint
    from ft_normalization import with_checkpoint_normalization
    resuming = latest_checkpoint(Path(settings["output_dir"])) is not None

    if settings.get("optimizer_state_offload", "none") == "cpu":
        # Import before native.main parses saved optimizer configs on resume.
        from optimizer_offload import offload_optimizer_config
        original_optimizer_factory = native.make_optimizer_and_scheduler

        def optimizer_factory(cfg, policy):
            cfg.optimizer = offload_optimizer_config(cfg.optimizer)
            return original_optimizer_factory(cfg, policy)

        native.make_optimizer_and_scheduler = optimizer_factory

    def make_datasets(cfg):
        train, val = original_factory(cfg)
        return (ground_dataset(train, settings["proprio_grounding"]),
                ground_dataset(val, settings["proprio_grounding"]))

    def engine(cfg):
        # All ranks must pass cfg.validate before rank 0 creates output_dir.
        # Otherwise a slower peer can mistake rank 0's new metadata for an old run.
        accelerator = original_engine(cfg)
        accelerator.wait_for_everyone()
        provenance(settings)
        accelerator.wait_for_everyone()
        return accelerator

    def checkpoint_dir(output, total, micro_step):
        if micro_step % accum:
            raise ValueError("Cannot checkpoint part-way through an accumulated update")
        width = max(6, len(str(settings["steps"])))
        return Path(output) / "checkpoints" / f"{micro_step // accum:0{width}d}"

    def update_last(path):
        payload = {"backend": "hf_lerobot", "proprio_grounding": settings["proprio_grounding"],
                   "normalization_stats": settings.get("normalization_stats", "ft" if settings["stage"] == "ft" else "dataset"),
                   "optimizer_state_offload": settings.get("optimizer_state_offload", "none"),
                   "optimizer_step": int(path.name), "accumulation_steps": accum,
                   "native_micro_step": int(path.name) * accum}
        (path / "pretrained_model/sbd_molmoact2.json").write_text(json.dumps(payload, indent=2))
        original_update_last(path)
        prune_checkpoints(Path(settings["output_dir"]), settings["keep_checkpoints"],
                          settings["protect_checkpoints"])

    def processors(*args, **kwargs):
        # Fresh FT chooses statistics; resume restores its own saved processors.
        if settings["stage"] == "ft" and not resuming:
            if settings.get("normalization_stats", "ft") == "pt":
                kwargs["dataset_stats"] = with_checkpoint_normalization(
                    kwargs["dataset_stats"], Path(settings["initial_checkpoint"]) / "pretrained_model",
                )
            kwargs["pretrained_path"] = None
            kwargs.pop("preprocessor_overrides", None)
            kwargs.pop("postprocessor_overrides", None)
        elif resuming:
            kwargs.pop("dataset_stats", None)
            for key in ("preprocessor_overrides", "postprocessor_overrides"):
                overrides = dict(kwargs.get(key) or {})
                for name in ("normalizer_processor", "unnormalizer_processor",
                             "molmoact2_masked_normalizer", "molmoact2_masked_unnormalizer",
                             "molmoact2_clamp_normalized"):
                    overrides.pop(name, None)
                kwargs[key] = overrides
        return original_processors(*args, **kwargs)

    MolmoAct2Config.get_scheduler_preset = lambda self: UpdateClockSchedulerConfig(
        num_warmup_steps=settings["warmup_steps"], accumulation_steps=accum,
        logical_steps=settings["steps"], mode=settings["lr_mode"])
    native.make_train_eval_datasets = make_datasets
    native.make_accelerator = engine
    native.get_step_checkpoint_dir = checkpoint_dir
    native.update_last_checkpoint = update_last
    native.make_pre_post_processors = processors

    class UpdateClockWandBLogger(native.WandBLogger):
        def log_dict(self, values, step, *args, **kwargs):
            values = dict(values)
            values["micro_step"] = step
            values["optimizer_step"] = step // accum
            if "steps" in values:
                values["steps"] = step // accum
            for key in ("update_s", "dataloading_s", "preprocessing_s", "step_s"):
                if key in values:
                    values[key] *= accum
            return super().log_dict(values, step // accum, *args, **kwargs)

    native.WandBLogger = UpdateClockWandBLogger
