"""FT-time PT-group probes, with one immutable baseline at FT step zero."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import shutil

import torch

from diagnostics import CurriculumObserver


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


class FTObserver(CurriculumObserver):
    """Reuse PT's loss/noise implementation; never change its phase-relative diagnostics.

    The raw PT probes already contain episode-relative XYZ. Loading them directly
    preserves both frames and grounding without decoding or grounding a second time.
    FT updates use only the new-task native DataLoader; probes are forward-only.
    """

    def __init__(self, cfg, policy, preprocessor, accelerator, logger, step, settings):
        self.cfg, self.policy, self.preprocessor = cfg, policy, preprocessor
        self.accelerator, self.logger = accelerator, logger
        self.root, self.step, self.mode = Path(cfg.output_dir), step, "ft"
        self.every, self.pending = settings["cyclic_ft_probe_every"], False
        source = Path(settings["cyclic_ft_source_run"])
        checkpoint = Path(settings["ft_pretrained_model_path"])
        groups = json.loads((source / "groups.json").read_text())
        probe_config = json.loads((source / "cyclic_probe_config.json").read_text())
        parent = json.loads((checkpoint / "train_config.json").read_text())
        digest = hashlib.sha256(json.dumps(groups, sort_keys=True).encode()).hexdigest()
        if probe_config["groups"] != digest:
            raise ValueError("Source groups.json does not match the saved PT probes.")
        if probe_config["dataset"] != parent["dataset"]["repo_id"]:
            raise ValueError("PT probes and source checkpoint use different datasets.")
        if probe_config["batch_size"] != parent["batch_size"]:
            raise ValueError("PT probe batch size differs from its checkpoint.")
        if cfg.dataset.repo_id == parent["dataset"]["repo_id"]:
            raise ValueError("Replay-free FT requires a new-task dataset, separate from the PT probes.")
        grounding = parent["policy"].get("proprio_grounding", "none")
        if cfg.policy.proprio_grounding != grounding or cfg.policy.chunk_size != parent["policy"]["chunk_size"]:
            raise ValueError("FT must inherit PT grounding and action horizon.")
        if cfg.normalization_stats != "pt":
            raise ValueError("FT probes require the saved PT normalization statistics.")
        if cfg.batch_size != settings["ft_batch_size"] or cfg.steps != settings["ft_steps"]:
            raise ValueError("Native training config differs from the selected FT recipe.")
        if Path(cfg.output_dir).resolve() != Path(settings["ft_output_dir"]).resolve():
            raise ValueError("Native FT output differs from the selected FT config.")
        self.seed = probe_config["probe_seed"]
        self.groups = groups
        runtime = cfg.to_dict()
        # No absolute dataset paths: identical checkpoint/probes may be moved between servers.
        signature = {
            "source_run": source.name, "source_step": checkpoint.parent.name,
            "source_train_sha256": file_digest(checkpoint / "train_config.json"),
            "source_policy_sha256": file_digest(checkpoint / "config.json"),
            "probe_config": probe_config,
            "probe_batches_sha256": file_digest(source / "cyclic_probe_batches.pt"),
            "dataset": cfg.dataset.repo_id, "batch_size": cfg.batch_size,
            "steps": cfg.steps, "seed": cfg.seed, "probe_every": self.every,
            "normalization_stats": "pt", "grounding": grounding,
            "optimizer": runtime["optimizer"], "scheduler": runtime["scheduler"],
            "policy": {key: runtime["policy"].get(key) for key in (
                "chunk_size", "dtype", "freeze_vision_encoder", "freeze_language_model",
                "freeze_action_expert", "gradient_checkpointing", "gradient_checkpointing_expert",
                "gradient_checkpointing_llm", "gradient_checkpointing_vision",
            )},
        }
        self.signature = signature
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = self.root / "cyclic_ft_manifest.json"
        raw_path = self.root / "cyclic_probe_batches.pt"
        if manifest.exists():
            if json.loads(manifest.read_text()) != signature:
                raise ValueError("Saved FT source/probes/recipe differs; use the original configuration on resume.")
            if not raw_path.is_file() or file_digest(raw_path) != signature["probe_batches_sha256"]:
                raise ValueError("Saved FT probe batches are missing or changed.")
            if json.loads((self.root / "groups.json").read_text()) != groups:
                raise ValueError("Saved FT groups differ from the PT source.")
            if not step:
                raise ValueError("Existing FT diagnostics require a checkpoint resume.")
        else:
            if step:
                raise ValueError("FT resume requires its original manifest and raw probes.")
            shutil.copy2(source / "cyclic_probe_batches.pt", raw_path)
            write_json(self.root / "groups.json", groups)
            write_json(self.root / "cyclic_probe_config.json", probe_config)
            write_json(manifest, signature)
        self.batches = torch.load(raw_path, map_location="cpu", weights_only=False)
        expected = set(range(len(groups)))
        if set(self.batches) != expected or [g["group_id"] for g in groups] != sorted(expected):
            raise ValueError("Incomplete PT group probe batches.")
        for group in groups:
            batches = self.batches[group["group_id"]]
            if not batches or any(not batch["task"] or not set(batch["task"]) <= set(group["tasks"])
                                  for batch in batches):
                raise ValueError("PT probe tasks do not match their saved groups.")
        self.metrics_path = self.root / "cyclic_ft_metrics.jsonl"
        self.baseline = {}
        if step:
            state = json.loads((Path(cfg.checkpoint_path) / "cyclic_ft_diagnostics.json").read_text())
            if state["step"] != step or state["signature"] != signature:
                raise ValueError("FT checkpoint diagnostics do not match the restored step or recipe.")
            self.baseline = {int(k): v for k, v in state["baseline"].items()}
            if set(self.baseline) != expected or not all(
                torch.isfinite(torch.tensor(v)).item() for v in self.baseline.values()
            ):
                raise ValueError("Invalid FT step-zero baseline.")
            if self.metrics_path.exists():
                rows = [line for line in self.metrics_path.read_text().splitlines()
                        if json.loads(line)["step"] < step]
                self.metrics_path.write_text("".join(line + "\n" for line in rows))
        self.measure(step)

    def on_batch(self, batch):
        if self.pending:
            raise RuntimeError("FT observer received a batch before the previous update committed.")
        self.pending = True

    def on_step(self, step):
        if not self.pending or step != self.step + 1:
            raise RuntimeError("FT observer must follow each committed optimizer update.")
        self.step, self.pending = step, False
        if step % self.every == 0 or step == self.cfg.steps:
            self.measure(step)

    def measure(self, step):
        flags = [(module, module.training) for module in self.policy.modules()]
        self.policy.eval()
        try:
            values = self.losses()
            if not self.baseline:
                if step:
                    raise ValueError("Cannot reset the FT baseline after step zero.")
                self.baseline = dict(values)
                write_json(self.root / "cyclic_ft_baseline.json", self.baseline)
            losses = {f"g{gid}": value for gid, value in values.items()}
            forget = {f"g{gid}": (value - self.baseline[gid]) / max(abs(self.baseline[gid]), 1e-8)
                      for gid, value in values.items()}
            self.emit(step, {"probe_loss": losses, "probe_forget": forget,
                             "probe": {**{f"{g}_loss": v for g, v in losses.items()},
                                       **{f"{g}_forget": v for g, v in forget.items()}}})
            logging.info("[cyclic FT probe] step=%d loss=%s forget=%s", step, losses, forget)
        finally:
            for module, training in flags:
                module.training = training

    def on_checkpoint(self, directory, step):
        if step != self.step or self.pending:
            raise RuntimeError("FT checkpoint diagnostic step mismatch.")
        write_json(Path(directory) / "cyclic_ft_diagnostics.json",
                   {"step": step, "signature": self.signature, "baseline": self.baseline})


def make_observer_factory(settings):
    def factory(cfg, dataset, policy, preprocessor, accelerator, logger, step):
        return FTObserver(cfg, policy, preprocessor, accelerator, logger, step, settings)

    return factory
