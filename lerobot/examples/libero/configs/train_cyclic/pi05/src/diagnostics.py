"""Shared IID/cyclic measurement, separate from the native optimizer and sampler.

Fixed raw frames and pinned flow noise provide the same ruler for both modes.
Probe gradients use autograd.grad: parameter .grad and optimizer state are untouched.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import logging
from pathlib import Path
import random

import numpy as np
import torch

from sampling import build_groups


@contextmanager
def pinned_rng(seed, device):
    """Do not consume training RNG, including while decoding fixed probe frames."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def clone_batch(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {key: clone_batch(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_batch(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_batch(item) for item in value)
    return copy.deepcopy(value)


def grad_cosine(previous, current):
    dot = norm_a = norm_b = 0.0
    for a, b in zip(previous, current, strict=True):
        if a is None or b is None:
            continue
        a, b = a.float(), b.float()
        dot += float((a * b).sum())
        norm_a += float(a.square().sum())
        norm_b += float(b.square().sum())
    if not norm_a or not norm_b:
        return None
    return dot / (norm_a * norm_b) ** 0.5


class CurriculumObserver:
    def __init__(self, cfg, dataset, policy, preprocessor, accelerator, logger, step, settings):
        self.cfg, self.policy, self.preprocessor = cfg, policy, preprocessor
        self.accelerator, self.logger, self.settings = accelerator, logger, settings
        self.step, self.mode = step, settings["cyclic_mode"]
        self.phase = settings["cyclic_phase_steps"]
        self.seed = settings["cyclic_probe_seed"]
        self.grad_group = settings["cyclic_probe_grad_group"]
        self.groups = build_groups(dataset, settings["cyclic_n_groups"], settings["cyclic_group_seed"])
        self.task_group = {task: g["group_id"] for g in self.groups for task in g["tasks"]}
        self.counts = [0] * len(self.groups)
        self.pending = None
        self.root = Path(cfg.output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        compact = [{key: value for key, value in g.items() if key != "frame_indices"} for g in self.groups]
        signature = {
            "groups": hashlib.sha256(json.dumps(compact, sort_keys=True).encode()).hexdigest(),
            "dataset": str(cfg.dataset.repo_id), "batch_size": cfg.batch_size,
            "batches_per_group": settings["cyclic_probe_batches_per_group"], "probe_seed": self.seed,
            "phase_steps": self.phase, "group_seed": settings["cyclic_group_seed"],
            "grad_group": self.grad_group,
        }
        group_path = self.root / "groups.json"
        if group_path.exists() and json.loads(group_path.read_text()) != compact:
            raise ValueError("Diagnostic groups differ from groups.json.")
        if not group_path.exists():
            group_path.write_text(json.dumps(compact, indent=2))
        metadata = self.root / "cyclic_probe_config.json"
        raw_path = self.root / "cyclic_probe_batches.pt"
        if metadata.exists():
            if json.loads(metadata.read_text()) != signature:
                raise ValueError("Saved probe configuration differs; use an unchanged ruler on resume.")
            self.batches = torch.load(raw_path, map_location="cpu", weights_only=False)
        else:
            if step:
                raise ValueError("Resume requires the original cyclic probe batches and configuration.")
            self.batches = {}
            rng = np.random.default_rng(self.seed)
            with pinned_rng(self.seed, accelerator.device):
                for g in self.groups:
                    count = min(len(g["frame_indices"]), cfg.batch_size * settings["cyclic_probe_batches_per_group"])
                    indices = rng.choice(g["frame_indices"], size=count, replace=False).tolist()
                    loader = torch.utils.data.DataLoader(
                        torch.utils.data.Subset(dataset, indices), batch_size=cfg.batch_size,
                        shuffle=False, num_workers=0,
                    )
                    self.batches[g["group_id"]] = list(loader)
            torch.save(self.batches, raw_path)
            metadata.write_text(json.dumps(signature, indent=2))
        self.signature = signature
        self.metrics_path = self.root / "cyclic_metrics.jsonl"
        if step:
            state_path = Path(cfg.checkpoint_path) / "cyclic_diagnostics.json"
            state = json.loads(state_path.read_text())
            if state["step"] != step or state["signature"] != signature or state["mode"] != self.mode:
                raise ValueError("Checkpoint cyclic diagnostics do not match the resumed optimizer step.")
            self.own_last = {int(k): v for k, v in state["own_last"].items()}
            self.counts = state["counts"]
            # A failed continuation may have logged beyond the checkpoint. Do not mix that future in.
            if self.metrics_path.exists():
                rows = [line for line in self.metrics_path.read_text().splitlines() if json.loads(line)["step"] < step]
                self.metrics_path.write_text("".join(line + "\n" for line in rows))
        else:
            self.own_last = {}
        self.previous_grad = None
        self.previous_grad_step = step
        self.measure(step, initial=True)

    def _order(self, cycle):
        return np.random.default_rng(self.settings["cyclic_group_seed"] + 1 + cycle).permutation(len(self.groups))

    def active_group(self, step):
        if self.mode == "iid" or not step:
            return -1
        phase_index = (step - 1) // self.phase
        return int(self._order(phase_index // len(self.groups))[phase_index % len(self.groups)])

    def on_batch(self, batch):
        # Count actual committed samples, not the sampler's prefetched cursor. Works for IID too.
        self.pending = [0] * len(self.groups)
        for task in batch["task"]:
            self.pending[self.task_group[task]] += 1

    def on_step(self, step):
        if self.pending is None or step != self.step + 1:
            raise RuntimeError("Diagnostic observer must follow each committed optimizer update.")
        if self.mode == "cyclic":
            active = self.active_group(step)
            if sum(self.pending) != self.pending[active]:
                raise RuntimeError("Observed training batch does not match the cyclic schedule.")
        self.counts = [old + new for old, new in zip(self.counts, self.pending, strict=True)]
        self.pending, self.step = None, step
        if step % self.phase == 0 or step == self.cfg.steps:
            self.measure(step)
        elif self.cfg.log_freq > 0 and step % self.cfg.log_freq == 0:
            self.emit(step, self.progress(step))

    def progress(self, step):
        return {
            "cycle": {"index": (step - 1) // (self.phase * len(self.groups)) if step else 0,
                      "position": step / (self.phase * len(self.groups)),
                      "active_group": self.active_group(step),
                      "phase_step": (step - 1) % self.phase + 1 if step else 0},
            "epoch": {**{f"g{g['group_id']}": self.counts[g["group_id"]] / len(g["frame_indices"]) for g in self.groups},
                      "global": sum(self.counts) / sum(len(g["frame_indices"]) for g in self.groups)},
        }

    def _loss(self, gid, index, batch):
        # The same noise is used for a group's loss and gradient diagnostic in both modes.
        with pinned_rng(self.seed + gid * 1000 + index, self.accelerator.device):
            prepared = self.preprocessor(clone_batch(batch))
            with self.accelerator.autocast():
                loss, _ = self.policy.forward(prepared)
            return loss

    def losses(self):
        result = {}
        with torch.no_grad():
            for gid, batches in self.batches.items():
                weighted = count = 0
                for index, batch in enumerate(batches):
                    size = len(batch["task"])
                    weighted += float(self._loss(gid, index, batch)) * size
                    count += size
                result[gid] = weighted / count
        if not all(np.isfinite(value) for value in result.values()):
            raise ValueError("Non-finite cyclic probe loss.")
        return result

    def gradient(self):
        params = [p for p in self.policy.parameters() if p.requires_grad]
        accumulated = [None] * len(params)
        batches = self.batches[self.grad_group]
        total = sum(len(batch["task"]) for batch in batches)
        for index, batch in enumerate(batches):
            loss = self._loss(self.grad_group, index, batch) * (len(batch["task"]) / total)
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            for i, grad in enumerate(grads):
                if grad is not None:
                    value = grad.detach().to(device="cpu", dtype=torch.bfloat16, copy=True)
                    if accumulated[i] is None:
                        accumulated[i] = value
                    else:
                        accumulated[i].add_(value)
            del grads, loss
        return accumulated

    def measure(self, step, initial=False):
        flags = [(module, module.training) for module in self.policy.modules()]
        self.policy.eval()
        try:
            values = self.losses()
            if not self.own_last:
                self.own_last = dict(values)
            metrics = self.progress(step)
            metrics["probe_loss"] = {f"g{gid}": value for gid, value in values.items()}
            metrics["probe_forget"] = {f"g{gid}": (value - self.own_last[gid]) / max(abs(self.own_last[gid]), 1e-8)
                                       for gid, value in values.items()}
            metrics["probe"] = {**{f"g{gid}_loss": value for gid, value in values.items()},
                                **{f"g{gid}_forget": metrics["probe_forget"][f"g{gid}"] for gid in values}}
            if self.grad_group >= 0:
                current = self.gradient()
                if self.previous_grad is not None:
                    cosine = grad_cosine(self.previous_grad, current)
                    if cosine is not None:
                        metrics["grad_cos"] = {f"g{self.grad_group}": cosine, "interval_steps": step - self.previous_grad_step}
                        metrics["probe"][f"grad_cos_g{self.grad_group}"] = cosine
                self.previous_grad, self.previous_grad_step = current, step
            self.emit(step, metrics)
            # Log BEFORE updating each group's own-phase reference, matching the original experiment.
            if not initial:
                if self.mode == "iid":
                    self.own_last = dict(values)
                else:
                    active = self.active_group(step)
                    self.own_last[active] = values[active]
            logging.info("[cyclic probe] mode=%s step=%d active=%d loss=%s", self.mode, step,
                         self.active_group(step), metrics["probe_loss"])
        finally:
            for module, was_training in flags:
                module.training = was_training

    def emit(self, step, sections):
        with self.metrics_path.open("a") as stream:
            stream.write(json.dumps({"step": step, "mode": self.mode, **sections}, allow_nan=False) + "\n")
        if self.logger:
            for section, metrics in sections.items():
                self.logger.log_dict(metrics, step, mode=section)

    def on_checkpoint(self, directory, step):
        if step != self.step:
            raise RuntimeError("Checkpoint diagnostic step mismatch.")
        # Gradient is reconstructed at the restored weights, not stored as a multi-GB checkpoint copy.
        state = {"step": step, "mode": self.mode, "signature": self.signature,
                 "own_last": self.own_last, "counts": self.counts}
        path = Path(directory) / "cyclic_diagnostics.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(path)


def make_observer_factory(settings):
    def factory(cfg, dataset, policy, preprocessor, accelerator, logger, step):
        return CurriculumObserver(cfg, dataset, policy, preprocessor, accelerator, logger, step, settings)
    return factory
