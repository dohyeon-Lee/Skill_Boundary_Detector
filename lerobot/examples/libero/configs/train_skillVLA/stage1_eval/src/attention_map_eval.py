"""Opt-in Arch3/Arch4 attention capture for Stage-1 closed-loop evaluation.

This measures routing attention, not causal pixel attribution: Cond-Gemma
mixes spatial positions and the recurrent bottleneck mixes earlier layers.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from lerobot.policies.skill_expert.layerwise_cond_bottleneck import (
    LayerwiseCondBottleneckSkillExpert,
)

from stage1_eval_config import load_config

TOP_IMAGE = "observation.images.image"
WRIST_IMAGE = "observation.images.wrist_image"


def attention_settings(config_path: str | Path | None) -> dict[str, Any] | None:
    """Read the snapshotted YAML; ordinary Stage-1 eval remains unaffected."""
    if not config_path:
        return None
    settings = load_config(config_path).get("attention_map", {}) or {}
    if not isinstance(settings, dict):
        raise ValueError("attention_map must be a YAML mapping.")
    if not settings.get("enabled", False):
        return None
    result = {
        "max_chunks_per_task": int(settings.get("max_chunks_per_task", 12)),
        "every_n_chunks": int(settings.get("every_n_chunks", 1)),
        "denoise_step": str(settings.get("denoise_step", "last")),
        "expert_layers": str(settings.get("expert_layers", "last")),
    }
    if result["max_chunks_per_task"] <= 0 or result["every_n_chunks"] <= 0:
        raise ValueError("Attention-map chunk counts must be positive.")
    if result["denoise_step"] not in {"first", "middle", "last"}:
        raise ValueError("attention_map.denoise_step must be first|middle|last.")
    if result["expert_layers"] not in {"last", "all"}:
        raise ValueError("attention_map.expert_layers must be last|all.")
    return result


def composed_camera_attention(
    bridge_weights: torch.Tensor, reader_weights: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Average heads/action queries, compose bridge->latent->Cond token maps.

    Both cameras are [CLS, square DINO patch grid].  The composition points
    to *contextualized* Cond-Gemma token positions, not unmixed DINO patches.
    """
    if bridge_weights.ndim != 4 or reader_weights.ndim != 4:
        raise ValueError("Expected attention tensors [batch, heads, queries, keys].")
    bridge = bridge_weights[0].float().mean(dim=(0, 1))
    reader = reader_weights[0].float().mean(dim=0)
    if bridge.numel() != reader.shape[0]:
        raise ValueError("Bridge/reader latent token counts do not match.")
    distribution = (bridge @ reader).clamp_min(0)
    distribution = distribution / distribution.sum().clamp_min(1e-12)
    if distribution.numel() % 2:
        raise ValueError("Arch3 visual token count is not evenly split by camera.")
    camera_tokens = distribution.numel() // 2
    side = math.isqrt(camera_tokens - 1)
    if side * side != camera_tokens - 1:
        raise ValueError(
            f"DINO patch positions are not square: {camera_tokens - 1} per camera."
        )
    top = distribution[:camera_tokens]
    wrist = distribution[camera_tokens:]
    mass = {
        "top_mass": float(top.sum()),
        "wrist_mass": float(wrist.sum()),
        "top_cls_mass": float(top[0]),
        "wrist_cls_mass": float(wrist[0]),
    }
    return (
        top[1:].reshape(side, side).numpy(),
        wrist[1:].reshape(side, side).numpy(),
        mass,
    )


def _rgb(image: torch.Tensor) -> np.ndarray:
    image = image.detach().float().cpu()
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError(f"Expected one RGB image [1,3,H,W], got {tuple(image.shape)}.")
    return (image[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)


def _overlay(rgb: np.ndarray, heatmap: np.ndarray) -> Image.Image:
    base = Image.fromarray(rgb, "RGB")
    values = np.asarray(heatmap, dtype=np.float32)
    # Per-camera contrast is for readability; raw masses are reported beside it.
    values = values / max(float(np.percentile(values, 99)), 1e-12)
    values = np.clip(values, 0, 1)
    color = np.zeros((*values.shape, 3), dtype=np.uint8)
    color[..., 0] = (255 * values).astype(np.uint8)
    color[..., 1] = (80 * np.sqrt(values)).astype(np.uint8)
    tint = Image.fromarray(color, "RGB").resize(base.size, Image.Resampling.BILINEAR)
    alpha = Image.fromarray((values * 210).astype(np.uint8), "L").resize(
        base.size, Image.Resampling.BILINEAR
    )
    return Image.composite(tint, base, alpha)


class AttentionMapCapture:
    """Temporarily request MHA weights during selected policy action chunks."""

    def __init__(self, wrapper, output_dir: Path, settings: dict[str, Any]):
        self.wrapper = wrapper
        self.model = getattr(wrapper.policy, "model", None)
        if not isinstance(self.model, LayerwiseCondBottleneckSkillExpert):
            raise ValueError("Attention-map capture needs an arch3/arch4/arch5/arch6 Stage-1 model.")
        self.output_dir = Path(output_dir)
        self.settings = settings
        self.originals: list[tuple[Any, Any]] = []
        self.active = False
        self.readers: dict[int, torch.Tensor] = {}
        self.bridges: list[torch.Tensor] = []
        self.task = "unknown"
        self.task_chunks = 0
        self.task_saved = 0
        self.last_step = -1
        self.episode = -1
        self.records: list[dict[str, Any]] = []
        self.seen_tasks: set[str] = set()

    def start_task(self, suite: str, task_id: int) -> None:
        self.task = f"{suite}_task{task_id:02d}"
        self.seen_tasks.add(self.task)
        self.task_chunks = 0
        self.task_saved = 0
        self.last_step = -1
        self.episode = -1

    def _patch(self, module, index: int | None) -> None:
        original = module.forward

        def forward(*args, **kwargs):
            if not self.active:
                return original(*args, **kwargs)
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            result, weights = original(*args, **kwargs)
            if weights is None:
                raise RuntimeError("Layerwise bottleneck MHA did not return attention weights.")
            if index is None:
                self.bridges.append(weights.detach())
            else:
                self.readers[index] = weights.detach()
            return result, weights

        module.forward = forward
        self.originals.append((module, original))

    def attach(self) -> None:
        layer_start = self.model.visual_bridge_start_layer
        last_layer = len(self.model.layerwise_condition_readers) - 1
        selected = (
            range(layer_start, last_layer + 1)
            if self.settings["expert_layers"] == "all"
            else [last_layer]
        )
        for index in selected:
            reader = self.model.layerwise_condition_readers[index]
            self._patch(reader.cross_attention, index)
        self._patch(self.model.visual_bridge_attention, None)
        original = self.wrapper.policy.predict_action_chunk

        def predict(batch, **kwargs):
            self.task_chunks += 1
            self.active = (
                self.task_saved < self.settings["max_chunks_per_task"]
                and (self.task_chunks - 1) % self.settings["every_n_chunks"] == 0
            )
            self.readers = {}
            self.bridges = []
            try:
                result = original(batch, **kwargs)
                if self.active:
                    self._save_chunk(batch)
                return result
            finally:
                self.active = False
                self.readers = {}
                self.bridges = []

        self.wrapper.policy.predict_action_chunk = predict
        self.originals.append((self.wrapper.policy, original))

    def detach(self) -> None:
        self.active = False
        for module, original in reversed(self.originals):
            if module is self.wrapper.policy:
                module.predict_action_chunk = original
            else:
                module.forward = original
        self.originals.clear()

    def _save_chunk(self, batch: dict) -> None:
        if not self.bridges:
            raise RuntimeError("No layerwise bridge attention was captured.")
        step = int(self.wrapper._episode_step)
        if step <= self.last_step:
            self.episode += 1
        self.last_step = step
        layer_start = self.model.visual_bridge_start_layer
        n_layers = len(self.model.layerwise_condition_readers) - layer_start
        if len(self.bridges) % n_layers:
            raise RuntimeError("Bridge calls did not form complete denoising steps.")
        n_denoise = len(self.bridges) // n_layers
        denoise_lookup = {"first": 0, "middle": n_denoise // 2, "last": n_denoise - 1}
        denoise = denoise_lookup[self.settings["denoise_step"]]
        layers = (
            range(layer_start, layer_start + n_layers)
            if self.settings["expert_layers"] == "all"
            else [layer_start + n_layers - 1]
        )
        skill = int(torch.as_tensor(batch["skill_code"]).reshape(-1)[0])
        folder = self.output_dir / self.task
        folder.mkdir(parents=True, exist_ok=True)
        identifier = f"ep{self.episode:02d}_step{step:04d}_skill{skill:02d}_chunk{self.task_chunks:03d}"
        top_rgb = _rgb(batch[TOP_IMAGE])
        wrist_rgb = _rgb(batch[WRIST_IMAGE])
        for layer in layers:
            bridge = self.bridges[denoise * n_layers + layer - layer_start].cpu()
            reader = self.readers.get(layer)
            if reader is None:
                raise RuntimeError(f"Missing Cond-Gemma reader attention for layer {layer}.")
            top, wrist, mass = composed_camera_attention(bridge, reader.cpu())
            stem = f"{identifier}_layer{layer + 1:02d}"
            _overlay(top_rgb, top).save(folder / f"{stem}_top.png")
            _overlay(wrist_rgb, wrist).save(folder / f"{stem}_wrist.png")
            np.savez_compressed(folder / f"{stem}.npz", top=top, wrist=wrist)
            self.records.append(
                {
                    "task": self.task,
                    "episode": self.episode,
                    "step": step,
                    "skill": skill,
                    "layer": layer + 1,
                    "denoise_step": denoise + 1,
                    "denoise_steps": n_denoise,
                    "gate_tanh": float(
                        self.model.visual_bridge_gates[layer].detach().tanh().item()
                    ),
                    **mass,
                    "top": f"{self.task}/{stem}_top.png",
                    "wrist": f"{self.task}/{stem}_wrist.png",
                    "map": f"{self.task}/{stem}.npz",
                }
            )
        self.task_saved += 1

    def finish(self) -> None:
        # Each task has exactly one Slurm owner. Shared panel indexes are built
        # separately under the eval.sbatch merge lock after workers finish.
        for task in sorted(self.seen_tasks):
            folder = self.output_dir / task
            folder.mkdir(parents=True, exist_ok=True)
            records = [row for row in self.records if row["task"] == task]
            (folder / "attention_maps.json").write_text(
                json.dumps(records, indent=2), encoding="utf-8"
            )
            rows = []
            for row in records:
                title = (
                    f"{task} · episode {row['episode']} · step {row['step']} · "
                    f"skill {row['skill']} · Expert layer {row['layer']} · "
                    f"denoise {row['denoise_step']}/{row['denoise_steps']}"
                )
                rows.append(
                    "<article><h2>" + html.escape(title) + "</h2>"
                    f"<p>gate tanh={row['gate_tanh']:.4f} · top mass={row['top_mass']:.3f} "
                    f"· wrist mass={row['wrist_mass']:.3f} · "
                    f"top CLS={row['top_cls_mass']:.3f} · wrist CLS={row['wrist_cls_mass']:.3f}</p>"
                    f"<div><figure><img src='{html.escape(Path(row['top']).name)}'><figcaption>VSA top input</figcaption></figure>"
                    f"<figure><img src='{html.escape(Path(row['wrist']).name)}'><figcaption>VSA wrist input</figcaption></figure></div>"
                    f"<p><a href='{html.escape(Path(row['map']).name)}'>raw patch map (.npz)</a></p></article>"
                )
            page = (
                "<!doctype html><html><head><meta charset='utf-8'><title>Layerwise attention maps</title>"
                "<style>body{font:15px system-ui;background:#111827;color:#eee;max-width:1100px;margin:32px auto}"
                "article{padding:18px;border:1px solid #455066;border-radius:12px;margin:20px 0}"
                "article div{display:flex;gap:16px}figure{margin:0}img{width:480px;max-width:100%}"
                "a{color:#93c5fd}</style></head><body><h1>Layerwise vision–action attention</h1>"
                "<p>Expert→latent and same-layer latent→Cond attention are multiplied. "
                "Cond-Gemma spatial mixing and recurrent latents mean this is an approximate "
                "routing view, not causal attribution to raw pixels. Heatmap contrast is "
                "normalized separately for each camera; compare the reported mass values.</p>"
                + ("".join(rows) if rows else "<p>No action chunks were captured for this task.</p>")
                + "</body></html>"
            )
            temporary = folder / "index.html.tmp"
            temporary.write_text(page, encoding="utf-8")
            temporary.replace(folder / "index.html")
