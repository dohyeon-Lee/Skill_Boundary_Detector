"""SkillVLA training/inference facade using native GR00T weights and state tokens."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy
from lerobot.policies.skill_expert.vlm_termination import vlm_termination_loss
from lerobot.policies.skill_expert.wrist_patch_alignment import wrist_patch_alignment_loss
from lerobot.policies.skill_expert.wrist_patch_target import camera_for_batch, patch_labels
from lerobot.policies.skillVLA.dataset_skillVLA import SKILL_FOCUS_UV
from lerobot.utils.training_performance import cached_training_target, reuse_training_targets, scalar_metric
from .model import GrootSkillModel


class GrootSkillExpertPolicy(SkillExpertPolicy):
    def __init__(self, config, **kwargs):
        PreTrainedPolicy.__init__(self, config)
        if config.skillvla_backbone not in {"groot_integrate", "groot_seperate"}:
            raise ValueError("GR00T SkillVLA requires groot_integrate or groot_seperate")
        if config.visual_bottleneck_tokens != 100:
            raise ValueError("Both GR00T bottleneck designs use 100 tokens per tower")
        self.config = config
        self.newtask_dsbc_head = None
        self.model = GrootSkillModel(config).float().to(config.device)
        self.reset()
        self.model.vlm.configure_checkpointing(
            llm=config.gradient_checkpointing if config.vlm_checkpointing is None else config.vlm_checkpointing,
            vision=config.gradient_checkpointing if config.vision_checkpointing is None else config.vision_checkpointing)

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, config=None, strict=True, **kwargs):
        source = json.loads((Path(pretrained_name_or_path) / "config.json").read_text())
        if source.get("type") == "skill_expert":
            if source.get("skillvla_backbone", "pi05") not in {"groot_integrate", "groot_seperate"}:
                raise ValueError("Cannot load a pi0.5 SkillVLA checkpoint as GR00T")
            if config is not None and source["skillvla_backbone"] != config.skillvla_backbone:
                raise ValueError("GR00T integrate/seperate checkpoints cannot be interchanged")
            return PreTrainedPolicy.from_pretrained.__func__(cls, pretrained_name_or_path, config=config, strict=True, **kwargs)
        if config is None:
            raise ValueError("Initializing from GR00T base requires a SkillExpertConfig")
        # Constructor performs native strict pretrained loading; new modules alone are random.
        return cls(config, **kwargs)

    def _inputs(self, batch):
        images = [batch[key] for key in self.config.skill_image_keys]
        images = [x.permute(0, 3, 1, 2) if x.shape[-1] == 3 else x for x in images]
        # skill_code is the dataset's coherent post-randomization selection;
        # skill_code_true describes the original frame and may differ from
        # the selected UV/fullskill/termination condition near a boundary.
        return images, batch["observation.state"], self._skill_code(batch), batch[SKILL_FOCUS_UV]

    def _visual_loss(self, batch):
        cache = self.model._camera_memory
        device = batch["observation.state"].device
        goal_valid = batch["skill_end_state_valid"].to(device).reshape(-1).bool()
        top = cache.uv(batch[SKILL_FOCUS_UV])
        raw = batch["skill_decoder_state"].float()
        # Use the same physical wrist projection and validity as pi0.5 SkillVLA,
        # then map it through GR00T's camera crop before assigning patch cells.
        grid = math.isqrt(self.model._aux[0][1].shape[-1])
        _, pixel, wrist_valid = patch_labels(raw[:, :3], raw[:, 3:6], batch["skill_end_state"][:, :3],
            camera_for_batch(batch), grid=grid, height=224, width=224)
        wrist = cache.uv(2 * pixel / 224 - 1, camera=1)
        total, metrics = 0., {}
        for tower, (top_logits, wrist_logits, xyz) in enumerate(self.model._aux):
            loss = xyz.sum() * 0
            for name, logits, uv, valid in (("top", top_logits, top, goal_valid),
                                            ("wrist", wrist_logits, wrist, wrist_valid & goal_valid)):
                grid = math.isqrt(logits.shape[-1])
                def labels():
                    usable = valid & torch.isfinite(uv).all(-1) & (uv.abs() <= 1).all(-1)
                    xy = ((uv.nan_to_num().clamp(-1, 1) + 1) * .5 * grid).floor().long().clamp(0, grid - 1)
                    return xy[:, 1] * grid + xy[:, 0], usable

                # The cache dies after this forward. Each subsequent native or
                # extra Top crop therefore gets newly transformed UV/labels.
                cell, valid = cached_training_target(cache, ('groot_patch_labels', name, grid), labels)
                term, values = wrist_patch_alignment_loss(logits, cell, valid,
                    grid=grid, sigma=self.config.wrist_patch_align_target_sigma, prefix=name)
                loss = loss + self.config.wrist_patch_align_loss_weight * term
                metrics.update({f"vision_aux/{tower}/{key}": value for key, value in values.items()})
            xyz_loss, xyz_metrics = self._agent_end_state_alignment_loss(batch, xyz, top_k=1)
            loss = loss + self.config.agent_end_state_loss_weight * xyz_loss
            metrics.update({f"vision_aux/{tower}/{key}": value for key, value in xyz_metrics.items()})
            total = total + .5 * loss
        return total, metrics

    @reuse_training_targets()
    def forward(self, batch, reduction="mean"):
        from lerobot.policies.skill_expert.vsa_top_crop import prepare_top_crop
        batch, self._vsa_top_crop = prepare_top_crop(batch, self.config, training=self.training)
        images, state, code, uv = self._inputs(batch)
        banks, uv = self.model.observe(images, code, uv)
        state_token, ids = self.model.state_token(state)
        actions = batch["action"][:, :self.config.chunk_size]
        error = self.model.flow_loss(actions, state_token, ids, code, uv, banks)
        valid = self._valid_action_steps(actions, batch)
        count = valid.sum(-1).clamp_min(1)
        per_sample = (error * valid[:, :, None]).sum((1, 2)) / (count * actions.shape[-1])
        metrics = {"action_loss": scalar_metric(per_sample.detach().mean())}
        loss = per_sample.mean()
        if not self.config.newtask_ft_enabled or self.config.newtask_ft_skill_flow_loss:
            target = self._prepare_independent_skill_plan(batch, code, None)
            plan_error = self.model.flow_loss(target, state_token, ids, code, uv, banks, planner=True)
            plan = plan_error.mean((1, 2))
            per_sample = per_sample + self.config.bspline_loss_weight * plan
            loss = loss + self.config.bspline_loss_weight * plan.mean()
            metrics["bspline/loss"] = scalar_metric(plan.detach().mean())
        if not self.config.newtask_ft_enabled or self.config.newtask_ft_visual_aux_loss:
            visual, values = self._visual_loss(batch)
            loss = loss + visual
            per_sample = per_sample + visual
            metrics.update(values)
        if self.config.vlm_termination_enabled and self.config.vlm_termination_loss_weight > 0:
            term, values = vlm_termination_loss(self.model, self.config, batch, state)
            per_sample = per_sample + self.config.vlm_termination_loss_weight * term
            loss = loss + self.config.vlm_termination_loss_weight * term.mean()
            metrics.update(values)
        self.model._aux = None
        self.model._camera_memory = None
        metrics["loss"] = scalar_metric(loss.detach())
        return (per_sample if reduction == "none" else loss), metrics

    @torch.no_grad()
    def predict_action_chunk(self, batch, **kwargs):
        from lerobot.policies.skill_expert.vsa_top_crop import prepare_top_crop
        batch, self._vsa_top_crop = prepare_top_crop(batch, self.config, training=False)
        images, state, code, uv = self._inputs(batch)
        banks, uv = self.model.observe(images, code, uv)
        state_token, ids = self.model.state_token(state)
        actions = torch.randn(state.shape[0], self.config.chunk_size, 7, device=state.device)
        steps = self.model.native_config.num_inference_timesteps
        for index in range(steps):
            time = torch.full((state.shape[0],), int(index / steps * self.model.native_config.num_timestep_buckets), device=state.device, dtype=torch.long)
            actions = actions + self.model.velocity(actions, state_token, ids, time, code, uv, banks) / steps
        return actions

    @torch.no_grad()
    def predict_termination_probability(self, batch):
        from lerobot.policies.skill_expert.vsa_top_crop import prepare_top_crop
        batch, self._vsa_top_crop = prepare_top_crop(batch, self.config, training=False)
        images, state, code, uv = self._inputs(batch)
        self.model.observe(images, code, uv)
        logits, _ = self.model.take_vlm_termination_outputs()
        self.model._aux = self.model._camera_memory = None
        return logits.sigmoid()

    def get_optim_params(self):
        return [{"params": [p for p in self.parameters() if p.requires_grad], "group_name": "skillvla_groot"}]

    def isolated_main_optimizer_grad_groups(self):
        branch = self.model.vlm_termination_bottleneck
        params = [p for p in branch.parameters() if p.requires_grad] if branch is not None else []
        return {"vlm_termination": params} if params else {}

    def parameter_counts(self):
        return {"total": sum(p.numel() for p in self.parameters()), "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad)}

    def training_debug_metrics(self):
        return {}

    def named_component_params(self):
        return {"vlm": list(self.model.vlm.named_parameters()), "expert": list(self.model.expert.named_parameters())}
