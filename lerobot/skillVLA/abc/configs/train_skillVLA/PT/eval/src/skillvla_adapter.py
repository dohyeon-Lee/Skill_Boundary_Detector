"""Closed-loop ABC SkillVLA: predicted FSQ + two UV goals, VSA, Term_Token_1."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from joint_overlay import JointPlanFK, denormalize_plan, draw_frame
from skillvla_config import CAMERA_KEYS


@dataclass
class WorldSkill:
    code: int = -1
    order: int = -1
    age: int = 0
    step: int = 0
    needs_skill: bool = True
    probability: float = 0.0
    uv: np.ndarray | None = None
    valid: np.ndarray | None = None
    start_state: np.ndarray | None = None
    xyz: np.ndarray | None = None
    events: list = field(default_factory=list)
    terminations: list = field(default_factory=list)
    plans: list = field(default_factory=list)
    auxiliaries: dict = field(default_factory=dict)
    auxiliary_points: list = field(default_factory=list)

    def activate(self, code, uv, state, valid=None):
        uv, state = np.asarray(uv, dtype=np.float32).reshape(2, 2), np.asarray(state, dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all() or not np.isfinite(uv).all():
            raise ValueError("Skill activation needs finite two-arm UV and 14D start state")
        previous = self.code
        self.code, self.order, self.age = int(code), self.order + 1, 0
        self.needs_skill, self.probability = False, 0.0
        self.uv, self.start_state, self.xyz = uv.clip(-1, 1).copy(), state.copy(), None
        self.valid = np.ones(2, dtype=bool) if valid is None else np.asarray(valid, dtype=bool).reshape(2).copy()
        self.auxiliaries = {}
        self.events.append({"step": self.step, "skill_order": self.order, "previous_code": previous,
                            "skill_code": self.code, "focus_uv": self.uv.tolist(), "start_state": state.tolist()})

    def advance(self, probability, *, threshold, minimum, maximum):
        self.step += 1
        self.age += 1
        self.probability = float(probability)
        if not np.isfinite(self.probability):
            raise ValueError("Terminator returned a non-finite probability")
        self.terminations.append((self.step, self.order, self.probability))
        reason = ("terminator" if self.age >= minimum and self.probability >= threshold else
                  "timeout" if self.age >= maximum else None)
        if reason:
            self.needs_skill = True
            self.events[-1].update(end_step=self.step, end_reason=reason, end_probability=self.probability)
        return self.needs_skill


class SkillVLAPolicyAdapter:
    def __init__(self, config):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy
        from lerobot.policies.skill_aux.modeling_skill_aux import SkillAuxPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from safetensors.torch import load_model
        try:
            from transformers.initialization import no_init_weights
        except ImportError:  # Transformers 4.x
            from transformers.modeling_utils import no_init_weights

        self.config, self.settings, self.torch = config, config.skillvla, torch
        self.device = torch.device(config.policy_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("ABC SkillVLA requested CUDA but no GPU is available")
        torch.manual_seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)

        self.gt = None
        if self.settings.get("condition_source") == "gt":
            from gt_conditions import GTConditions
            self.gt = GTConditions(self.settings, seed=config.seed, count=config.num_worlds)

        self.models, self.preprocessors, self.postprocessors = {}, {}, {}
        for component, cls in (("vsa", SkillExpertPolicy), ("predictor", SkillAuxPolicy), ("terminator", SkillAuxPolicy)):
            if component == "predictor" and self.settings.get("condition_source") == "gt":
                continue
            path = Path(self.settings[component])
            print(f"Loading ABC SkillVLA {component}: {path}", flush=True)
            model_config = PreTrainedConfig.from_pretrained(path)
            model_config.device = str(self.device)
            model_config.dtype = config.policy_dtype if self.device.type == "cuda" else "float32"
            model_config.pretrained_path = path
            if hasattr(model_config, "gradient_checkpointing"):
                model_config.gradient_checkpointing = False
            if component == "vsa":
                model_config.n_action_steps = config.action_chunk_size
                model_config.num_inference_steps = config.policy_num_inference_steps
                if not model_config.omits_start_vision:
                    raise ValueError("ABC runtime currently supports the trained current-vision dual-bottleneck VSA")
                if self.settings["fullskill_overlay"] and not (
                    model_config.uses_independent_skill_planner and model_config.skill_planner_mode == "denoising"
                ):
                    raise ValueError("fullskill_overlay needs a denoising full-skill planner")
            elif component == "predictor":
                if model_config.skill_predictor_mode != "mode2" or not model_config.skill_predictor_focus_uv_enabled:
                    raise ValueError("ABC runtime requires Predictor mode2 with two-arm UV output")
            elif model_config.terminator_architecture_label != "term_token_1_dinos_freeze":
                raise ValueError("ABC runtime currently supports Term_Token_1_dinoS_freeze")
            with no_init_weights():
                model = cls(model_config, **({"_skip_auxiliary_warm_start": True} if component != "vsa" else {}))
            # load_model handles tied VLM weights, and raises on ANY missing or
            # unexpected tensor. No legacy pi0.5 initialization fallback.
            load_model(model, str(path / "model.safetensors"), strict=True, device="cpu")
            model = model.to(self.device).eval().requires_grad_(False)
            self.models[component] = model
            if component != "terminator":
                pre, post = make_pre_post_processors(
                    model_config, pretrained_path=str(path),
                    preprocessor_overrides={"device_processor": {"device": str(self.device)}},
                    postprocessor_overrides={"device_processor": {"device": str(self.device)}},
                )
                self.preprocessors[component], self.postprocessors[component] = pre, post
        self.vsa = self.models["vsa"]
        self.predictor = self.models["predictor"].model.skill_predictor if "predictor" in self.models else None
        self.terminator = self.models["terminator"].model.fsq_term_train
        self.reset()

    def reset(self):
        self.worlds, self.fk, self.gt_world_indices = {}, {}, {}
        for model in self.models.values():
            model.reset()
        for processor in (*self.preprocessors.values(), *self.postprocessors.values()):
            processor.reset()

    def describe(self):
        return {"type": "skillvla", "checkpoint": self.settings["vsa"],
                "predictor_checkpoint": self.settings.get("predictor"), "terminator_checkpoint": self.settings["terminator"],
                "execution_chunk_size": self.config.action_chunk_size, "skillvla": self.settings,
                "condition_source": self.settings.get("condition_source", "predictor"),
                "evaluation_protocol": ("recorded training episodes, GT FSQ sequence and endpoint goals, learned termination"
                                        if self.gt is not None else "new randomized scenes, predicted skills and goals"),
                "gt_episode_ids": self.gt.episode_ids if self.gt is not None else None,
                "action_space": "absolute_joint_position_14d", "fullskill_visualization": "joint14 -> private FK -> two grasp-site XYZ -> live camera pixels"}

    def world_env_kwargs(self, task, world_index):
        return self.gt.environment_kwargs(world_index) if self.gt is not None else {}

    def reset_world(self, env, world_index, world_id):
        seed = self.config.seed + world_index
        if self.gt is not None:
            self.gt_world_indices[world_id] = world_index
            return self.gt.restore(env, world_index, seed)
        options = None if self.config.randomization is None else {"randomization": self.config.randomization}
        observation, _ = env.reset(seed=seed, options=options, randomize=True)
        return observation

    def world_done(self, world_id):
        if getattr(self, "gt", None) is None:
            return False
        c = self.worlds[world_id]
        return c.needs_skill and c.order + 1 >= self.gt.skill_count(self.gt_world_indices[world_id])

    def _autocast(self):
        if self.device.type == "cuda" and self.config.policy_dtype == "bfloat16":
            return self.torch.autocast("cuda", dtype=self.torch.bfloat16)
        return nullcontext()

    def _raw_batch(self, observation, rows=None):
        torch = self.torch
        state = np.asarray(observation["state"], dtype=np.float32)
        if rows is None:
            rows = list(range(len(state)))
        batch = {"observation.state": torch.as_tensor(state[rows].copy()),
                 "task": [observation["prompt"][i] for i in rows]}
        for camera, key in zip(("top", "left", "right"), CAMERA_KEYS, strict=True):
            images = np.asarray(observation["images"][camera])[rows]
            if images.ndim != 4 or images.shape[1] != 3:
                raise ValueError(f"ABC {camera} must be uint8 [B,3,H,W]")
            batch[key] = torch.as_tensor(images.copy()).float() / 255
        return batch

    def infer_for_worlds(self, observation, world_ids, environments):
        torch = self.torch
        with torch.inference_mode(), self._autocast():
            contexts = [self.worlds.setdefault(i, WorldSkill()) for i in world_ids]
            new_rows = [row for row, context in enumerate(contexts) if context.needs_skill]
            if new_rows and getattr(self, "gt", None) is not None:
                for row in new_rows:
                    context, env = contexts[row], environments[row]
                    code, uv, valid, identity = self.gt.condition(
                        self.gt_world_indices[world_ids[row]], context.order + 1, env,
                        height=self.config.camera_height, width=self.config.camera_width,
                    )
                    context.activate(code, uv, observation["state"][row], valid)
                    context.events[-1].update(identity, condition_source="gt")
                    print(f"  world={world_ids[row]} step={context.step} GT skill#{context.order} code={code}", flush=True)
            elif new_rows:
                batch = self.preprocessors["predictor"](self._raw_batch(observation, new_rows))
                previous = torch.tensor([contexts[i].code for i in new_rows], device=self.device)
                codes, uv = self.predictor.predict_focus_uv(
                    [batch[key] for key in CAMERA_KEYS], batch["observation.language.tokens"],
                    batch["observation.language.attention_mask"], previous_skill_code=previous,
                )
                for row, code, goal in zip(new_rows, codes.cpu().tolist(), uv.float().cpu().numpy(), strict=True):
                    contexts[row].activate(code, goal, observation["state"][row])
                    print(f"  world={world_ids[row]} step={contexts[row].step} skill#{contexts[row].order} code={code}", flush=True)
            raw = self._raw_batch(observation)
            raw.update(skill_code=torch.tensor([c.code for c in contexts]),
                       skill_focus_uv=torch.from_numpy(np.stack([c.uv for c in contexts])),
                       skill_focus_valid=torch.from_numpy(np.stack([c.valid for c in contexts])),
                       skill_start_state=torch.from_numpy(np.stack([c.start_state for c in contexts])),
                       skill_index=torch.tensor([c.order for c in contexts]))
            batch = self.preprocessors["vsa"](raw)
            # Current-only VSA has no recurrent state. Explicitly clear batch
            # caches: active world rows can shrink or advance at different times.
            self.vsa.reset()
            from lerobot.policies.skill_expert.bottleneck_video import capture_bottleneck, decode_auxiliaries
            with capture_bottleneck(getattr(self.vsa, "model", None), getattr(self.config, "save_video", False)) as capture:
                normalized = self.vsa.predict_action_chunk(batch)
            if capture is not None:
                decoded = decode_auxiliaries(capture["outputs"], self.vsa.config)
                for row, context in enumerate(contexts):
                    context.auxiliaries = {
                        tower: {name: value[row].copy() for name, value in values.items()}
                        for tower, values in decoded.items()
                    }
                    context.auxiliary_points.append({
                        "step": context.step, "skill_order": context.order,
                        "towers": {tower: {name: value.tolist() for name, value in values.items()}
                                   for tower, values in context.auxiliaries.items()},
                    })
            actions = self.postprocessors["vsa"](normalized).float().cpu().numpy()
            if actions.ndim != 3 or actions.shape[0] != len(world_ids) or actions.shape[2] != 14 or not np.isfinite(actions).all():
                raise ValueError(f"VSA returned invalid actions: {actions.shape}")
            if self.settings["fullskill_overlay"]:
                devices = [self.device.index or 0] if self.device.type == "cuda" else []
                # Visualization is a diagnostic denoising chain; never consume
                # the RNG state that will generate the next control action.
                with torch.random.fork_rng(devices=devices):
                    seed = (self.config.seed + 0x46554C4C + sum(c.step for c in contexts)) % (2**63 - 1)
                    torch.default_generator.manual_seed(seed)
                    if devices:
                        with torch.cuda.device(self.device):
                            torch.cuda.manual_seed(seed)
                    plans = self.vsa.predict_independent_plan_for_visualization(batch).float().cpu().numpy()
                for world_id, context, env, plan in zip(world_ids, contexts, environments, plans, strict=True):
                    if plan.shape != (self.vsa.config.bspline_tokens, 14):
                        raise ValueError(f"unexpected full-skill plan shape {plan.shape}")
                    joints = denormalize_plan(plan, self.vsa.config.chunk_end_state_q01, self.vsa.config.chunk_end_state_q99)
                    if world_id not in self.fk:
                        self.fk[world_id] = JointPlanFK(env)
                    xyz = self.fk[world_id].trajectory(env.data, joints)
                    context.xyz = xyz
                    context.plans.append((context.step, context.order, joints, xyz))
            return actions[:, :self.config.action_chunk_size]

    def observe_step(self, world_id, observation):
        torch, context = self.torch, self.worlds[world_id]
        batched = {"state": np.asarray(observation["state"])[None], "prompt": [observation["prompt"]],
                   "images": {k: np.asarray(v)[None] for k, v in observation["images"].items()}}
        batch = self._raw_batch(batched)
        with torch.inference_mode():
            # Terminator was trained in float32, with RAW (not VSA-normalized)
            # joint proprio and frozen DINO-S. Goal and start pose stay latched.
            cameras = [batch[key].to(self.device) for key in CAMERA_KEYS]
            codes = torch.tensor([context.code], device=self.device)
            zq = self.models["terminator"]._code_to_zq(codes)
            _, logits = self.terminator(
                zq, batch["observation.state"].to(self.device), cameras[0], torch.stack(cameras[1:], dim=1),
                goal_uv=torch.as_tensor(context.uv.reshape(1, 4), device=self.device),
                start_state=torch.as_tensor(context.start_state[None], device=self.device),
            )
            probability = logits.float().sigmoid().item()
        ended = context.advance(probability, threshold=self.settings["termination_threshold"],
                                minimum=self.settings["min_skill_steps"], maximum=self.settings["max_skill_steps"])
        # Admit another evaluation only after Predictor/VSA/plan and Terminator
        # have all completed their first real inference, including CUDA setup.
        from lerobot.utils.gpu_memory_guard import mark_ready

        mark_ready()
        if context.step % 60 == 0:
            print(f"  world={world_id} frame={context.step}/{self.config.max_steps} skill={context.code} age={context.age} term={probability:.3f}", flush=True)
        return ended

    def video_frame(self, world_id, env, images, cameras):
        c = self.worlds[world_id]
        label = f"skill {c.code} / #{c.order} | step {c.step} | term {c.probability:.2f} | arms L cyan/R orange | aux U coral/L purple"
        return draw_frame(env, images, cameras, c.xyz, c.uv, label,
                          auxiliaries=c.auxiliaries, goal_valid=c.valid)

    def finish_world(self, world_id, trace_base):
        c = self.worlds[world_id]
        path = Path(trace_base).with_suffix(".json")
        if c.events and "end_step" not in c.events[-1]:
            c.events[-1].update(end_step=c.step, end_reason="episode_end")
        path.write_text(json.dumps({"skills": c.events, "termination_trace": c.terminations,
                                    "bottleneck_points": c.auxiliary_points}, indent=2) + "\n")
        result = {"skill_trace_path": str(path), "num_skills": len(c.events)}
        if getattr(self, "gt", None) is not None:
            result.update(condition_source="gt", gt_episode_id=self.gt.episode_ids[self.gt_world_indices[world_id]],
                          gt_sequence_exhausted=self.world_done(world_id))
        if c.plans:
            plan_path = Path(trace_base).with_suffix(".npz")
            np.savez_compressed(plan_path, frame=np.array([p[0] for p in c.plans]),
                                skill_order=np.array([p[1] for p in c.plans]),
                                joint_plan=np.stack([p[2] for p in c.plans]),
                                grasp_xyz=np.stack([p[3] for p in c.plans]))
            result["fullskill_plan_path"] = str(plan_path)
        return result
