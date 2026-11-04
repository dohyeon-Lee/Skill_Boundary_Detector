"""Opt-in, privileged Task0 condition ablation; never an oracle success claim.

Skill order follows a successful demonstration, advanced only at accepted learned
boundaries. UV is a demo EEF endpoint rigidly retargeted from an object-local
frame, NOT the object centre and NOT per-rollout ground truth.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

PROTOCOL = "task0_demo_retarget_boundary_v1"
MODES = {"baseline", "skill_only", "uv_only", "skill_uv"}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference(path, *, code_space=None, vocabulary=None, fsq_path=None):
    path = Path(path)
    data = json.loads(path.read_text())
    if data.get("protocol") != PROTOCOL or data.get("task_id") != 0:
        raise ValueError("Task0 reference has an unsupported protocol/task.")
    if code_space is not None and data.get("skill_code_space_id") != code_space:
        # Legacy builds did not save a code-space name. Never infer equivalence
        # from similar run names: require the exact FSQ checkpoint bytes.
        if (data.get("skill_code_space_id") is not None or fsq_path is None
                or data.get("fsq_sha256") != file_sha256(fsq_path)):
            raise ValueError("Task0 reference/checkpoint FSQ code-space mismatch.")
    goals = (data["first_goal"], data["second_goal"])
    if any(len(g) != 3 or g[0] != "in" for g in goals) or goals[0][1] == goals[1][1] or goals[0][2] != goals[1][2]:
        raise ValueError("Task0 requires two distinct objects in the same basket.")
    if not data.get("phases"):
        raise ValueError("Task0 reference has no second-object phases.")
    container = data["container_object"]
    if not isinstance(container, str) or not (goals[0][2] == container or goals[0][2].startswith(container + "_")):
        raise ValueError("Task0 container does not match the BDDL goal region.")
    for phase in data["phases"]:
        code = phase["skill_code"]
        xyz = np.asarray(phase["eef_end_local_xyz"], dtype=float)
        if type(code) is not int or code < 0 or (vocabulary is not None and code >= vocabulary):
            raise ValueError("Invalid Task0 reference FSQ code.")
        if phase["anchor"] not in {goals[1][1], container} or xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("Invalid Task0 reference EEF endpoint/anchor.")
    return data


def reference_identity(path):
    return {"protocol": PROTOCOL, "reference_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def scene_context(domain, *, task_id, height, width):
    """Read simulator geometry only: no step/reset/render or RNG use."""
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    predicates = []
    for raw in domain.parsed_problem.get("goal_state", []):
        predicates.append({"state": [str(v) for v in raw], "value": bool(domain._eval_predicate(raw))})
    objects = {}
    for name, body_id in domain.obj_body_id.items():
        objects[str(name)] = {"position": domain.sim.data.body_xpos[body_id].copy().tolist(),
                              "rotation": domain.sim.data.body_xmat[body_id].copy().reshape(3, 3).tolist()}
    grasps = {
        str(name): bool(domain._check_grasp(domain.robots[0].gripper, obj))
        for name, obj in domain.objects_dict.items()
    }
    return {"task_id": int(task_id), "predicates": predicates, "objects": objects, "grasps": grasps,
            "eef_world": domain.sim.data.site_xpos[domain.robots[0].eef_site_id].copy().tolist(),
            "height": int(height), "width": int(width),
            "camera_transform": get_camera_transform_matrix(domain.sim, "agentview", height, width).tolist()}


def goal_value(snapshot, goal):
    matches = [p["value"] for p in snapshot["predicates"] if p["state"] == goal]
    if len(matches) != 1:
        raise ValueError(f"Task0 goal is missing/ambiguous: {goal}")
    return bool(matches[0])


def summarize_interventions(episodes):
    triggered = [ep for ep in episodes if ep["triggered"]]
    intervened = [ep for ep in episodes if ep["intervened"]]
    return {"episodes": len(episodes), "triggered": len(triggered), "intervened": len(intervened),
            "triggered_success_pct": 100 * sum(ep["success"] for ep in triggered) / len(triggered) if triggered else None,
            "intervened_success_pct": 100 * sum(ep["success"] for ep in intervened) / len(intervened) if intervened else None,
            "interpretation": "Report full-task SR AND intervention coverage; untriggered episodes are not second-can trials."}


class Task0Intervention:
    def __init__(self, reference, *, mode, stable_steps=3):
        if mode not in MODES or type(stable_steps) is not int or stable_steps < 1:
            raise ValueError("Invalid Task0 intervention mode/stable_steps.")
        self.reference, self.mode, self.stable_steps = reference, mode, stable_steps
        self.reset()

    def reset(self):
        self.snapshot = None
        self.stable_count = 0
        self.trigger_step = None
        self.phase_index = -1
        self.events = []
        self.rejections = []

    def observe(self, snapshots, step):
        if len(snapshots) != 1 or snapshots[0]["task_id"] != 0:
            raise ValueError("Task0 intervention requires LIBERO task0, eval_batch_size=1.")
        snapshot = snapshots[0]
        first = self.reference["first_goal"]
        if first[1] not in snapshot["grasps"]:
            raise ValueError("Task0 intervention requires a first-object grasp signal.")
        second_done = goal_value(snapshot, self.reference["second_goal"])
        complete = goal_value(snapshot, first) and not snapshot["grasps"][first[1]] and not second_done
        self.stable_count = self.stable_count + 1 if complete else 0
        if self.trigger_step is None and self.stable_count >= self.stable_steps:
            self.trigger_step = int(step)
        self.snapshot = snapshot

    def propose(self, predicted_code, predicted_uv, step, *, current_code=None):
        if self.trigger_step is None or goal_value(self.snapshot, self.reference["second_goal"]):
            return None
        from lerobot.policies.skillVLA.focus_projection import project_eef

        index = min(self.phase_index + 1, len(self.reference["phases"]) - 1)
        # Stability may be confirmed just after the first second-can skill
        # has already started. At its next learned boundary, do not replay it.
        aligned_existing = (self.phase_index < 0 and len(self.reference["phases"]) > 1
                            and current_code == self.reference["phases"][0]["skill_code"])
        if aligned_existing:
            index = 1
        phase = self.reference["phases"][index]
        anchor = self.snapshot["objects"][phase["anchor"]]
        rotation, position = np.asarray(anchor["rotation"]), np.asarray(anchor["position"])
        if (rotation.shape != (3, 3) or position.shape != (3,) or not np.isfinite(rotation).all()
                or not np.isfinite(position).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)):
            raise ValueError("Invalid Task0 anchor pose.")
        endpoint = position + rotation @ np.asarray(phase["eef_end_local_xyz"])
        point = project_eef(endpoint, np.asarray(self.snapshot["camera_transform"]),
                            height=self.snapshot["height"], width=self.snapshot["width"])
        raw_uv = np.asarray(predicted_uv, dtype=float)
        if raw_uv.shape != (2,) or not np.isfinite(raw_uv).all():
            raise ValueError("Invalid predicted Task0 UV.")
        result = {"mode": self.mode, "phase_index": index, "reference_skill_index": phase["skill_index"],
                "aligned_existing_first_phase": aligned_existing,
                "episode_step": int(step), "trigger_step": self.trigger_step,
                "predicted_skill": int(predicted_code), "predicted_uv": raw_uv.tolist(),
                "reference_skill": phase["skill_code"], "reference_uv": list(point.normalized_xy),
                "reference_uv_clipped": point.clipped, "reference_eef_world": endpoint.tolist(),
                "anchor": phase["anchor"], "anchor_pose": copy.deepcopy(anchor),
                "applied_skill": phase["skill_code"] if self.mode in {"skill_only", "skill_uv"} else int(predicted_code),
                "applied_uv": list(point.normalized_xy) if self.mode in {"uv_only", "skill_uv"} else raw_uv.tolist()}
        result["skill_changed"] = result["applied_skill"] != result["predicted_skill"]
        result["uv_changed"] = not np.allclose(result["applied_uv"], raw_uv, atol=1e-6, rtol=0)
        return result

    def commit(self, proposal):
        if proposal is not None:
            self.phase_index = proposal["phase_index"]
            self.events.append(copy.deepcopy(proposal))

    def reject(self, proposal):
        if proposal is not None:
            self.rejections.append({"episode_step": proposal["episode_step"],
                                    "phase_index": proposal["phase_index"], "reason": "same_skill_transition"})

    def report(self):
        return {"protocol": PROTOCOL, "mode": self.mode,
                "interpretation": "Demo-reference condition ablation, not per-rollout GT or a deployment score.",
                "trigger_step": self.trigger_step, "triggered": self.trigger_step is not None,
                "accepted_boundaries": len(self.events),
                "intervened": bool(self.events) and self.mode != "baseline",
                "changed_boundaries": sum(event["skill_changed"] or event["uv_changed"] for event in self.events),
                "first_boundary_delay": self.events[0]["episode_step"] - self.trigger_step if self.events else None,
                "events": copy.deepcopy(self.events), "rejections": copy.deepcopy(self.rejections)}
