"""Demonstration skill/goal oracle in its own recorded ABC scene.

The oracle supplies sequence order and goal only. It never uses demonstration
boundary times to advance the runtime skill: the learned Terminator does that.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

from joint_overlay import camera_projection, project_xyz


class GTConditions:
    def __init__(self, settings, *, seed, count):
        self.root = Path(settings["gt_dataset"])
        marker = json.loads((self.root / "native_dataset.json").read_text())
        info = json.loads((self.root / "meta/info.json").read_text())
        real_codes = int(np.prod(info["skill_fsq_levels"]))
        self.raw_root = (self.root / marker["raw_root"]).resolve()
        self.sources = marker["sources"]
        with np.load(self.root.parent / "skill_latents.npz", allow_pickle=False) as data:
            self.latents = {key: data[key].copy() for key in ("episode_id", "skill_index", "frame_start", "frame_end", "tokens")}
        with np.load(self.root.parent / "skill_spatial_targets.npz", allow_pickle=False) as data:
            for key, latent_key in (("skill_episode_id", "episode_id"), ("skill_index", "skill_index"),
                                    ("frame_start", "frame_start"), ("frame_end", "frame_end")):
                if not np.array_equal(data[key], self.latents[latent_key]):
                    raise ValueError(f"GT spatial target identity differs from skill_latents: {key}")
            self.end_xyz = data["skill_end_xyz"].copy()
        self.rows = {}
        for row, episode in enumerate(self.latents["episode_id"]):
            episode = int(episode)
            if Path(self.sources[episode]["path"]).parts[0] == "grab_clutter":
                self.rows.setdefault(episode, []).append(row)
        for episode, rows in self.rows.items():
            rows.sort(key=lambda row: int(self.latents["frame_start"][row]))
            codes = self.latents["tokens"][rows]
            if np.any(codes < 0) or np.any(codes >= real_codes) or not np.isfinite(self.end_xyz[rows]).all():
                raise ValueError(f"invalid GT skill/XYZ sequence for episode {episode}")
        selected = settings.get("gt_episode_ids")
        if selected is None:
            if count > len(self.rows):
                raise ValueError(f"requested {count} GT episodes but only {len(self.rows)} are available")
            selected = np.random.default_rng(seed).choice(sorted(self.rows), size=count, replace=False).tolist()
        if len(selected) != count or any(episode not in self.rows for episode in selected):
            raise ValueError("gt_episode_ids must contain exactly num_worlds valid grab_clutter episode IDs")
        self.episode_ids = list(map(int, selected))

    def episode_dir(self, world_index):
        return self.raw_root / self.sources[self.episode_ids[world_index]]["path"]

    def environment_kwargs(self, world_index):
        # Resolve old absolute asset prefixes and relative asset names without
        # changing the process cwd or altering any recorded scene geometry.
        import abc_sim
        models = Path(abc_sim.__file__).resolve().parent / "models"
        assets = models / "assets"
        xml = (self.episode_dir(world_index) / "scene_assembled.xml").read_text()
        xml = re.sub(r'"/[^\"]*?/models/assets(?=[/\"])', '"' + str(assets), xml)
        root = ET.fromstring(xml)
        for element in root.iter():
            filename = element.get("file")
            if filename:
                path = Path(filename)
                candidates = [path] if path.is_absolute() else [assets / path, models / path]
                # Recorded grab_clutter XML uses the shared put_relative object
                # library path. Its task-specific asset download installs the
                # same category/variant files under task_grab_clutter instead.
                relocated = filename.replace("task_put_relative/objects/", "task_grab_clutter/objects/")
                if relocated != filename:
                    alternative = Path(relocated)
                    candidates.extend([alternative] if alternative.is_absolute() else [assets / alternative, models / alternative])
                resolved = next((candidate for candidate in candidates if candidate.is_file()), None)
                if resolved is None:
                    raise FileNotFoundError(f"recorded ABC scene asset missing: {filename}")
                element.set("file", str(resolved.resolve()))
        compiler = root.find("compiler")
        if compiler is not None:
            compiler.set("meshdir", str(assets))
            compiler.set("texturedir", str(models))
        # grab_clutter's factory otherwise randomizes/replaces the supplied
        # model in prepare_env(), even before reset(randomize=False).
        return {"scene_xml_string": ET.tostring(root, encoding="unicode"), "enable_task_randomizer": False}

    def restore(self, env, world_index, seed):
        import mujoco
        from abc_sim.env import _GRIPPER_CTRL_MAX
        from abc_sim.randomization.core import RandomizationState
        folder = self.episode_dir(world_index)
        metadata = json.loads((folder / "episode_metadata.json").read_text())
        env.reset(seed=seed, randomize=False)
        # Frame zero is the native state/action alignment used for training.
        # All object and both-finger qpos values are restored, not just 14D arms.
        qpos = np.load(folder / "scene_qpos.npy", mmap_mode="r")[0]
        if qpos.shape != (env.model.nq,) or not np.isfinite(qpos).all():
            raise ValueError(f"recorded qpos/model mismatch: {folder}")
        env.data.qpos[:] = qpos
        env.data.qvel[:] = 0  # Recorded release contains qpos, not initial velocities.
        env.data.time = 0
        state = env.get_state().copy()
        state[env._gripper_indices] *= _GRIPPER_CTRL_MAX
        env.data.ctrl[env._ctrl_indices] = state
        env.prompt = str(metadata["instruction"])
        env._last_randomization = RandomizationState.from_dict(json.loads((folder / "randomization.json").read_text()))
        mujoco.mj_forward(env.model, env.data)
        env._task_evaluator.reset(nworld=1)
        return env.get_obs()

    def skill_count(self, world_index):
        return len(self.rows[self.episode_ids[world_index]])

    def condition(self, world_index, order, env, *, height, width):
        episode = self.episode_ids[world_index]
        row = self.rows[episode][order]
        xyz = self.end_xyz[row]
        pixels, valid = project_xyz(xyz, camera_projection(env.model, env.data, "top", height, width))
        uv = 2 * pixels / np.array([width - 1, height - 1]) - 1
        uv = np.where(valid[:, None], uv, 0).clip(-1, 1).astype(np.float32)
        return int(self.latents["tokens"][row]), uv, valid, {
            "gt_episode_id": episode, "gt_skill_index": int(self.latents["skill_index"][row]),
            "gt_frame_start": int(self.latents["frame_start"][row]),
            "gt_frame_end": int(self.latents["frame_end"][row]), "gt_end_xyz": xyz.tolist(),
        }
