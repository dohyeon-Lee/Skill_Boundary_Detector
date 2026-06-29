"""Stage-1 oracle-eval helpers (kept out of the model to keep the policy clean).

These support closed-loop sim eval of the SkillExpert with a GT skill sequence per task:
  - FsqTerminator: the frozen FSQ terminator (skill code -> z_q, and per-step
    progress/termination from [z, current 3rd-person image, current state]).
  - skill-sequence loading per task, keyed by LANGUAGE instruction.
  - env(task_id) <-> dataset(task_index) mapping by language (LIBERO_90 languages are
    NOT unique per task: ~12 are shared by 2 scenes, so this mapping is by language only).
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def _norm_lang(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


# FSQ keys SplineFSQAE accepts (mirrors modeling_skillVLA.load_vae_decoder).
_FSQ_KEYS = {
    "action_dim", "enc_dim", "state_dim", "n_control", "spline_degree", "hidden_dim", "fsq_levels",
    "num_layers", "dropout", "length_min", "length_max", "action_min", "action_max", "delta_min", "delta_max", "state_min", "state_max",
    "feat_dim", "n_tokens", "image_encoder_layers", "image_encoder_heads", "terminator_use_third", "terminator_use_wrist",
    "image_model_name", "image_size", "patch_grid", "n_patch_raw", "image_token_dim", "chunk_size",
}


class FsqTerminator:
    """Frozen FSQ terminator used only at eval to advance the GT skill sequence."""

    def __init__(self, fsq_path: str | Path, device: torch.device, *, dino_path: str | None = None,
                 libero_examples_dir: str | Path | None = None):
        import dataclasses

        if libero_examples_dir is not None:
            sys.path.insert(0, str(libero_examples_dir))
        from FSQ import SplineFSQAE  # noqa: PLC0415

        ckpt = torch.load(str(fsq_path), map_location="cpu", weights_only=False)
        cfg = ckpt["cfg"]
        cfg_dict = dataclasses.asdict(cfg)
        if dino_path:
            cfg_dict["image_model_name"] = dino_path
        vae = SplineFSQAE(**{k: v for k, v in cfg_dict.items() if k in _FSQ_KEYS})
        vae.load_state_dict(ckpt["model_state"])
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        self.vae = vae.to(device)
        self.device = device
        self.state_dim = int(cfg.state_dim)

        levels = [int(x) for x in cfg.fsq_levels]
        strides = [1]
        for i in range(1, len(levels)):
            strides.append(strides[-1] * levels[i - 1])
        self.register_levels = levels
        self._strides = torch.tensor(strides, dtype=torch.long)
        self._levels = torch.tensor(levels, dtype=torch.long)
        self._half = torch.tensor([(l - 1) / 2.0 for l in levels], dtype=torch.float32)
        self.num_codes = int(np.prod(levels))

    def code_to_z(self, codes: torch.Tensor) -> torch.Tensor:
        """(B,) FSQ code -> (B, D) z_q grid vector."""
        idx = codes.view(-1, 1).long().cpu()
        level_ids = torch.div(idx, self._strides[None, :], rounding_mode="floor") % self._levels[None, :]
        z = level_ids.to(torch.float32) - self._half[None, :]
        return z.to(self.device)

    @property
    def use_wrist(self) -> bool:
        return bool(getattr(self.vae, "terminator_use_wrist", False))

    @torch.no_grad()
    def terminate(self, codes: torch.Tensor, state: torch.Tensor, image: torch.Tensor,
                  wrist_image: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """codes (B,), state (B, state_dim) raw, image / wrist_image (B,C,H,W) raw current frames.
        wrist_image is used only by a dual-camera terminator (ignored / may be None for single).
        Returns (progress (B,), termination_prob (B,))."""
        z = self.code_to_z(codes)
        state = state.to(self.device, torch.float32)[:, : self.state_dim].unsqueeze(1)  # (B,1,state_dim)
        image = image.to(self.device)
        if wrist_image is not None:
            wrist_image = wrist_image.to(self.device)
        progress, term = self.vae.predict_termination(z, state, image, wrist_image, quantize=True)
        return progress[:, 0], term[:, 0]


def make_terminator(path: str | Path, device: torch.device, *, dino_path: str | None,
                    libero_examples_dir: str | Path):
    """Load the FSQ terminator (FSQ.pt) for eval — provides the skill-end signal (progress + termination)."""
    return FsqTerminator(path, device, dino_path=dino_path, libero_examples_dir=libero_examples_dir)


def build_task_name_to_id(suite_name: str) -> dict[str, int]:
    """LIBERO suite task .name -> task_id. The task name (e.g. 'KITCHEN_SCENE10_close_the_top_drawer')
    equals the original HDF5 ``scene_file`` with '_demo.hdf5' stripped, so it uniquely identifies the
    env scene (disambiguating the ~12 languages shared across two scenes in LIBERO_90)."""
    from libero.libero import benchmark  # noqa: PLC0415

    suite = benchmark.get_benchmark_dict()[suite_name]()
    return {str(t.name): i for i, t in enumerate(suite.tasks)}


def load_episode_oracle_data(
    skill_dataset_dir: str | Path,
    init_states_path: str | Path,
    suite_name: str,
    *,
    resample_n: int,
    spline_degree: int,
) -> dict[int, list[dict]]:
    """Per-EPISODE oracle eval data, grouped by LIBERO task_id and ordered by episode_index.

    Joins the skillvla dataset (GT skill sequence + per-frame state) with eval_init_states.npz
    (episode -> MuJoCo init_state + scene_file). Each record::

        {"episode_index": int,
         "init_state":  np.float64 (state_dim,)             — episode-exact env reset state,
         "skills": [{"token": fsq_code,
                     "gt_length": frames,                   — GT skill duration (timing + length token),
                     "state_traj": np.float32 (resample_n, state_dim)}, ...]}  — Oracle r input.

    ``state_traj`` is the skill's full state trajectory resampled EXACTLY as training
    (_spline_resample), so feeding it to the Oracle reproduces the training-time r (the oracle-r
    upper bound). Special tokens (code >= skill_num_embeddings) are dropped. Only episodes present in
    BOTH the dataset and the npz, with >=1 real skill, are kept."""
    import json  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    from lerobot.policies.skill_expert.dataset_skill_expert import STATE_KEY, _spline_resample  # noqa: PLC0415

    skill_dataset_dir = Path(skill_dataset_dir)
    info = json.loads((skill_dataset_dir / "meta" / "info.json").read_text())
    num_emb = int(info["skill_num_embeddings"])  # real FSQ codes < num_emb; BOS/EOS/PAD >= it

    npz = np.load(str(init_states_path), allow_pickle=True)
    inits = {int(e): (st, str(sf)) for e, st, sf in
             zip(npz["episode_index"], npz["init_states"], npz["scene_file"])}
    name_to_id = build_task_name_to_id(suite_name)

    cols = ["episode_index", "frame_index", STATE_KEY, "skill_sequence", "skill_length_sequence"]
    data_files = sorted((skill_dataset_dir / "data").glob("**/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet under {skill_dataset_dir / 'data'}")
    df = pd.concat([pd.read_parquet(p, columns=cols) for p in data_files], ignore_index=True)

    by_task: dict[int, list[dict]] = defaultdict(list)
    for ep, ep_df in df.groupby("episode_index"):
        ep = int(ep)
        if ep not in inits:
            continue
        init_state, scene_file = inits[ep]
        task_name = scene_file[: -len("_demo.hdf5")] if scene_file.endswith("_demo.hdf5") else scene_file
        task_id = name_to_id.get(task_name)
        if task_id is None:
            continue
        ep_df = ep_df.sort_values("frame_index")
        states = np.stack([np.asarray(s, np.float32) for s in ep_df[STATE_KEY].values])  # (T, state_dim)
        row0 = ep_df.iloc[0]
        seq = np.asarray(row0["skill_sequence"]).reshape(-1)
        lens = np.asarray(row0["skill_length_sequence"]).reshape(-1)
        offsets = np.concatenate([[0], np.cumsum(lens)])  # frame boundary before skill i
        skills = []
        for i in range(min(len(seq), len(lens))):
            if int(seq[i]) >= num_emb:  # special token
                continue
            s, e = int(offsets[i]), min(int(offsets[i] + lens[i]), len(states))
            if e <= s:
                continue
            skills.append({
                "token": int(seq[i]),
                "gt_length": int(lens[i]),
                "state_traj": _spline_resample(states[s:e], resample_n, spline_degree).astype(np.float32),
            })
        if skills:
            by_task[task_id].append(
                {"episode_index": ep, "init_state": np.asarray(init_state, np.float64), "skills": skills})

    for tid in by_task:
        by_task[tid].sort(key=lambda r: r["episode_index"])
    return dict(by_task)
