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


def load_skill_sequences_by_language(dataset_dir: str | Path) -> dict[str, list[list[dict]]]:
    """Per (normalized) language instruction -> list of episodes; each episode is an ordered
    list of skills ``{"token": fsq_code, "gt_length": frames}`` (BOS/EOS/PAD dropped).

    ``gt_length`` is the GT number of frames the skill spanned in the demonstration
    (``skill_length_sequence``, aligned to ``skill_sequence``) — used by the eval to compare
    GT skill-transition timing against the runtime terminator. Episodes ordered by
    episode_index."""
    import json

    import pandas as pd

    dataset_dir = Path(dataset_dir)
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    num_embeddings = int(info["skill_num_embeddings"])  # real FSQ codes < K; BOS/EOS/PAD >= K
    data_files = sorted((dataset_dir / "data").glob("**/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet under {dataset_dir / 'data'}")

    # task_index -> language (meta/tasks.parquet: index=language, col=task_index)
    tasks = pd.read_parquet(dataset_dir / "meta" / "tasks.parquet")
    idx_to_lang = {int(ti): str(lang) for lang, ti in zip(tasks.index, tasks["task_index"])}

    cols = ["episode_index", "frame_index", "task_index", "skill_sequence",
            "skill_length_sequence", "skill_sequence_len"]
    df = pd.concat([pd.read_parquet(p, columns=cols) for p in data_files], ignore_index=True)
    first = df[df["frame_index"] == 0].sort_values("episode_index")

    by_lang: dict[str, list[list[dict]]] = defaultdict(list)
    for _, row in first.iterrows():
        seq = [int(x) for x in np.asarray(row["skill_sequence"]).reshape(-1)]
        lens = [int(x) for x in np.asarray(row["skill_length_sequence"]).reshape(-1)]
        # Drop special tokens by VALUE (real FSQ codes < num_embeddings; BOS/EOS/PAD >= it).
        # Scheme-agnostic: works whether or not the dataset has a leading BOS.
        skills = [{"token": seq[i], "gt_length": lens[i]}
                  for i in range(min(len(seq), len(lens))) if seq[i] < num_embeddings]
        if skills:
            by_lang[_norm_lang(idx_to_lang[int(row["task_index"])])].append(skills)
    return dict(by_lang)


def map_env_tasks_to_skills(
    env_task_languages: dict[int, str],
    sequences_by_language: dict[str, list[list[dict]]],
) -> dict[int, list[list[dict]]]:
    """env task_id -> list of GT skill sequences (from the dataset task with the same
    language); each skill is a ``{"token", "gt_length"}`` dict. Env tasks whose language is
    absent from the dataset are dropped."""
    out: dict[int, list[list[dict]]] = {}
    for task_id, lang in env_task_languages.items():
        seqs = sequences_by_language.get(_norm_lang(lang))
        if seqs:
            out[task_id] = seqs
    return out
