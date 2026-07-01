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
                 libero_examples_dir: str | Path | None = None, finetuned_ckpt: str | Path | None = None):
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
        # Optionally override the terminator submodules with the CO-TRAINED (fine-tuned) terminator saved
        # in a SkillExpert checkpoint — the DINO backbone above stays from dino_path.
        self.finetuned_loaded = _load_finetuned_terminator(vae, finetuned_ckpt) if finetuned_ckpt else 0
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


def _load_finetuned_terminator(vae, ckpt_dir: str | Path) -> int:
    """Override the terminator submodule weights of `vae` (an FSQ.pt-built SplineFSQAE) with the CO-TRAINED
    terminator saved in a SkillExpert checkpoint (model.fsq_term_train.*, fine-tuned on the dataset via
    train_terminator). Only the intersecting terminator/reconstructor keys are replaced — the DINO backbone
    stays from dino_path (it isn't stored in the checkpoint). Returns the number of overridden tensors
    (0 → the checkpoint has no co-trained terminator, so the caller keeps the raw FSQ.pt terminator)."""
    from safetensors import safe_open  # noqa: PLC0415

    st_path = Path(ckpt_dir) / "model.safetensors"
    if not st_path.exists():
        return 0
    vsd = set(vae.state_dict().keys())
    override = {}
    with safe_open(str(st_path), framework="pt") as h:
        for k in h.keys():
            if "fsq_term_train." in k:
                sub = k.split("fsq_term_train.", 1)[1]
                if sub in vsd:
                    override[sub] = h.get_tensor(k)
    if override:
        vae.load_state_dict(override, strict=False)  # strict=False: DINO backbone keys stay from FSQ.pt build
    return len(override)


def make_terminator(path: str | Path, device: torch.device, *, dino_path: str | None,
                    libero_examples_dir: str | Path, finetuned_ckpt: str | Path | None = None):
    """The FSQ terminator (FSQ.pt). terminator_path (eval config) may point it at a DIFFERENT run's FSQ.pt
    to compare terminators — the codebook (code→z_q) is the same frozen FSQ, so any FSQ.pt drops straight in.
    When finetuned_ckpt is given, the terminator submodules are overridden with that checkpoint's CO-TRAINED
    (fine-tuned) terminator; the number actually loaded is on the returned obj's `.finetuned_loaded`."""
    return FsqTerminator(path, device, dino_path=dino_path, libero_examples_dir=libero_examples_dir,
                         finetuned_ckpt=finetuned_ckpt)


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
) -> dict[int, list[dict]]:
    """Per-EPISODE eval data, grouped by LIBERO task_id and ordered by episode_index — for EPISODE-EXACT
    closed-loop eval (each rollout reproduces a specific dataset episode's scene).

    Joins the skillvla dataset (GT skill sequence per episode) with eval_init_states.npz (episode ->
    MuJoCo init_state + scene_file). Each record::

        {"episode_index": int,
         "init_state": np.float64 (state_dim,),          — episode-exact env reset state (MuJoCo),
         "skills": [{"token": fsq_code, "gt_length": frames}, ...]}   — the episode's GT skill sequence.

    scene_file (minus '_demo.hdf5') == the LIBERO task .name → a unique task_id (build_task_name_to_id).
    Special tokens (code >= skill_num_embeddings, i.e. BOS/EOS/PAD) are dropped. Only episodes present in
    BOTH the dataset and the npz, with >=1 real skill, are kept."""
    import json  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    skill_dataset_dir = Path(skill_dataset_dir)
    info = json.loads((skill_dataset_dir / "meta" / "info.json").read_text())
    num_emb = int(info["skill_num_embeddings"])  # real FSQ codes < num_emb; BOS/EOS/PAD >= it

    npz = np.load(str(init_states_path), allow_pickle=True)
    inits = {int(e): (st, str(sf)) for e, st, sf in
             zip(npz["episode_index"], npz["init_states"], npz["scene_file"])}
    name_to_id = build_task_name_to_id(suite_name)

    cols = ["episode_index", "frame_index", "skill_sequence", "skill_length_sequence"]
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
        row0 = ep_df.sort_values("frame_index").iloc[0]
        seq = np.asarray(row0["skill_sequence"]).reshape(-1)
        lens = np.asarray(row0["skill_length_sequence"]).reshape(-1)
        skills = [{"token": int(seq[i]), "gt_length": int(lens[i])}
                  for i in range(min(len(seq), len(lens))) if int(seq[i]) < num_emb]
        if skills:
            by_task[task_id].append({"episode_index": ep, "skills": skills,
                                     "init_state": np.asarray(init_state, np.float64)})

    for tid in by_task:
        by_task[tid].sort(key=lambda r: r["episode_index"])
    return dict(by_task)


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
