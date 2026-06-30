#!/usr/bin/env python3
"""Standalone FSQ-terminator trainer (Stage-1 sibling).

Co-trains ONLY the FSQ terminator (progress + termination heads) on the skillvla dataset's GT signals,
warm-started from the run's FSQ.pt. This is the FT terminator co-training (modeling_skillVLA.py) lifted
out of the policy: a DISJOINT model (no expert / no VLM) that you can run once to get a libero_90
terminator — the idea being a *dedicated post-FSQ* terminator is cleaner than the FSQ-jointly-trained one
(no codebook-formation objective competing). The output is an FSQ.pt-format checkpoint (cfg + model_state)
so it drops straight into the eval as a terminator (same loader as the original FSQ.pt).

Inputs per frame come from the SAME dataset the FT terminator co-train uses (no bespoke dataset):
  SkillVLADataset (skill_code_true, skill_ds, skill_de, observation.state)
    wrapped by SkillVLADinoTokenDataset (skill_decoder_dino = precomputed current-frame 3rd-person tokens).
Targets: progress = ds/(ds+de); termination = exp(-de^2/2σ^2) (soft) — mirrors the FSQ terminator.

Run:  src/train.sbatch   (or: python src/train_terminator.py --config ../terminator_train_config.yaml)
Out:  {outputs_root}/skillVLA_stage1/terminator/{run_name}/checkpoints/{step}/FSQ.pt
"""
from __future__ import annotations

import argparse
import dataclasses
import re
import struct
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import numpy.lib.format as _npf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve()
# shared yaml helpers (load_config merges global_config.yaml → project_root/dataset_root/outputs_root)
sys.path.insert(0, str(_HERE.parents[4] / "train_skills" / "src"))
from train_skills_config import as_bool, get_value, load_config  # noqa: E402

# FSQ module (SplineFSQAE + its config dataclass, needed to unpickle FSQ.pt["cfg"])
_LIBERO = _HERE.parents[5]  # …/examples/libero
sys.path.insert(0, str(_LIBERO))
from FSQ import SplineFSQAE  # noqa: E402

# (lerobot's LeRobotDataset is imported lazily inside TerminatorDataset)

# The terminator-path submodules (warm-started, trainable); everything else (FSQ encoder, reconstructor)
# stays frozen. Mirrors modeling_skillVLA._TERM_TRAIN_MODULES.
_TERM_MODULES = ("dec_z_proj", "term_state_proj", "dec_image_encoder_term",
                 "dec_image_encoder_term_wrist", "term_pool", "progress_head", "termination_head")
# Fields the SplineFSQAE constructor accepts (filter the pickled cfg dataclass). Mirrors _construct_fsq.
_FSQ_CFG_KEYS = {"action_dim", "enc_dim", "state_dim", "n_control", "spline_degree", "hidden_dim", "fsq_levels",
                 "num_layers", "dropout", "length_min", "length_max", "action_min", "action_max", "delta_min", "delta_max", "state_min", "state_max",
                 "feat_dim", "n_tokens", "image_encoder_layers", "terminator_use_third", "terminator_use_wrist",
                 "image_encoder_heads", "image_model_name", "image_size",
                 "patch_grid", "n_patch_raw", "image_token_dim", "chunk_size", "reconstructor_mode"}
_NEEDED = ("skill_code_true", "skill_ds", "skill_de", "skill_decoder_dino", "observation.state")


def _open_dino_mmap(tokens_path: str):
    """Memmap dino.npz's UNCOMPRESSED features.npy in place (instant — no ~28GB .features.npy cache build)
    + build the (episode, frame)→row map. Returns (features memmap, frame_to_row)."""
    tp = Path(tokens_path)
    info = zipfile.ZipFile(tp).getinfo("features.npy")
    if info.compress_type != 0:
        raise RuntimeError("dino.npz features.npy is COMPRESSED — can't memmap (rebuild uncompressed).")
    meta = np.load(str(tp), mmap_mode="r", allow_pickle=False)  # small index members
    try:
        offsets = meta["offsets"].astype(np.int64)
        episode_ids = meta["episode_id"].astype(np.int64)
        frame_starts = meta["frame_start"].astype(np.int64)
        lengths = meta["length"].astype(np.int64)
    finally:
        meta.close()
    with open(tp, "rb") as f:                                   # locate the raw .npy bytes inside the zip
        f.seek(info.header_offset)
        fnl, efl = struct.unpack("<HH", f.read(30)[26:30])      # local-header name/extra lengths
        f.seek(info.header_offset + 30 + fnl + efl)
        ver = _npf.read_magic(f)
        shape, fortran, dtype = _npf._read_array_header(f, ver)
        arr_off = f.tell()
    features = np.memmap(str(tp), dtype=dtype, mode="r", offset=arr_off, shape=shape,
                         order="F" if fortran else "C")
    frame_to_row: dict[tuple[int, int], int] = {}
    for si, (ep, fs, length) in enumerate(zip(episode_ids, frame_starts, lengths, strict=True)):
        start = int(offsets[si])
        for j in range(int(length)):
            frame_to_row[(int(ep), int(fs) + j)] = start + j
    return features, frame_to_row


class TerminatorDataset(torch.utils.data.Dataset):
    """Lean, NO-VIDEO terminator dataset. Reads ONLY the parquet columns the terminator needs
    (observation.state, skill_ds, skill_de, skill_index, skill_sequence→skill_code_true) and the current
    frame's DINO tokens (npz mmapped) — skipping SkillVLADataset's per-item video decode (the terminator
    uses none of the raw images). Restricted to skill-covered frames (those with real DINO tokens), which is
    exactly the FSQ terminator's training set. Same inputs/targets as the FT co-train → same result, faster."""

    def __init__(self, repo_id: str, root: str, dino_path: str, output_key: str = "skill_decoder_dino"):
        self.output_key = output_key
        self.features, frame_to_row = _open_dino_mmap(dino_path)             # always (instant)
        # The per-frame index (state + targets + dino-row, already filtered to skill frames) is small but
        # SLOW to build from the parquet (~17 min over 569k rows), so cache it. Reuse → instant (no parquet).
        cache = Path(dino_path).with_name("dino.terminator_index.npz")
        if cache.exists():
            z = np.load(str(cache))
            self.state, self.skill_ds, self.skill_de = z["state"], z["ds"], z["de"]
            self.code_true, self._rows = z["code_true"], z["rows"]
        else:
            print("[terminator] building lean index from parquet (ONE-TIME, ~17 min over 569k frames; "
                  f"cached → {cache.name}, instant after)…", flush=True)
            from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415
            hf = LeRobotDataset(repo_id, root=root).hf_dataset            # parquet (Arrow); no video decode
            state = np.asarray(hf["observation.state"], dtype=np.float32)
            ds = np.asarray(hf["skill_ds"], dtype=np.float32)
            de = np.asarray(hf["skill_de"], dtype=np.float32)
            ep = np.asarray(hf["episode_index"], dtype=np.int64)
            fr = np.asarray(hf["frame_index"], dtype=np.int64)
            sk_idx = np.asarray(hf["skill_index"], dtype=np.int64)
            seqs = list(hf["skill_sequence"])                               # per-frame skill-code sequence
            code = np.array([int(np.asarray(seqs[i]).reshape(-1)[sk_idx[i]]) for i in range(len(ep))],
                            dtype=np.int64)                                 # un-jittered current code = ss[k]
            valid = np.array([i for i in range(len(ep))                     # skill-covered frames (have DINO)
                              if (int(ep[i]), int(fr[i])) in frame_to_row], dtype=np.int64)
            self._rows = np.array([frame_to_row[(int(ep[i]), int(fr[i]))] for i in valid], dtype=np.int64)
            self.state, self.skill_ds, self.skill_de = state[valid], ds[valid], de[valid]
            self.code_true = code[valid]
            np.savez(str(cache), state=self.state, ds=self.skill_ds, de=self.skill_de,
                     code_true=self.code_true, rows=self._rows)
            print(f"[terminator] cached lean index ({len(self._rows)} frames) → {cache.name}", flush=True)

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, j):
        return {
            "observation.state": torch.from_numpy(self.state[j].copy()),
            "skill_ds": torch.tensor(float(self.skill_ds[j])),
            "skill_de": torch.tensor(float(self.skill_de[j])),
            "skill_code_true": torch.tensor(int(self.code_true[j])),
            self.output_key: torch.from_numpy(np.asarray(self.features[int(self._rows[j])], dtype=np.float32)),
        }


def build_terminator(fsq_path: str, device: torch.device, warm_start: bool = True):
    """Load FSQ.pt → SplineFSQAE with ONLY the terminator-path submodules trainable. warm_start=True warm-
    starts those from the FSQ checkpoint (REFINE the FSQ terminator — but it starts at FSQ quality, so the
    loss barely moves); warm_start=False keeps their fresh random init (a dedicated FROM-SCRATCH terminator
    on the frozen codebook) — the real test of "is a dedicated terminator better than the FSQ-jointly-trained
    one". The codebook (fsq.*) + reconstructor are always loaded. Returns (fsq, cfg, trainable_names)."""
    ckpt = torch.load(fsq_path, map_location="cpu", weights_only=False)
    cfg_dict = dataclasses.asdict(ckpt["cfg"])
    fsq = SplineFSQAE(**{k: v for k, v in cfg_dict.items() if k in _FSQ_CFG_KEYS})
    trainable = {n for n in _TERM_MODULES if getattr(fsq, n, None) is not None}
    model_keys = set(fsq.state_dict().keys())
    state = {k: v for k, v in ckpt["model_state"].items() if k in model_keys}
    if not warm_start:  # from-scratch: keep the terminator modules' random init (load everything else)
        state = {k: v for k, v in state.items() if k.split(".", 1)[0] not in trainable}
    missing, _ = fsq.load_state_dict(state, strict=False)
    bad = [k for k in missing if k.split(".", 1)[0] not in trainable]  # missing terminator keys OK (random init)
    if bad:
        raise RuntimeError(f"FSQ checkpoint missing required (non-terminator) weights: {sorted(bad)}")
    for name, mod in fsq.named_children():
        for p in mod.parameters():
            p.requires_grad_(name in trainable)
    fsq.train()
    return fsq.to(device=device, dtype=torch.float32), ckpt["cfg"], sorted(trainable)


class CodeToZ:
    """Flat FSQ code → z_q (B, D) in the codebook coordinate frame. Mirrors SkillVLA._code_to_z."""

    def __init__(self, levels: list[int], device: torch.device):
        self.levels = torch.tensor(levels, dtype=torch.long, device=device)
        strides = torch.ones_like(self.levels)
        for i in range(1, len(levels)):
            strides[i] = strides[i - 1] * self.levels[i - 1]
        self.strides = strides
        self.half = (self.levels - 1).float() / 2.0

    def __call__(self, code: torch.Tensor) -> torch.Tensor:
        idx = code.view(-1, 1).long()
        level_ids = torch.div(idx, self.strides[None, :], rounding_mode="floor") % self.levels[None, :]
        return level_ids.float() - self.half[None, :]


def term_forward(fsq, code2z: CodeToZ, true_code, state, dino):
    """GT skill code + current state + current 3rd-person DINO tokens → (progress (B,), term_logits (B,)).
    Replicates SkillVLAPytorch.terminator_predict (wrist=None; this FSQ is terminator_use_wrist=False)."""
    dev = next(fsq.parameters()).device
    st = state.to(device=dev, dtype=torch.float32)
    if st.ndim == 2:
        st = st.unsqueeze(1)                                   # (B, 1, state_dim)
    st = st[..., : int(fsq.state_dim)]
    z = code2z(true_code.to(dev)).to(device=dev, dtype=st.dtype)
    dec = fsq._prepare_decoder_tokens(dino.to(dev), states=st)  # (B, 1, N, F)
    B, T = dec.shape[:2]
    lh = fsq.fsq.levels_half.to(z.device, z.dtype)
    zq = torch.maximum(torch.minimum(torch.round(z), lh), -lh)
    z_tok = fsq.dec_z_proj(zq.unsqueeze(1).expand(B, T, -1).to(st.dtype))
    progress, term_logits = fsq._terminate(z_tok, st, dec, None)  # (B, 1), (B, 1)
    return progress[:, 0], term_logits[:, 0]


def collate(items):
    return {k: torch.stack([torch.as_tensor(it[k]) for it in items]) for k in _NEEDED}


def save_ckpt(out_dir: Path, step: int, fsq, fsq_cfg, optimizer) -> Path:
    # optimizer_state + step let training RESUME; the eval loader (FsqTerminator) ignores those keys.
    blob = {"cfg": fsq_cfg, "model_state": {k: v.cpu() for k, v in fsq.state_dict().items()},
            "optimizer_state": optimizer.state_dict(), "step": step}
    d = out_dir / "checkpoints" / f"{step:06d}"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "FSQ.pt"  # FSQ.pt format → drop-in terminator for eval
    torch.save(blob, path)
    last = out_dir / "checkpoints" / "last"
    last.mkdir(parents=True, exist_ok=True)
    torch.save(blob, last / "FSQ.pt")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=_HERE.parents[1] / "terminator_train_config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    # ── paths (mirror FT resolver: run_dir = {dataset_root}/{skillvla_root}/{source}/{run_tag}) ──
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    dataset_root = project_root / str(get_value(cfg, "dataset_root", "dataset"))
    skillvla_root = dataset_root / str(get_value(cfg, "skillvla_dataset_root", "skillvla_dataset"))
    source = str(get_value(cfg, "source_dataset"))
    run_tag = str(get_value(cfg, "run_tag"))
    run_dir = skillvla_root / source / run_tag
    fsq_path = run_dir / "FSQ.pt"
    dino_path = run_dir / "dino.npz"
    skillvla_dir = run_dir / "skillvla"
    repo_id = f"dohyeon/{source}"
    outputs_root = project_root / str(get_value(cfg, "outputs_root", "outputs"))
    if not skillvla_dir.exists():                               # FSQ.pt/dino.npz checked per-arch below
        raise FileNotFoundError(f"missing skillvla/: {skillvla_dir}")

    # ── hyperparams ──
    steps = int(get_value(cfg, "steps", 10000))
    batch_size = int(get_value(cfg, "batch_size", 256))
    lr = float(get_value(cfg, "lr", 1.0e-4))
    num_workers = int(get_value(cfg, "num_workers", 16))
    save_freq = int(get_value(cfg, "save_freq", 2500))
    log_freq = int(get_value(cfg, "log_freq", 50))
    sigma = float(get_value(cfg, "terminator_end_target_sigma", 1.0))   # soft termination target std (frames)
    pos_weight = float(get_value(cfg, "terminator_end_pos_weight", 1.0))
    w_prog = float(get_value(cfg, "progress_loss_weight", 1.0))          # loss = w_prog·prog + w_term·term
    w_term = float(get_value(cfg, "termination_loss_weight", 1.0))       # (FSQ's progress_loss_weight/end_loss_weight)
    warm_start = as_bool(get_value(cfg, "warm_start", True))             # False = FROM-SCRATCH terminator
    exp = str(get_value(cfg, "exp", "")).strip()
    wandb_enable = as_bool(get_value(cfg, "wandb_enable", True))
    wandb_project = str(get_value(cfg, "wandb_project", "VLA_terminator"))
    # arch: "fsq" = refine/from-scratch the FSQ terminator (lean precomputed tokens); "custom" = a NEW
    # architecture (raw images → frozen DINO → full-token transformer + AdaRMS), built from the toggles.
    arch = str(get_value(cfg, "terminator_arch", "fsq")).strip().lower()
    use_third = as_bool(get_value(cfg, "use_third", True))
    use_wrist = as_bool(get_value(cfg, "use_wrist", True))
    use_state = as_bool(get_value(cfg, "use_state", True))
    custom_dino = str(get_value(cfg, "custom_dino", "s")).strip().lower()   # s|b|l
    custom_pool = as_bool(get_value(cfg, "custom_pool_8x8", True))          # 8×8 pool (FSQ-style) vs full grid
    custom_dim = int(get_value(cfg, "custom_dim", 256))
    custom_layers = int(get_value(cfg, "custom_layers", 3))
    custom_heads = int(get_value(cfg, "custom_heads", 4))

    run_name = f"{source}_{run_tag}_term_bs{batch_size}_sig{sigma:g}_pw{pos_weight:g}"
    if arch == "custom":                          # arch + dino + toggles + pool → own folder
        run_name += (f"_custom_{custom_dino}_t{int(use_third)}w{int(use_wrist)}s{int(use_state)}"
                     f"_p{'8' if custom_pool else 'full'}")
    elif not warm_start:                          # from-scratch (fsq arch) → its own folder
        run_name = f"{run_name}_scratch"
    if w_prog != 1.0 or w_term != 1.0:            # non-1:1 ratio → its own folder
        run_name = f"{run_name}_lp{w_prog:g}lt{w_term:g}"
    if exp:
        run_name = f"{run_name}_{exp}"
    out_dir = outputs_root / "skillVLA_stage1" / "terminator" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    levels_m = re.search(r"FSQ(\d+)", run_tag)
    levels = [int(d) for d in levels_m.group(1)] if levels_m else [5, 5, 5]
    vocab = int(np.prod(levels))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[terminator] run_dir={run_dir}\n[terminator] out={out_dir}\n[terminator] device={device} "
          f"levels={levels} steps={steps} bs={batch_size} lr={lr} σ={sigma} pos_w={pos_weight} "
          f"loss=({w_prog:g}·prog + {w_term:g}·term) warm_start={warm_start}")

    # ── model + data + per-arch forward/save (loss is shared) ──
    if arch == "custom":
        from custom_terminator import CustomTerminator  # noqa: PLC0415  (sibling module)
        model = CustomTerminator(vocab=vocab, state_dim=8, dino_size=custom_dino, patch_pool_8x8=custom_pool,
                                 use_third=use_third, use_wrist=use_wrist, use_state=use_state,
                                 dim=custom_dim, layers=custom_layers, heads=custom_heads,
                                 project_root=str(project_root)).to(device)
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[terminator] CUSTOM arch (dino={custom_dino} pool8={custom_pool} 3rd={use_third} "
              f"wrist={use_wrist} state={use_state}; {n_tr:,} trainable)", flush=True)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        from lerobot.policies.skillVLA.dataset_skillVLA import SkillVLADataset  # noqa: PLC0415
        print("[terminator] loading skillvla dataset (RAW images; video decode)…", flush=True)
        dataset = SkillVLADataset(repo_id, root=str(skillvla_dir))
        raw_keys = ["skill_code_true", "skill_ds", "skill_de", "observation.state"]
        if use_third:
            raw_keys.append("observation.images.image")
        if use_wrist:
            raw_keys.append("observation.images.wrist_image")

        def collate_fn(items):
            return {k: torch.stack([torch.as_tensor(it[k]) for it in items]) for k in raw_keys}

        def forward_fn(batch):
            sk = batch["skill_code_true"].view(-1).long().clamp(0, vocab - 1).to(device)
            st = batch["observation.state"].float().to(device)
            i3 = batch["observation.images.image"].to(device) if use_third else None
            iw = batch["observation.images.wrist_image"].to(device) if use_wrist else None
            return model(sk, st, i3, iw)

        custom_cfg = {"vocab": vocab, "state_dim": 8, "dino_size": custom_dino, "patch_pool_8x8": custom_pool,
                      "use_third": use_third, "use_wrist": use_wrist, "use_state": use_state,
                      "dim": custom_dim, "layers": custom_layers, "heads": custom_heads}

        net = model

        def save_fn(step):
            # drop the frozen DINO (reloaded from custom_dino at eval); +optimizer/step for resume.
            sd = {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("tok.dino.")}
            blob = {"arch": "custom", "cfg": custom_cfg, "model_state": sd,
                    "optimizer_state": optimizer.state_dict(), "step": step}
            for sub in (f"{step:06d}", "last"):
                d = out_dir / "checkpoints" / sub
                d.mkdir(parents=True, exist_ok=True)
                torch.save(blob, d / "terminator.pt")
            return out_dir / "checkpoints" / f"{step:06d}" / "terminator.pt"
    else:  # fsq
        for p, name in [(fsq_path, "FSQ.pt"), (dino_path, "dino.npz")]:
            if not p.exists():
                raise FileNotFoundError(f"missing {name}: {p}")
        fsq, fsq_cfg, trainable = build_terminator(str(fsq_path), device, warm_start=warm_start)
        n_tr = sum(p.numel() for p in fsq.parameters() if p.requires_grad)
        print(f"[terminator] FSQ arch, trainable modules={trainable} ({n_tr:,} params)", flush=True)
        code2z = CodeToZ(levels, device)
        optimizer = torch.optim.AdamW([p for p in fsq.parameters() if p.requires_grad], lr=lr)
        print("[terminator] building lean dataset (parquet cols + DINO mmap, no video)…", flush=True)
        dataset = TerminatorDataset(repo_id, str(skillvla_dir), str(dino_path))
        collate_fn = collate
        net = fsq

        def forward_fn(batch):
            tc = batch["skill_code_true"].view(-1).long().clamp(0, vocab - 1)
            return term_forward(fsq, code2z, tc, batch["observation.state"], batch["skill_decoder_dino"])

        def save_fn(step):
            return save_ckpt(out_dir, step, fsq, fsq_cfg, optimizer)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                        collate_fn=collate_fn, drop_last=True, pin_memory=False,  # pinned-mem alloc under concurrent jobs → CUDA "unknown error"
                        persistent_workers=num_workers > 0)
    print(f"[terminator] dataset frames={len(dataset)} repo={repo_id} arch={arch} — starting train", flush=True)

    # ── resume: auto-continue from checkpoints/last if present (e.g. to extend a finished run: raise `steps`) ──
    ckpt_name = "terminator.pt" if arch == "custom" else "FSQ.pt"
    last_ckpt = out_dir / "checkpoints" / "last" / ckpt_name
    start_step = 0
    if last_ckpt.exists():
        rs = torch.load(str(last_ckpt), map_location=device, weights_only=False)
        net.load_state_dict(rs["model_state"], strict=False)        # frozen DINO/codebook already in `net`
        if rs.get("optimizer_state") is not None:
            optimizer.load_state_dict(rs["optimizer_state"])
        start_step = int(rs.get("step", 0))
        print(f"[terminator] RESUMING from {last_ckpt} at step {start_step} (target {steps})", flush=True)

    wandb = None
    if wandb_enable:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=wandb_project, name=run_name, config={
            "source": source, "run_tag": run_tag, "steps": steps, "batch_size": batch_size, "lr": lr,
            "sigma": sigma, "pos_weight": pos_weight, "levels": levels, "warm_start": warm_start,
            "progress_loss_weight": w_prog, "termination_loss_weight": w_term})

    # ── train ──
    step = start_step
    t0 = time.perf_counter()
    done = step >= steps
    while not done:
        for batch in loader:
            prog, term_logits = forward_fn(batch)
            ds = batch["skill_ds"].float().view(-1).to(prog.device)
            de = batch["skill_de"].float().view(-1).to(prog.device)
            prog_tgt = (ds / (ds + de).clamp_min(1.0)).clamp(0.0, 1.0)
            term_tgt = torch.exp(-(de ** 2) / (2.0 * sigma ** 2)) if sigma > 0 else (de == 0).float()
            pw = torch.tensor(pos_weight, device=term_logits.device, dtype=term_logits.dtype)
            loss_prog = F.smooth_l1_loss(prog, prog_tgt.to(prog.dtype))
            loss_term = F.binary_cross_entropy_with_logits(term_logits, term_tgt.to(term_logits.dtype), pos_weight=pw)
            loss = w_prog * loss_prog + w_term * loss_term

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % log_freq == 0:
                sps = (step - start_step) / max(time.perf_counter() - t0, 1e-6)
                print(f"[{step:>6}/{steps}] loss={loss.item():.4f} (prog={loss_prog.item():.4f} "
                      f"term={loss_term.item():.4f}) {sps:.1f} it/s")
                if wandb is not None:
                    wandb.log({"loss": loss.item(), "loss_progress": loss_prog.item(),
                               "loss_termination": loss_term.item()}, step=step)
            if step % save_freq == 0 or step >= steps:
                print(f"[terminator] saved → {save_fn(step)}", flush=True)
            if step >= steps:
                done = True
                break

    print(f"[terminator] DONE → {out_dir}")
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
