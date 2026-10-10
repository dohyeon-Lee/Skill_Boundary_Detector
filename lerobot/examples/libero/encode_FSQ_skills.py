"""Encode skillset trajectories with a trained FSQ checkpoint.

This produces the skill_latents*.npz consumed by the SkillVLA data builders.
The default path instantiates only ``encoder.*``. Optional evaluation artifacts
also reuse that encoder for a boundary sweep and load the checkpoint terminator
for a small, task-balanced GT-boundary diagnostic pass.
"""

from __future__ import annotations

import json
import gc
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from FSQ import (
    SplineFSQEncoder,
    build_skill_initial_previous_actions,
    build_boundary_augmentation_contexts,
    encoder_grounding_position,
    load_fsq_encoder,
    load_fsq_model,
    spline_encode,
)
from train_FSQ import (
    _compute_skill_orders,
    attach_episode_offsets,
    load_skill_files,
)


@dataclass
class Args:
    skills_dir: str
    """Directory containing per-skill npz files."""

    model_path: str = ""
    """FSQ checkpoint path: FSQ.pt or FSQ_epochXXXX.pt. Omit when using --plan-path."""

    output_path: str = ""
    """Output skill_latents npz path. Omit when using --plan-path."""

    boundary_sweep_output_path: str = ""
    """Optional deterministic boundary-jitter sweep artifact."""

    boundary_sweep_max_offset: int = 10
    """Largest absolute boundary offset included in the optional sweep."""

    encode_batch_size: int = 256
    """Batch size used by the optional boundary sweep encoder."""

    raw_dataset_dir: str = ""
    """Source LeRobot dataset, required by terminator diagnostics."""

    terminator_samples_per_task: int = 8
    """Deterministic skill samples per requested task for terminator diagnostics."""

    terminator_extension_frames: int = 10
    """Next-skill frames evaluated after the GT boundary."""

    plan_path: str = ""
    """JSON file holding [{"model_path": ..., "output_path": ...}, ...].

    Reading the skillset dominates a single-checkpoint run (11k npz files versus
    ~40s of encoding), so a plan encodes many checkpoints of one run in one
    process and pays that cost once instead of once per checkpoint."""

    overwrite: bool = False
    """Re-encode a plan entry whose output already exists (default: skip it)."""

    device: str = "cuda"

    # ── 전이 안전망 (B): 새 데이터셋 인코딩 시 미지원 코드 → 최근접 지원 코드로 snap ──
    snap_to_supported: bool = False
    """raw FSQ code가 참조(training) 데이터셋에서 미지원(빈/희귀)이면 가장 가까운 지원 코드로 옮김.
    다운스트림 VLA가 학습 못 한 코드로 스킬이 배정되는 것을 방지 (OOD graceful degradation)."""
    supported_freq_path: str = ""
    """지원 코드 참조 npz. training 빌드의 skill_code_freq.npz(motion_counts) 또는 skill_latents.npz(tokens)."""
    min_code_freq: int = 1
    """이 값 미만 빈도의 코드는 '미지원'으로 간주 → snap 대상. 1 = 완전 빈칸만; ↑ 하면 희귀코드도."""
    snap_metric: str = "l1"
    """격자 거리 (l1|l2)."""


def load_model(model_path: Path, device: str) -> SplineFSQEncoder:
    model, _ = load_fsq_encoder(model_path, device)
    return model


def _grid_coords(model: SplineFSQEncoder) -> np.ndarray:
    """All code vectors in the quantizer's native encoder-output space."""
    fsq = model.fsq
    if hasattr(fsq, "bit_weights"):
        codes = torch.arange(fsq.codebook_size, device=fsq.bit_weights.device)
        normalized = fsq.code_to_normalized(codes)
        return (normalized / np.sqrt(fsq.latent_dim)).cpu().numpy().astype(np.float32)

    # FSQ.forward returns integer-spaced raw grid coordinates.
    L = np.array([int(round(2 * h + 1)) for h in fsq.levels_half.cpu().tolist()])   # levels_half=(L-1)/2
    strides = fsq.strides.cpu().numpy()
    half = fsq.half_width.cpu().numpy()
    codes = np.arange(int(np.prod(L)))
    lvl = (codes[:, None] // strides[None, :]) % L[None, :]                          # (C, D) level 0..L-1
    return (lvl - half[None, :]).astype(np.float32)                                  # (C, D) integer coord


def _supported_from_freq(freq: np.ndarray, n_codes: int, min_freq: int) -> np.ndarray:
    return np.where(np.asarray(freq).ravel()[:n_codes] >= min_freq)[0]


def _load_supported(ref_path: Path, n_codes: int, min_freq: int) -> np.ndarray:
    """Reference npz → indices of supported codes (freq >= min_freq). Accepts skill_code_freq.npz
    (motion_counts) or skill_latents.npz (tokens → bincount)."""
    raw = np.load(str(ref_path))
    if "motion_counts" in raw:
        freq = raw["motion_counts"]
    elif "tokens" in raw:
        freq = np.bincount(raw["tokens"].astype(np.int64).ravel(), minlength=n_codes)
    else:
        raise KeyError(f"{ref_path}: need 'motion_counts' or 'tokens' key for supported-code reference.")
    return _supported_from_freq(freq, n_codes, min_freq)


def _snap_to_supported(latents: np.ndarray, tokens: np.ndarray, model: SplineFSQEncoder, args: Args):
    """Remap each skill whose raw code is unsupported → nearest supported code (grid distance in the
    integer cell-coord space, which is exactly what `latents` holds). Returns (latents, tokens)."""
    coords = _grid_coords(model)                                    # (C, D)
    if str(args.supported_freq_path).strip().lower() == "self":
        # self-pruning: 방금 인코딩한 RAW 토큰 분포가 곧 기준표 (외부 파일 불필요 — 1-pass 자기완결).
        # un-snap 빌드의 skill_code_freq.npz(motion_counts)를 참조하는 것과 수치적으로 동일.
        freq = np.bincount(tokens.astype(np.int64).ravel(), minlength=len(coords))
        supported = _supported_from_freq(freq, len(coords), args.min_code_freq)
    else:
        supported = _load_supported(Path(args.supported_freq_path), len(coords), args.min_code_freq)
    if len(supported) == 0:
        raise ValueError("no supported codes at this min_code_freq — lower --min_code_freq.")
    sup_coords = coords[supported]                                  # (S, D)
    is_sup = np.zeros(len(coords), dtype=bool)
    is_sup[supported] = True

    lat, tok = latents.copy(), tokens.copy()
    dead = ~is_sup[tokens]
    dists = []
    for i in np.where(dead)[0]:
        diff = sup_coords - latents[i][None, :]
        d = np.abs(diff).sum(1) if args.snap_metric == "l1" else (diff ** 2).sum(1)
        j = int(np.argmin(d))
        tok[i] = supported[j]
        lat[i] = sup_coords[j]
        dists.append(float(d[j]))
    n = len(dead)
    print(f"[FSQ encode] snap: {int(dead.sum())}/{n} skills ({dead.mean()*100:.1f}%) landed on "
          f"unsupported codes (freq<{args.min_code_freq}) → snapped to nearest of {len(supported)} "
          f"supported codes | snap dist({args.snap_metric}) mean={np.mean(dists) if dists else 0:.2f} "
          f"max={np.max(dists) if dists else 0:.0f}")
    return lat, tok


def _encode_plan(args: Args) -> list[dict[str, Any]]:
    """Checkpoint/output entries from ``--plan-path`` or the single-run flags."""
    if args.plan_path:
        if args.model_path or args.output_path:
            raise ValueError("--plan-path replaces --model-path/--output-path; pass one form.")
        entries = json.loads(Path(args.plan_path).read_text())
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{args.plan_path} must hold a non-empty list of entries.")
        result = []
        for entry in entries:
            sweep_path = str(entry.get("boundary_sweep_output_path") or "").strip()
            result.append(
                {
                    "model_path": Path(entry["model_path"]),
                    "output_path": Path(entry["output_path"]),
                    "boundary_sweep_output_path": Path(sweep_path) if sweep_path else None,
                    "terminator_output_path": (
                        Path(str(entry["terminator_output_path"]))
                        if str(entry.get("terminator_output_path") or "").strip()
                        else None
                    ),
                    "terminator_task_ids": [
                        int(value) for value in entry.get("terminator_task_ids", [])
                    ],
                }
            )
        return result
    if not args.model_path or not args.output_path:
        raise ValueError("Pass --model-path and --output-path, or --plan-path.")
    return [
        {
            "model_path": Path(args.model_path),
            "output_path": Path(args.output_path),
            "boundary_sweep_output_path": (
                Path(args.boundary_sweep_output_path)
                if args.boundary_sweep_output_path.strip()
                else None
            ),
            "terminator_output_path": None,
            "terminator_task_ids": [],
        }
    ]


BOUNDARY_SWEEP_DIRECTIONS = (
    "early_start",
    "late_start",
    "early_end",
    "late_end",
)


def _boundary_sweep_segment(context, direction: str, offset: int, min_length: int):
    """Return one exact boundary shift, or ``None`` when that shift is invalid."""
    if offset < 0:
        raise ValueError(f"Boundary sweep offset must be non-negative, got {offset}.")
    if direction not in BOUNDARY_SWEEP_DIRECTIONS:
        raise ValueError(f"Unknown boundary sweep direction: {direction!r}.")
    if offset == 0:
        return context.trajectory[context.start : context.end].copy()
    start, end = int(context.start), int(context.end)
    if direction == "early_start":
        start -= offset
    elif direction == "late_start":
        start += offset
    elif direction == "early_end":
        end -= offset
    else:
        end += offset
    if start < 0 or end > len(context.trajectory) or end - start < min_length:
        return None
    return context.trajectory[start:end].copy()


@torch.inference_mode()
def _encode_trajectory_batch(
    model: SplineFSQEncoder,
    trajectories: list[np.ndarray],
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch the existing FSQ encoder contract for a list of variable trajectories."""
    if batch_size <= 0:
        raise ValueError(f"encode_batch_size must be positive, got {batch_size}.")
    action_seq_encoder = (
        getattr(getattr(model, "cfg", None), "encoder_arch", "spline") == "action_seq"
    )
    all_latents: list[np.ndarray] = []
    all_tokens: list[np.ndarray] = []
    for start in range(0, len(trajectories), batch_size):
        items = trajectories[start : start + batch_size]
        lengths = torch.as_tensor(
            [len(item) for item in items], dtype=torch.long, device=device
        )
        if action_seq_encoder:
            prepared = [model._prepare_actions_numpy(item).float() for item in items]
            steps = int(lengths.max().item())
            values = torch.zeros(
                len(items), steps, prepared[0].shape[-1], dtype=torch.float32
            )
            for row, item in enumerate(prepared):
                values[row, : len(item)] = item
            latent, token = model(values.to(device), lengths)
        else:
            controls = []
            grounding = []
            for item in items:
                control, _ = spline_encode(
                    item,
                    model.n_control,
                    model.spline_degree,
                    input_mode=model.encoder_input_mode,
                )
                controls.append(control)
                if model.enc_start_proj is not None:
                    grounding.append(encoder_grounding_position(item))
            control_tensor = torch.from_numpy(np.stack(controls)).float().to(device)
            grounding_tensor = (
                torch.from_numpy(np.stack(grounding)).float().to(device)
                if grounding
                else None
            )
            latent, token = model(
                control_tensor,
                lengths,
                grounding_tensor,
                normalized=False,
            )
        all_latents.append(latent.float().cpu().numpy())
        all_tokens.append(token.int().cpu().numpy())
    return np.concatenate(all_latents), np.concatenate(all_tokens)


def _encode_boundary_sweep(
    args: Args,
    *,
    model: SplineFSQEncoder,
    output_path: Path,
    clean_latents_path: Path,
    device: str,
    segments,
    skill_actions,
    metadata,
) -> None:
    """Encode every valid deterministic boundary offset around each clean skill."""
    max_offset = int(args.boundary_sweep_max_offset)
    if max_offset <= 0:
        raise ValueError(
            f"boundary_sweep_max_offset must be positive, got {max_offset}."
        )
    action_seq_encoder = (
        getattr(getattr(model, "cfg", None), "encoder_arch", "spline") == "action_seq"
    )
    source = skill_actions if action_seq_encoder else segments
    contexts = build_boundary_augmentation_contexts(source, metadata, pmax=max_offset)
    with np.load(clean_latents_path, allow_pickle=False) as clean:
        clean_latents = clean["latents"].astype(np.float32)
        clean_tokens = clean["tokens"].astype(np.int32)
    sample_count = len(source)
    if len(clean_tokens) != sample_count or len(clean_latents) != sample_count:
        raise ValueError(
            "Clean latent artifact and skillset lengths differ: "
            f"{len(clean_tokens)}/{len(clean_latents)} vs {sample_count}."
        )
    offsets = np.arange(max_offset + 1, dtype=np.int16)
    directions = np.asarray(BOUNDARY_SWEEP_DIRECTIONS, dtype="U16")
    latent_dim = int(clean_latents.shape[1])
    valid = np.zeros((len(directions), len(offsets), sample_count), dtype=np.bool_)
    tokens = np.full(valid.shape, -1, dtype=np.int32)
    latents = np.full((*valid.shape, latent_dim), np.nan, dtype=np.float32)
    min_length = max(1, int(round(float(model.cfg.length_min))))
    for direction_index, direction in enumerate(directions):
        valid[direction_index, 0] = True
        tokens[direction_index, 0] = clean_tokens
        latents[direction_index, 0] = clean_latents
        for offset in offsets[1:]:
            trajectories = []
            rows = []
            for row, context in enumerate(contexts):
                trajectory = _boundary_sweep_segment(
                    context, str(direction), int(offset), min_length
                )
                if trajectory is not None:
                    rows.append(row)
                    trajectories.append(trajectory)
            if not trajectories:
                continue
            encoded_latents, encoded_tokens = _encode_trajectory_batch(
                model,
                trajectories,
                device=device,
                batch_size=int(args.encode_batch_size),
            )
            valid[direction_index, int(offset), rows] = True
            tokens[direction_index, int(offset), rows] = encoded_tokens
            latents[direction_index, int(offset), rows] = encoded_latents
        print(
            f"[FSQ encode] boundary sweep {direction}: "
            f"valid@{max_offset}={int(valid[direction_index, -1].sum())}/{sample_count}"
        )

    save_dict: dict[str, np.ndarray] = {
        "format": np.asarray("fsq_boundary_jitter_sweep_v1"),
        "directions": directions,
        "offsets": offsets,
        "valid": valid,
        "tokens": tokens,
        "latents": latents,
    }
    for key in ("episode_id", "task_id", "skill_index", "frame_start", "frame_end"):
        save_dict[key] = np.asarray([item[key] for item in metadata])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f".tmp{os.getpid()}.npz")
    np.savez_compressed(temporary, **save_dict)
    temporary.replace(output_path)
    print(f"[FSQ encode] boundary sweep saved -> {output_path}")


class _TerminatorFrameSource:
    """Batch exact video frames, preferring the existing lossless FSQ cache."""

    def __init__(self, raw_dataset_dir: Path) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.root = raw_dataset_dir.resolve()
        self.dataset = LeRobotDataset(
            repo_id=f"local/{self.root.name}",
            root=self.root,
            video_keys_to_load=[
                "observation.images.image",
                "observation.images.wrist_image",
            ],
        )
        self.reader = self.dataset._ensure_reader()  # noqa: SLF001
        if self.reader.hf_dataset is None:
            self.reader.load_and_activate()
        self.cache = None
        project_root = Path(__file__).resolve().parents[3]
        candidates = sorted(
            project_root.glob(
                f".cache/fsq_frame_cache/*/{self.root.name}/rgb_zstd_v2/*/_SUCCESS"
            )
        )
        if candidates:
            from fsq_frame_cache import RGBFrameCache

            self.cache = RGBFrameCache(candidates[-1].parent, self.root)
            print(f"[FSQ encode] terminator frames: {candidates[-1].parent}")

    @staticmethod
    def _number(value, kind):
        return kind(value.item()) if isinstance(value, torch.Tensor) else kind(value)

    def frames(self, metadata: dict, count: int, camera_key: str) -> torch.Tensor:
        start = int(metadata["dataset_from_index"]) + int(metadata["frame_start"])
        indices = list(range(start, start + count))
        rows = self.reader.hf_dataset[indices]
        episode_ids = [self._number(value, int) for value in rows["episode_index"]]
        if len(set(episode_ids)) != 1:
            raise RuntimeError(f"Terminator sample crosses episodes: {episode_ids}.")
        episode_id = episode_ids[0]
        timestamps = [self._number(value, float) for value in rows["timestamp"]]
        episode = self.reader._meta.episodes[episode_id]  # noqa: SLF001
        video_path = self.reader.root / self.reader._meta.get_video_file_path(  # noqa: SLF001
            episode_id, camera_key
        )
        from_timestamp = float(episode[f"videos/{camera_key}/from_timestamp"])
        queries = [from_timestamp + timestamp for timestamp in timestamps]
        if self.cache is not None:
            return self.cache.get_frames(
                video_path, queries, self.reader._tolerance_s  # noqa: SLF001
            )
        from lerobot.datasets.video_utils import decode_video_frames

        return decode_video_frames(
            video_path,
            queries,
            self.reader._tolerance_s,  # noqa: SLF001
            self.reader._video_backend,  # noqa: SLF001
            decoder_num_threads=1,
        )


@torch.inference_mode()
def _terminator_probabilities(
    model,
    cfg,
    latent: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    initial_previous_action: np.ndarray | None,
    third: torch.Tensor | None,
    wrist: torch.Tensor | None,
    *,
    device: str,
) -> np.ndarray:
    steps = len(states)
    z_q = torch.from_numpy(latent.astype(np.float32)).unsqueeze(0).to(device)
    z_norm = model.fsq.normalized(z_q).repeat_interleave(steps, dim=0)
    raw_states = torch.from_numpy(states.astype(np.float32)).unsqueeze(0).to(device)
    flat_state = raw_states.reshape(steps, -1)[..., : cfg.state_dim]
    if cfg.terminator_context == "none":
        context_sequence = flat_context = None
    elif cfg.terminator_context == "prev_action":
        emitted = torch.from_numpy(actions.astype(np.float32)).unsqueeze(0).to(device)
        initial = (
            None
            if initial_previous_action is None
            else torch.from_numpy(initial_previous_action.astype(np.float32))
            .unsqueeze(0)
            .to(device)
        )
        context_sequence = model._previous_action_context(
            emitted, initial_previous_action=initial
        )
        flat_context = context_sequence.reshape(steps, cfg.action_dim)
    else:
        context_sequence = raw_states[..., : cfg.state_dim]
        flat_context = flat_state
    start_state = (
        flat_state[:1].expand(steps, -1)
        if bool(getattr(cfg, "terminator_start_proprio", False))
        else None
    )
    third = None if third is None else third.to(device, non_blocking=True)
    wrist = None if wrist is None else wrist.to(device, non_blocking=True)
    with torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.bfloat16,
        enabled=torch.device(device).type == "cuda",
    ):
        if cfg.state_rnn_terminator:
            _, logits, _ = model.terminator.forward_all_outputs(
                model.fsq.normalized(z_q), context_sequence
            )
            logits = logits.reshape(-1)
        elif cfg.terminator_input_space == "state":
            _, logits = model.terminator.forward_outputs(z_norm, flat_context)
        elif cfg.terminator_input_space == "image":
            _, logits = model.terminator(z_norm, third, wrist)
        else:
            _, logits = model.terminator(
                z_norm,
                flat_context,
                third,
                wrist,
                start_state=start_state,
            )
    return logits.float().sigmoid().cpu().numpy()


def _encode_terminator_diagnostics(
    args: Args,
    *,
    model_path: Path,
    output_path: Path,
    clean_latents_path: Path,
    device: str,
    dec_states: list[np.ndarray],
    actions: list[np.ndarray],
    metadata: list[dict],
    task_ids: list[int],
    frame_source: _TerminatorFrameSource,
) -> None:
    """Evaluate a deterministic, task-balanced subset around every GT boundary."""
    if args.terminator_samples_per_task <= 0:
        raise ValueError("terminator_samples_per_task must be positive.")
    model, cfg = load_fsq_model(model_path, device)
    if model.terminator is None or not bool(cfg.terminator_termination):
        raise ValueError(f"Terminator diagnostics requested for a non-terminator model: {model_path}")
    if bool(getattr(cfg, "terminator_goal_xyz", False)):
        raise ValueError("Terminator diagnostics do not yet support goal-XYZ terminators.")
    attach_episode_offsets(args.raw_dataset_dir, metadata)
    with np.load(clean_latents_path, allow_pickle=False) as clean:
        latents = clean["latents"].astype(np.float32)
        tokens = clean["tokens"].astype(np.int32)
    initial_actions = build_skill_initial_previous_actions(
        actions, metadata, actions[0].shape[-1]
    )
    by_episode_skill = {
        (int(item["episode_id"]), int(item["skill_index"])): index
        for index, item in enumerate(metadata)
    }
    following = {}
    for index, item in enumerate(metadata):
        candidate = by_episode_skill.get(
            (int(item["episode_id"]), int(item["skill_index"]) + 1)
        )
        if (
            candidate is not None
            and int(metadata[candidate]["frame_start"]) == int(item["frame_end"])
        ):
            following[index] = candidate
    requested = set(task_ids)
    grouped: dict[int, list[int]] = {}
    for index, item in enumerate(metadata):
        task = int(item["task_id"])
        if requested and task not in requested:
            continue
        grouped.setdefault(task, []).append(index)
    generator = np.random.default_rng(20261010)
    selected: list[int] = []
    for task in sorted(grouped):
        candidates = np.asarray(grouped[task], dtype=np.int64)
        generator.shuffle(candidates)
        selected.extend(candidates[: args.terminator_samples_per_task].tolist())
    selected.sort(key=lambda index: (int(metadata[index]["task_id"]), index))
    if not selected:
        raise ValueError("No skills matched terminator diagnostic task selection.")
    probability_rows: list[np.ndarray] = []
    boundary_indices = []
    output_tokens = []
    output_tasks = []
    output_episodes = []
    output_skills = []
    output_starts = []
    output_ends = []
    for sample_number, index in enumerate(tqdm(selected, desc="Terminator diagnostics")):
        item = metadata[index]
        current_length = len(dec_states[index])
        next_index = following.get(index)
        extension = (
            min(int(args.terminator_extension_frames), len(dec_states[next_index]))
            if next_index is not None
            else 0
        )
        states = dec_states[index]
        emitted_actions = actions[index]
        if extension:
            states = np.concatenate([states, dec_states[next_index][:extension]], axis=0)
            emitted_actions = np.concatenate(
                [emitted_actions, actions[next_index][:extension]], axis=0
            )
        camera_mode = str(getattr(cfg, "terminator_cameras", "both"))
        third = (
            frame_source.frames(item, len(states), "observation.images.image")
            if camera_mode in {"both", "top"}
            else None
        )
        wrist = (
            frame_source.frames(item, len(states), "observation.images.wrist_image")
            if camera_mode in {"both", "wrist"}
            else None
        )
        probabilities = _terminator_probabilities(
            model,
            cfg,
            latents[index],
            states,
            emitted_actions,
            initial_actions[index],
            third,
            wrist,
            device=device,
        )
        probability_rows.append(probabilities.astype(np.float32))
        boundary_indices.append(current_length - 1)
        output_tokens.append(int(tokens[index]))
        output_tasks.append(int(item["task_id"]))
        output_episodes.append(int(item["episode_id"]))
        output_skills.append(int(item["skill_index"]))
        output_starts.append(int(item["frame_start"]))
        output_ends.append(int(item["frame_end"]))
        if (sample_number + 1) % 32 == 0:
            print(f"[FSQ encode] terminator {sample_number + 1}/{len(selected)}")
    lengths = np.asarray([len(row) for row in probability_rows], dtype=np.int16)
    save_dict = {
        "format": np.asarray("fsq_terminator_diagnostics_v1"),
        "threshold": np.asarray(float(cfg.end_threshold), dtype=np.float32),
        "end_target_sigma": np.asarray(float(cfg.end_target_sigma), dtype=np.float32),
        "extension_frames": np.asarray(int(args.terminator_extension_frames), dtype=np.int16),
        "probabilities_cat": np.concatenate(probability_rows).astype(np.float32),
        "probabilities_len": lengths,
        "boundary_index": np.asarray(boundary_indices, dtype=np.int16),
        "tokens": np.asarray(output_tokens, dtype=np.int32),
        "task_id": np.asarray(output_tasks, dtype=np.int16),
        "episode_id": np.asarray(output_episodes, dtype=np.int32),
        "skill_index": np.asarray(output_skills, dtype=np.int16),
        "frame_start": np.asarray(output_starts, dtype=np.int32),
        "frame_end": np.asarray(output_ends, dtype=np.int32),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f".tmp{os.getpid()}.npz")
    np.savez_compressed(temporary, **save_dict)
    temporary.replace(output_path)
    print(f"[FSQ encode] terminator diagnostics saved -> {output_path}")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _encode_one(
    args: Args,
    *,
    model_path: Path,
    output_path: Path,
    device: str,
    segments,
    skill_actions,
    metadata,
) -> None:
    model = load_model(model_path, device)
    print(f"[FSQ encode] model={model_path}")
    print(f"[FSQ encode] device={device} skills={len(segments)}")

    latents = []
    tokens = []
    # action_seq checkpoints encode ACTION sequences; every other variant
    # encodes the state trajectory. No images either way.
    action_seq_encoder = getattr(getattr(model, "cfg", None), "encoder_arch", "spline") == "action_seq"
    source = skill_actions if action_seq_encoder else segments
    for item in tqdm(source, desc="Encoding FSQ skills"):
        if action_seq_encoder:
            latents.append(model.encode_actions_numpy(item, device=device))
            tokens.append(model.encode_actions_index(item, device=device))
        else:
            latents.append(model.encode_numpy(item, device=device))
            tokens.append(model.encode_index(item, device=device))

    latents_arr = np.stack(latents).astype(np.float32)
    tokens_arr = np.array(tokens, dtype=np.int32)

    if args.snap_to_supported:
        if not args.supported_freq_path:
            raise ValueError("--snap_to_supported requires --supported_freq_path (training code freq).")
        latents_arr, tokens_arr = _snap_to_supported(latents_arr, tokens_arr, model, args)

    save_dict: dict[str, np.ndarray] = {
        "latents": latents_arr,
        "tokens": tokens_arr.astype(np.int32),
        "skill_order": np.array(_compute_skill_orders(metadata), dtype=np.float32),
    }
    for key in ("episode_id", "task_id", "skill_index", "frame_start", "frame_end", "length"):
        save_dict[key] = np.array([m[key] for m in metadata])
    # Write through a temp file so a killed job never leaves a half-written npz
    # that later runs would treat as an already-encoded checkpoint. The name must
    # keep the .npz suffix: np.savez appends one when it is missing, and the
    # rename would then target a file that was never written.
    temporary = output_path.with_suffix(f".tmp{os.getpid()}.npz")
    np.savez(str(temporary), **save_dict)
    temporary.replace(output_path)
    print(f"[FSQ encode] saved -> {output_path}")


def main(args: Args) -> None:
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    plan = _encode_plan(args)
    for entry in plan:
        entry["output_path"].parent.mkdir(parents=True, exist_ok=True)

    # Read the skillset ONCE: it costs far more than a checkpoint's encoding.
    segments, dec_states, skill_actions, metadata = load_skill_files(Path(args.skills_dir))
    frame_source = None

    for index, entry in enumerate(plan, start=1):
        model_path = entry["model_path"]
        output_path = entry["output_path"]
        assert model_path is not None and output_path is not None
        sweep_output_path = entry["boundary_sweep_output_path"]
        terminator_output_path = entry["terminator_output_path"]
        clean_complete = output_path.is_file() and not args.overwrite
        sweep_complete = (
            sweep_output_path is None
            or (sweep_output_path.is_file() and not args.overwrite)
        )
        terminator_complete = (
            terminator_output_path is None
            or (terminator_output_path.is_file() and not args.overwrite)
        )
        if clean_complete and sweep_complete and terminator_complete:
            print(f"[FSQ encode] ({index}/{len(plan)}) exists, skipping -> {output_path}")
            continue
        print(f"[FSQ encode] ({index}/{len(plan)}) {model_path.name}")
        if not clean_complete:
            _encode_one(
                args,
                model_path=model_path,
                output_path=output_path,
                device=device,
                segments=segments,
                skill_actions=skill_actions,
                metadata=metadata,
            )
        if sweep_output_path is not None and not sweep_complete:
            model = load_model(model_path, device)
            _encode_boundary_sweep(
                args,
                model=model,
                output_path=sweep_output_path,
                clean_latents_path=output_path,
                device=device,
                segments=segments,
                skill_actions=skill_actions,
                metadata=metadata,
            )
        if terminator_output_path is not None and not terminator_complete:
            if not args.raw_dataset_dir:
                raise ValueError(
                    "--raw-dataset-dir is required for terminator diagnostics."
                )
            if frame_source is None:
                frame_source = _TerminatorFrameSource(Path(args.raw_dataset_dir))
            _encode_terminator_diagnostics(
                args,
                model_path=model_path,
                output_path=terminator_output_path,
                clean_latents_path=output_path,
                device=device,
                dec_states=dec_states,
                actions=skill_actions,
                metadata=metadata,
                task_ids=entry["terminator_task_ids"],
                frame_source=frame_source,
            )


if __name__ == "__main__":
    main(tyro.cli(Args))
