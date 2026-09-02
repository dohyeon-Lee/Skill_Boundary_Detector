"""Train the v3 FSQ tokenizer and selected decoder objectives jointly.

The terminator input is configurable as state, image, or both. Visual modes read
live third-person and wrist frames at sampled timesteps; the state RNN instead
supervises the complete skill sequence and never constructs a video reader.

Usage:
    python examples/libero/train_FSQ.py \
      --skills_dir      /path/to/skillset/skills \
      --raw_dataset_dir /path/to/lerobot_dataset   (videos/ + meta/ 포함) \
      --output_dir      /path/to/output

Data requirements:
  skills_dir/*.npz  — per-skill npz with keys: actions, states, episode_id, skill_index,
                       frame_start, frame_end, task_id (optional)
  raw_dataset_dir   — LeRobot v3 dataset (videos/{image_key}/chunk-*/file-*.mp4 + meta/)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def _enforce_inline_cuda_guard() -> None:
    """Reuse this process's torch import for the Slurm CUDA preflight."""
    if os.environ.get("LEROBOT_INLINE_CUDA_GUARD") != "1":
        return
    if torch.cuda.is_available():
        return

    marker = os.environ.get("LEROBOT_CUDA_GUARD_FAILURE_MARKER")
    if marker:
        Path(marker).touch()
    print("GPU GUARD: training process cannot initialize CUDA.", file=sys.stderr)
    raise SystemExit(86)


# ── Args ───────────────────────────────────────────────────────────────────────

@dataclass
class Args:
    # ── data
    skills_dir: str = ""
    """Directory (recursively searched) for per-skill .npz files."""
    raw_dataset_dir: str = ""
    """Raw LeRobot dataset. The default/both terminator reads selected camera
    frames from cache or live video when its termination objective is enabled."""
    output_dir: str = ""
    """Output directory. Defaults to parent of skills_dir."""

    # ── model
    hidden_dim: int = 256
    fsq_levels: list[int] = field(default_factory=lambda: [5, 5, 5])
    quantizer: str = "fsq"
    """fsq | bsq. BSQ uses bsq_code_dim bits and ignores fsq_levels."""
    bsq_code_dim: int = 5
    num_layers: int = 3
    dropout: float = 0.1
    n_control: int = 30
    spline_degree: int = 3
    autoencoder_mode: str = "action"
    """Indivisible encoder/decoder preset: raw | zero | action | norm_action."""
    action_gripper_weight: float = 1.0
    """Trailing gripper-axis MSE weight for every autoencoder mode."""
    start_state_conditioning: str = "none"
    """Optional decoder context in every preset: none | adaln."""
    encoder_input_mode: str = "zero_grounded"
    """raw | zero_grounded | start_grounded | optimal.

    start_grounded expresses XYZ and axis-angle rotation relative to the first
    EEF pose while leaving gripper state unchanged.
    """
    encoder_length_token: bool = False
    """Fixed false by the standard YAML path; retained only for direct CLI compatibility."""
    encoder_arch: str = "spline"
    """spline: fixed control-point tokens. action_seq: variable-length ACTION
    sequence transformer (no spline codec / grounding / length-token choices)."""
    reconstructor_start_state: bool = False
    """Internal compatibility flag resolved from start_state_conditioning."""
    reconstructor_arch: str = "oneshot"
    """skill/oneshot: full control-point-grid reconstruction once
    per trajectory. action_seq: raw full
    action sequence reconstructed from z by a GRU. action_seq_transformer:
    the same raw target decoded by a z-AdaLN causal query Transformer."""
    reconstructor_output_mode: str = "match_encoder"
    """Spline/oneshot output: raw | zero_grounded | start_grounded | match_encoder."""
    init_calibration: bool = False
    """Calibrate the fresh z_head once from clean training trajectories."""
    init_calibration_gain: float = 1.0
    """Target per-axis z standard deviation after one-shot calibration."""
    init_calibration_samples: int = 0
    """Clean trajectories used for calibration; 0 uses the full training split."""
    pair_loss: str = "none"
    """none | overlap | js | contrastive. The linear contrastive mode also
    repels one randomly selected adjacent skill by full-code overlap."""
    pair_weight: float = 0.1
    pair_inv_temperature: float = 5.0
    route_loss: bool = False
    """Use detached all-code reconstructor/termination costs for encoder routing.
    Reuses pair_inv_temperature, with no schedule or separate loss weight."""
    pair_warmup: bool = False
    """Enable reconstruction-only warm-up and pair-weight ramp."""
    pair_warmup_epochs: int = 0
    """Initial reconstruction-only epochs."""
    pair_ramp_epochs: int = 0
    """Epochs to linearly ramp pair weight to pair_weight."""
    boundary_aug_pmax: int = 0
    """Legacy shared fallback for directional augmentation windows."""
    boundary_aug_early_start_pmax: int = -1
    boundary_aug_late_start_pmax: int = -1
    boundary_aug_early_end_pmax: int = -1
    boundary_aug_late_end_pmax: int = -1
    """Directional maxima; -1 inherits boundary_aug_pmax and 0 disables."""
    boundary_aug_distribution: str = "half_normal"
    decoder_reconstructor: bool = True
    decoder_terminator_progress: bool = True
    decoder_terminator_termination: bool = True
    terminator_context: str = "prev_action"
    """prev_action, proprio, or none (skill+vision only)."""
    terminator_cameras: str = "both"
    """both, top, or wrist."""
    visual_terminator_arch: str = "small"
    """small | fusion; the terminator is always the default multimodal model."""
    vision_backbone: str = "dino"
    """dino, siglip, or resnet; shared by the selected terminator cameras."""
    freeze_vision_encoder: bool = True
    dino_image_size: int = 224
    siglip_image_size: int = 224
    resnet_image_size: int = 224
    frame_cache_dir: str = ""
    """Completed exact RGB frame cache; blank retains live video decoding."""
    skill_cond_mode: str = "token"
    """Skill conditioning shared by reconstructor and terminator: token/AdaRMS or hidden broadcast."""
    chunk_size: int = 10
    """Steps per motion action chunk."""
    samples_per_skill: int = 2
    """M: random decoder timesteps sampled from each of the B skill trajectories per update."""
    max_state_dim: int = 32
    max_action_dim: int = 32
    pi_base: str = "../models/pi05_base"
    """PI05 checkpoint used only to initialize SigLIP terminator vision when requested."""

    # ── loss
    action_loss_weight: float = 1.0
    progress_loss_weight: float = 0.1
    end_loss_weight: float = 0.1
    end_pos_weight: float = 1.0
    """BCE positive-class weight for the termination head."""
    end_threshold: float = 0.5
    end_target_sigma: float = 0.0
    """Soft termination target std in frames (Gaussian bump at the skill end). 0 = hard 1-frame
    spike. σ≈2-3 curbs the val overfit a sharp spike causes and adds ±tolerance to recall/precision."""

    # ── training
    epochs: int = 300
    lr: float = 3e-4
    """Shared learning rate for encoder, reconstructor, and terminator."""
    lr_schedule: str = "cosine"
    """cosine: decay to 1% of lr; constant: keep lr fixed."""
    batch_size: int = 64
    num_workers: int = 8
    val_num_workers: int = 0
    """Validation video workers. Keep at 0 to avoid AV1 decoder deadlocks across epochs."""
    grad_clip: float = 1.0
    gradient_checkpointing: bool = False
    """Enable activation checkpointing. Saves VRAM at the cost of extra recompute; false is faster."""
    val_split: float = 0.1
    val_every: int = 1
    """Run validation every N epochs; 0 disables it."""
    save_best_model: bool = True
    """Write FSQ.pt whenever the validation selection metric improves."""
    # Best-val SELECTION metric weights. Unset (None) → follow the actual loss (total val). Set to
    # override: FSQ.pt = argmin over epochs of wa*action + wp*progress + we*end on val.
    val_select_action_weight: float | None = None
    val_select_progress_weight: float | None = None
    val_select_end_weight: float | None = None
    log_every: int = 10
    seed: int = 42
    device: str = "cuda"
    checkpoint_every: int = 0
    resume_from: str | None = None
    wandb_project: str | None = None
    wandb_run_name: str = "spline_fsqae"


# ── Data loading ───────────────────────────────────────────────────────────────

def _compute_skill_orders(metadata: list[dict]) -> list[float]:
    by_ep: dict[int, list[int]] = {}
    for i, m in enumerate(metadata):
        by_ep.setdefault(int(m["episode_id"]), []).append(i)
    orders = [0.0] * len(metadata)
    for ids in by_ep.values():
        ids.sort(key=lambda i: (metadata[i]["frame_start"], metadata[i]["skill_index"]))
        denom = max(1, len(ids))
        for rank, idx in enumerate(ids):
            orders[idx] = float((rank + 1) / denom)
    return orders


def _make_encoder_traj(states):
    """Encoder trajectory = full observation.state (EEF pose + gripper STATE), same source as the
    decoder state. The selected encoder convention is applied downstream; mean grounding changes
    XYZ only, while rotation and trailing gripper-state dimensions stay absolute. For LIBERO this
    is 8D: ee_state(6) + gripper_state(2)."""
    return states.astype(np.float32)


def _make_decoder_state(states):
    """Decoder/terminator proprioception = full observation.state (EEF pose + gripper STATE).

    Gripper here is the observed gripper STATE (part of observation.state), not the
    action command, so there is no target leak — the current-step value is used as-is.
    For LIBERO this is 8D: ee_state(6) + gripper_state(2).
    """
    return states.astype(np.float32)


def _make_target_action(actions):
    """Decoder target is the real dataset action_t."""
    return actions.astype(np.float32)


def load_skill_files(
    skills_dir: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[dict]]:
    npz_files = sorted(skills_dir.rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files in {skills_dir}")

    segments, dec_states, dec_targets, metadata = [], [], [], []
    for f in npz_files:
        d = np.load(str(f))
        actions = d["actions"].astype(np.float32)
        states  = d["states"].astype(np.float32)
        segments.append(_make_encoder_traj(states))
        dec_states.append(_make_decoder_state(states))
        dec_targets.append(_make_target_action(actions))
        metadata.append({
            "file":        str(f.name),
            "episode_id":  int(d["episode_id"]),
            "task_id":     int(d["task_id"]) if "task_id" in d else -1,
            "skill_index": int(d["skill_index"]),
            "frame_start": int(d["frame_start"]),
            "frame_end":   int(d["frame_end"]),
            "length":      len(actions),
        })

    print(f"[FSQ] Loaded {len(segments)} skills from {skills_dir}")
    lengths = [m["length"] for m in metadata]
    print(f"[FSQ] Skill lengths — min:{min(lengths)} max:{max(lengths)} mean:{np.mean(lengths):.1f}")
    return segments, dec_states, dec_targets, metadata


def attach_episode_offsets(raw_dataset_dir: str, metadata: list[dict]) -> None:
    """Attach the absolute dataset index corresponding to each skill's episode."""
    from lerobot.datasets.io_utils import load_episodes

    episodes = load_episodes(Path(raw_dataset_dir))
    for item in metadata:
        episode = episodes[int(item["episode_id"])]
        item["dataset_from_index"] = int(episode["dataset_from_index"])
        episode_length = int(episode["dataset_to_index"]) - int(episode["dataset_from_index"])
        if int(item["frame_start"]) + int(item["length"]) > episode_length:
            raise ValueError(
                f"Skill {item['file']} exceeds episode {item['episode_id']}: "
                f"start={item['frame_start']} length={item['length']} episode_length={episode_length}"
            )


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: Args) -> None:
    from FSQ import (
        SplineFSQAEConfig,
        encoder_grounding_convention,
        is_action_sequence_reconstructor_arch,
        prepare_encoder_trajectory,
        resolve_boundary_augmentation_pmaxes,
        train_spline_fsqae,
    )

    skills_dir = Path(args.skills_dir)
    output_dir = Path(args.output_dir) if args.output_dir else skills_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    segments, dec_states, dec_targets, metadata = load_skill_files(skills_dir)

    if not args.raw_dataset_dir:
        raise ValueError("--raw_dataset_dir is required for dataset normalization metadata.")
    args.autoencoder_mode = args.autoencoder_mode.strip().lower()
    autoencoder_presets = {
        "raw": ("spline", "raw_state", "oneshot", "raw_state"),
        "zero": ("spline", "zero_grounded", "oneshot", "zero_grounded"),
        "action": (
            "action_seq",
            "zero_grounded",
            "action_seq_transformer",
            "zero_grounded",
        ),
        "norm_action": (
            "action_seq",
            "zero_grounded",
            "action_seq_transformer",
            "zero_grounded",
        ),
    }
    if args.autoencoder_mode not in autoencoder_presets:
        raise ValueError(
            "--autoencoder_mode must be raw|zero|action|norm_action, "
            f"got {args.autoencoder_mode!r}."
        )
    (
        args.encoder_arch,
        args.encoder_input_mode,
        args.reconstructor_arch,
        args.reconstructor_output_mode,
    ) = autoencoder_presets[args.autoencoder_mode]
    if (
        not np.isfinite(args.action_gripper_weight)
        or not 0.0 < args.action_gripper_weight <= 1.0
    ):
        raise ValueError("--action_gripper_weight must be in (0, 1].")
    args.start_state_conditioning = args.start_state_conditioning.strip().lower()
    if args.start_state_conditioning not in {"none", "adaln"}:
        raise ValueError(
            "--start_state_conditioning must be none|adaln, "
            f"got {args.start_state_conditioning!r}."
        )
    args.reconstructor_start_state = args.start_state_conditioning == "adaln"
    terminator_input_space = "both"
    terminator_model = "default"
    dino_model_path = "../models/dinov3-vits16"
    terminator_enabled = (
        args.decoder_terminator_progress or args.decoder_terminator_termination
    )
    if not args.decoder_reconstructor and not terminator_enabled:
        raise ValueError("At least one decoder output must be enabled.")
    if args.route_loss and not args.decoder_reconstructor:
        raise ValueError(
            "--route_loss requires --decoder_reconstructor."
        )
    reconstructor_only = args.decoder_reconstructor and not terminator_enabled
    terminator_only = not args.decoder_reconstructor and terminator_enabled
    state_rnn_terminator = False
    terminator_termination_only = (
        args.decoder_terminator_termination
        and not args.decoder_terminator_progress
    )
    uses_visual_terminator = terminator_enabled
    if uses_visual_terminator:
        attach_episode_offsets(args.raw_dataset_dir, metadata)

    args.quantizer = args.quantizer.strip().lower()
    if args.quantizer not in {"fsq", "bsq"}:
        raise ValueError(f"--quantizer must be fsq|bsq, got {args.quantizer!r}.")
    if args.bsq_code_dim < 2:
        raise ValueError("--bsq_code_dim must be >= 2.")
    if args.encoder_input_mode not in {
        "zero_grounded", "start_grounded", "raw_state", "optimal"
    }:
        raise ValueError(
            "--encoder_input_mode must be zero_grounded|start_grounded|raw_state|optimal, "
            f"got {args.encoder_input_mode!r}."
        )
    if args.reconstructor_output_mode not in {
        "zero_grounded", "start_grounded", "raw_state"
    }:
        raise ValueError(
            "--reconstructor_output_mode must be "
            "raw|zero_grounded|start_grounded|match_encoder, "
            f"got {args.reconstructor_output_mode!r}."
        )
    if args.encoder_arch not in {"spline", "action_seq"}:
        raise ValueError(f"--encoder_arch must be spline|action_seq, got {args.encoder_arch!r}.")
    if args.init_calibration_gain <= 0:
        raise ValueError("--init_calibration_gain must be positive.")
    if args.init_calibration_samples < 0:
        raise ValueError("--init_calibration_samples must be non-negative.")
    args.pair_loss = args.pair_loss.strip().lower()
    if args.pair_loss not in {"none", "overlap", "js", "contrastive"}:
        raise ValueError(
            "--pair_loss must be none|overlap|js|contrastive, "
            f"got {args.pair_loss!r}."
        )
    directional_pmaxes = resolve_boundary_augmentation_pmaxes(
        args.boundary_aug_pmax,
        early_start_pmax=args.boundary_aug_early_start_pmax,
        late_start_pmax=args.boundary_aug_late_start_pmax,
        early_end_pmax=args.boundary_aug_early_end_pmax,
        late_end_pmax=args.boundary_aug_late_end_pmax,
    )
    (
        args.boundary_aug_early_start_pmax,
        args.boundary_aug_late_start_pmax,
        args.boundary_aug_early_end_pmax,
        args.boundary_aug_late_end_pmax,
    ) = directional_pmaxes
    args.boundary_aug_pmax = max(directional_pmaxes)
    if args.pair_loss != "none" and not any(directional_pmaxes):
        raise ValueError(
            "--pair_loss requires at least one positive directional boundary augmentation pmax."
        )
    if args.pair_warmup_epochs < 0 or args.pair_ramp_epochs < 0:
        raise ValueError("--pair_warmup_epochs and --pair_ramp_epochs must be non-negative.")
    if args.reconstructor_arch not in {
        "oneshot", "action_seq", "action_seq_transformer"
    }:
        raise ValueError(
            "--reconstructor_arch must be skill|action_seq|"
            "action_seq_transformer, "
            f"got {args.reconstructor_arch!r}."
        )
    if is_action_sequence_reconstructor_arch(args.reconstructor_arch):
        if args.encoder_arch != "action_seq":
            raise ValueError(
                "An action-sequence reconstructor requires --encoder_arch action_seq."
            )
    print(
        f"[FSQ] autoencoder mode: {args.autoencoder_mode}; "
        f"encoder arch: {args.encoder_arch}, length token: {args.encoder_length_token}, "
        f"start-state conditioning: {args.start_state_conditioning}"
    )
    print(
        "[FSQ] z_head init calibration: "
        + (
            f"enabled (gain={args.init_calibration_gain:g}, "
            f"samples={args.init_calibration_samples or 'all training'})"
            if args.init_calibration
            else "disabled"
        )
    )
    print(
        f"[FSQ] pair loss: {args.pair_loss}"
        + (
            f" (weight={args.pair_weight}, inv_temperature={args.pair_inv_temperature}, "
            f"warmup={'on' if args.pair_warmup else 'off'}, "
            f"recon-only={args.pair_warmup_epochs} epochs, "
            f"ramp={args.pair_ramp_epochs} epochs, "
            "boundary=one-of-enabled-start/end, "
            "pmax(early_start/late_start/early_end/late_end)="
            f"{args.boundary_aug_early_start_pmax}/"
            f"{args.boundary_aug_late_start_pmax}/"
            f"{args.boundary_aug_early_end_pmax}/"
            f"{args.boundary_aug_late_end_pmax}, "
            f"distribution={args.boundary_aug_distribution})"
            if args.pair_loss != "none"
            else ""
        )
    )
    print(
        "[FSQ] joint decoder-aware routing: "
        + (
            "enabled (all codes, detached distortion, "
            f"inv_temperature={args.pair_inv_temperature:g})"
            if args.route_loss
            else "disabled"
        )
    )

    # Encoder normalization stats must follow the exact checkpointed input convention used before
    # spline fitting. Length stats are data-driven min/max over skill lengths.
    encoder_trajectories = np.concatenate([
        prepare_encoder_trajectory(s, args.encoder_input_mode)
        for s in segments
    ])
    encoder_min, encoder_max = encoder_trajectories.min(0), encoder_trajectories.max(0)
    grounding_convention = encoder_grounding_convention(args.encoder_input_mode)
    reconstructor_trajectories = np.concatenate([
        prepare_encoder_trajectory(s, args.reconstructor_output_mode)
        for s in segments
    ])
    reconstructor_min = reconstructor_trajectories.min(0)
    reconstructor_max = reconstructor_trajectories.max(0)
    encoder_start_min = encoder_start_max = None
    if args.encoder_input_mode == "optimal":
        from FSQ import encoder_grounding_position

        grounding_positions = np.stack([
            encoder_grounding_position(s) for s in segments
        ])
        encoder_start_min = grounding_positions.min(0)
        encoder_start_max = grounding_positions.max(0)
    all_state = np.concatenate(dec_states)   # decoder/terminator proprioception (raw, all timesteps)
    state_min,  state_max  = all_state.min(0), all_state.max(0)
    _lens = [int(m["length"]) for m in metadata]
    length_min, length_max = float(min(_lens)), float(max(_lens))
    stats_path = Path(args.raw_dataset_dir) / "meta" / "stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"Dataset normalization stats are required for Stage-1-compatible FSQ training: {stats_path}"
        )
    raw_stats = json.loads(stats_path.read_text())
    try:
        state_q01 = np.asarray(raw_stats["observation.state"]["q01"], dtype=np.float32)
        state_q99 = np.asarray(raw_stats["observation.state"]["q99"], dtype=np.float32)
        action_q01 = np.asarray(raw_stats["action"]["q01"], dtype=np.float32)
        action_q99 = np.asarray(raw_stats["action"]["q99"], dtype=np.float32)
    except KeyError as exc:
        raise KeyError(f"{stats_path} lacks the q01/q99 stats required by Stage-1: {exc}") from exc
    state_dim, action_dim = dec_states[0].shape[-1], dec_targets[0].shape[-1]
    if len(state_q01) != state_dim or len(action_q01) != action_dim:
        raise ValueError(
            f"Dataset stats dimensionality mismatch: state {len(state_q01)}!={state_dim}, "
            f"action {len(action_q01)}!={action_dim}."
        )
    raw_action_min = np.concatenate(dec_targets).min(axis=0).astype(np.float32)
    raw_action_max = np.concatenate(dec_targets).max(axis=0).astype(np.float32)
    if args.autoencoder_mode == "action":
        max_abs_action = max(
            float(np.abs(raw_action_min).max()),
            float(np.abs(raw_action_max).max()),
        )
        if max_abs_action > 1.0001:
            raise ValueError(
                "Raw action-sequence reconstruction requires native controller "
                "commands in [-1, 1] because the decoder output uses tanh; "
                f"observed min/max={raw_action_min}/{raw_action_max}."
            )
    stats = dict(
        encoder_min=encoder_min, encoder_max=encoder_max,
        encoder_input_mode=np.asarray(args.encoder_input_mode),
        encoder_grounding_convention=np.asarray(grounding_convention),
        reconstructor_min=reconstructor_min, reconstructor_max=reconstructor_max,
        reconstructor_output_mode=np.asarray(args.reconstructor_output_mode),
        state_q01=state_q01, state_q99=state_q99,
        action_q01=action_q01, action_q99=action_q99,
        state_min=state_min, state_max=state_max,
        raw_action_min=raw_action_min, raw_action_max=raw_action_max,
        action_gripper_weight=np.float32(args.action_gripper_weight),
        length_min=np.float32(length_min), length_max=np.float32(length_max),
    )
    if args.autoencoder_mode == "action":
        stats["action_sequence_convention"] = np.asarray(
            "raw_controller_action_gripper_weighted_v1"
        )
    elif args.autoencoder_mode == "norm_action":
        stats["action_sequence_convention"] = np.asarray(
            "q01_q99_clipped_gripper_scaled_v1"
        )
    if encoder_start_min is not None:
        stats.update(encoder_start_min=encoder_start_min, encoder_start_max=encoder_start_max)
    np.savez(str(output_dir / "action_stats.npz"), **stats)
    print(f"[FSQ] encoder input mode: {args.encoder_input_mode}")
    print(f"[FSQ] encoder grounding convention: {grounding_convention}")
    print(f"[FSQ] reconstructor output mode: {args.reconstructor_output_mode}")
    print(
        "[FSQ] autoencoder gripper MSE weight: "
        f"{args.action_gripper_weight:g}"
    )
    print(f"[FSQ] encoder_min: {np.round(encoder_min, 4)}")
    print(f"[FSQ] encoder_max: {np.round(encoder_max, 4)}")
    if encoder_start_min is not None:
        print(f"[FSQ] optimal grounding mean-XYZ min/max: {np.round(encoder_start_min, 4)} / "
              f"{np.round(encoder_start_max, 4)}")
    print(f"[FSQ] state q01/q99: {np.round(state_q01, 4)} / {np.round(state_q99, 4)}")
    if args.autoencoder_mode == "action":
        print(
            "[FSQ] action-sequence convention: raw controller action; "
            f"gripper_weight={args.action_gripper_weight:g}"
        )
        print(
            f"[FSQ] raw action min/max: {np.round(raw_action_min, 4)} / "
            f"{np.round(raw_action_max, 4)}"
        )
    elif args.autoencoder_mode == "norm_action":
        print(
            "[FSQ] action-sequence convention: q01/q99 -> [-1, 1], "
            f"clipped; gripper_weight={args.action_gripper_weight:g}"
        )
        print(
            f"[FSQ] action q01/q99: {np.round(action_q01, 4)} / "
            f"{np.round(action_q99, 4)}"
        )
    else:
        print(f"[FSQ] action q01/q99: {np.round(action_q01, 4)} / {np.round(action_q99, 4)}")
    print(f"[FSQ] state_min:  {np.round(state_min,  4)}")
    print(f"[FSQ] state_max:  {np.round(state_max,  4)}")
    print(f"[FSQ] length_min/max: {length_min:.0f} / {length_max:.0f}")

    device = args.device if torch.cuda.is_available() else "cpu"
    enc_dim    = segments[0].shape[-1]

    wandb_run = None
    if args.wandb_project:
        import os

        import wandb

        # Pin wandb's system-metrics GPU monitor to THIS job's GPU(s) so the panel shows only our device
        # instead of every GPU on the shared node (same fix as lerobot's WandBLogger — READ-only use of
        # the CUDA_VISIBLE_DEVICES Slurm already set; telemetry scoping only, never affects allocation).
        gpu_settings = None
        try:
            ids = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
            if ids:
                gpu_settings = wandb.Settings(x_stats_gpu_device_ids=ids)
        except Exception:  # noqa: BLE001 — cosmetic monitor scoping must never break training
            gpu_settings = None
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={**vars(args), "action_dim": action_dim, "enc_dim": enc_dim,
                    "state_dim": state_dim, "n_segments": len(segments)},
            settings=gpu_settings,
        )

    cfg = SplineFSQAEConfig(
        action_dim=action_dim,
        enc_dim=enc_dim,
        state_dim=state_dim,
        n_control=args.n_control,
        spline_degree=args.spline_degree,
        encoder_input_mode=args.encoder_input_mode,
        encoder_grounding_convention=grounding_convention,
        encoder_length_token=args.encoder_length_token,
        encoder_arch=args.encoder_arch,
        autoencoder_mode=args.autoencoder_mode,
        action_gripper_weight=args.action_gripper_weight,
        quantizer=args.quantizer,
        bsq_code_dim=args.bsq_code_dim,
        init_calibration=args.init_calibration,
        init_calibration_gain=args.init_calibration_gain,
        init_calibration_samples=args.init_calibration_samples,
        pair_loss=args.pair_loss,
        pair_weight=args.pair_weight,
        pair_inv_temperature=args.pair_inv_temperature,
        route_loss=args.route_loss,
        pair_warmup=args.pair_warmup,
        pair_warmup_epochs=args.pair_warmup_epochs,
        pair_ramp_epochs=args.pair_ramp_epochs,
        boundary_aug_pmax=args.boundary_aug_pmax,
        boundary_aug_early_start_pmax=args.boundary_aug_early_start_pmax,
        boundary_aug_late_start_pmax=args.boundary_aug_late_start_pmax,
        boundary_aug_early_end_pmax=args.boundary_aug_early_end_pmax,
        boundary_aug_late_end_pmax=args.boundary_aug_late_end_pmax,
        boundary_aug_distribution=args.boundary_aug_distribution,
        reconstructor_start_state=args.reconstructor_start_state,
        reconstructor_start_state_conditioning=(
            "adaln" if args.reconstructor_start_state else "concat"
        ),
        reconstructor_arch=args.reconstructor_arch,
        reconstructor_output_mode=args.reconstructor_output_mode,
        hidden_dim=args.hidden_dim,
        fsq_levels=args.fsq_levels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_state_dim=args.max_state_dim,
        max_action_dim=args.max_action_dim,
        skill_cond_mode=args.skill_cond_mode,
        pi_base=args.pi_base,
        terminator_arch=args.visual_terminator_arch,
        terminator_input_space=terminator_input_space,
        terminator_context=args.terminator_context,
        terminator_cameras=args.terminator_cameras,
        terminator_model=terminator_model,
        terminator_progress=args.decoder_terminator_progress,
        terminator_termination=args.decoder_terminator_termination,
        terminator_termination_only=terminator_termination_only,
        reconstructor_only=reconstructor_only,
        terminator_only=terminator_only,
        state_rnn_terminator=state_rnn_terminator,
        vision_backbone=args.vision_backbone,
        freeze_vision_encoder=args.freeze_vision_encoder,
        dino_model_path=dino_model_path,
        dino_image_size=args.dino_image_size,
        siglip_image_size=args.siglip_image_size,
        resnet_image_size=args.resnet_image_size,
        frame_cache_dir=args.frame_cache_dir,
        image_encoder_layers=3,
        image_encoder_heads=4,
        chunk_size=args.chunk_size,
        samples_per_skill=args.samples_per_skill,
        length_min=length_min,
        length_max=length_max,
        action_loss_weight=args.action_loss_weight,
        progress_loss_weight=args.progress_loss_weight,
        end_loss_weight=args.end_loss_weight,
        end_pos_weight=args.end_pos_weight,
        end_threshold=args.end_threshold,
        end_target_sigma=args.end_target_sigma,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_num_workers=args.val_num_workers,
        grad_clip=args.grad_clip,
        gradient_checkpointing=args.gradient_checkpointing,
        epochs=args.epochs,
        val_split=args.val_split,
        val_every=args.val_every,
        save_best_model=args.save_best_model,
        val_select_action_weight=args.val_select_action_weight,
        val_select_progress_weight=args.val_select_progress_weight,
        val_select_end_weight=args.val_select_end_weight,
        log_every=args.log_every,
        device=device,
        save_path=str(output_dir / "FSQ.pt"),
        checkpoint_every=args.checkpoint_every,
        encoder_min=encoder_min,
        encoder_max=encoder_max,
        encoder_start_min=encoder_start_min,
        encoder_start_max=encoder_start_max,
        reconstructor_min=reconstructor_min,
        reconstructor_max=reconstructor_max,
        state_min=state_min,
        state_max=state_max,
        state_q01=state_q01,
        state_q99=state_q99,
        action_q01=action_q01,
        action_q99=action_q99,
    )

    train_spline_fsqae(
        segments=segments,
        decoder_states=dec_states,
        decoder_targets=dec_targets,
        raw_dataset_dir=args.raw_dataset_dir,
        cfg=cfg,
        wandb_run=wandb_run,
        metadata=metadata,
        resume_from=args.resume_from,
    )

    # Skill latents (skill_latents.npz) are produced separately by encode_FSQ_skills.py
    # (build_data/encode_skills.sbatch) — the canonical, consumer-facing producer. We do
    # not re-emit them here to avoid a stale/duplicate copy under the training output_dir.
    if args.save_best_model:
        print(f"[FSQ] Saved best model → {output_dir / 'FSQ.pt'}")
    else:
        print(f"[FSQ] Saved periodic checkpoints → {output_dir / 'FSQ_epoch*.pt'}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    _enforce_inline_cuda_guard()

    # CLI parsing is needed only after CUDA is known to work. Keeping it here
    # also shortens the bad-node/requeue path.
    import tyro

    main(tyro.cli(Args))
