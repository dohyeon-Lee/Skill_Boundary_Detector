"""
build_skill_dataset.py — demos를 VF cosine divergence로 스킬 분할 후 .npz 저장.

Pipeline:
  1. 모든 task의 모든 episode 순회
  2. VF analysis (legacy spherical_xyz or generic pca_action probes)
  3. SG smooth + peak detection → skill boundaries
  4. actions / states를 boundary 단위로 잘라 .npz 저장
  5. 이미 처리된 episode는 skip (resume)

Output layout:
  output_dir/skills/ep{ep_id:05d}_skill{si:02d}.npz
    └── actions, states, episode_id, skill_index, frame_start, frame_end

Usage:
  python examples/libero/build_skill_dataset.py \
    --dataset_dir .../libero_90 \
    --policy_path .../checkpoints/080000/pretrained_model \
    --output_dir .../outputs/skill_dataset
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import tyro

sys.path.insert(0, str(Path(__file__).parent))
from skill_divider import (
    _savgol_smooth,
    get_episode_timestamps,
    get_video_path,
    load_data,
    load_dino_episode,
    load_episodes_meta,
    load_policy,
    run_vf_analysis,
)
from SBD_visualize import SkillVisualizer
from action_manifold import (
    ACTION_MODE_ANCHOR_RELATIVE,
    ACTION_MODE_DATASET,
    GRIPPER_CONTINUOUS,
    PCA_SCALE_NONE,
    PROBE_PCA_ACTION,
    PROBE_SPHERICAL_XYZ,
    SUPPORTED_PCA_SCALE_MODES,
    ActionPCA,
    NumpyActionNormalizer,
    RunningCovariance,
    get_or_fit_action_pca,
    relative_action_mask,
    resolve_indices,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_camera_frames(camera_keys, load_camera, *, serial_attempts: int = 3):
    """Decode cameras in parallel, then retry serially on transient ffmpeg failures.

    ``imageio_ffmpeg.read_frames`` waits only ten seconds for ffmpeg's metadata
    header.  When many Slurm array workers hit the shared filesystem at once,
    one of the camera decoders can exceed that timeout even though the video is
    valid.  Keep the fast parallel path, but fall back to a few serial attempts
    so a temporary startup delay does not leave a missing episode curve.
    """
    try:
        with ThreadPoolExecutor(max_workers=len(camera_keys)) as pool:
            return dict(pool.map(load_camera, camera_keys))
    except (OSError, RuntimeError) as parallel_error:
        last_error = parallel_error
        print(
            f" [video decode retry: {type(parallel_error).__name__}; serial fallback]",
            end="",
            flush=True,
        )

    for attempt in range(1, serial_attempts + 1):
        try:
            return dict(load_camera(cam_key) for cam_key in camera_keys)
        except (OSError, RuntimeError) as error:
            last_error = error
            if attempt < serial_attempts:
                time.sleep(float(attempt))

    raise RuntimeError(
        f"Camera video decoding failed after parallel attempt and "
        f"{serial_attempts} serial attempts"
    ) from last_error


# ── Args ──────────────────────────────────────────────────────────────────────

@dataclass
class Args:
    dataset_dir: str = str(PROJECT_ROOT / "dataset_filtered" / "libero_90_full_full")
    output_dir: str = str(PROJECT_ROOT / "outputs_filtered" / "skill_dataset")
    policy_path: str = ""
    device: str = "cuda"
    task_ids: list[int] | None = None
    """처리할 task index 목록. None이면 전체 task 처리."""
    # ── Diffusion scheduler ──────────────────────────────────────────────────
    noise_scheduler_type: str = "DDIM"
    num_inference_steps: int = 10
    eval_at_step: int = 7
    # ── VF analysis ──────────────────────────────────────────────────────────
    replan_interval: int = 3
    n_gmm_components: int = 5
    probe_mode: str = ""
    """User-facing mode: spherical, full, without_gripper, or std."""
    probe_type: str = PROBE_SPHERICAL_XYZ
    """spherical_xyz (legacy) or pca_action (generic selected-action probes)."""
    probe_count: int = 24
    """Number of PCA probes; GT is added separately."""
    probe_alpha: float = 0.1
    """Per-dimension RMS radius in PCA input coordinates."""
    pca_variance: float = 0.95
    """Cumulative explained variance retained by action-plan PCA."""
    pca_stride: int = 3
    """Anchor stride used while fitting the dataset action PCA."""
    pca_scale_mode: str = PCA_SCALE_NONE
    """none or std; std fits correlation PCA after per-dimension standardization."""
    probe_exclude_indices: str = ""
    """Comma-separated full-action dimensions excluded from PCA probes and scoring."""
    action_mode: str = ACTION_MODE_DATASET
    """dataset or anchor_relative."""
    relative_exclude_joints: str = "gripper"
    """Comma-separated action-name tokens left absolute in anchor_relative mode."""
    gripper_mode: str = GRIPPER_CONTINUOUS
    """continuous or discrete."""
    gripper_indices: str = "-1"
    """Comma-separated full-action indices used when gripper_mode=discrete."""
    gripper_values: str = "-1,1"
    """Raw valid low/high gripper commands for discrete projection."""
    gripper_threshold: float = 0.0
    # ── Peak detection ────────────────────────────────────────────────────────
    smooth_window: int = 7
    savgol_polyorder: int = 4
    peak_nms: bool = True
    nms_dist: int | None = None
    """NMS 거리. None이면 replan_interval * 2 사용."""
    boundary_threshold_mode: Literal["episode_mean", "global_mean"] = "episode_mean"
    """episode_mean(legacy) or one global_mean threshold shared by every episode."""
    boundary_threshold_scale: float = 1.0
    """Multiplier applied to the episode or global arithmetic mean."""
    global_threshold_path: str = ""
    """Path to global_boundary_threshold.json when boundary_threshold_mode=global_mean."""
    use_cached_curves: bool = False
    """Re-segment from curves/ep*.npz without rerunning DP/VF inference (global 2-pass)."""
    # ── Dataset filtering ─────────────────────────────────────────────────────
    min_skill_len: int = 2
    """스킬 세그먼트 최소 프레임 수 (미만이면 해당 세그먼트 제외)."""
    min_skills: int = 1
    """유효 스킬 수가 이 값 미만이면 episode 전체 skip."""
    # ── Misc ─────────────────────────────────────────────────────────────────
    dino_feature_dir: str = ""
    """Per-episode DINO feature dir (e.g. .../libero_90_DINO/dinov3_vits16_pg8). Required when policy uses DINO features."""
    seed: int | None = 42
    resume: bool = True
    """True면 이미 저장된 episode skip."""
    write_done_markers: bool = False
    """True면 episode 처리 완료 시 .done 마커 기록 (필터/0-skill 포함). 재시도 검증용 (verify_skillset.py)."""
    # ── Boundary curves (eval HTML 오버레이용) ────────────────────────────────
    dump_curves: bool = True
    """True면 episode별 multimodality(VF cos-divergence) 곡선을 output_dir/curves/에 저장 (eval HTML 그래프용)."""
    curves_only: bool = False
    """True면 skill npz는 만들지 않고 곡선만 저장 (이미 빌드된 run 백필용). resume은 곡선 파일 존재 기준."""
    # ── WandB ────────────────────────────────────────────────────────────────
    wandb_project: str | None = None
    wandb_run_name: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────


def _csv_values(value: str, cast) -> list:
    return [cast(token.strip()) for token in str(value).split(",") if token.strip()]


def _threshold_for_episode(
    sg_vals: np.ndarray, args: Args, global_threshold: float | None
) -> float:
    if args.boundary_threshold_mode == "global_mean" and global_threshold is not None:
        return float(global_threshold)
    if not len(sg_vals):
        return 0.0
    return float(np.mean(sg_vals)) * args.boundary_threshold_scale


def _infer_probe_mode(args: Args) -> str:
    if args.probe_mode:
        mode = args.probe_mode.strip().lower()
        if mode not in {"spherical", "full", "without_gripper", "std"}:
            raise ValueError(f"--probe_mode must be spherical, full, without_gripper, or std; got {mode}")
        return mode
    if args.probe_type == PROBE_SPHERICAL_XYZ:
        return "spherical"
    if args.pca_scale_mode == "std":
        return "std"
    return "without_gripper" if args.probe_exclude_indices else "full"


def _skillset_manifest(
    args: Args,
    dataset_dir: Path,
    policy_path: str,
    image_key: str,
    mode: str,
    action_dim: int,
    action_pca: ActionPCA | None,
) -> dict:
    return {
        "schema_version": 1,
        "mode": mode,
        "dataset_name": dataset_dir.name,
        "dataset_dir": str(dataset_dir.resolve()),
        "policy_path": str(Path(policy_path).resolve()),
        "image_key": image_key,
        "action": {
            "dim": action_dim,
            "mode": args.action_mode,
            "relative_exclude_joints": _csv_values(args.relative_exclude_joints, str),
            "gripper_mode": args.gripper_mode,
            "gripper_indices": _csv_values(args.gripper_indices, int),
            "gripper_values": _csv_values(args.gripper_values, float),
            "gripper_threshold": args.gripper_threshold,
        },
        "probe": {
            "type": args.probe_type,
            "count": args.probe_count,
            "alpha": args.probe_alpha,
            "pca_variance": args.pca_variance,
            "pca_stride": args.pca_stride,
            "pca_scale_mode": args.pca_scale_mode,
            "exclude_indices": _csv_values(args.probe_exclude_indices, int),
            "seed": args.seed,
            "pca_components": action_pca.n_components if action_pca is not None else None,
            "pca_artifact": "action_probe_pca.npz" if action_pca is not None else None,
        },
        "detector": {
            "noise_scheduler_type": args.noise_scheduler_type,
            "num_inference_steps": args.num_inference_steps,
            "eval_at_step": args.eval_at_step,
            "replan_interval": args.replan_interval,
            "n_gmm_components": args.n_gmm_components,
            "smooth_window": args.smooth_window,
            "savgol_polyorder": args.savgol_polyorder,
            "peak_nms": args.peak_nms,
            "nms_dist": args.nms_dist if args.nms_dist is not None else args.replan_interval * 2,
            "boundary_threshold_mode": args.boundary_threshold_mode,
            # Keep scale=1 manifests compatible so an interrupted historical
            # mean build can resume after this feature is added.
            **(
                {"boundary_threshold_scale": args.boundary_threshold_scale}
                if not np.isclose(args.boundary_threshold_scale, 1.0)
                else {}
            ),
            "global_threshold_path": (
                str(Path(args.global_threshold_path).expanduser().resolve())
                if args.global_threshold_path else ""
            ),
            "min_skill_len": args.min_skill_len,
            "min_skills": args.min_skills,
        },
    }


def _write_skillset_manifest(path: Path, payload: dict) -> None:
    """Write or validate the immutable configuration for a mode-keyed skillset."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            existing = json.loads(path.read_text())
            if existing != payload:
                raise ValueError(
                    f"Skillset manifest mismatch: {path}\n"
                    f"existing={json.dumps(existing, sort_keys=True)}\n"
                    f"requested={json.dumps(payload, sort_keys=True)}"
                )
            return
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        tmp.replace(path)


def _action_names(dataset_dir: Path) -> list[str] | None:
    import json

    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    names = (info.get("features", {}).get("action") or {}).get("names")
    return [str(name) for name in names] if names else None


def _iter_state_action_episodes(dataset_dir: Path):
    """Stream complete episodes from ordered LeRobot parquet shards."""
    files = sorted((dataset_dir / "data").rglob("file-*.parquet")) or sorted(
        (dataset_dir / "data").rglob("episode_*.parquet")
    )
    if not files:
        raise FileNotFoundError(f"No parquet data found under {dataset_dir / 'data'}")

    pending_id = None
    pending_actions: list[np.ndarray] = []
    pending_states: list[np.ndarray] = []

    def _flush():
        if pending_id is None:
            return None
        return (
            int(pending_id),
            np.concatenate(pending_actions, axis=0),
            np.concatenate(pending_states, axis=0),
        )

    for path in files:
        frame = pd.read_parquet(
            path, columns=["episode_index", "frame_index", "observation.state", "action"]
        )
        frame = frame.sort_values(["episode_index", "frame_index"], kind="stable")
        for episode_id, group in frame.groupby("episode_index", sort=False):
            actions = np.stack(group["action"].to_numpy()).astype(np.float32)
            states = np.stack(group["observation.state"].to_numpy()).astype(np.float32)
            episode_id = int(episode_id)
            if pending_id is not None and episode_id != pending_id:
                completed = _flush()
                if completed is not None:
                    yield completed
                pending_actions = []
                pending_states = []
            pending_id = episode_id
            pending_actions.append(actions)
            pending_states.append(states)

    completed = _flush()
    if completed is not None:
        yield completed


def _fit_dataset_action_pca(
    dataset_dir: Path,
    horizon: int,
    stride: int,
    action_dim: int,
    action_indices: tuple[int, ...],
    action_mode: str,
    rel_mask: np.ndarray | None,
    normalizer: NumpyActionNormalizer,
    variance_threshold: float,
    scale_mode: str,
    metadata: dict,
) -> ActionPCA:
    if stride < 1:
        raise ValueError(f"pca_stride must be positive, got {stride}.")
    accumulator = RunningCovariance(len(action_indices))
    anchor_batch_size = 4096
    n_episodes = 0
    for _, actions, states in _iter_state_action_episodes(dataset_dir):
        if actions.shape[1] != action_dim:
            raise ValueError(f"Dataset action dim {actions.shape[1]} != policy action dim {action_dim}.")
        anchors = np.arange(0, len(actions), stride, dtype=np.int64)
        offsets = np.arange(horizon, dtype=np.int64)
        for start in range(0, len(anchors), anchor_batch_size):
            selected = anchors[start:start + anchor_batch_size]
            indices = np.minimum(selected[:, None] + offsets[None], len(actions) - 1)
            chunks = actions[indices].copy()
            if action_mode == ACTION_MODE_ANCHOR_RELATIVE:
                if rel_mask is None:
                    raise ValueError("anchor_relative PCA fitting requires a relative-action mask.")
                dims = len(rel_mask)
                chunks[..., :dims] -= (
                    states[selected, None, :dims] * rel_mask.astype(np.float32)[None, None]
                )
            elif action_mode != ACTION_MODE_DATASET:
                raise ValueError(f"Unsupported action mode: {action_mode}")
            normalized = normalizer.normalize(chunks)
            accumulator.update_batch(normalized.mean(axis=1)[:, list(action_indices)])
        n_episodes += 1
    print(f"  [PCA] fitted from {accumulator.count:,} anchors across {n_episodes:,} episodes")
    return ActionPCA.from_covariance(
        accumulator,
        variance_threshold,
        metadata=metadata,
        scale_mode=scale_mode,
    )


def _detect_boundaries(
    replan_ts: list,
    div_cos: np.ndarray,
    n_frames: int,
    args: Args,
    global_threshold: float | None = None,
) -> list[int]:
    nms_dist = (args.nms_dist if args.nms_dist is not None
                else args.replan_interval * 2) if args.peak_nms else 0
    sg_vals = _savgol_smooth(list(div_cos), args.smooth_window, polyorder=args.savgol_polyorder)
    threshold = _threshold_for_episode(sg_vals, args, global_threshold)
    peak_ts, _ = _find_peaks_above_threshold(
        sg_vals, replan_ts, threshold=threshold, min_distance=nms_dist, margin=nms_dist
    )
    return sorted(set([0] + [int(p) for p in peak_ts] + [n_frames]))


def _save_boundary_curve(curves_dir: Path, ep_id: int, task_id: int, replan_ts,
                         div_cos: np.ndarray, boundaries: list[int], n_frames: int,
                         args: "Args", global_threshold: float | None = None) -> None:
    """Persist the per-episode multimodality (VF cos-divergence) curve so the eval
    HTML can overlay it. Stores raw + SG-smoothed values, the mean threshold, the
    detected peaks and the final boundaries — everything the plot needs, so eval
    needs neither scipy nor the detection params (mirrors _detect_boundaries)."""
    replan_ts = np.asarray(replan_ts, dtype=np.int64)
    div_cos = np.asarray(div_cos, dtype=np.float32)
    sg_vals = np.asarray(
        _savgol_smooth(list(div_cos), args.smooth_window, polyorder=args.savgol_polyorder),
        dtype=np.float32,
    )
    threshold = _threshold_for_episode(sg_vals, args, global_threshold)
    nms_dist = (args.nms_dist if args.nms_dist is not None
                else args.replan_interval * 2) if args.peak_nms else 0
    peak_ts, peak_vals = _find_peaks_above_threshold(
        sg_vals,
        replan_ts.tolist(),
        threshold=threshold,
        min_distance=nms_dist,
        margin=nms_dist,
    )
    curves_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(curves_dir / f"ep{ep_id:07d}.npz"),
        episode_id=np.array(ep_id), task_id=np.array(task_id),
        replan_ts=replan_ts, div_cos=div_cos, sg_vals=sg_vals,
        mean_val=np.array(threshold, dtype=np.float32),
        threshold_val=np.array(threshold, dtype=np.float32),
        threshold_scale=np.array(args.boundary_threshold_scale, dtype=np.float32),
        threshold_mode=np.array(
            "episode_mean_staging"
            if args.curves_only and args.boundary_threshold_mode == "global_mean"
            else args.boundary_threshold_mode
        ),
        peak_ts=np.asarray(peak_ts, dtype=np.int64),
        peak_vals=np.asarray(peak_vals, dtype=np.float32),
        boundaries=np.asarray(boundaries, dtype=np.int64),
        n_frames=np.array(n_frames),
        probe_mode=np.array(_infer_probe_mode(args)),
        probe_type=np.array(args.probe_type),
        probe_alpha=np.array(args.probe_alpha, dtype=np.float32),
        pca_scale_mode=np.array(args.pca_scale_mode),
        probe_exclude_indices=np.array(args.probe_exclude_indices),
    )


def _find_peaks_above_threshold(
    vals: np.ndarray,
    ts: list,
    *,
    threshold: float,
    min_distance: int = 0,
    margin: int = 0,
) -> tuple[list, list]:
    """Same peak/NMS policy as skill_divider, with an explicit threshold."""
    from scipy.signal import find_peaks

    peak_idxs, _ = find_peaks(vals)
    above = [i for i in peak_idxs if vals[i] > threshold]

    if margin > 0 and len(ts) >= 2:
        t_min, t_max = ts[0], ts[-1]
        above = [i for i in above if ts[i] - t_min > margin and t_max - ts[i] > margin]

    if min_distance > 0 and len(above) > 1:
        by_height = sorted(above, key=lambda i: vals[i], reverse=True)
        kept, suppressed = [], set()
        for idx in by_height:
            if idx in suppressed:
                continue
            kept.append(idx)
            for other in by_height:
                if other != idx and abs(ts[other] - ts[idx]) <= min_distance:
                    suppressed.add(other)
        above = kept

    return [ts[i] for i in above], [float(vals[i]) for i in above]


def _load_global_threshold(path: str, args: Args) -> float:
    if not path:
        raise ValueError("global_mean requires --global_threshold_path")
    threshold_path = Path(path)
    if not threshold_path.is_file():
        raise FileNotFoundError(
            f"global_mean threshold file is missing: {threshold_path}. "
            "Run the curves collection/global-threshold pass first."
        )
    payload = json.loads(threshold_path.read_text())
    if payload.get("boundary_threshold_mode") != "global_mean":
        raise ValueError(f"Not a global_mean threshold file: {threshold_path}")
    source_scale = float(payload.get("boundary_threshold_scale", 1.0))
    if not np.isclose(source_scale, args.boundary_threshold_scale):
        raise ValueError(
            "Global threshold scale mismatch: "
            f"config={args.boundary_threshold_scale:g}, source={source_scale:g} "
            f"({threshold_path})"
        )
    raw_value = (
        payload["global_threshold"]
        if "global_threshold" in payload
        else payload["global_mean"]
    )
    value = float(raw_value)
    if not np.isfinite(value):
        raise ValueError(f"Invalid global mean in {threshold_path}: {value}")
    return value


def _save_skills(skills_dir: Path, ep_id: int, task_id: int,
                  gt_actions: np.ndarray, states: np.ndarray,
                  boundaries: list[int], min_skill_len: int, min_skills: int) -> list[str]:
    valid_segs = [
        (si, s, e)
        for si, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:]))
        if e - s >= min_skill_len
    ]
    if len(valid_segs) < min_skills:
        return []

    task_dir = skills_dir / f"task{task_id:02d}"
    task_dir.mkdir(exist_ok=True)

    saved = []
    for si, s, e in valid_segs:
        fname = task_dir / f"ep{ep_id:05d}_task{task_id:02d}_skill{si:02d}.npz"
        np.savez(
            str(fname),
            actions=gt_actions[s:e].astype(np.float32),
            states=states[s:e].astype(np.float32),
            episode_id=np.array(ep_id),
            task_id=np.array(task_id),
            skill_index=np.array(si),
            frame_start=np.array(s),
            frame_end=np.array(e),
        )
        saved.append(str(fname))
    return saved


def _done_marker(skills_dir: Path, ep_id: int, task_id: int) -> Path:
    return skills_dir / f"task{task_id:02d}" / f"ep{ep_id:05d}_task{task_id:02d}.done"


def _touch_done(skills_dir: Path, ep_id: int, task_id: int) -> None:
    """Mark an (episode, task) as processed, even when it produced 0 skills
    (filtered / empty). Lets verify_skillset.py tell finished-but-empty episodes
    apart from ones a dead GPU never reached (so retries terminate)."""
    m = _done_marker(skills_dir, ep_id, task_id)
    m.parent.mkdir(parents=True, exist_ok=True)
    m.touch()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: Args) -> None:
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    probe_mode = _infer_probe_mode(args)
    skills_dir = output_dir / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = output_dir / "curves"
    if args.dump_curves or args.curves_only or args.use_cached_curves:
        curves_dir.mkdir(parents=True, exist_ok=True)

    if args.curves_only and args.use_cached_curves:
        raise ValueError("--curves_only and --use_cached_curves are mutually exclusive")
    if args.boundary_threshold_scale <= 0.0:
        raise ValueError("boundary_threshold_scale must be positive.")
    global_threshold = None
    if args.boundary_threshold_mode == "global_mean" and not args.curves_only:
        global_threshold = _load_global_threshold(args.global_threshold_path, args)
        print(f"Global boundary threshold: {global_threshold:.8f} ({args.global_threshold_path})")

    if args.seed is not None:
        import random, torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    print("Loading metadata...")
    episodes_meta = load_episodes_meta(dataset_dir)
    tasks_meta = pd.read_parquet(dataset_dir / "meta" / "tasks.parquet").reset_index()

    # Episode → task_index (the UNIQUE task key). The dataset's "tasks" field is the language
    # STRING, which is NOT unique across task_index — e.g. a merged FT+PT dataset can have two
    # task_index sharing one language. Selecting episodes by language would conflate them and
    # segment an episode under every task that shares its language (duplicate skills). Map each
    # episode to its task_index from the per-frame data instead (each episode has a single one).
    _ep_task: dict[int, int] = {}
    for _pq in sorted((dataset_dir / "data").glob("*/*.parquet")):
        _df = pd.read_parquet(_pq, columns=["episode_index", "task_index"]).drop_duplicates("episode_index")
        _ep_task.update({int(e): int(t) for e, t in zip(_df["episode_index"], _df["task_index"])})

    def _episodes_for_task(tid: int) -> list[int]:
        return sorted(ep for ep, ti in _ep_task.items() if ti == tid)

    video_cols = [c for c in episodes_meta.columns
                  if c.startswith("videos/") and c.endswith("/chunk_index")]
    camera_keys = [c.split("/")[1] for c in video_cols]
    print(f"Cameras: {camera_keys}")

    task_ids = args.task_ids if args.task_ids is not None else sorted(tasks_meta["task_index"].tolist())
    print(f"Tasks to process: {len(task_ids)}")

    # Pre-count total episodes for ETA (by task_index — see _ep_task note above).
    all_episode_ids = []
    for task_id in task_ids:
        all_episode_ids.extend(_episodes_for_task(task_id))
    n_total_eps_global = len(all_episode_ids)
    print(f"Total episodes: {n_total_eps_global}")

    print(f"Loading policy from {args.policy_path} ...")
    t0 = time.time()
    policy, preprocessor = load_policy(
        args.policy_path, args.device, args.noise_scheduler_type, args.num_inference_steps
    )
    print(f"  [time] policy load: {time.time()-t0:.1f}s")

    action_pca = None
    action_normalizer = None
    probe_directions = None
    rel_mask = None
    gripper_indices: tuple[int, ...] = ()
    probe_action_indices = None
    gripper_values = tuple(_csv_values(args.gripper_values, float))
    if len(gripper_values) != 2:
        raise ValueError(f"--gripper_values needs exactly two comma-separated values, got {gripper_values}.")

    if args.probe_type == PROBE_PCA_ACTION:
        import hashlib

        if args.pca_scale_mode not in SUPPORTED_PCA_SCALE_MODES:
            raise ValueError(
                f"--pca_scale_mode must be one of {SUPPORTED_PCA_SCALE_MODES}, got {args.pca_scale_mode}."
            )
        action_dim = int(policy.config.action_feature.shape[0])
        excluded_indices = resolve_indices(_csv_values(args.probe_exclude_indices, int), action_dim)
        probe_action_indices = tuple(i for i in range(action_dim) if i not in excluded_indices)
        if not probe_action_indices:
            raise ValueError("--probe_exclude_indices removed every action dimension.")
        action_normalizer = NumpyActionNormalizer.from_preprocessor(preprocessor)
        names = _action_names(dataset_dir)
        exclude_tokens = _csv_values(args.relative_exclude_joints, str)
        if args.action_mode == ACTION_MODE_ANCHOR_RELATIVE:
            rel_mask = relative_action_mask(action_dim, names, exclude_tokens)
        elif args.action_mode != ACTION_MODE_DATASET:
            raise ValueError(f"Unsupported --action_mode: {args.action_mode}")
        gripper_indices = resolve_indices(_csv_values(args.gripper_indices, int), action_dim)

        stats_hash = hashlib.sha256()
        for name in sorted(action_normalizer.stats):
            stats_hash.update(name.encode("utf-8"))
            stats_hash.update(np.asarray(action_normalizer.stats[name]).tobytes())
        pca_metadata = {
            "version": 1,
            "dataset": dataset_dir.name,
            "action_dim": action_dim,
            "probe_action_indices": list(probe_action_indices),
            "horizon": int(policy.config.horizon),
            "stride": int(args.pca_stride),
            "variance_threshold": float(args.pca_variance),
            "action_mode": args.action_mode,
            "relative_exclude_joints": exclude_tokens,
            "normalizer_mode": action_normalizer.mode,
            "normalizer_stats_sha256": stats_hash.hexdigest(),
        }
        if args.pca_scale_mode != PCA_SCALE_NONE:
            pca_metadata["pca_scale_mode"] = args.pca_scale_mode
        pca_path = output_dir / "action_probe_pca.npz"
        print(f"Loading/fitting action-plan PCA: {pca_path}")
        action_pca = get_or_fit_action_pca(
            pca_path,
            pca_metadata,
            lambda: _fit_dataset_action_pca(
                dataset_dir=dataset_dir,
                horizon=int(policy.config.horizon),
                stride=args.pca_stride,
                action_dim=action_dim,
                action_indices=probe_action_indices,
                action_mode=args.action_mode,
                rel_mask=rel_mask,
                normalizer=action_normalizer,
                variance_threshold=args.pca_variance,
                scale_mode=args.pca_scale_mode,
                metadata=pca_metadata,
            ),
        )
        probe_directions = action_pca.sample_directions(args.probe_count, seed=args.seed or 0)
        print(
            f"  [PCA] {action_pca.n_components}/{len(probe_action_indices)} components "
            f"from action indices {probe_action_indices}, "
            f"scale={args.pca_scale_mode}, "
            f"explained={action_pca.explained_variance_ratio.sum():.4f}, "
            f"probes={args.probe_count}+GT, alpha={args.probe_alpha}"
        )
    elif args.probe_type != PROBE_SPHERICAL_XYZ:
        raise ValueError(f"Unsupported --probe_type: {args.probe_type}")

    manifest = _skillset_manifest(
        args=args,
        dataset_dir=dataset_dir,
        policy_path=args.policy_path,
        image_key=next(
            (
                key
                for key in ("observation.images.image", "observation.images.top")
                if key in camera_keys
            ),
            camera_keys[0] if camera_keys else "observation.images.image",
        ),
        mode=probe_mode,
        action_dim=int(policy.config.action_feature.shape[0]),
        action_pca=action_pca,
    )
    _write_skillset_manifest(output_dir / "skillset_manifest.json", manifest)

    use_dino = policy.config.use_dino_features
    dino_feature_dir = Path(args.dino_feature_dir) if args.dino_feature_dir else None
    if use_dino and dino_feature_dir is None:
        raise ValueError("--dino_feature_dir is required when policy uses DINO features.")
    dino_image_key = policy.config.dino_image_keys[0] if use_dino else None

    viz = SkillVisualizer(output_dir)

    # ── WandB init ────────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb_project:
        import wandb
        run_name = args.wandb_run_name or f"skill_dataset_{len(task_ids)}tasks"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "dataset_dir": args.dataset_dir,
                "n_tasks": len(task_ids),
                "n_total_episodes": n_total_eps_global,
                "probe_mode": probe_mode,
                "probe_type": args.probe_type,
                "probe_count": args.probe_count,
                "probe_alpha": args.probe_alpha,
                "pca_variance": args.pca_variance,
                "pca_scale_mode": args.pca_scale_mode,
                "pca_components": action_pca.n_components if action_pca is not None else None,
                "probe_exclude_indices": args.probe_exclude_indices,
                "replan_interval": args.replan_interval,
                "eval_at_step": args.eval_at_step,
                "n_gmm_components": args.n_gmm_components,
                "smooth_window": args.smooth_window,
                "savgol_polyorder": args.savgol_polyorder,
                "nms_dist": args.nms_dist,
                "boundary_threshold_mode": args.boundary_threshold_mode,
                "boundary_threshold_scale": args.boundary_threshold_scale,
                "global_threshold": global_threshold,
                "use_cached_curves": args.use_cached_curves,
                "min_skill_len": args.min_skill_len,
                "min_skills": args.min_skills,
            },
        )

    n_total_eps = 0
    n_processed = 0  # resume 포함 전체 처리 완료 수
    n_saved = 0
    n_skipped_resume = 0
    n_skipped_filter = 0
    n_error = 0
    total_skills = 0
    t_start = time.time()

    for task_id in task_ids:
        task_row = tasks_meta[tasks_meta["task_index"] == task_id]
        if task_row.empty:
            print(f"  [warn] task_id {task_id} not found, skipping.")
            continue
        target_lang = task_row.iloc[0]["task"]          # for display / skill metadata
        episode_ids = _episodes_for_task(task_id)        # by task_index, NOT language (see note above)
        n_total_eps += len(episode_ids)
        print(f"\nTask {task_id}: '{target_lang}' — {len(episode_ids)} episodes")

        ep_data = load_data(dataset_dir, episode_ids=episode_ids)

        for ep_id in episode_ids:
            # Resume: an episode is done if it has a .done marker (covers filtered /
            # 0-skill episodes too) or any saved skill npz.
            task_dir = skills_dir / f"task{task_id:02d}"
            marker = _done_marker(skills_dir, ep_id, task_id)
            existing = list(task_dir.glob(f"ep{ep_id:05d}_task{task_id:02d}_skill*.npz"))
            curve_path = curves_dir / f"ep{ep_id:07d}.npz"
            if args.curves_only:
                # Backfill mode: resume keyed by the curve file only, so an already
                # fully-built run (all skills present) still regenerates its curves.
                if args.resume and curve_path.exists():
                    n_skipped_resume += 1
                    n_processed += 1
                    print(f"  [skip] ep{ep_id:05d} curve exists")
                    continue
            elif args.resume and (marker.exists() or existing):
                n_skipped_resume += 1
                n_processed += 1
                total_skills += len(existing)
                if args.write_done_markers and not marker.exists():
                    _touch_done(skills_dir, ep_id, task_id)
                print(f"  [skip] ep{ep_id:05d} already done ({len(existing)} skills)")
                continue

            if episodes_meta[episodes_meta["episode_index"] == ep_id].empty:
                if args.write_done_markers:
                    _touch_done(skills_dir, ep_id, task_id)
                continue
            ep_df = ep_data[ep_data["episode_index"] == ep_id].reset_index(drop=True)
            if len(ep_df) == 0:
                if args.write_done_markers:
                    _touch_done(skills_dir, ep_id, task_id)
                continue

            print(f"  ep{ep_id:05d} ...", end="", flush=True)
            t_ep = time.time()

            try:
                if args.use_cached_curves:
                    if not curve_path.is_file():
                        raise FileNotFoundError(
                            f"Cached curve missing for ep{ep_id:05d}: {curve_path}"
                        )
                    with np.load(curve_path) as curve:
                        vf_replan_ts = curve["replan_ts"].astype(np.int64).tolist()
                        div_cos = curve["div_cos"].astype(np.float32)
                else:
                    if use_dino:
                        cam_frames = {}
                        ep_dino_tokens = load_dino_episode(
                            dino_feature_dir, dino_image_key, ep_id
                        )
                    else:
                        def _load_cam(cam_key):
                            src = get_video_path(
                                dataset_dir, ep_id, cam_key, episodes_meta
                            ).resolve()
                            start_sec, end_sec = get_episode_timestamps(
                                dataset_dir, ep_id, episodes_meta, cam_key
                            )
                            return cam_key, viz.load_episode_frames(src, start_sec, end_sec)

                        cam_frames = _load_camera_frames(camera_keys, _load_cam)
                        ep_dino_tokens = None

                    vf_replan_ts, _, _, div_cos, _, _, _ = run_vf_analysis(
                        policy, preprocessor, ep_df, cam_frames, camera_keys,
                        args.eval_at_step, args.replan_interval,
                        n_gmm_components=args.n_gmm_components,
                        dino_tokens=ep_dino_tokens,
                        probe_type=args.probe_type,
                        action_pca=action_pca,
                        probe_directions=probe_directions,
                        probe_alpha=args.probe_alpha,
                        action_normalizer=action_normalizer,
                        action_mode=args.action_mode,
                        relative_mask=rel_mask,
                        gripper_mode=args.gripper_mode,
                        gripper_indices=gripper_indices,
                        gripper_values=gripper_values,
                        gripper_threshold=args.gripper_threshold,
                        probe_action_indices=probe_action_indices,
                    )
            except Exception as e:
                import traceback
                print(f" [ERROR] {e}")
                traceback.print_exc()
                n_error += 1
                n_processed += 1
                continue

            n_frames = len(ep_df)
            boundaries = _detect_boundaries(
                vf_replan_ts, div_cos, n_frames, args, global_threshold=global_threshold
            )

            if args.dump_curves or args.curves_only or args.use_cached_curves:
                _save_boundary_curve(curves_dir, ep_id, task_id, vf_replan_ts, div_cos,
                                     boundaries, n_frames, args,
                                     global_threshold=global_threshold)

            if args.curves_only:
                n_processed += 1
                print(f" curve saved  [{time.time() - t_ep:.1f}s]")
                continue

            gt_actions = np.stack(ep_df["action"].values[:n_frames])
            states_arr = np.stack(ep_df["observation.state"].values[:n_frames])

            saved = _save_skills(skills_dir, ep_id, task_id, gt_actions, states_arr,
                                  boundaries, args.min_skill_len, args.min_skills)

            elapsed = time.time() - t_ep
            n_processed += 1
            if saved:
                n_saved += 1
                total_skills += len(saved)
                print(f" {len(saved)} skills  [{elapsed:.1f}s]")
            else:
                n_skipped_filter += 1
                print(f" skipped (≤{args.min_skills - 1} valid skills)  [{elapsed:.1f}s]")

            # mark processed (errors above `continue` earlier → no marker → retried)
            if args.write_done_markers:
                _touch_done(skills_dir, ep_id, task_id)

            if wandb_run is not None:
                import wandb
                elapsed_total = time.time() - t_start
                eps_per_sec = n_processed / elapsed_total if elapsed_total > 0 else 0
                remaining = n_total_eps_global - n_processed
                eta_sec = remaining / eps_per_sec if eps_per_sec > 0 else 0
                wandb_run.log({
                    "progress/episodes_done": n_processed,
                    "progress/episodes_total": n_total_eps_global,
                    "progress/episodes_pct": n_processed / n_total_eps_global * 100,
                    "progress/episodes_saved": n_saved,
                    "progress/episodes_skipped_filter": n_skipped_filter,
                    "progress/episodes_skipped_resume": n_skipped_resume,
                    "progress/episodes_error": n_error,
                    "progress/skills_total": total_skills,
                    "progress/eta_min": eta_sec / 60,
                    "progress/current_task_id": task_id,
                    "progress/ep_time_sec": elapsed,
                })

    print(f"\n{'=' * 60}")
    print(f"Done.")
    print(f"  total episodes  : {n_total_eps_global}")
    print(f"  saved           : {n_saved}")
    print(f"  skipped/resume  : {n_skipped_resume}")
    print(f"  skipped/filter  : {n_skipped_filter}")
    print(f"  errors          : {n_error}")
    print(f"  total skills    : {total_skills}")
    print(f"  skills dir      : {skills_dir}")

    if wandb_run is not None:
        wandb_run.summary.update({
            "final/episodes_saved": n_saved,
            "final/episodes_skipped_filter": n_skipped_filter,
            "final/episodes_skipped_resume": n_skipped_resume,
            "final/episodes_error": n_error,
            "final/skills_total": total_skills,
            "final/total_time_min": (time.time() - t_start) / 60,
        })
        wandb_run.finish()

    # Never let an incomplete shard look successful to Slurm.  The submit
    # pipeline uses afterok dependencies, and the global threshold must include
    # every episode curve.  Existing outputs remain resumable on the next run.
    if n_error:
        raise RuntimeError(
            f"Skill dataset build finished with {n_error} episode error(s); "
            "rerun with --resume after fixing or retrying the failed inputs."
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
