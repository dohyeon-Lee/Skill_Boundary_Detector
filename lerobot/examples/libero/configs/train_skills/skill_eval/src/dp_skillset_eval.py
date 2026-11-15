"""DP skill-boundary eval for the train_skills (FSQ_dataset) pipeline.

Visualises how the DP segmented each demo into skills: per episode, the boxed
start/end frames of every skill plus the multimodality (VF cos-divergence) curve
the boundaries were cut from. Uses the shared skillset_boundary_viz renderer and
works off the FSQ_dataset only (skillset npz + curves + raw videos) — i.e. BEFORE
any skillvla dataset exists.

Inputs:
  --skillset_dir  {SKILLSET_DIR} (uses skillset_manifest.json to infer other paths)
  --skills_dir   {SKILLSET_DIR}/skills   (ep*_task*_skill*.npz: frame_start/end, episode_id, task_id)
  --curves_dir   {SKILLSET_DIR}/curves   (ep{ep:07d}.npz; optional → frames-only if absent)
  --dataset_dir  raw LeRobot dataset (videos + meta) for frames
Output:
  {out_dir}/index.html
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

import numpy as np

# Shared evaluation renderers live beside this module in skill_eval/src.
EVAL_SHARED_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_SHARED_SRC))
BUILD_DATA_SRC = EVAL_SHARED_SRC.parents[1] / "build_data" / "src"
sys.path.insert(0, str(BUILD_DATA_SRC))

from codebook_visualizer import (  # noqa: E402
    _episode_row,
    _load_episodes_meta,
    _read_episode_camera_clip,
    _read_episode_clip,
    _resolve_image_key,
    _video_path,
)
from skillset_boundary_viz import (  # noqa: E402
    build_gmm_visualizations,
    load_boundary_curve,
    render_skillset_card,
    save_gallery,
)
from sbd_multimodality import (  # noqa: E402
    covariance_gated_angular_chord_curve,
    covariance_gated_cosine_curve,
    gmm_overlap_entropy_curve,
    strong_covariance_gated_cosine_curve,
    within_normalized_l2_curve,
)

_SKILL_RE = re.compile(r"ep(\d+)_task(\d+)_skill(\d+)\.npz$")
_CURVE_RE = re.compile(r"ep(\d+)\.npz$")


def index_skillset(skills_dir: Path) -> tuple[dict[int, int], dict[int, list[Path]]]:
    """Scan skills_dir filenames (no npz load) → (episode→task_id, episode→[skill npz paths])."""
    ep_task: dict[int, int] = {}
    ep_files: dict[int, list[Path]] = defaultdict(list)
    for path in skills_dir.rglob("ep*_task*_skill*.npz"):
        m = _SKILL_RE.search(path.name)
        if not m:
            continue
        ep, task = int(m.group(1)), int(m.group(2))
        ep_task[ep] = task
        ep_files[ep].append(path)
    return ep_task, ep_files


def _resolve_indices(indices, dim: int) -> tuple[int, ...]:
    resolved = []
    for raw in indices:
        index = int(raw)
        if index < 0:
            index += dim
        if index < 0 or index >= dim:
            raise IndexError(f"Action index {raw} is out of range for action_dim={dim}")
        if index not in resolved:
            resolved.append(index)
    return tuple(resolved)


def load_episode_skills(paths: list[Path], gripper_indices=()):
    """Load skill bounds and reconstruct raw action gripper signals on episode time."""
    records = []
    for p in paths:
        with np.load(str(p)) as z:
            records.append((
                int(z["frame_start"]),
                int(z["frame_end"]),
                np.asarray(z["actions"], dtype=np.float32),
            ))
    records.sort(key=lambda t: t[0])
    skills = [(start, end, None) for start, end, _actions in records]
    if not records or not gripper_indices:
        return skills, None, ()

    action_dim = int(records[0][2].shape[-1])
    resolved = _resolve_indices(gripper_indices, action_dim)
    n_frames = max(end for _start, end, _actions in records)
    signal = np.full((n_frames, len(resolved)), np.nan, dtype=np.float32)
    for start, end, actions in records:
        length = min(end - start, len(actions))
        signal[start:start + length] = actions[:length, list(resolved)]
    return skills, signal, resolved


def global_curve_mean(curves_dir: Path, key: str) -> tuple[float, int]:
    """Pool one saved diagnostic over every terminal-valid curve anchor."""
    total = 0.0
    count = 0
    for path in sorted(Path(curves_dir).glob("ep*.npz")):
        with np.load(path, allow_pickle=False) as curve:
            if key not in curve.files:
                raise KeyError(f"{path} does not contain {key!r}")
            values = np.asarray(curve[key], dtype=np.float64).reshape(-1)
            valid_count = (
                int(np.asarray(curve["threshold_valid_count"]).reshape(-1)[0])
                if "threshold_valid_count" in curve.files
                else len(values)
            )
            valid = values[:valid_count]
            finite = valid[np.isfinite(valid)]
            total += float(finite.sum())
            count += int(len(finite))
    if count == 0:
        raise ValueError(f"No finite {key!r} values found under {curves_dir}")
    return total / count, count


def global_curve_power_mean(
    curves_dir: Path,
    key: str,
    *,
    source_power: float,
    target_power: float = 1.0,
) -> tuple[float, int]:
    """Pool a saved powered score after mapping it into another power space."""
    if source_power <= 0.0 or target_power <= 0.0:
        raise ValueError("source_power and target_power must be positive")
    total = 0.0
    count = 0
    exponent = float(target_power) / float(source_power)
    for path in sorted(Path(curves_dir).glob("ep*.npz")):
        with np.load(path, allow_pickle=False) as curve:
            if key not in curve.files:
                raise KeyError(f"{path} does not contain {key!r}")
            values = np.asarray(curve[key], dtype=np.float64).reshape(-1)
            valid_count = (
                int(np.asarray(curve["threshold_valid_count"]).reshape(-1)[0])
                if "threshold_valid_count" in curve.files
                else len(values)
            )
            valid = values[:valid_count]
            finite = valid[np.isfinite(valid)]
            mapped = np.power(np.maximum(finite, 0.0), exponent)
            total += float(mapped.sum())
            count += int(len(mapped))
    if count == 0:
        raise ValueError(f"No finite {key!r} values found under {curves_dir}")
    return total / count, count


def cached_gmm_metric_curves(
    raw_dirs,
    *,
    score_metric: str,
    initial_mask_ends: dict[int, int] | None = None,
    terminal_mask_starts: dict[int, int] | None = None,
    components: int,
    min_effective_samples: float = 0.0,
    terminal_mask_frames: int,
    terminal_mask_from_episode_end: bool = False,
    smooth_window: int,
    savgol_polyorder: int,
    score_power: float,
) -> tuple[dict[int, dict], float, int]:
    """Build an alternate GMM score from cached descriptor clouds.

    The global threshold is calibrated over every unique episode cache found in
    ``raw_dirs``. This is intentionally a CPU-only report path: no DP inference
    is repeated while testing a different statistic on the same action clouds.
    """
    if int(terminal_mask_frames) < 0:
        raise ValueError("terminal_mask_frames must be non-negative")
    if not np.isfinite(min_effective_samples) or float(min_effective_samples) < 0.0:
        raise ValueError("min_effective_samples must be finite and non-negative")
    if not np.isfinite(score_power) or float(score_power) <= 0.0:
        raise ValueError("score_power must be finite and positive")
    scorers = {
        "within_normalized_l2": within_normalized_l2_curve,
        "covariance_gated_cosine": covariance_gated_cosine_curve,
        "covariance_gated_angular_chord": covariance_gated_angular_chord_curve,
        "gmm_overlap_entropy": gmm_overlap_entropy_curve,
        "strong_covariance_gated_cosine": strong_covariance_gated_cosine_curve,
    }
    if score_metric not in scorers:
        raise ValueError(
            f"Unsupported cached GMM score metric {score_metric!r}; "
            f"expected one of {sorted(scorers)}"
        )
    scorer = scorers[score_metric]

    paths: dict[int, Path] = {}
    for raw_dir in raw_dirs:
        for path in sorted(Path(raw_dir).glob("ep*.npz")):
            match = _CURVE_RE.search(path.name)
            if match is None:
                continue
            episode_id = int(match.group(1))
            previous = paths.get(episode_id)
            if previous is not None and previous.resolve() != path.resolve():
                raise ValueError(
                    f"Duplicate GMM cache for episode {episode_id}: {previous} and {path}"
                )
            paths[episode_id] = path
    if not paths:
        raise FileNotFoundError(
            "No epXXXXXXX.npz descriptor cache was found under: "
            + ", ".join(str(value) for value in raw_dirs)
        )

    curves: dict[int, dict] = {}
    threshold_total = 0.0
    threshold_count = 0
    for position, (episode_id, path) in enumerate(sorted(paths.items()), start=1):
        with np.load(path, allow_pickle=False) as payload:
            timestamps = np.asarray(payload["replan_ts"], dtype=np.int64).reshape(-1)
            descriptors = np.asarray(payload["descriptors"], dtype=np.float32)
            n_frames = int(np.asarray(payload["n_frames"]).reshape(-1)[0])
        if len(timestamps) != len(descriptors) or not len(timestamps):
            raise ValueError(
                f"Invalid GMM descriptor cache {path}: "
                f"timestamps={len(timestamps)}, descriptors={len(descriptors)}"
            )
        print(
            f"[{score_metric} GMM {position}/{len(paths)}] ep{episode_id}",
            flush=True,
        )
        raw_score = scorer(
            descriptors,
            components=int(components),
            covariance="diag",
            n_init=10,
            max_iter=300,
            min_effective_samples=float(min_effective_samples),
            weighted=True,
        ).astype(np.float64)
        terminal_mask_start_frame = None
        if int(terminal_mask_frames):
            if terminal_mask_from_episode_end:
                cutoff = int(n_frames) - int(terminal_mask_frames)
                valid_count = int(np.searchsorted(timestamps, cutoff, side="left"))
            else:
                cutoff = int(timestamps[-1]) - int(terminal_mask_frames)
                valid_count = int(np.searchsorted(timestamps, cutoff, side="right"))
            terminal_mask_start_frame = int(cutoff)
        else:
            valid_count = len(timestamps)
        semantic_terminal_start = None
        if terminal_mask_starts is not None and episode_id in terminal_mask_starts:
            semantic_terminal_start = int(terminal_mask_starts[episode_id])
            valid_count = min(
                valid_count,
                int(np.searchsorted(timestamps, semantic_terminal_start, side="left")),
            )
        if valid_count < 1:
            raise ValueError(
                f"terminal_mask_frames={terminal_mask_frames} masks every anchor in {path}"
            )
        start_index = 0
        if initial_mask_ends is not None:
            if episode_id not in initial_mask_ends:
                raise ValueError(
                    f"Initial-motion mask is missing cached episode {episode_id}"
                )
            start_index = int(
                np.searchsorted(
                    timestamps[:valid_count],
                    int(initial_mask_ends[episode_id]),
                    side="right",
                )
            )
            if start_index >= valid_count:
                raise ValueError(
                    f"Initial mask through frame {initial_mask_ends[episode_id]} "
                    f"leaves no valid anchors in {path}"
                )
        display_base = _savgol(
            raw_score[:valid_count],
            int(smooth_window),
            int(savgol_polyorder),
        )
        display_transformed = (
            display_base
            if np.isclose(score_power, 1.0)
            else np.power(np.maximum(display_base, 0.0), float(score_power))
        )
        smooth = np.full(
            len(raw_score), float(display_transformed[-1]), dtype=np.float64
        )
        smooth[:valid_count] = display_transformed
        if start_index:
            eligible_base = _savgol(
                raw_score[start_index:valid_count],
                int(smooth_window),
                int(savgol_polyorder),
            )
            eligible_transformed = (
                eligible_base
                if np.isclose(score_power, 1.0)
                else np.power(np.maximum(eligible_base, 0.0), float(score_power))
            )
            # Prevent the excluded initialization burst from leaking through
            # Savitzky-Golay smoothing into the first eligible peak candidates.
            smooth[start_index:valid_count] = eligible_transformed
        finite = smooth[start_index:valid_count]
        finite = finite[np.isfinite(finite)]
        threshold_total += float(finite.sum())
        threshold_count += int(len(finite))
        curve = {
            "replan_ts": timestamps,
            "div_cos": raw_score.astype(np.float32),
            "sg_vals": smooth.astype(np.float32),
            "threshold_valid_count": np.asarray(valid_count, dtype=np.int64),
            "terminal_mask_frames": np.asarray(terminal_mask_frames, dtype=np.int64),
            "score_power": np.asarray(score_power, dtype=np.float32),
            "score_metric": np.asarray(score_metric),
            "gmm_min_effective_samples": np.asarray(
                min_effective_samples, dtype=np.float32
            ),
            "n_frames": np.asarray(n_frames, dtype=np.int64),
            "last_valid_anchor": np.asarray(timestamps[-1], dtype=np.int64),
        }
        if terminal_mask_start_frame is not None:
            curve["terminal_mask_start_frame"] = np.asarray(
                terminal_mask_start_frame, dtype=np.int64
            )
        if semantic_terminal_start is not None:
            curve["terminal_release_start_frame"] = np.asarray(
                semantic_terminal_start, dtype=np.int64
            )
            curve["terminal_mask_start_frame"] = np.asarray(
                semantic_terminal_start, dtype=np.int64
            )
            curve["terminal_mask_mode"] = np.asarray("abc_terminal_release")
            curve["terminal_release_detected"] = np.asarray(True)
        elif terminal_mask_starts is not None:
            curve["terminal_release_detected"] = np.asarray(False)
        if initial_mask_ends is not None:
            curve["threshold_valid_start"] = np.asarray(start_index, dtype=np.int64)
            curve["initial_mask_until_frame"] = np.asarray(
                initial_mask_ends[episode_id], dtype=np.int64
            )
        curves[episode_id] = curve
    if threshold_count == 0:
        raise ValueError(f"No finite {score_metric} GMM scores were produced")
    threshold = threshold_total / threshold_count
    for curve in curves.values():
        curve["mean_val"] = np.asarray(threshold, dtype=np.float32)
        curve["threshold_val"] = np.asarray(threshold, dtype=np.float32)
        curve["threshold_pool_points"] = np.asarray(threshold_count, dtype=np.int64)
    return curves, threshold, threshold_count


def detect_initial_motion_start(
    states: np.ndarray,
    *,
    excluded_indices=(),
    velocity_threshold: float = 0.005,
    sustain_frames: int = 5,
) -> int:
    """Locate the first sustained arm motion in an episode."""
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError(f"states must have shape (T, D) with T >= 2, got {values.shape}")
    if not np.isfinite(velocity_threshold) or velocity_threshold <= 0.0:
        raise ValueError("velocity_threshold must be finite and positive")
    if int(sustain_frames) < 1:
        raise ValueError("sustain_frames must be positive")
    excluded = set(_resolve_indices(excluded_indices, values.shape[1]))
    arm_indices = [index for index in range(values.shape[1]) if index not in excluded]
    if not arm_indices:
        raise ValueError("Initial-motion detection has no state dimensions after exclusions")
    velocity = np.linalg.norm(np.diff(values[:, arm_indices], axis=0), axis=1)
    moving = velocity > float(velocity_threshold)
    run = int(sustain_frames)
    if len(moving) >= run:
        sustained = np.convolve(
            moving.astype(np.int8), np.ones(run, dtype=np.int8), mode="valid"
        )
        starts = np.flatnonzero(sustained == run)
        if len(starts):
            # velocity[i] describes the transition from state i to state i + 1.
            return int(starts[0] + 1)
    # Deterministically guard even an extremely static or malformed episode.
    return 0


def abc_initial_motion_masks(
    dataset_dir: Path,
    episode_ids,
    *,
    excluded_indices=(),
    after_start_frames: int = 100,
    velocity_threshold: float = 0.005,
    sustain_frames: int = 5,
    reset_jump_window_frames: int = 10,
) -> tuple[dict[int, int], dict[int, int], dict[int, float]]:
    """Return motion starts, base mask ends, and early arm-speed maxima."""
    from lerobot.datasets.abc_native_io import iter_abc_native_state_action

    if int(after_start_frames) < 0:
        raise ValueError("after_start_frames must be non-negative")
    if int(reset_jump_window_frames) < 1:
        raise ValueError("reset_jump_window_frames must be positive")
    starts: dict[int, int] = {}
    mask_ends: dict[int, int] = {}
    reset_jump_speeds: dict[int, float] = {}
    wanted = {int(value) for value in episode_ids}
    for episode_id, _actions, states in iter_abc_native_state_action(
        dataset_dir, episode_ids=wanted
    ):
        start = detect_initial_motion_start(
            states,
            excluded_indices=excluded_indices,
            velocity_threshold=velocity_threshold,
            sustain_frames=sustain_frames,
        )
        episode_id = int(episode_id)
        excluded = set(_resolve_indices(excluded_indices, states.shape[1]))
        arm_indices = [index for index in range(states.shape[1]) if index not in excluded]
        arm_velocity = np.linalg.norm(
            np.diff(np.asarray(states, dtype=np.float64)[:, arm_indices], axis=0), axis=1
        )
        starts[episode_id] = start
        mask_ends[episode_id] = start + int(after_start_frames)
        reset_jump_speeds[episode_id] = float(
            np.max(arm_velocity[: int(reset_jump_window_frames)], initial=0.0)
        )
    missing = sorted(wanted - set(mask_ends))
    if missing:
        raise ValueError(f"Initial-motion masks are missing {len(missing)} episodes: {missing[:8]}")
    return starts, mask_ends, reset_jump_speeds


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive ``(start, end)`` runs from a one-dimensional mask."""
    values = np.asarray(mask, dtype=bool).reshape(-1)
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    edges = np.diff(padded)
    return list(
        zip(
            np.flatnonzero(edges == 1).tolist(),
            (np.flatnonzero(edges == -1) - 1).tolist(),
            strict=True,
        )
    )


def detect_home_return_start(
    states: np.ndarray,
    *,
    excluded_indices=(),
    joint_scale: np.ndarray | None = None,
    home_reference_frames: int = 30,
    distance_smooth_frames: int = 31,
    progress_horizon_frames: int = 15,
    max_search_frames: int = 300,
    final_home_ratio_max: float = 0.30,
) -> tuple[int | None, dict[str, float | str]]:
    """Detect the final sustained return toward the episode's initial arm pose.

    The initial median is an episode-specific home reference.  A return is only
    accepted when the episode finishes near that reference and a sufficiently
    large, sustained decrease in home distance reaches the final home band.
    """
    from scipy.signal import savgol_filter

    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or len(values) < 60:
        return None, {"status": "too_short"}
    excluded = set(_resolve_indices(excluded_indices, values.shape[1]))
    arm_indices = np.asarray(
        [index for index in range(values.shape[1]) if index not in excluded],
        dtype=np.int64,
    )
    if not len(arm_indices):
        raise ValueError("Home-return detection has no state dimensions after exclusions")
    arm = values[:, arm_indices]
    reference_count = min(int(home_reference_frames), max(5, len(arm) // 10))
    home = np.median(arm[:reference_count], axis=0)
    if joint_scale is None:
        scale = np.ones(len(arm_indices), dtype=np.float64)
    else:
        raw_scale = np.asarray(joint_scale, dtype=np.float64).reshape(-1)
        if len(raw_scale) == values.shape[1]:
            scale = raw_scale[arm_indices]
        elif len(raw_scale) == len(arm_indices):
            scale = raw_scale
        else:
            raise ValueError(
                f"joint_scale dim {len(raw_scale)} is incompatible with states dim {values.shape[1]}"
            )
        scale = np.maximum(scale, 0.05)
    distance = np.sqrt(np.mean(np.square((arm - home) / scale), axis=1))
    smooth_window = min(
        int(distance_smooth_frames) | 1,
        len(distance) if len(distance) % 2 else len(distance) - 1,
    )
    smoothed = (
        savgol_filter(distance, window_length=smooth_window, polyorder=2)
        if smooth_window >= 5
        else distance.copy()
    )
    excursion = float(np.max(smoothed))
    final_distance = float(np.median(distance[-reference_count:]))
    home_noise = float(np.percentile(distance[:reference_count], 90))
    home_band = max(3.0 * home_noise, 2.0 * final_distance, 0.12 * excursion, 0.04)
    final_ratio = final_distance / max(excursion, np.finfo(np.float64).eps)
    if final_ratio > float(final_home_ratio_max):
        return None, {
            "status": "final_not_home",
            "final_ratio": final_ratio,
            "excursion": excursion,
        }

    inside_home = smoothed <= home_band
    home_runs = [
        run
        for run in _true_runs(inside_home)
        if run[1] >= len(arm) - reference_count and run[1] - run[0] + 1 >= 15
    ]
    if not home_runs:
        return None, {"status": "no_final_home_hold", "final_ratio": final_ratio}
    arrival = int(home_runs[-1][0])

    horizon = int(progress_horizon_frames)
    progress = smoothed[:-horizon] - smoothed[horizon:]
    toward_home = progress > max(0.015, 0.015 * excursion)
    search_start = max(0, arrival - int(max_search_frames))
    search_end = max(search_start, arrival)
    tail = toward_home[search_start:search_end].copy()
    runs = _true_runs(tail)
    if not runs:
        return None, {"status": "no_toward_home_run", "final_ratio": final_ratio}
    # Bridge brief pauses inside one return motion, but not a separate task phase.
    for (_left_start, left_end), (right_start, _right_end) in zip(
        runs[:-1], runs[1:], strict=True
    ):
        if right_start - left_end - 1 <= horizon:
            tail[left_end + 1:right_start] = True
    candidates = []
    for local_start, local_end in _true_runs(tail):
        start = search_start + local_start
        end = min(len(arm) - 1, search_start + local_end + horizon)
        drop = float(smoothed[start] - smoothed[min(arrival, len(arm) - 1)])
        if local_end - local_start + 1 >= 10 and drop >= max(0.12, 0.15 * excursion):
            candidates.append((start, end, drop))
    if not candidates:
        return None, {"status": "weak_return", "final_ratio": final_ratio}

    start, _end, drop = candidates[-1]
    # Forward progress compares t with t+horizon and therefore becomes true up
    # to one horizon before motion begins. Compensate back to the motion onset.
    start = min(int(arrival), int(start) + horizon)
    return int(start), {
        "status": "ok",
        "final_ratio": final_ratio,
        "excursion": excursion,
        "arrival_frame": float(arrival),
        "distance_drop": drop,
    }


def detect_terminal_gripper_release_start(
    states: np.ndarray,
    *,
    gripper_indices,
    search_frames: int = 450,
    progress_horizon_frames: int = 10,
    bridge_gap_frames: int = 60,
) -> tuple[int | None, dict[str, float | str]]:
    """Locate the final lasting gripper opening before terminal motion.

    ABC grippers are continuous joint positions whose larger value is open.
    The detector looks for sustained positive progress that produces a lasting
    opening at episode end.  Forward progress is shifted by its horizon so the
    returned frame is immediately before the physical release begins.
    """
    from scipy.signal import savgol_filter

    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or len(values) < 30:
        return None, {"status": "too_short"}
    resolved = _resolve_indices(gripper_indices, values.shape[1])
    if not resolved:
        raise ValueError("Terminal-release detection requires gripper indices")
    horizon = int(progress_horizon_frames)
    if horizon < 1 or int(search_frames) < 1 or int(bridge_gap_frames) < 0:
        raise ValueError("Invalid terminal-release detection window")

    search_start = max(0, len(values) - int(search_frames))
    candidates: list[tuple[int, int, float, float, int, float]] = []
    for index in resolved:
        signal = values[:, index]
        smooth_window = min(11, len(signal) if len(signal) % 2 else len(signal) - 1)
        smoothed = (
            savgol_filter(signal, window_length=smooth_window, polyorder=2)
            if smooth_window >= 5
            else signal.copy()
        )
        dynamic_range = float(
            np.percentile(smoothed, 98) - np.percentile(smoothed, 2)
        )
        progress_threshold = max(0.008, 0.025 * dynamic_range)
        progress = smoothed[horizon:] - smoothed[:-horizon]
        opening = progress > progress_threshold
        opening[: max(0, search_start - horizon)] = False

        runs = _true_runs(opening)
        if runs:
            bridged = opening.copy()
            for (_left_start, left_end), (right_start, _right_end) in zip(
                runs[:-1], runs[1:], strict=True
            ):
                if right_start - left_end - 1 <= int(bridge_gap_frames):
                    bridged[left_end + 1:right_start] = True
            runs = _true_runs(bridged)

        final_open = float(np.median(signal[-20:]))
        for run_start, run_end in runs:
            onset = min(len(signal) - 1, int(run_start) + horizon)
            end = min(len(signal) - 1, int(run_end) + horizon)
            lasting_gain = final_open - float(smoothed[onset])
            local_gain = float(
                np.max(smoothed[onset:min(len(signal), end + 61)]) - smoothed[onset]
            )
            if lasting_gain < max(0.04, 0.08 * dynamic_range):
                continue
            if local_gain < max(0.035, 0.07 * dynamic_range):
                continue
            local_stop = min(len(smoothed), end + 61)
            post_peak = onset + int(np.argmax(smoothed[onset:local_stop]))
            post_reclose = float(
                smoothed[post_peak] - np.min(smoothed[post_peak:])
            )
            # A temporary opening followed by another grasp is part of the
            # task, not the terminal release.  Only accept openings that remain
            # open (apart from small settling motion) through episode end.
            if post_reclose > max(0.08, 0.35 * local_gain):
                continue
            candidates.append(
                (
                    int(onset),
                    int(index),
                    lasting_gain,
                    local_gain,
                    int(end),
                    post_reclose,
                )
            )

    if not candidates:
        return None, {"status": "no_terminal_release"}
    # In a bimanual episode one arm can release its object substantially before
    # the other and then remain open through the end.  Selecting by opening
    # magnitude masks the still-active arm (and valid task motion) too early.
    # The semantic terminal release is the last accepted lasting opening.
    onset, index, lasting_gain, local_gain, end, post_reclose = max(
        candidates, key=lambda item: item[0]
    )
    return onset, {
        "status": "release",
        "gripper_index": float(index),
        "candidate_count": float(len(candidates)),
        "earliest_candidate_frame": float(min(item[0] for item in candidates)),
        "lasting_open_gain": lasting_gain,
        "local_open_gain": local_gain,
        "release_end_frame": float(end),
        "post_release_reclose": post_reclose,
    }


def abc_home_return_starts(
    dataset_dir: Path,
    episode_ids,
    *,
    excluded_indices=(),
    cache_path: Path | None = None,
) -> tuple[dict[int, int], dict[int, str]]:
    """Detect or load per-episode ABC home-return boundaries."""
    wanted = {int(value) for value in episode_ids}
    if cache_path is not None and cache_path.is_file():
        payload = json.loads(cache_path.read_text())
        cached = {int(key): value for key, value in payload.get("episodes", {}).items()}
        if wanted.issubset(cached):
            starts = {
                episode_id: int(cached[episode_id]["start"])
                for episode_id in wanted
                if cached[episode_id].get("start") is not None
            }
            failures = {
                episode_id: str(cached[episode_id].get("status", "unknown"))
                for episode_id in wanted - set(starts)
            }
            print(
                f"Home-return cache: {len(starts)}/{len(wanted)} detected from {cache_path}",
                flush=True,
            )
            return starts, failures

    from lerobot.datasets.abc_native_io import iter_abc_native_state_action

    stats = json.loads((Path(dataset_dir) / "meta" / "stats.json").read_text())
    state_scale = np.asarray(stats["observation.state"]["std"], dtype=np.float64)
    starts: dict[int, int] = {}
    failures: dict[int, str] = {}
    cache_episodes: dict[str, dict] = {}
    for episode_id, _actions, states in iter_abc_native_state_action(
        dataset_dir, episode_ids=wanted
    ):
        episode_id = int(episode_id)
        start, diagnostics = detect_home_return_start(
            states,
            excluded_indices=excluded_indices,
            joint_scale=state_scale,
        )
        status = str(diagnostics.get("status", "unknown"))
        if start is None:
            failures[episode_id] = status
        else:
            starts[episode_id] = int(start)
        cache_episodes[str(episode_id)] = {
            "start": None if start is None else int(start),
            "status": status,
            "n_frames": int(len(states)),
            "diagnostics": diagnostics,
        }
    missing = wanted - set(starts) - set(failures)
    if missing:
        raise ValueError(f"Home-return detector did not visit episodes: {sorted(missing)[:8]}")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "dataset_dir": str(Path(dataset_dir).resolve()),
                    "excluded_indices": [int(value) for value in excluded_indices],
                    "episodes": cache_episodes,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        os.replace(temporary, cache_path)
    durations = [
        cache_episodes[str(episode_id)]["n_frames"] - start
        for episode_id, start in starts.items()
    ]
    summary = (
        np.percentile(durations, [10, 50, 90]).round(1).tolist()
        if durations
        else []
    )
    print(
        f"Home-return detection: {len(starts)}/{len(wanted)} episodes; "
        f"duration p10/p50/p90={summary} frames; failures={len(failures)}",
        flush=True,
    )
    return starts, failures


def abc_terminal_release_starts(
    dataset_dir: Path,
    episode_ids,
    *,
    gripper_indices,
    excluded_indices=(),
    pre_release_offset_frames: int = 16,
    cache_path: Path | None = None,
) -> tuple[dict[int, int], dict[int, str]]:
    """Return release-before-terminal boundaries, falling back to home return."""
    if int(pre_release_offset_frames) < 0:
        raise ValueError("pre_release_offset_frames must be non-negative")
    wanted = {int(value) for value in episode_ids}
    if cache_path is not None and cache_path.is_file():
        payload = json.loads(cache_path.read_text())
        cached = {int(key): value for key, value in payload.get("episodes", {}).items()}
        if (
            int(payload.get("schema_version", 0)) == 5
            and payload.get("detection_mode") == "terminal_gripper_release"
            and int(payload.get("pre_release_offset_frames", -1))
            == int(pre_release_offset_frames)
            and wanted.issubset(cached)
        ):
            starts = {
                episode_id: int(cached[episode_id]["start"])
                for episode_id in wanted
                if cached[episode_id].get("start") is not None
            }
            failures = {
                episode_id: str(cached[episode_id].get("status", "unknown"))
                for episode_id in wanted - set(starts)
            }
            releases = sum(
                cached[episode_id].get("status") == "release" for episode_id in wanted
            )
            print(
                f"Terminal-release cache: {releases} release, "
                f"{len(starts) - releases} home fallback, {len(failures)} failed "
                f"from {cache_path}",
                flush=True,
            )
            return starts, failures

    from lerobot.datasets.abc_native_io import iter_abc_native_state_action

    stats = json.loads((Path(dataset_dir) / "meta" / "stats.json").read_text())
    state_scale = np.asarray(stats["observation.state"]["std"], dtype=np.float64)
    starts: dict[int, int] = {}
    failures: dict[int, str] = {}
    cache_episodes: dict[str, dict] = {}
    release_count = 0
    fallback_count = 0
    for episode_id, _actions, states in iter_abc_native_state_action(
        dataset_dir, episode_ids=wanted
    ):
        episode_id = int(episode_id)
        release_start, release_diagnostics = detect_terminal_gripper_release_start(
            states,
            gripper_indices=gripper_indices,
        )
        home_start, home_diagnostics = detect_home_return_start(
            states,
            excluded_indices=excluded_indices,
            joint_scale=state_scale,
        )
        if release_start is not None:
            start = max(0, int(release_start) - int(pre_release_offset_frames))
            status = "release"
            release_count += 1
        elif home_start is not None:
            start = int(home_start)
            status = "fallback_home_return"
            fallback_count += 1
        else:
            start = None
            status = "no_release_or_home_return"
            failures[episode_id] = status
        if start is not None:
            starts[episode_id] = start
        cache_episodes[str(episode_id)] = {
            "start": start,
            "status": status,
            "release_start": None if release_start is None else int(release_start),
            "home_return_start": None if home_start is None else int(home_start),
            "n_frames": int(len(states)),
            "release_diagnostics": release_diagnostics,
            "home_return_diagnostics": home_diagnostics,
        }
    missing = wanted - set(starts) - set(failures)
    if missing:
        raise ValueError(
            f"Terminal-release detector did not visit episodes: {sorted(missing)[:8]}"
        )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": 5,
                    "detection_mode": "terminal_gripper_release",
                    "pre_release_offset_frames": int(pre_release_offset_frames),
                    "dataset_dir": str(Path(dataset_dir).resolve()),
                    "gripper_indices": [int(value) for value in gripper_indices],
                    "episodes": cache_episodes,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        os.replace(temporary, cache_path)
    durations = [
        cache_episodes[str(episode_id)]["n_frames"] - start
        for episode_id, start in starts.items()
    ]
    summary = (
        np.percentile(durations, [10, 50, 90]).round(1).tolist()
        if durations
        else []
    )
    print(
        f"Terminal-release detection: {release_count} release, "
        f"{fallback_count} home fallback, {len(failures)} failed; "
        f"masked duration p10/p50/p90={summary} frames",
        flush=True,
    )
    return starts, failures


def _savgol(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Apply Savitzky-Golay smoothing, including SciPy's even windows."""
    from scipy.signal import savgol_filter

    values = np.asarray(values, dtype=np.float64)
    if int(window) <= 1:
        return values.copy()
    win = min(int(window), len(values))
    if win <= int(polyorder):
        return values.copy()
    return savgol_filter(values, window_length=win, polyorder=int(polyorder))


def initial_masked_score_curve(
    curve: dict,
    *,
    initial_mask_until_frame: int,
    score_power: float,
    smooth_window: int,
    savgol_polyorder: int,
) -> tuple[np.ndarray, int, int]:
    """Re-smooth one curve strictly between its initial and terminal masks."""
    timestamps = np.asarray(curve["replan_ts"], dtype=np.int64).reshape(-1)
    raw = np.asarray(curve["div_cos"], dtype=np.float64).reshape(-1)
    valid_count = int(np.asarray(curve["threshold_valid_count"]).reshape(-1)[0])
    if len(timestamps) != len(raw) or not 0 < valid_count <= len(raw):
        raise ValueError(
            "Invalid SBD curve for initial masking: "
            f"timestamps={len(timestamps)}, raw={len(raw)}, terminal_valid={valid_count}."
        )
    start_index = int(
        np.searchsorted(
            timestamps[:valid_count], int(initial_mask_until_frame), side="right"
        )
    )
    if start_index >= valid_count:
        raise ValueError(
            f"Initial mask through frame {initial_mask_until_frame} leaves no valid anchors"
        )
    # Keep the excluded prefix visible, but independently re-smooth the eligible
    # slice so its large reset/lift peak cannot leak across the mask boundary.
    display = _savgol(raw[:valid_count], smooth_window, savgol_polyorder)
    eligible = _savgol(raw[start_index:valid_count], smooth_window, savgol_polyorder)
    transformed_display = np.power(np.maximum(display, 0.0), float(score_power))
    transformed_eligible = np.power(np.maximum(eligible, 0.0), float(score_power))
    result = np.full(len(raw), float(transformed_display[-1]), dtype=np.float64)
    result[:valid_count] = transformed_display
    result[start_index:valid_count] = transformed_eligible
    return result, start_index, valid_count


def global_initial_masked_score_mean(
    curves_dir: Path,
    mask_ends: dict[int, int],
    *,
    score_power: float,
    smooth_window: int,
    savgol_polyorder: int,
) -> tuple[float, int]:
    """Pool the re-smoothed power-space score outside both episode-edge masks."""
    total = 0.0
    count = 0
    seen: set[int] = set()
    for path in sorted(Path(curves_dir).glob("ep*.npz")):
        match = _CURVE_RE.search(path.name)
        if match is None:
            continue
        episode_id = int(match.group(1))
        if episode_id not in mask_ends:
            continue
        with np.load(path, allow_pickle=False) as loaded:
            curve = {key: loaded[key] for key in loaded.files}
        score, start_index, valid_count = initial_masked_score_curve(
            curve,
            initial_mask_until_frame=mask_ends[episode_id],
            score_power=score_power,
            smooth_window=smooth_window,
            savgol_polyorder=savgol_polyorder,
        )
        finite = score[start_index:valid_count]
        if not len(finite) or not np.isfinite(finite).all():
            raise ValueError(f"Invalid initial-masked score curve: {path}")
        total += float(finite.sum())
        count += int(len(finite))
        seen.add(episode_id)
    missing = sorted(set(mask_ends) - seen)
    if missing:
        raise ValueError(f"No saved SBD curve for {len(missing)} masked episodes: {missing[:8]}")
    if count == 0:
        raise ValueError("No initial/terminal-valid SBD anchors were found")
    return total / count, count


def recut_saved_standard_curves(
    curves_dir: Path,
    *,
    home_return_starts: dict[int, int] | None,
    smooth_window: int,
    savgol_polyorder: int,
    score_power: float,
    replan_interval: int | None = None,
    terminal_mask_frames: int | None = None,
) -> tuple[dict[int, dict], float, int]:
    """Re-smooth saved production scores and optionally change their cadence.

    A larger cadence that is an integer multiple of the saved cadence can be
    reproduced exactly by retaining aligned anchors.  A smaller or unaligned
    cadence needs fresh DP inference and is rejected here.
    """
    curves: dict[int, dict] = {}
    threshold_total = 0.0
    threshold_count = 0
    for path in sorted(Path(curves_dir).glob("ep*.npz")):
        match = _CURVE_RE.search(path.name)
        if match is None:
            continue
        episode_id = int(match.group(1))
        with np.load(path, allow_pickle=False) as loaded:
            curve = {key: loaded[key] for key in loaded.files}
        timestamps = np.asarray(curve["replan_ts"], dtype=np.int64).reshape(-1)
        raw = np.asarray(curve["div_cos"], dtype=np.float64).reshape(-1)
        built_valid_count = int(
            np.asarray(curve["threshold_valid_count"]).reshape(-1)[0]
        )
        if len(timestamps) != len(raw) or not 0 < built_valid_count <= len(raw):
            raise ValueError(f"Invalid saved standard curve: {path}")
        original_timestamps = timestamps
        original_length = len(timestamps)
        complete_future_last_anchor = int(
            np.asarray(
                curve.get("last_valid_anchor", original_timestamps[-1])
            ).reshape(-1)[0]
        )
        if replan_interval is not None:
            target_interval = int(replan_interval)
            if target_interval < 1:
                raise ValueError("replan_interval must be positive")
            if len(timestamps) >= 2:
                steps = np.diff(timestamps)
                source_interval = int(steps[0])
                if source_interval < 1 or not np.all(steps == source_interval):
                    raise ValueError(
                        f"Saved replan timestamps are not uniformly spaced in {path}"
                    )
                if (
                    target_interval < source_interval
                    or target_interval % source_interval != 0
                ):
                    raise ValueError(
                        f"replan_interval={target_interval} cannot be recovered from "
                        f"saved interval={source_interval}; fresh DP inference is required"
                    )
                keep = np.arange(
                    0,
                    len(timestamps),
                    target_interval // source_interval,
                    dtype=np.int64,
                )
                for key, value in list(curve.items()):
                    array = np.asarray(value)
                    if array.ndim >= 1 and array.shape[0] == original_length:
                        curve[key] = array[keep]
                timestamps = original_timestamps[keep]
                raw = raw[keep]
            curve["replan_interval"] = np.asarray(
                target_interval, dtype=np.int64
            )

        if terminal_mask_frames is not None:
            fixed_terminal_frames = int(terminal_mask_frames)
            if fixed_terminal_frames < 0:
                raise ValueError("terminal_mask_frames must be non-negative")
            if fixed_terminal_frames:
                cutoff = int(timestamps[-1]) - fixed_terminal_frames
                valid_count = int(np.searchsorted(timestamps, cutoff, side="right"))
            else:
                valid_count = len(timestamps)
            curve["terminal_mask_frames"] = np.asarray(
                fixed_terminal_frames, dtype=np.int64
            )
            curve["terminal_mask_mode"] = np.asarray("fixed_frames")
            curve["terminal_mask_start_frame"] = np.asarray(-1, dtype=np.int64)
            curve["terminal_mask_status"] = np.asarray("fixed_frames")
        else:
            if built_valid_count < original_length:
                built_terminal_start = int(original_timestamps[built_valid_count])
                valid_count = int(
                    np.searchsorted(timestamps, built_terminal_start, side="left")
                )
            else:
                valid_count = len(timestamps)
        if valid_count < 1:
            raise ValueError(f"Terminal mask removes every retained anchor in {path}")
        home_start = None
        if home_return_starts is not None and episode_id in home_return_starts:
            home_start = int(home_return_starts[episode_id])
            valid_count = min(
                valid_count,
                int(np.searchsorted(timestamps, home_start, side="left")),
            )
            if valid_count < 1:
                raise ValueError(
                    f"Home-return start {home_start} masks every anchor in {path}"
                )

        # Smooth only the eligible prefix so terminal motion cannot leak into
        # the detector.  The masked suffix is held flat for display.
        eligible_base = _savgol(raw[:valid_count], smooth_window, savgol_polyorder)
        eligible_score = np.power(
            np.maximum(eligible_base, 0.0), float(score_power)
        )
        score = np.full(len(raw), float(eligible_score[-1]), dtype=np.float64)
        score[:valid_count] = eligible_score
        finite = score[:valid_count]
        if not np.isfinite(finite).all():
            raise ValueError(f"Non-finite re-smoothed standard score in {path}")
        threshold_total += float(finite.sum())
        threshold_count += int(len(finite))

        curve["sg_vals"] = score.astype(np.float32)
        curve["replan_ts"] = timestamps.astype(np.int64)
        curve["div_cos"] = raw.astype(np.float32)
        curve["threshold_valid_count"] = np.asarray(valid_count, dtype=np.int64)
        curve["score_power"] = np.asarray(score_power, dtype=np.float32)
        curve["smooth_window"] = np.asarray(smooth_window, dtype=np.int64)
        curve["savgol_polyorder"] = np.asarray(savgol_polyorder, dtype=np.int64)
        curve["last_valid_anchor"] = np.asarray(
            complete_future_last_anchor, dtype=np.int64
        )
        if home_start is not None:
            curve["terminal_release_start_frame"] = np.asarray(
                home_start, dtype=np.int64
            )
            curve["terminal_mask_start_frame"] = np.asarray(home_start, dtype=np.int64)
            curve["terminal_release_detected"] = np.asarray(True)
        elif home_return_starts is not None:
            curve["terminal_release_detected"] = np.asarray(False)
        curves[episode_id] = curve

    if not threshold_count:
        raise ValueError(f"No valid saved standard scores found under {curves_dir}")
    threshold = threshold_total / threshold_count
    for curve in curves.values():
        curve["mean_val"] = np.asarray(threshold, dtype=np.float32)
        curve["threshold_val"] = np.asarray(threshold, dtype=np.float32)
        curve["threshold_pool_points"] = np.asarray(
            threshold_count, dtype=np.int64
        )
    return curves, threshold, threshold_count


def pooled_curve_power_mean(
    curves: dict[int, dict], *, source_power: float, target_power: float
) -> tuple[float, int]:
    """Pool an in-memory curve set after mapping between score powers."""
    exponent = float(target_power) / float(source_power)
    total = 0.0
    count = 0
    for curve in curves.values():
        valid_count = int(np.asarray(curve["threshold_valid_count"]).reshape(-1)[0])
        values = np.asarray(curve["sg_vals"], dtype=np.float64)[:valid_count]
        mapped = np.power(np.maximum(values, 0.0), exponent)
        total += float(mapped.sum())
        count += int(len(mapped))
    if not count:
        raise ValueError("No valid in-memory curve points")
    return total / count, count


def extend_reset_jump_masks(
    curves_dir: Path,
    base_mask_ends: dict[int, int],
    reset_jump_speeds: dict[int, float],
    *,
    provisional_threshold: float,
    reset_jump_velocity_threshold: float = 0.5,
    stable_anchors: int = 3,
    max_extra_frames: int = 100,
    smooth_window: int = 7,
    savgol_polyorder: int = 4,
) -> tuple[dict[int, int], dict[int, int]]:
    """Extend only severe reset-jump episodes until their SBD burst settles.

    Stability means consecutive eligible power=1 anchors below the provisional
    pooled mean.  The extension is capped so a genuinely immediate task cannot
    cause the whole demonstration to be classified as initialization.
    """
    if reset_jump_velocity_threshold <= 0.0:
        raise ValueError("reset_jump_velocity_threshold must be positive")
    if int(stable_anchors) < 1:
        raise ValueError("stable_anchors must be positive")
    if int(max_extra_frames) < 0:
        raise ValueError("max_extra_frames must be non-negative")
    extended = dict(base_mask_ends)
    reset_extensions: dict[int, int] = {}
    for episode_id, speed in reset_jump_speeds.items():
        if float(speed) <= float(reset_jump_velocity_threshold):
            continue
        path = Path(curves_dir) / f"ep{int(episode_id):07d}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Reset-jump curve not found: {path}")
        with np.load(path, allow_pickle=False) as loaded:
            curve = {key: loaded[key] for key in loaded.files}
        score, start_index, valid_count = initial_masked_score_curve(
            curve,
            initial_mask_until_frame=base_mask_ends[int(episode_id)],
            score_power=1.0,
            smooth_window=smooth_window,
            savgol_polyorder=savgol_polyorder,
        )
        timestamps = np.asarray(curve["replan_ts"], dtype=np.int64).reshape(-1)
        below = score[start_index:valid_count] < float(provisional_threshold)
        stable = np.convolve(
            below.astype(np.int8),
            np.ones(int(stable_anchors), dtype=np.int8),
            mode="valid",
        )
        matches = np.flatnonzero(stable == int(stable_anchors))
        base_end = int(base_mask_ends[int(episode_id)])
        cap = base_end + int(max_extra_frames)
        if len(matches):
            stable_end_index = start_index + int(matches[0]) + int(stable_anchors) - 1
            candidate = int(timestamps[stable_end_index])
        else:
            candidate = cap
        # Keep at least one terminal-valid anchor available to the detector.
        latest_end = int(timestamps[valid_count - 1]) - 1
        final_end = min(cap, latest_end, max(base_end, candidate))
        extended[int(episode_id)] = final_end
        reset_extensions[int(episode_id)] = final_end
    return extended, reset_extensions


def abc_reset_aware_initial_masks(
    dataset_dir: Path,
    episode_ids,
    curves_dir: Path,
    *,
    excluded_indices=(),
    after_start_frames: int = 100,
    velocity_threshold: float = 0.005,
    sustain_frames: int = 5,
    reset_jump_window_frames: int = 10,
    reset_jump_velocity_threshold: float = 0.5,
    stable_anchors: int = 3,
    max_extra_frames: int = 100,
    smooth_window: int = 7,
    savgol_polyorder: int = 4,
) -> tuple[dict[int, int], dict[int, int], dict[int, int], dict[int, float], dict[int, int]]:
    """Build the same reset-aware initialization masks used by ABC cosine SBD."""
    starts, base_ends, reset_speeds = abc_initial_motion_masks(
        dataset_dir,
        episode_ids,
        excluded_indices=excluded_indices,
        after_start_frames=after_start_frames,
        velocity_threshold=velocity_threshold,
        sustain_frames=sustain_frames,
        reset_jump_window_frames=reset_jump_window_frames,
    )
    provisional_threshold, _ = global_initial_masked_score_mean(
        curves_dir,
        base_ends,
        score_power=1.0,
        smooth_window=smooth_window,
        savgol_polyorder=savgol_polyorder,
    )
    mask_ends, reset_extensions = extend_reset_jump_masks(
        curves_dir,
        base_ends,
        reset_speeds,
        provisional_threshold=provisional_threshold,
        reset_jump_velocity_threshold=reset_jump_velocity_threshold,
        stable_anchors=stable_anchors,
        max_extra_frames=max_extra_frames,
        smooth_window=smooth_window,
        savgol_polyorder=savgol_polyorder,
    )
    return starts, base_ends, mask_ends, reset_speeds, reset_extensions


def global_curve_power_range(
    curves_dir: Path,
    keys: tuple[str, ...],
    *,
    source_power: float,
    target_power: float,
) -> tuple[float, float]:
    """Return one finite display range for saved metrics across all episodes."""
    if source_power <= 0.0 or target_power <= 0.0:
        raise ValueError("source_power and target_power must be positive")
    exponent = float(target_power) / float(source_power)
    low = np.inf
    high = -np.inf
    for path in sorted(Path(curves_dir).glob("ep*.npz")):
        with np.load(path, allow_pickle=False) as curve:
            for key in keys:
                if key not in curve.files:
                    raise KeyError(f"{path} does not contain {key!r}")
                values = np.asarray(curve[key], dtype=np.float64).reshape(-1)
                finite = values[np.isfinite(values)]
                if not len(finite):
                    continue
                mapped = np.power(np.maximum(finite, 0.0), exponent)
                low = min(low, float(mapped.min()))
                high = max(high, float(mapped.max()))
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError(f"No finite values found for {keys!r} under {curves_dir}")
    return low, high


def padded_axis_range(
    low: float,
    high: float,
    *,
    include: tuple[float, ...] = (),
    fraction: float = 0.05,
) -> tuple[float, float]:
    """Pad a global range while including fixed reference lines."""
    if include:
        low = min(low, *include)
        high = max(high, *include)
    span = high - low
    if span <= 0.0:
        span = max(abs(low), 1.0)
    pad = span * fraction
    return max(0.0, low - pad), high + pad


def skills_from_denoising_gain(
    curve: dict,
    *,
    threshold: float,
    nms_dist: int,
    min_skill_len: int,
) -> tuple[list[tuple[int, int, None]], np.ndarray, np.ndarray]:
    """Cut an episode at gain peaks above a global mean using the final SBD NMS rules."""
    from scipy.signal import find_peaks

    timestamps = np.asarray(curve["replan_ts"], dtype=np.int64).reshape(-1)
    values = np.asarray(curve["denoising_gain_sg"], dtype=np.float64).reshape(-1)
    valid_count = int(np.asarray(curve["threshold_valid_count"]).reshape(-1)[0])
    n_frames = int(np.asarray(curve["n_frames"]).reshape(-1)[0])
    if len(timestamps) != len(values) or not 0 < valid_count <= len(values):
        raise ValueError(
            "Invalid denoising-gain curve: "
            f"timestamps={len(timestamps)}, values={len(values)}, valid={valid_count}."
        )
    valid_values = values[:valid_count]
    peak_indices = find_peaks(valid_values)[0].tolist()
    # scipy excludes endpoints. Mirror the detector's terminal-edge check by
    # using the first masked anchor as the right-hand neighbour when available.
    if (
        valid_count >= 2
        and valid_count < len(values)
        and valid_values[-1] > valid_values[-2]
        and valid_values[-1] > values[valid_count]
    ):
        peak_indices.append(valid_count - 1)
    candidates = [
        int(index)
        for index in peak_indices
        if np.isfinite(valid_values[index])
        and valid_values[index] > float(threshold)
        and int(timestamps[index] - timestamps[0]) > int(nms_dist)
    ]
    by_height = sorted(candidates, key=lambda index: valid_values[index], reverse=True)
    kept: list[int] = []
    suppressed: set[int] = set()
    for index in by_height:
        if index in suppressed:
            continue
        kept.append(index)
        for other in by_height:
            if other != index and abs(int(timestamps[other] - timestamps[index])) <= int(nms_dist):
                suppressed.add(other)
    peak_by_frame = {
        int(timestamps[index]): float(valid_values[index])
        for index in kept
    }
    boundaries = sorted({0, *peak_by_frame, n_frames})
    while len(boundaries) > 2 and boundaries[-1] - boundaries[-2] < int(min_skill_len):
        del boundaries[-2]
    accepted_frames = boundaries[1:-1]
    skills = [
        (int(start), int(end), None)
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    return (
        skills,
        np.asarray(accepted_frames, dtype=np.int64),
        np.asarray([peak_by_frame[frame] for frame in accepted_frames], dtype=np.float32),
    )


def skills_from_sbd_score(
    curve: dict,
    *,
    threshold: float,
    nms_dist: int,
    min_skill_len: int,
    smooth_window: int,
    savgol_polyorder: int,
    initial_mask_until_frame: int | None = None,
    forced_terminal_boundary: int | None = None,
) -> tuple[list[tuple[int, int, None]], np.ndarray, np.ndarray]:
    """Recut one saved SBD curve with a different global score threshold."""
    from scipy.signal import find_peaks, savgol_filter

    timestamps = np.asarray(curve["replan_ts"], dtype=np.int64).reshape(-1)
    values = np.asarray(curve["sg_vals"], dtype=np.float64).reshape(-1)
    valid_count = int(np.asarray(curve["threshold_valid_count"]).reshape(-1)[0])
    n_frames = int(np.asarray(curve["n_frames"]).reshape(-1)[0])
    if len(timestamps) != len(values) or not 0 < valid_count <= len(values):
        raise ValueError(
            "Invalid SBD score curve: "
            f"timestamps={len(timestamps)}, values={len(values)}, valid={valid_count}."
        )
    start_index = (
        int(np.searchsorted(timestamps[:valid_count], int(initial_mask_until_frame), side="right"))
        if initial_mask_until_frame is not None
        else 0
    )
    if start_index >= valid_count:
        raise ValueError(
            f"Initial mask through frame {initial_mask_until_frame} leaves no peak candidates"
        )
    valid_values = values[:valid_count]
    peak_indices = (find_peaks(valid_values[start_index:])[0] + start_index).tolist()

    # Match build_data's terminal-edge topology test. Its first masked raw
    # anchor is used only as the right-hand neighbour of the final valid anchor.
    raw = np.asarray(curve.get("div_cos", []), dtype=np.float64).reshape(-1)
    if valid_count >= 2 and valid_count < len(raw):
        terminal_raw = raw[: valid_count + 1]
        window = min(int(smooth_window), len(terminal_raw))
        topology_polyorder = int(savgol_polyorder)
        if window > 1:
            window = min(window, len(terminal_raw))
        if window > topology_polyorder:
            topology = savgol_filter(
                terminal_raw,
                window_length=window,
                polyorder=topology_polyorder,
            )
        else:
            topology = terminal_raw
        score_power = float(np.asarray(curve.get("score_power", 1.0)).reshape(-1)[0])
        if not np.isclose(score_power, 1.0):
            topology = np.power(np.maximum(topology, 0.0), score_power)
        if (valid_count - 1) in set(find_peaks(topology)[0].tolist()):
            peak_indices.append(valid_count - 1)

    candidates = [
        int(index)
        for index in sorted(set(peak_indices))
        if np.isfinite(valid_values[index])
        and valid_values[index] > float(threshold)
        and int(timestamps[index] - timestamps[0]) > int(nms_dist)
    ]
    by_height = sorted(candidates, key=lambda index: valid_values[index], reverse=True)
    kept: list[int] = []
    suppressed: set[int] = set()
    for index in by_height:
        if index in suppressed:
            continue
        kept.append(index)
        for other in by_height:
            if other != index and abs(int(timestamps[other] - timestamps[index])) <= int(nms_dist):
                suppressed.add(other)
    peak_by_frame = {
        int(timestamps[index]): float(valid_values[index])
        for index in kept
    }
    forced = None
    if forced_terminal_boundary is not None:
        forced = int(forced_terminal_boundary)
        if not 0 < forced < n_frames:
            raise ValueError(
                f"Forced terminal boundary {forced} is outside episode length {n_frames}"
            )
        nearest = int(np.argmin(np.abs(timestamps - forced)))
        peak_by_frame[forced] = float(values[nearest])
        # Treat the semantic release cut exactly like the ordinary SBD peaks:
        # if it is too close to another candidate, retain the higher score.
        by_height = sorted(peak_by_frame.items(), key=lambda item: item[1], reverse=True)
        filtered: dict[int, float] = {}
        for frame, value in by_height:
            if any(abs(frame - kept_frame) <= int(nms_dist) for kept_frame in filtered):
                continue
            filtered[frame] = value
        peak_by_frame = filtered
        if forced not in peak_by_frame:
            forced = None
    boundaries = sorted({0, *peak_by_frame, n_frames})
    while len(boundaries) > 2 and boundaries[-1] - boundaries[-2] < int(min_skill_len):
        if forced is not None and boundaries[-2] == forced:
            # A detected return shorter than min_skill_len is a failed semantic
            # split; retain the full episode instead of silently moving the cut.
            del peak_by_frame[forced]
            forced = None
        del boundaries[-2]
    accepted_frames = boundaries[1:-1]
    skills = [
        (int(start), int(end), None)
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    return (
        skills,
        np.asarray(accepted_frames, dtype=np.int64),
        np.asarray([peak_by_frame[frame] for frame in accepted_frames], dtype=np.float32),
    )


def saved_abc_terminal_boundary(curve: dict) -> int | None:
    """Recover the semantic ABC release/home-return cut from a saved curve."""
    raw_mode = np.asarray(curve.get("terminal_mask_mode", "")).reshape(-1)[0]
    if isinstance(raw_mode, bytes):
        raw_mode = raw_mode.decode("utf-8")
    if str(raw_mode) != "abc_terminal_release":
        return None
    for key in ("terminal_release_start_frame", "terminal_mask_start_frame"):
        if key not in curve:
            continue
        candidate = int(np.asarray(curve[key]).reshape(-1)[0])
        if candidate >= 0:
            return candidate
    return None


def select_episodes(ep_task: dict[int, int], task_ids, n_episodes: int) -> list[tuple]:
    """Ordered [(task_label, [ep, ...]), ...] episode groups for the gallery.
    Empty task_ids → first n_episodes episodes overall (task_label=None); else
    n_episodes episodes per listed task."""
    if not task_ids:
        return [(None, sorted(ep_task)[:n_episodes])]
    groups = []
    for t in task_ids:
        eps = sorted(ep for ep, ti in ep_task.items() if ti == int(t))
        groups.append((int(t), eps[:n_episodes]))
    return groups


def _cap(task_label, ep) -> str:
    return f"episode {ep}" if task_label is None else f"task{int(task_label):02d} · episode {ep}"


def _skill_stats(lengths: list[int]) -> str:
    """'N skills · len min/max/mean' summary for a set of skill lengths (frames)."""
    if not lengths:
        return "0 skills"
    arr = np.asarray(lengths)
    return (
        f"{len(arr)} skills · len min {int(arr.min())} / max {int(arr.max())}"
        f" / mean {float(arr.mean()):.1f}"
    )


def make_frames_loader(dataset_dir: Path, image_key: str, episodes_meta=None):
    episodes_meta = _load_episodes_meta(dataset_dir) if episodes_meta is None else episodes_meta
    key = _resolve_image_key(episodes_meta, image_key)

    def load(ep_id: int):
        try:
            return _read_episode_camera_clip(dataset_dir, episodes_meta, ep_id, key)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] frames ep{ep_id}: {exc}")
            return None
    return load


def _task_instruction_map(episodes_meta, ep_task: dict[int, int]) -> dict[int, str]:
    instructions = {}
    if "tasks" not in episodes_meta.columns:
        return instructions
    for _, row in episodes_meta.iterrows():
        ep = int(row["episode_index"])
        task_id = ep_task.get(ep)
        if task_id is None or task_id in instructions:
            continue
        value = row["tasks"]
        if isinstance(value, str):
            text = value
        elif value is None:
            text = ""
        else:
            values = np.asarray(value, dtype=object).reshape(-1).tolist()
            text = str(values[0]) if values else ""
        if text:
            instructions[task_id] = text
    return instructions


def suite_to_dataset_task_ids(
    task_ids: list[int],
    *,
    suite_name: str,
    instructions: dict[int, str],
) -> list[int]:
    """Translate LIBERO suite IDs to the skillset's dataset task indices.

    LIBERO-90's filtered LeRobot dataset does not preserve the benchmark's
    numeric task order (for example suite task 1 is dataset task 26).  Skill
    artifacts store the latter, so bridge the two spaces through their common
    language instruction instead of silently selecting the wrong task.
    """
    from libero.libero import benchmark  # noqa: PLC0415

    suite_key = str(suite_name).strip()
    suites = benchmark.get_benchmark_dict()
    if suite_key not in suites:
        raise ValueError(
            f"Unknown LIBERO suite {suite_key!r}; available suites: {sorted(suites)}"
        )
    suite = suites[suite_key]()
    by_instruction = {
        str(instruction).strip(): int(dataset_id)
        for dataset_id, instruction in instructions.items()
        if str(instruction).strip()
    }
    translated = []
    mappings = []
    missing = []
    for suite_id in task_ids:
        if not 0 <= int(suite_id) < len(suite.tasks):
            missing.append(int(suite_id))
            continue
        instruction = str(suite.tasks[int(suite_id)].language).strip()
        dataset_id = by_instruction.get(instruction)
        if dataset_id is None:
            missing.append(int(suite_id))
            continue
        mappings.append((int(suite_id), dataset_id, instruction))
        if dataset_id not in translated:
            translated.append(dataset_id)
    if missing:
        raise ValueError(
            f"Suite task ids {missing} are absent from this skillset/dataset."
        )
    print(
        "Task ID mapping (suite -> dataset): "
        + ", ".join(f"{suite_id}->{dataset_id} ({text})" for suite_id, dataset_id, text in mappings)
    )
    return translated


def select_stage1_suite_episodes(
    ep_task: dict[int, int],
    task_ids: list[int],
    *,
    suite_name: str,
    dataset_dir: Path,
    n_episodes: int,
) -> list[tuple[int, list[int]]]:
    """Select the same scene-exact LIBERO episodes used by Stage-1 eval.

    A LIBERO-90 language instruction is not a unique task identifier: the same
    instruction can occur in multiple scenes. Stage-1 resolves this ambiguity
    with ``eval_init_states.npz``, whose ``scene_file`` records the exact source
    BDDL task for every filtered LeRobot episode. Use that same contract here
    instead of translating suite IDs through language alone.
    """
    from libero.libero import benchmark  # noqa: PLC0415

    dataset_dir = Path(dataset_dir)
    init_states_path = (
        dataset_dir.parent
        / "skillvla_dataset"
        / dataset_dir.name
        / "eval_init_states.npz"
    )
    if not init_states_path.is_file():
        raise FileNotFoundError(
            "Suite task selection requires the Stage-1 episode-exact map: "
            f"{init_states_path}"
        )

    suites = benchmark.get_benchmark_dict()
    suite_key = str(suite_name).strip()
    if suite_key not in suites:
        raise ValueError(
            f"Unknown LIBERO suite {suite_key!r}; available suites: {sorted(suites)}"
        )
    suite = suites[suite_key]()
    with np.load(init_states_path, allow_pickle=True) as init_data:
        if not {"episode_index", "scene_file"}.issubset(init_data.files):
            raise KeyError(
                f"{init_states_path} must contain episode_index and scene_file."
            )
        scene_by_episode = {
            int(episode): str(scene_file).removesuffix("_demo.hdf5")
            for episode, scene_file in zip(
                init_data["episode_index"], init_data["scene_file"], strict=True
            )
        }

    groups = []
    summaries = []
    available_episodes = sorted(int(episode) for episode in ep_task)
    for raw_task_id in task_ids:
        task_id = int(raw_task_id)
        if not 0 <= task_id < len(suite.tasks):
            raise ValueError(
                f"Suite task id {task_id} is outside {suite_key}'s "
                f"0..{len(suite.tasks) - 1} range."
            )
        task = suite.tasks[task_id]
        task_name = str(task.name)
        episodes = [
            episode
            for episode in available_episodes
            if scene_by_episode.get(episode) == task_name
        ]
        if not episodes:
            raise ValueError(
                f"Suite task {task_id} ({task_name}) has no scene-exact episode "
                f"in {dataset_dir}."
            )
        chosen = episodes[: int(n_episodes)]
        groups.append((task_id, chosen))
        summaries.append(
            f"{task_id}:{task_name} -> dataset task(s) "
            f"{sorted({int(ep_task[episode]) for episode in chosen})}, "
            f"episodes {chosen}"
        )
    print("Stage-1 exact task selection: " + "; ".join(summaries))
    return groups


def _gripper_labels(dataset_dir: Path, indices: tuple[int, ...]) -> list[str]:
    try:
        info = json.loads((dataset_dir / "meta" / "info.json").read_text())
        names = info["features"]["action"].get("names")
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
        names = None
    labels = []
    for index in indices:
        if names and index < len(names):
            labels.append(str(names[index]))
        elif len(indices) == 1:
            labels.append("gripper")
        else:
            labels.append(f"gripper action[{index}]")
    return labels


def _dataset_fps(dataset_dir: Path) -> float:
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    if fps <= 0:
        raise ValueError(f"Invalid dataset fps: {fps}")
    return fps


def _skill_video_metadata(
    dataset_dir: Path,
    episodes_meta,
    image_key: str,
    episode_id: int,
    skills,
    out_dir: Path,
    fps: float,
) -> list[dict]:
    key = _resolve_image_key(episodes_meta, image_key)
    row = _episode_row(episodes_meta, episode_id)
    video_path = _video_path(dataset_dir, episodes_meta, episode_id, key).resolve()
    relative_path = os.path.relpath(video_path, out_dir.expanduser().resolve())
    video_src = quote(Path(relative_path).as_posix(), safe="/.:@-_")
    if "video_path" in row.index:
        episode_start = 0.0
        episode_end = int(row["length"]) / fps
    else:
        episode_start = float(row[f"videos/{key}/from_timestamp"])
        episode_end = float(row[f"videos/{key}/to_timestamp"])
    result = []
    for frame_start, frame_end, _label in skills:
        start_sec = min(episode_end, episode_start + float(frame_start) / fps)
        end_sec = min(episode_end, episode_start + float(frame_end) / fps)
        result.append({
            "video_src": video_src,
            "start_sec": start_sec,
            "end_sec": max(start_sec, end_sec),
        })
    return result


def _cached_episode_video_clip(
    source_path: Path,
    *,
    episode_id: int,
    image_key: str,
    episode_start: float,
    episode_end: float,
    cache_dir: Path,
) -> Path:
    """Materialize one short browser-friendly H.264 episode clip.

    LeRobot stores many episodes in long AV1 chunk files. Browsers serving the
    report over a lightweight preview server can otherwise fetch and seek into
    a 50--200 MB source for every play request. The cache signature makes clips
    reusable across report variants while invalidating them if the source or
    episode interval changes.
    """
    source_path = Path(source_path).resolve()
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(image_key))
    output_path = cache_dir / f"ep{int(episode_id):07d}_{safe_key}.mp4"
    metadata_path = output_path.with_suffix(".json")
    source_stat = source_path.stat()
    duration = max(0.0, float(episode_end) - float(episode_start))
    if duration <= 0.0:
        raise ValueError(
            f"Episode {episode_id} has invalid video interval "
            f"[{episode_start}, {episode_end}]"
        )
    signature = {
        "schema_version": 1,
        "source_path": str(source_path),
        "source_size": int(source_stat.st_size),
        "source_mtime_ns": int(source_stat.st_mtime_ns),
        "episode_start_sec": round(float(episode_start), 9),
        "episode_end_sec": round(float(episode_end), 9),
        "video_codec": "libx264",
        "crf": 23,
        "pixel_format": "yuv420p",
    }
    if output_path.is_file() and output_path.stat().st_size > 0 and metadata_path.is_file():
        try:
            cached_signature = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            cached_signature = None
        if cached_signature == signature:
            print(f"  [episode clip cached] ep{episode_id}: {output_path}")
            return output_path

    temporary_path = output_path.with_name(output_path.stem + ".tmp.mp4")
    temporary_path.unlink(missing_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        f"{float(episode_start):.9f}",
        "-i",
        str(source_path),
        "-t",
        f"{duration:.9f}",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary_path),
    ]
    print(
        f"  [episode clip encode] ep{episode_id}: "
        f"{episode_start:.3f}..{episode_end:.3f}s from {source_path.name}",
        flush=True,
    )
    try:
        subprocess.run(command, check=True)
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg produced an empty clip: {temporary_path}")
        temporary_path.replace(output_path)
        metadata_path.write_text(json.dumps(signature, indent=2) + "\n")
    except FileNotFoundError as exc:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "ffmpeg is required by --episode_video_clips but was not found"
        ) from exc
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def _episode_video_metadata(
    dataset_dir: Path,
    episodes_meta,
    image_key: str,
    episode_id: int,
    out_dir: Path,
    fps: float,
    clip_cache_dir: Path | None = None,
) -> dict:
    """Describe one source episode for the browser-synchronised skill player."""
    key = _resolve_image_key(episodes_meta, image_key)
    row = _episode_row(episodes_meta, episode_id)
    video_path = _video_path(dataset_dir, episodes_meta, episode_id, key).resolve()
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())

    feature = info.get("features", {}).get(
        key, info.get("features", {}).get(f"observation.images.{key}", {})
    )
    shape = [int(value) for value in feature.get("shape", [])]
    if len(shape) >= 2:
        if len(shape) >= 3 and shape[0] in (1, 3, 4):
            height, width = shape[-2], shape[-1]
        else:
            height, width = shape[0], shape[1]
    else:
        height, width = 168, 224
    aspect_ratio = float(width) / max(1.0, float(height))

    object_position_y = 50.0
    camera_views = None
    if "video_path" in row.index:
        episode_start = 0.0
        episode_end = int(row["length"]) / fps
        layout = json.loads(str(row["camera_layout_json"]))
        camera = str(key).removeprefix("observation.images.")
        match = next(
            (
                item
                for item in layout
                if camera in {str(item.get("camera", "")), str(item.get("source", ""))}
            ),
            None,
        )
        if match is None:
            raise ValueError(
                f"ABC camera {camera!r} is unavailable; choices="
                f"{[item.get('camera') for item in layout]}"
            )
        source_height = max(int(item["row_end"]) for item in layout)
        crop_height = int(match["row_end"]) - int(match["row_start"])
        overflow = max(0, source_height - crop_height)
        object_position_y = (
            0.0 if overflow == 0 else 100.0 * int(match["row_start"]) / overflow
        )
        view_labels = {
            "left_wrist": "Left wrist",
            "top": "External",
            "right_wrist": "Right wrist",
        }
        camera_views = []
        for view_camera in ("left_wrist", "top", "right_wrist"):
            view = next(
                (
                    item
                    for item in layout
                    if view_camera
                    in {str(item.get("camera", "")), str(item.get("source", ""))}
                ),
                None,
            )
            if view is None:
                continue
            camera_views.append(
                {
                    "camera": view_camera,
                    "label": view_labels[view_camera],
                    "row_start": int(view["row_start"]),
                    "row_end": int(view["row_end"]),
                    "source_height": source_height,
                    "is_primary": view_camera == "top",
                }
            )
    else:
        episode_start = float(row[f"videos/{key}/from_timestamp"])
        episode_end = float(row[f"videos/{key}/to_timestamp"])

    if clip_cache_dir is not None and "video_path" not in row.index:
        video_path = _cached_episode_video_clip(
            video_path,
            episode_id=episode_id,
            image_key=key,
            episode_start=episode_start,
            episode_end=episode_end,
            cache_dir=clip_cache_dir,
        )
        episode_end = max(0.0, episode_end - episode_start)
        episode_start = 0.0
    relative_path = os.path.relpath(video_path, out_dir.expanduser().resolve())

    return {
        "video_src": quote(Path(relative_path).as_posix(), safe="/.:@-_"),
        "episode_start_sec": episode_start,
        "episode_end_sec": episode_end,
        "fps": fps,
        "aspect_ratio": aspect_ratio,
        "object_position_y": object_position_y,
        "camera_views": camera_views,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skillset_dir", default=None,
                   help="skillset root; reads skillset_manifest.json and infers skills/curves/dataset")
    p.add_argument("--skills_dir", default=None, help="{SKILLSET_DIR}/skills")
    p.add_argument("--curves_dir", default=None,
                   help="{SKILLSET_DIR}/curves (per-episode multimodality curves; "
                        "absent → frames only, no graph)")
    p.add_argument("--dataset_dir", default=None, help="raw LeRobot dataset (videos + meta)")
    p.add_argument("--image_key", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--out_html", default=None,
                   help="output HTML filename within out_dir; encode the DP (e.g. state_obs10_ck100000.html) "
                        "so different DPs don't overwrite each other in a shallow folder")
    p.add_argument("--title", default="DP skill boundary split")
    p.add_argument("--n_episodes", type=int, default=12,
                   help="episodes shown; per task when --task_ids is given")
    p.add_argument("--task_ids", type=int, nargs="*", default=None,
                   help="restrict to these tasks (n_episodes each); empty = first n_episodes overall")
    p.add_argument(
        "--task_id_space",
        choices=("dataset", "suite"),
        default="dataset",
        help="interpret --task_ids as skillset dataset indices or LIBERO suite indices",
    )
    p.add_argument(
        "--target_task",
        default="",
        help="LIBERO suite name used when --task_id_space=suite (for example libero_90)",
    )
    p.add_argument("--thumb_size", type=int, default=160)
    p.add_argument(
        "--skill_video",
        action="store_true",
        help="show a lazy-loaded looping source-video slot between each skill's start/end frames",
    )
    p.add_argument(
        "--episode_video",
        action="store_true",
        help="show one full-episode video with a live skill badge and clickable skill timeline",
    )
    p.add_argument(
        "--episode_video_clips",
        action="store_true",
        help=(
            "for chunked LeRobot video, cache each selected episode as a short "
            "H.264 clip next to sibling report directories"
        ),
    )
    p.add_argument("--hide_start_end_frames", action="store_true", help="omit start/end still images")
    p.add_argument("--hide_cos_graph", action="store_true", help="omit the VF cosine-divergence graph")
    p.add_argument("--hide_gain_graph", action="store_true", help="omit the denoising-gain graph")
    p.add_argument("--hide_bic_graph", action="store_true", help="omit the delta-BIC graph")
    p.add_argument("--hide_gripper_graph", action="store_true", help="omit all gripper signal graphs")
    p.add_argument(
        "--boundary_source",
        choices=(
            "skillset",
            "denoising_gain",
            "all_peaks",
            "power_half_mean",
            "power_one_initial_motion_mask",
            "standard_recut",
            "within_normalized_l2",
            "covariance_gated_cosine",
            "covariance_gated_cosine_initial_motion_mask",
            "covariance_gated_angular_chord",
            "covariance_gated_angular_chord_initial_motion_mask",
            "gmm_overlap_entropy",
            "strong_covariance_gated_cosine",
            "strong_covariance_gated_cosine_initial_motion_mask",
        ),
        default="skillset",
        help=(
            "use built skill cuts, recut saved denoising-gain peaks above their global mean, "
            "recut every SBD local maximum without a height threshold, or recut in the "
            "score-power=0.5 space above its pooled global mean, or recut cached GMM "
            "clouds by center-distance / within-mode-radius or covariance-gated "
            "angular chord or strongly reliability-gated cosine, optionally with "
            "ABC's reset-aware initial-motion mask"
        ),
    )
    p.add_argument("--initial_motion_after_start_frames", type=int, default=100)
    p.add_argument("--initial_motion_velocity_threshold", type=float, default=0.005)
    p.add_argument("--initial_motion_sustain_frames", type=int, default=5)
    p.add_argument("--initial_reset_jump_window_frames", type=int, default=10)
    p.add_argument("--initial_reset_jump_velocity_threshold", type=float, default=0.5)
    p.add_argument("--initial_reset_stable_anchors", type=int, default=3)
    p.add_argument("--initial_reset_max_extra_frames", type=int, default=100)
    p.add_argument("--boundary_nms_dist", type=int, default=None)
    p.add_argument("--boundary_min_skill_len", type=int, default=None)
    p.add_argument(
        "--boundary_smooth_window",
        type=int,
        default=None,
        help="override the standard score Savitzky-Golay window for report-only recutting",
    )
    p.add_argument(
        "--boundary_replan_interval",
        type=int,
        default=None,
        help=(
            "report-only cadence override; must be an integer multiple of the "
            "saved replan interval so existing DP/GMM anchors can be reused"
        ),
    )
    p.add_argument(
        "--boundary_savgol_polyorder",
        type=int,
        default=None,
        help="override the standard score Savitzky-Golay polynomial order",
    )
    p.add_argument(
        "--boundary_threshold_value",
        type=float,
        default=None,
        help=(
            "override the pooled global-mean threshold for standard recutting; "
            "use 0 to retain every non-negative local peak before NMS/min-length filtering"
        ),
    )
    p.add_argument(
        "--home_return_mask",
        action="store_true",
        help=(
            "with --boundary_source=standard_recut, exclude ABC terminal motion "
            "from immediately before the final gripper release; fall back to the "
            "home-return onset only when no release is detected"
        ),
    )
    p.add_argument(
        "--home_return_cache",
        default="",
        help="optional JSON cache for per-episode ABC terminal-release detections",
    )
    p.add_argument(
        "--terminal_release_pre_offset_frames",
        type=int,
        default=8,
        help="start the terminal mask this many frames before detected gripper release",
    )
    p.add_argument(
        "--boundary_terminal_mask_frames",
        type=int,
        default=None,
        help=(
            "override the manifest terminal mask for standard recutting or cached "
            "alternate GMM metrics; this report-only recut does not modify the built skillset"
        ),
    )
    p.add_argument(
        "--boundary_terminal_mask_from_episode_end",
        action="store_true",
        help=(
            "interpret --boundary_terminal_mask_frames from the true episode end "
            "instead of from the last complete-future detector anchor"
        ),
    )
    p.add_argument(
        "--boundary_score_power",
        type=float,
        default=None,
        help=(
            "override the manifest score-power transform for report-only recutting; "
            "primarily used with cached alternate GMM metrics"
        ),
    )
    p.add_argument(
        "--action_error_dir",
        default="",
        help="optional per-model directory containing epXXXXXXX.npz fresh-inference curves",
    )
    p.add_argument(
        "--gmm_raw_dir",
        action="append",
        default=[],
        help=(
            "optional raw/iid96 directory containing epXXXXXXX.npz descriptor clouds; "
            "adds a video-synchronised 5-GMM 3-D panel"
        ),
    )
    p.add_argument("--gmm_components", type=int, default=5)
    p.add_argument(
        "--gmm_min_effective_samples",
        type=float,
        default=0.0,
        help=(
            "discard fitted GMM components whose mixture weight corresponds to "
            "fewer than this many of the cached descriptor samples"
        ),
    )
    p.add_argument(
        "--hide_gmm_visualization",
        action="store_true",
        help="use --gmm_raw_dir for a derived metric without rendering the 3-D GMM panel",
    )
    return p.parse_args()


def _resolve_eval_inputs(args):
    if args.skillset_dir is None:
        missing = [name for name, value in (
            ("--skills_dir", args.skills_dir),
            ("--dataset_dir", args.dataset_dir),
            ("--out_dir", args.out_dir),
        ) if value is None]
        if missing:
            raise ValueError(f"Without --skillset_dir, these arguments are required: {', '.join(missing)}")
        skills_dir = Path(args.skills_dir)
        manifest_path = skills_dir.parent / "skillset_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        return (
            skills_dir,
            Path(args.curves_dir) if args.curves_dir else None,
            Path(args.dataset_dir),
            args.image_key or "observation.images.image",
            Path(args.out_dir),
            args.out_html or "index.html",
            manifest,
        )

    skillset_dir = Path(args.skillset_dir).expanduser().resolve()
    manifest_path = skillset_dir / "skillset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Skillset manifest not found: {manifest_path}. "
            "Rebuild the skillset with the current builder or provide explicit paths."
        )
    manifest = json.loads(manifest_path.read_text())
    skills_dir = Path(args.skills_dir) if args.skills_dir else skillset_dir / "skills"
    curves_dir = Path(args.curves_dir) if args.curves_dir else skillset_dir / "curves"
    if not curves_dir.is_dir():
        curves_dir = None
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path(manifest["dataset_dir"])
    image_key = args.image_key or manifest.get("image_key", "observation.images.image")
    out_dir = Path(args.out_dir) if args.out_dir else skillset_dir / "eval"
    out_html = args.out_html or "index.html"
    return skills_dir, curves_dir, dataset_dir, image_key, out_dir, out_html, manifest


def main():
    args = parse_args()
    if args.skill_video and args.episode_video:
        raise ValueError("--skill_video and --episode_video are mutually exclusive")
    if args.episode_video_clips and not args.episode_video:
        raise ValueError("--episode_video_clips requires --episode_video")
    skills_dir, curves_dir, dataset_dir, image_key, out_dir, out_html, manifest = _resolve_eval_inputs(args)
    if not skills_dir.is_dir():
        raise FileNotFoundError(f"skills_dir not found: {skills_dir}")

    ep_task, ep_files = index_skillset(skills_dir)
    if not ep_task:
        raise FileNotFoundError(f"No skill npz under {skills_dir}")
    episodes_meta = _load_episodes_meta(dataset_dir)
    frames_src = (
        make_frames_loader(dataset_dir, image_key, episodes_meta=episodes_meta)
        if not args.hide_start_end_frames
        else None
    )
    instructions = _task_instruction_map(episodes_meta, ep_task)
    selected_task_ids = args.task_ids
    if selected_task_ids and args.task_id_space == "suite":
        if not args.target_task.strip():
            raise ValueError("--target_task is required when --task_id_space=suite")
        selected_groups = select_stage1_suite_episodes(
            ep_task,
            selected_task_ids,
            suite_name=args.target_task,
            dataset_dir=dataset_dir,
            n_episodes=args.n_episodes,
        )
        from libero.libero import benchmark  # noqa: PLC0415

        suite = benchmark.get_benchmark_dict()[args.target_task]()
        display_instructions = {
            int(task_id): str(suite.tasks[int(task_id)].language)
            for task_id in selected_task_ids
        }
    else:
        selected_groups = select_episodes(ep_task, selected_task_ids, args.n_episodes)
        display_instructions = instructions
    selected_episode_ids = [
        int(episode_id)
        for _task_label, episode_ids in selected_groups
        for episode_id in episode_ids
    ]
    gmm_visualizations = (
        build_gmm_visualizations(
            args.gmm_raw_dir,
            selected_episode_ids,
            components=int(args.gmm_components),
        )
        if args.gmm_raw_dir and not args.hide_gmm_visualization
        else {}
    )
    if args.gmm_raw_dir and not args.hide_gmm_visualization:
        print(
            f"GMM 3-D visualization: {len(gmm_visualizations)}/"
            f"{len(selected_episode_ids)} selected episodes"
        )
    configured_gripper_indices = (
        [] if args.hide_gripper_graph else manifest.get("action", {}).get("gripper_indices", [])
    )
    fps = _dataset_fps(dataset_dir)
    action_error_dir = Path(args.action_error_dir) if args.action_error_dir else None
    episode_clip_cache_dir = (
        out_dir.parent / "_media" / "episode_clips"
        if args.episode_video_clips
        else None
    )
    if episode_clip_cache_dir is not None:
        print(f"Episode-video H.264 cache: {episode_clip_cache_dir}")
    gain_threshold = None
    gain_threshold_count = 0
    power_half_threshold = None
    power_half_threshold_count = 0
    initial_motion_starts: dict[int, int] = {}
    initial_motion_base_mask_ends: dict[int, int] = {}
    initial_motion_mask_ends: dict[int, int] = {}
    initial_reset_jump_speeds: dict[int, float] = {}
    initial_reset_extensions: dict[int, int] = {}
    power_one_initial_threshold = None
    power_one_initial_threshold_count = 0
    cached_metric_curves: dict[int, dict] = {}
    cached_metric_threshold = None
    cached_metric_threshold_count = 0
    detector = manifest.get("detector", {})
    score_power = float(
        args.boundary_score_power
        if args.boundary_score_power is not None
        else detector.get("score_power", 1.0)
    )
    if not np.isfinite(score_power) or score_power <= 0.0:
        raise ValueError("--boundary_score_power must be finite and positive")
    power1_mean = None
    power1_mean_count = 0
    if (
        curves_dir is not None
        and not args.hide_cos_graph
        and args.boundary_source not in {
            "standard_recut",
            "within_normalized_l2",
            "covariance_gated_cosine",
            "covariance_gated_cosine_initial_motion_mask",
            "covariance_gated_angular_chord",
            "covariance_gated_angular_chord_initial_motion_mask",
            "gmm_overlap_entropy",
            "strong_covariance_gated_cosine",
            "strong_covariance_gated_cosine_initial_motion_mask",
        }
        and not np.isclose(score_power, 1.0)
    ):
        power1_mean, power1_mean_count = global_curve_power_mean(
            curves_dir,
            "sg_vals",
            source_power=score_power,
            target_power=1.0,
        )
        print(
            f"SBD power=1 global mean={power1_mean:.8f} "
            f"from {power1_mean_count:,} valid anchors"
        )
    gain_power2_mean = None
    gain_power2_mean_count = 0
    gain_ylim = None
    gain_power2_ylim = None
    if curves_dir is not None and not args.hide_gain_graph:
        gain_power2_mean, gain_power2_mean_count = global_curve_power_mean(
            curves_dir,
            "denoising_gain_sg",
            source_power=1.0,
            target_power=2.0,
        )
        print(
            f"Denoising-gain power=2 global mean={gain_power2_mean:.8f} "
            f"from {gain_power2_mean_count:,} valid anchors"
        )
        gain_low, gain_high = global_curve_power_range(
            curves_dir,
            ("denoising_gain", "denoising_gain_sg"),
            source_power=1.0,
            target_power=1.0,
        )
        gain_power2_low, gain_power2_high = global_curve_power_range(
            curves_dir,
            ("denoising_gain_sg",),
            source_power=1.0,
            target_power=2.0,
        )
        gain_ylim = padded_axis_range(gain_low, gain_high, include=(1.0,))
        gain_power2_ylim = padded_axis_range(
            gain_power2_low,
            gain_power2_high,
            include=(gain_power2_mean,),
        )
        print(
            "Denoising-gain fixed axes: "
            f"power=1 [{gain_ylim[0]:.6f}, {gain_ylim[1]:.6f}], "
            f"power=2 [{gain_power2_ylim[0]:.6f}, {gain_power2_ylim[1]:.6f}]"
        )
    boundary_nms_dist = (
        int(args.boundary_nms_dist)
        if args.boundary_nms_dist is not None
        else int(detector.get("nms_dist", 0))
    )
    boundary_min_skill_len = (
        int(args.boundary_min_skill_len)
        if args.boundary_min_skill_len is not None
        else int(detector.get("min_skill_len", 1))
    )
    boundary_smooth_window = int(
        args.boundary_smooth_window
        if args.boundary_smooth_window is not None
        else detector.get("smooth_window", 7)
    )
    boundary_savgol_polyorder = int(
        args.boundary_savgol_polyorder
        if args.boundary_savgol_polyorder is not None
        else detector.get("savgol_polyorder", 4)
    )
    if boundary_smooth_window < 1 or boundary_savgol_polyorder < 0:
        raise ValueError("Boundary smoothing parameters must be non-negative")
    cached_gmm_sources = {
        "within_normalized_l2",
        "covariance_gated_cosine",
        "covariance_gated_cosine_initial_motion_mask",
        "covariance_gated_angular_chord",
        "covariance_gated_angular_chord_initial_motion_mask",
        "gmm_overlap_entropy",
        "strong_covariance_gated_cosine",
        "strong_covariance_gated_cosine_initial_motion_mask",
    }
    if args.home_return_mask and args.boundary_source not in {
        "standard_recut",
        *cached_gmm_sources,
    }:
        raise ValueError(
            "--home_return_mask requires --boundary_source=standard_recut or a "
            "cached GMM metric"
        )

    standard_recut_curves: dict[int, dict] = {}
    standard_recut_threshold = None
    standard_recut_threshold_count = 0
    home_return_starts: dict[int, int] = {}
    home_return_failures: dict[int, str] = {}
    if args.home_return_mask:
        gripper_indices = manifest.get("action", {}).get("gripper_indices", [])
        home_return_starts, home_return_failures = abc_terminal_release_starts(
            dataset_dir,
            ep_task,
            gripper_indices=gripper_indices,
            excluded_indices=gripper_indices,
            pre_release_offset_frames=args.terminal_release_pre_offset_frames,
            cache_path=(
                Path(args.home_return_cache).expanduser().resolve()
                if args.home_return_cache
                else None
            ),
        )
    if args.boundary_source == "standard_recut":
        if curves_dir is None:
            raise FileNotFoundError("standard recutting requires saved SBD curves")
        (
            standard_recut_curves,
            standard_recut_threshold,
            standard_recut_threshold_count,
        ) = recut_saved_standard_curves(
            curves_dir,
            home_return_starts=(home_return_starts if args.home_return_mask else None),
            smooth_window=boundary_smooth_window,
            savgol_polyorder=boundary_savgol_polyorder,
            score_power=score_power,
            replan_interval=args.boundary_replan_interval,
            terminal_mask_frames=args.boundary_terminal_mask_frames,
        )
        pooled_standard_recut_mean = standard_recut_threshold
        if args.boundary_threshold_value is not None:
            if not np.isfinite(args.boundary_threshold_value):
                raise ValueError("boundary_threshold_value must be finite")
            standard_recut_threshold = float(args.boundary_threshold_value)
        if not np.isclose(score_power, 1.0):
            power1_mean, power1_mean_count = pooled_curve_power_mean(
                standard_recut_curves,
                source_power=score_power,
                target_power=1.0,
            )
        if args.boundary_terminal_mask_frames is not None:
            terminal_summary = (
                f"fixed {int(args.boundary_terminal_mask_frames)} frames"
            )
        elif args.home_return_mask:
            terminal_summary = (
                f"detected {len(home_return_starts)}, "
                f"{len(home_return_failures)} unmasked"
            )
        else:
            terminal_summary = "saved"
        threshold_summary = (
            f"override {standard_recut_threshold:.8f} "
            f"(pooled mean {pooled_standard_recut_mean:.8f})"
            if args.boundary_threshold_value is not None
            else f"global mean {standard_recut_threshold:.8f}"
        )
        print(
            f"standard_recut threshold={threshold_summary} from "
            f"{standard_recut_threshold_count:,} anchors; "
            f"replan={args.boundary_replan_interval or detector.get('replan_interval', 'saved')}; "
            f"smooth={boundary_smooth_window}/{boundary_savgol_polyorder}; "
            f"terminal-mask={terminal_summary}",
            flush=True,
        )

    if args.boundary_source == "denoising_gain":
        if curves_dir is None:
            raise FileNotFoundError("denoising_gain boundaries require the saved curves directory")
        gain_threshold, gain_threshold_count = global_curve_mean(
            curves_dir, "denoising_gain_sg"
        )
        print(
            f"Denoising-gain global mean={gain_threshold:.8f} "
            f"from {gain_threshold_count:,} valid anchors"
        )
    elif args.boundary_source == "all_peaks" and curves_dir is None:
        raise FileNotFoundError("all-peaks SBD boundaries require the saved curves directory")
    elif args.boundary_source == "power_half_mean":
        if curves_dir is None:
            raise FileNotFoundError("power=0.5 SBD boundaries require the saved curves directory")
        power_half_threshold, power_half_threshold_count = global_curve_power_mean(
            curves_dir,
            "sg_vals",
            source_power=score_power,
            target_power=0.5,
        )
        print(
            f"SBD power=0.5 global mean={power_half_threshold:.8f} "
            f"from {power_half_threshold_count:,} valid anchors"
        )
    elif args.boundary_source == "power_one_initial_motion_mask":
        if curves_dir is None:
            raise FileNotFoundError(
                "power=1 initial-motion-mask boundaries require the saved curves directory"
            )
        (
            initial_motion_starts,
            initial_motion_base_mask_ends,
            initial_reset_jump_speeds,
        ) = abc_initial_motion_masks(
            dataset_dir,
            ep_task,
            excluded_indices=manifest.get("action", {}).get("gripper_indices", []),
            after_start_frames=args.initial_motion_after_start_frames,
            velocity_threshold=args.initial_motion_velocity_threshold,
            sustain_frames=args.initial_motion_sustain_frames,
            reset_jump_window_frames=args.initial_reset_jump_window_frames,
        )
        provisional_threshold, _provisional_count = global_initial_masked_score_mean(
            curves_dir,
            initial_motion_base_mask_ends,
            score_power=1.0,
            smooth_window=int(detector.get("smooth_window", 7)),
            savgol_polyorder=int(detector.get("savgol_polyorder", 4)),
        )
        initial_motion_mask_ends, initial_reset_extensions = extend_reset_jump_masks(
            curves_dir,
            initial_motion_base_mask_ends,
            initial_reset_jump_speeds,
            provisional_threshold=provisional_threshold,
            reset_jump_velocity_threshold=args.initial_reset_jump_velocity_threshold,
            stable_anchors=args.initial_reset_stable_anchors,
            max_extra_frames=args.initial_reset_max_extra_frames,
            smooth_window=int(detector.get("smooth_window", 7)),
            savgol_polyorder=int(detector.get("savgol_polyorder", 4)),
        )
        power_one_initial_threshold, power_one_initial_threshold_count = (
            global_initial_masked_score_mean(
                curves_dir,
                initial_motion_mask_ends,
                score_power=1.0,
                smooth_window=int(detector.get("smooth_window", 7)),
                savgol_polyorder=int(detector.get("savgol_polyorder", 4)),
            )
        )
        starts = np.asarray(list(initial_motion_starts.values()), dtype=np.int64)
        ends = np.asarray(list(initial_motion_mask_ends.values()), dtype=np.int64)
        print(
            "ABC initial-motion mask: "
            f"start median={int(np.median(starts))}, p10/p90="
            f"{int(np.percentile(starts, 10))}/{int(np.percentile(starts, 90))}; "
            f"mask-end median={int(np.median(ends))}"
        )
        if initial_reset_extensions:
            details = ", ".join(
                f"ep{episode_id}: {initial_motion_base_mask_ends[episode_id]}"
                f"->{mask_end} (jump={initial_reset_jump_speeds[episode_id]:.3f})"
                for episode_id, mask_end in sorted(initial_reset_extensions.items())
            )
            print(
                f"ABC reset-jump extensions ({len(initial_reset_extensions)}): {details}"
            )
        print(
            f"SBD power=1 initial-masked global mean={power_one_initial_threshold:.8f} "
            f"from {power_one_initial_threshold_count:,} valid anchors"
        )
    elif args.boundary_source in {
        "within_normalized_l2",
        "covariance_gated_cosine",
        "covariance_gated_cosine_initial_motion_mask",
        "covariance_gated_angular_chord",
        "covariance_gated_angular_chord_initial_motion_mask",
        "gmm_overlap_entropy",
        "strong_covariance_gated_cosine",
        "strong_covariance_gated_cosine_initial_motion_mask",
    }:
        if not args.gmm_raw_dir:
            raise FileNotFoundError(
                f"{args.boundary_source} boundaries require --gmm_raw_dir descriptor caches"
            )
        cached_score_metric = args.boundary_source
        cached_initial_mask_ends = None
        cached_initial_mask_source = args.boundary_source in {
            "covariance_gated_cosine_initial_motion_mask",
            "covariance_gated_angular_chord_initial_motion_mask",
            "strong_covariance_gated_cosine_initial_motion_mask",
        }
        if cached_initial_mask_source:
            if curves_dir is None:
                raise FileNotFoundError(
                    "angular-chord initial-motion masking requires the saved cosine "
                    "curves to reproduce reset-aware mask endpoints"
                )
            (
                initial_motion_starts,
                initial_motion_base_mask_ends,
                initial_motion_mask_ends,
                initial_reset_jump_speeds,
                initial_reset_extensions,
            ) = abc_reset_aware_initial_masks(
                dataset_dir,
                ep_task,
                curves_dir,
                excluded_indices=manifest.get("action", {}).get("gripper_indices", []),
                after_start_frames=args.initial_motion_after_start_frames,
                velocity_threshold=args.initial_motion_velocity_threshold,
                sustain_frames=args.initial_motion_sustain_frames,
                reset_jump_window_frames=args.initial_reset_jump_window_frames,
                reset_jump_velocity_threshold=args.initial_reset_jump_velocity_threshold,
                stable_anchors=args.initial_reset_stable_anchors,
                max_extra_frames=args.initial_reset_max_extra_frames,
                smooth_window=int(detector.get("smooth_window", 7)),
                savgol_polyorder=int(detector.get("savgol_polyorder", 4)),
            )
            cached_score_metric = {
                "covariance_gated_cosine_initial_motion_mask": "covariance_gated_cosine",
                "covariance_gated_angular_chord_initial_motion_mask": (
                    "covariance_gated_angular_chord"
                ),
                "strong_covariance_gated_cosine_initial_motion_mask": (
                    "strong_covariance_gated_cosine"
                ),
            }[args.boundary_source]
            cached_initial_mask_ends = initial_motion_mask_ends
            starts = np.asarray(list(initial_motion_starts.values()), dtype=np.int64)
            ends = np.asarray(list(initial_motion_mask_ends.values()), dtype=np.int64)
            print(
                f"ABC {cached_score_metric} initial-motion mask: "
                f"start median={int(np.median(starts))}, p10/p90="
                f"{int(np.percentile(starts, 10))}/{int(np.percentile(starts, 90))}; "
                f"mask-end median={int(np.median(ends))}; "
                f"reset extensions={len(initial_reset_extensions)}"
            )
        (
            cached_metric_curves,
            cached_metric_threshold,
            cached_metric_threshold_count,
        ) = cached_gmm_metric_curves(
            args.gmm_raw_dir,
            score_metric=cached_score_metric,
            initial_mask_ends=cached_initial_mask_ends,
            terminal_mask_starts=(
                home_return_starts if args.home_return_mask else None
            ),
            components=int(args.gmm_components),
            min_effective_samples=float(args.gmm_min_effective_samples),
            terminal_mask_frames=int(
                args.boundary_terminal_mask_frames
                if args.boundary_terminal_mask_frames is not None
                else detector.get("terminal_mask_frames", 0)
            ),
            terminal_mask_from_episode_end=bool(
                args.boundary_terminal_mask_from_episode_end
            ),
            smooth_window=int(detector.get("smooth_window", 7)),
            savgol_polyorder=int(detector.get("savgol_polyorder", 4)),
            score_power=score_power,
        )
        missing_metric_episodes = sorted(
            set(selected_episode_ids) - set(cached_metric_curves)
        )
        if missing_metric_episodes:
            raise FileNotFoundError(
                f"{args.boundary_source} GMM caches are missing selected episodes: "
                f"{missing_metric_episodes}"
            )
        print(
            f"{args.boundary_source} GMM global mean={cached_metric_threshold:.8f} "
            f"from {cached_metric_threshold_count:,} valid anchors across "
            f"{len(cached_metric_curves)} cached episodes",
            flush=True,
        )

    cards = []
    for task_label, eps in selected_groups:
        task_cards = []
        task_lengths = []
        for ep in eps:
            built_skills, gripper_signal, resolved_gripper_indices = load_episode_skills(
                ep_files[ep], configured_gripper_indices
            )
            skills = built_skills
            raw = frames_src(int(ep)) if frames_src is not None else None
            hide_all_metrics = args.hide_cos_graph and args.hide_gain_graph and args.hide_bic_graph
            curve = (
                None
                if hide_all_metrics
                else load_boundary_curve(str(curves_dir) if curves_dir else None, ep)
            )
            if args.boundary_source == "standard_recut":
                if standard_recut_threshold is None or int(ep) not in standard_recut_curves:
                    raise FileNotFoundError(f"Re-cut standard curve is unavailable for episode {ep}")
                curve = dict(standard_recut_curves[int(ep)])
                forced_home_boundary = home_return_starts.get(int(ep))
                if forced_home_boundary is None:
                    forced_home_boundary = saved_abc_terminal_boundary(curve)
                skills, peak_ts, peak_vals = skills_from_sbd_score(
                    curve,
                    threshold=standard_recut_threshold,
                    nms_dist=boundary_nms_dist,
                    min_skill_len=boundary_min_skill_len,
                    smooth_window=boundary_smooth_window,
                    savgol_polyorder=boundary_savgol_polyorder,
                    forced_terminal_boundary=forced_home_boundary,
                )
                curve["peak_ts"] = peak_ts
                curve["peak_vals"] = peak_vals
                curve["mean_val"] = np.asarray(
                    standard_recut_threshold, dtype=np.float32
                )
                curve["threshold_val"] = np.asarray(
                    standard_recut_threshold, dtype=np.float32
                )
                curve["threshold_pool_points"] = np.asarray(
                    standard_recut_threshold_count, dtype=np.int64
                )
                if args.home_return_mask and forced_home_boundary is None:
                    curve["terminal_release_detection_status"] = np.asarray(
                        home_return_failures.get(int(ep), "fallback")
                    )
            elif args.boundary_source == "denoising_gain":
                if curve is None or gain_threshold is None:
                    raise FileNotFoundError(f"Denoising-gain curve is unavailable for episode {ep}")
                skills, gain_peak_ts, gain_peak_vals = skills_from_denoising_gain(
                    curve,
                    threshold=gain_threshold,
                    nms_dist=boundary_nms_dist,
                    min_skill_len=boundary_min_skill_len,
                )
                curve["denoising_gain_threshold"] = np.asarray(gain_threshold, dtype=np.float32)
                curve["denoising_gain_threshold_count"] = np.asarray(
                    gain_threshold_count, dtype=np.int64
                )
                curve["denoising_gain_peak_ts"] = gain_peak_ts
                curve["denoising_gain_peak_vals"] = gain_peak_vals
            elif args.boundary_source == "all_peaks":
                if curve is None:
                    raise FileNotFoundError(f"SBD score curve is unavailable for episode {ep}")
                skills, all_peak_ts, all_peak_vals = skills_from_sbd_score(
                    curve,
                    threshold=-np.inf,
                    nms_dist=boundary_nms_dist,
                    min_skill_len=boundary_min_skill_len,
                    smooth_window=int(detector.get("smooth_window", 7)),
                    savgol_polyorder=int(detector.get("savgol_polyorder", 4)),
                )
                curve["peak_ts"] = all_peak_ts
                curve["peak_vals"] = all_peak_vals
                curve["hide_score_threshold"] = np.asarray(True)
            elif args.boundary_source == "power_half_mean":
                if curve is None or power_half_threshold is None:
                    raise FileNotFoundError(f"SBD score curve is unavailable for episode {ep}")
                exponent = 0.5 / score_power
                power_half_curve = dict(curve)
                power_half_curve["sg_vals"] = np.power(
                    np.maximum(np.asarray(curve["sg_vals"], dtype=np.float64), 0.0),
                    exponent,
                ).astype(np.float32)
                power_half_curve["score_power"] = np.asarray(0.5, dtype=np.float32)
                skills, power_half_peak_ts, power_half_peak_vals = skills_from_sbd_score(
                    power_half_curve,
                    threshold=power_half_threshold,
                    nms_dist=boundary_nms_dist,
                    min_skill_len=boundary_min_skill_len,
                    smooth_window=int(detector.get("smooth_window", 7)),
                    savgol_polyorder=int(detector.get("savgol_polyorder", 4)),
                )
                curve["sg_vals"] = power_half_curve["sg_vals"]
                curve["score_power"] = np.asarray(0.5, dtype=np.float32)
                curve["mean_val"] = np.asarray(power_half_threshold, dtype=np.float32)
                curve["threshold_val"] = np.asarray(power_half_threshold, dtype=np.float32)
                curve["peak_ts"] = power_half_peak_ts
                curve["peak_vals"] = power_half_peak_vals
                curve["hide_score_threshold"] = np.asarray(False)
                curve["threshold_pool_points"] = np.asarray(
                    power_half_threshold_count, dtype=np.int64
                )
            elif args.boundary_source == "power_one_initial_motion_mask":
                if curve is None or power_one_initial_threshold is None:
                    raise FileNotFoundError(f"SBD score curve is unavailable for episode {ep}")
                mask_end = int(initial_motion_mask_ends[int(ep)])
                score, start_index, valid_count = initial_masked_score_curve(
                    curve,
                    initial_mask_until_frame=mask_end,
                    score_power=1.0,
                    smooth_window=int(detector.get("smooth_window", 7)),
                    savgol_polyorder=int(detector.get("savgol_polyorder", 4)),
                )
                power_one_curve = dict(curve)
                power_one_curve["sg_vals"] = score.astype(np.float32)
                power_one_curve["score_power"] = np.asarray(1.0, dtype=np.float32)
                skills, peak_ts, peak_vals = skills_from_sbd_score(
                    power_one_curve,
                    threshold=power_one_initial_threshold,
                    nms_dist=boundary_nms_dist,
                    min_skill_len=boundary_min_skill_len,
                    smooth_window=int(detector.get("smooth_window", 7)),
                    savgol_polyorder=int(detector.get("savgol_polyorder", 4)),
                    initial_mask_until_frame=mask_end,
                )
                curve["sg_vals"] = score.astype(np.float32)
                curve["score_power"] = np.asarray(1.0, dtype=np.float32)
                curve["mean_val"] = np.asarray(power_one_initial_threshold, dtype=np.float32)
                curve["threshold_val"] = np.asarray(
                    power_one_initial_threshold, dtype=np.float32
                )
                curve["peak_ts"] = peak_ts
                curve["peak_vals"] = peak_vals
                curve["threshold_valid_start"] = np.asarray(start_index, dtype=np.int64)
                curve["threshold_valid_count"] = np.asarray(valid_count, dtype=np.int64)
                curve["initial_motion_start_frame"] = np.asarray(
                    initial_motion_starts[int(ep)], dtype=np.int64
                )
                curve["initial_mask_base_end_frame"] = np.asarray(
                    initial_motion_base_mask_ends[int(ep)], dtype=np.int64
                )
                curve["initial_mask_until_frame"] = np.asarray(mask_end, dtype=np.int64)
                curve["initial_reset_jump_speed"] = np.asarray(
                    initial_reset_jump_speeds[int(ep)], dtype=np.float32
                )
                curve["threshold_pool_points"] = np.asarray(
                    power_one_initial_threshold_count, dtype=np.int64
                )
            elif args.boundary_source in {
                "within_normalized_l2",
                "covariance_gated_cosine",
                "covariance_gated_cosine_initial_motion_mask",
                "covariance_gated_angular_chord",
                "covariance_gated_angular_chord_initial_motion_mask",
                "gmm_overlap_entropy",
                "strong_covariance_gated_cosine",
                "strong_covariance_gated_cosine_initial_motion_mask",
            }:
                if cached_metric_threshold is None:
                    raise RuntimeError(
                        f"{args.boundary_source} threshold was not calibrated"
                    )
                diagnostic_curve = curve
                curve = dict(cached_metric_curves[int(ep)])
                if diagnostic_curve is not None:
                    for key in (
                        "denoising_gain",
                        "denoising_gain_sg",
                        "delta_bic",
                        "delta_bic_sg",
                        "bic_k1",
                        "bic_best_multi",
                        "bic_best_k",
                        "future_horizon",
                        "tail_excluded_frames",
                    ):
                        if key in diagnostic_curve:
                            curve[key] = diagnostic_curve[key]
                skills, peak_ts, peak_vals = skills_from_sbd_score(
                    curve,
                    threshold=cached_metric_threshold,
                    nms_dist=boundary_nms_dist,
                    min_skill_len=boundary_min_skill_len,
                    smooth_window=int(detector.get("smooth_window", 7)),
                    savgol_polyorder=int(detector.get("savgol_polyorder", 4)),
                    initial_mask_until_frame=(
                        int(initial_motion_mask_ends[int(ep)])
                        if args.boundary_source in {
                            "covariance_gated_cosine_initial_motion_mask",
                            "covariance_gated_angular_chord_initial_motion_mask",
                            "strong_covariance_gated_cosine_initial_motion_mask",
                        }
                        else None
                    ),
                    forced_terminal_boundary=(
                        home_return_starts.get(int(ep))
                        if args.home_return_mask
                        else None
                    ),
                )
                curve["peak_ts"] = peak_ts
                curve["peak_vals"] = peak_vals
                curve["mean_val"] = np.asarray(
                    cached_metric_threshold, dtype=np.float32
                )
                curve["threshold_val"] = np.asarray(
                    cached_metric_threshold, dtype=np.float32
                )
                curve["threshold_pool_points"] = np.asarray(
                    cached_metric_threshold_count, dtype=np.int64
                )
                if (
                    args.boundary_source in {
                        "covariance_gated_cosine_initial_motion_mask",
                        "covariance_gated_angular_chord_initial_motion_mask",
                        "strong_covariance_gated_cosine_initial_motion_mask",
                    }
                ):
                    curve["initial_motion_start_frame"] = np.asarray(
                        initial_motion_starts[int(ep)], dtype=np.int64
                    )
                    curve["initial_mask_base_end_frame"] = np.asarray(
                        initial_motion_base_mask_ends[int(ep)], dtype=np.int64
                    )
                    curve["initial_reset_jump_speed"] = np.asarray(
                        initial_reset_jump_speeds[int(ep)], dtype=np.float32
                    )
            if curve is not None and power1_mean is not None:
                curve["score_power1_mean"] = np.asarray(power1_mean, dtype=np.float32)
                curve["score_power1_mean_count"] = np.asarray(
                    power1_mean_count, dtype=np.int64
                )
            if curve is not None and gain_power2_mean is not None:
                curve["denoising_gain_power2_mean"] = np.asarray(
                    gain_power2_mean, dtype=np.float32
                )
                curve["denoising_gain_power2_mean_count"] = np.asarray(
                    gain_power2_mean_count, dtype=np.int64
                )
                curve["denoising_gain_ylim"] = np.asarray(gain_ylim, dtype=np.float32)
                curve["denoising_gain_power2_ylim"] = np.asarray(
                    gain_power2_ylim, dtype=np.float32
                )
            if curve is not None:
                if args.hide_cos_graph:
                    for key in (
                        "div_cos",
                        "sg_vals",
                        "mean_val",
                        "threshold_val",
                        "peak_ts",
                        "peak_vals",
                    ):
                        curve.pop(key, None)
                if args.hide_gain_graph:
                    for key in ("denoising_gain", "denoising_gain_sg"):
                        curve.pop(key, None)
                if args.hide_bic_graph:
                    for key in (
                        "delta_bic",
                        "delta_bic_sg",
                        "bic_k1",
                        "bic_best_multi",
                        "bic_best_k",
                    ):
                        curve.pop(key, None)
            if action_error_dir is not None:
                action_error_path = action_error_dir / f"ep{int(ep):07d}.npz"
                if action_error_path.is_file():
                    if curve is None:
                        curve = {"n_frames": np.array(skills[-1][1] if skills else 0)}
                    with np.load(action_error_path, allow_pickle=False) as action_error:
                        for key in action_error.files:
                            curve[key] = action_error[key].copy()
            skill_videos = (
                _skill_video_metadata(
                    dataset_dir,
                    episodes_meta,
                    image_key,
                    int(ep),
                    skills,
                    out_dir,
                    fps,
                )
                if args.skill_video
                else None
            )
            episode_video = (
                _episode_video_metadata(
                    dataset_dir,
                    episodes_meta,
                    image_key,
                    int(ep),
                    out_dir,
                    fps,
                    clip_cache_dir=episode_clip_cache_dir,
                )
                if args.episode_video
                else None
            )
            media = render_skillset_card(
                skills,
                raw,
                curve,
                thumb=args.thumb_size,
                gripper_signal=gripper_signal,
                gripper_labels=_gripper_labels(dataset_dir, resolved_gripper_indices),
                skill_videos=skill_videos,
                episode_video=episode_video,
                gmm_visualization=gmm_visualizations.get(int(ep)),
                show_frames=not args.hide_start_end_frames,
            )
            lengths = [end - start for start, end, _video in skills]
            task_lengths.extend(lengths)
            task_cards.append((f"{_cap(task_label, ep)} — {_skill_stats(lengths)}", media))
        # Section header (task id + instruction) gets the stats aggregated over the
        # episodes shown for that task; each episode caption carries its own stats.
        section_label = None
        if task_label is not None:
            section_label = (
                f"task{int(task_label):02d}: "
                f"{display_instructions.get(int(task_label), '(instruction unavailable)')}"
                f" — {_skill_stats(task_lengths)}"
            )
        cards.extend((section_label, caption, media) for caption, media in task_cards)
    save_gallery(out_dir, args.title, cards, filename=out_html)
    print(f"[dp_eval] done → {out_dir / out_html}")


if __name__ == "__main__":
    main()
