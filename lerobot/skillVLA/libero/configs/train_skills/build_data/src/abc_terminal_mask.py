"""ABC bimanual terminal-motion detection shared by production skill builds."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def _resolve_indices(indices, dim: int) -> tuple[int, ...]:
    resolved: list[int] = []
    for raw in indices:
        index = int(raw)
        if index < 0:
            index += dim
        if index < 0 or index >= dim:
            raise IndexError(f"Index {raw} is out of range for dim={dim}")
        if index not in resolved:
            resolved.append(index)
    return tuple(resolved)


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    edges = np.diff(np.concatenate(([False], values, [False])).astype(np.int8))
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
    """Detect the final sustained return toward the episode's initial arm pose."""
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or len(values) < 60:
        return None, {"status": "too_short"}
    excluded = set(_resolve_indices(excluded_indices, values.shape[1]))
    arm_indices = np.asarray(
        [index for index in range(values.shape[1]) if index not in excluded],
        dtype=np.int64,
    )
    if not len(arm_indices):
        raise ValueError("Home-return detection has no state dimensions")
    arm = values[:, arm_indices]
    reference_count = min(int(home_reference_frames), max(5, len(arm) // 10))
    home = np.median(arm[:reference_count], axis=0)
    if joint_scale is None:
        scale = np.ones(len(arm_indices), dtype=np.float64)
    else:
        raw_scale = np.asarray(joint_scale, dtype=np.float64).reshape(-1)
        scale = raw_scale[arm_indices] if len(raw_scale) == values.shape[1] else raw_scale
        if len(scale) != len(arm_indices):
            raise ValueError("joint_scale is incompatible with the state dimension")
        scale = np.maximum(scale, 0.05)
    distance = np.sqrt(np.mean(np.square((arm - home) / scale), axis=1))
    window = min(
        int(distance_smooth_frames) | 1,
        len(distance) if len(distance) % 2 else len(distance) - 1,
    )
    smoothed = (
        savgol_filter(distance, window_length=window, polyorder=2)
        if window >= 5
        else distance.copy()
    )
    excursion = float(np.max(smoothed))
    final_distance = float(np.median(distance[-reference_count:]))
    home_noise = float(np.percentile(distance[:reference_count], 90))
    home_band = max(3.0 * home_noise, 2.0 * final_distance, 0.12 * excursion, 0.04)
    final_ratio = final_distance / max(excursion, np.finfo(np.float64).eps)
    if final_ratio > float(final_home_ratio_max):
        return None, {"status": "final_not_home", "final_ratio": final_ratio}
    home_runs = [
        run
        for run in _true_runs(smoothed <= home_band)
        if run[1] >= len(arm) - reference_count and run[1] - run[0] + 1 >= 15
    ]
    if not home_runs:
        return None, {"status": "no_final_home_hold", "final_ratio": final_ratio}
    arrival = int(home_runs[-1][0])
    horizon = int(progress_horizon_frames)
    progress = smoothed[:-horizon] - smoothed[horizon:]
    toward_home = progress > max(0.015, 0.015 * excursion)
    search_start = max(0, arrival - int(max_search_frames))
    tail = toward_home[search_start:arrival].copy()
    runs = _true_runs(tail)
    if not runs:
        return None, {"status": "no_toward_home_run", "final_ratio": final_ratio}
    for (_left_start, left_end), (right_start, _right_end) in zip(
        runs[:-1], runs[1:], strict=True
    ):
        if right_start - left_end - 1 <= horizon:
            tail[left_end + 1:right_start] = True
    candidates: list[tuple[int, float]] = []
    for local_start, local_end in _true_runs(tail):
        start = search_start + local_start
        drop = float(smoothed[start] - smoothed[min(arrival, len(arm) - 1)])
        if local_end - local_start + 1 >= 10 and drop >= max(0.12, 0.15 * excursion):
            candidates.append((start, drop))
    if not candidates:
        return None, {"status": "weak_return", "final_ratio": final_ratio}
    start, drop = candidates[-1]
    start = min(arrival, int(start) + horizon)
    return start, {
        "status": "ok",
        "final_ratio": final_ratio,
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
    """Locate the last lasting gripper opening before terminal motion."""
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or len(values) < 30:
        return None, {"status": "too_short"}
    resolved = _resolve_indices(gripper_indices, values.shape[1])
    if not resolved:
        raise ValueError("Terminal-release detection requires gripper indices")
    horizon = int(progress_horizon_frames)
    search_start = max(0, len(values) - int(search_frames))
    candidates: list[tuple[int, int, float, float, int, float]] = []
    for index in resolved:
        signal = values[:, index]
        window = min(11, len(signal) if len(signal) % 2 else len(signal) - 1)
        smoothed = (
            savgol_filter(signal, window_length=window, polyorder=2)
            if window >= 5
            else signal.copy()
        )
        dynamic_range = float(np.percentile(smoothed, 98) - np.percentile(smoothed, 2))
        progress = smoothed[horizon:] - smoothed[:-horizon]
        opening = progress > max(0.008, 0.025 * dynamic_range)
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
            post_reclose = float(smoothed[post_peak] - np.min(smoothed[post_peak:]))
            if post_reclose > max(0.08, 0.35 * local_gain):
                continue
            candidates.append(
                (onset, index, lasting_gain, local_gain, end, post_reclose)
            )
    if not candidates:
        return None, {"status": "no_terminal_release"}
    onset, index, lasting_gain, local_gain, end, post_reclose = max(
        candidates, key=lambda item: item[0]
    )
    return onset, {
        "status": "release",
        "gripper_index": float(index),
        "candidate_count": float(len(candidates)),
        "lasting_open_gain": lasting_gain,
        "local_open_gain": local_gain,
        "release_end_frame": float(end),
        "post_release_reclose": post_reclose,
    }


def detect_terminal_mask_start(
    states: np.ndarray,
    *,
    gripper_indices,
    state_scale: np.ndarray | None,
    pre_release_offset_frames: int,
) -> tuple[int | None, str, dict]:
    """Return ABC's semantic terminal mask start, with home-return fallback."""
    release, release_info = detect_terminal_gripper_release_start(
        states, gripper_indices=gripper_indices
    )
    home, home_info = detect_home_return_start(
        states,
        excluded_indices=gripper_indices,
        joint_scale=state_scale,
    )
    if release is not None:
        return (
            max(0, int(release) - int(pre_release_offset_frames)),
            "release",
            {"release": release_info, "home": home_info},
        )
    if home is not None:
        return int(home), "fallback_home_return", {"release": release_info, "home": home_info}
    return None, "no_release_or_home_return", {"release": release_info, "home": home_info}
