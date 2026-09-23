#!/usr/bin/env python3
"""Fresh-inference action-error diagnostic for DP boundary comparisons.

This deliberately does not rebuild or edit a skillset.  It resolves the DP
checkpoint from each selected skillset manifest, runs the checkpoints on the
same episode anchors with deterministic shared diffusion noise, and compares
their predicted action chunks with the demonstration targets.
"""

from __future__ import annotations

import gc
import html
import json
import re
from pathlib import Path

import numpy as np

LIBERO_EXAMPLES = Path(__file__).resolve().parents[4]
PROJECT_ROOT = LIBERO_EXAMPLES.parents[2]


def _manifest(skillset_dir: Path) -> dict:
    path = skillset_dir / "skillset_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Skillset manifest not found: {path}")
    return json.loads(path.read_text())


def _portable_path(raw_path: str, marker: str) -> Path:
    """Use a manifest path locally, with a project-root fallback after marker."""
    path = Path(str(raw_path)).expanduser()
    if path.exists():
        return path
    parts = path.parts
    if marker in parts:
        candidate = PROJECT_ROOT.joinpath(*parts[parts.index(marker) :])
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Manifest path does not exist locally: {path}")


def _selected_episode_ids(
    skillset_dir: Path,
    *,
    task_ids: list[int],
    task_id_space: str,
    target_task: str,
    n_episodes: int,
) -> list[int]:
    from codebook_visualizer import _load_episodes_meta
    from dp_skillset_eval import (
        _task_instruction_map,
        index_skillset,
        select_episodes,
        suite_to_dataset_task_ids,
    )

    ep_task, _ = index_skillset(skillset_dir / "skills")
    manifest = _manifest(skillset_dir)
    dataset_dir = _portable_path(str(manifest["dataset_dir"]), "dataset_filtered")
    episodes_meta = _load_episodes_meta(dataset_dir)
    selected = list(task_ids)
    if selected and task_id_space == "suite":
        selected = suite_to_dataset_task_ids(
            selected,
            suite_name=target_task,
            instructions=_task_instruction_map(episodes_meta, ep_task),
        )
    return [
        int(episode_id)
        for _task_id, episodes in select_episodes(ep_task, selected, n_episodes)
        for episode_id in episodes
    ]


def _moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return values
    window = min(int(window), len(values))
    if window <= 1:
        return values
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._-")
    return text or "model"


def _episode_action_error(
    *,
    policy,
    preprocessor,
    normalizer,
    ep_df,
    replan_interval: int,
    batch_size: int,
    seed: int,
    action_mode: str,
    relative_mask,
) -> dict[str, np.ndarray]:
    import torch
    from lerobot.datasets.proprio_grounding import ground_state_xyz, normalize_proprio_grounding
    from action_manifold import to_model_action_chunk
    from skill_divider import _aligned_action_chunk, _valid_replan_anchors
    from lerobot.utils.constants import OBS_STATE

    if not bool(getattr(policy.config, "state_only", False)):
        raise ValueError(
            "dp_eval_action_error currently targets the state-only DP comparison; "
            "the selected checkpoint consumes visual features."
        )

    states = np.stack(ep_df["observation.state"].values).astype(np.float32)
    actions = np.stack(ep_df["action"].values).astype(np.float32)
    grounding = normalize_proprio_grounding(
        getattr(policy.config, "proprio_grounding", "none")
    )
    if grounding == "episode_start_xyz":
        states = ground_state_xyz(states, states[0, :3])

    # Apply exactly the checkpoint's state normalizer/device pipeline once per
    # frame, then assemble its configured observation histories below.
    processed_states = []
    for state in states:
        processed = preprocessor({OBS_STATE: torch.from_numpy(state).float()})[OBS_STATE]
        if processed.ndim == 2 and processed.shape[0] == 1:
            processed = processed[0]
        processed_states.append(processed)
    state_tensor = torch.stack(processed_states)

    cfg = policy.config
    horizon = int(cfg.horizon)
    future_start = int(cfg.action_execution_start_index)
    future_horizon = int(cfg.action_prediction_horizon)
    action_indices = [int(value) for value in cfg.action_delta_indices]
    anchors = _valid_replan_anchors(len(states), future_horizon, replan_interval)
    n_obs = int(cfg.n_obs_steps)

    histories = []
    target_model_chunks = []
    target_normalized_chunks = []
    for anchor in anchors:
        indices = list(range(max(0, anchor - n_obs + 1), anchor + 1))
        indices = [indices[0]] * (n_obs - len(indices)) + indices
        histories.append(state_tensor[indices])
        raw_chunk = _aligned_action_chunk(actions, anchor, action_indices)
        model_chunk = to_model_action_chunk(
            raw_chunk,
            states[anchor],
            action_mode=action_mode,
            relative_mask=relative_mask,
        )
        target_model_chunks.append(model_chunk)
        target_normalized_chunks.append(normalizer.normalize(model_chunk))

    target_model = np.stack(target_model_chunks).astype(np.float32)
    target_normalized = np.stack(target_normalized_chunks).astype(np.float32)
    predictions = []
    parameter = next(policy.parameters())
    # Generate one anchor-specific noise tensor up front. Models with the same
    # horizon/action shape therefore receive exactly the same stochastic input,
    # independent of diagnostic batch size.
    rng = np.random.default_rng(int(seed))
    shared_noise = rng.standard_normal(
        (len(anchors), horizon, target_normalized.shape[-1]), dtype=np.float32
    )
    with torch.inference_mode():
        for start in range(0, len(anchors), batch_size):
            end = min(len(anchors), start + batch_size)
            batch_states = torch.stack(histories[start:end])
            global_cond = policy.diffusion._prepare_global_conditioning(
                {OBS_STATE: batch_states}
            )
            noise = torch.from_numpy(shared_noise[start:end]).to(
                device=parameter.device, dtype=parameter.dtype
            )
            prediction = policy.diffusion.conditional_sample(
                end - start,
                global_cond=global_cond,
                noise=noise,
            )
            predictions.append(prediction.float().cpu().numpy())
    predicted_normalized = np.concatenate(predictions, axis=0)
    predicted_model = normalizer.denormalize(predicted_normalized)

    region = slice(future_start, future_start + future_horizon)
    norm_delta = predicted_normalized[:, region] - target_normalized[:, region]
    raw_delta = predicted_model[:, region] - target_model[:, region]
    dim = raw_delta.shape[-1]

    def component(indices: list[int]) -> np.ndarray:
        valid = [index for index in indices if index < dim]
        if not valid:
            return np.full(len(anchors), np.nan, dtype=np.float32)
        return np.mean(raw_delta[..., valid] ** 2, axis=(1, 2)).astype(np.float32)

    return {
        "anchors": np.asarray(anchors, dtype=np.int64),
        "normalized_mse": np.mean(norm_delta**2, axis=(1, 2)).astype(np.float32),
        "xyz_mse": component([0, 1, 2]),
        "rotation_mse": component([3, 4, 5]),
        "gripper_mse": component(list(range(6, dim))),
    }


def _plot_episode(
    episode_id: int,
    curves: list[tuple[str, dict[str, np.ndarray]]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        ("normalized_mse", "Normalized action MSE"),
        ("xyz_mse", "Raw XYZ MSE"),
        ("rotation_mse", "Raw rotation MSE"),
        ("gripper_mse", "Raw gripper MSE"),
    )
    figure, axes = plt.subplots(len(metrics), 1, figsize=(12, 9), sharex=True)
    for axis, (key, title) in zip(axes, metrics, strict=True):
        for label, curve in curves:
            x = curve["anchors"]
            y = curve[key]
            axis.plot(x, y, alpha=0.2, linewidth=1)
            axis.plot(x, _moving_average(y), linewidth=2, label=label)
        axis.set_ylabel(title)
        axis.grid(alpha=0.2)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("episode frame / replanning anchor")
    figure.suptitle(f"DP action prediction error · episode {episode_id}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=145)
    plt.close(figure)


_CACHED_CURVE_KEYS = {
    "anchors": "action_error_replan_ts",
    "normalized_mse": "action_error_normalized_mse",
    "xyz_mse": "action_error_xyz_mse",
    "rotation_mse": "action_error_rotation_mse",
    "gripper_mse": "action_error_gripper_mse",
}


def _load_cached_curve(path: Path, expected: dict) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            required = set(_CACHED_CURVE_KEYS.values()) | {
                "cache_schema",
                "policy_path",
                "episode_seed",
                "replan_interval",
                "num_inference_steps",
            }
            if not required.issubset(data.files):
                return None
            actual = {
                "cache_schema": int(data["cache_schema"]),
                "policy_path": str(data["policy_path"]),
                "episode_seed": int(data["episode_seed"]),
                "replan_interval": int(data["replan_interval"]),
                "num_inference_steps": int(data["num_inference_steps"]),
            }
            if actual != expected:
                return None
            return {
                internal: data[stored].copy()
                for internal, stored in _CACHED_CURVE_KEYS.items()
            }
    except (OSError, ValueError, KeyError):
        return None


def _save_cached_curve(path: Path, curve: dict[str, np.ndarray], expected: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        action_error_replan_ts=curve["anchors"],
        action_error_normalized_mse=curve["normalized_mse"],
        action_error_normalized_mse_smooth=_moving_average(
            curve["normalized_mse"]
        ).astype(np.float32),
        action_error_xyz_mse=curve["xyz_mse"],
        action_error_rotation_mse=curve["rotation_mse"],
        action_error_gripper_mse=curve["gripper_mse"],
        cache_schema=np.array(expected["cache_schema"], dtype=np.int16),
        policy_path=np.array(expected["policy_path"]),
        episode_seed=np.array(expected["episode_seed"], dtype=np.int64),
        replan_interval=np.array(expected["replan_interval"], dtype=np.int64),
        num_inference_steps=np.array(expected["num_inference_steps"], dtype=np.int64),
    )


def run_action_error_comparison(
    specs: list[dict],
    *,
    labels: list[str],
    task_ids: list[int],
    task_id_space: str,
    target_task: str,
    n_episodes: int,
    output_dir: Path,
    batch_size: int,
    seed: int,
    resume: bool = False,
) -> Path:
    import torch
    from action_manifold import (
        ACTION_MODE_ANCHOR_RELATIVE,
        ACTION_MODE_DATASET,
        NumpyActionNormalizer,
    )
    from skill_divider import load_data, load_policy

    by_label = {str(spec["label"]): spec for spec in specs}
    selected_specs = [by_label[label] for label in labels]
    first_skillset = Path(selected_specs[0]["skillset_dir"])
    episode_ids = _selected_episode_ids(
        first_skillset,
        task_ids=task_ids,
        task_id_space=task_id_space,
        target_task=target_task,
        n_episodes=n_episodes,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_curves: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    summaries = []

    for model_index, spec in enumerate(selected_specs):
        label = str(spec["label"])
        skillset_dir = Path(spec["skillset_dir"])
        manifest = _manifest(skillset_dir)
        dataset_dir = _portable_path(str(manifest["dataset_dir"]), "dataset_filtered")
        policy_path = _portable_path(str(manifest["policy_path"]), "outputs_filtered")
        detector = manifest.get("detector", {})
        action = manifest.get("action", {})
        policy_config = json.loads((policy_path / "config.json").read_text())
        if not bool(policy_config.get("state_only", False)):
            raise ValueError(f"Action-error model {label!r} is not state-only.")
        replan_interval = int(detector.get("replan_interval", 3))
        num_inference_steps = int(detector.get("num_inference_steps", 10))
        curve_dir = output_dir / "curves" / _slug(label)
        curve_dir.mkdir(parents=True, exist_ok=True)
        curves = {}
        missing_episode_ids = []
        expected_by_episode = {}
        for episode_id in episode_ids:
            expected = {
                "cache_schema": 1,
                "policy_path": str(policy_path.resolve()),
                "episode_seed": seed + int(episode_id) * 1009,
                "replan_interval": replan_interval,
                "num_inference_steps": num_inference_steps,
            }
            expected_by_episode[int(episode_id)] = expected
            cached = (
                _load_cached_curve(
                    curve_dir / f"ep{int(episode_id):07d}.npz", expected
                )
                if resume
                else None
            )
            if cached is None:
                missing_episode_ids.append(int(episode_id))
            else:
                curves[int(episode_id)] = cached
                print(f"[resume action-error] {label} · ep{episode_id}", flush=True)

        action_mode = str(action.get("mode", ACTION_MODE_DATASET))
        relative_mask = None
        if action_mode == ACTION_MODE_ANCHOR_RELATIVE:
            action_shape = policy_config.get("output_features", {}).get("action", {}).get("shape")
            if not action_shape:
                action_shape = policy_config.get("input_features", {}).get("action", {}).get("shape")
            dim = int(action_shape[0])
            relative_mask = np.ones(dim, dtype=bool)
            for raw_index in action.get("gripper_indices", []):
                index = int(raw_index)
                relative_mask[index if index >= 0 else dim + index] = False
        elif action_mode != ACTION_MODE_DATASET:
            raise ValueError(f"Unsupported action mode for {label}: {action_mode}")

        policy = preprocessor = normalizer = data = None
        if missing_episode_ids:
            policy, preprocessor = load_policy(
                str(policy_path),
                "cuda",
                str(detector.get("noise_scheduler_type", "DDIM")),
                num_inference_steps,
            )
            normalizer = NumpyActionNormalizer.from_preprocessor(preprocessor)
            data = load_data(dataset_dir, episode_ids=missing_episode_ids)
            for episode_id in missing_episode_ids:
                ep_df = data[data["episode_index"] == episode_id].reset_index(drop=True)
                if ep_df.empty:
                    raise ValueError(f"Episode {episode_id} missing from {dataset_dir}")
                print(
                    f"[action-error {model_index + 1}/{len(selected_specs)}] "
                    f"{label} · ep{episode_id}",
                    flush=True,
                )
                curve = _episode_action_error(
                    policy=policy,
                    preprocessor=preprocessor,
                    normalizer=normalizer,
                    ep_df=ep_df,
                    replan_interval=replan_interval,
                    batch_size=batch_size,
                    seed=expected_by_episode[episode_id]["episode_seed"],
                    action_mode=action_mode,
                    relative_mask=relative_mask,
                )
                curves[int(episode_id)] = curve
                # Save immediately: a preemption later in this model resumes at
                # the next missing episode rather than repeating completed GPU work.
                _save_cached_curve(
                    curve_dir / f"ep{int(episode_id):07d}.npz",
                    curve,
                    expected_by_episode[episode_id],
                )
        model_curves[label] = curves
        metric_summary = {
            key: float(np.mean(np.concatenate([curve[key] for curve in curves.values()])))
            for key in ("normalized_mse", "xyz_mse", "rotation_mse", "gripper_mse")
        }
        summaries.append(
            {
                "label": label,
                "policy_path": str(policy_path),
                "n_obs_steps": int(policy_config["n_obs_steps"]),
                "future_horizon": int(
                    policy_config.get("action_prediction_horizon")
                    or policy_config["horizon"]
                ),
                **metric_summary,
            }
        )
        del policy, preprocessor, normalizer, data
        gc.collect()
        torch.cuda.empty_cache()

    image_rows = []
    for episode_id in episode_ids:
        image_name = f"episode_{episode_id:07d}.png"
        image_path = output_dir / image_name
        source_paths = [
            output_dir / "curves" / _slug(label) / f"ep{episode_id:07d}.npz"
            for label in labels
        ]
        image_current = (
            resume
            and image_path.is_file()
            and all(path.is_file() for path in source_paths)
            and image_path.stat().st_mtime
            >= max(path.stat().st_mtime for path in source_paths)
        )
        if not image_current:
            _plot_episode(
                episode_id,
                [(label, model_curves[label][episode_id]) for label in labels],
                image_path,
            )
        image_rows.append(
            f"<article><h2>episode {episode_id}</h2>"
            f"<img loading='lazy' src='{image_name}'></article>"
        )

    headers = "".join(
        f"<th>{html.escape(key)}</th>"
        for key in ("model", "obs", "future", "norm MSE", "XYZ", "rotation", "gripper")
    )
    table_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['label'])}</td><td>{row['n_obs_steps']}</td>"
        f"<td>{row['future_horizon']}</td>"
        f"<td>{row['normalized_mse']:.6g}</td><td>{row['xyz_mse']:.6g}</td>"
        f"<td>{row['rotation_mse']:.6g}</td><td>{row['gripper_mse']:.6g}</td>"
        "</tr>"
        for row in summaries
    )
    report = output_dir / "index.html"
    report.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>DP action-error diagnostic</title><style>"
        "body{font-family:Inter,system-ui,sans-serif;background:#f4f6fa;color:#172033;"
        "margin:0;padding:24px}h1{margin-top:0}article,table{background:white;"
        "border:1px solid #d7deea;border-radius:12px;margin:16px 0;padding:14px}"
        "img{display:block;width:100%;height:auto}table{border-collapse:collapse;width:100%}"
        "th,td{border-bottom:1px solid #e2e8f0;padding:8px;text-align:right}"
        "th:first-child,td:first-child{text-align:left}</style></head><body>"
        "<h1>DP action prediction error</h1>"
        "<p>동일 episode와 동일 anchor별 diffusion noise로 비교합니다. "
        "희미한 선은 raw 값, 진한 선은 5-anchor moving average입니다.</p>"
        f"<table><thead><tr>{headers}</tr></thead><tbody>{table_rows}</tbody></table>"
        + "".join(image_rows)
        + "</body></html>",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
