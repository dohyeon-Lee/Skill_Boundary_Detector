#!/usr/bin/env python
"""Episode-exact, single-skill Stage-1 rollout evaluation."""

import gc
import json
import logging
import os
import sys
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs, preprocess_observation
from lerobot.scripts.lerobot_skillvla_eval import _libero_task_descriptions
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.random_utils import set_seed

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / "stage1_eval" / "src"))

from html_report import write_html_report  # noqa: E402
from merge_results import maybe_merge_chunks, report_payload  # noqa: E402
from run_eval import RAW_IMAGE, RAW_STATE, RAW_WRIST, _build_context  # noqa: E402
from skill_data import SkillEvaluationDataset, SkillOccurrence  # noqa: E402

log = logging.getLogger(__name__)

BRANCHES = (
    ("gt", "GT actions", "#2e7d32"),
    ("policy", "Policy · exact start", "#1565c0"),
    ("policy_alt_noise", "Policy · exact start · different noise", "#00838f"),
    ("policy_early", "Policy · early start", "#ef6c00"),
    ("policy_late", "Policy · late start", "#8e24aa"),
)
ALT_NOISE_SEED_OFFSET = 1_000_003


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _terminator_fired(
    *,
    mode: str,
    progress: float,
    termination: float,
    progress_threshold: float,
    end_threshold: float,
) -> bool:
    progress_high = progress >= progress_threshold
    termination_high = termination >= end_threshold
    if mode == "progress":
        return progress_high
    if mode == "termination":
        return termination_high
    if mode == "and":
        return progress_high and termination_high
    return progress_high or termination_high


def _restore_state(base_env, state: np.ndarray):
    # Reset controller internals first, then install the exact per-frame MuJoCo
    # state. No settling/no-op step is allowed: that would change the requested
    # skill start state before the first recorded or predicted action.
    base_env._env.reset()
    raw_obs = base_env._env.set_init_state(np.asarray(state, dtype=np.float64))
    for robot in base_env._env.robots:
        robot.controller.use_delta = True
    return raw_obs


def _render(base_env) -> np.ndarray:
    return np.asarray(base_env.render(), dtype=np.uint8).copy()


def _add_batch_dimension(value):
    """Batch one direct (non-VectorEnv) LIBERO observation recursively."""
    if isinstance(value, dict):
        return {key: _add_batch_dimension(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value[None, ...]
    return value


def _prepare_observation(
    *,
    base_env,
    raw_obs,
    env_preprocessor,
    preprocessor,
) -> tuple[dict[str, Any], np.ndarray]:
    # preprocess_observation adds a batch axis for a bare image, but not for the
    # nested robot_state arrays. Batch the complete direct-env observation here
    # so LiberoProcessorStep receives e.g. quaternion (1, 4), exactly as in the
    # standard SyncVectorEnv Stage-1 evaluator.
    formatted = _add_batch_dimension(base_env._format_raw_obs(raw_obs))
    observation = preprocess_observation(formatted)
    observation["task"] = [str(base_env.task_description)]
    env_observation = env_preprocessor(observation)
    restored_state = (
        env_observation[OBS_STATE].detach().cpu().numpy()[0].astype(np.float32)
    )
    return preprocessor(env_observation), restored_state


def _postprocess_action(action, postprocessor, env_postprocessor) -> np.ndarray:
    action = postprocessor(action)
    action = env_postprocessor({ACTION: action})[ACTION]
    action_numpy = action.detach().to("cpu").numpy()
    if action_numpy.shape != (1, 7):
        raise ValueError(f"Expected one LIBERO action with shape (1, 7), got {action_numpy.shape}.")
    return action_numpy[0].astype(np.float32)


def _query_terminator(
    *,
    base_env,
    raw_obs,
    token: int,
    context: dict,
    env_preprocessor,
) -> tuple[dict[str, Any], np.ndarray, float, float]:
    batch, restored_state = _prepare_observation(
        base_env=base_env,
        raw_obs=raw_obs,
        env_preprocessor=env_preprocessor,
        preprocessor=context["preprocessor"],
    )
    policy = context["policy"].policy
    terminator = context["policy"].terminator
    if terminator is None:
        raise RuntimeError("Predicted-end skill eval requires a terminator.")
    device = next(policy.parameters()).device
    codes = torch.tensor([int(token)], dtype=torch.long, device=device)
    missing = [key for key in (RAW_STATE, RAW_IMAGE, RAW_WRIST) if key not in batch]
    if missing:
        raise ValueError(f"Policy preprocessor omitted terminator inputs: {missing}.")
    current_progress, current_termination = terminator.terminate(
        codes,
        batch[RAW_STATE],
        batch[RAW_IMAGE],
        batch[RAW_WRIST],
    )
    return (
        batch,
        restored_state,
        float(current_progress[0]),
        float(current_termination[0]),
    )


def _load_font(size: int):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _annotate_frames(
    frames: list[np.ndarray],
    *,
    progress: list[float | None] | None = None,
    termination: list[float | None] | None = None,
    progress_threshold: float,
    end_threshold: float,
) -> list[np.ndarray]:
    """Append progress/termination gauges without adding title/frame banners."""
    if not frames:
        return []
    height, width = frames[0].shape[:2]
    gauge_width = max(44, width // 6)
    font = _load_font(max(9, gauge_width // 5))

    def gauge(
        label: str,
        value: float | None,
        threshold: float,
        below_color: tuple[int, int, int],
        above_color: tuple[int, int, int],
    ) -> np.ndarray:
        panel = Image.new("RGB", (gauge_width, height), (22, 24, 28))
        draw = ImageDraw.Draw(panel)
        track_x0 = max(7, int(gauge_width * 0.30))
        track_x1 = min(gauge_width - 7, int(gauge_width * 0.70))
        track_top = max(22, height // 9)
        track_bottom = height - max(22, height // 10)
        track_height = max(1, track_bottom - track_top)
        title_width = draw.textlength(label, font=font)
        draw.text(((gauge_width - title_width) / 2, 4), label, fill=(235, 235, 235), font=font)
        valid = value is not None and np.isfinite(float(value))
        if valid:
            clipped = float(np.clip(float(value), 0.0, 1.0))
            fill_bottom = track_bottom - 2
            fill_top = min(
                fill_bottom,
                track_bottom - max(2, round(track_height * clipped)),
            )
            draw.rectangle(
                (track_x0 + 2, fill_top, track_x1 - 2, fill_bottom),
                fill=above_color if clipped >= threshold else below_color,
            )
            value_label = f"{round(clipped * 100):d}%"
        else:
            value_label = "N/A"
        draw.rectangle(
            (track_x0, track_top, track_x1, track_bottom),
            outline=(210, 210, 210),
            width=2,
        )
        threshold_y = track_bottom - round(track_height * float(np.clip(threshold, 0.0, 1.0)))
        draw.line(
            (track_x0 - 4, threshold_y, track_x1 + 4, threshold_y),
            fill=(255, 196, 48),
            width=3,
        )
        label_width = draw.textlength(value_label, font=font)
        draw.text(
            ((gauge_width - label_width) / 2, height - max(18, height // 12)),
            value_label,
            fill=(245, 245, 245),
            font=font,
        )
        return np.asarray(panel, dtype=np.uint8)

    annotated = []
    for index, frame in enumerate(frames):
        progress_value = progress[index] if progress and index < len(progress) else None
        termination_value = (
            termination[index] if termination and index < len(termination) else None
        )
        annotated.append(
            np.concatenate(
                [
                    np.asarray(frame, dtype=np.uint8),
                    gauge(
                        "PROG",
                        progress_value,
                        progress_threshold,
                        (52, 152, 219),
                        (46, 204, 113),
                    ),
                    gauge(
                        "TERM",
                        termination_value,
                        end_threshold,
                        (155, 89, 182),
                        (231, 76, 60),
                    ),
                ],
                axis=1,
            )
        )
    return annotated


def _write_branch_video(
    path: Path,
    frames: list[np.ndarray],
    *,
    frame_stride: int,
    fps: int,
    progress: list[float | None] | None = None,
    termination: list[float | None] | None = None,
    progress_threshold: float,
    end_threshold: float,
) -> None:
    if not frames:
        raise ValueError(f"Cannot write an empty video: {path}")
    indices = list(range(0, len(frames), frame_stride))
    if indices[-1] != len(frames) - 1:
        indices.append(len(frames) - 1)
    selected = [frames[index] for index in indices]
    selected_progress = [progress[index] for index in indices] if progress else None
    selected_termination = [termination[index] for index in indices] if termination else None
    selected = _annotate_frames(
        selected,
        progress=selected_progress,
        termination=selected_termination,
        progress_threshold=progress_threshold,
        end_threshold=end_threshold,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        str(path),
        np.stack(selected),
        fps=int(fps),
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )


def _read_video(path: Path) -> tuple[list[np.ndarray], float]:
    reader = imageio.get_reader(str(path))
    try:
        metadata = reader.get_meta_data()
        frames = [np.asarray(frame, dtype=np.uint8) for frame in reader]
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"Cannot compose an empty video: {path}")
    return frames, float(metadata.get("fps", 10.0))


def _write_comparison_video(
    path: Path,
    *,
    output_dir: Path,
    branches: list[dict],
    fps: int,
) -> None:
    """Horizontally synchronize all GT/policy branches into one playable video."""
    frame_sets: list[list[np.ndarray] | None] = []
    reference_shape = None
    for branch in branches:
        relative_path = branch.get("path")
        if relative_path:
            frames, _ = _read_video(output_dir / relative_path)
            if reference_shape is None:
                reference_shape = frames[0].shape
            elif frames[0].shape != reference_shape:
                raise ValueError(
                    f"Comparison panels must share one shape: {frames[0].shape} != {reference_shape}."
                )
            frame_sets.append(frames)
        else:
            frame_sets.append(None)
    if reference_shape is None:
        raise ValueError("At least one branch video is required to build a comparison.")
    frame_count = max(len(frames) for frames in frame_sets if frames is not None)
    unavailable = np.full(reference_shape, 28, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(path),
        fps=int(fps),
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    try:
        for frame_index in range(frame_count):
            panels = [
                unavailable
                if frames is None
                else frames[min(frame_index, len(frames) - 1)]
                for frames in frame_sets
            ]
            writer.append_data(np.hstack(panels))
    finally:
        writer.close()


def _write_final_frame_image(path: Path, frame: np.ndarray) -> None:
    """Atomically save the five-panel terminal state without loading the MP4 in HTML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".png.tmp")
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
        temporary,
        format="PNG",
        optimize=True,
    )
    temporary.replace(path)


def _extract_boundary_frames(
    video_path: Path,
    start_image_path: Path,
    final_image_path: Path,
) -> None:
    """Extract a poster and terminal comparison image in one decode pass."""
    frames, _ = _read_video(video_path)
    _write_final_frame_image(start_image_path, frames[0])
    _write_final_frame_image(final_image_path, frames[-1])


def _run_gt_actions(
    *,
    base_env,
    state: np.ndarray,
    actions: np.ndarray,
    token: int,
    context: dict,
    env_preprocessor,
) -> dict:
    raw_obs = _restore_state(base_env, state)
    frames = [_render(base_env)]
    progress_values: list[float | None] = []
    termination_values: list[float | None] = []
    stop_reason = "gt_frame_end"
    steps = 0
    for action in np.asarray(actions, dtype=np.float32):
        _, _, progress, termination = _query_terminator(
            base_env=base_env,
            raw_obs=raw_obs,
            token=token,
            context=context,
            env_preprocessor=env_preprocessor,
        )
        progress_values.append(progress)
        termination_values.append(termination)
        raw_obs, _, done, _ = base_env._env.step(action)
        steps += 1
        frames.append(_render(base_env))
        if bool(done):
            stop_reason = "environment_done"
            break
    # Also annotate the state reached by the final GT action.
    _, _, progress, termination = _query_terminator(
        base_env=base_env,
        raw_obs=raw_obs,
        token=token,
        context=context,
        env_preprocessor=env_preprocessor,
    )
    progress_values.append(progress)
    termination_values.append(termination)
    return {
        "frames": frames,
        "steps": steps,
        "stop_reason": stop_reason,
        "progress": progress_values,
        "termination": termination_values,
    }


def _run_policy(
    *,
    base_env,
    state: np.ndarray,
    expected_filtered_state: np.ndarray,
    token: int,
    context: dict,
    env_preprocessor,
    env_postprocessor,
    max_skill_length: int,
    n_action_steps: int,
    end_mode: str,
    end_threshold: float,
    progress_threshold: float,
    finish_action_chunk_on_end: bool,
    seed: int,
) -> dict:
    set_seed(int(seed))
    policy = context["policy"].policy
    policy.reset()
    action_queue: deque[torch.Tensor] = deque()
    raw_obs = _restore_state(base_env, state)
    frames = [_render(base_env)]
    progress_values: list[float | None] = []
    termination_values: list[float | None] = []
    pending_end = False
    stop_reason = "max_skill_length"
    steps = 0
    restored_state_rms = None

    while steps < int(max_skill_length):
        batch, restored_state, progress, termination = _query_terminator(
            base_env=base_env,
            raw_obs=raw_obs,
            token=token,
            context=context,
            env_preprocessor=env_preprocessor,
        )
        if restored_state_rms is None:
            expected = np.asarray(expected_filtered_state, dtype=np.float32)
            restored_state_rms = float(np.sqrt(np.mean((restored_state - expected) ** 2)))
        progress_values.append(progress)
        termination_values.append(termination)
        fired = _terminator_fired(
            mode=end_mode,
            progress=progress,
            termination=termination,
            progress_threshold=progress_threshold,
            end_threshold=end_threshold,
        )
        if fired:
            pending_end = True
            stop_reason = "predicted_end"
        if pending_end and (not finish_action_chunk_on_end or not action_queue):
            break

        if not action_queue:
            device = next(policy.parameters()).device
            codes = torch.tensor([int(token)], dtype=torch.long, device=device)
            action_batch = dict(batch)
            action_batch["skill_code"] = codes
            action_batch["skill_sequence"] = codes[:, None]
            action_batch["skill_index"] = torch.zeros(1, dtype=torch.long, device=device)
            chunk = policy.predict_action_chunk(action_batch)
            action_queue.extend(chunk[:, :n_action_steps].transpose(0, 1))
        action_numpy = _postprocess_action(
            action_queue.popleft(),
            context["postprocessor"],
            env_postprocessor,
        )
        raw_obs, _, done, _ = base_env._env.step(action_numpy)
        steps += 1
        frames.append(_render(base_env))
        if bool(done):
            stop_reason = "environment_done"
            break

    # One value per rendered state. If max length or env done occurred directly
    # after an action, carry the latest signal into the final video frame.
    if not progress_values:
        progress_values = [None]
        termination_values = [None]
    while len(progress_values) < len(frames):
        progress_values.append(progress_values[-1])
        termination_values.append(termination_values[-1])
    return {
        "frames": frames,
        "steps": steps,
        "stop_reason": stop_reason,
        "progress": progress_values,
        "termination": termination_values,
        "restored_state_rms": restored_state_rms,
    }


def _branch_start_frame(occurrence: SkillOccurrence, branch: str, offset: int) -> tuple[int | None, str | None]:
    if branch in {"gt", "policy", "policy_alt_noise"}:
        return occurrence.frame_start, None
    if branch == "policy_early":
        frame = occurrence.frame_start - int(offset)
        if frame < 0:
            return None, f"f{occurrence.frame_start}-{offset} is before episode start"
        return frame, None
    if branch == "policy_late":
        frame = occurrence.frame_start + int(offset)
        if frame >= occurrence.frame_end:
            return None, (
                f"f{occurrence.frame_start}+{offset} reaches/passes skill end f{occurrence.frame_end}"
            )
        return frame, None
    raise ValueError(f"Unknown branch {branch!r}.")


def _manifest_signature(spec: dict, cfg, selected: dict[int, list[int]]) -> dict:
    return {
        "format": "stage1_skill_eval_v5_video_poster",
        "policy_path": str(spec["policy_path"]),
        "external_skill_model": str(spec.get("external_skill_model") or ""),
        "advance_mode": str(spec["advance_mode"]),
        "target_task": str(cfg.env.task),
        "selected_episodes": {str(key): value for key, value in selected.items()},
        "time_shift_offset": int(os.environ["TIME_SHIFT_OFFSET"]),
        "n_action_steps": int(cfg.policy.n_action_steps),
        "end_mode": os.environ["SKILL_END_MODE"],
        "end_threshold": float(os.environ["SKILL_END_THRESHOLD"]),
        "progress_threshold": float(os.environ["SKILL_END_PROGRESS_THRESHOLD"]),
        "max_skill_length": int(os.environ["INFERENCE_SKILL_MAX_LENGTH"]),
        "finish_action_chunk_on_end": _as_bool(os.environ["FINISH_ACTION_CHUNK_ON_END"]),
        "seed": int(cfg.seed),
    }


def _save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    spec = json.loads(os.environ.get("SPEC_JSON", "") or "{}")
    if not spec:
        raise ValueError("SPEC_JSON is empty; resolve stage1_skill_eval_config.yaml first.")
    device = get_safe_torch_device(cfg.policy.device, log=True)
    set_seed(cfg.seed)

    dataset = SkillEvaluationDataset(
        skill_dataset_dir=os.environ["SKILL_DATASET_DIR"],
        skill_latents_path=os.environ["SKILL_LATENTS_PATH"],
        eval_init_states_path=os.environ["EVAL_INIT_STATES_PATH"],
        original_dataset_dir=os.environ["ORIGINAL_DATASET_DIR"],
        suite_name=cfg.env.task,
    )
    selected = dataset.select_episodes(
        task_ids=list(cfg.env.task_ids or []),
        episodes_per_task=int(os.environ["EPISODES_PER_TASK"]),
        selection=os.environ["EPISODE_SELECTION"],
        seed=cfg.seed,
        explicit_episode_ids=json.loads(os.environ.get("EPISODE_IDS", "[]")),
    )
    all_occurrences = dataset.occurrences(selected)
    if not all_occurrences:
        raise RuntimeError("No skill occurrences were found in the selected exact episodes.")
    worker_count = int(os.environ.get("SKILL_EVAL_WORKER_COUNT", "1"))
    worker_index = int(os.environ.get("SKILL_EVAL_WORKER_INDEX", "0"))
    if worker_count <= 0 or not 0 <= worker_index < worker_count:
        raise ValueError(
            f"Invalid skill-eval worker index/count: {worker_index}/{worker_count}."
        )
    occurrences = all_occurrences[worker_index::worker_count]
    if not occurrences:
        raise RuntimeError(
            f"Worker {worker_index}/{worker_count} received no occurrences; "
            f"total={len(all_occurrences)}."
        )
    log.info(
        "worker=%d/%d occurrences=%d/%d",
        worker_index,
        worker_count,
        len(occurrences),
        len(all_occurrences),
    )
    # Preflight only the source episodes used by this worker before allocating
    # the policy. Other workers independently verify their own exact mappings.
    for episode_id in sorted({occurrence.episode_id for occurrence in occurrences}):
        aligned = dataset.load_aligned_episode(episode_id)
        log.info(
            "episode=%s task=%s demo=%s aligned=%s max_action_error=%.3e",
            episode_id,
            aligned.source.task_id,
            aligned.source.demo,
            len(aligned.original_action_indices),
            aligned.alignment_max_error,
        )

    envs = make_env(
        cfg.env,
        n_envs=1,
        use_async_envs=False,
        trust_remote_code=cfg.trust_remote_code,
    )
    output_dir = Path(cfg.output_dir)
    consolidated_path = output_dir / "metrics" / "manifest.json"
    manifest_path = (
        consolidated_path
        if worker_count == 1
        else output_dir / "metrics" / "chunks" / f"chunk_{worker_index:03d}.json"
    )
    signature = _manifest_signature(spec, cfg, selected)
    resume = _as_bool(os.environ.get("EVAL_RESUME", "false"))
    if worker_count > 1 and consolidated_path.is_file() and not resume:
        raise FileExistsError(
            f"Output already contains {consolidated_path}; set resume: true or choose another output_name."
        )
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if not resume:
            raise FileExistsError(
                f"Output already contains {manifest_path}; set resume: true or choose another output_name."
            )
        if existing.get("signature") != signature:
            raise ValueError("resume=true but existing manifest signature does not match this evaluation.")
        manifest = existing
    else:
        manifest = {
            "signature": signature,
            "model_label": spec["label"],
            "architecture_label": spec.get("architecture_label", ""),
            "chunk_index": worker_index,
            "chunk_count": worker_count,
            "completed": False,
            "records": {},
        }
        _save_manifest(manifest_path, manifest)

    try:
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=cfg.env,
            policy_cfg=cfg.policy,
        )
        context = _build_context(spec, cfg, device)
        try:
            levels = [int(value) for value in context["config"].skill_fsq_levels]
            manifest["levels"] = levels
            _save_manifest(manifest_path, manifest)
            max_token = int(np.prod(levels))
            invalid_tokens = sorted({occ.token for occ in occurrences if not 0 <= occ.token < max_token})
            if invalid_tokens:
                raise ValueError(f"Skill tokens outside FSQ{levels}: {invalid_tokens}")

            shift = int(os.environ["TIME_SHIFT_OFFSET"])
            frame_stride = int(os.environ["VIDEO_FRAME_STRIDE"])
            video_fps = int(os.environ["VIDEO_FPS"])
            end_mode = os.environ["SKILL_END_MODE"]
            end_threshold = float(os.environ["SKILL_END_THRESHOLD"])
            progress_threshold = float(os.environ["SKILL_END_PROGRESS_THRESHOLD"])
            max_skill_length = int(os.environ["INFERENCE_SKILL_MAX_LENGTH"])
            finish_chunk = _as_bool(os.environ["FINISH_ACTION_CHUNK_ON_END"])
            task_descriptions = _libero_task_descriptions(cfg.env.task)

            inference_context = (
                torch.autocast(device_type=device.type)
                if context["config"].use_amp
                else nullcontext()
            )
            with torch.inference_mode(), inference_context:
                for occurrence_index, occurrence in enumerate(occurrences):
                    aligned = dataset.load_aligned_episode(occurrence.episode_id)
                    vec_env = envs[cfg.env.task][occurrence.task_id]
                    base_env = vec_env.envs[0].unwrapped
                    record = manifest["records"].get(occurrence.uid)
                    if record is None:
                        record = {
                            "uid": occurrence.uid,
                            "token": occurrence.token,
                            "task_id": occurrence.task_id,
                            "task_description": task_descriptions.get(occurrence.task_id, ""),
                            "episode_id": occurrence.episode_id,
                            "skill_index": occurrence.skill_index,
                            "frame_start": occurrence.frame_start,
                            "frame_end": occurrence.frame_end,
                            "length": occurrence.length,
                            "scene_file": aligned.source.scene_file,
                            "demo": aligned.source.demo,
                            "alignment_mean_error": aligned.alignment_mean_error,
                            "alignment_max_error": aligned.alignment_max_error,
                            "branches": [],
                        }
                        manifest["records"][occurrence.uid] = record
                    existing_branches = {branch["name"]: branch for branch in record["branches"]}
                    branch_records = []
                    common_seed = int(cfg.seed) + occurrence.episode_id * 1009 + occurrence.skill_index * 17
                    for branch_name, branch_label, branch_color in BRANCHES:
                        branch_seed = common_seed + (
                            ALT_NOISE_SEED_OFFSET
                            if branch_name == "policy_alt_noise"
                            else 0
                        )
                        start_frame, unavailable = _branch_start_frame(occurrence, branch_name, shift)
                        existing_branch = existing_branches.get(branch_name)
                        if existing_branch is not None:
                            existing_path = existing_branch.get("path")
                            if existing_path is None or (output_dir / existing_path).is_file():
                                branch_records.append(existing_branch)
                                continue
                        if unavailable is not None:
                            branch_records.append(
                                {
                                    "name": branch_name,
                                    "label": branch_label,
                                    "color": branch_color,
                                    "path": None,
                                    "unavailable_reason": unavailable,
                                    "start_frame": None,
                                    "original_start_frame": None,
                                    "requested_offset": (
                                        -shift if branch_name == "policy_early" else shift
                                    ),
                                    "steps": 0,
                                    "stop_reason": "invalid_shift",
                                    "final_progress": None,
                                    "final_termination": None,
                                }
                            )
                            continue
                        assert start_frame is not None
                        state = aligned.state_at(start_frame)
                        offset = start_frame - occurrence.frame_start
                        relative_path = (
                            Path("videos")
                            / f"task_{occurrence.task_id:02d}"
                            / f"token_{occurrence.token:04d}"
                            / occurrence.uid
                            / f"{branch_name}.mp4"
                        )
                        log.info(
                            "[%d/%d] token=%d ep=%d skill=%d branch=%s start=%d (offset=%+d)",
                            occurrence_index + 1,
                            len(occurrences),
                            occurrence.token,
                            occurrence.episode_id,
                            occurrence.skill_index,
                            branch_name,
                            start_frame,
                            offset,
                        )
                        if branch_name == "gt":
                            result = _run_gt_actions(
                                base_env=base_env,
                                state=state,
                                actions=aligned.filtered_actions[
                                    occurrence.frame_start : occurrence.frame_end
                                ],
                                token=occurrence.token,
                                context=context,
                                env_preprocessor=env_preprocessor,
                            )
                        else:
                            result = _run_policy(
                                base_env=base_env,
                                state=state,
                                expected_filtered_state=aligned.filtered_states[start_frame],
                                token=occurrence.token,
                                context=context,
                                env_preprocessor=env_preprocessor,
                                env_postprocessor=env_postprocessor,
                                max_skill_length=max_skill_length,
                                n_action_steps=int(cfg.policy.n_action_steps),
                                end_mode=end_mode,
                                end_threshold=end_threshold,
                                progress_threshold=progress_threshold,
                                finish_action_chunk_on_end=finish_chunk,
                                seed=branch_seed,
                            )
                        _write_branch_video(
                            output_dir / relative_path,
                            result["frames"],
                            frame_stride=frame_stride,
                            fps=video_fps,
                            progress=result["progress"],
                            termination=result["termination"],
                            progress_threshold=progress_threshold,
                            end_threshold=end_threshold,
                        )
                        branch_records.append(
                            {
                                "name": branch_name,
                                "label": branch_label,
                                "color": branch_color,
                                "path": relative_path.as_posix(),
                                "unavailable_reason": None,
                                "start_frame": start_frame,
                                "original_start_frame": aligned.original_frame_at(start_frame),
                                "requested_offset": offset,
                                "steps": int(result["steps"]),
                                "stop_reason": result["stop_reason"],
                                "final_progress": (
                                    None
                                    if not result["progress"] or result["progress"][-1] is None
                                    else float(result["progress"][-1])
                                ),
                                "final_termination": (
                                    None
                                    if not result["termination"] or result["termination"][-1] is None
                                    else float(result["termination"][-1])
                                ),
                                "restored_state_rms": result.get("restored_state_rms"),
                                "noise_seed": None if branch_name == "gt" else branch_seed,
                            }
                        )
                        record["branches"] = branch_records
                        _save_manifest(manifest_path, manifest)
                    record["branches"] = branch_records
                    comparison_path = (
                        Path("videos")
                        / f"task_{occurrence.task_id:02d}"
                        / f"token_{occurrence.token:04d}"
                        / occurrence.uid
                        / "comparison.mp4"
                    )
                    comparison_final_path = comparison_path.with_name(
                        "comparison_final.png"
                    )
                    comparison_start_path = comparison_path.with_name(
                        "comparison_start.png"
                    )
                    if not (output_dir / comparison_path).is_file():
                        _write_comparison_video(
                            output_dir / comparison_path,
                            output_dir=output_dir,
                            branches=branch_records,
                            fps=video_fps,
                        )
                    if not (
                        (output_dir / comparison_start_path).is_file()
                        and (output_dir / comparison_final_path).is_file()
                    ):
                        _extract_boundary_frames(
                            output_dir / comparison_path,
                            output_dir / comparison_start_path,
                            output_dir / comparison_final_path,
                        )
                    record["comparison_path"] = comparison_path.as_posix()
                    record["comparison_start_path"] = comparison_start_path.as_posix()
                    record["comparison_final_path"] = comparison_final_path.as_posix()
                    _save_manifest(manifest_path, manifest)

            manifest["completed"] = True
            _save_manifest(manifest_path, manifest)
            if worker_count == 1:
                report = write_html_report(
                    output_dir,
                    report_payload(manifest, levels=levels),
                )
                print(f"Saved report: {report}")
            else:
                report = maybe_merge_chunks(
                    output_dir,
                    expected_chunks=worker_count,
                )
                if report is None:
                    print(
                        f"Saved worker chunk {worker_index + 1}/{worker_count}; "
                        "HTML will be generated by the last finishing worker."
                    )
                else:
                    print(f"All {worker_count} chunks complete; saved report: {report}")
            print(
                f"Worker occurrences: {len(manifest['records'])} / "
                f"total {len(all_occurrences)}"
            )
        finally:
            del context
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        close_envs(envs)


if __name__ == "__main__":
    eval_main()
