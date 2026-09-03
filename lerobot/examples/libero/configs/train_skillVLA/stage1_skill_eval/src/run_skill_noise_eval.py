#!/usr/bin/env python
"""Repeated skill-start policy-noise rollouts with trajectory-only recording."""

import gc
import json
import logging
import os
import sys
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from robosuite.utils.camera_utils import get_camera_transform_matrix

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.random_utils import set_seed

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from noise_merge_results import (  # noqa: E402
    maybe_merge_noise_chunks,
    write_noise_html_reports,
)
from run_skill_eval import (  # noqa: E402
    _as_bool,
    _build_context,
    _environment_layout_seed,
    _is_environment_done,
    _mark_startup_ready,
    _postprocess_action,
    _query_terminator,
    _render,
    _reset_terminators,
    _restore_skill_start,
    _rollout_max_skill_length,
    _run_inline_cuda_guard,
    _save_manifest,
    _set_episode_grounding_reference,
    _task_success,
    _terminator_fired,
)
from skill_data import SkillEvaluationDataset  # noqa: E402

log = logging.getLogger(__name__)


def _worker_units(
    *,
    model_count: int,
    selected: dict[int, list[int]],
    noise_rollouts: int,
    worker_index: int,
    worker_count: int,
) -> dict[int, list[tuple[int, int]]]:
    """Assign model x selected environment x noise-index units to this worker."""
    if model_count <= 0 or noise_rollouts <= 0:
        raise ValueError("model_count and noise_rollouts must be positive.")
    if worker_count <= 0 or not 0 <= worker_index < worker_count:
        raise ValueError(f"Invalid worker index/count {worker_index}/{worker_count}.")
    episode_ids = [
        int(episode_id)
        for task_id in sorted(selected)
        for episode_id in selected[task_id]
    ]
    if not episode_ids:
        raise ValueError("No environments were selected.")
    all_units = [
        (model_index, episode_id, noise_index)
        for model_index in range(model_count)
        for episode_id in episode_ids
        for noise_index in range(noise_rollouts)
    ]
    assigned = all_units[worker_index::worker_count]
    if not assigned:
        raise RuntimeError(
            f"Worker {worker_index}/{worker_count} received no work; "
            f"total units={len(all_units)}."
        )
    grouped: dict[int, list[tuple[int, int]]] = {}
    for model_index, episode_id, noise_index in assigned:
        grouped.setdefault(model_index, []).append((episode_id, noise_index))
    return grouped


def _token_to_coord(token: int, levels: list[int]) -> list[int]:
    token = int(token)
    codebook_size = int(np.prod(levels, dtype=np.int64))
    if not 0 <= token < codebook_size:
        raise ValueError(
            f"Code token {token} is outside codebook [0, {codebook_size})."
        )
    coord: list[int] = []
    base = 1
    for level in levels:
        coord.append((token // base) % int(level))
        base *= int(level)
    return coord


def _coord_to_token(coord: list[int], levels: list[int]) -> int:
    if len(coord) != len(levels):
        raise ValueError(f"Coordinate {coord} does not match levels {levels}.")
    token = 0
    base = 1
    for value, level in zip(coord, levels, strict=True):
        if not 0 <= int(value) < int(level):
            raise ValueError(f"Coordinate {coord} is outside levels {levels}.")
        token += int(value) * base
        base *= int(level)
    return int(token)


def _evaluated_tokens(
    token: int,
    levels: list[int],
    *,
    probe_mode: str,
) -> list[int]:
    """Return assigned, local-neighbor, opposite-neighbor, or all codes."""
    token = int(token)
    mode = str(probe_mode).strip().lower()
    if mode == "off":
        return [token]
    codebook_size = int(np.prod(levels, dtype=np.int64))
    if mode == "all":
        return [token, *(value for value in range(codebook_size) if value != token)]
    if mode not in {"neighbor", "neighbor_and_opposite"}:
        raise ValueError(f"Unknown code probe mode: {probe_mode!r}.")
    coord = _token_to_coord(token, levels)

    def immediate_neighbors(center: list[int]) -> set[int]:
        result: set[int] = set()
        for dimension, level in enumerate(levels):
            for delta in (-1, 1):
                candidate = center.copy()
                candidate[dimension] += delta
                if 0 <= candidate[dimension] < int(level):
                    result.add(_coord_to_token(candidate, levels))
        return result

    neighbors = immediate_neighbors(coord)
    if mode == "neighbor":
        return [token, *sorted(neighbors)]

    # Mirror the selected coordinate through every FSQ axis and sample both
    # that antipodal code and its one-hop shell. A 3x3x3 corner thus contributes
    # three local codes, the three codes beside the opposite corner, and the
    # opposite corner itself, instead of all 27.
    opposite_coord = [
        int(level) - 1 - value for value, level in zip(coord, levels, strict=True)
    ]
    opposite = immediate_neighbors(opposite_coord)
    opposite.add(_coord_to_token(opposite_coord, levels))
    opposite.difference_update({token, *neighbors})

    # An exact center is its own antipode, so the two one-hop shells coincide.
    # In that special case retain a useful long-range probe by including every
    # code tied for maximum Manhattan distance (the 8 corners for FSQ333).
    if not opposite and opposite_coord == coord:
        max_distance = sum(
            max(value, int(level) - 1 - value)
            for value, level in zip(coord, levels, strict=True)
        )
        codebook_size = int(np.prod(levels, dtype=np.int64))
        opposite = {
            candidate_token
            for candidate_token in range(codebook_size)
            if sum(
                abs(candidate_value - value)
                for candidate_value, value in zip(
                    _token_to_coord(candidate_token, levels), coord, strict=True
                )
            )
            == max_distance
        }
        opposite.difference_update({token, *neighbors})
    return [token, *sorted(neighbors), *sorted(opposite)]


def _code_probe_mode(value: Any) -> str:
    if isinstance(value, bool):
        return "neighbor" if value else "off"
    text = str(value).strip().lower()
    mode = {
        "true": "neighbor",
        "false": "off",
        "none": "off",
        "assigned": "off",
        "neighbors": "neighbor",
        "neighbor+opposite": "neighbor_and_opposite",
        "neighbors_and_opposite": "neighbor_and_opposite",
    }.get(text, text)
    if mode not in {"off", "neighbor", "neighbor_and_opposite", "all"}:
        raise ValueError(
            "NEIGHBOR_CODE_PROBE must be "
            "off|neighbor|neighbor_and_opposite|all."
        )
    return mode


def _evaluated_token_roles(
    token: int,
    levels: list[int],
    *,
    probe_mode: str,
) -> dict[int, str]:
    """Label evaluated codes for consistent report colors."""
    tokens = _evaluated_tokens(token, levels, probe_mode=probe_mode)
    roles = {int(token): "original"}
    if probe_mode in {"neighbor", "neighbor_and_opposite"}:
        local = set(_evaluated_tokens(token, levels, probe_mode="neighbor"))
        local.discard(int(token))
        roles.update({candidate: "neighbor" for candidate in local})
    for candidate in tokens:
        if candidate not in roles:
            roles[candidate] = (
                "opposite" if probe_mode == "neighbor_and_opposite" else "other"
            )
    return roles


def _rollout_sampling_seed(
    seed: int,
    *,
    rollout_path: str,
    skill_flow_target: str,
) -> int:
    """Honor the auxiliary route's training-time noise-sharing contract."""
    value = int(seed)
    if rollout_path == "skill_only" and skill_flow_target == "canonical":
        value = (value + 1_000_000_007) % (2**31 - 1)
    return value


def _eef_position(raw_obs: dict[str, Any]) -> np.ndarray:
    value = raw_obs.get("robot0_eef_pos")
    if value is None:
        raise KeyError("LIBERO observation has no robot0_eef_pos for trajectory capture.")
    position = np.asarray(value, dtype=np.float64).reshape(-1)
    if position.shape != (3,):
        raise ValueError(f"Expected robot0_eef_pos shape (3,), got {position.shape}.")
    return position.copy()


def _project_trajectory(
    points: list[np.ndarray],
    transform: np.ndarray,
    *,
    height: int,
    width: int,
    stride: int,
) -> list[list[int] | None]:
    """Project world EEF points onto LiberoEnv.render()'s flipped agent view."""
    array = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate(
        [array, np.ones((len(array), 1), dtype=np.float64)], axis=1
    )
    projected = (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T
    result: list[list[int] | None] = []
    sample_indices = list(range(0, len(array), int(stride)))
    if sample_indices[-1] != len(array) - 1:
        sample_indices.append(len(array) - 1)
    for index in sample_indices:
        depth = float(projected[index, 2])
        if not np.isfinite(depth) or depth <= 1e-8:
            result.append(None)
            continue
        column = float(projected[index, 0] / depth)
        row = float(projected[index, 1] / depth)
        if (
            not np.isfinite(column)
            or not np.isfinite(row)
            or column < 0
            or column >= width
            or row < 0
            or row >= height
        ):
            result.append(None)
            continue
        # robosuite's projection is aligned with its conventional top-left
        # image. LiberoEnv.render() additionally mirrors the agent view in X.
        display_column = int(np.clip(round(width - 1 - column), 0, width - 1))
        display_row = int(np.clip(round(row), 0, height - 1))
        result.append([display_column, display_row])
    return result


def _save_start_image(path: Path, frame: np.ndarray) -> None:
    if path.is_file() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
        temporary,
        format="JPEG",
        quality=90,
        optimize=True,
    )
    os.replace(temporary, path)


def _run_noise_policy(
    *,
    base_env,
    state: np.ndarray,
    token: int,
    context: dict,
    env_preprocessor,
    env_postprocessor,
    max_skill_length: int,
    gt_skill_length: int,
    n_action_steps: int,
    end_mode: str,
    end_threshold: float,
    progress_threshold: float,
    finish_action_chunk_on_end: bool,
    seed: int,
    initial_previous_action: np.ndarray | None,
    capture_start_image: bool,
    episode_start_xyz: np.ndarray | None = None,
    model_xml: str | None = None,
    layout_seed: int | None = None,
    exact_init_state_index: int | None = None,
    replay_actions: np.ndarray | None = None,
    task_description: str | None = None,
    rollout_path: str = "main",
) -> dict:
    """Run one sample without per-step rendering or video encoding."""
    rollout_path = str(rollout_path).strip().lower()
    if rollout_path not in {"main", "skill_only"}:
        raise ValueError(f"Unknown rollout path: {rollout_path!r}.")
    policy = context["policy"].policy
    canonical_skill_only = bool(
        rollout_path == "skill_only"
        and str(getattr(policy.config, "skill_flow_target", "")) == "canonical"
    )
    policy.reset()
    _reset_terminators(context)
    _set_episode_grounding_reference(context, episode_start_xyz)
    action_queue: deque[torch.Tensor] = deque()
    raw_obs = _restore_skill_start(
        base_env,
        state,
        model_xml=model_xml,
        layout_seed=layout_seed,
        exact_init_state_index=exact_init_state_index,
        replay_actions=replay_actions,
    )
    # Keep policy sampling independent from fixture-layout randomization. The
    # canonical auxiliary was trained with noise independent of the main flow;
    # extended-chunk auxiliaries shared the main noise prefix and therefore use
    # the same seed as the corresponding main rollout.
    sampling_seed = _rollout_sampling_seed(
        seed,
        rollout_path=rollout_path,
        skill_flow_target=str(getattr(policy.config, "skill_flow_target", "")),
    )
    set_seed(sampling_seed)
    height = int(base_env.observation_height)
    width = int(base_env.observation_width)
    start_image = _render(base_env) if capture_start_image else None
    camera_transform = get_camera_transform_matrix(
        base_env._env.sim,
        camera_name="agentview",
        camera_height=height,
        camera_width=width,
    )
    trajectory = [_eef_position(raw_obs)]
    previous_action = (
        None
        if initial_previous_action is None
        else np.asarray(initial_previous_action, dtype=np.float32).copy()
    )
    pending_end = False
    skill_only_generated = False
    stop_reason = "max_skill_length"
    steps = 0
    final_progress = None
    final_termination = None
    task_success_seen = _task_success(base_env)
    task_success_step = 0 if task_success_seen else None
    environment_done_step = None

    while steps < int(max_skill_length):
        batch, _, progress, termination, _ = _query_terminator(
            base_env=base_env,
            raw_obs=raw_obs,
            token=token,
            context=context,
            env_preprocessor=env_preprocessor,
            previous_action=previous_action,
            task_description=task_description,
        )
        final_progress = None if progress is None else float(progress)
        final_termination = None if termination is None else float(termination)
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
            if canonical_skill_only and skill_only_generated:
                stop_reason = "skill_only_horizon"
                break
            device = next(policy.parameters()).device
            codes = torch.tensor([int(token)], dtype=torch.long, device=device)
            action_batch = dict(batch)
            action_batch["skill_code"] = codes
            action_batch["skill_sequence"] = codes[:, None]
            action_batch["skill_index"] = torch.zeros(
                1, dtype=torch.long, device=device
            )
            if rollout_path == "skill_only":
                chunk = policy.predict_skill_only_action_chunk(
                    action_batch,
                    horizon=(int(gt_skill_length) if canonical_skill_only else None),
                )
                skill_only_generated = True
                # This route was trained to produce one complete canonical
                # trajectory (arch0_skill) or one extended chunk
                # (*_skill_chunk). Execute that complete generated object.
                # Canonical rollout then ends; chunk modes sample another
                # extended chunk only when the rollout limit outlives it.
                queue_length = min(
                    int(chunk.shape[1]), int(max_skill_length) - steps
                )
            else:
                chunk = policy.predict_action_chunk(action_batch)
                queue_length = min(int(n_action_steps), int(chunk.shape[1]))
            action_queue.extend(chunk[:, :queue_length].transpose(0, 1))
        action_numpy = _postprocess_action(
            action_queue.popleft(),
            context["postprocessor"],
            env_postprocessor,
        )
        raw_obs, _, done, _ = base_env._env.step(action_numpy)
        previous_action = np.asarray(action_numpy, dtype=np.float32).copy()
        steps += 1
        trajectory.append(_eef_position(raw_obs))
        task_success_now = _task_success(base_env)
        if not task_success_seen and task_success_now:
            task_success_seen = True
            task_success_step = steps
        if _is_environment_done(raw_done=done, task_success=task_success_now):
            environment_done_step = steps
            stop_reason = "environment_done"
            break

    return {
        "start_image": start_image,
        "image_height": height,
        "image_width": width,
        "trajectory_world": trajectory,
        "camera_transform": camera_transform,
        "steps": steps,
        "stop_reason": stop_reason,
        "final_progress": final_progress,
        "final_termination": final_termination,
        "task_success_seen": task_success_seen,
        "task_success_step": task_success_step,
        "environment_done_step": environment_done_step,
        "sampling_seed": sampling_seed,
    }


def _signature(
    specs: list[dict],
    cfg,
    selected: dict[int, list[int]],
    noise_rollouts: int,
    trajectory_stride: int,
    code_probe_mode: str,
    skill_only_rollout_probe: bool,
) -> dict:
    return {
        "format": "stage1_skill_noise_eval_v7_skill_only_rollout",
        "policies": [
            {
                "label": str(spec["label"]),
                "policy_path": str(spec["policy_path"]),
                "fsq_path": str(spec["fsq_path"]),
                "skill_latents_path": str(spec["skill_latents_path"]),
                "raw_dataset_dir": str(spec.get("raw_dataset_dir", "")),
                "fsq_levels": [int(value) for value in spec["fsq_levels"]],
                "main_terminator": spec.get("main_terminator", {}),
                "main_terminator_path": str(spec.get("external_skill_model", "")),
                "main_terminator_variant": str(
                    spec.get("external_skill_model_variant", "")
                ),
            }
            for spec in specs
        ],
        "target_task": str(cfg.env.task),
        "episode_exact": _as_bool(os.environ.get("EPISODE_EXACT", "true")),
        "environment_mode": (
            "langgap_episode_replay"
            if str(cfg.env.task).startswith("langgap_")
            and _as_bool(os.environ.get("EPISODE_EXACT", "true"))
            else "source_demo_xml"
            if _as_bool(os.environ.get("EPISODE_EXACT", "true"))
            else "seeded_random_layout"
        ),
        "selected_episodes": {
            str(task_id): [int(value) for value in episode_ids]
            for task_id, episode_ids in selected.items()
        },
        "noise_rollouts_per_env": int(noise_rollouts),
        "code_probe_mode": str(code_probe_mode),
        "skill_only_rollout_probe": bool(skill_only_rollout_probe),
        "trajectory_stride": int(trajectory_stride),
        "n_action_steps": int(cfg.policy.n_action_steps),
        "seed": int(cfg.seed),
    }


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    _run_inline_cuda_guard()
    _mark_startup_ready()
    specs = json.loads(os.environ.get("MODELS_JSON", "") or "[]")
    if not specs:
        raise ValueError("MODELS_JSON is empty; resolve noise eval YAML first.")
    noise_rollouts = int(os.environ["NOISE_ROLLOUTS_PER_ENV"])
    code_probe_mode = _code_probe_mode(
        os.environ.get("NEIGHBOR_CODE_PROBE", "off")
    )
    skill_only_rollout_probe = _as_bool(
        os.environ.get("SKILL_ONLY_ROLLOUT_PROBE", "false")
    )
    trajectory_stride = int(os.environ["TRAJECTORY_STRIDE"])
    episode_exact = _as_bool(os.environ.get("EPISODE_EXACT", "true"))
    worker_count = int(os.environ.get("SKILL_EVAL_WORKER_COUNT", "1"))
    worker_index = int(os.environ.get("SKILL_EVAL_WORKER_INDEX", "0"))
    device = get_safe_torch_device(cfg.policy.device, log=True)
    set_seed(cfg.seed)

    dataset = SkillEvaluationDataset(
        skill_dataset_dir=os.environ["SKILL_DATASET_DIR"],
        skill_latents_path=os.environ["SKILL_LATENTS_PATH"],
        eval_init_states_path=os.environ["EVAL_INIT_STATES_PATH"],
        original_dataset_dir=os.environ["ORIGINAL_DATASET_DIR"],
        suite_name=cfg.env.task,
        raw_dataset_dir=specs[0].get("raw_dataset_dir"),
    )
    if dataset.uses_langgap_replay and not episode_exact:
        raise ValueError(
            "LangGap skill-noise evaluation requires episode_exact=true because "
            "middle-skill states are reconstructed by episode action replay."
        )
    selected = dataset.select_episodes(
        task_ids=json.loads(
            os.environ.get("DATASET_TASK_IDS", json.dumps(list(cfg.env.task_ids or [])))
        ),
        episodes_per_task=int(os.environ["ENVS_PER_TASK"]),
        selection=os.environ["EPISODE_SELECTION"],
        seed=cfg.seed,
        explicit_episode_ids=json.loads(os.environ.get("EPISODE_IDS", "[]")),
    )
    datasets = [
        SkillEvaluationDataset(
            skill_dataset_dir=os.environ["SKILL_DATASET_DIR"],
            skill_latents_path=spec["skill_latents_path"],
            eval_init_states_path=os.environ["EVAL_INIT_STATES_PATH"],
            original_dataset_dir=os.environ["ORIGINAL_DATASET_DIR"],
            suite_name=cfg.env.task,
            raw_dataset_dir=spec.get("raw_dataset_dir"),
        )
        for spec in specs
    ]
    occurrences_by_model = [item.occurrences(selected) for item in datasets]
    if not occurrences_by_model[0]:
        raise RuntimeError("No skill occurrences were found in selected environments.")
    reference_ids = [item.identity_uid for item in occurrences_by_model[0]]
    for model_index, occurrences in enumerate(occurrences_by_model[1:], start=1):
        candidate_ids = [item.identity_uid for item in occurrences]
        if candidate_ids != reference_ids:
            raise ValueError(
                "Models may use different codes but must share GT segmentation; "
                f"model={specs[model_index]['label']!r}."
            )

    for episode_ids in selected.values():
        for episode_id in episode_ids:
            dataset.load_aligned_episode(episode_id)
            if episode_exact:
                dataset.exact_model_xml(episode_id)

    assigned_by_model = _worker_units(
        model_count=len(specs),
        selected=selected,
        noise_rollouts=noise_rollouts,
        worker_index=worker_index,
        worker_count=worker_count,
    )
    log.info(
        "noise worker=%d/%d assigned_units=%d",
        worker_index,
        worker_count,
        sum(len(units) for units in assigned_by_model.values()),
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
    signature = _signature(
        specs,
        cfg,
        selected,
        noise_rollouts,
        trajectory_stride,
        code_probe_mode,
        skill_only_rollout_probe,
    )
    resume = _as_bool(os.environ.get("EVAL_RESUME", "false"))
    if worker_count > 1 and consolidated_path.is_file() and not resume:
        raise FileExistsError(
            f"Output already contains {consolidated_path}; enable resume or change output_name."
        )
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if not resume:
            raise FileExistsError(
                f"Output already contains {manifest_path}; enable resume or change output_name."
            )
        if manifest.get("signature") != signature:
            raise ValueError("resume=true but the existing noise-eval signature differs.")
    else:
        manifest = {
            "signature": signature,
            "chunk_index": worker_index,
            "chunk_count": worker_count,
            "model_levels": [
                [int(value) for value in spec["fsq_levels"]] for spec in specs
            ],
            "completed": False,
            "records": {},
        }
        _save_manifest(manifest_path, manifest)

    try:
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=cfg.env,
            policy_cfg=cfg.policy,
        )
        for model_index, spec in enumerate(specs):
            units = assigned_by_model.get(model_index, [])
            if not units:
                continue
            occurrences_for_episode: dict[int, list] = {}
            for occurrence in occurrences_by_model[model_index]:
                occurrences_for_episode.setdefault(occurrence.episode_id, []).append(
                    occurrence
                )
            context = _build_context(spec, cfg, device)
            context["display_terminators"] = []
            try:
                runtime_levels = [
                    int(value) for value in context["config"].skill_fsq_levels
                ]
                if runtime_levels != [int(value) for value in spec["fsq_levels"]]:
                    raise ValueError(
                        f"{spec['label']} runtime levels {runtime_levels} differ "
                        f"from checkpoint levels {spec['fsq_levels']}."
                    )
                if skill_only_rollout_probe:
                    runtime_config = context["config"]
                    if not bool(getattr(runtime_config, "skill_flow_enabled", False)):
                        raise ValueError(
                            "skill_only_rollout_probe=true requires every selected "
                            "checkpoint to use arch0_skill, arch0_skill_chunk, or "
                            "arch0_2_skill_chunk; "
                            f"{spec['label']!r} has no trained skill-flow path."
                        )
                    if not callable(
                        getattr(
                            context["policy"].policy,
                            "predict_skill_only_action_chunk",
                            None,
                        )
                    ):
                        raise RuntimeError(
                            f"{spec['label']!r} policy does not expose skill-only inference."
                        )
                main_rule = spec.get("main_terminator", {})
                end_mode = str(
                    main_rule.get("end_mode", os.environ["SKILL_END_MODE"])
                )
                end_threshold = float(
                    main_rule.get(
                        "end_threshold", os.environ["SKILL_END_THRESHOLD"]
                    )
                )
                progress_threshold = float(
                    main_rule.get(
                        "progress_threshold",
                        os.environ["SKILL_END_PROGRESS_THRESHOLD"],
                    )
                )
                finish_chunk = bool(
                    main_rule.get(
                        "finish_action_chunk_on_end",
                        _as_bool(os.environ["FINISH_ACTION_CHUNK_ON_END"]),
                    )
                )
                inference_context = (
                    torch.autocast(device_type=device.type)
                    if context["config"].use_amp
                    else nullcontext()
                )
                with torch.inference_mode(), inference_context:
                    for unit_index, (episode_id, noise_index) in enumerate(units):
                        aligned = datasets[model_index].load_aligned_episode(
                            episode_id
                        )
                        model_xml = (
                            dataset.exact_model_xml(episode_id)
                            if episode_exact
                            else None
                        )
                        occurrences = occurrences_for_episode.get(episode_id, [])
                        if not occurrences:
                            raise RuntimeError(
                                f"No skills for assigned episode {episode_id}."
                            )
                        for occurrence in occurrences:
                            rollout_tokens = _evaluated_tokens(
                                occurrence.token,
                                runtime_levels,
                                probe_mode=code_probe_mode,
                            )
                            rollout_token_roles = _evaluated_token_roles(
                                occurrence.token,
                                runtime_levels,
                                probe_mode=code_probe_mode,
                            )
                            record_uid = (
                                f"model_{model_index:02d}__{occurrence.identity_uid}"
                            )
                            task_description = datasets[model_index].episode_task_description(
                                occurrence.episode_id
                            )
                            start_image_relative = (
                                Path("assets")
                                / "start"
                                / f"task_{occurrence.task_id:02d}"
                                / f"{occurrence.identity_uid}.jpg"
                            )
                            record = manifest["records"].setdefault(
                                record_uid,
                                {
                                    "uid": record_uid,
                                    "occurrence_uid": occurrence.identity_uid,
                                    "model_index": model_index,
                                    "model_label": str(spec["label"]),
                                    "token": int(occurrence.token),
                                    "evaluated_tokens": rollout_tokens,
                                    "evaluated_token_roles": {
                                        str(token): rollout_token_roles[token]
                                        for token in rollout_tokens
                                    },
                                    "evaluated_coords": {
                                        str(token): _token_to_coord(
                                            token, runtime_levels
                                        )
                                        for token in rollout_tokens
                                    },
                                    "task_id": int(occurrence.task_id),
                                    "task_description": task_description,
                                    "episode_id": int(occurrence.episode_id),
                                    "skill_index": int(occurrence.skill_index),
                                    "frame_start": int(occurrence.frame_start),
                                    "frame_end": int(occurrence.frame_end),
                                    "length": int(occurrence.length),
                                    "scene_file": aligned.source.scene_file,
                                    "demo": aligned.source.demo,
                                    "start_image_path": start_image_relative.as_posix(),
                                    "rollouts": [],
                                },
                            )
                            expected_roles = {
                                str(token): rollout_token_roles[token]
                                for token in rollout_tokens
                            }
                            previous_roles = record.get("evaluated_token_roles")
                            if previous_roles is None:
                                record["evaluated_token_roles"] = expected_roles
                            elif previous_roles != expected_roles:
                                raise ValueError(
                                    "Existing record uses different evaluated-token "
                                    f"roles: {record_uid}."
                                )
                            existing = {
                                (
                                    str(item.get("rollout_path", "main")),
                                    int(
                                        item.get(
                                            "eval_token", occurrence.token
                                        )
                                    ),
                                    int(item["noise_index"]),
                                )
                                for item in record.get("rollouts", [])
                            }
                            rollout_paths = ["main"]
                            if skill_only_rollout_probe:
                                rollout_paths.append("skill_only")
                            missing_routes = [
                                (rollout_path, token)
                                for rollout_path in rollout_paths
                                for token in rollout_tokens
                                if (
                                    rollout_path,
                                    int(token),
                                    int(noise_index),
                                )
                                not in existing
                            ]
                            if not missing_routes:
                                continue
                            if main_rule.get("max_skill_length_scale") is not None:
                                max_length = _rollout_max_skill_length(
                                    gt_length=occurrence.length,
                                    mode="gt_scale",
                                    fixed_length=1,
                                    scale=float(
                                        main_rule["max_skill_length_scale"]
                                    ),
                                )
                            else:
                                max_length = _rollout_max_skill_length(
                                    gt_length=occurrence.length,
                                    mode="fixed",
                                    fixed_length=int(
                                        main_rule.get(
                                            "max_skill_length",
                                            os.environ[
                                                "INFERENCE_SKILL_MAX_LENGTH"
                                            ],
                                        )
                                    ),
                                    scale=0.0,
                                )
                            (
                                state,
                                replay_actions,
                                exact_init_state_index,
                            ) = aligned.restoration_at(occurrence.frame_start)
                            previous_action = (
                                None
                                if occurrence.frame_start == 0
                                else np.asarray(
                                    aligned.filtered_actions[
                                        occurrence.frame_start - 1
                                    ],
                                    dtype=np.float32,
                                ).copy()
                            )
                            noise_seed = (
                                int(cfg.seed)
                                + int(episode_id) * 1_000_003
                                + int(occurrence.skill_index) * 10_007
                                + int(noise_index) * 97
                            ) % (2**31 - 1)
                            layout_seed = (
                                None
                                if episode_exact
                                else _environment_layout_seed(
                                    base_seed=int(cfg.seed),
                                    task_id=occurrence.task_id,
                                    episode_id=episode_id,
                                )
                            )
                            vec_env = envs[cfg.env.task][aligned.source.env_task_id]
                            base_env = vec_env.envs[0].unwrapped
                            start_image_path = output_dir / start_image_relative
                            for rollout_path, eval_token in missing_routes:
                                # Reuse one base seed across tested codes and
                                # paths. The runner salts only arch0_skill's
                                # canonical auxiliary, whose noise was independent
                                # of the main route during training; chunk routes
                                # preserve their trained shared-noise prefix.
                                result = _run_noise_policy(
                                    base_env=base_env,
                                    state=state,
                                    token=eval_token,
                                    context=context,
                                    env_preprocessor=env_preprocessor,
                                    env_postprocessor=env_postprocessor,
                                    max_skill_length=max_length,
                                    gt_skill_length=occurrence.length,
                                    n_action_steps=int(cfg.policy.n_action_steps),
                                    end_mode=end_mode,
                                    end_threshold=end_threshold,
                                    progress_threshold=progress_threshold,
                                    finish_action_chunk_on_end=finish_chunk,
                                    seed=noise_seed,
                                    initial_previous_action=previous_action,
                                    capture_start_image=not (
                                        start_image_path.is_file()
                                        and start_image_path.stat().st_size > 0
                                    ),
                                    episode_start_xyz=aligned.episode_start_xyz,
                                    model_xml=model_xml,
                                    layout_seed=layout_seed,
                                    exact_init_state_index=exact_init_state_index,
                                    replay_actions=replay_actions,
                                    task_description=task_description,
                                    rollout_path=rollout_path,
                                )
                                if result["start_image"] is not None:
                                    _save_start_image(
                                        start_image_path,
                                        result["start_image"],
                                    )
                                height = int(result["image_height"])
                                width = int(result["image_width"])
                                trajectory = _project_trajectory(
                                    result["trajectory_world"],
                                    result["camera_transform"],
                                    height=height,
                                    width=width,
                                    stride=trajectory_stride,
                                )
                                record["rollouts"].append(
                                    {
                                        "rollout_path": rollout_path,
                                        "eval_token": int(eval_token),
                                        "eval_coord": _token_to_coord(
                                            eval_token, runtime_levels
                                        ),
                                        "probe_role": rollout_token_roles[
                                            int(eval_token)
                                        ],
                                        "is_original_code": bool(
                                            int(eval_token)
                                            == int(occurrence.token)
                                        ),
                                        "noise_index": int(noise_index),
                                        "seed": int(result["sampling_seed"]),
                                        "paired_base_seed": int(noise_seed),
                                        "layout_seed": layout_seed,
                                        "environment_mode": (
                                            "langgap_episode_replay"
                                            if aligned.requires_episode_replay
                                            else "source_demo_xml"
                                            if episode_exact
                                            else "seeded_random_layout"
                                        ),
                                        "trajectory": trajectory,
                                        "steps": int(result["steps"]),
                                        "max_skill_length": int(max_length),
                                        "stop_reason": str(result["stop_reason"]),
                                        "final_progress": result[
                                            "final_progress"
                                        ],
                                        "final_termination": result[
                                            "final_termination"
                                        ],
                                        "task_success_seen": bool(
                                            result["task_success_seen"]
                                        ),
                                        "task_success_step": result[
                                            "task_success_step"
                                        ],
                                        "environment_done_step": result[
                                            "environment_done_step"
                                        ],
                                    }
                                )
                            record["rollouts"].sort(
                                key=lambda item: (
                                    0
                                    if str(item.get("rollout_path", "main"))
                                    == "main"
                                    else 1,
                                    0
                                    if int(
                                        item.get(
                                            "eval_token", occurrence.token
                                        )
                                    )
                                    == int(occurrence.token)
                                    else 1,
                                    int(
                                        item.get(
                                            "eval_token", occurrence.token
                                        )
                                    ),
                                    int(item["noise_index"]),
                                )
                            )
                        _save_manifest(manifest_path, manifest)
                        log.info(
                            "[%s] unit %d/%d episode=%d noise=%d skills=%d code_probe=%s",
                            spec["label"],
                            unit_index + 1,
                            len(units),
                            episode_id,
                            noise_index,
                            len(occurrences),
                            (
                                f"{code_probe_mode}+skill_only"
                                if skill_only_rollout_probe
                                else code_probe_mode
                            ),
                        )
            finally:
                del context
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        manifest["completed"] = True
        _save_manifest(manifest_path, manifest)
        if worker_count == 1:
            report = write_noise_html_reports(output_dir, manifest)
            print(f"Saved noise trajectory report: {report}")
        else:
            report = maybe_merge_noise_chunks(
                output_dir,
                expected_chunks=worker_count,
            )
            if report is None:
                print(
                    f"Saved noise worker chunk {worker_index + 1}/{worker_count}; "
                    "the last worker will write HTML."
                )
            else:
                print(f"All chunks complete; saved noise trajectory report: {report}")
    finally:
        close_envs(envs)


if __name__ == "__main__":
    eval_main()
