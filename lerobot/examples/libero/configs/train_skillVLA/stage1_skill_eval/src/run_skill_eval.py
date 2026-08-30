#!/usr/bin/env python
"""Episode-exact skill rollout with one MAIN and multiple display terminators."""

import gc
import json
import logging
import math
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
from lerobot.policies.skill_expert.modeling_skill_expert import (
    _load_complete_terminator_parameters,
)
from lerobot.policies.skill_expert.modeling_utils import (
    build_fsq_image_only_terminator,
    build_fsq_terminator,
    build_trainable_fsq_terminator,
    build_fsq_wrist_only_terminator,
)
from lerobot.scripts.lerobot_skillvla_eval import _libero_task_descriptions
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.random_utils import set_seed

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / "stage1_eval" / "src"))

from html_report import write_html_report  # noqa: E402
from merge_results import maybe_merge_chunks, report_payload  # noqa: E402
from run_eval import (  # noqa: E402
    RAW_IMAGE,
    RAW_STATE,
    RAW_WRIST,
    _build_context as _build_stage1_context,
    _ensure_skill_runtime_steps,
)
from skill_data import SkillEvaluationDataset, SkillOccurrence  # noqa: E402

log = logging.getLogger(__name__)

_INLINE_CUDA_GUARD_EXIT_CODE = 86

BRANCHES = (
    ("gt", "GT actions", "#2e7d32"),
    ("policy", "Policy · exact start", "#1565c0"),
    ("policy_alt_noise", "Policy · exact start · different noise", "#00838f"),
    ("policy_early", "Policy · early start", "#ef6c00"),
    ("policy_late", "Policy · late start", "#8e24aa"),
)
ALT_NOISE_SEED_OFFSET = 1_000_003


def _run_inline_cuda_guard() -> None:
    """Validate CUDA in this evaluator process, avoiding a second torch import."""
    if os.environ.get("LEROBOT_INLINE_CUDA_GUARD", "0") != "1":
        return
    if torch.cuda.is_available():
        return
    marker = os.environ.get("LEROBOT_CUDA_GUARD_FAILURE_MARKER", "")
    if marker:
        Path(marker).write_text("torch.cuda.is_available()=false\n", encoding="utf-8")
    print(
        "GPU GUARD: torch.cuda.is_available() is false; refusing CPU fallback.",
        flush=True,
    )
    raise SystemExit(_INLINE_CUDA_GUARD_EXIT_CODE)


def _mark_startup_ready() -> None:
    marker = os.environ.get("LEROBOT_STARTUP_READY_MARKER", "")
    if marker:
        Path(marker).touch()


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class IndependentTerminator:
    """Run one display-only terminator without occupying a policy module slot."""

    def __init__(self, policy, module, variant: str):
        self.policy_model = policy.model
        self.module = module
        self.variant = variant
        self.termination_only = bool(getattr(module, "termination_only", False))
        self.context_mode = str(getattr(module, "context_mode", "proprio"))

    def reset(self) -> None:
        """Keep one reset interface for independent rollout boundaries."""

    @torch.no_grad()
    def terminate(
        self,
        codes: torch.Tensor,
        state: torch.Tensor | None,
        image: torch.Tensor,
        wrist_image: torch.Tensor,
        previous_action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.module.parameters()).device
        dtype = next(self.module.parameters()).dtype
        z_q = self.policy_model._code_to_zq(  # noqa: SLF001
            codes.to(self.policy_model._fsq_strides.device)  # noqa: SLF001
        ).to(device=device, dtype=dtype)
        if self.variant in {"state_image", "fsq_initial"}:
            if self.context_mode == "prev_action":
                if previous_action is None:
                    context = torch.zeros(
                        codes.shape[0],
                        int(self.module.state_dim),
                        device=device,
                        dtype=dtype,
                    )
                else:
                    context = self.module.normalize_previous_action(
                        previous_action.to(device=device, dtype=dtype)
                    )
            else:
                if state is None:
                    raise ValueError(f"{self.variant} terminator requires robot state.")
                context = state.to(device=device, dtype=dtype)
            progress, logits = self.module(
                z_q,
                context,
                image.to(device=device, dtype=dtype),
                wrist_image.to(device=device, dtype=dtype),
            )
        elif self.variant == "image_only":
            progress, logits = self.module(
                z_q,
                image.to(device=device, dtype=dtype),
                wrist_image.to(device=device, dtype=dtype),
            )
        elif self.variant == "wrist_only":
            progress, logits = self.module(
                z_q,
                wrist_image.to(device=device, dtype=dtype),
            )
        else:
            raise ValueError(f"Unknown display terminator variant: {self.variant!r}.")
        return progress, torch.sigmoid(logits)


def _checkpoint_termination_only(checkpoint_path: str, field: str) -> bool:
    if not checkpoint_path:
        return False
    config_path = Path(checkpoint_path) / "config.json"
    if not config_path.is_file():
        return False
    return bool(json.loads(config_path.read_text()).get(field, False))


def _checkpoint_config(checkpoint_path: str) -> dict:
    config_path = Path(checkpoint_path) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Terminator config not found: {config_path}")
    return json.loads(config_path.read_text())


def _load_display_terminator(policy, model_spec: dict, fsq_path: str | Path):
    variant = str(model_spec["variant"])
    checkpoint_path = str(model_spec.get("path") or "")
    source_config = None
    if variant != "fsq_initial":
        source_config = _checkpoint_config(checkpoint_path)
    if variant == "fsq_initial":
        if not checkpoint_path:
            raise ValueError("fsq_initial requires its resolved raw FSQ.pt path.")
        module = build_fsq_terminator(checkpoint_path)
        prefix = "model.fsq_term_train."
    elif variant == "state_image":
        assert source_config is not None
        module = build_trainable_fsq_terminator(
            fsq_path,
            termination_only=_checkpoint_termination_only(
                checkpoint_path, "terminator_termination_only"
            ),
            context=source_config.get("terminator_context"),
            default_arch=source_config.get("terminator_arch"),
            vision_backbone=source_config.get("terminator_vision_backbone"),
            freeze_vision_encoder=source_config.get(
                "terminator_freeze_vision_encoder"
            ),
        )
        prefix = "model.fsq_term_train."
    elif variant == "image_only":
        module = build_fsq_image_only_terminator(
            fsq_path,
            termination_only=_checkpoint_termination_only(
                checkpoint_path, "image_only_terminator_termination_only"
            ),
        )
        prefix = "model.fsq_image_term_train."
    elif variant == "wrist_only":
        module = build_fsq_wrist_only_terminator(
            fsq_path,
            termination_only=_checkpoint_termination_only(
                checkpoint_path, "wrist_only_terminator_termination_only"
            ),
        )
        prefix = "model.fsq_wrist_term_train."
    else:
        raise ValueError(f"Unknown display terminator variant: {variant!r}.")

    if variant != "fsq_initial":
        _load_complete_terminator_parameters(
            module,
            checkpoint_path,
            prefix=prefix,
            label=f"{variant} terminator",
        )
    device = next(policy.model.parameters()).device
    module.to(device=device, dtype=torch.float32)
    module.requires_grad_(False).eval()
    return IndependentTerminator(policy, module, variant)


def _display_reuses_main(spec: dict, display_model: dict) -> bool:
    """Whether MAIN and display can share one terminator forward pass."""
    main_path = str(spec.get("external_skill_model") or "").strip()
    display_path = str(display_model.get("path") or "").strip()
    return bool(
        main_path
        and display_path
        and str(spec.get("external_skill_model_variant", ""))
        == str(display_model.get("variant", ""))
        and Path(main_path).resolve() == Path(display_path).resolve()
    )


def _build_context(spec: dict, cfg, device: torch.device) -> dict:
    """Build evaluation context, including the raw-FSQ MAIN special case."""
    use_fsq_initial_main = (
        str(spec.get("advance_mode", "")) in {"external", "original"}
        and str(spec.get("external_skill_model_variant", "checkpoint"))
        == "fsq_initial"
    )
    if not use_fsq_initial_main:
        return _build_stage1_context(spec, cfg, device)

    # The shared Stage-1 loader treats every external source as a trained
    # pretrained_model directory. Build the action policy without an overlay,
    # then attach the pristine terminator reconstructed directly from FSQ.pt.
    base_spec = dict(spec)
    base_spec["advance_mode"] = "gt"
    # The shared loader validates this field before it examines advance_mode.
    # It is unused in GT mode, but must still carry one of its public variants.
    base_spec["terminator_variant"] = "state_image"
    context = _build_stage1_context(base_spec, cfg, device)
    _ensure_skill_runtime_steps(
        context["preprocessor"],
        context["config"],
        needs_predictor=False,
        needs_terminator=True,
    )
    wrapper = context["policy"]
    action_policy = wrapper.policy
    if hasattr(action_policy.model, "fsq_term_train"):
        # The action checkpoint's co-trained copy is not the requested baseline
        # and can be released before constructing the pristine FSQ module.
        action_policy.model.fsq_term_train = None
    module = build_fsq_terminator(spec["external_skill_model"])
    module.to(
        device=next(action_policy.model.parameters()).device,
        dtype=torch.float32,
    )
    module.requires_grad_(False).eval()
    wrapper.terminator = IndependentTerminator(
        action_policy,
        module,
        "fsq_initial",
    )
    wrapper.advance_mode = "external"
    log.info(
        "[%s] attached pristine FSQ terminator as MAIN from %s.",
        spec["label"],
        spec["external_skill_model"],
    )
    return context


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
    # state. The reset-created controller still caches the episode-initial EE
    # pose after set_init_state(), so force it to observe the restored pose and
    # reset its goal before the first recorded or predicted action. Otherwise
    # the first OSC torque pulls the arm toward that stale pose. No settling or
    # no-op step is used because that would alter the requested skill start.
    base_env._env.reset()
    raw_obs = base_env._env.set_init_state(np.asarray(state, dtype=np.float64))
    for robot in base_env._env.robots:
        controller = robot.controller
        controller.use_delta = True
        controller.update(force=True)
        controller.reset_goal()
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
    previous_action: np.ndarray | torch.Tensor | None = None,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    float,
    float,
    list[tuple[float, float]],
]:
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
    previous_action_tensor = None
    if previous_action is not None:
        previous_action_tensor = torch.as_tensor(
            previous_action,
            dtype=torch.float32,
            device=device,
        )
        if previous_action_tensor.ndim == 1:
            previous_action_tensor = previous_action_tensor.unsqueeze(0)
        if previous_action_tensor.ndim != 2:
            raise ValueError(
                "previous_action must have shape (A,) or (B,A), got "
                f"{tuple(previous_action_tensor.shape)}."
            )
    missing = [key for key in (RAW_STATE, RAW_IMAGE, RAW_WRIST) if key not in batch]
    if missing:
        raise ValueError(f"Policy preprocessor omitted terminator inputs: {missing}.")
    current_progress, current_termination = terminator.terminate(
        codes,
        batch[RAW_STATE],
        batch[RAW_IMAGE],
        batch[RAW_WRIST],
        previous_action=previous_action_tensor,
    )
    display_signals = []
    for display_entry in context.get("display_terminators", []):
        if display_entry.get("reuse_main", False):
            display_progress_tensor = current_progress
            display_termination_tensor = current_termination
        else:
            display_progress_tensor, display_termination_tensor = display_entry[
                "terminator"
            ].terminate(
                codes,
                batch[RAW_STATE],
                batch[RAW_IMAGE],
                batch[RAW_WRIST],
                previous_action=previous_action_tensor,
            )
        display_signals.append(
            (
                float(display_progress_tensor[0]),
                float(display_termination_tensor[0]),
            )
        )
    return (
        batch,
        restored_state,
        float(current_progress[0]),
        float(current_termination[0]),
        display_signals,
    )


def _reset_terminators(context: dict) -> None:
    """Reset MAIN/display terminator state before each independent rollout."""
    terminators = [
        getattr(context["policy"], "terminator", None),
        *[
            entry.get("terminator")
            for entry in context.get("display_terminators", [])
            if not entry.get("reuse_main", False)
        ],
    ]
    for terminator in terminators:
        reset = getattr(terminator, "reset", None)
        if callable(reset):
            reset()


def _load_font(size: int):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _signal_row_height(camera_height: int) -> int:
    return max(36, int(camera_height) // 7)


def _apply_display_fired_tint(
    frame: np.ndarray,
    *,
    alpha: float = 0.18,
) -> np.ndarray:
    """Apply a translucent pale-green overlay to a camera frame."""
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    overlay = Image.new("RGB", image.size, (132, 255, 166))
    return np.asarray(Image.blend(image, overlay, float(alpha)), dtype=np.uint8)


def _annotate_frames(
    frames: list[np.ndarray],
    *,
    progress: list[float | None] | None = None,
    termination: list[float | None] | None = None,
    display_traces: list[dict[str, Any]] | None = None,
    progress_threshold: float,
    end_threshold: float,
) -> list[np.ndarray]:
    """Place full-width signal bars below one branch without repeated labels."""
    if not frames:
        return []
    height, width = frames[0].shape[:2]
    row_height = _signal_row_height(height)
    term_width = width // 2
    progress_width = width - term_width

    def metric_panel(
        panel_width: int,
        value: float | None,
        threshold: float,
        below_color: tuple[int, int, int],
        above_color: tuple[int, int, int],
    ) -> np.ndarray:
        panel = Image.new("RGB", (panel_width, row_height), (22, 24, 28))
        draw = ImageDraw.Draw(panel)
        valid = value is not None and np.isfinite(float(value))
        if valid:
            clipped = float(np.clip(float(value), 0.0, 1.0))
        else:
            clipped = 0.0

        track_x0 = 4
        track_x1 = max(track_x0 + 4, panel_width - 4)
        track_top = 5
        track_bottom = row_height - 5
        inner_x0 = track_x0 + 2
        inner_x1 = track_x1 - 2
        if valid and inner_x1 >= inner_x0:
            fill_x1 = min(
                inner_x1,
                inner_x0 + max(1, round((inner_x1 - inner_x0) * clipped)),
            )
            draw.rectangle(
                (inner_x0, track_top + 2, fill_x1, track_bottom - 2),
                fill=above_color if clipped >= threshold else below_color,
            )
        draw.rectangle(
            (track_x0, track_top, track_x1, track_bottom),
            outline=(210, 210, 210),
            width=1,
        )
        threshold_x = track_x0 + round(
            (track_x1 - track_x0) * float(np.clip(threshold, 0.0, 1.0))
        )
        draw.line(
            (threshold_x, track_top - 2, threshold_x, track_bottom + 1),
            fill=(255, 196, 48),
            width=2,
        )
        return np.asarray(panel, dtype=np.uint8)

    def signal_row(
        progress_value: float | None,
        termination_value: float | None,
        *,
        row_progress_threshold: float,
        row_end_threshold: float,
    ) -> np.ndarray:
        return np.concatenate(
            [
                metric_panel(
                    term_width,
                    termination_value,
                    row_end_threshold,
                    (155, 89, 182),
                    (231, 76, 60),
                ),
                metric_panel(
                    progress_width,
                    progress_value,
                    row_progress_threshold,
                    (52, 152, 219),
                    (46, 204, 113),
                ),
            ],
            axis=1,
        )

    annotated = []
    for index, frame in enumerate(frames):
        progress_value = progress[index] if progress and index < len(progress) else None
        termination_value = (
            termination[index] if termination and index < len(termination) else None
        )
        signal_rows = []
        display_has_fired = False
        for trace in display_traces or []:
            trace_progress = trace.get("progress") or []
            trace_termination = trace.get("termination") or []
            trace_fired = trace.get("fired") or []
            display_has_fired |= bool(
                index < len(trace_fired) and trace_fired[index]
            )
            signal_rows.append(
                signal_row(
                    trace_progress[index] if index < len(trace_progress) else None,
                    (
                        trace_termination[index]
                        if index < len(trace_termination)
                        else None
                    ),
                    row_progress_threshold=float(
                        trace.get("progress_threshold", progress_threshold)
                    ),
                    row_end_threshold=float(
                        trace.get("end_threshold", end_threshold)
                    ),
                )
            )
        annotated.append(
            np.concatenate(
                [
                    (
                        _apply_display_fired_tint(frame)
                        if display_has_fired
                        else np.asarray(frame, dtype=np.uint8)
                    ),
                    *signal_rows,
                    signal_row(
                        progress_value,
                        termination_value,
                        row_progress_threshold=progress_threshold,
                        row_end_threshold=end_threshold,
                    ),
                ],
                axis=0,
            )
        )
    return annotated


def _latch_termination_trace(
    values: list[float | None],
    *,
    end_threshold: float,
) -> list[float | None]:
    """Freeze a displayed termination trace at its first threshold crossing."""
    latched_value: float | None = None
    latched_values: list[float | None] = []
    for value in values:
        if latched_value is not None:
            latched_values.append(latched_value)
            continue
        latched_values.append(value)
        if value is not None and np.isfinite(float(value)) and float(value) >= end_threshold:
            latched_value = float(value)
    return latched_values


def _write_branch_video(
    path: Path,
    frames: list[np.ndarray],
    *,
    frame_stride: int,
    fps: int,
    progress: list[float | None] | None = None,
    termination: list[float | None] | None = None,
    display_traces: list[dict[str, Any]] | None = None,
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
    latched_termination = (
        _latch_termination_trace(termination, end_threshold=end_threshold)
        if termination
        else None
    )
    selected_termination = (
        [latched_termination[index] for index in indices]
        if latched_termination
        else None
    )
    selected_display_traces = []
    for trace in display_traces or []:
        selected_display_traces.append(
            {
                "label": trace["label"],
                "end_mode": trace.get("end_mode", "termination"),
                "end_threshold": trace.get("end_threshold", end_threshold),
                "progress_threshold": trace.get(
                    "progress_threshold", progress_threshold
                ),
                "progress": [trace["progress"][index] for index in indices],
                # Policy traces are frozen at the MAIN FSQ firing point by
                # _run_policy, so the final bar is the trained terminator's
                # value at that exact decision rather than its own crossing.
                "termination": [trace["termination"][index] for index in indices],
                "fired": [trace["fired"][index] for index in indices],
            }
        )
    selected = _annotate_frames(
        selected,
        progress=selected_progress,
        termination=selected_termination,
        display_traces=selected_display_traces,
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


def _comparison_label_gutter(
    reference_shape: tuple[int, ...],
    row_labels: list[str],
) -> np.ndarray:
    """Build the single left-side label column shared by all five branches."""
    total_height, panel_width = reference_shape[:2]
    row_count = len(row_labels)
    candidates = [
        camera_height
        for camera_height in range(1, total_height + 1)
        if camera_height + _signal_row_height(camera_height) * row_count
        == total_height
    ]
    if not candidates:
        raise ValueError(
            "Could not infer camera/signal-row heights for comparison labels: "
            f"shape={reference_shape}, rows={row_count}."
        )
    camera_height = max(candidates)
    row_height = _signal_row_height(camera_height)
    gutter_width = max(112, panel_width // 2)
    gutter = Image.new("RGB", (gutter_width, total_height), (16, 19, 24))
    draw = ImageDraw.Draw(gutter)
    draw.line(
        (gutter_width - 1, 0, gutter_width - 1, total_height),
        fill=(76, 82, 94),
        width=1,
    )
    for index, label in enumerate(row_labels):
        top = camera_height + index * row_height
        bottom = top + row_height - 1
        is_main = label == "MAIN"
        draw.rectangle(
            (0, top, gutter_width - 1, bottom),
            fill=(35, 42, 54) if is_main else (22, 24, 28),
            outline=(58, 63, 72),
            width=1,
        )
        font_size = max(8, row_height // 3)
        font = _load_font(font_size)
        while font_size > 8 and draw.textlength(label, font=font) > gutter_width - 12:
            font_size -= 1
            font = _load_font(font_size)
        text_width = draw.textlength(label, font=font)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_height = bbox[3] - bbox[1]
        draw.text(
            ((gutter_width - text_width) / 2, top + (row_height - text_height) / 2 - bbox[1]),
            label,
            fill=(255, 214, 102) if is_main else (225, 229, 236),
            font=font,
        )
    return np.asarray(gutter, dtype=np.uint8)


def _write_comparison_video(
    path: Path,
    *,
    output_dir: Path,
    branches: list[dict],
    row_labels: list[str],
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
    label_gutter = _comparison_label_gutter(reference_shape, row_labels)
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
            writer.append_data(np.hstack([label_gutter, *panels]))
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


def _new_display_traces(context: dict) -> list[dict[str, Any]]:
    return [
        {
            "label": entry["label"],
            "end_mode": entry.get("end_mode", "termination"),
            "end_threshold": float(entry.get("end_threshold", 0.5)),
            "progress_threshold": float(entry.get("progress_threshold", 0.95)),
            "progress": [],
            "termination": [],
            "fired": [],
        }
        for entry in context.get("display_terminators", [])
    ]


def _append_display_signals(
    traces: list[dict[str, Any]],
    signals: list[tuple[float, float]],
) -> None:
    if len(traces) != len(signals):
        raise RuntimeError(
            "Display terminator count changed during rollout: "
            f"traces={len(traces)}, signals={len(signals)}."
        )
    for trace, (progress, termination) in zip(traces, signals, strict=True):
        trace["progress"].append(progress)
        trace["termination"].append(termination)
        fired_now = _terminator_fired(
            mode=str(trace["end_mode"]),
            progress=float(progress),
            termination=float(termination),
            progress_threshold=float(trace["progress_threshold"]),
            end_threshold=float(trace["end_threshold"]),
        )
        trace["fired"].append(
            bool(trace["fired"] and trace["fired"][-1]) or fired_now
        )


def _task_success(base_env) -> bool:
    checker = getattr(base_env._env, "check_success", None)
    return bool(checker()) if callable(checker) else False


def _is_environment_done(*, raw_done: bool, task_success: bool) -> bool:
    """Exclude LIBERO's success-derived ``done`` from rollout termination."""
    return bool(raw_done) and not bool(task_success)


def _run_gt_actions(
    *,
    base_env,
    state: np.ndarray,
    actions: np.ndarray,
    token: int,
    context: dict,
    env_preprocessor,
    initial_previous_action: np.ndarray | None = None,
) -> dict:
    _reset_terminators(context)
    raw_obs = _restore_state(base_env, state)
    frames = [_render(base_env)]
    progress_values: list[float | None] = []
    termination_values: list[float | None] = []
    display_traces = _new_display_traces(context)
    stop_reason = "gt_frame_end"
    steps = 0
    environment_done_step: int | None = None
    task_success_seen = _task_success(base_env)
    task_success_step = 0 if task_success_seen else None
    previous_action = (
        None
        if initial_previous_action is None
        else np.asarray(initial_previous_action, dtype=np.float32).copy()
    )
    for action in np.asarray(actions, dtype=np.float32):
        batch, _, progress, termination, display_signals = _query_terminator(
            base_env=base_env,
            raw_obs=raw_obs,
            token=token,
            context=context,
            env_preprocessor=env_preprocessor,
            previous_action=previous_action,
        )
        progress_values.append(progress)
        termination_values.append(termination)
        _append_display_signals(display_traces, display_signals)
        raw_obs, _, done, _ = base_env._env.step(action)
        previous_action = np.asarray(action, dtype=np.float32).copy()
        steps += 1
        frames.append(_render(base_env))
        task_success_now = _task_success(base_env)
        if not task_success_seen and task_success_now:
            task_success_seen = True
            task_success_step = steps
        if _is_environment_done(raw_done=done, task_success=task_success_now):
            environment_done_step = steps
            stop_reason = "environment_done"
            break
    # Also annotate the state reached by the final GT action.
    _, _, progress, termination, display_signals = _query_terminator(
        base_env=base_env,
        raw_obs=raw_obs,
        token=token,
        context=context,
        env_preprocessor=env_preprocessor,
        previous_action=previous_action,
    )
    progress_values.append(progress)
    termination_values.append(termination)
    _append_display_signals(display_traces, display_signals)
    return {
        "frames": frames,
        "steps": steps,
        "stop_reason": stop_reason,
        "progress": progress_values,
        "termination": termination_values,
        "display_traces": display_traces,
        "task_success_seen": task_success_seen,
        "task_success_step": task_success_step,
        "environment_done_step": environment_done_step,
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
    initial_previous_action: np.ndarray | None = None,
) -> dict:
    set_seed(int(seed))
    policy = context["policy"].policy
    policy.reset()
    _reset_terminators(context)
    action_queue: deque[torch.Tensor] = deque()
    raw_obs = _restore_state(base_env, state)
    frames = [_render(base_env)]
    progress_values: list[float | None] = []
    termination_values: list[float | None] = []
    display_traces = _new_display_traces(context)
    pending_end = False
    stop_reason = "max_skill_length"
    steps = 0
    environment_done_step: int | None = None
    task_success_seen = _task_success(base_env)
    task_success_step = 0 if task_success_seen else None
    restored_state_rms = None
    main_boundary: dict[str, Any] | None = None
    boundary_display_signals: list[tuple[float, float]] | None = None
    previous_action = (
        None
        if initial_previous_action is None
        else np.asarray(initial_previous_action, dtype=np.float32).copy()
    )

    while steps < int(max_skill_length):
        (
            batch,
            restored_state,
            progress,
            termination,
            display_signals,
        ) = _query_terminator(
            base_env=base_env,
            raw_obs=raw_obs,
            token=token,
            context=context,
            env_preprocessor=env_preprocessor,
            previous_action=previous_action,
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
        if fired and main_boundary is None:
            boundary_display_signals = list(display_signals)
            main_boundary = {
                "step": steps,
                "progress": float(progress),
                "termination": float(termination),
                "display_terminators": [
                    {
                        "label": trace["label"],
                        "progress": float(display_progress),
                        "termination": float(display_termination),
                        "end_mode": trace["end_mode"],
                        "end_threshold": float(trace["end_threshold"]),
                        "progress_threshold": float(
                            trace["progress_threshold"]
                        ),
                        "would_fire": _terminator_fired(
                            mode=str(trace["end_mode"]),
                            progress=float(display_progress),
                            termination=float(display_termination),
                            progress_threshold=float(
                                trace["progress_threshold"]
                            ),
                            end_threshold=float(trace["end_threshold"]),
                        ),
                        "fired_by_main_boundary": bool(
                            trace["fired"] and trace["fired"][-1]
                        )
                        or _terminator_fired(
                            mode=str(trace["end_mode"]),
                            progress=float(display_progress),
                            termination=float(display_termination),
                            progress_threshold=float(
                                trace["progress_threshold"]
                            ),
                            end_threshold=float(trace["end_threshold"]),
                        ),
                    }
                    for trace, (display_progress, display_termination) in zip(
                        display_traces,
                        boundary_display_signals,
                        strict=True,
                    )
                ],
            }
        _append_display_signals(
            display_traces,
            boundary_display_signals or display_signals,
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
        previous_action = np.asarray(action_numpy, dtype=np.float32).copy()
        steps += 1
        frames.append(_render(base_env))
        # Task success is diagnostic only. As in the raw skill evaluator, it
        # must not stop/reset the rollout before the learned terminator sees the
        # resulting observations.
        task_success_now = _task_success(base_env)
        if not task_success_seen and task_success_now:
            task_success_seen = True
            task_success_step = steps
        if _is_environment_done(raw_done=done, task_success=task_success_now):
            environment_done_step = steps
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
    for trace in display_traces:
        while len(trace["progress"]) < len(frames):
            trace["progress"].append(trace["progress"][-1])
            trace["termination"].append(trace["termination"][-1])
            trace["fired"].append(trace["fired"][-1])
    return {
        "frames": frames,
        "steps": steps,
        "stop_reason": stop_reason,
        "progress": progress_values,
        "termination": termination_values,
        "display_traces": display_traces,
        "main_boundary": main_boundary,
        "restored_state_rms": restored_state_rms,
        "task_success_seen": task_success_seen,
        "task_success_step": task_success_step,
        "environment_done_step": environment_done_step,
    }


def _branch_start_frame(
    occurrence: SkillOccurrence,
    branch: str,
    offset: int,
) -> tuple[int | None, str | None]:
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


def _rollout_max_skill_length(
    *,
    gt_length: int,
    mode: str,
    fixed_length: int,
    scale: float,
) -> int:
    if int(gt_length) <= 0:
        raise ValueError(f"GT skill length must be positive, got {gt_length}.")
    if mode == "gt_scale":
        if float(scale) < 1.0:
            raise ValueError(f"GT max-length scale must be >= 1.0, got {scale}.")
        return int(math.ceil(int(gt_length) * float(scale)))
    if mode == "fixed":
        if int(fixed_length) <= 0:
            raise ValueError(f"Fixed max skill length must be positive, got {fixed_length}.")
        return int(fixed_length)
    raise ValueError(f"Unknown max skill length mode: {mode!r}.")


def _branch_max_skill_length(
    *,
    base_max_skill_length: int,
    branch: str,
    time_shift_offset: int,
) -> int:
    """Give an early-start rollout enough budget to reach the GT start."""
    if int(base_max_skill_length) <= 0:
        raise ValueError(
            f"Base max skill length must be positive, got {base_max_skill_length}."
        )
    if int(time_shift_offset) < 0:
        raise ValueError(
            f"Time-shift offset must be non-negative, got {time_shift_offset}."
        )
    return int(base_max_skill_length) + (
        int(time_shift_offset) if branch == "policy_early" else 0
    )


def _manifest_signature(specs: list[dict], cfg, selected: dict[int, list[int]]) -> dict:
    terminator_models = []
    main_terminators = []
    for model_index, spec in enumerate(specs):
        for model in spec.get("terminator_models", []):
            terminator_models.append({"model_index": model_index, **model})
        main_spec = spec.get("main_terminator", {})
        main_terminators.append(
            {
                "model_index": model_index,
                "label": str(
                    main_spec.get("label", os.environ["MAIN_TERMINATOR_LABEL"])
                ),
                "variant": str(spec["external_skill_model_variant"]),
                "path": str(spec["external_skill_model"]),
            }
        )
    return {
        "format": "stage1_skill_eval_v16_ignore_success_done",
        "policies": [
            {
                "label": str(spec["label"]),
                "policy_path": str(spec["policy_path"]),
                "architecture_label": str(spec.get("architecture_label", "")),
                "fsq_path": str(spec["fsq_path"]),
                "skill_latents_path": str(spec["skill_latents_path"]),
                "fsq_levels": [int(value) for value in spec["fsq_levels"]],
            }
            for spec in specs
        ],
        "main_terminator": {
            "label": os.environ["MAIN_TERMINATOR_LABEL"],
            "variant": str(specs[0]["external_skill_model_variant"]),
            "path": str(specs[0]["external_skill_model"]),
            "end_mode": os.environ["SKILL_END_MODE"],
            "end_threshold": float(os.environ["SKILL_END_THRESHOLD"]),
            "progress_threshold": float(
                os.environ["SKILL_END_PROGRESS_THRESHOLD"]
            ),
            "max_skill_length_mode": os.environ.get(
                "SKILL_MAX_LENGTH_MODE", "fixed"
            ),
            "max_skill_length": (
                int(os.environ["INFERENCE_SKILL_MAX_LENGTH"])
                if os.environ.get("SKILL_MAX_LENGTH_MODE", "fixed") == "fixed"
                else None
            ),
            "max_skill_length_scale": float(
                os.environ.get("SKILL_MAX_LENGTH_SCALE", "0")
            ),
            "finish_action_chunk_on_end": _as_bool(
                os.environ["FINISH_ACTION_CHUNK_ON_END"]
            ),
        },
        "main_terminators": main_terminators,
        "terminator_models": terminator_models,
        "target_task": str(cfg.env.task),
        "selected_episodes": {str(key): value for key, value in selected.items()},
        "time_shift_offset": int(os.environ["TIME_SHIFT_OFFSET"]),
        "n_action_steps": int(cfg.policy.n_action_steps),
        "end_mode": os.environ["SKILL_END_MODE"],
        "end_threshold": float(os.environ["SKILL_END_THRESHOLD"]),
        "progress_threshold": float(os.environ["SKILL_END_PROGRESS_THRESHOLD"]),
        "max_skill_length_mode": os.environ.get("SKILL_MAX_LENGTH_MODE", "fixed"),
        "max_skill_length": (
            int(os.environ["INFERENCE_SKILL_MAX_LENGTH"])
            if os.environ.get("SKILL_MAX_LENGTH_MODE", "fixed") == "fixed"
            else None
        ),
        "max_skill_length_scale": float(
            os.environ.get("SKILL_MAX_LENGTH_SCALE", "0")
        ),
        "finish_action_chunk_on_end": _as_bool(os.environ["FINISH_ACTION_CHUNK_ON_END"]),
        "seed": int(cfg.seed),
    }


def _save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)


def _worker_model_episode_units(
    *,
    model_count: int,
    selected: dict[int, list[int]],
    worker_index: int,
    worker_count: int,
) -> dict[int, tuple[int, ...]]:
    """Return this worker's policy x episode units, grouped by policy."""
    if model_count <= 0:
        raise ValueError("model_count must be positive.")
    if worker_count <= 0 or not 0 <= worker_index < worker_count:
        raise ValueError(
            f"Invalid skill-eval worker index/count: {worker_index}/{worker_count}."
        )

    episode_ids = [
        int(episode_id)
        for task_id in sorted(selected)
        for episode_id in selected[task_id]
    ]
    if not episode_ids:
        raise ValueError("No selected episodes are available for worker assignment.")
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError(f"Selected episode IDs must be unique, got {episode_ids}.")

    all_units = [
        (model_index, episode_id)
        for model_index in range(model_count)
        for episode_id in episode_ids
    ]
    assigned_units = all_units[worker_index::worker_count]
    if not assigned_units:
        raise RuntimeError(
            f"Worker {worker_index}/{worker_count} received no policy x episode work unit; "
            f"total={len(all_units)}. Reduce eval_num_gpus."
        )

    grouped: dict[int, list[int]] = {}
    for model_index, episode_id in assigned_units:
        grouped.setdefault(model_index, []).append(episode_id)
    return {
        model_index: tuple(episode_ids)
        for model_index, episode_ids in grouped.items()
    }


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    _run_inline_cuda_guard()
    _mark_startup_ready()
    specs = json.loads(os.environ.get("MODELS_JSON", "") or "[]")
    if not specs:
        raise ValueError("MODELS_JSON is empty; resolve stage1_skill_eval_config.yaml first.")
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
    datasets = [
        SkillEvaluationDataset(
            # The filtered observations/actions and exact simulator provenance
            # are shared. Only the FSQ assignment file changes by skill space.
            skill_dataset_dir=os.environ["SKILL_DATASET_DIR"],
            skill_latents_path=spec["skill_latents_path"],
            eval_init_states_path=os.environ["EVAL_INIT_STATES_PATH"],
            original_dataset_dir=os.environ["ORIGINAL_DATASET_DIR"],
            suite_name=cfg.env.task,
        )
        for spec in specs
    ]
    occurrences_by_model = [model_dataset.occurrences(selected) for model_dataset in datasets]
    if not occurrences_by_model[0]:
        raise RuntimeError("No skill occurrences were found in the selected exact episodes.")
    reference_ids = [occurrence.identity_uid for occurrence in occurrences_by_model[0]]
    for model_index, occurrences in enumerate(occurrences_by_model[1:], start=1):
        candidate_ids = [occurrence.identity_uid for occurrence in occurrences]
        if candidate_ids != reference_ids:
            missing = sorted(set(reference_ids) - set(candidate_ids))
            extra = sorted(set(candidate_ids) - set(reference_ids))
            raise ValueError(
                "Different FSQ spaces may assign different tokens, but must use the "
                "same GT segmentation for linked evaluation. "
                f"model={specs[model_index]['label']!r}, "
                f"missing={missing[:5]}, extra={extra[:5]}."
            )
    worker_count = int(os.environ.get("SKILL_EVAL_WORKER_COUNT", "1"))
    worker_index = int(os.environ.get("SKILL_EVAL_WORKER_INDEX", "0"))
    assigned_by_model = _worker_model_episode_units(
        model_count=len(specs),
        selected=selected,
        worker_index=worker_index,
        worker_count=worker_count,
    )
    assigned_episode_ids = {
        episode_id
        for episode_ids in assigned_by_model.values()
        for episode_id in episode_ids
    }
    log.info(
        "worker=%d/%d policy_episode_units=%d/%d assignments=%s",
        worker_index,
        worker_count,
        sum(len(episode_ids) for episode_ids in assigned_by_model.values()),
        len(specs) * sum(len(episode_ids) for episode_ids in selected.values()),
        assigned_by_model,
    )
    # Preflight only the source episodes used by this worker before allocating
    # the policy. Other workers independently verify their own exact mappings.
    for episode_id in sorted(assigned_episode_ids):
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
    signature = _manifest_signature(specs, cfg, selected)
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
            "model_label": f"{len(specs)} policies",
            "models": signature["policies"],
            "architecture_label": ",".join(
                str(spec.get("architecture_label", "")) for spec in specs
            ),
            "chunk_index": worker_index,
            "chunk_count": worker_count,
            "levels": [int(value) for value in specs[0]["fsq_levels"]],
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
            model_episode_ids = assigned_by_model.get(model_index)
            if not model_episode_ids:
                continue
            model_episode_id_set = set(model_episode_ids)
            occurrences = [
                occurrence
                for occurrence in occurrences_by_model[model_index]
                if occurrence.episode_id in model_episode_id_set
            ]
            if not occurrences:
                raise RuntimeError(
                    f"Policy {model_index} was assigned episodes {model_episode_ids}, "
                    "but they contain no skill occurrences."
                )
            log.info(
                "Evaluating policy %d/%d: %s; episodes=%s occurrences=%d; "
                "MAIN=%s, display=%s.",
                model_index + 1,
                len(specs),
                spec["label"],
                model_episode_ids,
                len(occurrences),
                spec.get("main_terminator", {}).get(
                    "label", os.environ["MAIN_TERMINATOR_LABEL"]
                ),
                ",".join(
                    str(model.get("label", "terminator"))
                    for model in spec.get("terminator_models", [])
                ),
            )
            context = _build_context(spec, cfg, device)
            try:
                action_policy = context["policy"].policy
                context["display_terminators"] = []
                for display_model in spec.get("terminator_models", []):
                    reuse_main = _display_reuses_main(spec, display_model)
                    context["display_terminators"].append(
                        {
                            "label": str(display_model["label"]),
                            "variant": str(display_model["variant"]),
                            "path": str(display_model.get("path") or ""),
                            "end_mode": str(
                                display_model.get("end_mode", "termination")
                            ),
                            "end_threshold": float(
                                display_model.get("end_threshold", 0.5)
                            ),
                            "progress_threshold": float(
                                display_model.get("progress_threshold", 0.95)
                            ),
                            "reuse_main": reuse_main,
                            "terminator": (
                                None
                                if reuse_main
                                else _load_display_terminator(
                                    action_policy,
                                    display_model,
                                    spec["fsq_path"],
                                )
                            ),
                        }
                    )
                    if reuse_main:
                        log.info(
                            "Display terminator %s reuses the MAIN forward pass "
                            "(variant=%s source=%s).",
                            display_model["label"],
                            display_model["variant"],
                            display_model.get("path"),
                        )
                    else:
                        log.info(
                            "Loaded display-only terminator %s variant=%s source=%s.",
                            display_model["label"],
                            display_model["variant"],
                            display_model.get("path") or spec["fsq_path"],
                        )
                if not context["display_terminators"]:
                    raise RuntimeError("No display terminators were configured.")
                levels = [int(value) for value in context["config"].skill_fsq_levels]
                expected_levels = [int(value) for value in spec["fsq_levels"]]
                if levels != expected_levels:
                    raise ValueError(
                        f"Policy {spec['label']} runtime FSQ levels {levels} do not "
                        f"match its checkpoint contract {expected_levels}."
                    )
                _save_manifest(manifest_path, manifest)
                max_token = int(np.prod(levels))
                invalid_tokens = sorted({occ.token for occ in occurrences if not 0 <= occ.token < max_token})
                if invalid_tokens:
                    raise ValueError(f"Skill tokens outside FSQ{levels}: {invalid_tokens}")

                shift = int(os.environ["TIME_SHIFT_OFFSET"])
                frame_stride = int(os.environ["VIDEO_FRAME_STRIDE"])
                video_fps = int(os.environ["VIDEO_FPS"])
                main_rule = spec.get("main_terminator", {})
                end_mode = str(main_rule.get("end_mode", os.environ["SKILL_END_MODE"]))
                end_threshold = float(
                    main_rule.get("end_threshold", os.environ["SKILL_END_THRESHOLD"])
                )
                progress_threshold = float(
                    main_rule.get(
                        "progress_threshold",
                        os.environ["SKILL_END_PROGRESS_THRESHOLD"],
                    )
                )
                if main_rule.get("max_skill_length_scale") is not None:
                    max_skill_length_mode = "gt_scale"
                    fixed_max_skill_length = 1
                    max_skill_length_scale = float(
                        main_rule["max_skill_length_scale"]
                    )
                else:
                    max_skill_length_mode = "fixed"
                    fixed_max_skill_length = int(
                        main_rule.get(
                            "max_skill_length",
                            os.environ["INFERENCE_SKILL_MAX_LENGTH"],
                        )
                    )
                    max_skill_length_scale = 0.0
                finish_chunk = bool(
                    main_rule.get(
                        "finish_action_chunk_on_end",
                        _as_bool(os.environ["FINISH_ACTION_CHUNK_ON_END"]),
                    )
                )
                task_descriptions = _libero_task_descriptions(cfg.env.task)

                inference_context = (
                    torch.autocast(device_type=device.type)
                    if context["config"].use_amp
                    else nullcontext()
                )
                with torch.inference_mode(), inference_context:
                    for occurrence_index, occurrence in enumerate(occurrences):
                        max_skill_length = _rollout_max_skill_length(
                            gt_length=occurrence.length,
                            mode=max_skill_length_mode,
                            fixed_length=fixed_max_skill_length,
                            scale=max_skill_length_scale,
                        )
                        aligned = dataset.load_aligned_episode(occurrence.episode_id)
                        vec_env = envs[cfg.env.task][occurrence.task_id]
                        base_env = vec_env.envs[0].unwrapped
                        record_uid = f"model_{model_index:02d}__{occurrence.identity_uid}"
                        record = manifest["records"].get(record_uid)
                        if record is None:
                            record = {
                                "uid": record_uid,
                                "occurrence_uid": occurrence.identity_uid,
                                "model_index": model_index,
                                "model_label": spec["label"],
                                "architecture_label": spec.get("architecture_label", ""),
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
                            manifest["records"][record_uid] = record
                        existing_branches = {branch["name"]: branch for branch in record["branches"]}
                        branch_records = []
                        common_seed = (
                            int(cfg.seed)
                            + occurrence.episode_id * 1009
                            + occurrence.skill_index * 17
                        )
                        for branch_name, branch_label, branch_color in BRANCHES:
                            branch_max_skill_length = _branch_max_skill_length(
                                base_max_skill_length=max_skill_length,
                                branch=branch_name,
                                time_shift_offset=shift,
                            )
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
                                        "green_tint": False,
                                    }
                                )
                                continue
                            assert start_frame is not None
                            state = aligned.state_at(start_frame)
                            initial_previous_action = (
                                None
                                if start_frame == 0
                                else np.asarray(
                                    aligned.filtered_actions[start_frame - 1],
                                    dtype=np.float32,
                                ).copy()
                            )
                            offset = start_frame - occurrence.frame_start
                            relative_path = (
                                Path("models")
                                / f"model_{model_index:02d}"
                                / "videos"
                                / f"task_{occurrence.task_id:02d}"
                                / f"token_{occurrence.token:04d}"
                                / occurrence.uid
                                / f"{branch_name}.mp4"
                            )
                            log.info(
                                "[%d/%d] token=%d ep=%d skill=%d branch=%s start=%d "
                                "(offset=%+d max_steps=%d gt_length=%d)",
                                occurrence_index + 1,
                                len(occurrences),
                                occurrence.token,
                                occurrence.episode_id,
                                occurrence.skill_index,
                                branch_name,
                                start_frame,
                                offset,
                                branch_max_skill_length,
                                occurrence.length,
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
                                    initial_previous_action=initial_previous_action,
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
                                    max_skill_length=branch_max_skill_length,
                                    n_action_steps=int(cfg.policy.n_action_steps),
                                    end_mode=end_mode,
                                    end_threshold=end_threshold,
                                    progress_threshold=progress_threshold,
                                    finish_action_chunk_on_end=finish_chunk,
                                    seed=branch_seed,
                                    initial_previous_action=initial_previous_action,
                                )
                            _write_branch_video(
                                output_dir / relative_path,
                                result["frames"],
                                frame_stride=frame_stride,
                                fps=video_fps,
                                progress=result["progress"],
                                termination=result["termination"],
                                display_traces=result.get("display_traces"),
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
                                    "max_skill_length": branch_max_skill_length,
                                    "stop_reason": result["stop_reason"],
                                    "task_success_seen": bool(
                                        result.get("task_success_seen", False)
                                    ),
                                    "task_success_step": result.get(
                                        "task_success_step"
                                    ),
                                    "environment_done_step": result.get(
                                        "environment_done_step"
                                    ),
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
                                    "final_display_terminators": [
                                        {
                                            "label": trace["label"],
                                            "progress": (
                                                None
                                                if not trace["progress"]
                                                else float(trace["progress"][-1])
                                            ),
                                            "termination": (
                                                None
                                                if not trace["termination"]
                                                else float(trace["termination"][-1])
                                            ),
                                        }
                                        for trace in result.get("display_traces", [])
                                    ],
                                    "main_boundary": result.get("main_boundary"),
                                    "green_tint": any(
                                        bool(trace.get("fired"))
                                        and bool(trace["fired"][-1])
                                        for trace in result.get("display_traces", [])
                                    ),
                                    "restored_state_rms": result.get("restored_state_rms"),
                                    "noise_seed": None if branch_name == "gt" else branch_seed,
                                }
                            )
                            record["branches"] = branch_records
                            _save_manifest(manifest_path, manifest)
                        record["branches"] = branch_records
                        comparison_path = (
                            Path("models")
                            / f"model_{model_index:02d}"
                            / "videos"
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
                            row_labels=[
                                str(model["label"])
                                for model in spec.get("terminator_models", [])
                            ]
                            + [
                                str(
                                    spec.get("main_terminator", {}).get(
                                        "label", os.environ["MAIN_TERMINATOR_LABEL"]
                                    )
                                )
                            ],
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

            finally:
                del context
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        manifest["completed"] = True
        _save_manifest(manifest_path, manifest)
        if worker_count == 1:
            report = write_html_report(
                output_dir,
                report_payload(manifest),
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
            f"Worker policy-occurrences: {len(manifest['records'])} / "
            f"total {len(all_occurrences) * len(specs)}"
        )
    finally:
        close_envs(envs)


if __name__ == "__main__":
    eval_main()
