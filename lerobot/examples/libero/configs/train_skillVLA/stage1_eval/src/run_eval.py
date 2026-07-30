#!/usr/bin/env python
"""Multi-checkpoint closed-loop LIBERO evaluation for Stage 1 and Stage 2."""

import gc
import json
import logging
import os
import sys
from collections import deque
from contextlib import nullcontext
from itertools import cycle, islice
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.skill_expert.configuration_skill_expert import SkillExpertConfig
from lerobot.scripts.lerobot_skillvla_eval import (
    _libero_task_descriptions,
    close_envs,
    eval_policy_all,
)
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.random_utils import set_seed

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from eval_oracle import (  # noqa: E402
    load_episode_exact_data,
    load_sequences_by_language,
    map_sequences_to_tasks,
)

RAW_STATE = "skill_decoder_state"
RAW_IMAGE = "skill_decoder_image"
RAW_WRIST = "skill_decoder_wrist"
log = logging.getLogger(__name__)


class CheckpointTerminator:
    """Inference adapter around the terminator stored in a policy checkpoint."""

    use_wrist = True

    def __init__(self, policy):
        if policy.model.fsq_term_train is None:
            raise ValueError("The policy checkpoint has no co-trained terminator.")
        self.model = policy.model

    @torch.no_grad()
    def terminate(
        self,
        codes: torch.Tensor,
        state: torch.Tensor,
        image: torch.Tensor,
        wrist_image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        progress, logits = self.model.terminator_predict(
            codes, state, image, wrist_image
        )
        return progress, torch.sigmoid(logits)


def _parse_sequences(sequences) -> tuple[list[list[int]], list[list[int]]]:
    codes, lengths = [], []
    for sequence in sequences:
        episode_codes, episode_lengths = [], []
        for skill in sequence:
            episode_codes.append(
                int(skill["token"] if isinstance(skill, dict) else skill)
            )
            episode_lengths.append(
                int(skill.get("gt_length", 0)) if isinstance(skill, dict) else 0
            )
        if not episode_codes:
            raise ValueError("Every reference skill sequence must be non-empty.")
        codes.append(episode_codes)
        lengths.append(episode_lengths)
    return codes, lengths


class Stage1OraclePolicy(PreTrainedPolicy):
    """Run a skill-conditioned policy with GT skills or its learned predictor."""

    config_class = SkillExpertConfig
    name = "stage1_oracle"

    def __init__(
        self,
        policy,
        terminator,
        *,
        skill_source: str = "gt",
        advance_mode: str,
        end_mode: str,
        end_threshold: float,
        progress_threshold: float,
        max_skill_length: int,
        n_action_steps: int,
    ):
        super().__init__(policy.config)
        if skill_source not in {"gt", "predictor"}:
            raise ValueError(f"skill_source must be gt|predictor, got {skill_source!r}.")
        if advance_mode not in {"terminator", "gt"}:
            raise ValueError(f"advance_mode must be terminator|gt, got {advance_mode!r}.")
        if skill_source == "predictor" and advance_mode != "terminator":
            raise ValueError("Predictor skills require terminator-based advancement.")
        if advance_mode == "terminator" and terminator is None:
            raise ValueError("advance_mode=terminator requires a checkpoint terminator.")
        if end_mode not in {"termination", "progress", "or", "and"}:
            raise ValueError(
                f"end_mode must be termination|progress|or|and, got {end_mode!r}."
            )
        self.policy = policy
        self.terminator = terminator
        self.skill_source = skill_source
        self.advance_mode = advance_mode
        self.end_mode = end_mode
        self.end_threshold = float(end_threshold)
        self.progress_threshold = float(progress_threshold)
        self.max_skill_length = int(max_skill_length)
        self.n_action_steps = int(n_action_steps)
        self._sequences: list[list[int]] | None = None
        self._gt_lengths: list[list[int]] | None = None
        self._references: list[list[int]] | None = None
        self._reference_lengths: list[list[int]] | None = None
        self._action_queue: deque = deque(maxlen=self.n_action_steps)
        self.reset()

    def set_forced_skill_token_sequences(self, sequences) -> None:
        self._sequences, self._gt_lengths = _parse_sequences(sequences)
        self.reset()

    def set_reference_skill_token_sequences(self, sequences) -> None:
        self._references, self._reference_lengths = _parse_sequences(sequences)
        self.reset()

    def reset(self) -> None:
        if hasattr(self, "policy"):
            self.policy.reset()
        self._action_queue.clear()
        source = self._sequences if self.skill_source == "gt" else self._references
        count = len(source) if source is not None else 0
        self._cursor = [0] * count
        self._skill_step = [0] * count
        self._skill_order = [-1] * count
        self._active_trace = [None] * count
        self._predicted_codes: torch.Tensor | None = None
        self._trace: list[dict] = []
        self._episode_step = 0
        self._started = False

    def _predict_codes(self, batch: dict) -> torch.Tensor:
        return self.policy.predict_skill_code(batch).view(-1).long()

    def _current_codes(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.skill_source == "predictor":
            if self._predicted_codes is None or len(self._predicted_codes) != batch_size:
                raise RuntimeError("Predictor skill codes have not been initialized.")
            return self._predicted_codes.to(device)
        if self._sequences is None or len(self._sequences) != batch_size:
            raise RuntimeError(
                "set_forced_skill_token_sequences must provide one sequence per environment."
            )
        return torch.tensor(
            [
                self._sequences[index][self._cursor[index]]
                for index in range(batch_size)
            ],
            dtype=torch.long,
            device=device,
        )

    def _start_skill(self, batch_index: int, codes: torch.Tensor) -> None:
        self._skill_order[batch_index] += 1
        self._trace.append(
            {
                "batch_index": batch_index,
                "codebook_token": int(codes[batch_index]),
                "skill_index": self._skill_order[batch_index],
                "episode_timestep": self._episode_step,
                "length": 0,
                "end_probs": [],
                "skill_source": self.skill_source,
            }
        )
        self._active_trace[batch_index] = len(self._trace) - 1

    def _terminator_fired(self, progress: float, probability: float) -> bool:
        progress_high = progress >= self.progress_threshold
        termination_high = probability >= self.end_threshold
        if self.end_mode == "progress":
            return progress_high
        if self.end_mode == "termination":
            return termination_high
        if self.end_mode == "and":
            return progress_high and termination_high
        return progress_high or termination_high

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        device = next(self.policy.parameters()).device
        batch_size = int(batch[OBS_STATE].shape[0])
        if not self._started:
            if self.skill_source == "predictor":
                if self._references is None or len(self._references) != batch_size:
                    raise RuntimeError(
                        "Predictor evaluation needs one GT reference sequence per environment."
                    )
                self._predicted_codes = self._predict_codes(batch).to(device)
            codes = self._current_codes(batch_size, device)
            for batch_index in range(batch_size):
                self._start_skill(batch_index, codes)
            self._started = True

        codes = self._current_codes(batch_size, device)
        progress = probability = None
        if self.advance_mode == "terminator":
            missing = [key for key in (RAW_STATE, RAW_IMAGE, RAW_WRIST) if key not in batch]
            if missing:
                raise ValueError(
                    "The saved policy preprocessor must preserve raw terminator inputs; "
                    f"missing={missing}."
                )
            progress, probability = self.terminator.terminate(
                codes, batch[RAW_STATE], batch[RAW_IMAGE], batch[RAW_WRIST]
            )

        gt_advanced = []
        repredict = []
        for batch_index in range(batch_size):
            trace = self._trace[self._active_trace[batch_index]]
            if self.advance_mode == "terminator":
                trace["end_probs"].append(
                    {
                        "skill_step": self._skill_step[batch_index],
                        "prob": float(probability[batch_index]),
                        "progress": float(progress[batch_index]),
                    }
                )
            self._skill_step[batch_index] += 1
            trace["length"] = self._skill_step[batch_index]
            if self.advance_mode == "gt":
                target = self._gt_lengths[batch_index][self._cursor[batch_index]]
                fired = self._skill_step[batch_index] >= max(1, int(target))
            else:
                fired = self._terminator_fired(
                    float(progress[batch_index]), float(probability[batch_index])
                )
                if self.max_skill_length > 0:
                    fired |= self._skill_step[batch_index] >= self.max_skill_length

            if not fired:
                continue
            if self.skill_source == "predictor":
                repredict.append(batch_index)
            else:
                last = len(self._sequences[batch_index]) - 1
                if self._cursor[batch_index] < last:
                    self._cursor[batch_index] += 1
                    gt_advanced.append(batch_index)

        advanced = gt_advanced + repredict
        if repredict:
            new_codes = self._predict_codes(batch).to(device)
            self._predicted_codes[repredict] = new_codes[repredict]
        if advanced:
            codes = self._current_codes(batch_size, device)
            for batch_index in advanced:
                self._skill_step[batch_index] = 0
                self._start_skill(batch_index, codes)
            self._action_queue.clear()

        if not self._action_queue:
            codes = self._current_codes(batch_size, device)
            action_batch = dict(batch)
            action_batch["skill_code"] = codes
            action_batch["skill_sequence"] = codes[:, None]
            action_batch["skill_index"] = torch.zeros(
                batch_size, dtype=torch.long, device=device
            )
            chunk = self.policy.predict_action_chunk(action_batch)
            self._action_queue.extend(
                chunk[:, : self.n_action_steps].transpose(0, 1)
            )
        self._episode_step += 1
        return self._action_queue.popleft()

    def get_skill_trace(self) -> list[dict]:
        return self._trace

    def get_gt_timeline(self) -> dict[int, list[dict]]:
        sequences = self._sequences if self.skill_source == "gt" else self._references
        lengths = self._gt_lengths if self.skill_source == "gt" else self._reference_lengths
        if sequences is None or lengths is None:
            return {}
        return {
            batch_index: [
                {"token": int(code), "length": int(length)}
                for code, length in zip(
                    sequences[batch_index], lengths[batch_index], strict=True
                )
            ]
            for batch_index in range(len(sequences))
        }

    def get_optim_params(self):
        return self.policy.get_optim_params()

    def forward(self, batch, *args, **kwargs):
        return self.policy.forward(batch, *args, **kwargs)

    def predict_action_chunk(self, batch, **kwargs):
        return self.policy.predict_action_chunk(batch, **kwargs)


def _repeat_to_length(sequences: list, count: int) -> list:
    return list(islice(cycle(sequences), count)) if sequences else []


def _language_oracle_maps(
    envs: dict,
    specs: list[dict],
    task_descriptions: dict[int, str],
    n_episodes: int,
) -> list[dict]:
    per_model = [
        map_sequences_to_tasks(
            task_descriptions,
            load_sequences_by_language(spec["skill_dataset_dir"]),
        )
        for spec in specs
    ]
    maps = [dict() for _ in specs]
    for task_group, group in envs.items():
        for task_id in list(group):
            sequences = [model.get(int(task_id), []) for model in per_model]
            if not all(sequences):
                log.warning("task_id=%s is absent from at least one model dataset; dropping it.", task_id)
                group[task_id].close()
                del group[task_id]
                continue
            for index, model_sequences in enumerate(sequences):
                maps[index][(task_group, int(task_id))] = _repeat_to_length(
                    model_sequences, n_episodes
                )
    return maps


def _episode_exact_oracle_maps(
    envs: dict,
    specs: list[dict],
    suite_name: str,
    n_episodes: int,
) -> list[dict]:
    episode_data = [
        load_episode_exact_data(
            spec["skill_dataset_dir"], spec["eval_init_states_path"], suite_name
        )
        for spec in specs
    ]
    maps = [dict() for _ in specs]
    for task_group, group in envs.items():
        for task_id in list(group):
            indexed = [
                {record["episode_index"]: record for record in data.get(int(task_id), [])}
                for data in episode_data
            ]
            common = sorted(set.intersection(*(set(model) for model in indexed))) if all(indexed) else []
            if len(common) < n_episodes:
                log.warning(
                    "task_id=%s has %d shared exact episodes; dropping it.",
                    task_id,
                    len(common),
                )
                group[task_id].close()
                del group[task_id]
                continue
            common = common[:n_episodes]
            init_states = np.stack(
                [indexed[0][episode]["init_state"] for episode in common]
            ).astype(np.float64)
            sub_envs = getattr(group[task_id], "envs", None)
            if sub_envs is None:
                raise RuntimeError("Episode-exact eval requires SyncVectorEnv.")
            for sub_env in sub_envs:
                base = sub_env.unwrapped
                base.init_states = True
                base._init_states = init_states
            for index, model in enumerate(indexed):
                maps[index][(task_group, int(task_id))] = [
                    model[episode]["skills"] for episode in common
                ]
    return maps


def _reset_init_state_ids(envs: dict) -> None:
    for group in envs.values():
        for vector_env in group.values():
            for sub_env in getattr(vector_env, "envs", []):
                base = sub_env.unwrapped
                base.init_state_id = base.episode_index


def _policy_config(spec: dict, base, device: torch.device):
    config = PreTrainedConfig.from_pretrained(spec["policy_path"])
    config.pretrained_path = Path(spec["policy_path"])
    config.device = str(device)
    config.use_amp = base.use_amp
    config.n_action_steps = base.n_action_steps
    config.compile_model = False
    config.gradient_checkpointing = False
    for field in (
        "fsq_path",
        "dino_model_path",
        "terminator_dino_model_path",
        "tokenizer_path",
    ):
        setattr(config, field, spec[field])
    return config


def _build_context(spec: dict, cfg, device: torch.device) -> dict:
    policy_config = _policy_config(spec, cfg.policy, device)
    policy = make_policy(
        cfg=policy_config, env_cfg=cfg.env, rename_map=cfg.rename_map
    )
    policy.eval()
    terminator = (
        CheckpointTerminator(policy)
        if spec["advance_mode"] == "terminator"
        else None
    )
    # A Stage-1 GT run does not need its predictor/VLM and can release it.  Stage 2
    # is different: the likelihood blocks always consume the pristine frozen VLM
    # memory owned by skill_predictor, even when the injected skill itself is GT.
    if (
        spec["skill_source"] == "gt"
        and policy_config.type == "skill_expert"
        and policy.model.skill_predictor is not None
    ):
        policy.model.skill_predictor = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
        log.info("[%s] released unused predictor.", spec["label"])
    if spec["skill_source"] == "predictor" and policy.model.skill_predictor is None:
        raise ValueError(f"[{spec['label']}] checkpoint predictor is unavailable.")

    wrapper = Stage1OraclePolicy(
        policy,
        terminator,
        skill_source=spec["skill_source"],
        advance_mode=spec["advance_mode"],
        end_mode=os.environ["SKILL_END_MODE"],
        end_threshold=float(os.environ["SKILL_END_THRESHOLD"]),
        progress_threshold=float(os.environ["SKILL_END_PROGRESS_THRESHOLD"]),
        max_skill_length=int(os.environ["INFERENCE_SKILL_MAX_LENGTH"]),
        n_action_steps=policy_config.n_action_steps,
    )
    wrapper.eval()
    overrides = {
        "device_processor": {"device": str(device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    if spec.get("tokenizer_path"):
        overrides["tokenizer_processor"] = {
            "tokenizer_name": spec["tokenizer_path"]
        }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=policy_config.pretrained_path,
        preprocessor_overrides=overrides,
    )
    return {
        "policy": wrapper,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "config": policy_config,
    }


def _panel_dir(index: int, label: str) -> str:
    safe = "".join(character if character.isalnum() or character in "._-" else "-" for character in label)
    return f"{index:02d}_{safe.strip('-_') or 'model'}"


def _stitch_panels(panels: list[tuple[Path, str]], output_dir: Path, per_row: int) -> None:
    if len(panels) < 2:
        return
    try:
        import imageio.v2 as imageio
        from compare_videos import even, label_bar, load_font, make_panel, read_video
    except Exception as error:  # noqa: BLE001
        log.warning("Side-by-side video generation skipped: %s", error)
        return

    height = 256
    bar_height = even(max(20, height // 9))
    font = load_font(int(bar_height * 0.62))
    first_dir = panels[0][0]
    written = 0
    for task_dir in sorted(path for path in first_dir.glob("*") if path.is_dir()):
        for first_video in sorted(task_dir.glob("eval_episode_*.mp4")):
            videos = [directory / task_dir.name / first_video.name for directory, _ in panels]
            if not all(video.is_file() for video in videos):
                continue
            reads = [read_video(video) for video in videos]
            frame_sets = [read[0] for read in reads]
            if any(not frames for frames in frame_sets):
                continue
            bars = []
            for (_, label), frames in zip(panels, frame_sets, strict=True):
                frame_height, frame_width = frames[0].shape[:2]
                width = even(max(2, round(frame_width * height / frame_height)))
                bars.append(label_bar(width, bar_height, label, font))
            destination = output_dir / task_dir.name / first_video.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(
                str(destination),
                fps=reads[0][1],
                codec="libx264",
                quality=8,
                macro_block_size=None,
            )
            columns = per_row if per_row > 0 else len(panels)
            for frame_index in range(max(len(frames) for frames in frame_sets)):
                tiles = [
                    make_panel(frames[min(frame_index, len(frames) - 1)], height, bar)
                    for frames, bar in zip(frame_sets, bars, strict=True)
                ]
                rows = [
                    np.hstack(tiles[start : start + columns])
                    for start in range(0, len(tiles), columns)
                ]
                max_width = max(row.shape[1] for row in rows)
                rows = [
                    row
                    if row.shape[1] == max_width
                    else np.pad(row, ((0, 0), (0, max_width - row.shape[1]), (0, 0)))
                    for row in rows
                ]
                frame = np.vstack(rows)
                frame = frame[
                    : frame.shape[0] - frame.shape[0] % 2,
                    : frame.shape[1] - frame.shape[1] % 2,
                ]
                writer.append_data(frame)
            writer.close()
            written += 1
    log.info("Wrote %d side-by-side videos to %s.", written, output_dir)


def _maybe_log_wandb(cfg, infos: dict[str, dict], specs: list[dict]) -> None:
    if not cfg.wandb_project:
        return
    try:
        import wandb

        wandb.init(
            project=cfg.wandb_project,
            name=cfg.job_name,
            config={
                "models": [
                    {
                        "label": spec["label"],
                        "policy_path": spec["policy_path"],
                        "skill_source": spec["skill_source"],
                        "conditioning_route": spec.get("conditioning_route"),
                        "action_loss_mode": spec.get("action_loss_mode"),
                    }
                    for spec in specs
                ],
                "n_episodes": cfg.eval.n_episodes,
            },
        )
        payload = {}
        for label, info in infos.items():
            for key, value in info.get("overall", {}).items():
                if isinstance(value, (int, float)):
                    payload[f"{label}/overall/{key}"] = float(value)
            for task in info.get("per_task", []):
                task_id = task.get("task_id")
                metrics = task.get("metrics", task)
                success = metrics.get("pc_success", metrics.get("success_rate"))
                if task_id is not None and success is not None:
                    payload[f"{label}/task_{int(task_id):02d}/success"] = float(success)
        wandb.log(payload)
        wandb.finish()
    except Exception as error:  # noqa: BLE001
        log.warning("wandb logging failed: %s", error)


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    supported = {"skill_expert", "skill_vla_stage2"}
    if cfg.policy is None or cfg.policy.type not in supported:
        raise ValueError(
            "Skill-conditioned eval requires a skill_expert or "
            "skill_vla_stage2 checkpoint."
        )
    specs = json.loads(os.environ.get("MODELS_JSON", "") or "[]")
    if not specs:
        raise ValueError("MODELS_JSON is empty; resolve the eval config first.")

    device = get_safe_torch_device(cfg.policy.device, log=True)
    set_seed(cfg.seed)
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )
    try:
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=cfg.env, policy_cfg=cfg.policy
        )
        task_descriptions = _libero_task_descriptions(cfg.env.task)
        episode_exact = all(spec.get("eval_init_states_path") for spec in specs)
        oracle_maps = (
            _episode_exact_oracle_maps(
                envs, specs, cfg.env.task, cfg.eval.n_episodes
            )
            if episode_exact
            else _language_oracle_maps(
                envs, specs, task_descriptions, cfg.eval.n_episodes
            )
        )
        if not oracle_maps or not oracle_maps[0]:
            raise RuntimeError("No requested LIBERO tasks matched every model dataset.")

        output_dir = Path(cfg.output_dir)
        infos = {}
        video_panels = []
        for index, (spec, oracle_map) in enumerate(zip(specs, oracle_maps, strict=True)):
            log.info(
                "[%s] loading %s (skill_source=%s, advance=%s).",
                spec["label"],
                spec["policy_path"],
                spec["skill_source"],
                spec["advance_mode"],
            )
            context = _build_context(spec, cfg, device)
            panel_root = output_dir / "panels" / _panel_dir(index, spec["label"])
            try:
                _reset_init_state_ids(envs)
                use_gt = spec["skill_source"] == "gt"
                with torch.no_grad(), (
                    torch.autocast(device_type=device.type)
                    if context["config"].use_amp
                    else nullcontext()
                ):
                    info = eval_policy_all(
                        envs=envs,
                        policy=context["policy"],
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=context["preprocessor"],
                        postprocessor=context["postprocessor"],
                        n_episodes=cfg.eval.n_episodes,
                        max_episodes_rendered=cfg.eval.max_videos_per_task,
                        video_frame_stride=cfg.eval.video_frame_stride,
                        video_fps=cfg.eval.video_fps,
                        videos_dir=panel_root / "videos",
                        return_episode_data=False,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                        forced_skill_token_sequences_by_task=(oracle_map if use_gt else None),
                        reference_skill_token_sequences_by_task=(None if use_gt else oracle_map),
                        skill_html_dir=(panel_root / "skill_html" if cfg.eval.skill_html else None),
                        skill_html_train_samples=cfg.eval.skill_html_train_samples,
                        skill_html_skill_latents_path=spec["skill_latents_path"],
                        skill_html_raw_dataset_dir=spec["raw_dataset_dir"],
                        skill_html_image_key=cfg.eval.skill_html_image_key,
                        task_descriptions=task_descriptions,
                    )
                infos[spec["label"]] = info
                video_panels.append((panel_root / "videos", spec["label"]))
                log.info("[%s] overall=%s", spec["label"], info.get("overall"))
            finally:
                del context
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        _stitch_panels(
            video_panels,
            output_dir / "side_by_side",
            int(os.environ.get("MODELS_PER_ROW", "0") or 0),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        task_tag = os.environ.get("TASK_TAG", "").strip()
        info_path = output_dir / (
            f"eval_info_{task_tag}.json" if task_tag else "eval_info.json"
        )
        info_path.write_text(json.dumps(infos, indent=2))
        for label, info in infos.items():
            print(f"{label}: {info['overall']}")
        print("Saved:", info_path)
        _maybe_log_wandb(cfg, infos, specs)
    finally:
        close_envs(envs)


if __name__ == "__main__":
    eval_main()
