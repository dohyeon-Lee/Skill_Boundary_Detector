#!/usr/bin/env python
"""Stage-1 (skill_expert) closed-loop oracle eval on the LIBERO sim — EPISODE-EXACT.

The action expert has no skill predictor, so the GT skill sequence is supplied (oracle) and the
FSQ terminator advances the skill each step ([z, current 3rd-person image, current state] ->
termination > threshold). The clean SkillExpert policy is left untouched; all orchestration lives
in OracleSkillExpertPolicy (below) and the verified lerobot eval harness is reused.

Each rollout reproduces a SPECIFIC dataset episode: the env is reset to that episode's exact MuJoCo
init_state (eval_init_states.npz, built by oracle_matching/) and fed THAT episode's GT skill sequence.
Episode<->env alignment: per task the env's init_state_id and the forced-sequence index are BOTH the
global episode index (batch_ix*num_envs + b), so overriding each task env's `_init_states` with the
matched per-episode init states (ordered by episode_index) keeps every scene paired with its skills.
Needs SyncVectorEnv (eval.use_async_envs=false). See eval_oracle.py for the data/FSQ helpers.
"""

import json
import logging
import sys
from collections import deque
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.skill_expert.configuration_skill_expert import SkillExpertConfig
from lerobot.processor import NormalizerProcessorStep
from lerobot.policies.skillVLA.processor_skillVLA import SkillVLAPreserveRawStateProcessorStep
from lerobot.scripts.lerobot_skillvla_eval import (
    _libero_task_descriptions,
    close_envs,
    eval_policy_all,
)
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.random_utils import set_seed

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # local eval_oracle
from eval_oracle import load_episode_oracle_data, make_terminator  # noqa: E402

# Path to examples/libero so eval_oracle can import the FSQ model definition.
_LIBERO_EXAMPLES = _HERE.parents[3]  # .../examples/libero
_IMAGE_KEY = "observation.images.image"        # 3rd-person view for the terminator
_WRIST_KEY = "observation.images.wrist_image"  # wrist view for the terminator (2nd DINO encoder)

log = logging.getLogger(__name__)


class OracleSkillExpertPolicy(PreTrainedPolicy):
    """Wraps the clean SkillExpert policy for closed-loop oracle eval.

    Holds the per-env GT skill sequence + cursor, runs the FSQ terminator every step to
    advance the cursor, and injects the current skill code into the batch before calling
    the policy. Subclasses PreTrainedPolicy (the eval harness requires it); the action
    abstractmethods delegate to the wrapped policy, while select_action does the oracle
    orchestration.
    """

    config_class = SkillExpertConfig
    name = "skill_expert_oracle"

    def __init__(self, policy, terminator, *, end_threshold: float, progress_threshold: float,
                 end_mode: str, advance_mode: str, max_skill_len: int, n_action_steps: int):
        super().__init__(policy.config)
        self.policy = policy
        self.terminator = terminator
        self.end_threshold = float(end_threshold)              # gate on the TERMINATION-prob signal
        self.progress_threshold = float(progress_threshold)    # gate on the PROGRESS signal
        self.end_mode = str(end_mode)  # "termination" | "progress" | "or" (either) | "and" (both)
        if self.end_mode not in ("termination", "progress", "or", "and"):
            raise ValueError(f"skill_end_mode must be termination|progress|or|and, got {end_mode!r}")
        self.advance_mode = str(advance_mode)  # "terminator" (FSQ gates) | "gt" (advance by GT duration)
        if self.advance_mode not in ("terminator", "gt"):
            raise ValueError(f"skill_advance_mode must be 'terminator' or 'gt', got {advance_mode!r}")
        self.max_skill_len = int(max_skill_len)
        self.n_action_steps = int(n_action_steps)
        self._seqs: list[list[int]] | None = None
        self._gt_lengths: list[list[int]] | None = None   # GT demo frames per skill (for timing compare)
        self._cursors: list[int] = []
        self._skill_step: list[int] = []
        self._queue: deque = deque(maxlen=n_action_steps)
        self._trace: list[dict] = []     # per-skill records for the HTML (codes are GT, but
        self._active: list = []          # timing + progress are runtime → recorded here)
        self._order: list[int] = []
        self._t = 0
        self._started = False

    # ── oracle skill sequence interface ──
    def set_forced_skill_token_sequences(self, sequences) -> None:
        """Each sequence is a per-episode list of skills: either bare codes or
        ``{"token": code, "gt_length": frames}`` dicts (gt_length feeds the timing compare)."""
        self._seqs, self._gt_lengths = [], []
        for seq in sequences:
            codes, lens = [], []
            for x in seq:
                codes.append(int(x["token"] if isinstance(x, dict) else x))
                lens.append(int(x.get("gt_length", 0)) if isinstance(x, dict) else 0)
            self._seqs.append(codes)
            self._gt_lengths.append(lens)
        self.reset()

    def set_reference_skill_token_sequences(self, sequences) -> None:  # unused (no skill predictor)
        return None

    def get_skill_trace(self) -> list:
        return self._trace

    def get_gt_timeline(self) -> dict[int, list[dict]]:
        """Per batch index -> the full GT skill timeline ``[{"token", "length"}, ...]`` (length =
        GT demo frame count per skill), for comparing GT vs runtime terminator transition timing."""
        if self._seqs is None or self._gt_lengths is None:
            return {}
        return {
            b: [{"token": int(c), "length": int(n)} for c, n in zip(self._seqs[b], self._gt_lengths[b])]
            for b in range(len(self._seqs))
        }

    def reset(self) -> None:
        self.policy.reset()
        self._queue.clear()
        n = len(self._seqs) if self._seqs is not None else 0
        self._cursors = [0] * n
        self._skill_step = [0] * n
        self._order = [-1] * n
        self._active = [None] * n
        self._trace = []
        self._t = 0
        self._started = False

    def _current_codes(self, batch_size: int, device) -> torch.Tensor:
        codes = [self._seqs[b][min(self._cursors[b], len(self._seqs[b]) - 1)] for b in range(batch_size)]
        return torch.tensor(codes, dtype=torch.long, device=device)

    def _start_skill(self, b: int) -> None:
        code = self._seqs[b][min(self._cursors[b], len(self._seqs[b]) - 1)]
        self._order[b] += 1
        self._trace.append({
            "batch_index": b, "codebook_token": int(code), "skill_index": self._order[b],
            "episode_timestep": self._t, "length": 0, "end_probs": [], "skill_source": "oracle",
        })
        self._active[b] = len(self._trace) - 1

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        device = next(self.policy.parameters()).device
        bsize = batch[OBS_STATE].shape[0]
        if self._seqs is None:
            raise RuntimeError("set_forced_skill_token_sequences must be called before eval.")
        if not self._started:
            for b in range(bsize):
                self._start_skill(b)
            self._started = True

        # 1) FSQ terminator every step → record progress, then advance the per-env skill cursor.
        codes = self._current_codes(bsize, device)
        # wrist is used only by a dual-camera terminator; pass it when present (single ignores it).
        progress, term = self.terminator.terminate(
            codes, batch["skill_decoder_state"], batch[_IMAGE_KEY],
            batch.get(_WRIST_KEY) if self.terminator.use_wrist else None,
        )
        advanced = False
        for b in range(bsize):
            rec = self._trace[self._active[b]]
            rec["end_probs"].append(
                {"skill_step": self._skill_step[b], "prob": float(term[b]), "progress": float(progress[b])}
            )
            self._skill_step[b] += 1
            rec["length"] = self._skill_step[b]
            # Skill-transition gate: GT duration (oracle timing) or the FSQ terminator. The
            # terminator still runs every step above so its curves are recorded either way.
            if self.advance_mode == "gt":
                gt_len = self._gt_lengths[b][min(self._cursors[b], len(self._gt_lengths[b]) - 1)]
                fired = self._skill_step[b] >= max(1, int(gt_len))
            else:
                term_hi = float(term[b]) >= self.end_threshold
                prog_hi = float(progress[b]) >= self.progress_threshold
                if self.end_mode == "or":
                    sig = term_hi or prog_hi
                elif self.end_mode == "and":
                    sig = term_hi and prog_hi
                elif self.end_mode == "progress":
                    sig = prog_hi
                else:  # "termination"
                    sig = term_hi
                fired = sig or (self.max_skill_len > 0 and self._skill_step[b] >= self.max_skill_len)
            if fired and self._cursors[b] < len(self._seqs[b]) - 1:
                self._cursors[b] += 1
                self._skill_step[b] = 0
                self._start_skill(b)
                advanced = True
        if advanced:
            self._queue.clear()  # re-predict with the new skill

        # 2) Action expert at chunk cadence: predict a chunk for the current skill.
        if len(self._queue) == 0:
            codes = self._current_codes(bsize, device)
            inj = dict(batch)
            inj["skill_sequence"] = codes.view(bsize, 1)
            inj["skill_index"] = torch.zeros(bsize, dtype=torch.long, device=device)
            chunk = self.policy.predict_action_chunk(inj)[:, : self.n_action_steps]
            self._queue.extend(chunk.transpose(0, 1))
        self._t += 1
        return self._queue.popleft()

    # PreTrainedPolicy abstractmethods (unused at eval; delegate to the wrapped policy).
    def get_optim_params(self):
        return self.policy.get_optim_params()

    def forward(self, batch, *args, **kwargs):
        return self.policy.forward(batch, *args, **kwargs)

    def predict_action_chunk(self, batch, **kwargs):
        return self.policy.predict_action_chunk(batch, **kwargs)


def _override_init_states(envs: dict, episode_data: dict[int, list[dict]]) -> dict:
    """Replace each task env's LIBERO built-in init states with the matched per-episode init states
    (ordered by episode_index), so each rollout reproduces a specific dataset episode's scene. Tasks
    with no matched episode are dropped (closed). Returns the forced-sequence map
    {(task_group, task_id): [skills_ep0, skills_ep1, ...]} (each entry the episode's GT skill list,
    ordered the SAME as init states → both indexed by the global episode index → auto-paired)."""
    forced: dict[tuple[str, int], list[list[dict]]] = {}
    for task_group, group in envs.items():
        for task_id in list(group.keys()):
            records = episode_data.get(int(task_id))
            if not records:
                log.warning("No matched episodes for task_id=%s — dropping it from the eval.", task_id)
                group[task_id].close()
                del group[task_id]
                continue
            vec = group[task_id]
            subs = getattr(vec, "envs", None)
            if subs is None:
                raise RuntimeError("Episode-exact eval needs SyncVectorEnv (set eval.use_async_envs=false).")
            init_arr = np.stack([r["init_state"] for r in records]).astype(np.float64)  # (n_ep, state_dim)
            for sub in subs:
                base = sub.unwrapped
                base.init_states = True             # ensure reset() takes the set_init_state path
                base._init_states = init_arr         # env indexes by init_state_id (= global episode index)
            # forced[(tg,tid)][i] pairs with init_arr[i] via the same global episode index (eval wraps both).
            forced[(task_group, int(task_id))] = [r["skills"] for r in records]
    return forced


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    if cfg.policy is None or cfg.policy.type != "skill_expert":
        raise ValueError(f"stage1 eval expects policy.type='skill_expert', got {getattr(cfg.policy,'type',None)!r}")
    if not cfg.policy.fsq_path or not cfg.policy.skill_label_dataset_dir:
        raise ValueError("--policy.fsq_path and --policy.skill_label_dataset_dir are required for oracle eval.")
    if not cfg.policy.eval_init_states_path:
        raise ValueError("--policy.eval_init_states_path (eval_init_states.npz) is required for episode-exact eval.")

    device = get_safe_torch_device(cfg.policy.device, log=True)
    set_seed(cfg.seed)
    logging.info("Making environment.")
    envs = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs,
                    trust_remote_code=cfg.trust_remote_code)

    logging.info("Making policy + FSQ terminator.")
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.eval()
    # FSQ terminator from fsq_path (an FSQ.pt) — exposes terminate(codes, state, image[, wrist]). When
    # eval_use_trained_terminator, its terminator submodules are overridden with the CO-TRAINED (fine-tuned)
    # terminator saved in THIS checkpoint (model.fsq_term_train.*); the DINO backbone stays from FSQ.pt.
    finetuned_ckpt = cfg.policy.pretrained_path if cfg.policy.eval_use_trained_terminator else None
    terminator = make_terminator(
        cfg.policy.fsq_path, device, dino_path=cfg.policy.terminator_dino_model_path,
        libero_examples_dir=_LIBERO_EXAMPLES, finetuned_ckpt=finetuned_ckpt)
    logging.info("Terminator: %s (%s)",
                 "co-trained (fine-tuned from checkpoint)" if getattr(terminator, "finetuned_loaded", 0)
                 else "raw FSQ.pt (no co-trained terminator loaded)",
                 f"{terminator.finetuned_loaded} tensors overridden" if getattr(terminator, "finetuned_loaded", 0)
                 else str(cfg.policy.fsq_path))

    # Episode-exact oracle data: join the skillvla dataset (GT skill sequence) with the init-state npz
    # (episode -> MuJoCo init_state + scene), grouped by LIBERO task_id; then override each task env's
    # init states so every rollout reproduces that episode's exact scene, paired with its GT skills.
    episode_data = load_episode_oracle_data(
        cfg.policy.skill_label_dataset_dir, cfg.policy.eval_init_states_path, cfg.env.task)
    forced_by_task = _override_init_states(envs, episode_data)
    logging.info("Episode-exact eval over %d tasks (n_episodes=%d per task).",
                 len(forced_by_task), cfg.eval.n_episodes)

    oracle = OracleSkillExpertPolicy(
        policy, terminator,
        end_threshold=cfg.policy.skill_end_threshold,
        progress_threshold=cfg.policy.skill_end_progress_threshold,
        end_mode=cfg.policy.skill_end_mode,
        advance_mode=cfg.policy.skill_advance_mode,
        max_skill_len=cfg.policy.inference_skill_max_length,
        n_action_steps=cfg.policy.n_action_steps,
    )

    # Processors: load the checkpoint's, then insert PreserveRawState so the terminator
    # sees the RAW observation.state (skill_decoder_state) before normalization.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}, "rename_observations_processor": {"rename_map": cfg.rename_map}},
    )
    norm_idx = next(i for i, s in enumerate(preprocessor.steps) if isinstance(s, NormalizerProcessorStep))
    preprocessor.steps.insert(norm_idx, SkillVLAPreserveRawStateProcessorStep())
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
    task_id_to_desc = _libero_task_descriptions(cfg.env.task)

    skill_html_dir = (Path(cfg.output_dir) / "skill_html") if cfg.eval.skill_html else None
    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy_all(
            envs=envs, policy=oracle,
            env_preprocessor=env_preprocessor, env_postprocessor=env_postprocessor,
            preprocessor=preprocessor, postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=cfg.eval.max_videos_per_task,
            video_frame_stride=cfg.eval.video_frame_stride, video_fps=cfg.eval.video_fps,
            videos_dir=Path(cfg.output_dir) / "videos",
            start_seed=cfg.seed, max_parallel_tasks=cfg.env.max_parallel_tasks,
            forced_skill_token_sequences_by_task=forced_by_task,
            reference_skill_token_sequences_by_task=None,
            skill_html_dir=skill_html_dir,
            skill_html_train_samples=cfg.eval.skill_html_train_samples,
            skill_html_skill_latents_path=cfg.eval.skill_html_skill_latents_path,
            skill_html_raw_dataset_dir=cfg.eval.skill_html_raw_dataset_dir,
            skill_html_image_key=cfg.eval.skill_html_image_key,
            task_descriptions=task_id_to_desc,
        )
    print("Overall:", info["overall"])
    close_envs(envs)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)
    _maybe_log_wandb(cfg, info)


def _maybe_log_wandb(cfg, info) -> None:
    project = getattr(cfg, "wandb_project", None)
    if not project:
        return
    try:
        import wandb

        wandb.init(project=project, name=cfg.job_name,
                   config={"policy_path": str(cfg.policy.pretrained_path), "n_episodes": cfg.eval.n_episodes})
        payload = {f"overall/{k}": float(v) for k, v in info.get("overall", {}).items() if isinstance(v, (int, float))}
        for ti in info.get("per_task", []):
            tid, sr = ti.get("task_id"), ti.get("pc_success", ti.get("success_rate"))
            if tid is not None and sr is not None:
                payload[f"task_{int(tid):02d}/success"] = float(sr)
        wandb.log(payload)
        wandb.finish()
    except Exception as exc:  # noqa: BLE001
        logging.warning("wandb logging failed: %s", exc)


if __name__ == "__main__":
    eval_main()
