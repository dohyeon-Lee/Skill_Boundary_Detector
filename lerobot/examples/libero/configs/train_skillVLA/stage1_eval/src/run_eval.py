#!/usr/bin/env python
"""Stage-1 (skill_expert) closed-loop oracle eval on the LIBERO sim — EPISODE-EXACT.

The action expert has no skill predictor, so the GT skill sequence is supplied (oracle). Unlike a
task-level eval, here every rollout reproduces a SPECIFIC dataset episode: the env is reset to that
episode's exact MuJoCo init_state (eval_init_states.npz, built by oracle_matching/) and fed that
episode's own GT skill sequence. Scene and skills therefore come from the same demonstration.

Per skill the FSQ terminator advances the cursor ([z, current image, current state] -> progress +
termination; skill_end_mode chooses how they gate). When the checkpoint has an Oracle (1-2 / single),
that skill's GT state-trajectory is fed to the Oracle to produce r (the oracle-r upper bound); a 1-1
checkpoint (no Oracle) runs the learned null token. The clean SkillExpert policy is untouched; all
orchestration lives in OracleSkillExpertPolicy and the verified lerobot eval harness is reused.

Episode<->env alignment: per task the env's init_state_id and the forced-sequence index are both the
global episode index (batch_ix*num_envs + b), so overriding each task env's `_init_states` with the
matched per-episode init states (ordered by episode_index) keeps the scene paired with its skills.
See eval_oracle.py for the data/FSQ helpers.
"""

import json
import logging
import shutil
import sys
from collections import deque
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.skill_expert.configuration_skill_expert import SkillExpertConfig
from lerobot.policies.skill_expert.dataset_skill_expert import SKILL_STATE_TRAJ
from lerobot.processor import NormalizerProcessorStep
from lerobot.policies.skillVLA.processor_skillVLA import SkillVLAPreserveRawStateProcessorStep
from lerobot.scripts.lerobot_skillvla_eval import (
    _libero_task_descriptions,
    close_envs,
    eval_policy_all,
)
from lerobot.utils.constants import ACTION, OBS_STATE
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
    """Wraps the clean SkillExpert policy for closed-loop episode-exact oracle eval.

    Holds the per-env GT skill sequence (+ per-skill GT length and resampled state-trajectory) and a
    cursor; runs the FSQ terminator every step to advance the cursor; injects the current skill code
    (and, for an Oracle checkpoint in oracle-r mode, that skill's state-trajectory -> r) before calling
    the policy. Subclasses PreTrainedPolicy (the harness requires it); action abstractmethods delegate
    to the wrapped policy while select_action does the orchestration.
    """

    config_class = SkillExpertConfig
    name = "skill_expert_oracle"

    def __init__(self, policy, terminator, *, end_threshold: float, end_mode: str, progress_threshold: float,
                 advance_mode: str, max_skill_len: int, n_action_steps: int, oracle_r: bool, input_source: str):
        super().__init__(policy.config)
        self.policy = policy
        self.terminator = terminator
        self.end_threshold = float(end_threshold)
        self.progress_threshold = float(progress_threshold)
        self.end_mode = str(end_mode)        # "termination" | "progress" | "and"
        if self.end_mode not in ("termination", "progress", "and", "or"):
            raise ValueError(f"skill_end_mode must be 'termination'|'progress'|'and'|'or', got {end_mode!r}")
        self.advance_mode = str(advance_mode)  # "terminator" (FSQ gates) | "gt" (advance by GT duration)
        if self.advance_mode not in ("terminator", "gt"):
            raise ValueError(f"skill_advance_mode must be 'terminator' or 'gt', got {advance_mode!r}")
        self.max_skill_len = int(max_skill_len)
        self.n_action_steps = int(n_action_steps)
        self.oracle_r = bool(oracle_r)       # feed GT Oracle input -> r (else learned null token)
        self.input_source = str(input_source)  # "state" (per-skill state-traj) | "action" (per-step GT chunk)
        self.chunk_size = int(policy.config.chunk_size)
        self._seqs: list[list[int]] | None = None
        self._gt_lengths: list[list[int]] | None = None   # GT demo frames per skill
        self._state_trajs: list[list[np.ndarray | None]] = []  # per skill: (N, state_dim) Oracle input ("state")
        self._gt_actions: list[np.ndarray | None] = []    # per episode: (T, action_dim) GT actions ("action")
        self._cursors: list[int] = []
        self._skill_step: list[int] = []
        self._queue: deque = deque(maxlen=n_action_steps)
        self._trace: list[dict] = []     # per-skill records for the HTML (codes are GT, timing is runtime)
        self._active: list = []
        self._order: list[int] = []
        self._t = 0
        self._started = False

    # ── oracle skill sequence interface ──
    def set_forced_skill_token_sequences(self, sequences) -> None:
        """Each entry is a per-episode dict ``{"skills": [{"token", "gt_length", "state_traj"?}, ...],
        "gt_actions": (T, action_dim) | None}``. ``skills`` give the GT skill sequence (+ per-skill length
        for the terminator/timing, and the resampled state-traj for the "state"-mode Oracle); ``gt_actions``
        is the episode's per-frame GT action sequence for the "action"-mode Oracle. A bare list of skills
        (no dict) is tolerated (→ no gt_actions)."""
        self._seqs, self._gt_lengths, self._state_trajs, self._gt_actions = [], [], [], []
        for entry in sequences:
            skills = entry["skills"] if isinstance(entry, dict) else entry
            gt_act = entry.get("gt_actions") if isinstance(entry, dict) else None
            codes, lens, trajs = [], [], []
            for x in skills:
                if isinstance(x, dict):
                    codes.append(int(x["token"]))
                    lens.append(int(x.get("gt_length", 0)))
                    trajs.append(np.asarray(x["state_traj"], np.float32) if x.get("state_traj") is not None else None)
                else:
                    codes.append(int(x)); lens.append(0); trajs.append(None)
            self._seqs.append(codes)
            self._gt_lengths.append(lens)
            self._state_trajs.append(trajs)
            self._gt_actions.append(np.asarray(gt_act, np.float32) if gt_act is not None else None)
        self.reset()

    def set_reference_skill_token_sequences(self, sequences) -> None:  # unused (no skill predictor)
        return None

    def get_skill_trace(self) -> list:
        return self._trace

    def get_gt_timeline(self) -> dict[int, list[dict]]:
        """Per batch index -> the full GT skill timeline ``[{"token", "length"}, ...]`` (length = GT demo
        frame count per skill), for comparing GT vs runtime terminator transition timing."""
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

    def _gt_action_chunk(self, b: int) -> torch.Tensor:
        """env b's GT action chunk at the CURRENT rollout step: GT_actions[t : t+chunk_size], tail-padded
        with the last action (hold). (chunk_size, action_dim) float32. The "action"-mode Oracle input."""
        ga = self._gt_actions[b]
        K = self.chunk_size
        if ga is None or len(ga) == 0:
            return torch.zeros(K, self.config.output_features[ACTION].shape[0], dtype=torch.float32)
        t = min(self._t, len(ga) - 1)
        seg = ga[t : t + K]
        if seg.shape[0] < K:
            seg = np.concatenate([seg, np.repeat(ga[-1:], K - seg.shape[0], axis=0)], axis=0)
        return torch.from_numpy(seg.astype(np.float32))

    def _start_skill(self, b: int) -> None:
        code = self._seqs[b][min(self._cursors[b], len(self._seqs[b]) - 1)]
        self._order[b] += 1
        self._trace.append({
            "batch_index": b, "codebook_token": int(code), "skill_index": self._order[b],
            "episode_timestep": self._t, "length": 0, "end_probs": [], "skill_source": "oracle",
        })
        self._active[b] = len(self._trace) - 1

    def _fired(self, b: int, progress: torch.Tensor, term: torch.Tensor) -> bool:
        """Whether skill b ends this step. GT mode = by demo duration; else the FSQ terminator:
        'and' = end-prob AND progress (BOTH at this step), 'or' = EITHER (end-prob OR progress — fires on
        whichever crosses first, robust to term/progress peaking at different steps), 'termination'|'progress'
        = that single signal. A max-length cap force-advances regardless."""
        if self.advance_mode == "gt":
            gt_len = self._gt_lengths[b][min(self._cursors[b], len(self._gt_lengths[b]) - 1)]
            return self._skill_step[b] >= max(1, int(gt_len))
        term_hi, prog_hi = float(term[b]) >= self.end_threshold, float(progress[b]) >= self.progress_threshold
        if self.end_mode == "and":
            sig = term_hi and prog_hi
        elif self.end_mode == "or":
            sig = term_hi or prog_hi
        else:
            sig = (float(progress[b]) if self.end_mode == "progress" else float(term[b])) >= self.end_threshold
        return sig or (self.max_skill_len > 0 and self._skill_step[b] >= self.max_skill_len)

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
            if self._fired(b, progress, term) and self._cursors[b] < len(self._seqs[b]) - 1:
                self._cursors[b] += 1
                self._skill_step[b] = 0
                self._start_skill(b)
                advanced = True
        if advanced:
            self._queue.clear()  # re-predict with the new skill

        # 2) Action expert at chunk cadence: predict a chunk for the current skill (+ Oracle r).
        if len(self._queue) == 0:
            codes = self._current_codes(bsize, device)
            inj = dict(batch)
            inj["skill_sequence"] = codes.view(bsize, 1)
            inj["skill_index"] = torch.zeros(bsize, dtype=torch.long, device=device)
            if self.oracle_r and self.input_source == "action":
                # "action" Oracle: per-step GT action chunk, sliced from THIS rollout step (skill-agnostic
                # → robust to imperfect terminator skill boundaries). Pad the tail with the last action.
                inj[ACTION] = torch.stack([self._gt_action_chunk(b) for b in range(bsize)]).to(device)
            elif self.oracle_r:                                                        # "state" Oracle
                trajs, des = [], []
                for b in range(bsize):
                    k = min(self._cursors[b], len(self._state_trajs[b]) - 1)
                    trajs.append(torch.from_numpy(self._state_trajs[b][k]))
                    des.append(max(1, int(self._gt_lengths[b][k])))
                inj[SKILL_STATE_TRAJ] = torch.stack(trajs).to(device)                  # (B, N, state_dim)
                inj["skill_ds"] = torch.zeros(bsize, dtype=torch.long, device=device)  # ds=0, de=len-1
                inj["skill_de"] = torch.tensor([d - 1 for d in des], dtype=torch.long, device=device)
            chunk = self.policy.predict_action_chunk(inj, use_r=self.oracle_r)[:, : self.n_action_steps]
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
    with no matched episode are dropped (closed) — they have no GT sequence to inject. Returns the
    forced-sequence map {(task_group, task_id): [per-episode skills]} for the kept tasks."""
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
                base.init_states = True            # ensure reset() takes the set_init_state path
                base._init_states = init_arr        # env indexes by init_state_id (= global episode index)
            # forced[(tg,tid)][i] pairs with init_arr[i % n_ep] via the same global index (eval wraps both).
            # Each entry carries the skill sequence + (action mode) the episode's GT action sequence.
            forced[(task_group, int(task_id))] = [
                {"skills": r["skills"], "gt_actions": r.get("gt_actions")} for r in records]
    return forced


def _reset_init_state_ids(envs: dict) -> None:
    """Rewind every task env's init_state_id to its episode_index, so a SECOND eval pass replays the
    SAME per-episode scenes as the first (side-by-side A/B over identical init states)."""
    for group in envs.values():
        for vec in group.values():
            for sub in getattr(vec, "envs", []):
                base = sub.unwrapped
                base.init_state_id = base.episode_index


def _stitch_side_by_side(dir_a: Path, dir_b: Path, out_dir: Path, label_a: str, label_b: str,
                         height: int = 256) -> None:
    """Glue dir_a | dir_b rollout videos (same videos/{task}/eval_episode_{ep}.mp4 layout) horizontally,
    each panel labelled, → out_dir/{task}/eval_episode_{ep}.mp4. Reuses the video_compare/ helpers."""
    try:
        sys.path.insert(0, str(_HERE.parent / "video_compare"))
        from compare_videos import even, label_bar, load_font, make_panel, read_video  # noqa: PLC0415
        import imageio.v2 as imageio  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 — stitching is a convenience; never fail the eval over it
        log.warning("side-by-side stitch skipped (video libs unavailable): %s", exc)
        return
    H, bar_h = even(height), even(max(20, height // 9))
    font = load_font(int(bar_h * 0.62))

    def panel_bar(frames, label):
        h, w = frames[0].shape[:2]
        return label_bar(even(max(2, round(w * H / h))), bar_h, label, font)

    n = 0
    for taskdir_a in sorted(p for p in dir_a.glob("*") if p.is_dir()):
        taskdir_b = dir_b / taskdir_a.name
        for mp4_a in sorted(taskdir_a.glob("eval_episode_*.mp4")):
            mp4_b = taskdir_b / mp4_a.name
            if not mp4_b.exists():
                continue
            fa, fps = read_video(mp4_a)
            fb, _ = read_video(mp4_b)
            if not fa or not fb:
                continue
            bar_a, bar_b = panel_bar(fa, label_a), panel_bar(fb, label_b)
            (out_dir / taskdir_a.name).mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(out_dir / taskdir_a.name / mp4_a.name), fps=fps,
                                        codec="libx264", quality=8, macro_block_size=None)
            for i in range(max(len(fa), len(fb))):
                frame = np.hstack([make_panel(fa[min(i, len(fa) - 1)], H, bar_a),
                                   make_panel(fb[min(i, len(fb) - 1)], H, bar_b)])
                writer.append_data(frame[:, :-1] if frame.shape[1] % 2 else frame)
            writer.close()
            n += 1
    log.info("side-by-side: wrote %d stitched clips → %s", n, out_dir)


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
    # FSQ terminator (FSQ.pt) → terminate(codes, state, image[, wrist]) → (progress, termination_prob).
    terminator = make_terminator(
        cfg.policy.fsq_path, device, dino_path=cfg.policy.terminator_dino_model_path,
        libero_examples_dir=_LIBERO_EXAMPLES)

    # Episode-exact oracle data: join skillvla dataset (GT skills + per-skill state-traj OR per-frame GT
    # actions) with the init-state npz (episode -> MuJoCo init_state + scene), grouped by LIBERO task_id.
    input_source = str(cfg.policy.oracle_input_source)
    episode_data = load_episode_oracle_data(
        cfg.policy.skill_label_dataset_dir, cfg.policy.eval_init_states_path, cfg.env.task,
        input_source=input_source,
        resample_n=cfg.policy.oracle_resample_n, spline_degree=cfg.policy.oracle_spline_degree)
    forced_by_task = _override_init_states(envs, episode_data)
    logging.info("Episode-exact eval over %d tasks (n_episodes=%d per task).",
                 len(forced_by_task), cfg.eval.n_episodes)

    # An Oracle checkpoint (1-2/single) → ALWAYS the A/B side-by-side: each scene is rolled out with r
    # (skill+residual) AND with the null token (skill-only). A 1-1 checkpoint has no Oracle → a single
    # null pass (the two passes would be identical). The phase + r regime are inferred from the checkpoint
    # (model_dir path + use_oracle); no eval flag needed.
    side_by_side = bool(policy.model._oracle_active)
    gt_in = "GT state-traj" if input_source == "state" else "GT action chunk"
    logging.info("r regime: %s", (f"side-by-side (skill+residual [oracle-r, {gt_in}] vs skill-only [null])"
                                  if side_by_side else "null token (no Oracle)"))

    oracle = OracleSkillExpertPolicy(
        policy, terminator,
        end_threshold=cfg.policy.skill_end_threshold,
        end_mode=cfg.policy.skill_end_mode,
        progress_threshold=cfg.policy.skill_end_progress_threshold,
        advance_mode=cfg.policy.skill_advance_mode,
        max_skill_len=cfg.policy.inference_skill_max_length,
        n_action_steps=cfg.policy.n_action_steps,
        input_source=input_source,
        oracle_r=side_by_side,   # placeholder; _run_pass sets it per pass
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

    # ── Rollout(s). Oracle ckpt → A/B side-by-side, but PER TASK so the STITCHED combined videos stream
    # out in task/episode order (no separate per-pass video files kept): for each task, pass A (r=skill+
    # residual) and pass B (null=skill-only) over the SAME scenes (init_state_id rewound between), stitch
    # immediately → side_by_side/{task}/, then drop that task's raw per-pass videos. skill_html (per pass)
    # is kept under skillR/ & skillonly/. A 1-1 checkpoint (no Oracle) → a single null pass. ──
    out = Path(cfg.output_dir)
    common = dict(
        policy=oracle, env_preprocessor=env_preprocessor, env_postprocessor=env_postprocessor,
        preprocessor=preprocessor, postprocessor=postprocessor,
        n_episodes=cfg.eval.n_episodes, max_episodes_rendered=cfg.eval.max_videos_per_task,
        video_frame_stride=cfg.eval.video_frame_stride, video_fps=cfg.eval.video_fps,
        start_seed=cfg.seed, max_parallel_tasks=cfg.env.max_parallel_tasks,
        reference_skill_token_sequences_by_task=None,
        skill_html_train_samples=cfg.eval.skill_html_train_samples,
        skill_html_skill_latents_path=cfg.eval.skill_html_skill_latents_path,
        skill_html_raw_dataset_dir=cfg.eval.skill_html_raw_dataset_dir,
        skill_html_image_key=cfg.eval.skill_html_image_key, task_descriptions=task_id_to_desc,
    )

    def _eval(envs_, use_r, videos_dir, html_dir, forced):
        oracle.oracle_r = bool(use_r)
        with torch.no_grad(), (torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext()):
            return eval_policy_all(envs=envs_, videos_dir=videos_dir, skill_html_dir=html_dir,
                                   forced_skill_token_sequences_by_task=forced, **common)

    def _merge(tagged):                                               # [(tid, info)] → one {overall, per_task}
        per_task = [{"task_id": tid, **info.get("overall", {})} for tid, info in tagged]
        succ = [d.get("pc_success") for d in per_task if isinstance(d.get("pc_success"), (int, float))]
        return {"overall": {"pc_success": float(np.mean(succ)) if succ else 0.0}, "per_task": per_task}

    if side_by_side:
        sbs_dir = out / "side_by_side"
        r_html = (out / "skillR" / "skill_html") if cfg.eval.skill_html else None
        n_html = (out / "skillonly" / "skill_html") if cfg.eval.skill_html else None
        tagged_r, tagged_n = [], []
        for tg, grp in envs.items():
            for tid in list(grp.keys()):
                one_env, one_forced = {tg: {tid: grp[tid]}}, {(tg, tid): forced_by_task[(tg, tid)]}
                tmp_r, tmp_n = out / "_tmp" / "R" / "videos", out / "_tmp" / "null" / "videos"
                iA = _eval(one_env, True, tmp_r, r_html, one_forced)
                _reset_init_state_ids(one_env)                        # rewind → pass B sees the same scenes
                iB = _eval(one_env, False, tmp_n, n_html, one_forced)
                _stitch_side_by_side(tmp_r, tmp_n, sbs_dir, "skill + residual", "skill only")
                shutil.rmtree(out / "_tmp", ignore_errors=True)       # drop raw per-pass videos (keep combined)
                tagged_r.append((tid, iA)); tagged_n.append((tid, iB))
                log.info("task %s → side_by_side (skill+residual %s | skill-only %s)",
                         tid, iA.get("overall"), iB.get("overall"))
        info = {"skill+residual": _merge(tagged_r), "skill_only": _merge(tagged_n)}
        wandb_infos = {"skillR/": info["skill+residual"], "skillonly/": info["skill_only"]}
        print("side_by_side done →", sbs_dir)
    else:                                                             # 1-1: no Oracle → single null pass
        info = _eval(envs, False, out / "videos",
                     (out / "skill_html") if cfg.eval.skill_html else None, forced_by_task)
        print("Overall:", info["overall"])
        wandb_infos = {"": info}

    close_envs(envs)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)
    _maybe_log_wandb(cfg, wandb_infos)


def _maybe_log_wandb(cfg, infos: dict) -> None:
    """infos: {prefix: eval_info}. Single pass → {"": info}; side-by-side → {"skillR/":.., "skillonly/":..}
    (both variants logged to ONE run under their prefixes)."""
    project = getattr(cfg, "wandb_project", None)
    if not project:
        return
    try:
        import wandb

        wandb.init(project=project, name=cfg.job_name,
                   config={"policy_path": str(cfg.policy.pretrained_path), "n_episodes": cfg.eval.n_episodes})
        payload: dict[str, float] = {}
        for pref, info in infos.items():
            payload.update({f"{pref}overall/{k}": float(v)
                            for k, v in info.get("overall", {}).items() if isinstance(v, (int, float))})
            for ti in info.get("per_task", []):
                tid, sr = ti.get("task_id"), ti.get("pc_success", ti.get("success_rate"))
                if tid is not None and sr is not None:
                    payload[f"{pref}task_{int(tid):02d}/success"] = float(sr)
        wandb.log(payload)
        wandb.finish()
    except Exception as exc:  # noqa: BLE001
        logging.warning("wandb logging failed: %s", exc)


if __name__ == "__main__":
    eval_main()
