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
import os
import shutil
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


def _reset_init_state_ids(envs: dict) -> None:
    """Rewind every task env's init_state_id to its episode_index, so a REPEAT pass replays the SAME
    per-episode scenes — used between models in a side-by-side so each model sees identical init states."""
    for group in envs.values():
        for vec in group.values():
            for sub in getattr(vec, "envs", []):
                base = sub.unwrapped
                base.init_state_id = base.episode_index


def _align_and_override_multi(envs: dict, episode_datas: list[dict]) -> list[dict]:
    """MULTI-model episode alignment. Keep only tasks+episodes present in EVERY model's dataset (episodes
    align by episode_index across FSQ runs — same filtered demos), override each task env's `_init_states`
    with those common episodes' init states (FSQ-independent → identical across models), and return a
    per-model forced-skill map keyed to that SAME common-episode order. Tasks/episodes not shared by all
    models are dropped (closed)."""
    forced_maps: list[dict] = [dict() for _ in episode_datas]
    for task_group, group in envs.items():
        for task_id in list(group.keys()):
            per_model = [{r["episode_index"]: r for r in ed.get(int(task_id), [])} for ed in episode_datas]
            common = sorted(set.intersection(*[set(d) for d in per_model])) if all(per_model) else []
            if not common:
                log.warning("task_id=%s not shared by all models — dropping it.", task_id)
                group[task_id].close(); del group[task_id]; continue
            subs = getattr(group[task_id], "envs", None)
            if subs is None:
                raise RuntimeError("Episode-exact eval needs SyncVectorEnv (set eval.use_async_envs=false).")
            init_arr = np.stack([per_model[0][ep]["init_state"] for ep in common]).astype(np.float64)
            for sub in subs:
                base = sub.unwrapped
                base.init_states = True
                base._init_states = init_arr
            for i, d in enumerate(per_model):
                forced_maps[i][(task_group, int(task_id))] = [d[ep]["skills"] for ep in common]
    return forced_maps


def _merge(tagged: list) -> dict:
    """[(task_id, per-task eval_info)] → one {overall: {pc_success}, per_task: [...]} (mirrors eval_policy_all)."""
    per_task = [{"task_id": tid, **info.get("overall", {})} for tid, info in tagged]
    succ = [d.get("pc_success") for d in per_task if isinstance(d.get("pc_success"), (int, float))]
    return {"overall": {"pc_success": float(np.mean(succ)) if succ else 0.0}, "per_task": per_task}


def _stitch_models(panels: list, out_dir: Path, height: int = 256, per_row: int = 0) -> None:
    """panels = [(videos_dir, label), ...]; for every task/episode present in ALL panels, glue the rollouts
    into a GRID of labelled panels — ``per_row`` panels per row (0 = all in ONE row, the old behaviour;
    e.g. 6 models @ per_row=3 → 2×3, 9 → 3×3; a short last row is right-padded black) →
    out_dir/{task}/eval_episode_{ep}.mp4. Reuses video_compare/ helpers; never fails the eval (skips if
    video libs are unavailable)."""
    try:
        sys.path.insert(0, str(_HERE.parent / "video_compare"))
        from compare_videos import even, label_bar, load_font, make_panel, read_video  # noqa: PLC0415
        import imageio.v2 as imageio  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 — stitching is a convenience; never fail the eval over it
        log.warning("side-by-side stitch skipped (video libs unavailable): %s", exc)
        return
    H, bar_h = even(height), even(max(20, height // 9))
    font = load_font(int(bar_h * 0.62))
    dir0 = Path(panels[0][0])
    n = 0
    for taskdir0 in sorted(p for p in dir0.glob("*") if p.is_dir()):
        for mp4_0 in sorted(taskdir0.glob("eval_episode_*.mp4")):
            mp4s = [Path(d) / taskdir0.name / mp4_0.name for d, _ in panels]
            if not all(p.exists() for p in mp4s):
                continue
            reads = [read_video(p) for p in mp4s]
            frames_list, fps = [r[0] for r in reads], reads[0][1]
            if any(not fr for fr in frames_list):
                continue
            bars = []
            for (_, lbl), fr in zip(panels, frames_list):
                h, w = fr[0].shape[:2]
                bars.append(label_bar(even(max(2, round(w * H / h))), bar_h, lbl, font))
            (out_dir / taskdir0.name).mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(out_dir / taskdir0.name / mp4_0.name), fps=fps,
                                        codec="libx264", quality=8, macro_block_size=None)
            ncols = per_row if per_row and per_row > 0 else len(panels)
            for i in range(max(len(fr) for fr in frames_list)):
                tiles = [make_panel(fr[min(i, len(fr) - 1)], H, bar)
                         for fr, bar in zip(frames_list, bars)]
                rows = [np.hstack(tiles[r : r + ncols]) for r in range(0, len(tiles), ncols)]
                w_max = max(r.shape[1] for r in rows)                 # short last row → right-pad black
                rows = [r if r.shape[1] == w_max else
                        np.pad(r, ((0, 0), (0, w_max - r.shape[1]), (0, 0))) for r in rows]
                frame = np.vstack(rows)
                # libx264 needs even dims (crop a pixel if odd — width AND height once gridded)
                frame = frame[: frame.shape[0] - frame.shape[0] % 2, : frame.shape[1] - frame.shape[1] % 2]
                writer.append_data(frame)
            writer.close()
            n += 1
    log.info("side-by-side: wrote %d stitched clips → %s", n, out_dir)


def _build_context(pcfg, *, cfg, device, label):
    """Build one model's eval context: policy + FSQ terminator (optionally the checkpoint's CO-TRAINED one)
    + OracleSkillExpertPolicy wrapper + processors (PreserveRawState inserted before Normalizer) + the
    episode-exact oracle data. pcfg carries the model ARCHITECTURE and the shared eval knobs (skill_end_*,
    terminator paths, device). Returns {oracle, pre, post, ep, label}."""
    policy = make_policy(cfg=pcfg, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.eval()
    ft = pcfg.pretrained_path if pcfg.eval_use_trained_terminator else None
    terminator = make_terminator(pcfg.fsq_path, device, dino_path=pcfg.terminator_dino_model_path,
                                 libero_examples_dir=_LIBERO_EXAMPLES, finetuned_ckpt=ft)
    log.info("[%s] terminator: %s", label or "model",
             f"co-trained ({terminator.finetuned_loaded} tensors overridden)"
             if getattr(terminator, "finetuned_loaded", 0) else f"raw FSQ.pt ({pcfg.fsq_path})")
    oracle = OracleSkillExpertPolicy(
        policy, terminator, end_threshold=pcfg.skill_end_threshold,
        progress_threshold=pcfg.skill_end_progress_threshold, end_mode=pcfg.skill_end_mode,
        advance_mode=pcfg.skill_advance_mode, max_skill_len=pcfg.inference_skill_max_length,
        n_action_steps=pcfg.n_action_steps)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=pcfg, pretrained_path=pcfg.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": str(device)},
                                "rename_observations_processor": {"rename_map": cfg.rename_map}})
    norm_idx = next(i for i, s in enumerate(preprocessor.steps) if isinstance(s, NormalizerProcessorStep))
    preprocessor.steps.insert(norm_idx, SkillVLAPreserveRawStateProcessorStep())
    episode_data = load_episode_oracle_data(pcfg.skill_label_dataset_dir, pcfg.eval_init_states_path, cfg.env.task)
    return {"oracle": oracle, "pre": preprocessor, "post": postprocessor, "ep": episode_data, "label": label}


def _model_cfg_from_spec(spec: dict, base):
    """Load one model's ARCHITECTURE config from its checkpoint, then copy the SHARED eval knobs from `base`
    (= cfg.policy, i.e. model 0's config with the CLI overrides): device, skill_end_*, terminator paths,
    n_action_steps, ... — so every model is evaluated under the SAME skill-advance policy. fsq_path and
    skill_label_dataset_dir come from the per-model spec."""
    from lerobot.configs.policies import PreTrainedConfig  # noqa: PLC0415
    mcfg = PreTrainedConfig.from_pretrained(spec["policy_path"])
    mcfg.pretrained_path = spec["policy_path"]
    for f in ("device", "use_amp", "terminator_dino_model_path", "eval_init_states_path",
              "eval_use_trained_terminator", "skill_end_mode", "skill_end_threshold",
              "skill_end_progress_threshold", "skill_advance_mode", "inference_skill_max_length",
              "n_action_steps", "compile_model", "gradient_checkpointing"):
        if hasattr(base, f) and hasattr(mcfg, f):
            setattr(mcfg, f, getattr(base, f))
    mcfg.fsq_path = spec["fsq_path"]
    mcfg.skill_label_dataset_dir = spec["skill_label_dataset_dir"]
    return mcfg


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
    specs = json.loads(os.environ.get("MODELS_JSON", "") or "[]")   # >=2 entries → multi-model side-by-side
    models_per_row = int(os.environ.get("MODELS_PER_ROW", "0") or 0)  # 0 = one row; 3 → 6 models = 2×3 grid
    logging.info("Making environment.")
    envs = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs,
                    trust_remote_code=cfg.trust_remote_code)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
    task_id_to_desc = _libero_task_descriptions(cfg.env.task)
    out = Path(cfg.output_dir)
    common = dict(   # eval_policy_all kwargs shared by every rollout (single or per-model per-task)
        env_preprocessor=env_preprocessor, env_postprocessor=env_postprocessor,
        n_episodes=cfg.eval.n_episodes, max_episodes_rendered=cfg.eval.max_videos_per_task,
        video_frame_stride=cfg.eval.video_frame_stride, video_fps=cfg.eval.video_fps,
        start_seed=cfg.seed, max_parallel_tasks=cfg.env.max_parallel_tasks,
        reference_skill_token_sequences_by_task=None,
        skill_html_train_samples=cfg.eval.skill_html_train_samples,
        skill_html_skill_latents_path=cfg.eval.skill_html_skill_latents_path,
        skill_html_raw_dataset_dir=cfg.eval.skill_html_raw_dataset_dir,
        skill_html_image_key=cfg.eval.skill_html_image_key, task_descriptions=task_id_to_desc,
    )

    if len(specs) >= 2:
        # ── MULTI-model side-by-side: build each model's context, align episodes across models, then PER
        # TASK roll out every model over the SAME scenes (init_state_id rewound between models) and stitch
        # horizontally → side_by_side/{task}/ (streams in task order; no per-model videos kept). ──
        logging.info("Multi-model side-by-side over %d models: %s", len(specs), [s["label"] for s in specs])
        contexts = [_build_context(_model_cfg_from_spec(s, cfg.policy), cfg=cfg, device=device, label=s["label"])
                    for s in specs]
        forced_maps = _align_and_override_multi(envs, [c["ep"] for c in contexts])
        sbs_dir = out / "side_by_side"
        # Per-job scratch: task-split jobs (submit_eval.sh eval_num_gpus>1) share ONE out dir — a fixed
        # "_tmp" would let concurrent jobs delete each other's in-flight videos.
        tmp_root = out / f"_tmp_{os.environ.get('SLURM_JOB_ID') or os.getpid()}"
        tagged = [[] for _ in contexts]
        for tg, grp in envs.items():
            for tid in list(grp.keys()):
                one_env = {tg: {tid: grp[tid]}}
                panels = []
                for i, ctx in enumerate(contexts):
                    _reset_init_state_ids(one_env)                    # rewind → each model sees the same scenes
                    vdir = tmp_root / f"m{i}" / "videos"
                    with torch.no_grad(), (torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext()):
                        info_i = eval_policy_all(
                            envs=one_env, policy=ctx["oracle"], preprocessor=ctx["pre"], postprocessor=ctx["post"],
                            videos_dir=vdir, skill_html_dir=None,
                            forced_skill_token_sequences_by_task={(tg, int(tid)): forced_maps[i][(tg, int(tid))]},
                            **common)
                    panels.append((vdir, ctx["label"]))
                    tagged[i].append((tid, info_i))
                _stitch_models(panels, sbs_dir, per_row=models_per_row)
                shutil.rmtree(tmp_root, ignore_errors=True)           # drop raw per-model videos (keep combined)
                log.info("task %s → side_by_side (%s)", tid,
                         {c["label"]: t[-1][1].get("overall") for c, t in zip(contexts, tagged)})
        info = {c["label"]: _merge(t) for c, t in zip(contexts, tagged)}
        wandb_infos = {f"{c['label']}/": _merge(t) for c, t in zip(contexts, tagged)}
        print("side_by_side done →", sbs_dir)
    else:
        # ── Single model: episode-exact eval over all tasks (env init states overridden once). ──
        logging.info("Making policy + FSQ terminator.")
        ctx = _build_context(cfg.policy, cfg=cfg, device=device, label="")
        forced_by_task = _override_init_states(envs, ctx["ep"])
        logging.info("Episode-exact eval over %d tasks (n_episodes=%d per task).",
                     len(forced_by_task), cfg.eval.n_episodes)
        skill_html_dir = (out / "skill_html") if cfg.eval.skill_html else None
        with torch.no_grad(), (torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext()):
            info = eval_policy_all(
                envs=envs, policy=ctx["oracle"], preprocessor=ctx["pre"], postprocessor=ctx["post"],
                videos_dir=out / "videos", skill_html_dir=skill_html_dir,
                forced_skill_token_sequences_by_task=forced_by_task, **common)
        print("Overall:", info["overall"])
        wandb_infos = {"": info}

    close_envs(envs)
    out.mkdir(parents=True, exist_ok=True)
    # Task-split jobs share this out dir → suffix the summary per chunk (TASK_TAG, e.g. "t0-4") so
    # concurrent jobs don't clobber each other's eval_info.json.
    task_tag = os.environ.get("TASK_TAG", "").strip()
    with open(out / (f"eval_info_{task_tag}.json" if task_tag else "eval_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    _maybe_log_wandb(cfg, wandb_infos)


def _maybe_log_wandb(cfg, infos: dict) -> None:
    """infos: {prefix: eval_info}. Single model → {"": info}; multi-model → {"plain/":.., "weighted/":..}
    (each model's metrics logged to ONE run under its label prefix)."""
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
