#!/usr/bin/env python
"""Side-by-side comparison eval for cycle-PT models — incremental, task-by-task.

Loads N policies ONCE, then for each LIBERO task id:
  1. rolls out every model on IDENTICAL init states (fresh env per model, same seed/offset),
  2. immediately stitches each episode's videos side-by-side with a label banner
     (green = success, red = fail) → {out_dir}/videos/task{ID:02d}_ep{E}.mp4,
  3. updates {out_dir}/compare_summary.json.
So the first task's comparison video is watchable while later tasks are still running.

Models come as JSON (mirrors the other-server video_compare convention):
  --models '[{"model_dir": "PTcyc_..._g8p250", "label": "cyc250"},
             {"model_dir": "PTiid_..._g8p250", "label": "iid250"}]'
"""

import argparse
import json
import logging
from contextlib import nullcontext
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

RENAME_MAP = {"observation.images.image2": "observation.images.wrist_image"}
BANNER_H = 26


def load_model(models_root: Path, model_dir: str, checkpoint: str, n_action_steps: int, env_cfg):
    # model_dir may be absolute (e.g., a run under pi05_PT) — lets a known-good reference
    # model ride through the exact same pipeline as the cycle models.
    base = Path(model_dir) if Path(model_dir).is_absolute() else models_root / model_dir
    path = base / "checkpoints" / checkpoint / "pretrained_model"
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    policy_cfg = PreTrainedConfig.from_pretrained(str(path))
    policy_cfg.pretrained_path = path
    # Load & PARK on CPU — N policies don't fit a 24GB GPU together; each model is moved to
    # cuda only for its own rollout turn (swap ≈ seconds, negligible vs rollout time).
    policy_cfg.device = "cpu"
    policy_cfg.n_action_steps = n_action_steps
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=RENAME_MAP)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(path),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},  # batches go to GPU; policy is swapped in
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    return {"policy": policy, "cfg": policy_cfg, "pre": pre, "post": post,
            "env_pre": env_pre, "env_post": env_post}


def extract_task_metrics(info: dict, tid: int) -> tuple[list, list]:
    """(successes, video_paths) for task tid from eval_policy_all's result dict."""
    for t in info.get("per_task", []):
        if int(t.get("task_id", -1)) == tid:
            m = t.get("metrics", {})
            return list(m.get("successes", [])), list(m.get("video_paths", []))
    return [], []


def banner(width: int, label: str, success: bool | None) -> np.ndarray:
    color = (60, 60, 60) if success is None else ((30, 120, 40) if success else (150, 40, 40))
    img = Image.new("RGB", (width, BANNER_H), color)
    d = ImageDraw.Draw(img)
    text = label if success is None else f"{label}  {'success' if success else 'fail'}"
    d.text((6, 6), text, fill=(255, 255, 255))
    return np.asarray(img)


def stitch_episode(videos: list[tuple[str, Path, bool]], out_path: Path, fps_fallback: int = 30):
    """videos: [(label, path, success)] → horizontal concat with per-model banner."""
    clips, fps = [], fps_fallback
    for _, path, _ in videos:
        reader = imageio.get_reader(str(path))
        fps = reader.get_meta_data().get("fps", fps_fallback)
        clips.append([np.asarray(f) for f in reader])
        reader.close()
    n = max(len(c) for c in clips)
    with imageio.get_writer(str(out_path), fps=fps, macro_block_size=1) as w:
        for i in range(n):
            cols = []
            for (label, _, success), clip in zip(videos, clips):
                frame = clip[min(i, len(clip) - 1)]
                cols.append(np.vstack([banner(frame.shape[1], label, success), frame]))
            h = max(c.shape[0] for c in cols)
            cols = [np.pad(c, ((0, h - c.shape[0]), (0, 0), (0, 0))) for c in cols]
            w.append_data(np.hstack(cols))


def draw_chart(summary: dict, out_path: Path):
    """Per-task grouped bars (model vs model) + overall bars. Redrawn after every task."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = summary["tasks"]
    labels = list(summary["models"].keys())
    if not tasks:
        return
    cmap = plt.get_cmap("tab10")
    n_t, n_m = len(tasks), len(labels)
    fig, axes = plt.subplots(1, 2, figsize=(11, max(3.0, 0.32 * n_t + 1)),
                             gridspec_kw={"width_ratios": [3, 1]})
    y = np.arange(n_t)
    h = 0.8 / n_m
    for j, lbl in enumerate(labels):
        vals = [t["success_rate"].get(lbl, 0.0) for t in tasks]
        axes[0].barh(y + (j - (n_m - 1) / 2) * h, vals, height=h, color=cmap(j % 10), label=lbl)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"t{t['task_id']:02d}" for t in tasks], fontsize=7)
    axes[0].invert_yaxis(); axes[0].set_xlim(0, 1)
    axes[0].grid(axis="x", alpha=0.25); axes[0].legend(fontsize=8)
    axes[0].set_title("success by task")
    overall = [float(np.mean([t["success_rate"].get(lbl, 0.0) for t in tasks])) for lbl in labels]
    axes[1].bar(labels, overall, color=[cmap(j % 10) for j in range(n_m)])
    axes[1].set_ylim(0, 1); axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_title(f"overall ({n_t} tasks)")
    for xi, v in enumerate(overall):
        axes[1].text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help='JSON: [{"model_dir":..., "label":...}, ...]')
    ap.add_argument("--models_root", type=Path, required=True)
    ap.add_argument("--checkpoint", default="020000")
    ap.add_argument("--suite", default="libero_90")
    ap.add_argument("--task_ids", required=True, help="JSON list of env task ids")
    ap.add_argument("--n_episodes", type=int, default=5)
    ap.add_argument("--init_state_offset", type=int, default=25)
    ap.add_argument("--n_action_steps", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out_dir", type=Path, required=True)
    # sharding (SLURM array): shard k of n takes task_ids[k::n]; merge_compare_summaries.py joins
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    args = ap.parse_args()

    init_logging()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    models_spec = json.loads(args.models)
    task_ids = [int(t) for t in json.loads(args.task_ids)]
    tag = ""
    if args.n_shards > 1:
        task_ids = task_ids[args.shard :: args.n_shards]  # round-robin → shards get similar mix
        tag = f"_shard{args.shard}"
        logging.info(f"shard {args.shard}/{args.n_shards} → tasks {task_ids}")
    summary_path = args.out_dir / f"compare_summary{tag}.json"
    chart_path = args.out_dir / f"compare_chart{tag}.png"
    videos_out = args.out_dir / "videos"
    videos_out.mkdir(parents=True, exist_ok=True)

    def env_cfg_for(tids: list[int]) -> LiberoEnv:
        return LiberoEnv(task=args.suite, task_ids=tids, init_state_offset=args.init_state_offset)

    logging.info(f"Loading {len(models_spec)} models (once)")
    models = []
    for spec in models_spec:
        ckpt = str(spec.get("checkpoint", args.checkpoint))  # per-model override (e.g., ref@015000)
        m = load_model(args.models_root, spec["model_dir"], ckpt,
                       args.n_action_steps, env_cfg_for(task_ids[:1]))
        m["label"] = spec["label"]
        models.append(m)
        logging.info(f"  loaded [{spec['label']}] {spec['model_dir']} @ {ckpt}")

    summary = {"checkpoint": args.checkpoint, "suite": args.suite,
               "models": {m["label"]: s["model_dir"] for m, s in zip(models, models_spec)},
               "tasks": []}

    for tid in task_ids:
        row = {"task_id": tid, "success_rate": {}, "successes": {}}
        per_model_videos: dict[str, list] = {}
        for m in models:
            # fresh env per model → both see the exact same deterministic init-state sequence
            envs = make_env(env_cfg_for([tid]), n_envs=args.batch_size)
            m["policy"].to("cuda")  # swap in (parked on CPU; N models don't fit 24GB together)
            amp = nullcontext()
            if getattr(m["cfg"], "use_amp", False):
                amp = torch.autocast(device_type="cuda")
            with torch.no_grad(), amp:
                info = eval_policy_all(
                    envs=envs, policy=m["policy"],
                    env_preprocessor=m["env_pre"], env_postprocessor=m["env_post"],
                    preprocessor=m["pre"], postprocessor=m["post"],
                    n_episodes=args.n_episodes,
                    max_episodes_rendered=args.n_episodes,
                    videos_dir=args.out_dir / "tmp" / m["label"] / f"task{tid:02d}",
                    start_seed=args.seed,
                )
            if len(models) > 1:
                m["policy"].to("cpu")  # swap out (single-model runs just stay on GPU)
                torch.cuda.empty_cache()
            close_envs(envs)
            succs, vids = extract_task_metrics(info, tid)
            row["successes"][m["label"]] = [bool(s) for s in succs]
            row["success_rate"][m["label"]] = float(np.mean(succs)) if succs else 0.0
            per_model_videos[m["label"]] = vids

        # stitch episode-by-episode, right now (incremental visibility)
        n_vids = min(len(v) for v in per_model_videos.values()) if per_model_videos else 0
        for e in range(n_vids):
            stitch_episode(
                [(m["label"], Path(per_model_videos[m["label"]][e]),
                  row["successes"][m["label"]][e] if e < len(row["successes"][m["label"]]) else None)
                 for m in models],
                videos_out / f"task{tid:02d}_ep{e}.mp4",
            )

        summary["tasks"].append(row)
        summary_path.write_text(json.dumps(summary, indent=2))
        draw_chart(summary, chart_path)
        rates = "  ".join(f"{lbl}={r:.2f}" for lbl, r in row["success_rate"].items())
        logging.info(f"[task {tid:02d}] {rates}  → {n_vids} stitched videos")

    # final per-model overall
    overall = {m["label"]: float(np.mean([t["success_rate"][m["label"]] for t in summary["tasks"]]))
               for m in models}
    summary["overall"] = overall
    summary_path.write_text(json.dumps(summary, indent=2))
    logging.info("OVERALL  " + "  ".join(f"{k}={v:.3f}" for k, v in overall.items()))


if __name__ == "__main__":
    main()
