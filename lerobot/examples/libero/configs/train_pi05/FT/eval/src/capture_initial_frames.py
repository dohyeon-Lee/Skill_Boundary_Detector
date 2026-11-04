#!/usr/bin/env python3
"""Capture initial agentview scenes using pi05 eval's reset/settling/render path.

No policy is loaded and no rollout is performed. The default source follows the
eval YAML (benchmark init states, or episode-exact dataset states). Random mode
instead samples the BDDL's placement distribution; it does NOT widen its bounds.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EVAL_DIR.parents[6]
sys.path.insert(0, str(EVAL_DIR / "../../PT/src"))
from train_pi05_config import as_bool, load_config, resolve_path
sys.path.insert(0, str(EVAL_DIR / "../../PT/eval/src"))
from random_initial_states import prepare_bank, range_scale, range_tag


def capture_output_dir(label, scale, explicit=None):
    if explicit is not None:
        return Path(explicit).resolve()
    label = re.sub(r"[^\w.-]+", "_", label)
    tag = f"range{range_tag(scale)}"
    if scale > 1 and label != tag and not label.endswith(f"_{tag}"):
        label += f"_{tag}"
    return (EVAL_DIR / "initial_frames" / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{label}").resolve()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EVAL_DIR / "pi05_eval_config.yaml")
    parser.add_argument("--label", default="baseline", help="Separate comparison run label")
    parser.add_argument("--output-dir", type=Path, help="New/empty directory; default: initial_frames/<timestamp>_<label>")
    parser.add_argument("--init-source", choices=("config", "benchmark", "random"), default="config")
    parser.add_argument("--random-range", type=float, help="Override YAML random_range; >1 uses the same expanded bank as eval")
    parser.add_argument("--task-ids", type=int, nargs="+", help="Override YAML task_ids")
    parser.add_argument("--episodes", type=int, help="Override YAML n_episodes")
    parser.add_argument("--episode-offset", type=int, help="Override benchmark init-state offset")
    parser.add_argument("--seed", type=int, default=1000, help="Eval default: 1000; episode k uses seed+k")
    parser.add_argument("--settle-steps", type=int, default=10, help="Same default as LiberoEnv.reset")
    parser.add_argument("--size", type=int, default=360, help="Square image size; pi05 eval default: 360")
    parser.add_argument("--libero-root", type=Path, default=PROJECT_ROOT / "tools/lerobot-libero/libero/libero")
    parser.add_argument("--bddl-root", type=Path, help="Alternate BDDL directory, e.g. widened placement ranges")
    parser.add_argument("--init-states-root", type=Path, help="Alternate benchmark init_files directory")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings without creating a simulator")
    return parser.parse_args()


def exact_states(path, np):
    grouped = {}
    with np.load(path, allow_pickle=True) as data:
        poses = data["model_body_poses"] if "model_body_poses" in data else [None] * len(data["episode_index"])
        for ep, state, scene, pose in zip(data["episode_index"], data["init_states"], data["scene_file"], poses, strict=True):
            grouped.setdefault(str(scene).removesuffix("_demo.hdf5"), []).append(
                (int(ep), np.asarray(state, dtype=np.float64), json.loads(str(pose)) if pose is not None else None)
            )
    return {scene: sorted(rows, key=lambda row: row[0]) for scene, rows in grouped.items()}


def save_predictor_input(path, observation):
    """Same image orientation and 8D world state as closed-loop LIBERO processing."""
    import numpy as np
    import torch
    from lerobot.processor.env_processor import LiberoProcessorStep

    robot = observation["robot_state"]
    state = LiberoProcessorStep().observation({"observation.robot_state": {
        "eef": {key: torch.as_tensor(robot["eef"][key])[None] for key in ("pos", "quat")},
        "gripper": {"qpos": torch.as_tensor(robot["gripper"]["qpos"])[None]},
    }})["observation.state"][0].numpy()
    np.savez_compressed(path,
                        top=np.ascontiguousarray(observation["pixels"]["image"][::-1, ::-1]),
                        wrist=np.ascontiguousarray(observation["pixels"]["image2"][::-1, ::-1]),
                        state_world=state, contract="libero_processed_rgb_state_v1")


def write_gallery(output, settings, records, Image, ImageDraw):
    """One row per task, one column per episode, plus full-resolution originals."""
    columns = settings["n_episodes"]
    thumb, caption, task_header = 240, 24, 28
    task_ids = settings["task_ids"]
    sheet = Image.new("RGB", (columns * thumb, len(task_ids) * (thumb + caption + task_header)), "white")
    draw = ImageDraw.Draw(sheet)
    sections = []
    for row, task_id in enumerate(task_ids):
        frames = [record for record in records if record["task_id"] == task_id]
        y = row * (thumb + caption + task_header)
        title = f"Task {task_id}: {frames[0]['task_name']}"
        draw.text((5, y + 5), title, fill="black")
        cards = []
        for col, record in enumerate(frames):
            with Image.open(output / record["image"]) as original:
                sheet.paste(original.resize((thumb, thumb)), (col * thumb, y + task_header))
            text = f"ep {record['episode']} | seed {record['seed']}"
            draw.text((col * thumb + 5, y + task_header + thumb + 3), text, fill="black")
            url = record["image"]
            cards.append(f'<figure><a href="{url}"><img loading="lazy" src="{url}"></a>'
                         f'<figcaption>{html.escape(text)}<br>init index: {record["init_state_index"]}</figcaption></figure>')
        sections.append(f'<h2>{html.escape(title)}</h2><div class="row">{"".join(cards)}</div>')
    sheet.save(output / "overview.png")
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Initial top-view scenes</title>'
        '<style>body{font-family:system-ui;margin:24px;background:#f5f5f5}.row{display:flex;overflow-x:auto;gap:8px}'
        'figure{margin:0;flex:0 0 240px}img{width:240px}figcaption{font-size:13px}pre{white-space:pre-wrap}</style>'
        f'<h1>{html.escape(settings["label"])} — initial top-view frames</h1>'
        '<p><a href="overview.png">All tasks PNG</a> · <a href="manifest.json">Settings / per-frame metadata</a></p>'
        f'<pre>{html.escape(json.dumps(settings, ensure_ascii=False, indent=2))}</pre>' + "".join(sections),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    oracle = config.get("oracle") or {}
    scale = range_scale(args.random_range if args.random_range is not None else config.get("random_range", 1.0))
    source = args.init_source
    if source == "config":
        source = "exact" if as_bool(oracle.get("episode_exact", False)) else "benchmark"
    if scale > 1:
        if source == "exact" or args.init_source == "benchmark":
            raise ValueError("random_range > 1 requires episode_exact=false; cannot use benchmark override")
        source = "range"
    tasks = args.task_ids if args.task_ids is not None else config.get("task_ids")
    if tasks is None:
        tasks = list(range(10))
    episodes = args.episodes if args.episodes is not None else int(config.get("n_episodes", 10))
    offset = args.episode_offset if args.episode_offset is not None else int(config.get("episode_offset", 0))
    if episodes < 1 or not tasks or len(tasks) != len(set(tasks)) or offset < 0 or args.size < 1 or args.settle_steps < 0:
        raise ValueError("Need unique task IDs, positive episodes/size, and nonnegative offset/settle steps")
    if source != "benchmark" and offset:
        raise ValueError("episode-offset applies only to benchmark init states")
    suite_name = str(config.get("target_task", "libero_10"))
    if "," in suite_name:
        raise ValueError("Capture one task suite at a time")
    root = args.libero_root.resolve()
    paths = {
        "benchmark_root": str(root),
        "bddl_files": str((args.bddl_root or root / "bddl_files").resolve()),
        "init_states": str((args.init_states_root or root / "init_files").resolve()),
        "assets": str(root / "assets"),
        "datasets": str(root.parent / "datasets"),
    }
    exact_path = None
    if source == "exact":
        dataset = str(oracle.get("dataset_source") or "").strip()
        if not dataset:
            raise ValueError("Episode-exact capture needs oracle.dataset_source")
        dataset_root = Path(resolve_path(config.get("project_root", PROJECT_ROOT), config.get("dataset_root", "dataset")))
        exact_path = dataset_root / "skillvla_dataset" / dataset / "eval_init_states.npz"
    settings = {
        "label": args.label, "config": str(args.config.resolve()), "target_task": suite_name,
        "task_ids": tasks, "n_episodes": episodes, "init_source": source, "episode_offset": offset,
        "seed": args.seed, "settle_steps": args.settle_steps, "image_size": args.size,
        "random_range": scale,
        "model_input_contract": "libero_processed_rgb_state_v1",
        "camera": "agentview_image", "orientation": "LiberoEnv.render (same as eval videos)",
        "exact_archive": str(exact_path) if exact_path else None,
        "repeat_episodes": as_bool(oracle.get("repeat_episodes", False)),
        "libero_paths": paths, "renderer": os.environ.get("MUJOCO_GL", "osmesa"),
    }
    print(json.dumps(settings, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    if os.environ["MUJOCO_GL"] in {"osmesa", "egl"}:
        os.environ["PYOPENGL_PLATFORM"] = os.environ["MUJOCO_GL"]
    import numpy as np
    import torch
    import yaml
    from PIL import Image, ImageDraw

    output = capture_output_dir(args.label, scale, args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "eval_config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    # Private LIBERO config: never overwrite ~/.libero or another running job's config.
    with tempfile.TemporaryDirectory(prefix="pi05_initial_frames_") as local_config:
        Path(local_config, "config.yaml").write_text(yaml.safe_dump(paths), encoding="utf-8")
        os.environ["LIBERO_CONFIG_PATH"] = local_config
        from libero.libero import benchmark
        from lerobot.envs.libero import LiberoEnv

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        suite = benchmark.get_benchmark_dict()[suite_name]()
        if source == "range":
            exact_path = prepare_bank(suite_name=suite_name, task_ids=tasks, episodes=episodes,
                                      scale=scale, seed=args.seed, bank_root=EVAL_DIR / "initial_state_banks")
            settings["exact_archive"] = str(exact_path)
        exact = exact_states(exact_path, np) if exact_path else {}
        records = []
        manifest = {"settings": settings, "frames": records, "complete": False}
        for task_id in tasks:
            task = suite.get_task(task_id)
            matched = exact.get(str(task.name), [])
            if source in {"exact", "range"} and (not matched or (len(matched) < episodes and not settings["repeat_episodes"])):
                raise ValueError(f"Task {task_id}: only {len(matched)} exact episodes; cannot capture {episodes}")
            bddl = Path(paths["bddl_files"]) / task.problem_folder / task.bddl_file
            env = LiberoEnv(
                task_suite=suite, task_id=task_id, task_suite_name=suite_name,
                observation_width=args.size, observation_height=args.size,
                obs_type="pixels_agent_pos",
                init_states=source == "benchmark", episode_index=offset,
                num_steps_wait=args.settle_steps,
            )
            try:
                if source in {"exact", "range"}:
                    env.init_states = True
                    env._init_states = np.stack([matched[i % len(matched)][1] for i in range(episodes)])
                    env._init_model_body_poses = [matched[i % len(matched)][2] for i in range(episodes)]
                folder = output / f"task{task_id:02d}"
                folder.mkdir()
                for episode in range(episodes):
                    index = env.init_state_id % len(env._init_states) if env.init_states else None
                    observation, _ = env.reset(seed=args.seed + episode)
                    relative = f"task{task_id:02d}/episode{episode:03d}.png"
                    Image.fromarray(np.ascontiguousarray(env.render())).save(output / relative)
                    model_input = f"task{task_id:02d}/episode{episode:03d}_model_input.npz"
                    save_predictor_input(output / model_input, observation)
                    record = {
                        "task_id": task_id, "task_name": str(task.name), "episode": episode,
                        "task_description": str(task.language), "model_input": model_input,
                        "seed": args.seed + episode, "init_state_index": index, "image": relative,
                        "dataset_episode": matched[episode % len(matched)][0] if source == "exact" else None,
                        "bddl_file": str(bddl), "bddl_sha256": hashlib.sha256(bddl.read_bytes()).hexdigest(),
                    }
                    if hasattr(env, "get_goal_predicate_diagnostics"):
                        record["object_positions"] = env.get_goal_predicate_diagnostics()["object_positions"]
                    records.append(record)
                    print(f"Task {task_id}: {episode + 1}/{episodes} saved", flush=True)
                (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            finally:
                env.close()
        write_gallery(output, settings, records, Image, ImageDraw)
        manifest["complete"] = True
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"HTML: {output / 'index.html'}\nPNG:  {output / 'overview.png'}", flush=True)


if __name__ == "__main__":
    main()
