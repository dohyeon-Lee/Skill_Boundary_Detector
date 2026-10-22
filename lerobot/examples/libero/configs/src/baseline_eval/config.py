"""Portable MolmoAct2/GR00T evaluation settings; no model imports at submission."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import sys

CONFIGS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CONFIGS / "src"))
from global_config_loader import load_global_config, read_yaml
from groot_action_contract import groot_action_decode_transform

FAMILIES = {
    "molmoact2": (".venv_molmoact2_lerobot", "third_party/lerobot_molmoact2", "sbd_molmoact2.json"),
    "groot": (".venv_groot_lerobot", "third_party/lerobot_groot", "sbd_groot.json"),
}


def load_config(path: Path) -> dict:
    global_path = next((p / "global_config.yaml" for p in path.resolve().parents
                        if (p / "global_config.yaml").is_file()), None)
    if global_path is None:
        raise FileNotFoundError(f"No global_config.yaml above {path}")
    return {**load_global_config(global_path), **read_yaml(path)}


def name(value, field):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value in {".", ".."}:
        raise ValueError(f"{field} must be a single safe folder name: {value!r}")
    return value


def integer(value, field, minimum=1, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{field} must be an integer >= {minimum}" + (f" and <= {maximum}" if maximum else ""))
    return value


def boolean(value, field):
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be true or false")
    return value


def checkpoint_step(value):
    if str(value).lower() == "last":
        return "last"
    if isinstance(value, bool) or not str(value).isdigit():
        raise ValueError(f"Invalid checkpoint: {value!r}; use a step or last")
    return f"{int(value):06d}"


def checkpoint_contract(path: Path, family: str) -> dict:
    missing = [f for f in ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json", FAMILIES[family][2])
               if not (path / f).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete {family} LeRobot checkpoint at {path}: missing {missing}")
    policy = json.loads((path / "config.json").read_text())
    if policy.get("type") != family:
        raise ValueError(f"Expected {family}, found {policy.get('type')!r}: {path}")
    provenance = json.loads((path / FAMILIES[family][2]).read_text())
    grounding = provenance.get("proprio_grounding")
    if grounding not in {"none", "episode_start_xyz"}:
        raise ValueError(f"Missing/unsupported rollout grounding in {path}: {grounding!r}")
    state = policy.get("input_features", {}).get("observation.state", {}).get("shape")
    action = policy.get("output_features", {}).get("action", {}).get("shape")
    if state != [8] or action != [7]:
        raise ValueError(f"This LIBERO adapter requires state=8, delta-EEF action=7; got {state}, {action}")
    if int(policy.get("n_obs_steps", 1)) != 1:
        raise ValueError("This baseline evaluator currently supports the trained one-observation setup only")
    for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
        pipeline = json.loads((path / filename).read_text())
        for step in pipeline.get("steps", []):
            state_file = step.get("state_file")
            if state_file and not (path / state_file).is_file():
                raise FileNotFoundError(f"Missing saved processor statistics: {path / state_file}")
    contract = {"grounding": grounding, "chunk_size": int(policy["chunk_size"])}
    if family == "groot":
        contract["action_decode_transform"] = groot_action_decode_transform(path)
    return contract


def build_settings(config: dict, *, work_dir: Path) -> dict:
    family = config.get("baseline")
    if family not in FAMILIES:
        raise ValueError(f"baseline must be one of {list(FAMILIES)}")
    root = Path(config["project_root"]).resolve()
    runtime = config.get("runtime", {})
    default_venv, default_source, _ = FAMILIES[family]
    resolve = lambda p: str((root / p).resolve())
    model_python = Path(resolve(runtime.get("venv", default_venv))) / "bin/python"
    main_python = Path(resolve(runtime.get("sim_venv", ".venv"))) / "bin/python"
    source = Path(resolve(runtime.get("source", default_source)))
    for path in (model_python, main_python, source / f"src/lerobot/policies/{family}"):
        if not path.exists():
            raise FileNotFoundError(f"Evaluation runtime missing: {path}; run setup_env.sh first")
    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list")
    defaults = config.get("model_defaults", {})
    panels, labels, sequences = [], set(), []
    playback = integer(config.get("n_action_steps", 5), "n_action_steps")
    for model in models:
        unknown = set(model) - {"model_dir", "checkpoint", "label", "stage"}
        if unknown:
            raise ValueError(f"Unknown model fields: {sorted(unknown)}")
        run = name(model["model_dir"], "model_dir")
        label = name(model.get("label", run), "label")
        if label in labels:
            raise ValueError(f"Duplicate model label: {label}")
        labels.add(label)
        stage = str(model.get("stage", defaults.get("stage", "PT"))).upper()
        if stage not in {"PT", "FT"}:
            raise ValueError("stage must be PT or FT")
        steps = model.get("checkpoint", defaults.get("checkpoint", "last"))
        steps = [checkpoint_step(s) for s in (steps if isinstance(steps, list) else [steps])]
        if not steps or len(set(steps)) != len(steps):
            raise ValueError("checkpoint list must be non-empty, without duplicates")
        sequences.append(steps)
        run_dir = root / config.get("outputs_root", "outputs_filtered") / f"{family}_{stage}" / run
        for checkpoint_index, step in enumerate(steps):
            if step == "last":
                candidates = sorted((p for p in (run_dir / "checkpoints").glob("*")
                                     if p.name.isdigit() and p.is_dir()), key=lambda p: int(p.name), reverse=True)
                # Never silently select an older save while the latest is incomplete.
                if not candidates:
                    raise FileNotFoundError(f"No saved checkpoints: {run_dir / 'checkpoints'}")
                step = candidates[0].name
            policy_path = run_dir / "checkpoints" / step / "pretrained_model"
            contract = checkpoint_contract(policy_path, family)
            if playback > contract["chunk_size"]:
                raise ValueError(f"n_action_steps={playback} exceeds chunk_size={contract['chunk_size']}")
            panel_label = f"{label} | ckpt {step}" if len(steps) > 1 else label
            panel_dir = f"{len(panels):02d}_" + re.sub(r"[^A-Za-z0-9_.-]", "-", panel_label)
            panels.append(dict(label=panel_label, model_dir=run, checkpoint=step, stage=stage, checkpoint_index=checkpoint_index,
                               policy_path=str(policy_path), panel_dir=panel_dir, **contract))
    if any(seq != sequences[0] for seq in sequences[1:]):
        raise ValueError("Use the same ordered checkpoint list for every model (model_defaults.checkpoint)")
    # Match the checkpoint-major layout used by pi05/SkillVLA video comparisons.
    panels.sort(key=lambda p: p["checkpoint_index"])
    for index, panel in enumerate(panels):
        panel.pop("checkpoint_index")
        panel["panel_dir"] = f"{index:02d}_" + re.sub(r"[^A-Za-z0-9_.-]", "-", panel["label"])
    task_ids = config.get("task_ids", list(range(10)))
    if not isinstance(task_ids, list) or not task_ids:
        raise ValueError("task_ids must be a non-empty list")
    task_ids = [integer(i, "task_ids", 0) for i in task_ids]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task_ids contains duplicates")
    suite = config.get("target_task", "libero_90")
    suite_counts = {"libero_90": 90, "libero_10": 10, "libero_spatial": 10, "libero_object": 10, "libero_goal": 10}
    if suite not in suite_counts or max(task_ids) >= suite_counts[suite]:
        raise ValueError(f"Invalid LIBERO suite/task_ids: {suite} {task_ids}")
    # A policy and its processor have episode-local state. Do not share them between task threads.
    if config.get("max_parallel_tasks", 1) != 1:
        raise ValueError("max_parallel_tasks must be 1; use eval_num_gpus/workers for parallelism")
    oracle = config.get("oracle", {})
    exact = boolean(oracle.get("episode_exact", False), "oracle.episode_exact")
    repeat = boolean(oracle.get("repeat_episodes", False), "oracle.repeat_episodes")
    offset = integer(config.get("episode_offset", 0), "episode_offset", 0)
    init_path = ""
    if repeat and not exact:
        raise ValueError("repeat_episodes requires episode_exact")
    if exact:
        if offset:
            raise ValueError("episode_exact requires episode_offset=0")
        source_name = name(oracle.get("dataset_source", ""), "oracle.dataset_source")
        init_path = resolve(str(Path(config.get("dataset_root", "dataset_filtered")) / "skillvla_dataset" / source_name / "eval_init_states.npz"))
        if not Path(init_path).is_file():
            raise FileNotFoundError(init_path)
    video = config.get("video", {})
    slurm = config.get("slurm", {})
    gres = str(slurm.get("gres", "gpu:1"))
    if not gres.startswith("gpu:") or gres.rsplit(":", 1)[-1] != "1":
        raise ValueError("Each eval job allocates one GPU; use slurm.gres: gpu:1 and eval_num_gpus for parallelism")
    output = name(config.get("output_name") or f"{family}_{suite}_eval", "output_name")
    return dict(
        baseline=family, project_root=str(root), model_python=str(model_python), sim_python=str(main_python),
        model_source=str(source), eval_work_dir=str(work_dir), models_json=json.dumps(panels),
        models_labels=json.dumps([p["label"] for p in panels]), panel_count=len(panels),
        panel_summary=", ".join(p["label"] for p in panels), target_task=suite, task_ids=json.dumps(task_ids),
        eval_expected_tasks=len(task_ids), n_episodes=integer(config.get("n_episodes", 10), "n_episodes"),
        eval_batch_size=integer(config.get("eval_batch_size", 1), "eval_batch_size"),
        n_action_steps=playback, episode_offset=offset, episode_exact=exact, episode_exact_repeat=repeat,
        eval_init_states_path=init_path, seed=integer(config.get("seed", 1000), "seed", 0),
        eval_resume=boolean(config.get("resume", True), "resume"), eval_out_dir=str(work_dir / "outputs" / output),
        eval_num_gpus=integer(config.get("eval_num_gpus", 1), "eval_num_gpus"),
        eval_max_workers_per_gpu=integer(config.get("eval_max_workers_per_gpu", 1), "eval_max_workers_per_gpu", 1, 5),
        video_enable=boolean(video.get("enable", True), "video.enable"),
        max_videos_per_task=integer(video.get("max_per_task", 3), "video.max_per_task", 0) if video.get("enable", True) else 0,
        video_show_wrist=boolean(video.get("show_wrist", False), "video.show_wrist"),
        video_frame_stride=integer(video.get("frame_stride", 2), "video.frame_stride"),
        video_fps=integer(video.get("fps", 10), "video.fps"),
        grid_columns=integer(video.get("grid_columns", 0), "video.grid_columns", 0),
        worker_timeout=integer(config.get("worker_timeout", 900), "worker_timeout"),
        eval_partition=",".join(config.get("train_partition", ["debug"])), eval_qos=config.get("train_qos", "base_qos"),
        eval_nodelist=config.get("train_nodelist", ""), eval_exclude_nodes=",".join(config.get("train_exclude_nodes", [])),
        eval_gres=gres, eval_cpus_per_task=integer(slurm.get("cpus", 8), "slurm.cpus"),
        eval_mem=slurm.get("memory", "128G"), eval_time=slurm.get("time", "6:00:00"),
        wandb_enable=boolean(config.get("logging", {}).get("wandb", {}).get("enable", True), "logging.wandb.enable"),
        wandb_project=config.get("logging", {}).get("wandb", {}).get("project", "VLA_eval"), wandb_run_name=output,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    settings = build_settings(load_config(args.config), work_dir=args.work_dir or args.config.parent)
    if args.shell:
        for key, value in settings.items():
            value = str(value).lower() if isinstance(value, bool) else str(value)
            print(f"export {key.upper()}={shlex.quote(value)}")
    else:
        print(json.dumps(settings, indent=2))


if __name__ == "__main__":
    main()
