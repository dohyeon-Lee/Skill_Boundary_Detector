"""Run native pi05 in Coffee, merge standard metrics, and make a small video HTML."""
from pathlib import Path
import html
import json
import os
import subprocess
import sys


def main():
    env = os.environ
    policy = Path(env["POLICY_PATH"])
    if not (policy / "model.safetensors").is_file() and not (policy / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"FT checkpoint not ready: {policy}")
    out = Path(env["EVAL_OUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    video = json.loads(env["VIDEO_JSON"])
    child_env = {**env, "TASK_TAG": "t0", "LEROBOT_EVAL_PANEL_LABEL": "pi05",
                 "LEROBOT_EVAL_METRICS_DIR": str(out / "src/metrics"),
                 "LEROBOT_EVAL_ANNOTATE_VIDEOS": "1",
                 "LEROBOT_EVAL_VIDEO_WRIST": "1" if video["show_wrist"] else "0"}
    args = {
        "policy.path": str(policy), "policy.tokenizer_path": env["TOKENIZER_PATH"],
        "policy.n_action_steps": int(env["N_ACTION_STEPS"]), "policy.device": "cuda",
        "policy.compile_model": False, "policy.gradient_checkpointing": False,
        "env.type": "mimicgen", "env.task": "Coffee_D0", "env.task_ids": [0],
        "env.dataset_dir": env["DATASET_DIR"], "env.simulator_python": env["SIMULATOR_PYTHON"],
        "env.init_states": False, "env.episode_length": int(env["MAX_EPISODE_STEPS"]),
        "env.max_parallel_tasks": 1, "eval.batch_size": 1,
        "eval.n_episodes": int(env["N_EPISODES"]), "eval.use_async_envs": False,
        "eval.init_states_path": env["INIT_STATES_PATH"],
        "eval.max_videos_per_task": video["max_per_task"],
        "rename_map": {"observation.images.image2": "observation.images.wrist_image"},
        "output_dir": str(out / "src"), "job_name": env["RUN_NAME"], "seed": int(env["SEED"]),
    }
    (out / "src").mkdir(exist_ok=True)
    provenance = {"arguments": args, "repeat_episodes": env["REPEAT_EPISODES"],
                  "checkpoint": str(policy)}
    previous = out / "src/evaluation_config.json"
    if previous.exists() and json.loads(previous.read_text()) != provenance:
        raise ValueError("Evaluation settings changed; use a different output folder")
    previous.write_text(json.dumps(provenance, indent=2))
    if env.get("SLURM_JOB_ID") or env.get("SBD_LOCAL_JOB_ID"):
        from lerobot.utils.eval_progress import publish

        progress_env = {**env, "EVAL_OUT_DIR": str(out / "src"), "TARGET_TASK": "Coffee_D0",
                        "STAGE1_EVAL_WORK_DIR": str(Path(__file__).resolve().parents[1]),
                        "EVAL_BATCH_SIZE": "1", "SBD_EVAL_EPISODE_PROGRESS": "1",
                        "EVAL_GPU_GROUPS_JSON": '[[{"items":[0],"tag":"t0"}]]'}
        publish(progress_env, env.get("SBD_TRAINING_PARENT_ID", env.get("SLURM_JOB_ID", "local")))
    command = [sys.executable, "-m", "lerobot.scripts.lerobot_eval"]
    command += [f"--{key}={json.dumps(value) if not isinstance(value, str) else value}" for key, value in args.items()]
    subprocess.run(command, env=child_env, check=True)
    merge = Path(env["CODE_ROOT"]) / "lerobot/skillVLA/libero/configs/train_pi05/PT/eval/src/merge_eval_chunks.py"
    subprocess.run([sys.executable, str(merge), "--out_dir", str(out / "src"), "--expected_tasks", "1",
                    "--labels", "pi05", "--wandb_project", env["WANDB_PROJECT"],
                    "--job_name", env["RUN_NAME"]], check=True)
    metrics = json.loads((out / "src/metrics/eval_info_merged.json").read_text())
    success = metrics["pi05"]["overall"]["pc_success"]
    clips = []
    for path in sorted((out / "src/videos").rglob("*.mp4")):
        url = html.escape(path.relative_to(out).as_posix(), quote=True)
        clips.append(f'<figure><video controls loop preload="metadata" src="{url}"></video>'
                     f'<figcaption>{html.escape(path.stem)}</figcaption></figure>')
    (out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Coffee pi05 evaluation</title>'
        '<style>body{background:#11151c;color:#eee;text-align:center;font:16px sans-serif;'
        'margin:24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));'
        'gap:16px}figure{margin:0}video{width:100%}</style>'
        f'<h1>{html.escape(env["RUN_NAME"])}</h1><p>Success: {success:.1f}%</p>'
        '<p><a href="src/metrics/eval_info_merged.json">Metrics</a></p>'
        '<img style="max-width:100%;max-height:320px" src="src/task_success_rates.png">'
        '<main>' + ''.join(clips) + '</main>', encoding="utf-8")
    print(f"Coffee pi05 success: {success:.1f}% -> {out / 'index.html'}")


if __name__ == "__main__":
    main()
