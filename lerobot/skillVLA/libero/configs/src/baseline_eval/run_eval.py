"""Use the exact in-tree pi05 LIBERO rollout/metrics/video code for isolated baselines."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import tempfile


def ensure_output_contract(settings):
    out = Path(settings["eval_out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    keys = ("baseline", "models_json", "target_task", "task_ids", "n_episodes", "eval_batch_size",
            "n_action_steps", "episode_offset", "episode_exact", "episode_exact_repeat", "eval_init_states_path", "seed")
    contract = {key: settings[key] for key in keys}
    if settings["baseline"] == "groot":
        # Also protects jobs submitted before the fix: their frozen model list
        # lacks the new decoder contract, but must never resume pre-fix metrics.
        contract["action_decoding"] = "saved-gripper-range-v1"
    path = out / "evaluation_contract.json"
    with (out / ".contract.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.is_file():
            if json.loads(path.read_text()) != contract:
                raise ValueError(f"{out} contains another evaluation recipe; choose a different output_name")
        else:
            path.write_text(json.dumps(contract, indent=2))


def panel_complete(path, label, task_ids, episodes):
    if not path.is_file():
        return False
    info = json.loads(path.read_text()).get(label, {})
    tasks = info.get("per_task", [])
    return (set(item["task_id"] for item in tasks) == set(task_ids)
            and all(len(item["metrics"]["successes"]) == episodes for item in tasks))


class Identity:
    def __call__(self, value):
        return value

    def reset(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    # The frozen resolution is exported by submit.sh; jobs never reread a changing YAML.
    keys = ("baseline", "project_root", "model_python", "sim_python", "model_source", "eval_work_dir", "models_json",
            "target_task", "task_ids", "n_episodes", "eval_batch_size", "n_action_steps", "episode_offset", "episode_exact",
            "episode_exact_repeat", "eval_init_states_path", "seed", "eval_resume", "eval_out_dir", "max_videos_per_task",
            "video_show_wrist", "video_frame_stride", "video_fps", "worker_timeout")
    settings = {key: os.environ[key.upper()] for key in keys}
    def terminated(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminated)
    for key in ("n_episodes", "eval_batch_size", "n_action_steps", "episode_offset", "seed", "max_videos_per_task",
                "video_frame_stride", "video_fps", "worker_timeout"):
        settings[key] = int(settings[key])
    for key in ("episode_exact", "episode_exact_repeat", "eval_resume", "video_show_wrist"):
        settings[key] = settings[key] == "true"
    ensure_output_contract(settings)
    project = Path(settings["project_root"])
    libero = project / "tools/lerobot-libero/libero/libero"
    if not (libero / "init_files").is_dir():
        raise FileNotFoundError(f"LIBERO environment assets missing: {libero}")
    # Never rewrite a shared LIBERO config while other jobs are reading it.
    with tempfile.TemporaryDirectory(prefix="baseline_libero_") as config_dir:
        values = {"benchmark_root": libero, "bddl_files": libero / "bddl_files", "init_states": libero / "init_files",
                  "datasets": libero.parent / "datasets", "assets": libero / "assets"}
        Path(config_dir, "config.yaml").write_text("".join(f"{key}: {value}\n" for key, value in values.items()))
        os.environ["LIBERO_CONFIG_PATH"] = config_dir
        os.environ["LEROBOT_EVAL_ANNOTATE_VIDEOS"] = "1"
        os.environ["LEROBOT_EVAL_VIDEO_WRIST"] = "1" if settings["video_show_wrist"] else "0"
        # This is also the environment variable consumed by pi05's episode-exact helper.
        os.environ["EPISODE_EXACT_REPEAT"] = str(settings["episode_exact_repeat"]).lower()
        import torch
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env, make_env_pre_post_processors
        from lerobot.envs.utils import close_envs
        from lerobot.scripts import lerobot_eval as native
        from lerobot.utils.random_utils import set_seed
        from remote_policy import IsolatedPolicy, ModelClient

        # Only presentation is wrapped; frames/actions/success/time limits come from native rollout.
        original_video = native._write_annotated_video

        def write_video(path, frames, fps, *, success, task_description, wrist_frames):
            stride = settings["video_frame_stride"]
            original_video(path, frames[::stride], settings["video_fps"], success=success,
                           task_description=task_description,
                           wrist_frames=None if wrist_frames is None else wrist_frames[::stride])

        native._write_annotated_video = write_video
        task_ids = json.loads(args.tasks)
        out = Path(settings["eval_out_dir"])
        metric_path = out / "metrics" / f"eval_info_{args.tag}.json"
        for panel in json.loads(settings["models_json"]):
            if settings["eval_resume"] and panel_complete(metric_path, panel["label"], task_ids, settings["n_episodes"]):
                print(f"[{args.tag}] resume: {panel['label']} complete", flush=True)
                continue
            set_seed(settings["seed"])
            panel_out = out / "panels" / panel["panel_dir"]
            print(f"[{args.tag}] loading {panel['label']} for tasks {task_ids}", flush=True)
            client = ModelClient(settings, panel)
            envs = None
            try:
                policy = IsolatedPolicy(client, settings["n_action_steps"])
                print(f"[{args.tag}] model/processors ready; starting LIBERO", flush=True)
                env_cfg = LiberoEnv(task=settings["target_task"], task_ids=task_ids,
                                    init_state_offset=settings["episode_offset"], max_parallel_tasks=1)
                envs = make_env(env_cfg, n_envs=settings["eval_batch_size"], use_async_envs=False)
                if settings["episode_exact"]:
                    native._apply_exact_init_states(envs, settings["eval_init_states_path"], settings["n_episodes"])
                env_pre, env_post = make_env_pre_post_processors(env_cfg, policy.config)
                with torch.inference_mode():
                    info = native.eval_policy_all(
                        envs, policy, env_pre, env_post, Identity(), Identity(), settings["n_episodes"],
                        max_episodes_rendered=settings["max_videos_per_task"], videos_dir=panel_out / "videos",
                        start_seed=settings["seed"], max_parallel_tasks=1,
                    )
                panel_out.mkdir(parents=True, exist_ok=True)
                (panel_out / f"eval_info_{args.tag}.json").write_text(json.dumps(info, indent=2))
                native._write_panel_metrics(info, str(out / "metrics"), panel["label"], args.tag)
                print(f"[{args.tag}] {panel['label']}: {info['overall']}", flush=True)
            finally:
                if envs is not None:
                    close_envs(envs)
                client.close()


if __name__ == "__main__":
    main()
