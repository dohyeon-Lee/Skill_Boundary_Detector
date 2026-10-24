"""Exercise real submit/batch scripts with CPU stubs, never Slurm or CUDA."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CONFIGS = Path(__file__).resolve().parents[2] / "examples/libero/configs"
EVAL = CONFIGS / "train_skillVLA/PT/eval"
spec = importlib.util.spec_from_file_location("stage1_fanout_packing", CONFIGS / "src/eval_gpu_packing.py")
packing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packing)


def _checkout(tmp_path, monkeypatch, *, gpus=10, workers=5):
    for key in list(os.environ):
        if key.startswith(("SLURM_", "STAGE1_EVAL_", "EVAL_")) or key in ("POLICY_PATH", "PANEL_INDICES"):
            monkeypatch.delenv(key, raising=False)
    repo = tmp_path / "repo"
    configs = repo / "lerobot/examples/libero/configs"
    work = configs / "train_skillVLA/PT/eval"
    (work / "src").mkdir(parents=True)
    shared = configs / "src"
    shared.mkdir()
    shutil.copy2(EVAL / "submit_eval.sh", work / "submit_eval.sh")
    shutil.copy2(EVAL / "src/eval.sbatch", work / "src/eval.sbatch")
    shutil.copy2(CONFIGS / "src/eval_gpu_packing.py", shared / "eval_gpu_packing.py")
    (work / "stage1_eval_config.yaml").write_text("{}\n")
    (shared / "snapshot_config.sh").write_text('snapshot_config() { printf "%s\\n" "$1"; }\n')
    for name in ("runtime_env.sh", "node_local_venv.sh", "gpu_guard.sh", "stall_guard.sh"):
        (shared / name).write_text(":\n")
    (repo / ".venv").symlink_to(sys.prefix, target_is_directory=True)
    (repo / "tools/lerobot-libero/libero/libero/init_files").mkdir(parents=True)
    artifact = repo / "artifact"
    artifact.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "nvidia-smi").write_text("#!/usr/bin/env bash\nexit 0\n")
    (fake_bin / "nvidia-smi").chmod(0o755)
    (fake_bin / "sbatch").write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys\nfrom pathlib import Path\n"
        "Path(os.environ['FAKE_SUBMIT_LOG']).write_text(json.dumps({"
        "'args':sys.argv[1:],'groups':json.loads(os.environ['EVAL_GPU_GROUPS_JSON']),"
        "'models':json.loads(os.environ['MODELS_JSON'])}))\n"
    )
    (fake_bin / "sbatch").chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("EVAL_NODE_LOCAL_VENV", "0")
    monkeypatch.setenv("EVAL_LOCAL_WORKER_STAGGER_SECONDS", "0")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("STAGE1_EVAL_BOOTSTRAP_PYTHON", sys.executable)
    monkeypatch.setenv("FAKE_SUBMIT_LOG", str(tmp_path / "submitted.json"))
    values = {
        "PROJECT_ROOT": str(repo), "LEROBOT_ROOT": str(repo / "lerobot"),
        "MODEL_COUNT": 9, "CHECKPOINT_COUNT": 1, "PANEL_COUNT": 9,
        "MODELS_JSON": json.dumps([{"label": f"model{i}"} for i in range(9)]),
        "MODEL_ARCHITECTURES": "nine models", "GRID_COLUMNS": 2,
        "TASK_IDS": json.dumps(list(range(10))), "DATASET_TASK_IDS": json.dumps(list(range(10))),
        "EVAL_NUM_GPUS": gpus, "EVAL_MAX_WORKERS_PER_GPU": workers,
        "EVAL_EXPECTED_TASKS": 10, "EVAL_OUT_DIR": str(work / "outputs/test"),
        "EVAL_PARTITION": "debug", "EVAL_QOS": "", "EVAL_GRES": "gpu:1",
        "EVAL_CPUS_PER_TASK": 8, "EVAL_MEM": "128G", "EVAL_TIME": "6:00:00",
        "EVAL_NODELIST": "", "EVAL_EXCLUDE_NODES": "", "EVAL_INIT_STATES_PATH": "",
        "SKILL_HTML": "false", "SKILL_HTML_TRAIN_SAMPLES": 0,
        "WANDB_ENABLE": "false", "WANDB_PROJECT": "", "WANDB_RUN_NAME": "test",
        "TARGET_TASK": "libero_90", "TERMINATOR_VARIANT": "state_image",
        "N_ACTION_STEPS": 5, "IMMEDIATE_REPLAN_ON_SKILL_END": "true",
        "MIN_SKILL_STEPS": 0, "REJECT_SAME_SKILL_TRANSITION": "false",
        "N_EPISODES": 10, "MAX_PARALLEL_TASKS": 1, "EVAL_BATCH_SIZE": 1,
        "MAX_VIDEOS_PER_TASK": 3, "VIDEO_FRAME_STRIDE": 4, "VIDEO_FPS": 10,
    }
    for key in ("POLICY_PATH", "FSQ_PATH", "SKILL_DATASET_DIR", "DINO_MODEL_PATH",
                "TOKENIZER_PATH", "SKILL_LATENTS_PATH", "RAW_DATASET_DIR"):
        values[key] = str(artifact)
    exports = "\n".join(f"export {key}={shlex.quote(str(value))}" for key, value in values.items())
    resolver = work / "src/stage1_eval_config.py"
    resolver.write_text(f"print({exports!r})\n")
    # A worker records the real batch launcher's panel selection and writes the
    # same uniquely tagged chunk filenames as the production evaluator.
    (work / "src/run_eval.py").write_text(
        "import json,os\nfrom pathlib import Path\n"
        "out=Path(os.environ['EVAL_OUT_DIR'])\n"
        "metrics=out/'metrics'; metrics.mkdir(parents=True,exist_ok=True)\n"
        "row={key:os.environ.get(key) for key in ['TASK_IDS','TASK_TAG','PANEL_INDICES','MODELS_JSON']}\n"
        "(metrics/('eval_info_'+row['TASK_TAG']+'.json')).write_text(json.dumps(row))\n"
        "Path(os.environ['LEROBOT_STARTUP_READY_MARKER']).touch()\n"
    )
    (work / "src/merge_eval_chunks.py").write_text(
        "import os\nfrom pathlib import Path\nPath(os.environ['EVAL_OUT_DIR'],'merged').touch()\n"
    )
    return work, values


@pytest.mark.parametrize("gpus,workers", [(10, 5), (1, 10)])
def test_submitter_uses_all_ninety_task_panel_units(tmp_path, monkeypatch, gpus, workers):
    work, _ = _checkout(tmp_path, monkeypatch, gpus=gpus, workers=workers)
    result = subprocess.run(["bash", str(work / "submit_eval.sh")], cwd=tmp_path,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    submitted = json.loads((tmp_path / "submitted.json").read_text())
    if gpus > 1:
        assert "--array=0-9%10" in submitted["args"]
    else:
        assert not any(arg.startswith("--array") for arg in submitted["args"])
    groups = submitted["groups"]
    assert len(groups) == gpus and all(len(group) == workers for group in groups)
    units = [(task, panel) for group in groups for worker in group
             for task in worker["items"] for panel in worker["panels"]]
    assert sorted(units) == [(task, panel) for task in range(10) for panel in range(9)]
    assert len(submitted["models"]) == 9


def test_panel_chunks_merge_into_the_same_nine_model_comparison(tmp_path):
    spec = importlib.util.spec_from_file_location("fanout_merge", EVAL / "src/merge_eval_chunks.py")
    merge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(merge)
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    plan = packing.plan_panel_item_chunks(list(range(10)), 9, 10, 5)
    for group in plan["groups"]:
        for worker in group:
            info = {
                f"model{panel}": {"per_task": [
                    {"task_group": "libero_90", "task_id": task,
                     "metrics": {"successes": [task % 2 == 0] * 10}}
                    for task in worker["items"]
                ]}
                for panel in worker["panels"]
            }
            (metrics / f"eval_info_{worker['tag']}.json").write_text(json.dumps(info))
    merged, found = merge.merge(tmp_path)
    labels = [f"model{panel}" for panel in range(9)]
    assert merge.ordered_labels(merged, found, labels) == labels
    for label in labels:
        assert merged[label]["overall"] == {"pc_success": 50.0, "n_tasks": 10}
        assert [task["task_id"] for task in merged[label]["per_task"]] == list(range(10))


def _json_writer():
    # Load the filesystem publisher without importing the GPU policy stack.
    tree = ast.parse((EVAL / "src/run_eval.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_write_eval_json")
    scope = {"Path": Path, "json": json, "os": os}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "eval_json", "exec"), scope)
    return scope["_write_eval_json"]


def test_metric_publication_preserves_previous_result_on_failed_write(tmp_path):
    publish = _json_writer()
    path = tmp_path / "metrics/eval_info_worker.json"
    publish(path, {"complete": True})
    with pytest.raises(TypeError):
        publish(path, {"not_json": object()})
    assert json.loads(path.read_text()) == {"complete": True}
    assert list(path.parent.iterdir()) == [path]
    publish(path, {"complete": "updated"})
    assert json.loads(path.read_text()) == {"complete": "updated"}


def test_comparison_waits_for_every_panel_video_producer(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("fanout_videos", EVAL / "src/compare_videos.py")
    videos = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(videos)
    publish = _json_writer()
    panels = [(tmp_path / "panels" / f"{index:02d}_model" / "videos", f"model{index}")
              for index in range(2)]
    for directory, _ in panels:
        target = directory / "task0/eval_episode_0.mp4"
        target.parent.mkdir(parents=True)
        target.touch()  # Exists, but represents a video still being encoded.
    cache0 = tmp_path / "metrics/panel_cache/00_model/eval_info_w0.json"
    cache1 = tmp_path / "metrics/panel_cache/01_model/eval_info_w1.json"
    read = []
    monkeypatch.setattr(videos, "read_video", lambda path: (read.append(path) or [], 10))
    publish(cache0, {"signature": {"tasks": ["task0"]}})
    assert videos.stitch_panel_videos(panels, tmp_path / "side_by_side", 2,
                                     require_completed_panels=True) == 0
    assert read == []
    cache1.parent.mkdir(parents=True)
    cache1.write_text('{"signature":')  # A legacy writer may still be in flight.
    assert videos.stitch_panel_videos(panels, tmp_path / "side_by_side", 2,
                                     require_completed_panels=True) == 0
    assert read == []
    publish(cache1, {"signature": {"tasks": ["task0"]}})
    videos.stitch_panel_videos(panels, tmp_path / "side_by_side", 2,
                              task_names={"task0"}, require_completed_panels=True)
    assert read == [directory / "task0/eval_episode_0.mp4" for directory, _ in panels]


@pytest.mark.parametrize("mode", ["array", "direct", "legacy"])
def test_batch_passes_panel_selection_and_keeps_full_comparison(tmp_path, monkeypatch, mode):
    work, values = _checkout(tmp_path, monkeypatch)
    if mode == "array":
        plan = packing.plan_panel_item_chunks(list(range(10)), 9, 10, 5)
        monkeypatch.setenv("EVAL_GPU_GROUPS_JSON", json.dumps(plan["groups"]))
        monkeypatch.setenv("EVAL_LOGICAL_WORKER_COUNT", str(plan["logical_worker_count"]))
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
        expected = plan["groups"][0]
    elif mode == "direct":
        # Direct sbatch has one GPU and fewer slots than panels: workers must
        # retain comma-separated panel groups, not drop every panel after one.
        values["EVAL_MAX_WORKERS_PER_GPU"] = 2
        expected = packing.plan_panel_item_chunks(list(range(10)), 9, 1, 2)["groups"][0]
    else:
        # Frozen task-only plans from earlier submitters still evaluate all panels.
        expected = [{"worker_index": 0, "items": [2], "tag": "w000_t2-2"}]
        monkeypatch.setenv("EVAL_GPU_GROUPS_JSON", json.dumps([expected]))
        monkeypatch.setenv("EVAL_LOGICAL_WORKER_COUNT", "1")
    for key, value in values.items():
        monkeypatch.setenv(key, str(value))
    result = subprocess.run(["bash", str(work / "src/eval.sbatch")], cwd=tmp_path,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    out = Path(values["EVAL_OUT_DIR"])
    records = [json.loads(path.read_text()) for path in (out / "metrics").glob("eval_info*.json")]
    assert len(records) == len(expected)
    by_tag = {record["TASK_TAG"]: record for record in records}
    for worker in expected:
        record = by_tag[worker["tag"]]
        assert json.loads(record["TASK_IDS"]) == worker["items"]
        assert record["PANEL_INDICES"] == ",".join(map(str, worker.get("panels", [])))
        assert json.loads(record["MODELS_JSON"]) == json.loads(values["MODELS_JSON"])
    assert (out / "merged").is_file()
