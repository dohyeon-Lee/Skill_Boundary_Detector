"""CPU regression tests: settings, grounding, saved-processor order, and private IPC."""
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest

SHARED = Path(__file__).resolve().parents[2] / "examples/libero/configs/src/baseline_eval"


def load(name, filename):
    sys.path.insert(0, str(SHARED))
    spec = importlib.util.spec_from_file_location(name, SHARED / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


settings_module = load("baseline_settings_tests", "config.py")
rpc = load("baseline_rpc_tests", "rpc.py")


def setup(tmp_path, family="molmoact2", steps=(1000, 10000)):
    venv, source, provenance = settings_module.FAMILIES[family]
    for python in (tmp_path / venv / "bin/python", tmp_path / ".venv/bin/python"):
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()
    (tmp_path / source / f"src/lerobot/policies/{family}").mkdir(parents=True)
    for step in steps:
        checkpoint = tmp_path / f"outputs_filtered/{family}_PT/run/checkpoints/{step:06d}/pretrained_model"
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text(json.dumps({
            "type": family, "chunk_size": 10, "input_features": {"observation.state": {"shape": [8]}},
            "output_features": {"action": {"shape": [7]}}}))
        (checkpoint / "model.safetensors").touch()
        (checkpoint / "policy_preprocessor.json").write_text('{"steps": []}')
        post = {"steps": []}
        if family == "groot":
            post["steps"].append({"registry_name": "groot_n1_7_action_decode_v1", "config": {
                "action_decode_transform": "libero",
                "raw_stats": {"action": {"gripper": {"min": [-1.], "max": [1.]}}},
            }})
        (checkpoint / "policy_postprocessor.json").write_text(json.dumps(post))
        (checkpoint / provenance).write_text('{"proprio_grounding": "episode_start_xyz"}')
    return dict(project_root=str(tmp_path), baseline=family, models=[dict(model_dir="run")], task_ids=[0, 1])


@pytest.mark.parametrize("family", settings_module.FAMILIES)
def test_latest_and_native_runtime(tmp_path, family):
    cfg = setup(tmp_path, family)
    settings = settings_module.build_settings(cfg, work_dir=tmp_path / "eval")
    panel = json.loads(settings["models_json"])[0]
    assert panel["checkpoint"] == "010000"
    assert panel["grounding"] == "episode_start_xyz"
    assert settings["model_python"] != settings["sim_python"]
    assert settings["eval_max_workers_per_gpu"] == 1
    if family == "groot":
        assert panel["action_decode_transform"] == "none"


@pytest.mark.parametrize("family", settings_module.FAMILIES)
@pytest.mark.parametrize("partition,excluded", [("debug", "node200"), (["debug", "batch"], ["node200", "node01"])])
def test_server_slurm_names_accept_scalar_and_list(tmp_path, family, partition, excluded):
    cfg = setup(tmp_path, family)
    cfg.update(train_partition=partition, train_exclude_nodes=excluded, train_qos="")
    settings = settings_module.build_settings(cfg, work_dir=tmp_path / "eval")
    assert settings["eval_partition"] == (partition if isinstance(partition, str) else ",".join(partition))
    assert settings["eval_exclude_nodes"] == (excluded if isinstance(excluded, str) else ",".join(excluded))
    assert settings["eval_qos"] == ""


@pytest.mark.parametrize("low,high,expected", [(-1., 1., "none"), (0., 1., "libero")])
def test_groot_legacy_gripper_contract_uses_saved_stats(tmp_path, low, high, expected):
    cfg = setup(tmp_path, "groot")
    path = tmp_path / "outputs_filtered/groot_PT/run/checkpoints/010000/pretrained_model/policy_postprocessor.json"
    pipeline = json.loads(path.read_text())
    pipeline["steps"][0]["config"]["raw_stats"]["action"]["gripper"] = {"min": [low], "max": [high]}
    path.write_text(json.dumps(pipeline))
    panel = json.loads(settings_module.build_settings(cfg, work_dir=tmp_path)["models_json"])[0]
    assert panel["action_decode_transform"] == expected


def test_groot_padded_fallback_decoder_checks_real_gripper_dimension(tmp_path):
    from groot_action_contract import groot_action_decode_transform
    path = tmp_path / "policy_postprocessor.json"
    low, high = [0.] * 132, [1.] * 132
    low[6] = -1.
    path.write_text(json.dumps({"steps": [{"registry_name": "groot_action_unpack_unnormalize_v2",
        "config": {"env_action_dim": 7, "stats": {"action": {"min": low, "max": high}}}}]}))
    assert groot_action_decode_transform(tmp_path) == "none"


@pytest.mark.parametrize("low,high", [(0., 0.), (-2., 2.), (False, True), (float("nan"), 1.)])
def test_groot_unknown_gripper_range_is_not_guessed(tmp_path, low, high):
    from groot_action_contract import groot_action_decode_transform
    path = tmp_path / "policy_postprocessor.json"
    path.write_text(json.dumps({"steps": [{"registry_name": "groot_n1_7_action_decode_v1",
        "config": {"raw_stats": {"action": {"gripper": {"min": [low], "max": [high]}}}}}]}))
    with pytest.raises(ValueError, match="gripper training range"):
        groot_action_decode_transform(tmp_path)


def test_groot_pre_fix_results_cannot_be_resumed_after_decode_repair(tmp_path):
    runner = load("baseline_groot_contract_tests", "run_eval.py")
    settings = settings_module.build_settings(setup(tmp_path, "groot"), work_dir=tmp_path / "eval")
    runner.ensure_output_contract(settings)
    path = Path(settings["eval_out_dir"]) / "evaluation_contract.json"
    contract = json.loads(path.read_text())
    assert contract.pop("action_decoding") == "saved-gripper-range-v1"
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="another evaluation"):
        runner.ensure_output_contract(settings)


@pytest.mark.parametrize("family", settings_module.FAMILIES)
def test_sweep_preserves_requested_checkpoint_order(tmp_path, family):
    cfg = setup(tmp_path, family)
    cfg["model_defaults"] = {"checkpoint": [10000, "001000"]}
    cfg["models"] = [dict(model_dir="run", label="A"), dict(model_dir="run", label="B")]
    panels = json.loads(settings_module.build_settings(cfg, work_dir=tmp_path)["models_json"])
    assert [p["label"] for p in panels] == ["A | ckpt 010000", "B | ckpt 010000", "A | ckpt 001000", "B | ckpt 001000"]
    assert len({p["panel_dir"] for p in panels}) == 4


@pytest.mark.parametrize("change", ["missing_provenance", "wrong_type", "wrong_shape", "missing_statistics"])
def test_incomplete_or_incompatible_checkpoints_rejected(tmp_path, change):
    cfg = setup(tmp_path)
    path = tmp_path / "outputs_filtered/molmoact2_PT/run/checkpoints/010000/pretrained_model"
    if change == "missing_provenance":
        (path / "sbd_molmoact2.json").unlink()
    elif change == "missing_statistics":
        (path / "policy_preprocessor.json").write_text('{"steps": [{"state_file": "missing.safetensors"}]}')
    else:
        policy = json.loads((path / "config.json").read_text())
        if change == "wrong_type":
            policy["type"] = "pi05"
        else:
            policy["output_features"]["action"]["shape"] = [8]
        (path / "config.json").write_text(json.dumps(policy))
    with pytest.raises((ValueError, FileNotFoundError)):
        settings_module.build_settings(cfg, work_dir=tmp_path)


@pytest.mark.parametrize("key,value", [("n_action_steps", 11), ("eval_max_workers_per_gpu", 6), ("max_parallel_tasks", 2), ("task_ids", [0, 0]), ("task_ids", [90])])
def test_invalid_settings_rejected(tmp_path, key, value):
    cfg = setup(tmp_path)
    cfg[key] = value
    with pytest.raises(ValueError):
        settings_module.build_settings(cfg, work_dir=tmp_path)


def test_rpc_arrays_roundtrip_and_truncation():
    stream = io.BytesIO()
    value = {"image": np.arange(768, dtype=np.uint8).reshape(1, 3, 16, 16), "task": ["open drawer"]}
    rpc.write_message(stream, value)
    stream.seek(0)
    decoded = rpc.read_message(stream)
    np.testing.assert_array_equal(decoded["image"], value["image"])
    assert decoded["task"] == value["task"]
    with pytest.raises(EOFError):
        rpc.read_message(io.BytesIO(b"short"))


def test_predictor_grounding_is_before_saved_preprocessor_and_resets():
    import torch
    worker = load("baseline_worker_tests", "policy_worker.py")
    received = []
    class Policy:
        def reset(self): pass
        def predict_action_chunk(self, batch):
            return torch.zeros(batch["observation.state"].shape[0], 10, 7)
    class Pre:
        def reset(self): pass
        def __call__(self, batch):
            received.append(batch["observation.state"].clone())
            assert batch["observation.images.image"].dtype == torch.float32
            return batch
    class Post:
        def reset(self): pass
        def __call__(self, action): return action + 2
    predictor = worker.Predictor(Policy(), Pre(), Post(), "episode_start_xyz")
    def batch(x):
        return {"observation.state": np.array([[x, 3, 4, 5, 6, 7, 8, 9]], np.float32),
                "observation.images.image": np.zeros((1, 3, 4, 4), np.uint8),
                "observation.images.wrist_image": np.zeros((1, 3, 4, 4), np.uint8)}
    assert (predictor.predict(batch(1)) == 2).all()
    predictor.predict(batch(2))
    predictor.reset()
    predictor.predict(batch(10))
    assert [state[0, 0].item() for state in received] == [0, 1, 0]
    assert all(state[0, 3:].tolist() == [5, 6, 7, 8, 9] for state in received)


def test_output_contract_and_resume_task_completeness(tmp_path):
    runner = load("baseline_runner_tests", "run_eval.py")
    cfg = setup(tmp_path)
    settings = settings_module.build_settings(cfg, work_dir=tmp_path / "eval")
    runner.ensure_output_contract(settings)
    runner.ensure_output_contract(settings)
    with pytest.raises(ValueError, match="another evaluation"):
        runner.ensure_output_contract({**settings, "n_action_steps": 1})
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"A": {"per_task": [{"task_id": 0, "metrics": {"successes": [True, False]}}]}}))
    assert runner.panel_complete(path, "A", [0], 2)
    assert not runner.panel_complete(path, "A", [0, 1], 2)
    assert not runner.panel_complete(path, "A", [0], 10)


def test_proxy_replays_only_selected_actions_and_clears_history():
    import torch
    remote = load("baseline_remote_tests", "remote_policy.py")
    class Client:
        def __init__(self): self.requests = []
        def request(self, operation, **kwargs):
            self.requests.append((operation, kwargs))
            return {"action": np.arange(70, dtype=np.float32).reshape(1, 10, 7)}
    client = Client()
    proxy = remote.IsolatedPolicy(client, n_action_steps=5)
    obs = {"observation.state": torch.ones(1, 8), "observation.images.image": torch.ones(1, 3, 2, 2),
           "observation.images.image2": torch.zeros(1, 3, 2, 2), "task": ["open"]}
    for index in range(6):
        assert proxy.select_action(obs)[0, 0].item() == (index % 5) * 7
    assert [op for op, _ in client.requests] == ["predict", "predict"]
    payload = client.requests[0][1]["batch"]
    assert payload["observation.images.image"].dtype == np.uint8
    assert (payload["observation.images.image"] == 255).all()
    assert "observation.images.wrist_image" in payload
    proxy.reset()
    assert proxy.select_action(obs)[0, 0].item() == 0
    assert [op for op, _ in client.requests][-2:] == ["reset", "predict"]


def test_rpc_timeout_is_bounded():
    import os
    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd, "rb", buffering=0) as incoming, os.fdopen(write_fd, "wb", buffering=0):
        with pytest.raises(TimeoutError):
            rpc.read_message(incoming, timeout=.02)


def test_rpc_short_writes():
    class ShortWrite(io.BytesIO):
        def write(self, data): return super().write(data[:3])
    stream = ShortWrite()
    rpc.write_message(stream, {"status": "ready"})
    stream.seek(0)
    assert rpc.read_message(stream) == {"status": "ready"}


@pytest.mark.parametrize("family", settings_module.FAMILIES)
def test_private_pipe_roundtrip_in_actual_isolated_interpreter(family):
    import os
    import subprocess
    root = SHARED.parents[5]
    python = root / settings_module.FAMILIES[family][0] / "bin/python"
    if not python.exists():
        pytest.skip(f"Optional {family} venv not installed")
    env = {**os.environ, "PYTHONPATH": str(SHARED)}
    code = "from rpc import read_message, write_message; import sys; write_message(sys.stdout.buffer, read_message(sys.stdin.buffer))"
    process = subprocess.Popen([str(python), "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, bufsize=0)
    try:
        value = {"observation.state": np.arange(16, dtype=np.float32).reshape(2, 8)}
        rpc.write_message(process.stdin, value)
        received = rpc.read_message(process.stdout, timeout=20)
        np.testing.assert_array_equal(received["observation.state"], value["observation.state"])
        assert process.wait(timeout=20) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdin.close()
        process.stdout.close()


def test_existing_rollout_accepts_proxy_and_resets_between_episodes(monkeypatch):
    import gymnasium as gym
    import torch
    from lerobot.scripts import lerobot_eval as native
    remote = load("baseline_proxy_rollout_tests", "remote_policy.py")
    runner = load("baseline_identity_rollout_tests", "run_eval.py")
    class Client:
        def __init__(self): self.resets = 0; self.predictions = 0
        def request(self, operation, **kwargs):
            if operation == "reset":
                self.resets += 1
                return {}
            self.predictions += 1
            return {"action": np.zeros((1, 10, 7), np.float32)}
    class Toy(gym.Env):
        metadata = {"render_fps": 10}
        _max_episode_steps = 8
        action_space = gym.spaces.Box(-1, 1, (7,), dtype=np.float32)
        observation_space = gym.spaces.Dict({
            "observation.state": gym.spaces.Box(-10, 10, (8,), dtype=np.float32),
            "observation.images.image": gym.spaces.Box(0, 1, (3, 2, 2), dtype=np.float32),
            "observation.images.image2": gym.spaces.Box(0, 1, (3, 2, 2), dtype=np.float32),
        })
        def obs(self): return {key: np.zeros(space.shape, np.float32) for key, space in self.observation_space.spaces.items()}
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.count = 0
            return self.obs(), {}
        def step(self, action):
            assert action.shape == (7,) and np.isfinite(action).all()
            self.count += 1
            done = self.count == 8
            return self.obs(), float(done), done, False, {"is_success": done}
    monkeypatch.setattr(native, "preprocess_observation", lambda obs: {key: torch.from_numpy(value) for key, value in obs.items()})
    monkeypatch.setattr(native, "add_envs_task", lambda env, obs: {**obs, "task": ["open drawer"]})
    monkeypatch.setattr(native, "check_env_attributes_and_types", lambda env: None)
    client = Client()
    proxy = remote.IsolatedPolicy(client, 5)
    env = gym.vector.SyncVectorEnv([Toy], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    try:
        result = native.eval_policy_all({"toy": {0: env}}, proxy, runner.Identity(), runner.Identity(),
                                        runner.Identity(), runner.Identity(), 2, max_parallel_tasks=1, start_seed=1000)
        assert result["overall"]["pc_success"] == 100.0
        assert result["overall"]["n_episodes"] == 2
        assert client.resets == 2 and client.predictions == 4  # two chunks per 8-step episode
    finally:
        env.close()


@pytest.mark.parametrize("family", settings_module.FAMILIES)
@pytest.mark.parametrize("worker_exit", [0, 13])
def test_slurm_copied_batch_uses_project_sources_and_preserves_failure(tmp_path, family, worker_exit):
    """Execute a real batch-script copy, as Slurm does, without GPU/model dependencies."""
    import os
    import shutil
    import subprocess

    project = tmp_path / "project with spaces"
    configs = project / "lerobot/examples/libero/configs"
    shared = configs / "src/baseline_eval"
    shared.mkdir(parents=True)
    (configs / "src/runtime_env.sh").write_text(":\n")
    (configs / "src/gpu_guard.sh").write_text("require_cuda_or_requeue() { return 0; }\n")
    (shared / "run_eval.py").write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        "out = Path(os.environ['EVAL_OUT_DIR'])\n"
        "(out / 'worker_source.txt').write_text(__file__)\n"
        f"status = {worker_exit}\n"
        "if status:\n    print('Simulated model failure', file=sys.stderr)\n    sys.exit(status)\n"
        "(out / 'metrics/eval_info_w000_t0-0.json').write_text(json.dumps({'test': {}}))\n"
    )
    merge = configs / "train_pi05/PT/eval/src"
    merge.mkdir(parents=True)
    (merge / "merge_eval_chunks.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "Path(os.environ['EVAL_OUT_DIR'], 'merge_ran').touch()\n"
    )
    spool = tmp_path / "var/spool/slurmd/job123"
    spool.mkdir(parents=True)
    script = spool / "slurm_script"
    shutil.copy2(SHARED / "eval.sbatch", script)
    work = configs / f"train_{family}/PT/eval"
    (work / "logs").mkdir(parents=True)  # submit.sh creates logs before calling sbatch.
    out = work / "outputs/test"
    env = {
        **os.environ, "PROJECT_ROOT": str(project), "BASELINE": family,
        "SIM_PYTHON": sys.executable, "MODEL_PYTHON": sys.executable,
        "EVAL_WORK_DIR": str(work), "EVAL_OUT_DIR": str(out), "EVAL_CPUS_PER_TASK": "1",
        "BASELINE_GPU_GROUP": "0", "SLURM_ARRAY_TASK_ID": "0", "SLURM_ARRAY_JOB_ID": "123",
        "EVAL_GPU_GROUPS_JSON": json.dumps([[{"items": [0], "tag": "w000_t0-0"}]]),
        "WANDB_ENABLE": "false", "WANDB_PROJECT": "", "WANDB_RUN_NAME": "test",
        "EVAL_EXPECTED_TASKS": "1", "VIDEO_ENABLE": "false", "PANEL_COUNT": "1",
        "GRID_COLUMNS": "0", "TMPDIR": str(tmp_path),
    }
    result = subprocess.run(["bash", str(script)], cwd=spool, env=env,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == worker_exit, result.stdout + result.stderr
    assert Path((out / "worker_source.txt").read_text()) == shared / "run_eval.py"
    assert (out / "metrics").is_dir()
    assert (out / "merge_ran").exists() == (worker_exit == 0)
    assert "No such file or directory" not in result.stderr
    if worker_exit:
        assert "Simulated model failure" in result.stderr
