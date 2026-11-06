"""Replay-free FT inheritance, unchanged updates, and step-zero group baselines."""
from __future__ import annotations

from contextlib import nullcontext
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_cyclic/pi05"


def module(name):
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / "src"))
        spec = importlib.util.spec_from_file_location(f"_test_cyclic_ft_{name}", ROOT / f"src/{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


CONFIG = module("ft_config")
DIAG = module("ft_diagnostics")


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.delenv("SBD_SERVER", raising=False)
    (tmp_path / "lerobot/examples/libero/configs").mkdir(parents=True)
    tokenizer = tmp_path / "models/tokenizer"
    tokenizer.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (tokenizer / name).write_text("{}")
    (tmp_path / "global_config.yaml").write_text(yaml.safe_dump({
        "project_root": str(tmp_path), "outputs_root": "outputs", "dataset_root": "datasets",
        "train_partition": ["test"], "train_qos": "test", "server": "fixture",
    }))
    (tmp_path / "servers").mkdir()
    (tmp_path / "servers/fixture.yaml").write_text("detect: []\nscheduler: slurm\n")
    source = tmp_path / "outputs/pi05_cyclic_PT/cyclic_run"
    checkpoint = source / "checkpoints/100000/pretrained_model"
    checkpoint.mkdir(parents=True)
    policy = {"type": "pi05", "chunk_size": 10, "proprio_grounding": "episode_start_xyz",
              "freeze_vision_encoder": False, "freeze_language_model": False,
              "freeze_action_expert": False, "dtype": "bfloat16", "gradient_checkpointing": True}
    parent = {"policy": policy, "dataset": {"repo_id": "lerobot/libero_90_full_full"}, "batch_size": 2}
    (checkpoint / "config.json").write_text(json.dumps(policy))
    (checkpoint / "train_config.json").write_text(json.dumps(parent))
    (checkpoint / "model.safetensors").write_bytes(b"fixture")
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        (checkpoint / name).write_text("{}")
    state = checkpoint.parent / "training_state"
    state.mkdir()
    (state / "training_step.json").write_text('{"step": 100000}')
    groups = [{"group_id": i, "tasks": [f"old{i}"], "episode_indices": [i],
               "frame_ranges": [[i * 4, (i + 1) * 4]]} for i in range(2)]
    (source / "groups.json").write_text(json.dumps(groups))
    probe_cfg = {"groups": hashlib.sha256(json.dumps(groups, sort_keys=True).encode()).hexdigest(),
                 "dataset": "lerobot/libero_90_full_full", "batch_size": 2,
                 "batches_per_group": 1, "probe_seed": 12345, "phase_steps": 200,
                 "group_seed": 0, "grad_group": 0}
    (source / "cyclic_probe_config.json").write_text(json.dumps(probe_cfg))
    batches = {i: [{"task": [f"old{i}"] * 2, "x": torch.tensor([[i + .1], [i + .2]])}]
               for i in range(2)}
    torch.save(batches, source / "cyclic_probe_batches.pt")
    raw = yaml.safe_load((ROOT / "FT/ft_config.yaml").read_text())
    raw.update(pi05_tokenizer="models/tokenizer", ft_source={"model_dir": "cyclic_run", "label": "cyclic"},
               ft_source_checkpoint="100000")
    raw["training"]["gradient_checkpointing"] = True
    path = tmp_path / "ft_config.yaml"
    path.write_text(yaml.safe_dump(raw))
    settings = CONFIG.build_settings(CONFIG.load_config(path), require_source=True)
    return path, source, checkpoint, settings, policy


def test_default_recipe_matches_real_ft_and_uses_full_libero10(experiment):
    path, _, checkpoint, settings, _ = experiment
    assert settings["ft_dataset"] == "libero_10_full_full"
    assert settings["ft_pretrained_model_path"] == checkpoint
    assert settings["ft_steps"] == 15000
    assert settings["ft_lr"] == "2.5e-05"
    assert settings["ft_lr_mode"] == "cosine_decay"
    assert settings["ft_warmup_steps"] == 1000
    assert settings["ft_decay_steps"] == 30000
    assert settings["ft_normalization_stats"] == "pt"
    assert settings["ft_proprio_grounding"] == "episode_start_xyz"
    assert settings["ft_chunk_size"] == 10
    assert settings["ft_freeze_expert"] is False
    assert settings["ft_freeze_vision_encoder"] is False
    assert settings["ft_freeze_language_model"] is False
    assert settings["ft_gradient_checkpointing_expert"] is True
    assert settings["ft_probe_dataset_root"] == ""  # only the group observer supplies probes
    changed = CONFIG.load_config(path)
    changed["ft_source"] = {"model_dir": "iid_run", "label": "iid"}
    assert CONFIG.build_settings(changed)["ft_output_dir"] != settings["ft_output_dir"]


@pytest.mark.parametrize("patch", [
    {"ft_source": {"model_dir": "../bad"}}, {"ft_source": "cyclic_run"},
    {"ft_source_checkpoint": "last"}, {"ft_source_checkpoint": 100000},
    {"ft_probe_every": 0}, {"ft_num_gpus": 2},
])
def test_invalid_config_rejected(experiment, patch):
    cfg = CONFIG.load_config(experiment[0])
    cfg.update(patch)
    with pytest.raises(ValueError):
        CONFIG.build_settings(cfg)


@pytest.mark.parametrize("mode", ["ft", "pt"])
def test_ft_statistics_are_selectable(experiment, mode):
    cfg = CONFIG.load_config(experiment[0])
    cfg["training"]["normalization_stats"] = mode
    settings = CONFIG.build_settings(cfg)
    assert settings["ft_normalization_stats"] == mode
    assert ("_ptstats" in settings["ft_run_name"]) == (mode == "pt")


def test_source_freeze_and_horizon_inherit_from_checkpoint(experiment):
    path, _, checkpoint, _, policy = experiment
    policy.update(freeze_language_model=True, freeze_vision_encoder=True, chunk_size=20)
    (checkpoint / "train_config.json").write_text(json.dumps({"policy": policy}))
    settings = CONFIG.build_settings(CONFIG.load_config(path))
    assert settings["ft_freeze_language_model"] and settings["ft_freeze_vision_encoder"]
    assert settings["ft_chunk_size"] == 20


def test_snapshot_freezes_global_and_source_selection(experiment):
    path = experiment[0]
    before = CONFIG.build_settings(CONFIG.load_config(path))
    frozen = CONFIG.freeze_config(path)
    raw = yaml.safe_load(path.read_text())
    raw["ft_source"]["model_dir"] = "iid_run"
    path.write_text(yaml.safe_dump(raw))
    (path.parent / "global_config.yaml").write_text("project_root: /changed\n")
    assert CONFIG.build_settings(CONFIG.load_config(frozen)) == before


def test_missing_source_only_allowed_for_planning_and_incomplete_outputs_preserved(experiment):
    path, source, _, settings, _ = experiment
    (source / "cyclic_probe_batches.pt").unlink()
    CONFIG.build_settings(CONFIG.load_config(path))
    with pytest.raises(FileNotFoundError):
        CONFIG.build_settings(CONFIG.load_config(path), require_source=True)
    output = settings["ft_output_dir"]
    output.mkdir(parents=True)
    marker = output / "keep.txt"
    marker.write_text("preserve")
    with pytest.raises(ValueError, match="without resumable"):
        CONFIG.build_settings(CONFIG.load_config(path))
    assert marker.read_text() == "preserve"
    incomplete = output / "checkpoints/last"
    incomplete.mkdir(parents=True)
    (incomplete / "cyclic_ft_diagnostics.json").write_text("{}")
    with pytest.raises(ValueError, match="without resumable"):
        CONFIG.build_settings(CONFIG.load_config(path))


def test_incomplete_source_weights_rejected(experiment):
    path, _, checkpoint, _, _ = experiment
    (checkpoint / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="weights are incomplete"):
        CONFIG.build_settings(CONFIG.load_config(path), require_source=True)


def observer_fixture(experiment):
    _, _, _, settings, policy_config = experiment
    settings = {**settings, "ft_steps": 5, "ft_batch_size": 2, "cyclic_ft_probe_every": 2}

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(1, 1)
            with torch.no_grad():
                self.layer.weight.fill_(.25)
                self.layer.bias.fill_(.1)

        def forward(self, batch):
            return (self.layer(batch["x"]) - torch.randn_like(batch["x"])).square().mean(), {}

    runtime = {"policy": policy_config, "optimizer": {"lr": .01}, "scheduler": {"type": "constant"}}
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(repo_id="lerobot/libero_10_full_full"),
        policy=SimpleNamespace(**policy_config), batch_size=2, steps=5, seed=1000,
        output_dir=settings["ft_output_dir"], checkpoint_path=None, normalization_stats="pt",
        to_dict=lambda: copy.deepcopy(runtime),
    )
    accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext)
    return cfg, Policy(), accelerator, settings


@pytest.mark.parametrize("mode", ["ft", "pt"])
def test_ft_probes_use_selected_preprocessor_and_preserve_rng_gradients_and_module_flags(experiment, mode):
    cfg, policy, accelerator, settings = observer_fixture(experiment)
    cfg.normalization_stats = settings["ft_normalization_stats"] = mode
    preprocess = (lambda b: {**b, "x": b["x"] + 2}) if mode == "ft" else (lambda b: b)
    policy.train()
    policy.layer.eval()
    for p in policy.parameters():
        p.grad = torch.full_like(p, .75)
    gradients = [p.grad.clone() for p in policy.parameters()]
    before = torch.get_rng_state(), np.random.get_state(), random.getstate()
    observer = DIAG.FTObserver(cfg, policy, preprocess, accelerator, None, 0, settings)
    pt = module("diagnostics").CurriculumObserver.__new__(module("diagnostics").CurriculumObserver)
    pt.policy, pt.preprocessor, pt.accelerator = policy, preprocess, accelerator
    pt.batches = torch.load(experiment[1] / "cyclic_probe_batches.pt", weights_only=False)
    pt.seed = 12345
    assert observer.losses() == pt.losses()
    assert observer.signature["normalization_stats"] == mode
    assert torch.equal(before[0], torch.get_rng_state())
    assert np.array_equal(before[1][1], np.random.get_state()[1])
    assert before[2] == random.getstate()
    assert policy.training and not policy.layer.training
    assert all(torch.equal(p.grad, g) for p, g in zip(policy.parameters(), gradients, strict=True))
    assert DIAG.file_digest(observer.root / "cyclic_probe_batches.pt") == DIAG.file_digest(
        experiment[1] / "cyclic_probe_batches.pt"
    )


def test_ft_resume_cannot_switch_statistics_mode(experiment):
    cfg, policy, accelerator, settings = observer_fixture(experiment)
    observer = DIAG.FTObserver(cfg, policy, lambda b: b, accelerator, None, 0, settings)
    cfg.normalization_stats = "ft"
    with pytest.raises(ValueError, match="recipe differs"):
        DIAG.FTObserver(cfg, policy, lambda b: b, accelerator, None, 1,
                        {**settings, "ft_normalization_stats": "ft"})


def test_probes_preserve_updates_and_resume_keeps_original_ft_baseline(experiment):
    cfg, policy, accelerator, settings = observer_fixture(experiment)
    baseline_model = copy.deepcopy(policy)
    observer = DIAG.FTObserver(cfg, policy, lambda b: b, accelerator, None, 0, settings)
    initial = dict(observer.baseline)
    opts = [torch.optim.Adam(m.parameters(), lr=.01) for m in (baseline_model, policy)]
    for step in range(1, 4):
        batch = {"task": ["new0", "new1"], "x": torch.tensor([[.8], [.9]])}
        for model, optimizer in zip((baseline_model, policy), opts, strict=True):
            torch.manual_seed(100 + step)
            if model is policy:
                observer.on_batch(batch)
            loss, _ = model(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if model is policy:
                observer.on_step(step)
        assert all(torch.equal(a, b) for a, b in zip(baseline_model.parameters(), policy.parameters(), strict=True))
    checkpoint = observer.root / "checkpoints/000003"
    checkpoint.mkdir(parents=True)
    observer.on_checkpoint(checkpoint, 3)  # mid-probe-interval checkpoint
    observer.metrics_path.write_text(observer.metrics_path.read_text() + '{"step": 4}\n')
    cfg.checkpoint_path = checkpoint
    resumed = DIAG.FTObserver(cfg, policy, lambda b: b, accelerator, None, 3, settings)
    assert resumed.baseline == initial
    assert resumed.losses() != initial
    resumed.on_batch({})
    resumed.on_step(4)
    resumed.on_batch({})
    resumed.on_step(5)
    rows = [json.loads(line) for line in resumed.metrics_path.read_text().splitlines()]
    assert [row["step"] for row in rows] == [0, 2, 3, 4, 5]
    assert rows[0]["probe_forget"] == {"g0": 0., "g1": 0.}
    for gid, loss in rows[-1]["probe_loss"].items():
        ref = initial[int(gid[1:])]
        assert rows[-1]["probe_forget"][gid] == pytest.approx((loss - ref) / abs(ref))


@pytest.mark.parametrize("change", ["grounding", "dataset", "normalization", "horizon", "groups", "probe_recipe"])
def test_observer_rejects_incompatible_coordinates_or_probes(experiment, change):
    cfg, policy, accelerator, settings = observer_fixture(experiment)
    if change == "grounding":
        cfg.policy.proprio_grounding = "none"
    elif change == "dataset":
        cfg.dataset.repo_id = "lerobot/libero_90_full_full"
    elif change == "normalization":
        cfg.normalization_stats = "ft"
    elif change == "horizon":
        cfg.policy.chunk_size = 20
    elif change == "groups":
        (experiment[1] / "groups.json").write_text("[]")
    else:
        p = experiment[1] / "cyclic_probe_config.json"
        saved = json.loads(p.read_text())
        saved["batch_size"] = 16
        p.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        DIAG.FTObserver(cfg, policy, lambda b: b, accelerator, None, 0, settings)


def test_resume_rejects_changed_schedule_missing_state_and_modified_probe_cache(experiment):
    cfg, policy, accelerator, settings = observer_fixture(experiment)
    observer = DIAG.FTObserver(cfg, policy, lambda b: b, accelerator, None, 0, settings)
    cfg.checkpoint_path = observer.root / "checkpoints/000001"
    cfg.checkpoint_path.mkdir(parents=True)
    with pytest.raises(ValueError, match="recipe differs"):
        DIAG.FTObserver(cfg, policy, lambda b: b, accelerator, None, 1,
                        {**settings, "cyclic_ft_probe_every": 3})
    with pytest.raises(FileNotFoundError):
        DIAG.FTObserver(cfg, policy, lambda b: b, accelerator, None, 1, settings)
    (observer.root / "cyclic_probe_batches.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="batches are missing or changed"):
        DIAG.FTObserver(cfg, policy, lambda b: b, accelerator, None, 1, settings)


def test_dry_run_never_submits_or_creates_outputs_or_snapshots(experiment, tmp_path):
    path = experiment[0]
    env = {**os.environ, "TRAIN_CYCLIC_FT_DRY_RUN": "1", "TRAIN_CYCLIC_FT_CONFIG": str(path)}
    result = subprocess.run(["bash", str(ROOT / "FT/submit_train.sh")], env=env,
                            capture_output=True, text=True, check=True)
    assert "no job submitted" in result.stdout
    assert not (tmp_path / ".config_snapshots").exists()
    assert not experiment[3]["ft_output_dir"].exists()


def test_eval_pairs_pre_and_post_ft_on_all_90_tasks_and_configured_output(experiment):
    evaluation = yaml.safe_load((ROOT / "FT/eval/pi05_eval_config.yaml").read_text())
    before, after = evaluation["models"]
    raw = CONFIG.load_config(experiment[0])
    default = yaml.safe_load((ROOT / "FT/ft_config.yaml").read_text())
    raw["ft_source"] = default["ft_source"]
    raw["ft_source_checkpoint"] = str(before["checkpoint"]).zfill(6)
    settings = CONFIG.build_settings(raw)
    assert before["stage"] == "PT" and after["stage"] == "FT"
    assert before["model_dir"] == default["ft_source"]["model_dir"]
    assert after["model_dir"] == settings["ft_run_name"]
    assert int(after["checkpoint"]) == settings["ft_steps"]
    assert evaluation["task_ids"] == list(range(90))
    assert evaluation["target_task"] == "libero_90"
    assert evaluation["random_range"] == 1.


def test_plain_pt_can_share_cyclic_probes_without_replacing_checkpoint_identity(experiment):
    path, source, checkpoint, _, _ = experiment
    plain = path.parent / "outputs/pi05_PT/ordinary_run"
    shutil.copytree(checkpoint.parent, plain / "checkpoints/100000")
    raw = CONFIG.load_config(path)
    raw.update(pi05_outputs_root="pi05_PT", ft_source={"model_dir": "ordinary_run", "label": "iid"},
               ft_probe_source_run="outputs/pi05_cyclic_PT/cyclic_run")
    settings = CONFIG.build_settings(raw, require_source=True)
    assert settings["cyclic_ft_source_run"] == plain
    assert settings["cyclic_ft_probe_source_run"] == source
    assert not (plain / "cyclic_probe_batches.pt").exists()
    cfg, policy, accelerator, _ = observer_fixture(experiment)
    cfg.output_dir = settings["ft_output_dir"]
    settings.update(ft_steps=cfg.steps, ft_batch_size=cfg.batch_size, cyclic_ft_probe_every=2)
    observer = DIAG.FTObserver(cfg, policy, lambda x: x, accelerator, None, 0, settings)
    assert observer.signature["source_run"] == "ordinary_run"
    assert observer.signature["probe_source_run"] == "cyclic_run"
    assert observer.signature["source_train_sha256"] == DIAG.file_digest(
        plain / "checkpoints/100000/pretrained_model/train_config.json")
    assert observer.signature["probe_batches_sha256"] == DIAG.file_digest(source / "cyclic_probe_batches.pt")
    assert set(observer.baseline) == {0, 1}


@pytest.mark.parametrize("field,value,match", [
    ("batch_size", 9, "batch size"), ("dataset", "lerobot/other", "different datasets"),
])
def test_shared_probes_reject_incompatible_source(experiment, field, value, match):
    cfg, policy, accelerator, settings = observer_fixture(experiment)
    probe_dir = experiment[0].parent / "shared_probes"
    probe_dir.mkdir()
    for name in ("groups.json", "cyclic_probe_config.json", "cyclic_probe_batches.pt"):
        shutil.copy2(experiment[1] / name, probe_dir / name)
    f = probe_dir / "cyclic_probe_config.json"
    meta = json.loads(f.read_text())
    meta[field] = value
    f.write_text(json.dumps(meta))
    settings["cyclic_ft_probe_source_run"] = probe_dir
    with pytest.raises(ValueError, match=match):
        DIAG.FTObserver(cfg, policy, lambda x: x, accelerator, None, 0, settings)


def test_entrypoint_uses_native_ft_without_cyclic_loader_or_generic_probe():
    entry = (ROOT / "src/train_ft.py").read_text()
    assert "native.train(training_observer_factory=make_observer_factory(settings))" in entry
    assert "retention.install_retention(native, settings)" in entry
    assert "offline_dataloader_factory" not in entry


def test_entrypoint_installs_real_retention_and_calls_native_observer_hook(experiment, monkeypatch):
    import sys

    calls = []
    def update_last(path):
        return path
    native = SimpleNamespace(
        update_last_checkpoint=update_last,
        _run_inline_cuda_guard=lambda: calls.append("guard"),
        register_third_party_plugins=lambda: calls.append("plugins"),
        train=lambda **kwargs: calls.append(kwargs),
    )
    scripts = ModuleType("lerobot.scripts")
    scripts.lerobot_train = native
    monkeypatch.setitem(sys.modules, "lerobot.scripts", scripts)
    monkeypatch.setenv("TRAIN_PI05_CONFIG", str(experiment[0]))
    module("train_ft").main()
    assert calls[:2] == ["guard", "plugins"]
    assert set(calls[2]) == {"training_observer_factory"}
    assert callable(calls[2]["training_observer_factory"])
    assert native.update_last_checkpoint is not update_last


@pytest.mark.parametrize("filename", ["pi05_eval_config.yaml", "libero10_eval_config.yaml"])
def test_dedicated_eval_configs_resolve_with_current_engine(experiment, filename):
    path, _, _, settings, policy = experiment
    config = yaml.safe_load((ROOT / "FT/eval" / filename).read_text())
    config.update(project_root=str(path.parent), outputs_root="outputs", pi05_tokenizer="models/tokenizer",
                  slurm_submit_mode="srun", srun_job_id="123")
    config["models"][0]["model_dir"] = "cyclic_run"
    config["models"][1]["model_dir"] = settings["ft_run_name"]
    after = settings["ft_output_dir"] / "checkpoints/015000/pretrained_model"
    after.mkdir(parents=True)
    (after / "config.json").write_text(json.dumps(policy))
    eval_path = ROOT.parents[1] / "train_pi05/PT/eval/src/pi05_eval_config.py"
    spec = importlib.util.spec_from_file_location("_cyclic_ft_eval_test", eval_path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    resolved = evaluator.build_settings(config)
    panels = json.loads(resolved["models_json"])
    assert resolved["panel_count"] == 2
    assert resolved["eval_expected_tasks"] == (90 if filename == "pi05_eval_config.yaml" else 10)
    assert all(p["proprio_grounding"] == "episode_start_xyz" for p in panels)


def test_sequential_ft_uses_ft_weights_and_original_pt_probes(experiment):
    path, source, checkpoint, _, _ = experiment
    previous = source.parents[0] / 'task0_ft'
    shutil.copytree(source, previous)
    previous_checkpoint = previous / 'checkpoints/005000/pretrained_model'
    previous_checkpoint.parent.mkdir(parents=True)
    shutil.copytree(checkpoint, previous_checkpoint, dirs_exist_ok=True)
    shutil.copytree(checkpoint.parent / 'training_state', previous_checkpoint.parent / 'training_state')
    (previous_checkpoint.parent / 'training_state/training_step.json').write_text('{"step":5000}')
    parent = json.loads((previous_checkpoint / 'train_config.json').read_text())
    parent['dataset']['repo_id'] = 'lerobot/libero_10_0_full'
    (previous_checkpoint / 'train_config.json').write_text(json.dumps(parent))
    raw = CONFIG.load_config(path)
    raw.update(ft_source={'model_dir': previous.name, 'label': 'cyclic', 'stage': 'ft'},
               ft_source_checkpoint='005000', ft_reference_pt_checkpoint=str(checkpoint),
               ft_probe_source_run=str(source), ft_dataset='libero_10_3_full')
    settings = CONFIG.build_settings(raw, require_source=True)
    assert settings['ft_pretrained_model_path'] == previous_checkpoint
    assert '_FT5k_' in settings['ft_run_name']
    assert settings['cyclic_ft_reference_pt_checkpoint'] == checkpoint
    cfg, policy, accelerator, old_settings = observer_fixture(experiment)
    cfg.dataset.repo_id = 'lerobot/libero_10_3_full'
    cfg.output_dir = settings['ft_output_dir']
    settings.update(ft_steps=5, ft_batch_size=2, cyclic_ft_probe_every=2)
    observer = DIAG.FTObserver(cfg, policy, lambda b: b, accelerator, None, 0, settings)
    assert observer.signature['source_run'] == previous.name
    assert 'reference_pt_train_sha256' in observer.signature
    cfg.dataset.repo_id = 'lerobot/libero_10_0_full'
    with pytest.raises(ValueError, match='new-task'):
        DIAG.FTObserver(cfg, policy, lambda b: b, accelerator, None, 0, settings)
    del raw['ft_reference_pt_checkpoint']
    with pytest.raises(ValueError, match='requires ft_reference'):
        CONFIG.build_settings(raw)
