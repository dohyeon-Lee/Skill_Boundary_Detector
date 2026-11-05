"""Keep pi05's native recipe intact; test curriculum order, inheritance and queued snapshots."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

CONFIGS = Path(__file__).resolve().parents[2] / "examples/libero/configs"
ROOT = CONFIGS / "train_cyclic/pi05"


def module(name):
    spec = importlib.util.spec_from_file_location(f"_test_cyclic_pi05_{name}", ROOT / f"src/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CONFIG, SAMPLING = module("config"), module("sampling")


@pytest.fixture
def files(tmp_path, monkeypatch):
    monkeypatch.delenv("SBD_SERVER", raising=False)
    (tmp_path / "lerobot/examples/libero/configs").mkdir(parents=True)
    tokenizer = tmp_path / "models/tokenizer"
    tokenizer.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (tokenizer / name).write_text("{}")
    global_path = tmp_path / "global_config.yaml"
    global_path.write_text(yaml.safe_dump({
        "project_root": str(tmp_path), "outputs_root": "outputs", "dataset_root": "dataset",
        "train_partition": ["alpha", "beta"], "train_qos": "base_qos", "server": "fixture",
    }))
    (tmp_path / "servers").mkdir()
    (tmp_path / "servers/fixture.yaml").write_text("detect: []\nscheduler: slurm\n")
    source = tmp_path / "pi05_PT_config.yaml"
    raw = yaml.safe_load((CONFIGS / "train_pi05/PT/pi05_PT_config.yaml").read_text())
    raw["pi05_tokenizer"] = "models/tokenizer"
    source.write_text(yaml.safe_dump(raw))
    paths = {}
    for mode in ("iid", "cyclic"):
        p = tmp_path / f"{mode}_config.yaml"
        p.write_text(yaml.safe_dump({"pt_config": source.name, "mode": mode,
                                     "cycle": {"n_groups": 2, "phase_steps": 3, "group_seed": 4}}))
        paths[mode] = p
    return source, paths, global_path


def test_modes_inherit_every_native_setting_except_run_identity(files):
    source, paths, _ = files
    native = CONFIG.base.build_settings(CONFIG.base.load_config(source))
    allowed = {"pt_run_name", "pt_output_dir", "pi05_pt_outputs_root", "train_bin", "pt_wandb_project"}
    for mode, p in paths.items():
        actual = CONFIG.build_settings(CONFIG.load_config(p))
        for key, value in native.items():
            if key not in allowed:
                assert actual[key] == value, key
        assert actual["pt_batch_size"] == 16
        assert actual["pt_num_workers"] == 8
        assert actual["pt_cpus_per_task"] == 16
        assert actual["pt_proprio_grounding"] == "episode_start_xyz"
        assert actual["pt_steps"] == 100000
        assert actual["pt_chunk_size"] == 10
        assert actual["pt_lr"] == 2.5e-5
        assert actual["pt_lr_mode"] == "warmup_constant"
        assert actual["pt_qos"] == native["pt_qos"]
        assert actual["pt_output_dir"] != native["pt_output_dir"]
        assert actual["pt_wandb_project"] == "cyclic_PT"
    iid = CONFIG.build_settings(CONFIG.load_config(paths["iid"]))
    assert iid["train_bin"] == ROOT / "src/train.py"
    assert CONFIG.build_settings(CONFIG.load_config(paths["cyclic"]))["train_bin"] == ROOT / "src/train.py"


def test_source_edits_propagate_to_both_modes_but_not_snapshots(files):
    source, paths, global_path = files
    before = CONFIG.build_settings(CONFIG.load_config(paths["cyclic"]))
    snapshot = CONFIG.freeze_config(paths["cyclic"])
    assert (snapshot.parent / "inherited_pi05_PT_config.yaml").read_text() == source.read_text()
    assert (snapshot.parent / "servers/fixture.yaml").is_file()
    raw = yaml.safe_load(source.read_text())
    raw["training"]["dataloader"]["batch_size"] = 7
    raw["slurm"]["cpus"] = 9
    source.write_text(yaml.safe_dump(raw))
    global_raw = yaml.safe_load(global_path.read_text())
    global_raw["train_qos"] = "big_qos"
    global_path.write_text(yaml.safe_dump(global_raw))
    for p in paths.values():
        s = CONFIG.build_settings(CONFIG.load_config(p))
        assert s["pt_batch_size"] == 7 and s["pt_cpus_per_task"] == 9
        assert s["pt_qos"] == "big_qos"
    frozen = CONFIG.build_settings(CONFIG.load_config(snapshot))
    assert frozen["pt_batch_size"] == before["pt_batch_size"]
    assert frozen["pt_cpus_per_task"] == before["pt_cpus_per_task"]
    assert frozen["pt_qos"] == before["pt_qos"]


@pytest.mark.parametrize("patch", [
    {"mode": "reptile"}, {"cycle": {"reptile_beta": 0.5}},
    {"cycle": {"n_groups": 0}}, {"cycle": {"phase_steps": True}},
    {"cycle": {"group_seed": -1}}, {"cycle": [1]},
])
def test_invalid_curricula_fail_early(files, patch):
    cfg = CONFIG.load_config(files[1]["cyclic"])
    cfg.update(patch)
    with pytest.raises(ValueError):
        CONFIG.build_settings(cfg)


def test_training_overrides_belong_in_shared_pt_config(files):
    p = files[1]["iid"]
    raw = yaml.safe_load(p.read_text())
    raw["training"] = {"dataloader": {"batch_size": 999}}
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="shared training"):
        CONFIG.load_config(p)


def test_multi_gpu_is_rejected_for_comparison(files):
    cfg = CONFIG.load_config(files[1]["cyclic"])
    cfg["pt"]["training"]["dataloader"]["gpus"] = 2
    with pytest.raises(ValueError, match="gpus: 1"):
        CONFIG.build_settings(cfg)


def test_incomplete_outputs_are_preserved(files):
    settings = CONFIG.build_settings(CONFIG.load_config(files[1]["iid"]))
    CONFIG.check_output(settings)
    output = settings["pt_output_dir"]
    output.mkdir(parents=True)
    sentinel = output / "keep.txt"
    sentinel.write_text("important")
    with pytest.raises(ValueError, match="preserve/inspect"):
        CONFIG.check_output(settings)
    assert sentinel.read_text() == "important"
    checkpoint = output / "checkpoints/last/pretrained_model/train_config.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("{}")
    CONFIG.check_output(settings)


class TinyDataset:
    episodes = None

    def __init__(self):
        self.meta = SimpleNamespace(episodes={
            "dataset_from_index": [0, 8, 14, 18], "dataset_to_index": [8, 14, 18, 24],
            "tasks": [["a"], ["b"], ["c"], ["a"]],
        })

    def __len__(self):
        return 24

    def __getitem__(self, index):
        return index


def test_grouping_is_atomic_frame_balanced_and_covers_every_frame():
    dataset = TinyDataset()
    groups = SAMPLING.build_groups(dataset, 2, 0)
    frames = [int(x) for g in groups for x in g["frame_indices"]]
    assert sorted(frames) == list(range(24))
    assert len(set(frames)) == len(frames)
    assert sorted(t for g in groups for t in g["tasks"]) == ["a", "b", "c"]
    assert sorted(len(g["frame_indices"]) for g in groups) == [10, 14]
    assert groups[0]["tasks"] == SAMPLING.build_groups(dataset, 2, 0)[0]["tasks"]


@pytest.mark.parametrize("groups", [0, 4, True])
def test_empty_or_excessive_groups_are_rejected(groups):
    with pytest.raises(ValueError):
        SAMPLING.build_groups(TinyDataset(), groups, 0)


def test_partial_overlapping_and_multitask_datasets_are_rejected():
    dataset = TinyDataset()
    dataset.episodes = [0]
    with pytest.raises(ValueError, match="subset"):
        SAMPLING.build_groups(dataset, 2, 0)
    dataset.episodes = None
    dataset.meta.episodes["dataset_from_index"][1] = 7
    with pytest.raises(ValueError, match="gaps or overlaps"):
        SAMPLING.build_groups(dataset, 2, 0)
    dataset = TinyDataset()
    dataset.meta.episodes["tasks"][0] = ["a", "b"]
    with pytest.raises(ValueError, match="exactly one"):
        SAMPLING.build_groups(dataset, 2, 0)


def test_cursor_continues_epochs_without_skipping_frames():
    cursor = SAMPLING.GroupCursor(np.arange(7), 42)
    frames = cursor.take(3) + cursor.take(2) + cursor.take(9)
    assert sorted(frames[:7]) == list(range(7))
    assert sorted(frames[7:]) == list(range(7))
    assert cursor.take(0) == []
    with pytest.raises(ValueError):
        cursor.take(-1)


def test_sampler_phase_lengths_cycle_shuffle_and_full_batches():
    groups = SAMPLING.build_groups(TinyDataset(), 2, 4)
    sampler = SAMPLING.BlockCyclicBatchSampler(groups, 5, 3, 19, 4)
    batches = list(sampler)
    assert len(batches) == len(sampler) == 19
    frame_group = {int(i): g["group_id"] for g in groups for i in g["frame_indices"]}
    active = []
    for batch in batches:
        assert len(batch) == 5
        gids = {frame_group[i] for i in batch}
        assert len(gids) == 1
        active.append(gids.pop())
    for phase in range(0, 18, 3):
        assert len(set(active[phase:phase+3])) == 1
    for cycle in range(3):
        order = np.random.default_rng(5 + cycle).permutation(2).tolist()
        assert [active[cycle*6], active[cycle*6+3]] == order
    assert batches == list(sampler)


@pytest.mark.parametrize("start", [0, 1, 3, 6, 11, 19])
def test_resume_replays_exact_index_schedule_independent_of_prefetch(start):
    groups = SAMPLING.build_groups(TinyDataset(), 2, 4)
    full = list(SAMPLING.BlockCyclicBatchSampler(groups, 5, 3, 19, 4))
    resumed = SAMPLING.BlockCyclicBatchSampler(groups, 5, 3, 19, 4, start)
    assert list(resumed) == full[start:]
    assert len(resumed) == len(full) - start


def test_loader_uses_native_worker_settings_and_accelerate_preserves_order(tmp_path):
    import torch
    from accelerate import Accelerator

    dataset = TinyDataset()
    cfg = SimpleNamespace(dataset=SimpleNamespace(streaming=False, repo_id="tiny"),
                          batch_size=5, steps=19, seed=1000, output_dir=tmp_path / "run")
    settings = {"cyclic_n_groups": 2, "cyclic_group_seed": 4, "cyclic_phase_steps": 3}
    factory = SAMPLING.make_loader_factory(settings)
    kwargs = dict(dataset=dataset, num_workers=0, pin_memory=False, prefetch_factor=None,
                  persistent_workers=False, timeout=0)
    # Independent grouping/ordering must not consume the policy's torch/global NumPy RNG.
    rng = torch.get_rng_state().clone()
    np_before = np.random.get_state()
    loader = factory(cfg, dataset, kwargs, None, 0)
    assert torch.equal(rng, torch.get_rng_state())
    assert np.array_equal(np_before[1], np.random.get_state()[1])
    expected = list(loader.batch_sampler)
    prepared = Accelerator(cpu=True).prepare(loader)
    assert [x.tolist() for x in prepared] == expected
    # A prefetched original loader may be arbitrarily far ahead; only saved step matters.
    resumed = factory(cfg, dataset, kwargs, None, 7)
    assert [x.tolist() for x in resumed] == expected[7:]
    changed = dict(settings, cyclic_phase_steps=2)
    with pytest.raises(ValueError, match="curriculum differs"):
        SAMPLING.make_loader_factory(changed)(cfg, dataset, kwargs, None, 7)
    with pytest.raises(ValueError, match="without another sampler"):
        factory(cfg, dataset, kwargs, object(), 0)


def test_shell_and_dry_run_do_not_submit(files, tmp_path):
    source, paths, _ = files
    env = {**os.environ, "TRAIN_CYCLIC_DRY_RUN": "1", "TRAIN_CYCLIC_CONFIG": str(paths["iid"])}
    subprocess.run(["bash", "-n", str(ROOT / "submit_train.sh")], check=True)
    result = subprocess.run(["bash", str(ROOT / "submit_train.sh"), "iid"], env=env,
                            check=True, capture_output=True, text=True)
    assert "Dry run: no job submitted" in result.stdout
    assert not (source.parent / ".config_snapshots").exists()
    assert not (tmp_path / "outputs").exists()
    script = (ROOT / "submit_train.sh").read_text()
    assert '"${CONFIGS}/train_pi05/PT/src/train.sbatch"' in script
    assert "TRAIN_PI05_CONFIG_RESOLVER" in script


@pytest.mark.parametrize("logging_patch", [
    {"wandb_project": ""}, {"probe_batches_per_group": 0}, {"probe_seed": -1},
    {"probe_grad_group": 2}, {"probe_grad_group": True}, {"unknown": 1},
])
def test_invalid_diagnostics_rejected(files, logging_patch):
    cfg = CONFIG.load_config(files[1]["cyclic"])
    cfg["logging"] = logging_patch
    with pytest.raises(ValueError):
        CONFIG.build_settings(cfg)


def diagnostics_module():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT / "src"))
        return module("diagnostics")


class ProbeDataset(TinyDataset):
    def __getitem__(self, index):
        import torch
        task = next(tasks[0] for tasks, lo, hi in zip(
            self.meta.episodes["tasks"], self.meta.episodes["dataset_from_index"],
            self.meta.episodes["dataset_to_index"], strict=True) if lo <= index < hi)
        return {"task": task, "x": torch.tensor([index / 24.0])}


def observer_fixture(tmp_path, mode, grad_group=0):
    import torch
    from contextlib import nullcontext

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(1, 1)
            with torch.no_grad():
                self.layer.weight.fill_(0.25)
                self.layer.bias.fill_(0.1)
        def forward(self, batch):
            return (self.layer(batch["x"]) - torch.randn_like(batch["x"])).square().mean(), {}

    cfg = SimpleNamespace(dataset=SimpleNamespace(repo_id="tiny"), batch_size=2,
                          output_dir=tmp_path, steps=12, log_freq=1, checkpoint_path=None)
    accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext)
    settings = {"cyclic_mode": mode, "cyclic_phase_steps": 3, "cyclic_group_seed": 4,
                "cyclic_probe_seed": 12345, "cyclic_probe_grad_group": grad_group,
                "cyclic_probe_batches_per_group": 2, "cyclic_n_groups": 2}
    return cfg, ProbeDataset(), Policy(), accelerator, settings


def test_probe_preserves_rng_gradients_flags_and_stored_batches(tmp_path):
    import random
    import torch

    diag = diagnostics_module()
    cfg, ds, policy, accelerator, settings = observer_fixture(tmp_path, "cyclic")
    policy.train()
    policy.layer.eval()
    for param in policy.parameters():
        param.grad = torch.full_like(param, 0.75)
    old_grads = [p.grad.clone() for p in policy.parameters()]
    # In-place preprocessing must only affect a fresh clone, not the fixed ruler.
    def preprocess(batch):
        batch["x"].add_(0.5)
        batch["task"][0] += " changed"
        return batch
    torch_state, numpy_state, python_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
    observer = diag.CurriculumObserver(cfg, ds, policy, preprocess, accelerator, None, 0, settings)
    original = observer.losses()
    observer.measure(0, initial=True)
    assert observer.losses() == original
    assert torch.equal(torch_state, torch.get_rng_state())
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])
    assert python_state == random.getstate()
    assert policy.training and not policy.layer.training
    assert all(torch.equal(p.grad, before) for p, before in zip(policy.parameters(), old_grads, strict=True))


@pytest.mark.parametrize("mode", ["iid", "cyclic"])
def test_probes_do_not_change_native_updates_and_count_committed_samples(tmp_path, mode):
    import json
    import torch
    from torch.utils.data import default_collate

    diag = diagnostics_module()
    cfg, ds, policy, accelerator, settings = observer_fixture(tmp_path, mode)
    baseline = observer_fixture(tmp_path / "unused", mode)[2]
    observer = diag.CurriculumObserver(cfg, ds, policy, lambda x: x, accelerator, None, 0, settings)
    groups = SAMPLING.build_groups(ds, 2, 4)
    indices = list(SAMPLING.BlockCyclicBatchSampler(groups, 2, 3, 12, 4)) if mode == "cyclic" else [
        [i % 24, (i + 11) % 24] for i in range(12)]
    opts = [torch.optim.Adam(p.parameters(), lr=0.01) for p in (baseline, policy)]
    for step, ids in enumerate(indices, 1):
        batch = default_collate([ds[i] for i in ids])
        for model, optimizer in zip((baseline, policy), opts, strict=True):
            torch.manual_seed(100 + step)
            if model is policy:
                observer.on_batch(batch)
            loss, _ = model(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if model is policy:
                state = torch.get_rng_state()
                observer.on_step(step)
                assert torch.equal(state, torch.get_rng_state())
        for a, b in zip(baseline.parameters(), policy.parameters(), strict=True):
            assert torch.equal(a, b)
    assert sum(observer.counts) == 24
    rows = [json.loads(line) for line in observer.metrics_path.read_text().splitlines()]
    boundaries = [row for row in rows if "probe_loss" in row]
    assert [row["step"] for row in boundaries] == [0, 3, 6, 9, 12]
    assert all("grad_cos" in row for row in boundaries[1:])
    assert boundaries[0]["probe_forget"] == {"g0": 0.0, "g1": 0.0}
    assert boundaries[-1]["cycle"]["active_group"] == (observer.active_group(12) if mode == "cyclic" else -1)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    observer.on_checkpoint(checkpoint, 12)
    cfg.checkpoint_path = checkpoint
    resumed = diag.CurriculumObserver(cfg, ds, policy, lambda x: x, accelerator, None, 12, settings)
    assert resumed.counts == observer.counts
    assert resumed.own_last == observer.own_last
    assert len([line for line in resumed.metrics_path.read_text().splitlines() if json.loads(line)["step"] == 12]) == 1


def test_iid_and_cyclic_share_identical_probe_frames_noise_and_initial_losses(tmp_path):
    diag = diagnostics_module()
    observers = []
    for mode in ("iid", "cyclic"):
        cfg, ds, policy, accelerator, settings = observer_fixture(tmp_path / mode, mode, -1)
        observers.append(diag.CurriculumObserver(cfg, ds, policy, lambda x: x, accelerator, None, 0, settings))
    assert observers[0].signature == observers[1].signature
    assert observers[0].losses() == observers[1].losses()


def test_native_observer_hooks_do_not_replace_training_or_iid_loader():
    trainer = (CONFIGS.parents[2] / "src/lerobot/scripts/lerobot_train.py").read_text()
    entrypoint = (ROOT / "src/train.py").read_text()
    assert 'if settings["cyclic_mode"] == "cyclic" else None' in entrypoint
    assert "training_observer.on_batch(batch)" in trainer
    assert "training_observer.on_step(step)" in trainer
    assert trainer.count("training_observer.on_checkpoint(checkpoint_dir, step)") == 2


def test_separate_checkpointing_variant_is_named_snapshotted_and_does_not_edit_base(files):
    source, paths, _ = files
    before = source.read_text()
    original = CONFIG.build_settings(CONFIG.load_config(paths["cyclic"]))
    raw = yaml.safe_load(paths["cyclic"].read_text())
    raw.update(gradient_checkpointing=True, run_suffix="gckpt")
    variant = paths["cyclic"].with_name("cyclic_gckpt_config.yaml")
    variant.write_text(yaml.safe_dump(raw))
    settings = CONFIG.build_settings(CONFIG.load_config(variant))
    assert settings["pt_gradient_checkpointing"] is True
    assert original["pt_gradient_checkpointing"] is False
    assert settings["pt_output_dir"].name == original["pt_output_dir"].name + "_gckpt"
    assert source.read_text() == before
    frozen = CONFIG.freeze_config(variant)
    assert CONFIG.build_settings(CONFIG.load_config(frozen))["pt_gradient_checkpointing"] is True
    assert CONFIG.build_settings(CONFIG.load_config(paths["cyclic"])) == original


@pytest.mark.parametrize("patch", [
    {"gradient_checkpointing": "true", "run_suffix": "gckpt"},
    {"gradient_checkpointing": True}, {"run_suffix": "../unsafe"},
])
def test_checkpointing_variant_rejects_ambiguous_or_unsafe_settings(files, patch):
    cfg = CONFIG.load_config(files[1]["cyclic"])
    cfg.update(patch)
    with pytest.raises(ValueError):
        CONFIG.build_settings(cfg)
