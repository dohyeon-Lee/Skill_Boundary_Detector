"""Native save/cleanup integration with tiny temporary files; no torch/GPU/network."""
from __future__ import annotations

import ast
import importlib.util
from os.path import join
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "lerobot/examples/libero/configs/train_molmoact2/PT/src/checkpoint_retention.py"
SPEC = importlib.util.spec_from_file_location("molmo_checkpoint_retention", SOURCE)
RETENTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RETENTION)
PROTECTED = [30000, 50000, 100000]


@pytest.fixture
def trainer(tmp_path):
    upstream = ROOT / "third_party/molmoact2/experiments/olmo/train/trainer.py"
    if not upstream.is_file():
        pytest.skip("Original MolmoAct2 checkout is not installed")
    definition = next(node for node in ast.parse(upstream.read_text()).body
                      if isinstance(node, ast.ClassDef) and node.name == "Trainer")
    namespace = dict(join=join, gc_cuda=lambda: None, barrier=lambda: None,
                     get_fs_local_rank=lambda: 0, torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda _: None)),
                     CheckpointType=SimpleNamespace(sharded="sharded", sharded_ephemeral="ephemeral"))
    removed = []
    root = tmp_path / "run"
    root.mkdir()

    def clear_directory(path):
        assert Path(path).resolve().parent == root.resolve()
        removed.append(Path(path).name)
        if Path(path).is_dir():
            shutil.rmtree(path)

    namespace["clear_directory"] = clear_directory
    # Execute the real native save ordering, list mutation, and filesystem-rank
    # barrier logic, with only CUDA/large tensors and the writer replaced.
    for name in ("save_checkpoint", "remove_checkpoints", "_remove_sharded_checkpoint"):
        method = next(node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == name)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(upstream), "exec"), namespace)

    class NativeHarness:
        save_checkpoint = namespace["save_checkpoint"]
        remove_checkpoints = RETENTION.protected_removal(namespace["remove_checkpoints"], PROTECTED)
        _remove_sharded_checkpoint = namespace["_remove_sharded_checkpoint"]

        def trainer_state_dict(self):
            # Native state is serialized BEFORE cleanup; it may contain stale
            # registered paths when a new process resumes from this checkpoint.
            self.saved_state = list(self.checkpoints)
            return {"checkpoints": self.saved_state}

    instance = NativeHarness()
    instance.cfg = SimpleNamespace(save_folder=str(root), save_num_checkpoints_to_keep=2)
    instance.checkpoints, instance.ephemeral_checkpoints = [], []
    instance.optim = SimpleNamespace(zero_grad=lambda **_: None)
    instance.fsdp_model, instance.device = None, None
    instance.removed = removed

    def save(path, *_, **kwargs):
        for name in ("config.yaml", "model_and_optim/.metadata", "train/rank0.pt"):
            file = Path(path) / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("fixture")

    instance.checkpointer = SimpleNamespace(save=save)
    return instance


def existing_steps(trainer):
    return {int(path.name[4:]) for path in Path(trainer.cfg.save_folder).iterdir()
            if path.name.startswith("step") and path.name[4:].isdigit()}


def save_steps(trainer, steps):
    for step in steps:
        trainer.global_step = step
        trainer.save_checkpoint("sharded")


def test_native_saves_keep_latest_two_union_protected_with_full_state(trainer):
    save_steps(trainer, range(10000, 100001, 10000))
    assert existing_steps(trainer) == {30000, 50000, 90000, 100000}
    for checkpoint in trainer.checkpoints:
        assert (Path(checkpoint) / "model_and_optim/.metadata").is_file()
        assert (Path(checkpoint) / "train/rank0.pt").is_file()
    assert len(trainer.removed) == 6


def test_resuming_from_native_pre_cleanup_state_keeps_protection(trainer):
    save_steps(trainer, range(10000, 60001, 10000))
    assert existing_steps(trainer) == {30000, 50000, 60000}
    trainer.checkpoints = list(trainer.saved_state)  # includes a pruned step40k record
    save_steps(trainer, range(70000, 100001, 10000))
    assert existing_steps(trainer) == {30000, 50000, 90000, 100000}


def test_failed_native_save_does_not_prune_previous_checkpoints(trainer):
    save_steps(trainer, [10000, 20000])
    trainer.checkpointer.save = lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
    trainer.global_step = 30000
    with pytest.raises(OSError, match="disk full"):
        trainer.save_checkpoint("sharded")
    assert existing_steps(trainer) == {10000, 20000} and trainer.removed == []


def test_keep_all_still_disables_pruning(trainer):
    trainer.cfg.save_num_checkpoints_to_keep = -1
    save_steps(trainer, range(10000, 100001, 10000))
    assert len(existing_steps(trainer)) == 10 and trainer.removed == []


def test_ephemeral_cleanup_does_not_acquire_regular_protection(trainer):
    trainer.global_step = 30000
    trainer.save_checkpoint("ephemeral")
    trainer.global_step = 40000
    trainer.save_checkpoint("ephemeral")
    assert existing_steps(trainer) == {40000}


def test_final_model_only_save_is_not_scanned_or_pruned(trainer):
    model_only = Path(trainer.cfg.save_folder) / "step100000-unsharded"
    model_only.mkdir()
    (model_only / "model.pt").write_text("fixture")
    save_steps(trainer, range(10000, 100001, 10000))
    assert (model_only / "model.pt").read_text() == "fixture"


def test_latest_is_numeric_and_protected_latest_does_not_add_extra_slots(tmp_path):
    paths = [str(tmp_path / f"step{s}") for s in [50000, 90000, 30000, 100000]]
    indices = RETENTION.removal_indices(paths, tmp_path, 2, [100000])
    assert [paths[index] for index in indices] == [str(tmp_path / "step30000"), str(tmp_path / "step50000")]


@pytest.mark.parametrize("name", ["step123-unsharded", "checkpoint123", "../other/step123"])
def test_unsafe_paths_are_rejected_before_any_deletion(tmp_path, name):
    root = tmp_path / "run"
    paths = [str(root / "step1"), str(root / name), str(root / "step3")]
    with pytest.raises(ValueError, match="Refusing checkpoint cleanup"):
        RETENTION.removal_indices(paths, root, 2, PROTECTED)


def test_symlink_checkpoint_is_never_deleted(tmp_path):
    root, other = tmp_path / "run", tmp_path / "other"
    root.mkdir()
    other.mkdir()
    (root / "step1").symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="Refusing checkpoint cleanup"):
        RETENTION.removal_indices([root / "step1"], root, 2, PROTECTED)
    assert other.is_dir()


def test_install_can_replace_or_disable_policy_without_stacking(monkeypatch, tmp_path):
    removed = []

    class FakeTrainer:
        def remove_checkpoints(self, checkpoints, keep):
            removed.append("native")

        def _remove_sharded_checkpoint(self, index, checkpoints):
            removed.append(Path(checkpoints.pop(index)).name)

    original = FakeTrainer.remove_checkpoints
    monkeypatch.setitem(sys.modules, "olmo.train.trainer", SimpleNamespace(Trainer=FakeTrainer))
    RETENTION.install([10000])
    first = FakeTrainer.remove_checkpoints
    RETENTION.install([20000])
    assert FakeTrainer.remove_checkpoints._sbd_original_remove is original
    assert FakeTrainer.remove_checkpoints is not first
    instance = FakeTrainer()
    instance.cfg = SimpleNamespace(save_folder=str(tmp_path))
    instance.checkpoints = [str(tmp_path / f"step{s}") for s in (10000, 20000, 30000, 40000)]
    instance.remove_checkpoints(instance.checkpoints, 2)
    assert removed == ["step10000"]  # new list only, not the old protected 10k
    RETENTION.install([])
    assert FakeTrainer.remove_checkpoints is original
