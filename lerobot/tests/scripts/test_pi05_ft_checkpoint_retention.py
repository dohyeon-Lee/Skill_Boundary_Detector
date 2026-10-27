"""pi05 FT keeps latest N plus protected steps after each complete native save."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_pi05/FT/src/train.py"
SPEC = importlib.util.spec_from_file_location("pi05_ft_retention_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_checkpoint(run, step):
    path = run / "checkpoints" / f"{step:06d}"
    for name in (
        "pretrained_model/config.json", "pretrained_model/train_config.json",
        "pretrained_model/policy_preprocessor.json", "pretrained_model/policy_postprocessor.json",
        "pretrained_model/model.safetensors", "training_state/training_step.json",
        "training_state/optimizer_state.safetensors", "training_state/optimizer_param_groups.json",
        "training_state/scheduler_state.json", "training_state/rng_state.safetensors",
    ):
        file = path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"step": step} if name.endswith("training_step.json") else {"type": "pi05"}
        file.write_text(json.dumps(payload))
    return path


def test_ft_callback_prunes_after_native_save_and_keeps_latest_two_union_protected(tmp_path):
    run = tmp_path / "run"
    calls = []

    def native_update_last(path):
        assert MODULE._is_training_checkpoint(path)
        last = path.parent / "last"
        if last.is_symlink():
            last.unlink()
        last.symlink_to(path.name)
        calls.append(path)
        return last

    native = SimpleNamespace(update_last_checkpoint=native_update_last)
    settings = {"ft_output_dir": run, "ft_keep_checkpoints": 2,
                "ft_protect_checkpoints": [1000, 3000, 5000, 10000]}
    MODULE.install_retention(native, settings)
    for step in range(1000, 10001, 1000):
        path = make_checkpoint(run, step)
        assert native.update_last_checkpoint(path).resolve() == path
    assert len(calls) == 10
    assert {int(p.name) for p in (run / "checkpoints").iterdir() if p.name.isdigit()} == {1000, 3000, 5000, 9000, 10000}
    assert (run / "checkpoints/last").resolve().name == "010000"
    # A resumed process installs the same hook and preserves the old protected saves.
    resumed = SimpleNamespace(update_last_checkpoint=native_update_last)
    MODULE.install_retention(resumed, settings)
    resumed.update_last_checkpoint(make_checkpoint(run, 11000))
    assert not (run / "checkpoints/009000").exists()
    assert (run / "checkpoints/001000").is_dir()
    assert (run / "checkpoints/last").resolve().name == "011000"


def test_retention_preserves_partial_foreign_and_symlinked_saves(tmp_path):
    run = tmp_path / "run"
    for step in (1000, 2000, 3000):
        make_checkpoint(run, step)
    partial = make_checkpoint(run, 4000)
    (partial / "training_state/optimizer_state.safetensors").unlink()
    foreign = make_checkpoint(run, 5000)
    (foreign / "pretrained_model/config.json").write_text('{"type": "groot"}')
    external = make_checkpoint(tmp_path / "external", 6000)
    (run / "checkpoints/006000").symlink_to(external)
    assert MODULE.prune_checkpoints(run, -1, []) == []
    assert [path.name for path in MODULE.prune_checkpoints(run, 2, [])] == ["001000"]
    assert partial.is_dir() and foreign.is_dir() and external.is_dir()
    assert (run / "checkpoints/006000").is_symlink()


def test_sharded_weights_and_malformed_training_state_are_not_treated_as_complete(tmp_path):
    path = make_checkpoint(tmp_path, 1000)
    (path / "pretrained_model/model.safetensors").unlink()
    index = path / "pretrained_model/model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a": "shard-1.safetensors", "b": "shard-2.safetensors"}}))
    (index.parent / "shard-1.safetensors").write_text("weights")
    assert not MODULE._is_training_checkpoint(path)
    (index.parent / "shard-2.safetensors").write_text("weights")
    assert MODULE._is_training_checkpoint(path)
    (path / "training_state/training_step.json").write_text('{"step": 999}')
    assert not MODULE._is_training_checkpoint(path)


def test_retention_refuses_symlinked_root_or_callback_outside_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "checkpoints").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlinked"):
        MODULE.prune_checkpoints(run, 2, [])
    called = []
    native = SimpleNamespace(update_last_checkpoint=lambda path: called.append(path))
    MODULE.install_retention(native, {"ft_output_dir": run, "ft_keep_checkpoints": 2, "ft_protect_checkpoints": []})
    with pytest.raises(ValueError, match="Unexpected"):
        native.update_last_checkpoint(tmp_path / "another_run/checkpoints/001000")
    assert called == []
