"""Real temporary checkpoint files; no user checkpoints or GPUs are touched."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from safetensors.torch import save_file
import torch

from lerobot.utils.checkpoint_retention import (
    _is_training_checkpoint, prune_local_checkpoints, validate_checkpoint_retention,
    validate_retention_mode,
)
from lerobot.utils.train_utils import update_last_checkpoint


def checkpoint(run, step):
    path = run / "checkpoints" / f"{step:06d}"
    pretrained, state = path / "pretrained_model", path / "training_state"
    pretrained.mkdir(parents=True)
    state.mkdir()
    for name, payload in (
        (pretrained / "config.json", {"type": "skill_expert"}),
        (pretrained / "train_config.json", {"scheduler": {"type": "cosine"}}),
        (pretrained / "policy_preprocessor.json", {"steps": [{"state_file": "normalizer.safetensors"}]}),
        (pretrained / "policy_postprocessor.json", {"steps": []}),
        (state / "training_step.json", {"step": step}),
        (state / "optimizer_param_groups.json", []),
        (state / "scheduler_state.json", {}),
    ):
        name.write_text(json.dumps(payload))
    for file in (pretrained / "model.safetensors", pretrained / "normalizer.safetensors",
                 state / "optimizer_state.safetensors", state / "rng_state.safetensors"):
        save_file({"weight": torch.ones(1)}, file)
    return path


def config(run, keep=2, protected=None):
    return SimpleNamespace(output_dir=run, keep_checkpoints=keep,
                           protect_checkpoints=[50000, 100000] if protected is None else protected,
                           checkpoint_archive="", policy=SimpleNamespace(type="skill_expert"))


def steps(run):
    return {int(path.name) for path in (run / "checkpoints").iterdir() if path.name.isdigit()}


@pytest.mark.parametrize("keep", [1, 2])
def test_sweep_and_resume_keep_recent_union_protected(tmp_path, keep):
    cfg = config(tmp_path, keep)
    for step in range(10000, 200001, 10000):
        path = checkpoint(tmp_path, step)
        update_last_checkpoint(path)
        prune_local_checkpoints(cfg, path)
        expected = set(range(max(10000, step - (keep - 1) * 10000), step + 1, 10000))
        expected |= {protected for protected in cfg.protect_checkpoints if protected <= step}
        assert steps(tmp_path) == expected
        assert (path.parent / "last").resolve() == path
    # Same saved policy on resume, including targets above the original budget.
    resumed = config(tmp_path, keep)
    path = checkpoint(tmp_path, 210000)
    update_last_checkpoint(path)
    prune_local_checkpoints(resumed, path)
    assert {50000, 100000, 210000} <= steps(tmp_path)


@pytest.mark.parametrize("broken", ["model", "optimizer", "normalizer", "step", "scheduler", "last"])
def test_failed_save_never_prunes_old_recovery_checkpoints(tmp_path, broken):
    for step in (10000, 20000):
        update_last_checkpoint(checkpoint(tmp_path, step))
    path = checkpoint(tmp_path, 30000)
    if broken != "last":
        update_last_checkpoint(path)
    files = {"model": "pretrained_model/model.safetensors",
             "optimizer": "training_state/optimizer_state.safetensors",
             "normalizer": "pretrained_model/normalizer.safetensors",
             "step": "training_state/training_step.json", "scheduler": "training_state/scheduler_state.json"}
    if broken in files:
        (path / files[broken]).write_bytes(b"truncated")
    assert prune_local_checkpoints(config(tmp_path, keep=1), path) == []
    assert steps(tmp_path) == {10000, 20000, 30000}


def test_partial_foreign_symlink_and_unrelated_files_are_preserved(tmp_path):
    run = tmp_path / "run"
    old = checkpoint(run, 10000)
    foreign = checkpoint(run, 20000)
    (foreign / "pretrained_model/config.json").write_text('{"type":"pi05"}')
    partial = checkpoint(run, 30000)
    (partial / "training_state/optimizer_state.safetensors").unlink()
    external = checkpoint(tmp_path / "external", 40000)
    (run / "checkpoints/040000").symlink_to(external)
    note = run / "checkpoints/readme.txt"
    note.write_text("keep")
    current = checkpoint(run, 50000)
    update_last_checkpoint(current)
    assert prune_local_checkpoints(config(run, keep=1, protected=[]), current) == [old]
    assert foreign.is_dir() and partial.is_dir() and external.is_dir() and note.is_file()


def test_disabled_and_archived_modes_do_not_delete(tmp_path):
    old, current = checkpoint(tmp_path, 10000), checkpoint(tmp_path, 20000)
    update_last_checkpoint(current)
    assert prune_local_checkpoints(config(tmp_path, keep=-1), current) == []
    cfg = config(tmp_path, keep=1, protected=[])
    cfg.checkpoint_archive = "scratch2"
    assert prune_local_checkpoints(cfg, current) == []
    assert old.is_dir()


def test_protected_initial_save_can_have_uninitialized_adam_state(tmp_path):
    initial = checkpoint(tmp_path, 0)
    save_file({}, initial / "training_state/optimizer_state.safetensors")
    assert _is_training_checkpoint(initial)
    current = checkpoint(tmp_path, 10000)
    update_last_checkpoint(current)
    assert prune_local_checkpoints(config(tmp_path, keep=1, protected=[0]), current) == []
    assert initial.is_dir()


def test_sharded_model_must_have_all_complete_shards(tmp_path):
    path = checkpoint(tmp_path, 10000)
    pretrained = path / "pretrained_model"
    (pretrained / "model.safetensors").rename(pretrained / "shard1.safetensors")
    (pretrained / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"a": "shard1.safetensors", "b": "shard2.safetensors"}}))
    assert not _is_training_checkpoint(path)
    save_file({"b": torch.ones(2)}, pretrained / "shard2.safetensors")
    assert _is_training_checkpoint(path)


def test_refuses_symlinked_root_and_cross_run_callback(tmp_path):
    run, external = tmp_path / "run", tmp_path / "external"
    run.mkdir(); external.mkdir()
    (run / "checkpoints").symlink_to(external)
    with pytest.raises(ValueError, match="symlinked"):
        prune_local_checkpoints(config(run), external / "001000")
    (run / "checkpoints").unlink()
    with pytest.raises(ValueError, match="Unexpected"):
        prune_local_checkpoints(config(run), external / "001000")


@pytest.mark.parametrize("keep", [0, -2, True, "2", 1.5])
def test_invalid_counts_rejected(keep):
    with pytest.raises(ValueError, match="keep_checkpoints"):
        validate_checkpoint_retention(keep, [])


@pytest.mark.parametrize("protected", [None, "50000", [-1], [True], [1.5], ["50000"]])
def test_invalid_protection_rejected(protected):
    with pytest.raises(ValueError, match="protect_checkpoints"):
        validate_checkpoint_retention(2, protected)


def test_local_retention_is_explicit_vsa_only_and_archive_mode_stays_separate():
    assert validate_retention_mode(2, [100000, 50000, 50000], "", "skill_expert") == [50000, 100000]
    assert validate_retention_mode(2, [], "scratch2", "skill_aux") == []
    with pytest.raises(ValueError, match="checkpoint_archive"):
        validate_retention_mode(2, [], "", "skill_aux")
    with pytest.raises(ValueError, match="without checkpoint_archive"):
        validate_retention_mode(2, [50000], "scratch2", "skill_expert")


def test_native_config_validates_serializes_and_resumes_retention(tmp_path):
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.skill_expert.configuration_skill_expert import SkillExpertConfig

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/no_dataset_access"),
        policy=SkillExpertConfig(device="cpu", push_to_hub=False),
        output_dir=tmp_path / "future_run", steps=200000,
        keep_checkpoints=2, protect_checkpoints=[100000, 50000, 50000],
    )
    cfg.validate()
    assert cfg.protect_checkpoints == [50000, 100000]
    metadata = tmp_path / "metadata"
    cfg.save_pretrained(metadata)
    restored = TrainPipelineConfig.from_pretrained(metadata)
    assert restored.keep_checkpoints == 2
    assert restored.protect_checkpoints == [50000, 100000]
    # Explicit CLI changes still override the previous run's saved settings.
    resumed = TrainPipelineConfig.from_pretrained(metadata, cli_args=["--keep_checkpoints=1"])
    assert resumed.keep_checkpoints == 1 and resumed.protect_checkpoints == [50000, 100000]


def test_native_hooks_only_run_after_save_and_last_update():
    source = Path(__file__).resolve().parents[2] / "src/lerobot/scripts/lerobot_train.py"
    tree = ast.parse(source.read_text())
    hits = 0
    for node in ast.walk(tree):
        for _, body in ast.iter_fields(node):
            if not isinstance(body, list): continue
            for index, statement in enumerate(body):
                if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call): continue
                if isinstance(statement.value.func, ast.Name) and statement.value.func.id == "update_last_checkpoint":
                    assert [body[index + offset].value.func.id for offset in (1, 2)] == ["queue_checkpoint_archive", "prune_local_checkpoints"]
                    hits += 1
    assert hits == 2
