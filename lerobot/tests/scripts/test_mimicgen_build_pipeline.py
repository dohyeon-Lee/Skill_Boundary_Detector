"""The one-call Coffee build owns SBD prerequisites without another Slurm step."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "lerobot/skillVLA/mimicgen/configs/train_skillVLA/build_data/src"


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, SRC / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = load("coffee_pipeline_test", "sbd_pipeline.py")
B = load("coffee_builder_test", "build_dataset.py")


@pytest.mark.parametrize("percent, steps, expected", [(30, 20, 150), (0, 20, 20), (0.5, 20, 3)])
def test_jitter_storage_matches_libero_percentage_capacity(percent, steps, expected):
    assert B.jitter_storage_pmax({"max_length": 500, "jitter_max_steps": steps,
                                 "jitter_max_percent": percent}) == expected


def test_primary_target_and_mode_override_the_standalone_sbd_recipe(monkeypatch):
    seen = {}
    def resolve(source):
        seen.update(source)
        return {"skillset_single_gpu_workers": 4, "skillset_cpus_per_task": 8}
    monkeypatch.setattr(P, "build_data_settings", resolve)
    cfg = {"target_dataset": "new_coffee", "skillvla_data_mode": "pt", "slurm": {"cpus": 8},
           "project_root": "/project", "dataset_root": "new_data", "outputs_root": "new_outputs",
           "sbd_recipe": {"target_dataset": "old", "dataset_root": "old_data", "skillvla_data_mode": "ft"}}
    P.settings(cfg)
    assert seen["target_dataset"] == "new_coffee" and seen["dataset_root"] == "new_data"
    assert seen["skillvla_data_mode"] == "pt" and seen["skillset_global_threshold_source"] == ""


def test_snapshot_keeps_the_submitted_sbd_recipe_when_live_yaml_changes(tmp_path, monkeypatch):
    source = tmp_path / "lerobot/skillVLA/mimicgen/configs/train_skills/build_data/build_data_config.yaml"
    source.parent.mkdir(parents=True)
    source.write_text("skillset_single_gpu_workers: 4\n")
    cfg_path = tmp_path / "snapshot.yaml"
    cfg_path.write_text("target_dataset: coffee\n")
    monkeypatch.setattr(B.sbd_pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(B, "load_config", lambda path: yaml.safe_load(Path(path).read_text()))
    monkeypatch.setattr(B, "resolve", lambda cfg: {})
    monkeypatch.setattr(sys, "argv", ["build", "--config", str(cfg_path), "--freeze-sbd"])
    B.main()
    source.write_text("skillset_single_gpu_workers: 9\n")
    frozen = yaml.safe_load(cfg_path.read_text())
    assert B.sbd_pipeline.recipe(frozen)["skillset_single_gpu_workers"] == 4


@pytest.mark.parametrize("finish", [True, False])
def test_missing_sbd_runs_in_the_same_allocation_and_gates_downstream(tmp_path, monkeypatch, finish):
    raw = tmp_path / "raw"
    metadata = raw / "meta/episodes/chunk-000"
    metadata.mkdir(parents=True)
    (raw / 'meta/info.json').write_text('{"total_episodes":2}')
    pd.DataFrame({"episode_index": [0, 1]}).to_parquet(metadata / "file-000.parquet")
    pd.DataFrame({"task_index": [0]}).to_parquet(raw / "meta/tasks.parquet")
    policy = tmp_path / "dp"
    policy.mkdir()
    cfg = {"target_dataset": "coffee", "skillvla_data_mode": "ft_own"}
    resolved = {"raw_dataset_dir": raw, "dp_policy_path": policy, "skillset_dir": tmp_path / "sbd"}
    monkeypatch.setattr(P, "settings", lambda c: (cfg, resolved))
    monkeypatch.setattr(P, "validate_manifest", lambda *args: None)
    complete = [False]
    monkeypatch.setattr(P, "is_complete", lambda s: complete[0])
    monkeypatch.setenv("SLURM_JOB_ID", "74714")
    envs = []
    class Runner:
        workers = 4
        def __init__(self, env, log_dir):
            envs.append(env)
        def run(self):
            assert envs[0]["SLURM_JOB_ID"] == "74714"
            assert envs[0]["LIBERO_SINGLE_GPU_BUILD"] == "true"
            assert envs[0]["LIBERO_EXPECTED_EPISODES"] == "2"
            assert json.loads(envs[0]["LIBERO_TASK_INDICES_JSON"]) == [0]
            assert yaml.safe_load(Path(envs[0]["TRAIN_SKILLS_CONFIG"]).read_text()) == cfg
            complete[0] = finish
        def stop(self, *args):
            pass
    monkeypatch.setattr(P, "_load", lambda *args: SimpleNamespace(LIBEROSharedGPUBuilder=Runner))
    if finish:
        P.ensure(cfg, tmp_path / "output")
    else:
        with pytest.raises(RuntimeError, match="SBD is incomplete"):
            P.ensure(cfg, tmp_path / "output")
    assert len(envs) == 1


def test_completed_sbd_is_reused_without_starting_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "settings", lambda cfg: ({}, {}))
    monkeypatch.setattr(P, "validate_manifest", lambda *args: None)
    monkeypatch.setattr(P, "is_complete", lambda resolved: True)
    monkeypatch.setattr(P, "progress_metadata", lambda *args: {})
    monkeypatch.setattr(P, "_load", lambda *args: pytest.fail("No worker should start"))
    P.ensure({}, tmp_path / "output")


def test_sbd_failure_never_starts_fsq_or_dataset_generation(tmp_path, monkeypatch):
    model = tmp_path / "FSQ.pt"
    model.write_bytes(b"fixture")
    (tmp_path / "fsq_meta.json").write_text(json.dumps({"fsq_levels": [3, 3, 3]}))
    monkeypatch.setattr(B, "resolve", lambda cfg: {"raw": tmp_path / "raw", "output": tmp_path / "out",
                                                "skillset": tmp_path / "skills", "model": model})
    def fail(*args):
        raise RuntimeError("SBD failed")
    monkeypatch.setattr(B.sbd_pipeline, "ensure", fail)
    monkeypatch.setattr(B.subprocess, "run", lambda *a, **kw: pytest.fail("No encoding after SBD failure"))
    with pytest.raises(RuntimeError, match="SBD failed"):
        B.build({"fsq_levels": [3, 3, 3], "max_length": 500,
                 "jitter_max_steps": 20, "jitter_max_percent": 30})
