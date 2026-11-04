"""Offline Predictor reports: no VLM download, GPU, MuJoCo or Slurm required."""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

CONFIGS = Path(__file__).resolve().parents[2] / "examples/libero/configs"
EVAL = CONFIGS / "train_skillVLA/FT/eval"
sys.path.insert(0, str(EVAL / "src"))
import ft_pred_eval as offline  # noqa: E402
import ft_pred_eval_config as resolver  # noqa: E402


def _dataset(tmp_path):
    run = tmp_path / "dataset_filtered/skillvla_dataset/libero_10_full_full/test_run"
    data = run / "skillvla/data/chunk-000"
    data.mkdir(parents=True)
    (run / "skillvla/meta").mkdir()
    info = {
        "skill_fsq_levels": [3, 3, 3],
        "skill_code_space_id": "same",
        "proprio_grounding": "episode_start_xyz",
        "repo_id": "test/offline",
    }
    (run / "skillvla/meta/info.json").write_text(json.dumps(info))
    episodes = [10] * 4 + [20] * 4
    table = pa.table(
        {
            "index": list(range(8)),
            "episode_index": episodes,
            "task_index": [0] * 8,
            "frame_index": [0, 1, 2, 3] * 2,
            "skill_index": [0, 0, 1, 1] * 2,
            "skill_sequence": [[1, 2, 27]] * 8,
            "skill_sequence_len": [3] * 8,
            "skill_initial_frame": [[0, 2, -1]] * 8,
            "skill_ds": [0, 1, 0, 1] * 2,
            "observation.state": [[i / 10, 0.2, 1.0, 0, 0, 0, 0.1, 0.1] for i in range(8)],
        }
    )
    pq.write_table(table, data / "file-000.parquet")
    np.savez_compressed(
        run / "skill_latents.npz",
        episode_id=[10, 10, 20, 20],
        skill_index=[0, 1, 0, 1],
        frame_start=[0, 2, 0, 2],
        frame_end=[2, 4, 2, 4],
        tokens=[1, 2, 1, 2],
    )
    np.savez_compressed(
        run.parent / "eval_init_states.npz",
        episode_index=[10, 20],
        scene_file=["scene_a.hdf5", "scene_b.hdf5"],
        demo=["demo_1", "demo_2"],
    )
    training = run.parent.parent / "libero_10_full_1"
    training.mkdir()
    # Same demo as episode 10, but a completely different local episode number.
    np.savez_compressed(
        training / "eval_init_states.npz", episode_index=[999], scene_file=["scene_a.hdf5"], demo=["demo_1"]
    )
    return run, info, table.to_pylist()


def _checkpoint(tmp_path, info):
    path = tmp_path / "checkpoint/001000/pretrained_model"
    path.mkdir(parents=True)
    config = {
        "type": "skill_aux",
        "train_skill_predictor": True,
        "skill_predictor_end_state_mode": "xyz",
        "skill_fsq_levels": [3, 3, 3],
        "skill_code_space_id": "same",
        "proprio_grounding": info["proprio_grounding"],
        "dtype": "bfloat16",
        "skill_predictor_start_proprio": False,
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / "policy_preprocessor.json").write_text('{"steps": []}')
    (path / "train_config.json").write_text('{"dataset": {}}')
    (path / "model.safetensors").write_bytes(b"fake weights; inference is mocked")
    return path


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    run, info, rows = _dataset(tmp_path)
    checkpoint = _checkpoint(tmp_path, info)
    monkeypatch.setattr(
        resolver, "_resolve_predictor_checkpoint", lambda *args: (checkpoint, str(args[-1]).zfill(6))
    )
    monkeypatch.setattr(resolver, "EVAL_DIR", tmp_path / "FT/eval")
    config = {
        "project_root": str(tmp_path),
        "dataset_root": "dataset_filtered",
        "outputs_root": "outputs_filtered",
        "dataset": {
            "source": "libero_10_full_full",
            "run": "test_run",
            "exclude_sources": ["libero_10_full_1"],
            "episodes_per_task": 0,
        },
        "models": [{"label": "test", "predictor_dir": "model", "predictor_checkpoint": "001000"}],
        "device": "cpu",
        "visualization": {"enabled": False},
        "output_name": "test_report",
        "train_partition": ["gpu_a", "gpu_b"],
        "train_qos": "base_qos",
        "slurm_cpu": {"partition": "dell_cpu", "qos": "cpu_qos"},
    }
    return config, run, info, checkpoint, rows


def test_config_separates_outputs_and_cpu_allocations(fixture):
    config, *_ = fixture
    settings = resolver.build_settings(config)
    assert Path(settings["output_dir"]).parts[-2:] == ("outputs_predictor", "test_report")
    assert settings["slurm"]["partition"] == "dell_cpu"
    assert settings["slurm"]["qos"] == "cpu_qos"
    config["device"] = "cuda"
    settings = resolver.build_settings(config)
    assert settings["slurm"]["partition"] == "gpu_a,gpu_b"
    assert settings["slurm"]["qos"] == "base_qos"


def test_preskill_source_is_selectable_and_defaults_to_gt(fixture):
    config, *_ = fixture
    config["batch_size"] = 8
    settings = resolver.build_settings(config)
    assert settings["preskill_source"] == "gt"
    assert settings["inference_batch_size"] == 8
    config["preskill_source"] = "predicted"
    settings = resolver.build_settings(config)
    assert settings["preskill_source"] == "predicted"
    assert settings["batch_size"] == 8 and settings["inference_batch_size"] == 1


@pytest.mark.parametrize("value", ["auto", "pred", "", None, True, ["gt"]])
def test_invalid_preskill_source_is_not_silently_gt(fixture, value):
    config, *_ = fixture
    config["preskill_source"] = value
    with pytest.raises(ValueError, match="preskill_source must be gt\\|predicted"):
        resolver.build_settings(config)


def _history_samples(episodes=(10, 20), count=2):
    return [
        {"uid": f"{ep}-{i}", "task_id": 0, "episode_index": ep, "skill_index": i,
         "frame_start": i * 10, "previous_gt_skill": -1 if i == 0 else 19}
        for ep in episodes for i in range(count)
    ]


def test_predicted_chunks_sort_reset_bos_and_never_teacher_force():
    samples = list(reversed(_history_samples()))
    records, inputs, order = {}, [], []
    for chunk, previous in offline.inference_chunks(
        samples, records, source="predicted", batch_size=8, wants_image=lambda s: False
    ):
        assert len(chunk) == len(previous) == 1
        sample = chunk[0][0]
        inputs.extend(previous)
        order.append(sample["uid"])
        records[sample["uid"]] = {"pred_skill": previous[0] + 3,
                                   "preskill_source": "predicted", "previous_skill_input": previous[0]}
    assert inputs == [-1, 2, -1, 2]
    assert order == ["10-0", "10-1", "20-0", "20-1"]
    # GT previous code is 19; it must never replace the wrong previous prediction 2.
    assert 19 not in inputs


@pytest.mark.parametrize("damage", ["missing_first", "gap", "duplicate", "reversed_frames"])
def test_predicted_chunks_reject_broken_episode_chains_before_inference(damage):
    samples = _history_samples(episodes=(10,), count=3)
    if damage == "missing_first":
        samples = samples[1:]
    elif damage == "gap":
        samples = [samples[0], samples[2]]
    elif damage == "duplicate":
        samples.append(dict(samples[0]))
    else:
        samples[1]["frame_start"] = -1
    with pytest.raises(ValueError, match="contiguous|chronological"):
        next(offline.inference_chunks(samples, {}, source="predicted", batch_size=8, wants_image=lambda s: False))


@pytest.mark.parametrize("new_prediction,expected_work", [(5, ["10-1", "10-2"]), (6, ["10-1"])])
def test_predicted_partial_resume_restores_prefix_and_checks_cached_successors(new_prediction, expected_work):
    samples = _history_samples(episodes=(10,), count=3)
    records = {
        "10-0": {"preskill_source": "predicted", "previous_skill_input": -1, "pred_skill": 2},
        "10-2": {"preskill_source": "predicted", "previous_skill_input": 6, "pred_skill": 9},
    }
    work = []
    for chunk, previous in offline.inference_chunks(
        samples, records, source="predicted", batch_size=8, wants_image=lambda s: False
    ):
        uid = chunk[0][0]["uid"]
        work.append(uid)
        if uid == "10-1":
            assert previous == [2]
        else:
            assert previous == [new_prediction]
        records[uid] = {"preskill_source": "predicted", "previous_skill_input": previous[0],
                        "pred_skill": new_prediction if uid == "10-1" else 10}
    assert work == expected_work


def test_gt_chunks_keep_existing_batching_and_gt_inputs():
    samples = _history_samples()
    chunks = list(offline.inference_chunks(samples, {}, source="gt", batch_size=8, wants_image=lambda s: False))
    assert len(chunks) == 1 and len(chunks[0][0]) == 4
    assert chunks[0][1] == [-1, 19, -1, 19]


def test_history_mode_is_part_of_resume_signature_with_legacy_gt_compatibility(fixture):
    config, *_ = fixture
    settings = resolver.build_settings(config)
    samples, _ = offline.load_samples(settings)
    panel = settings["models"][0]
    legacy = dict(settings)
    legacy.pop("preskill_source")
    assert offline.panel_signature(settings, panel, samples) == offline.panel_signature(legacy, panel, samples)
    assert offline.panel_signature(settings, panel, samples) != offline.panel_signature(
        {**settings, "preskill_source": "predicted"}, panel, samples
    )


@pytest.mark.parametrize("source", ["gt", "predicted"])
def test_history_end_to_end_inputs_targets_reports_and_resume(fixture, monkeypatch, source):
    config, _, _, _, rows = fixture
    config["dataset"]["exclude_sources"] = []
    config.update(preskill_source=source, batch_size=8)
    settings = resolver.build_settings(config)
    seen, loaded = [], []

    class Dataset:
        def __getitem__(self, index):
            row = rows[index]
            return {**row, "task": "task", "observation.state": torch.tensor(row["observation.state"]),
                    "observation.images.image": torch.zeros(3, 8, 8),
                    "observation.images.wrist_image": torch.zeros(3, 8, 8)}

    def load(*args):
        loaded.append(True)
        return None, lambda x: x, Dataset()

    def infer(predictor, processed, gt_codes):
        previous = processed["skill_previous_code"].tolist()
        seen.extend(previous)
        return np.array([(v + 3) % 27 for v in previous]), np.zeros((len(previous), 3)), np.zeros((len(previous), 3))

    monkeypatch.setattr(offline, "load_predictor", load)
    monkeypatch.setattr(offline, "infer_batch", infer)
    offline.run(settings)
    assert seen == ([-1, 1, -1, 1] if source == "gt" else [-1, 2, -1, 2])
    assert len(loaded) == 1
    folder = Path(settings["output_dir"]) / settings["models"][0]["folder"]
    records = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
    assert [r["gt_skill"] for r in records] == [1, 2, 1, 2]
    assert [r["previous_skill_input"] for r in records] == seen
    assert all(r["preskill_source"] == source for r in records)
    report = (folder / "index.html").read_text()
    assert "Preskill input:" in report
    assert ("teacher forcing" if source == "gt" else "Previous skill is the previous prediction") in report
    summary = json.loads((folder / "metrics.json").read_text())
    assert summary["preskill_source"] == source
    # A complete resume uses persisted predictions, with no model load or GT fallback.
    monkeypatch.setattr(offline, "load_predictor", lambda *args: pytest.fail("No model needed"))
    offline.run(settings)
    # Partial resume: legacy GT records omit the new fields; predicted records restore actual history.
    if source == "gt":
        for path in (folder / "records").glob("*.json"):
            record = json.loads(path.read_text())
            record.pop("preskill_source")
            record.pop("previous_skill_input")
            path.write_text(json.dumps(record))
        missing = records[0]
    else:
        missing = records[1]
    (folder / "records" / f"{missing['uid']}.json").unlink()
    seen.clear()
    monkeypatch.setattr(offline, "load_predictor", load)
    offline.run(settings)
    assert seen == ([-1] if source == "gt" else [2])
    other_source = "predicted" if source == "gt" else "gt"
    with pytest.raises(ValueError, match="Refusing to mix"):
        offline.run({**settings, "preskill_source": other_source})


@pytest.mark.parametrize("output_name", ["predictor_fullvlm_eval", "pred-mode2_10k.v2"])
def test_custom_output_name_only_changes_report_destination(fixture, output_name):
    config, *_ = fixture
    original = resolver.build_settings(config)
    config["output_name"] = output_name
    settings = resolver.build_settings(config)
    assert Path(settings["output_dir"]) == Path(original["output_dir"]).with_name(output_name)
    assert settings["models"] == original["models"]
    assert not Path(settings["output_dir"]).exists()


def test_requires_completed_build_not_just_oracle_map(fixture):
    config, run, *_ = fixture
    (run / "skillvla/meta/info.json").unlink()
    with pytest.raises(FileNotFoundError, match="eval_init_states.npz alone is not enough"):
        resolver.build_settings(config)


@pytest.mark.parametrize(
    "field,value", [("device", "auto"), ("batch_size", 0), ("output_name", "../outputs"), ("cpu_threads", 0)]
)
def test_rejects_invalid_settings(fixture, field, value):
    config, *_ = fixture
    config[field] = value
    with pytest.raises(ValueError):
        resolver.build_settings(config)


def test_checkpoint_lists_and_integer_padding(fixture):
    config, *_ = fixture
    config["models"][0]["predictor_checkpoint"] = [1000, 2000]
    settings = resolver.build_settings(config)
    assert [m["checkpoint"] for m in settings["models"]] == ["001000", "002000"]
    assert len({m["folder"] for m in settings["models"]}) == 2


@pytest.mark.parametrize("requested,expected", [(1, 1), (2, 2), (5, 5), (10, 5)])
def test_gpu_workers_partition_checkpoints_without_duplicates(fixture, requested, expected):
    config, *_ = fixture
    config.update(device="cuda", eval_num_gpus=requested)
    config["models"][0]["predictor_checkpoint"] = [0, 500, 1000, 2000, 5000]
    settings = resolver.build_settings(config)
    assert settings["worker_count"] == expected
    assert len(settings["panel_groups"]) == expected
    assert [i for group in settings["panel_groups"] for i in group] == list(range(5))
    assert max(map(len, settings["panel_groups"])) - min(map(len, settings["panel_groups"])) <= 1
    assert f"export FT_PRED_WORKER_COUNT={expected}" in resolver.shell_settings(settings)


def test_cpu_keeps_one_worker_even_with_multiple_requested_gpus(fixture):
    config, *_ = fixture
    config["eval_num_gpus"] = 5
    config["models"][0]["predictor_checkpoint"] = [500, 1000, 2000]
    settings = resolver.build_settings(config)
    assert settings["worker_count"] == 1
    assert settings["panel_groups"] == [[0, 1, 2]]


@pytest.mark.parametrize("value", [0, -1, 1.5, "5", True])
def test_rejects_invalid_gpu_limits(fixture, value):
    config, *_ = fixture
    config["eval_num_gpus"] = value
    with pytest.raises(ValueError, match="eval_num_gpus must be a positive integer"):
        resolver.build_settings(config)


def test_rejects_duplicate_normalized_checkpoint_folders(fixture):
    config, *_ = fixture
    config["models"][0]["predictor_checkpoint"] = [1000, "001000"]
    with pytest.raises(ValueError, match="unique output folders"):
        resolver.build_settings(config)


@pytest.fixture
def zero_shot_sources(tmp_path):
    _, info, _ = _dataset(tmp_path)
    template = _checkpoint(tmp_path, info)
    outputs = tmp_path / "outputs_filtered"
    pt = outputs / "skillVLA_PT/Predictor/pt_run/checkpoints/100000/pretrained_model"
    ft = outputs / "skillVLA_FT/Predictor/ft_run/checkpoints/000000/pretrained_model"
    for path, stats in ((pt, "PT stats"), (ft, "FT stats")):
        shutil.copytree(template, path)
        (path / "policy_preprocessor.json").write_text(json.dumps({"steps": [], "test_stats": stats}))
    policy = json.loads((ft / "config.json").read_text())
    policy["skill_predictor_checkpoint_path"] = (
        "/another/server/outputs_filtered/skillVLA_stage1/Predictor/"
        "pt_run/checkpoints/100000/pretrained_model"
    )
    (ft / "config.json").write_text(json.dumps(policy))
    for step in ("000500", "001000", "002000", "005000", "010000"):
        shutil.copytree(ft, ft.parent.parent / step / "pretrained_model")
    return tmp_path, outputs, pt, ft


def test_ft_zero_uses_original_pt_weights_and_preprocessor_and_rebases_server(zero_shot_sources):
    root, outputs, pt, ft = zero_shot_sources
    config = {
        "project_root": str(root),
        "dataset": {"source": "libero_10_full_full", "run": "test_run"},
        "models": [{"label": "ft", "predictor_dir": "ft_run", "predictor_checkpoint": [0, 500, 1000]}],
    }
    panels = resolver.build_settings(config)["models"]
    assert [p["checkpoint"] for p in panels] == ["000000", "000500", "001000"]
    assert panels[0]["path"] == str(pt)
    assert panels[0]["source_checkpoint"] == "100000"
    assert panels[0]["zero_shot"] is True
    assert panels[0]["ft_reference_path"] == str(ft)
    assert panels[0]["folder"].endswith("_000000")
    assert "PT zero-shot" in panels[0]["label"]
    assert (
        json.loads((Path(panels[0]["path"]) / "policy_preprocessor.json").read_text())["test_stats"]
        == "PT stats"
    )
    for panel in panels[1:]:
        assert not panel.get("zero_shot")
        assert (
            json.loads((Path(panel["path"]) / "policy_preprocessor.json").read_text())["test_stats"]
            == "FT stats"
        )


def test_pt_zero_is_not_redirected(zero_shot_sources):
    _, outputs, pt, _ = zero_shot_sources
    pt_zero = pt.parent.parent / "000000/pretrained_model"
    shutil.copytree(pt, pt_zero)
    path, step, provenance = resolver._resolve_eval_predictor_checkpoint(outputs, "pt_run", "000000")
    assert (path, step, provenance) == (pt_zero, "000000", {})


def test_pruned_ft_zero_can_use_later_checkpoint_metadata(zero_shot_sources):
    _, outputs, pt, ft = zero_shot_sources
    shutil.rmtree(ft.parent)
    path, step, provenance = resolver._resolve_eval_predictor_checkpoint(outputs, "ft_run", "000000")
    assert path == pt and step == "000000" and provenance["zero_shot"]


def test_resume_self_pointer_falls_back_to_original_saved_lineage(zero_shot_sources):
    _, outputs, pt, ft = zero_shot_sources
    policy = json.loads((ft / "config.json").read_text())
    policy["skill_predictor_checkpoint_path"] = str(ft.parent.parent / "001000/pretrained_model")
    (ft / "config.json").write_text(json.dumps(policy))
    assert resolver._resolve_eval_predictor_checkpoint(outputs, "ft_run", "000000")[0] == pt


@pytest.mark.parametrize("artifact", ["model.safetensors", "policy_preprocessor.json"])
def test_missing_pt_artifacts_never_fall_back_to_ft_zero(zero_shot_sources, artifact):
    _, outputs, pt, _ = zero_shot_sources
    (pt / artifact).unlink()
    with pytest.raises(FileNotFoundError, match="original PT checkpoint|Incomplete PT zero-shot"):
        resolver._resolve_eval_predictor_checkpoint(outputs, "ft_run", "000000")


def test_ft_zero_missing_source_metadata_fails_explicitly(zero_shot_sources):
    _, outputs, _, ft = zero_shot_sources
    for path in ft.parent.parent.glob("*/pretrained_model/config.json"):
        policy = json.loads(path.read_text())
        policy.pop("skill_predictor_checkpoint_path")
        path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="Cannot determine the original PT Predictor"):
        resolver._resolve_eval_predictor_checkpoint(outputs, "ft_run", "000000")


def test_runtime_loads_zero_shot_weights_and_processor_from_same_pt_source(zero_shot_sources, monkeypatch):
    from unittest.mock import MagicMock

    from lerobot.datasets import lerobot_dataset
    from lerobot.policies import factory
    from lerobot.policies.skill_expert import modeling_skill_expert, modeling_skill_predictor

    root, _, pt, _ = zero_shot_sources
    settings = {
        "device": "cpu",
        "cpu_threads": 1,
        "tokenizer_path": "/tokenizer",
        "run_dir": str(root / "dataset_filtered/skillvla_dataset/libero_10_full_full/test_run"),
    }
    predictor = MagicMock()
    predictor.to.return_value = predictor
    predictor.requires_grad_.return_value = predictor
    monkeypatch.setattr(offline, "load_predictor_config", lambda *args: SimpleNamespace(dtype="bfloat16"))
    monkeypatch.setattr(modeling_skill_predictor, "FrozenVLMSkillPredictor", lambda config: predictor)
    calls = []
    monkeypatch.setattr(
        modeling_skill_expert,
        "_load_complete_predictor_parameters",
        lambda model, path: calls.append(("weights", path)),
    )

    def processors(**kwargs):
        calls.append(("preprocessor", kwargs["pretrained_path"]))
        return SimpleNamespace(steps=[]), None

    monkeypatch.setattr(factory, "make_pre_post_processors", processors)
    monkeypatch.setattr(offline, "_preprocessor_overrides", lambda *args: {})
    monkeypatch.setattr(offline, "_prepare_relabel_preprocessor", lambda value: value)
    monkeypatch.setattr(lerobot_dataset, "LeRobotDataset", lambda **kwargs: None)
    offline.load_predictor(settings, {"path": str(pt), "zero_shot": True})
    assert calls == [("weights", str(pt)), ("preprocessor", str(pt))]


def test_grounding_mismatch_is_not_silently_accepted(fixture):
    config, _, _, checkpoint, _ = fixture
    policy = json.loads((checkpoint / "config.json").read_text())
    policy["proprio_grounding"] = "none"
    (checkpoint / "config.json").write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="grounding mismatch"):
        resolver.build_settings(config)


def test_staged_training_root_uses_stable_fsq_dataset(fixture):
    config, run, info, checkpoint, _ = fixture
    policy = json.loads((checkpoint / "config.json").read_text())
    del policy["proprio_grounding"]
    policy["fsq_path"] = "/another/server/dataset/skillvla_dataset/libero_10_full_full/test_run/FSQ.pt"
    (checkpoint / "config.json").write_text(json.dumps(policy))
    (checkpoint / "train_config.json").write_text('{"dataset":{"root":"/tmp/deleted/skillvla"}}')
    assert resolver.build_settings(config)["run_dir"] == str(run)


def test_cpu_config_does_not_validate_saved_cuda_device_first(fixture, monkeypatch):
    _, _, _, checkpoint, _ = fixture
    policy = json.loads((checkpoint / "config.json").read_text())
    policy["device"] = "cuda"
    policy.pop("proprio_grounding")  # auxiliary configs recover this from the dataset
    (checkpoint / "config.json").write_text(json.dumps(policy))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("LEROBOT_ALLOW_CPU_FALLBACK", raising=False)
    config = offline.load_predictor_config(checkpoint, "cpu", "/tokenizer")
    assert config.device == "cpu"
    assert config.dtype == "float32"
    assert config.tokenizer_path == "/tokenizer"


def test_holdout_uses_original_identity_and_endpoint_is_not_end_minus_one(fixture):
    config, *_ = fixture
    settings = resolver.build_settings(config)
    samples, _ = offline.load_samples(settings)
    assert {s["episode_index"] for s in samples} == {20}
    assert [s["frame_start"] for s in samples] == [0, 2]
    assert [s["target_frame"] for s in samples] == [2, 3]
    np.testing.assert_allclose([s["gt_xyz"][0] for s in samples], [0.6, 0.7])
    assert [s["previous_gt_skill"] for s in samples] == [-1, 1]
    assert [s["gt_skill"] for s in samples] == [1, 2]


def test_includes_training_demos_but_marks_and_summarizes_them(fixture):
    config, *_ = fixture
    config["dataset"]["exclude_sources"] = []
    config["dataset"]["training_sources"] = ["libero_10_full_1"]
    samples, _ = offline.load_samples(resolver.build_settings(config))
    assert len(samples) == 4
    assert [s["is_training_demo"] for s in samples] == [True, True, False, False]
    records = [
        offline.make_record(s, s["gt_skill"], s["gt_xyz"], s["gt_xyz"], [3, 3, 3], "task") for s in samples
    ]
    metrics = offline.summarize(records)
    assert metrics["by_demo_split"]["training_demo"]["samples"] == 2
    assert metrics["by_demo_split"]["held_out_demo"]["samples"] == 2


def test_default_yaml_keeps_all_demos_and_uses_separate_outputs():
    config = resolver.load_config(resolver.DEFAULT_CONFIG_PATH)
    assert not config["dataset"].get("exclude_sources")
    assert config["dataset"]["training_sources"] == ["libero_10_full_1"]
    assert config["dataset"]["episodes_per_task"] == 0
    assert config["visualization"]["max_samples"] == 0
    # This is an editable experiment selection, not a fixed six-checkpoint contract.
    steps = config["models"][0]["predictor_checkpoint"]
    assert isinstance(steps, list) and steps
    assert len(set(steps)) == len(steps)
    assert all(isinstance(step, str) and step.isdigit() and len(step) == 6 for step in steps)
    assert isinstance(config["eval_num_gpus"], int) and config["eval_num_gpus"] >= 1


def test_no_hidden_repetition_when_not_enough_heldout_episodes(fixture):
    config, *_ = fixture
    config["dataset"]["episodes_per_task"] = 2
    with pytest.raises(ValueError, match="has 1 held-out demos"):
        offline.load_samples(resolver.build_settings(config))


def test_selection_is_reproducible_and_task_aware():
    segments = [SimpleNamespace(episode_index=i, task_index=i // 10) for i in range(20)]
    mapping = {i: (f"scene{i // 10}", f"demo{i}") for i in range(20)}
    kwargs = {"task_ids": [0, 1], "episodes_per_task": 3, "seed": 42}
    first = offline.select_segments(segments, mapping, [], **kwargs)
    second = offline.select_segments(segments, mapping, [], **kwargs)
    assert [s.episode_index for s in first] == [s.episode_index for s in second]
    assert len(first) == 6


def test_one_vlm_pass_two_xyz_conditions_and_previous_skill():
    calls = []

    class Predictor:
        config = SimpleNamespace(skill_predictor_start_proprio=True)
        head = SimpleNamespace(decode=lambda _: torch.tensor([2]))

        def reader_memory(self, *args):
            assert not torch.is_grad_enabled()
            calls.append("vlm")
            return torch.zeros(1, 1, 4), torch.zeros(1, 1, dtype=torch.bool)

        def skill_reader_hidden(self, memory, ignore, state, previous):
            assert previous.tolist() == [-1]
            assert state.shape == (1, 8)
            return torch.zeros(1, 4)

        def predict_end_state_from_memory(self, memory, ignore, code, state):
            calls.append(code.item())
            return torch.ones(1, 3) * code[:, None]

    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    processed = {
        "observation.images.image": torch.zeros(1, 3, 8, 8),
        "observation.images.wrist_image": torch.zeros(1, 3, 8, 8),
        "observation.state": torch.zeros(1, 8),
        "skill_previous_code": torch.tensor([-1]),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 5, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 5, dtype=torch.bool),
    }
    codes, gt_xyz, pred_xyz = offline.infer_batch(Predictor(), processed, [1])
    assert calls == ["vlm", 1, 2]
    assert codes.tolist() == [2]
    np.testing.assert_array_equal(gt_xyz, np.ones((1, 3)))
    np.testing.assert_array_equal(pred_xyz, np.ones((1, 3)) * 2)


def test_relative_xyz_offline_restores_measured_origin_for_both_skill_conditions():
    from lerobot.configs.skill_predictor_modes import PREDICTOR_START_XYZ
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    class Predictor:
        config = SimpleNamespace(skill_predictor_start_proprio=False)
        head = SimpleNamespace(decode=lambda _: torch.tensor([2]))
        relative_xyz = True
        uses_wrist = False

        def reader_memory(self, images, *args):
            assert len(images) == 1
            return torch.zeros(1, 1, 4), torch.zeros(1, 1, dtype=torch.bool)

        def skill_reader_hidden(self, memory, ignore, state, previous):
            assert state is None and previous.tolist() == [-1]
            return torch.zeros(1, 4)

        def predict_end_state_from_memory(self, memory, ignore, code, state):
            return code.float()[:, None].expand(-1, 3) * .1

        def absolute_end_state(self, prediction, start_xyz):
            return prediction + start_xyz

    origin = torch.tensor([[.2, .3, .4]])
    batch = {
        "observation.images.image": torch.zeros(1, 3, 8, 8),
        PREDICTOR_START_XYZ: origin, "skill_previous_code": torch.tensor([-1]),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 2, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 2, dtype=torch.bool),
    }
    _, gt, pred = offline.infer_batch(Predictor(), batch, [1])
    np.testing.assert_allclose(gt, (origin + .1).numpy())
    np.testing.assert_allclose(pred, (origin + .2).numpy())


def test_metrics_are_physical_units_and_not_rollout_sr():
    sample = {"uid": "sample", "episode_index": 1, "task_id": 0, "gt_skill": 1, "gt_xyz": [0, 0, 0]}
    row = offline.make_record(sample, 2, [0, 0, 0.01], [0.03, 0.04, 0], [3, 3, 3], "task")
    summary = offline.summarize([row])
    assert summary["overall"]["skill_accuracy_pct"] == 0
    assert summary["overall"]["xyz_l2_mean_m"] == pytest.approx(0.05)
    assert summary["overall"]["xyz_given_gt_skill_l2_mean_m"] == pytest.approx(0.01)
    assert summary["confusion"] == [{"gt_skill": 1, "pred_skill": 2, "count": 1}]
    assert "NOT simulator success rate" in summary["interpretation"]


class _ReportParser(HTMLParser):
    def __init__(self, document):
        super().__init__()
        self.articles, self.buttons, self.summaries, self.skill_options = [], [], [], []
        self.script, self.in_script = "", False
        self.feed(document)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "article":
            self.articles.append(attrs)
        if "data-task-filter" in attrs:
            self.buttons.append(attrs)
        if "data-task-summary" in attrs:
            self.summaries.append(attrs)
        if tag == "option" and "data-skill-index" in attrs:
            self.skill_options.append(attrs)
        if tag == "script":
            self.in_script = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_script = False

    def handle_data(self, text):
        if self.in_script:
            self.script += text


def _report_records():
    records = []
    for task, episode, skill, correct, training in [
        (10, 1, 1, False, None),
        (2, 3, 2, True, False),
        (2, 3, 0, False, True),
        (2, 1, 0, True, False),
        (10, 1, 0, True, True),
    ]:
        sample = {
            "uid": f"t{task}e{episode}s{skill}",
            "episode_index": episode,
            "task_id": task,
            "skill_index": skill,
            "frame_start": skill * 10,
            "gt_skill": 1,
            "gt_xyz": [0, 0, 0],
            "scene_file": "scene.hdf5",
            "demo": f"demo_{episode}",
            "is_training_demo": training,
        }
        records.append(
            offline.make_record(sample, 1 if correct else 2, [0, 0, 0], [0.1, 0, 0], [3, 3, 3], "<task>")
        )
    return records


def test_html_task_tabs_default_to_first_task_and_sort_without_mutating_records():
    records = _report_records()
    before = json.dumps(records)
    document = offline.render_html_report(
        {"label": "<model>", "path": "/checkpoint"}, records, offline.summarize(records)
    )
    parsed = _ReportParser(document)
    assert [b["data-task-filter"] for b in parsed.buttons] == ["2", "10", "all"]
    assert [b["data-task-filter"] for b in parsed.buttons if b["aria-pressed"] == "true"] == ["2"]
    assert [a["data-task"] for a in parsed.articles] == ["2", "2", "2", "10", "10"]
    assert [a["data-task"] for a in parsed.articles if "hidden" not in a] == ["2"] * 3
    assert [p["data-task-summary"] for p in parsed.summaries if "hidden" not in p] == ["2"]
    assert [o["value"] for o in parsed.skill_options] == ["0", "1", "2"]
    assert [a["data-skill-index"] for a in parsed.articles] == ["0", "0", "2", "0", "1"]
    assert "Skill 0 = first skill, not FSQ code 0" in document
    assert (
        document.index("episode 1 · skill 0")
        < document.index("episode 3 · skill 0")
        < document.index("episode 3 · skill 2")
    )
    assert "&lt;task&gt;" in document and "&lt;model&gt;" in document
    assert json.dumps(records) == before


def test_html_task_tabs_and_existing_filters_work_together():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute the HTML's actual filter JavaScript")
    records = _report_records()
    parsed = _ReportParser(
        offline.render_html_report({"label": "model", "path": "/checkpoint"}, records, offline.summarize(records))
    )
    harness = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
function element(attrs) {
  return {dataset: Object.fromEntries(Object.entries(attrs).filter(([k])=>k.startsWith('data-')).map(([k,v])=>[k.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase()),v])),
    value: attrs.value, hidden: Object.hasOwn(attrs,'hidden'), attrs, setAttribute(k,v) { this.attrs[k]=v; }};
}
const articles=input.articles.map(element), buttons=input.buttons.map(element), summaries=input.summaries.map(element);
const skillOptions=input.skill_options.map(element);
const ids={split:{value:'all'},wrong:{checked:false},'skill-index':{value:'all'},'sample-count':{textContent:''}};
const document={getElementById:id=>ids[id],querySelectorAll:selector=>({'article[data-task]':articles,'[data-task-filter]':buttons,'[data-task-summary]':summaries,'#skill-index option[data-skill-index]':skillOptions})[selector]};
vm.runInNewContext(input.script,{document});
const visible=()=>articles.filter(a=>!a.hidden);
const click=task=>buttons.find(b=>b.dataset.taskFilter===task).onclick();
assert.equal(visible().length,3);
assert(visible().every(a=>a.dataset.task==='2'));
click('10'); assert.equal(visible().length,2);
assert.deepEqual(summaries.filter(p=>!p.hidden).map(p=>p.dataset.taskSummary),['10']);
ids.wrong.checked=true; ids.wrong.onchange(); assert.equal(visible().length,1);
assert.equal(visible()[0].dataset.split,'unknown');
ids.split.value='held_out'; ids.split.onchange(); assert.equal(visible().length,0);
click('2'); assert.equal(visible().length,0);
ids.wrong.checked=false; ids.wrong.onchange(); assert.equal(visible().length,2);
ids.split.value='training'; ids.split.onchange(); assert.equal(visible().length,1);
click('all'); assert.equal(visible().length,2);
ids.split.value='all'; ids.split.onchange(); assert.equal(visible().length,5);
assert.match(ids['sample-count'].textContent,/Showing 5 \/ 5/);
assert.deepEqual(buttons.filter(b=>b.attrs['aria-pressed']==='true').map(b=>b.dataset.taskFilter),['all']);
// Skill order is episode position, independent of the FSQ code (all GT codes are 1).
click('2'); ids['skill-index'].value='0'; ids['skill-index'].onchange();
assert.equal(visible().length,2);
assert(visible().every(a=>a.dataset.task==='2'&&a.dataset.skillIndex==='0'));
assert.match(ids['sample-count'].textContent,/Showing 2 \/ 2/);
ids.wrong.checked=true; ids.wrong.onchange(); assert.equal(visible().length,1);
ids.split.value='held_out'; ids.split.onchange(); assert.equal(visible().length,0);
ids.split.value='all'; ids.wrong.checked=false;
click('10'); assert.equal(ids['skill-index'].value,'0'); assert.equal(visible().length,1);
assert(visible().every(a=>a.dataset.skillIndex==='0'));
click('all'); assert.equal(visible().length,3);
click('2'); ids['skill-index'].value='2'; ids['skill-index'].onchange(); assert.equal(visible().length,1);
click('10'); assert.equal(ids['skill-index'].value,'all'); assert.equal(visible().length,2);
assert.equal(skillOptions.find(o=>o.value==='2').disabled,true);
assert.equal(skillOptions.find(o=>o.value==='2').hidden,true);
click('2'); assert.equal(skillOptions.find(o=>o.value==='2').disabled,false);
ids['skill-index'].value='all'; ids['skill-index'].onchange(); assert.equal(visible().length,3);
"""
    payload = {key: getattr(parsed, key) for key in ("articles", "buttons", "summaries", "skill_options", "script")}
    subprocess.run(
        [node, "-e", harness], input=json.dumps(payload), text=True, capture_output=True, check=True, timeout=15
    )


def test_refresh_reports_changes_only_html_without_loading_model_or_dataset(tmp_path, monkeypatch):
    records = _report_records()
    folder = tmp_path / "00_model_002000"
    panel = {"label": "model", "path": "/missing/checkpoint"}
    offline.write_report(folder, panel, records, offline.summarize(records))
    (folder / "manifest.json").write_text(json.dumps({"panel": panel, "signature": "saved"}))
    (folder / "completion.json").write_text('{"signature":"saved"}')
    (folder / "index.html").write_text("old HTML")
    before = {p.name: p.read_bytes() for p in folder.iterdir() if p.name != "index.html"}
    for name in ("load_predictor", "load_samples", "infer_batch"):
        monkeypatch.setattr(offline, name, lambda *a, **k: pytest.fail("HTML refresh must not evaluate"))
    incomplete = tmp_path / "01_pending"
    incomplete.mkdir()
    (incomplete / "manifest.json").write_text('{"panel":{}}')
    assert offline.refresh_reports(tmp_path) == 1
    assert 'data-task-filter="2"' in (folder / "index.html").read_text()
    assert {p.name: p.read_bytes() for p in folder.iterdir() if p.name != "index.html"} == before
    assert not (incomplete / "index.html").exists()


def test_refresh_cli_does_not_resolve_current_yaml_or_checkpoints(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ft_pred_eval.py", "--refresh-reports", str(tmp_path)])
    monkeypatch.setattr(offline, "build_settings", lambda *a: pytest.fail("Saved HTML needs no checkpoints"))
    offline.main()
    assert "Refreshed 0 checkpoint HTML reports" in capsys.readouterr().out


def test_projection_reports_behind_camera_without_crashing():
    point = offline.project_xyz([0, 0, -1], np.eye(4), 8, 8)
    assert not point["visible"]
    assert point["pixel_xy"] is None
    assert "behind" in point["error"]


def test_overlapping_gt_and_pred_xyz_keep_three_distinct_markers(tmp_path):
    from PIL import Image

    record = offline.make_record(
        {"gt_skill": 1, "gt_xyz": [0, 0, 1]}, 1, [0, 0, 1], [0, 0, 1], [3, 3, 3], "task"
    )
    record["projection"] = {
        name: {"pixel_xy": [24, 24], "visible": True} for name in ("gt", "pred", "gt_skill")
    }
    path = tmp_path / "overlap.png"
    offline.draw_overlay(path, torch.zeros(3, 64, 64), torch.zeros(3, 64, 64), record)
    with Image.open(path) as image:
        assert image.getpixel((24, 24)) == (255, 48, 48)  # red cross
        assert image.getpixel((18, 24)) == (0, 221, 85)  # green ring
        assert image.getpixel((14, 24)) == (255, 176, 0)  # orange square


def test_saved_grounded_camera_projection_needs_no_simulator(fixture):
    config, run, info, *_ = fixture
    transform = np.eye(4)
    transform[0, 3] = 0.5
    np.savez_compressed(
        run / "skill_focus_uv.npz",
        camera="agentview",
        image_height=8,
        image_width=8,
        camera_episode_id=[20],
        camera_grounded_transform=[transform],
    )
    projector = offline.CameraProjector(resolver.build_settings(config), info)
    np.testing.assert_array_equal(projector.matrix({"episode_index": 20}, 8, 8), transform)
    with pytest.raises(ValueError, match="resolution"):
        projector.matrix({"episode_index": 20}, 16, 16)


def test_legacy_projection_uses_aligned_original_eef_origin(fixture):
    import h5py

    from lerobot.policies.skillVLA.focus_projection import camera_transform_from_recorded_xml

    config, run, info, _, _ = fixture
    settings = resolver.build_settings(config)
    path = run / "skillvla/data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    actions = [[float(i % 4)] * 7 for i in range(8)]
    pq.write_table(table.append_column("action", pa.array(actions)), path)
    original = Path(settings["original_dataset_dir"])
    original.mkdir(parents=True)
    xml = '<mujoco><worldbody><camera name="agentview" pos="0 0 2" quat="1 0 0 0"/></worldbody></mujoco>'
    with h5py.File(original / "scene_b.hdf5", "w") as handle:
        demo = handle.create_group("data/demo_2")
        demo.attrs["model_file"] = xml
        demo.create_dataset("actions", data=[[-9.0] * 7, *actions[:4]])
        demo.create_dataset(
            "obs/ee_states",
            data=[[9.0, 9.0, 9.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )
    matrix = offline.CameraProjector(settings, info).matrix(
        {"episode_index": 20, "scene_file": "scene_b.hdf5", "demo": "demo_2"}, 8, 8
    )
    expected = camera_transform_from_recorded_xml(xml, camera_name="agentview", height=8, width=8)
    expected[:, 3] += expected[:, :3] @ [1.0, 2.0, 3.0]
    np.testing.assert_allclose(matrix, expected)


def test_end_to_end_html_csv_resume_and_changed_checkpoint_guard(fixture, monkeypatch):
    config, run, _, checkpoint, rows = fixture
    config["visualization"] = {"enabled": True}
    settings = resolver.build_settings(config)
    # Calibrations are sufficient: no original HDF5, XML parsing, LIBERO or EGL.
    np.savez_compressed(
        run / "skill_focus_uv.npz",
        camera="agentview",
        image_height=8,
        image_width=8,
        camera_episode_id=[20],
        camera_grounded_transform=[np.eye(4)],
    )

    class Dataset:
        def __getitem__(self, index):
            row = rows[index]
            return {
                **row,
                "task": '<task & "language">',
                "observation.state": torch.tensor(row["observation.state"]),
                "observation.images.image": torch.zeros(3, 8, 8),
                "observation.images.wrist_image": torch.zeros(3, 8, 8),
            }

    monkeypatch.setattr(offline, "load_predictor", lambda *args: (None, lambda x: x, Dataset()))
    monkeypatch.setattr(
        offline,
        "infer_batch",
        lambda predictor, processed, codes: (
            np.asarray(codes),
            np.zeros((len(codes), 3)),
            np.ones((len(codes), 3)),
        ),
    )
    offline.run(settings)
    output = Path(settings["output_dir"])
    folder = output / settings["models"][0]["folder"]
    assert (output / "index.html").is_file()
    assert len(list((folder / "images").glob("*.png"))) == 2
    report = (folder / "index.html").read_text()
    assert "GT skill: 1" in report and "GT skill: 2" in report
    assert "Wrong skill only" in report and "XYZ given GT skill" in report
    assert "Training demos" in report and "Held-out demos" in report
    assert "&lt;task &amp; &quot;language&quot;&gt;" in report
    assert '<task & "language">' not in report
    assert (folder / "predictions.csv").is_file()
    assert len((folder / "predictions.jsonl").read_text().splitlines()) == 2
    assert json.loads((folder / "metrics.json").read_text())["overall"]["skill_accuracy_pct"] == 100
    monkeypatch.setattr(
        offline, "load_predictor", lambda *args: pytest.fail("Should resume without model loading")
    )
    offline.run(settings)
    (checkpoint / "model.safetensors").write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="Refusing to mix"):
        offline.run(settings)


def test_sharded_evaluation_matches_serial_and_resumes_with_different_gpu_count(fixture, monkeypatch):
    config, _, _, _, rows = fixture
    config["models"][0]["predictor_checkpoint"] = [500, 1000, 2000, 5000, 10000]
    settings = resolver.build_settings(config)
    loaded = []

    class Dataset:
        def __getitem__(self, index):
            row = rows[index]
            return {
                **row,
                "task": "test task",
                "observation.state": torch.tensor(row["observation.state"]),
                "observation.images.image": torch.zeros(3, 8, 8),
                "observation.images.wrist_image": torch.zeros(3, 8, 8),
            }

    def load(_, panel):
        loaded.append(panel["checkpoint"])
        return None, lambda x: x, Dataset()

    monkeypatch.setattr(offline, "load_predictor", load)
    monkeypatch.setattr(
        offline,
        "infer_batch",
        lambda predictor, processed, codes: (
            np.asarray(codes),
            np.zeros((len(codes), 3)),
            np.ones((len(codes), 3)),
        ),
    )
    offline.run(settings)
    serial = Path(settings["output_dir"])
    parallel = serial.with_name("parallel_report")
    settings = {**settings, "output_dir": str(parallel)}
    loaded.clear()
    # Finish out of order, as independently scheduled GPUs do.
    for index in reversed(range(5)):
        offline.run(settings, worker_index=index, worker_count=5)
    assert sorted(loaded) == ["000500", "001000", "002000", "005000", "010000"]
    overview = (parallel / "index.html").read_text()
    assert "Completed checkpoints: 5/5" in overview
    for panel in settings["models"]:
        assert overview.count(f'href="{panel["folder"]}/index.html"') == 1
        for name in ("metrics.json", "predictions.jsonl", "predictions.csv"):
            assert (serial / panel["folder"] / name).read_bytes() == (
                parallel / panel["folder"] / name
            ).read_bytes()
    monkeypatch.setattr(offline, "load_predictor", lambda *args: pytest.fail("Resume must not load a model"))
    # GPU packing is deliberately not part of the cache signature.
    for index in range(2):
        offline.run(settings, worker_index=index, worker_count=2)
    assert "Completed checkpoints: 5/5" in (parallel / "index.html").read_text()


def test_concurrent_workers_do_not_lose_overview_links(fixture, monkeypatch):
    config, *_ = fixture
    config["models"][0]["predictor_checkpoint"] = [500, 1000, 2000, 5000, 10000]
    settings = resolver.build_settings(config)
    samples, _ = offline.load_samples(settings)

    def fake_panel(settings, panel, samples, info):
        folder = Path(settings["output_dir"]) / panel["folder"]
        folder.mkdir(parents=True, exist_ok=True)
        time.sleep(0.02 * (int(panel["checkpoint"]) % 3))
        for name in ("index.html", "predictions.csv", "predictions.jsonl"):
            (folder / name).write_text("complete")
        offline.atomic_json(
            folder / "metrics.json", {"overall": {"skill_accuracy_pct": 50, "xyz_l2_mean_m": 0.1}}
        )
        offline.atomic_json(
            folder / "completion.json", {"signature": offline.panel_signature(settings, panel, samples)}
        )

    monkeypatch.setattr(offline, "run_panel", fake_panel)
    context = multiprocessing.get_context("fork")
    processes = [
        context.Process(target=offline.run, args=(settings,), kwargs={"worker_index": i, "worker_count": 3})
        for i in range(3)
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=15)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    root = Path(settings["output_dir"])
    report = (root / "index.html").read_text()
    assert "Completed checkpoints: 5/5" in report
    assert json.loads((root / "selected_samples.json").read_text()) == samples
    for panel in settings["models"]:
        assert report.count(f'href="{panel["folder"]}/index.html"') == 1


@pytest.mark.parametrize("missing,stale", [("predictions.csv", False), (None, True)])
def test_overview_does_not_publish_partial_or_stale_panel(fixture, missing, stale):
    config, *_ = fixture
    settings = resolver.build_settings(config)
    samples, _ = offline.load_samples(settings)
    panel = settings["models"][0]
    root = Path(settings["output_dir"])
    folder = root / panel["folder"]
    folder.mkdir(parents=True)
    for name in ("index.html", "predictions.csv", "predictions.jsonl"):
        if name != missing:
            (folder / name).write_text("report")
    offline.atomic_json(folder / "metrics.json", {"overall": {}})
    signature = "stale" if stale else offline.panel_signature(settings, panel, samples)
    offline.atomic_json(folder / "completion.json", {"signature": signature})
    offline.update_overview(settings, samples)
    report = (root / "index.html").read_text()
    assert "Completed checkpoints: 0/1" in report
    assert 'href="' not in report


@pytest.mark.parametrize("index,count", [(0, 0), (-1, 1), (1, 1), (0, 2)])
def test_invalid_worker_indices_fail_before_loading_data(fixture, monkeypatch, index, count):
    config, *_ = fixture
    settings = resolver.build_settings(config)
    monkeypatch.setattr(offline, "load_samples", lambda *args: pytest.fail("Must validate workers first"))
    with pytest.raises(ValueError, match="Invalid Predictor worker"):
        offline.run(settings, worker_index=index, worker_count=count)


@pytest.mark.parametrize("script", ["submit_pred_eval.sh", "src/pred_eval.sbatch"])
def test_shell_syntax(script):
    subprocess.run(["bash", "-n", str(EVAL / script)], check=True)


@pytest.mark.parametrize(
    "device,workers,scheduler",
    [("cuda", 1, "slurm"), ("cuda", 5, "slurm"), ("cpu", 1, "slurm"), ("cuda", 5, "local")],
)
def test_launcher_submits_correct_resources_without_real_slurm(tmp_path, device, workers, scheduler):
    root = tmp_path / "repo"
    configs = root / "lerobot/examples/libero/configs"
    target = configs / "train_skillVLA/FT/eval"
    (target / "src").mkdir(parents=True)
    (configs / "src").mkdir()
    shutil.copy2(EVAL / "submit_pred_eval.sh", target)
    for name in ("submit_job.sh", "snapshot_config.sh"):
        shutil.copy2(CONFIGS / "src" / name, configs / "src" / name)
    if scheduler == "local":
        # Exercise the multi-job loop without creating any real local jobs.
        (configs / "src/submit_job.sh").write_text(
            "submit_job_scheduler() { echo local; }\n"
            'submit_job() { printf "worker=%s/%s\\n" "$FT_PRED_WORKER_INDEX" "$FT_PRED_WORKER_COUNT"; '
            'printf "%s\\n" "$@"; test -f "$FT_PRED_EVAL_CONFIG"; }\n'
        )
    (configs / "global_config.yaml").write_text("server: auto\n")
    (target / "ft_pred_eval_config.yaml").write_text("device: " + device + "\n")
    (target / "src/pred_eval.sbatch").write_text("#!/bin/bash\n")
    (root / ".venv/bin").mkdir(parents=True)
    (root / ".venv/bin/python").symlink_to(sys.executable)
    values = {
        "DEVICE": device,
        "PARTITION": "gpu_partition" if device == "cuda" else "dell_cpu",
        "QOS": "base_qos" if device == "cuda" else "cpu_qos",
        "CPUS": 8,
        "MEM": "64G",
        "TIME": "4:00:00",
        "NODELIST": "",
        "EXCLUDE": "node40",
        "WORKER_COUNT": workers,
        "OUTPUT_DIR": str(target / "outputs_predictor/test"),
    }
    emitter = "import sys\nif '--shell' in sys.argv:\n"
    emitter += "\n".join(f" print('export FT_PRED_{key}={value}')" for key, value in values.items())
    (target / "src/ft_pred_eval_config.py").write_text(emitter)
    binary = tmp_path / "bin"
    binary.mkdir()
    sbatch = binary / "sbatch"
    sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\ntest -f "$FT_PRED_EVAL_CONFIG"\n')
    sbatch.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FT_PRED_", "SBD_"))}
    # This fixture mocks sbatch, not the newer srun submission modes/server loader.
    env.update(PATH=f"{binary}:{env['PATH']}", SBD_SCHEDULER=scheduler, SBD_SLURM_SUBMIT_MODE="sbatch")
    dry_run = subprocess.run(
        ["bash", str(target / "submit_pred_eval.sh"), "--dry-run"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    assert not (target / ".config_snapshots").exists()
    assert "--gres" not in dry_run.stdout and "worker=" not in dry_run.stdout
    result = subprocess.run(
        ["bash", str(target / "submit_pred_eval.sh")], env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert ("--gres=gpu:1" in result.stdout) == (device == "cuda")
    assert ("--exclude=node40" in result.stdout) == (device == "cuda")
    assert f"--partition={values['PARTITION']}" in result.stdout
    assert f"--qos={values['QOS']}" in result.stdout
    array = scheduler == "slurm" and workers > 1
    assert (f"--array=0-{workers - 1}%{workers}" in result.stdout) == array
    assert ("--output=logs/%x_%A_%a.out" in result.stdout) == array
    expected_submissions = workers if scheduler == "local" else 1
    assert result.stdout.count("--gres=gpu:1") == (expected_submissions if device == "cuda" else 0)
    if scheduler == "local":
        assert [line for line in result.stdout.splitlines() if line.startswith("worker=")] == [
            f"worker={i}/{workers}" for i in range(workers)
        ]
    assert len(list((target / ".config_snapshots").iterdir())) == 1


@pytest.mark.parametrize("array_index", [None, "4"])
def test_batch_script_forwards_worker_identity_without_loading_a_model(tmp_path, array_index):
    root = tmp_path / "repo"
    (root / ".venv/bin").mkdir(parents=True)
    python = root / ".venv/bin/python"
    python.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    runtime = root / "lerobot/examples/libero/configs/src"
    runtime.mkdir(parents=True)
    (runtime / "runtime_env.sh").write_text("# No runtime caches needed by this stub.\n")
    env = {key: value for key, value in os.environ.items() if not key.startswith(("FT_PRED_", "SLURM_"))}
    env.update(
        FT_PRED_EVAL_PROJECT_ROOT=str(root),
        FT_PRED_DEVICE="cpu",
        FT_PRED_EVAL_SRC=str(tmp_path),
        FT_PRED_EVAL_CONFIG="test.yaml",
        FT_PRED_WORKER_COUNT="5",
        FT_PRED_WORKER_INDEX="2",
    )
    if array_index is not None:
        env["SLURM_ARRAY_TASK_ID"] = array_index
    result = subprocess.run(
        ["bash", str(EVAL / "src/pred_eval.sbatch")],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-4:] == ["--worker-index", array_index or "2", "--worker-count", "5"]


def test_report_runtime_does_not_import_simulator():
    code = (EVAL / "src/ft_pred_eval.py").read_text()
    assert "from libero" not in code and "import mujoco" not in code
    assert "make_policy(" not in code and "load_external_skill_predictor" not in code


def _uv_fixture(fixture):
    config, run, info, checkpoint, rows = fixture
    policy = json.loads((checkpoint / "config.json").read_text())
    policy.update(skill_predictor_end_state_mode="off", skill_predictor_focus_uv_enabled=True)
    (checkpoint / "config.json").write_text(json.dumps(policy))
    # Reverse row order: joining by position would assign wrong episodes/skills.
    targets = {
        "camera": "agentview", "image_width": 101, "image_height": 51,
        "episode_id": [20, 20, 10, 10], "skill_index": [1, 0, 1, 0],
        "frame_start": [2, 0, 2, 0], "frame_end": [4, 2, 4, 2], "task_id": [0] * 4,
        "focus_uv": [[1., 0.], [0., 0.], [1., 0.], [0., 0.]],
        "focus_valid": [False, True, False, True], "focus_clipped": [True, False, True, False],
    }
    np.savez_compressed(run / "skill_focus_uv.npz", **targets)
    return targets


def test_uv_only_config_and_keyed_targets_keep_clipped_rows(fixture):
    _uv_fixture(fixture)
    settings = resolver.build_settings(fixture[0])
    assert settings["models"][0]["target_type"] == "uv"
    samples, _ = offline.load_samples(settings)
    assert len(samples) == 2
    assert [s["gt_uv"] for s in samples] == [[0., 0.], [1., 0.]]
    assert [s["uv_target_clipped"] for s in samples] == [False, True]
    assert samples[1]["uv_target_valid"] is False
    assert samples[0]["uv_image_width"] == 101 and samples[0]["uv_image_height"] == 51


def test_uv_requires_target_archive_and_at_least_one_head(fixture):
    _uv_fixture(fixture)
    config, run, _, checkpoint, _ = fixture
    (run / "skill_focus_uv.npz").unlink()
    with pytest.raises(FileNotFoundError, match="skill_focus_uv"):
        resolver.build_settings(config)
    policy = json.loads((checkpoint / "config.json").read_text())
    policy["skill_predictor_focus_uv_enabled"] = False
    (checkpoint / "config.json").write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="no XYZ/end-state or UV head"):
        resolver.build_settings(config)


@pytest.mark.parametrize("bad", ["duplicate", "missing", "interval", "task", "nan", "range", "camera", "resolution"])
def test_uv_target_archive_rejects_bad_alignment_or_coordinates(fixture, bad):
    targets = _uv_fixture(fixture)
    config, run, *_ = fixture
    if bad == "duplicate":
        targets["skill_index"][0] = 0
    elif bad == "missing":
        targets["episode_id"][0] = 99
    elif bad == "interval":
        targets["frame_end"][0] = 3
    elif bad == "task":
        targets["task_id"][0] = 9
    elif bad == "nan":
        targets["focus_uv"][0][0] = float("nan")
    elif bad == "range":
        targets["focus_uv"][0][0] = 1.5
    elif bad == "camera":
        targets["camera"] = "wristview"
    else:
        targets["image_width"] = 1
    np.savez_compressed(run / "skill_focus_uv.npz", **targets)
    with pytest.raises(ValueError):
        offline.load_samples(resolver.build_settings(config))


@pytest.mark.parametrize("crop_enabled", [True, False])
@pytest.mark.parametrize("gt_available", [True, False])
def test_uv_shared_vlm_and_crop_inverse_both_skill_conditions(crop_enabled, gt_available):
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    calls, crop = [], object()

    class Predictor:
        config = SimpleNamespace(skill_predictor_start_proprio=False,
                                 skill_predictor_end_state_mode="off", skill_predictor_focus_uv_enabled=True)
        uses_top_crop = crop_enabled
        head = SimpleNamespace(decode=lambda _: torch.tensor([2]))

        def top_crop(self, top, training):
            assert training is False
            calls.append("crop")
            return crop

        def reader_memory(self, images, tokens, mask, **kwargs):
            assert not torch.is_grad_enabled()
            assert kwargs == ({"top_crop": crop} if crop_enabled else {})
            calls.append("vlm")
            return torch.zeros(1, 2, 4), torch.zeros(1, 2, dtype=torch.bool)

        def skill_reader_hidden(self, memory, ignore, state, previous):
            assert state is None and previous.tolist() == [7]
            return torch.zeros(1, 4)

        def predict_focus_uv_from_memory(self, memory, ignore, code):
            calls.append(code.item())
            return code.float()[:, None].expand(-1, 2)

        def focus_uv_in_original_frame(self, uv, top, *, top_crop):
            assert top_crop is crop
            calls.append("inverse")
            return uv * .5

        def predict_end_state_from_memory(self, *args):
            pytest.fail("UV-only checkpoint has no XYZ head")

    batch = {
        "observation.images.image": torch.zeros(1, 3, 8, 8),
        "observation.images.wrist_image": torch.zeros(1, 3, 8, 8),
        "skill_previous_code": torch.tensor([7]),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 2, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 2, dtype=torch.bool),
    }
    codes, gt, pred = offline.infer_batch(Predictor(), batch, [1] if gt_available else None)
    assert codes.tolist() == [2]
    assert calls == ((["crop"] if crop_enabled else []) + ["vlm"] + ([1] if gt_available else []) + [2]
                     + (["inverse"] * (2 if gt_available else 1) if crop_enabled else []))
    if gt_available:
        np.testing.assert_allclose(gt, np.full((1, 2), .5 if crop_enabled else 1.))
    else:
        assert gt is None
    np.testing.assert_allclose(pred, np.full((1, 2), 1. if crop_enabled else 2.))


def test_uv_metrics_non_square_pixels_and_unclipped_predictions():
    sample = {"episode_index": 1, "task_id": 0, "gt_skill": 1, "gt_uv": [0., 0.],
              "uv_image_width": 101, "uv_image_height": 51,
              "uv_target_clipped": True, "uv_target_valid": False}
    record = offline.make_uv_record(sample, 2, [.1, .2], [2., 2.], [3, 3, 3], "task")
    assert record["uv_error"] == pytest.approx(np.sqrt(8))
    assert record["uv_error_px"] == pytest.approx(np.sqrt(100 ** 2 + 50 ** 2))
    assert record["uv_gt_skill_error_px"] == pytest.approx(np.sqrt(50))
    assert record["pred_uv"] == [2., 2.]
    summary = offline.summarize([record])
    assert summary["target_type"] == "uv"
    assert summary["overall"]["uv_clipped_targets"] == 1
    assert summary["overall"]["uv_valid_targets"] == 0
    assert "xyz_l2_mean_m" not in summary["overall"]
    assert summary["per_task"]["0"]["uv_l2_mean_px"] == record["uv_error_px"]
    point = offline.project_uv([2., -2.], 51, 101)
    assert point["raw_xy"] == [150., -25.]
    assert point["pixel_xy"] == [100., 0.] and not point["visible"]


@pytest.mark.parametrize("source", ["gt", "predicted"])
def test_uv_end_to_end_history_overlay_reports_resume_and_refresh(fixture, monkeypatch, source):
    _uv_fixture(fixture)
    config, _, _, _, rows = fixture
    config.update(preskill_source=source, batch_size=8, visualization={"enabled": True})
    settings = resolver.build_settings(config)
    seen = []

    class Dataset:
        def __getitem__(self, index):
            row = rows[index]
            return {**row, "task": "task", "observation.state": torch.tensor(row["observation.state"]),
                    "observation.images.image": torch.zeros(3, 51, 101),
                    "observation.images.wrist_image": torch.zeros(3, 51, 101)}

    def infer(predictor, processed, codes):
        previous = processed["skill_previous_code"].tolist()
        seen.extend(previous)
        return np.array([(v + 3) % 27 for v in previous]), np.zeros((len(codes), 2)), np.ones((len(codes), 2)) * 2

    monkeypatch.setattr(offline, "load_predictor", lambda *args: (None, lambda x: x, Dataset()))
    monkeypatch.setattr(offline, "infer_batch", infer)
    monkeypatch.setattr(offline, "CameraProjector", lambda *args: pytest.fail("UV needs no calibration"))
    offline.run(settings)
    assert seen == ([-1, 1] if source == "gt" else [-1, 2])
    root = Path(settings["output_dir"])
    folder = root / settings["models"][0]["folder"]
    records = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
    assert all(r["target_type"] == "uv" and r["pred_uv"] == [2., 2.] for r in records)
    assert all("pred_xyz" not in r and not r["projection"]["pred"]["visible"] for r in records)
    assert len(list((folder / "images").glob("*.png"))) == 2
    assert "uv_error_px" in (folder / "predictions.csv").read_text()
    report = (folder / "index.html").read_text()
    assert "GT UV" in report and "UV given GT skill" in report and "UV L2 (px)" in report
    assert "XYZ" not in report and " cm" not in report
    assert "data-task-filter" in report and "Wrong skill only" in report and "Skill order" in report
    assert "UV L2" in (root / "index.html").read_text()
    monkeypatch.setattr(offline, "load_predictor", lambda *args: pytest.fail("Complete cache needs no model"))
    offline.run(settings)
    assert offline.refresh_reports(root) == 1
    # Missing middle/history predecessor is recomputed; cached successor is checked.
    first = folder / "records" / f"{records[0]['uid']}.json"
    first.unlink()
    monkeypatch.setattr(offline, "load_predictor", lambda *args: (None, lambda x: x, Dataset()))
    seen.clear()
    offline.run(settings)
    assert seen == [-1]


def _initial_capture(tmp_path):
    import ft_pred_initial_frames as preview

    root = tmp_path / "captured"
    root.mkdir()
    rows = []
    for episode in range(2):
        name = f"ep{episode}.npz"
        np.savez(root / name, contract=preview.CONTRACT,
                 top=np.full((8, 8, 3), 51, dtype=np.uint8),
                 wrist=np.full((8, 8, 3), 102, dtype=np.uint8),
                 state_world=[.1, .2, .3, 1., 2., 3., .04, -.04])
        rows.append({"task_id": 0, "episode": episode, "model_input": name,
                     "task_description": "put the can in the basket"})
    (root / "manifest.json").write_text(json.dumps({"complete": True,
        "settings": {"random_range": 2., "model_input_contract": preview.CONTRACT}, "frames": rows}))
    return root


def test_initial_frame_inputs_are_real_and_episode_grounded(tmp_path):
    import ft_pred_initial_frames as preview

    root = _initial_capture(tmp_path)
    source = preview.capture_settings(tmp_path, {"capture_dir": str(root), "task_ids": [0], "episodes_per_task": 2})
    samples = preview.load_initial_samples({"initial_frames": source})
    raw = preview.read_input(samples[0], "episode_start_xyz")
    torch.testing.assert_close(raw["observation.state"][:3], torch.zeros(3))
    torch.testing.assert_close(raw["observation.state"][3:], torch.tensor([1., 2., 3., .04, -.04]))
    assert raw["observation.images.image"].mean().item() == pytest.approx(.2)
    assert raw["observation.images.wrist_image"].mean().item() == pytest.approx(.4)
    assert not any(key.startswith("gt_") for key in samples[0])
    (root / "ep0.npz").unlink()
    with pytest.raises(ValueError, match="Missing/invalid model_input"):
        preview.load_initial_samples({"initial_frames": source})


def test_initial_frame_mode_has_no_gt_metrics_or_history_chain(fixture, tmp_path, monkeypatch):
    config, run, info, checkpoint, _ = fixture
    root = _initial_capture(tmp_path)
    policy = json.loads((checkpoint / "config.json").read_text())
    policy.update(skill_predictor_end_state_mode="off", skill_predictor_focus_uv_enabled=True)
    (checkpoint / "config.json").write_text(json.dumps(policy))
    # Only dataset metadata remains: none of these demo GT archives are needed.
    (run / "skill_latents.npz").unlink()
    (run.parent / "eval_init_states.npz").unlink()
    config.update(input_mode="initial_frames", initial_frames={"capture_dir": str(root), "task_ids": [0], "episodes_per_task": 2})
    config["visualization"] = {"enabled": True}
    settings = resolver.build_settings(config)
    assert settings["output_dir"].endswith("_initial_range2")
    class Preprocessor:
        def reset(self):
            pass
        def __call__(self, raw):
            return raw
    predictor = SimpleNamespace(config=SimpleNamespace(skill_fsq_levels=[3, 3, 3]), uses_wrist=True)
    monkeypatch.setattr(offline, "load_predictor", lambda *args: (predictor, Preprocessor(), None))
    monkeypatch.setattr(offline, "load_samples", lambda *args: pytest.fail("No demo samples in initial-frame mode"))
    seen = []
    def infer(model, processed, gt_codes=None):
        assert gt_codes is None
        seen.extend(processed["skill_previous_code"].tolist())
        return np.array([2]), None, np.array([[.1, .2]])
    monkeypatch.setattr(offline, "infer_batch", infer)
    offline.run(settings)
    assert seen == [-1, -1]
    folder = Path(settings["output_dir"]) / settings["models"][0]["folder"]
    overall = json.loads((folder / "metrics.json").read_text())["overall"]
    assert overall["gt_available"] is False and overall["skill_accuracy_pct"] is None and overall["uv_l2_mean_px"] is None
    records = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
    assert all(record["previous_skill_input"] == -1 and "gt_uv" not in record for record in records)
    assert (folder / records[0]["image"]).is_file()
    monkeypatch.setattr(offline, "load_predictor", lambda *args: pytest.fail("Complete preview resumes without loading a model"))
    offline.run(settings)
    assert offline.refresh_reports(Path(settings["output_dir"])) == 1
