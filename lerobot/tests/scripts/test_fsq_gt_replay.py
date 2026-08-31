from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "examples/libero/configs/train_skills/skill_eval/src"
sys.path.insert(0, str(SRC))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CONFIG = _load("fsq_gt_replay_config_test", "fsq_gt_replay_config.py")
REPORT = _load("fsq_gt_replay_report_test", "fsq_gt_replay_report.py")
CATEGORIZATION = _load(
    "fsq_gt_replay_categorization_test", "fsq_gt_replay_categorization.py"
)
RUNNER = _load("run_fsq_gt_replay_test", "run_fsq_gt_replay.py")


def test_output_name_accepts_nested_run_and_epoch() -> None:
    assert CONFIG._safe_relative_output("", default="run/epoch0500") == Path(
        "run/epoch0500"
    )


def test_checkpoint_list_accepts_multiple_noncontiguous_epochs() -> None:
    assert CONFIG._fsq_checkpoints(
        {"fsq_eval_checkpoint": [125, 175, 300]}
    ) == ["125", "175", "300"]


def test_run_names_accept_scalar_and_list() -> None:
    assert CONFIG._run_names({"fsq_eval_run_name": "solo"}) == ["solo"]
    assert CONFIG._run_names({"fsq_eval_run_name": ["a", "b"]}) == ["a", "b"]
    with pytest.raises(ValueError, match="duplicates"):
        CONFIG._run_names({"fsq_eval_run_name": ["a", "a"]})
    with pytest.raises(ValueError, match="at least one"):
        CONFIG._run_names({"fsq_eval_run_name": []})


def test_model_names_are_optional_and_pair_with_runs() -> None:
    runs = ["folder_a", "folder_b"]
    assert CONFIG._model_names({}, runs) == runs
    assert CONFIG._model_names({"fsq_eval_model_name": "Model with spaces"}, ["solo"]) == [
        "Model with spaces"
    ]
    assert CONFIG._model_names(
        {"fsq_eval_model_name": ["Model A", "모델 B"]}, runs
    ) == ["Model A", "모델 B"]
    with pytest.raises(ValueError, match="one non-empty display name"):
        CONFIG._model_names({"fsq_eval_model_name": ["only one"]}, runs)
    with pytest.raises(ValueError, match="duplicates"):
        CONFIG._model_names({"fsq_eval_model_name": ["same", "same"]}, runs)


def test_model_entries_pair_run_folder_and_html_name() -> None:
    config = {
        "fsq_eval_models": [
            {"run_name": "folder_a", "model_name": "Model A"},
            {"run_name": "folder_b", "model_name": "모델 B"},
        ]
    }
    assert CONFIG._model_entries(config) == config["fsq_eval_models"]
    assert CONFIG._model_entries(
        {
            "fsq_eval_run_name": ["folder_a", "folder_b"],
            "fsq_eval_model_name": ["Model A", "Model B"],
        }
    ) == [
        {"run_name": "folder_a", "model_name": "Model A"},
        {"run_name": "folder_b", "model_name": "Model B"},
    ]


@pytest.mark.parametrize(
    "entries, message",
    [
        ([], "non-empty list"),
        (["folder_a"], "must be a mapping"),
        ([{"run_name": "folder_a"}], "requires non-empty"),
        (
            [
                {"run_name": "folder_a", "model_name": "same"},
                {"run_name": "folder_b", "model_name": "same"},
            ],
            "duplicate model_name",
        ),
    ],
)
def test_model_entries_reject_invalid_pairs(entries: list, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CONFIG._model_entries({"fsq_eval_models": entries})


@pytest.mark.parametrize("value", ["../escape", "/absolute", "bad name"])
def test_output_name_rejects_unsafe_paths(value: str) -> None:
    with pytest.raises(ValueError, match="output_name"):
        CONFIG._safe_relative_output(value, default="unused")


@pytest.mark.parametrize("checkpoint", ["250", "last"])
def test_resolve_artifact_missing_ok_skips_untrained_checkpoint(
    tmp_path: Path, checkpoint: str
) -> None:
    (tmp_path / "outputs" / "FSQ" / "run").mkdir(parents=True)
    cfg = {"fsq_eval_run_name": "run"}
    assert (
        CONFIG._resolve_fsq_artifact(
            cfg,
            dataset_root=tmp_path / "dataset",
            outputs_root=tmp_path / "outputs",
            checkpoint=checkpoint,
            missing_ok=True,
        )
        is None
    )
    with pytest.raises(FileNotFoundError):
        CONFIG._resolve_fsq_artifact(
            cfg,
            dataset_root=tmp_path / "dataset",
            outputs_root=tmp_path / "outputs",
            checkpoint=checkpoint,
        )


def test_resolve_artifact_missing_ok_still_rejects_unknown_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run folder"):
        CONFIG._resolve_fsq_artifact(
            {"fsq_eval_run_name": "no_such_run"},
            dataset_root=tmp_path / "dataset",
            outputs_root=tmp_path / "outputs",
            checkpoint="100",
            missing_ok=True,
        )


def test_filter_task_ids_drops_unavailable_and_supports_all() -> None:
    assert RUNNER._filter_task_ids([3, 51, 5], [3, 4, 5]) == [3, 5]
    assert RUNNER._filter_task_ids(None, [3, 4, 5]) == [3, 4, 5]
    with pytest.raises(RuntimeError, match="No requested task"):
        RUNNER._filter_task_ids([51], [3, 4, 5])


class _Occurrence:
    uid = "occ"

    def __init__(self, frame_start: int, frame_end: int) -> None:
        self.frame_start = frame_start
        self.frame_end = frame_end


def test_fsq_levels_reads_dict_and_dataclass_cfg(tmp_path: Path) -> None:
    import torch
    from types import SimpleNamespace

    v3_path = tmp_path / "v3.pt"
    torch.save({"cfg": {"fsq_levels": [3, 3, 3]}}, v3_path)
    assert RUNNER._fsq_levels(v3_path) == [3, 3, 3]

    oneshot_path = tmp_path / "oneshot.pt"
    torch.save({"cfg": SimpleNamespace(fsq_levels=[7, 5])}, oneshot_path)
    assert RUNNER._fsq_levels(oneshot_path) == [7, 5]

    joint_bsq_path = tmp_path / "joint_bsq.pt"
    torch.save(
        {
            "cfg": {
                "quantizer": "bsq",
                "bsq_code_dim": 5,
                # Deliberately retain an FSQ-looking placeholder: BSQ metadata
                # must win so the HTML renders the 32-corner 5D hypercube.
                "fsq_levels": [3, 3, 3],
            }
        },
        joint_bsq_path,
    )
    assert RUNNER._fsq_levels(joint_bsq_path) == [2, 2, 2, 2, 2]


def _replay_run(tmp_path: Path, *, epochs: list[int]) -> dict:
    """A minimal on-disk FSQ run plus the config that build_settings resolves."""
    run_dir = tmp_path / "outputs" / "FSQ" / "run"
    run_dir.mkdir(parents=True)
    for epoch in epochs:
        (run_dir / f"FSQ_epoch{epoch:04d}.pt").write_bytes(b"x")
    (run_dir / "fsq_meta.json").write_text(
        json.dumps(
            {
                "fsq_dataset_root": "FSQ_dataset",
                "target_dataset": "ds",
                "fsq_inputs_name": "FSQ_inputs",
                "skillset_seg_name": "seg",
                "skillset_name": "skillset",
            }
        )
    )
    dataset_root = tmp_path / "dataset"
    (
        dataset_root / "FSQ_dataset" / "ds" / "FSQ_inputs" / "seg" / "skillset" / "skills"
    ).mkdir(parents=True)
    (dataset_root / "ds" / "videos").mkdir(parents=True)
    (dataset_root / "skillvla_dataset" / "ds").mkdir(parents=True)
    (dataset_root / "skillvla_dataset" / "ds" / "eval_init_states.npz").write_bytes(b"x")
    (tmp_path / "libero_original_dataset" / "libero_90").mkdir(parents=True)
    return {
        "project_root": str(tmp_path),
        "dataset_root": "dataset",
        "outputs_root": "outputs",
        "fsq_eval_run_name": "run",
        "fsq_eval_checkpoint": [50, 100, 150],
        "target_task": "libero_90",
        "task_ids": [3],
        "episodes_per_task": 2,
        "output_name": "out",
    }


def test_episode_source_dataset_drops_the_exact_map_requirements(tmp_path: Path) -> None:
    """Neither the rendered map nor the original HDF5s are consulted in this mode.

    A dataset whose scenes span several LIBERO suites has neither, and its own
    task table numbers every episode instead.
    """
    config = _replay_run(tmp_path, epochs=[50])
    config["episode_source"] = "dataset"
    config["target_task"] = "no_such_suite"
    settings = CONFIG.build_settings(config)
    assert settings["eval_init_states_path"] == ""
    assert settings["original_dataset_dir"] == ""
    assert settings["episode_source"] == "dataset"


def test_episode_source_exact_still_requires_the_map(tmp_path: Path) -> None:
    config = _replay_run(tmp_path, epochs=[50])
    (tmp_path / "dataset" / "skillvla_dataset" / "ds" / "eval_init_states.npz").unlink()
    with pytest.raises(FileNotFoundError, match="episode-exact map"):
        CONFIG.build_settings(config)


def test_episode_source_rejects_unknown_values(tmp_path: Path) -> None:
    config = _replay_run(tmp_path, epochs=[50])
    config["episode_source"] = "sim"
    with pytest.raises(ValueError, match="episode_source"):
        CONFIG.build_settings(config)


def test_missing_latents_lists_only_checkpoints_without_an_npz(tmp_path: Path) -> None:
    """Encoding latents is the pipeline's only GPU work, so the submitter needs
    to know exactly which checkpoints still require it."""
    config = _replay_run(tmp_path, epochs=[50, 100, 150])
    run_dir = tmp_path / "outputs" / "FSQ" / "run"
    (run_dir / "skill_latents_epoch0100.npz").write_bytes(b"x")

    settings = CONFIG.build_settings(config)

    assert settings["fsq_missing_latents"] == "50 150"
    assert settings["fsq_missing_latents_count"] == 2


def test_missing_latents_is_empty_once_every_checkpoint_is_encoded(tmp_path: Path) -> None:
    config = _replay_run(tmp_path, epochs=[50, 100, 150])
    run_dir = tmp_path / "outputs" / "FSQ" / "run"
    for tag in ("epoch0050", "epoch0100", "epoch0150"):
        (run_dir / f"skill_latents_{tag}.npz").write_bytes(b"x")

    settings = CONFIG.build_settings(config)

    assert settings["fsq_missing_latents"] == ""
    assert settings["fsq_missing_latents_count"] == 0


def _write_complete_replay_manifest(
    collection_dir: Path,
    run_dir: Path,
    *,
    epoch: int,
    request: dict,
) -> None:
    tag = f"epoch{epoch:04d}"
    output_dir = collection_dir / "checkpoints" / tag
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "start.png").write_bytes(b"start")
    (image_dir / "final.png").write_bytes(b"final")
    manifest = {
        "signature": {
            "format": "fsq_gt_replay_v3",
            "model_path": str((run_dir / f"FSQ_epoch{epoch:04d}.pt").resolve()),
            "latents_path": str((run_dir / f"skill_latents_{tag}.npz").resolve()),
            "target_task": request["target_task"],
            "selected_episodes": {"3": [10, 11]},
            "seed": request["seed"],
        },
        "request": request,
        "run_name": "run",
        "epoch_tag": tag,
        "completed": True,
        "records": {
            "occ": {
                "start_image_path": "images/start.png",
                "final_image_path": "images/final.png",
            }
        },
    }
    manifest_path = output_dir / "metrics" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest))


def test_incremental_submission_keeps_only_new_or_incomplete_checkpoints(
    monkeypatch, tmp_path: Path
) -> None:
    """Completed work submits no no-op job; a later checkpoint is still added."""
    monkeypatch.setattr(CONFIG, "_HERE", tmp_path / "skill_eval" / "src")
    config = _replay_run(tmp_path, epochs=[50, 100])
    config["resume"] = True
    run_dir = tmp_path / "outputs" / "FSQ" / "run"
    for tag in ("epoch0050", "epoch0100"):
        (run_dir / f"skill_latents_{tag}.npz").write_bytes(b"latent")

    initial = CONFIG.build_settings(config)
    assert initial["fsq_pending_checkpoints"] == "50 100"
    request = CONFIG._replay_request(
        episode_source="exact",
        target_task="libero_90",
        task_ids=[3],
        episode_ids=[],
        episodes_per_task=2,
        episode_selection="first",
        seed=42,
    )
    collection_dir = Path(initial["eval_collection_dir"])
    for epoch in (50, 100):
        _write_complete_replay_manifest(
            collection_dir, run_dir, epoch=epoch, request=request
        )

    complete = CONFIG.build_settings(config)
    assert complete["fsq_pending_checkpoints"] == ""
    assert complete["fsq_pending_checkpoint_count"] == 0
    assert complete["fsq_completed_checkpoints"] == "50 100"
    assert complete["fsq_missing_latents"] == ""
    assert complete["eval_max_concurrent"] == 0
    assert complete["eval_total_jobs"] == 0

    # Training later produces another requested checkpoint: only that one is
    # submitted, while the two completed checkpoints stay untouched.
    (run_dir / "FSQ_epoch0150.pt").write_bytes(b"checkpoint")
    (run_dir / "skill_latents_epoch0150.npz").write_bytes(b"latent")
    incremental = CONFIG.build_settings(config)
    assert incremental["fsq_pending_checkpoints"] == "150"
    assert incremental["fsq_pending_checkpoint_count"] == 1
    assert incremental["fsq_completed_checkpoints"] == "50 100"
    assert incremental["eval_checkpoints_per_job"] == 1
    assert incremental["eval_max_concurrent"] == 1
    assert incremental["eval_total_jobs"] == 1


def test_incremental_submission_replays_a_checkpoint_with_incomplete_records(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(CONFIG, "_HERE", tmp_path / "skill_eval" / "src")
    config = _replay_run(tmp_path, epochs=[50])
    config["resume"] = True
    run_dir = tmp_path / "outputs" / "FSQ" / "run"
    (run_dir / "skill_latents_epoch0050.npz").write_bytes(b"latent")
    settings = CONFIG.build_settings(config)
    request = CONFIG._replay_request(
        episode_source="exact",
        target_task="libero_90",
        task_ids=[3],
        episode_ids=[],
        episodes_per_task=2,
        episode_selection="first",
        seed=42,
    )
    _write_complete_replay_manifest(
        Path(settings["eval_collection_dir"]), run_dir, epoch=50, request=request
    )
    manifest_path = (
        Path(settings["eval_collection_dir"])
        / "checkpoints/epoch0050/metrics/manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["records"]["occ"].pop("final_image_path")
    manifest_path.write_text(json.dumps(manifest))

    incomplete = CONFIG.build_settings(config)
    assert incomplete["fsq_pending_checkpoints"] == "50"
    assert incomplete["fsq_completed_checkpoints"] == ""


def test_blank_replay_qos_inherits_train_gpu_reservation(tmp_path: Path) -> None:
    """Inherited GPU QOS gets a GPU reservation although replay remains CPU-only."""
    config = _replay_run(tmp_path, epochs=[50])
    settings = CONFIG.build_settings(config)
    assert settings["eval_replay_gres"] == "gpu:1"
    assert settings["eval_gres"] == "gpu:1"

    config["slurm"] = {
        "replay_partition": "dell_cpu",
        "replay_qos": "cpu_qos",
        "replay_gres": "",
    }
    settings = CONFIG.build_settings(config)
    assert settings["eval_replay_gres"] == ""
    assert settings["eval_replay_partition"] == "dell_cpu"
    assert settings["eval_replay_qos"] == "cpu_qos"


def test_blank_replay_slurm_values_inherit_global_train_defaults(tmp_path: Path) -> None:
    config = _replay_run(tmp_path, epochs=[50])
    config.update(
        train_partition=["gpu_a", "gpu_b"],
        train_qos="global_train_qos",
        slurm={"replay_partition": "", "replay_qos": ""},
    )

    settings = CONFIG.build_settings(config)

    assert settings["eval_replay_partition"] == "gpu_a,gpu_b"
    assert settings["eval_replay_qos"] == "global_train_qos"
    assert settings["eval_replay_gres"] == "gpu:1"


def test_nonempty_module_replay_slurm_values_override_global_defaults(
    tmp_path: Path,
) -> None:
    config = _replay_run(tmp_path, epochs=[50])
    config.update(
        train_partition="global_gpu",
        train_qos="global_train_qos",
        slurm={"replay_partition": "local_cpu", "replay_qos": "local_cpu_qos"},
    )

    settings = CONFIG.build_settings(config)

    assert settings["eval_replay_partition"] == "local_cpu"
    assert settings["eval_replay_qos"] == "local_cpu_qos"
    assert settings["eval_replay_gres"] == ""


def test_max_concurrent_still_accepts_the_old_gpu_count_name(tmp_path: Path) -> None:
    """eval_num_gpus was always the array throttle, never a GPU count."""
    config = _replay_run(tmp_path, epochs=[50, 100, 150])
    config["checkpoints_per_job"] = 1
    config["eval_num_gpus"] = 2

    assert CONFIG.build_settings(config)["eval_max_concurrent"] == 2


def test_max_concurrent_wins_over_the_old_name(tmp_path: Path) -> None:
    config = _replay_run(tmp_path, epochs=[50, 100, 150])
    config["checkpoints_per_job"] = 1
    config["eval_num_gpus"] = 2
    config["eval_max_concurrent"] = 3

    assert CONFIG.build_settings(config)["eval_max_concurrent"] == 3


def test_total_jobs_uses_pending_chunks_and_workers(tmp_path: Path) -> None:
    config = _replay_run(tmp_path, epochs=[50, 100, 150])
    config["checkpoints_per_job"] = 2
    config["workers_per_checkpoint"] = 3
    config["episodes_per_task"] = 4

    settings = CONFIG.build_settings(config)

    assert settings["fsq_checkpoint_count"] == 3
    assert settings["fsq_pending_checkpoint_count"] == 3
    assert settings["eval_total_jobs"] == 6


def test_max_concurrent_is_capped_by_the_number_of_tasks(tmp_path: Path) -> None:
    """Asking for more slots than there are tasks must not inflate the array."""
    config = _replay_run(tmp_path, epochs=[50])
    config["eval_max_concurrent"] = 50

    assert CONFIG.build_settings(config)["eval_max_concurrent"] == 1


def test_checkpoints_per_job_all_packs_one_replay_task_per_run(tmp_path: Path) -> None:
    config = _replay_run(tmp_path, epochs=[50, 100, 150])
    config["checkpoints_per_job"] = "all"
    config["eval_max_concurrent"] = 20

    settings = CONFIG.build_settings(config)

    assert settings["eval_checkpoints_per_job"] == 3
    assert settings["eval_max_concurrent"] == 1
    assert settings["eval_total_jobs"] == 1


def test_expected_epoch_tags_track_disk_without_a_frozen_list(tmp_path: Path) -> None:
    config = _replay_run(tmp_path, epochs=[50, 100])
    settings = CONFIG.build_settings(config)
    assert json.loads(settings["fsq_expected_epoch_tags"]) == ["epoch0050", "epoch0100"]
    assert settings["fsq_skipped_checkpoints"] == "150"


def test_frozen_checkpoint_list_ignores_checkpoints_trained_after_submission(
    tmp_path: Path,
) -> None:
    """A job must expect exactly the checkpoints its array was sized for.

    Training that keeps writing checkpoints while the array runs would otherwise
    enlarge the expected set, and the collection report would wait forever for
    jobs that were never submitted.
    """
    config = _replay_run(tmp_path, epochs=[50, 100])
    frozen = json.loads(CONFIG.build_settings(config)["fsq_expected_epoch_tags"])
    (tmp_path / "outputs" / "FSQ" / "run" / "FSQ_epoch0150.pt").write_bytes(b"x")

    settings = CONFIG.build_settings(
        config, checkpoint_override="100", checkpoint_list_override=["50", "100"]
    )
    assert json.loads(settings["fsq_expected_epoch_tags"]) == frozen
    assert settings["fsq_epoch_tag"] == "epoch0100"


def test_frozen_checkpoint_list_rejects_a_vanished_checkpoint(tmp_path: Path) -> None:
    config = _replay_run(tmp_path, epochs=[50])
    with pytest.raises(FileNotFoundError):
        CONFIG.build_settings(
            config, checkpoint_override="50", checkpoint_list_override=["50", "100"]
        )


def _meta_only_dataset(tmp_path: Path, *, tasks: list[str], episode_tasks: list[str]):
    """A skill dataset carrying only the metadata episode sourcing reads."""
    import pandas as pd

    dataset_dir = tmp_path / "ds"
    (dataset_dir / "meta" / "episodes").mkdir(parents=True)
    pd.DataFrame({"task_index": range(len(tasks))}, index=pd.Index(tasks, name="task")).to_parquet(
        dataset_dir / "meta" / "tasks.parquet"
    )
    pd.DataFrame(
        {
            "episode_index": range(len(episode_tasks)),
            "data/chunk_index": [0] * len(episode_tasks),
            "data/file_index": [0] * len(episode_tasks),
            "length": [10] * len(episode_tasks),
            "tasks": [[task] for task in episode_tasks],
        }
    ).to_parquet(dataset_dir / "meta" / "episodes" / "chunk.parquet")
    latents = tmp_path / "latents.npz"
    count = len(episode_tasks)
    np.savez(
        latents,
        tokens=np.zeros(count, dtype=np.int64),
        episode_id=np.arange(count, dtype=np.int64),
        skill_index=np.zeros(count, dtype=np.int64),
        frame_start=np.zeros(count, dtype=np.int64),
        frame_end=np.full(count, 10, dtype=np.int64),
    )
    return dataset_dir, latents


def test_dataset_meta_sourcing_numbers_every_episode(tmp_path: Path) -> None:
    """The dataset's own table covers scenes no single LIBERO suite contains."""
    import skill_data

    dataset_dir, latents = _meta_only_dataset(
        tmp_path, tasks=["pick up a", "pick up b"], episode_tasks=["pick up b", "pick up a"]
    )
    dataset = skill_data.SkillEvaluationDataset(
        skill_dataset_dir=dataset_dir,
        skill_latents_path=latents,
        eval_init_states_path=None,
        original_dataset_dir=None,
        suite_name="not_a_libero_suite",
    )
    assert {episode: source.task_id for episode, source in dataset.sources.items()} == {0: 1, 1: 0}
    assert dataset.task_descriptions == {0: "pick up a", 1: "pick up b"}
    assert RUNNER._available_task_ids(dataset, episodes_per_task=1) == [0, 1]


def test_dataset_meta_sourcing_refuses_state_alignment(tmp_path: Path) -> None:
    import skill_data

    dataset_dir, latents = _meta_only_dataset(
        tmp_path, tasks=["pick up a"], episode_tasks=["pick up a"]
    )
    dataset = skill_data.SkillEvaluationDataset(
        skill_dataset_dir=dataset_dir,
        skill_latents_path=latents,
        eval_init_states_path=None,
        original_dataset_dir=None,
        suite_name="not_a_libero_suite",
    )
    with pytest.raises(RuntimeError, match="episode-exact map"):
        dataset.load_aligned_episode(0)


def test_dataset_meta_sourcing_rejects_a_task_outside_the_table(tmp_path: Path) -> None:
    import skill_data

    dataset_dir, latents = _meta_only_dataset(
        tmp_path, tasks=["pick up a"], episode_tasks=["pick up z"]
    )
    with pytest.raises(ValueError, match="absent from"):
        skill_data.SkillEvaluationDataset(
            skill_dataset_dir=dataset_dir,
            skill_latents_path=latents,
            eval_init_states_path=None,
            original_dataset_dir=None,
            suite_name="not_a_libero_suite",
        )


def test_frame_pair_uses_next_frame_start() -> None:
    assert RUNNER._frame_pair(_Occurrence(0, 2), episode_length=5) == (0, 2)


def test_frame_pair_clamps_final_episode_skill() -> None:
    assert RUNNER._frame_pair(_Occurrence(3, 5), episode_length=5) == (3, 4)


def test_frame_pair_rejects_empty_segment() -> None:
    with pytest.raises(RuntimeError, match="no GT frame"):
        RUNNER._frame_pair(_Occurrence(2, 2), episode_length=5)


def test_episode_frame_reader_decodes_each_frame_only_once(monkeypatch, tmp_path: Path) -> None:
    import pandas as pd
    import torch

    reader = RUNNER._EpisodeFrameReader.__new__(RUNNER._EpisodeFrameReader)
    reader.dataset_dir = tmp_path
    reader.video_key = RUNNER.VIDEO_KEY
    reader.fps = 10.0
    reader.path_template = "videos/{video_key}/{chunk_index}/{file_index}.mp4"
    reader._frame_cache = {}
    reader.index = pd.DataFrame(
        {
            "episode_index": [7],
            "length": [10],
            f"videos/{RUNNER.VIDEO_KEY}/chunk_index": [0],
            f"videos/{RUNNER.VIDEO_KEY}/file_index": [0],
            f"videos/{RUNNER.VIDEO_KEY}/from_timestamp": [0.0],
        }
    ).set_index("episode_index", drop=False)
    calls: list[list[float]] = []

    def fake_decode(_path, timestamps, *, tolerance_s):
        assert tolerance_s == pytest.approx(0.05)
        calls.append(list(timestamps))
        return torch.stack(
            [torch.full((3, 2, 2), timestamp) for timestamp in timestamps]
        )

    monkeypatch.setattr(RUNNER, "decode_video_frames", fake_decode)

    first = reader.frames(7, [2, 1, 2])
    second = reader.frames(7, [1, 2])
    third = reader.frames(7, [2, 3])

    assert calls == [[0.1, 0.2], [0.3]]
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[2], third[2])
    assert reader.cached_frame_count == 3
    assert reader.cached_bytes == 3 * 2 * 2 * 3


def test_report_groups_occurrences_by_fsq_token() -> None:
    manifest = {
        "levels": [3, 3, 3],
        "run_name": "fsq333",
        "epoch_tag": "epoch0500",
        "train_codebook_counts": [0, 0, 0, 0, 2] + [0] * 22,
        "train_codebook_used": 18,
        "signature": {
            "target_task": "libero_90",
            "selected_episodes": {"0": [2]},
        },
        "records": {
            "b": {
                "token": 4,
                "task_id": 0,
                "episode_id": 2,
                "frame_start": 20,
            },
            "a": {
                "token": 4,
                "task_id": 0,
                "episode_id": 2,
                "frame_start": 0,
            },
        },
    }

    payload = REPORT.report_payload(manifest)

    assert payload["occurrence_count"] == 2
    assert payload["train_codebook_counts"][4] == 2
    assert payload["train_codebook_used"] == 18
    assert payload["skills"][0]["token"] == 4
    assert payload["skills"][0]["coord"] == [1, 1, 0]
    assert [row["frame_start"] for row in payload["skills"][0]["occurrences"]] == [
        0,
        20,
    ]


def test_write_image_preserves_rendered_frame(tmp_path: Path) -> None:
    frame = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    path = tmp_path / "frame.png"

    RUNNER._write_image(path, frame)

    np.testing.assert_array_equal(np.asarray(Image.open(path)), frame)


def test_report_shows_start_and_final_image_pair(tmp_path: Path) -> None:
    payload = {
        "levels": [3, 3, 3],
        "run_name": "fsq333",
        "model_name": "Readable model",
        "title": "term comparison <2>",
        "epoch_tag": "epoch0500",
        "target_task": "libero_90",
        "task_ids": [0],
        "episode_count": 1,
        "occurrence_count": 1,
        "train_codebook_counts": [1] + [0] * 26,
        "skills": [
            {
                "token": 0,
                "coord": [0, 0, 0],
                "occurrences": [
                    {
                        "task_id": 0,
                        "task_description": "test",
                        "episode_id": 0,
                        "skill_index": 0,
                        "frame_start": 0,
                        "frame_end": 2,
                        "length": 2,
                        "start_image_path": "images/test_start.png",
                        "final_image_path": "images/test_final.png",
                    }
                ],
            }
        ],
    }

    report = REPORT.write_html_report(tmp_path, payload)
    html = report.read_text(encoding="utf-8")
    data_paths = sorted(tmp_path.glob("report-data-*.js"))
    data = "".join(path.read_text(encoding="utf-8") for path in data_paths)

    assert data_paths
    assert all(
        path.stat().st_size <= REPORT._REPORT_DATA_CHUNK_BYTES for path in data_paths
    )
    assert all(f'<script src="{path.name}"></script>' in html for path in data_paths)
    assert "window.FSQ_GT_REPLAY_DATA=" in html
    assert "<title>term comparison &lt;2&gt;</title>" in html
    assert "<h1>term comparison &lt;2&gt;</h1>" in html
    assert "Readable model" in data
    assert "images/test_start.png" in data
    assert "images/test_final.png" in data
    assert "<video" not in html
    assert 'class="pair"' in html
    assert "GT start" in html
    assert "GT end" in html
    assert 'id="checkpoint"' in html
    assert 'id="tasks"' in html
    assert 'class="task-group"' in html
    assert 'class="occ-row"' in html
    assert 'id="positionMode"' in html
    assert "Cohesion" in html
    assert "cohesionTable()" in html
    assert "codebook used (train)" in html
    assert "mean effective codes per cell" in html
    assert "full-data codebook utilization" in html
    assert "independent of this report's task_ids selection" in html
    assert "effective (purity)" not in html
    assert 'id="modelTabs"' in html
    assert 'class="small-button cube-mode' in html
    assert 'data-mode="length"' in html
    assert 'data-mode="count"' in html
    assert 'class="full-cube"' in html
    assert 'id="fullCubeLegend"' in html
    assert "Full training skillset" in html
    assert "renderFullCube()" in html
    assert "full-data elements" in html
    assert "COUNT_BORDER_THRESHOLD=10" in html
    assert "stats.get(t).count>COUNT_BORDER_THRESHOLD" in html
    assert "selectedCode?'#d62728'" in html
    assert "&gt;${COUNT_BORDER_THRESHOLD} elements" in html
    assert "renderCubeLegend" in html
    assert "Math.max(0,...models.flatMap" not in html
    assert "maximumSkillId=Math.max(maximumSkillId" in html


def _write_collection_manifest(
    root: Path,
    run: str,
    epoch: int,
    episodes: list[int],
    *,
    model_name: str = "",
    title: str = "",
) -> None:
    tag = f"epoch{epoch:04d}"
    manifest = {
        "signature": {
            "format": "fsq_gt_replay_v2",
            "target_task": "libero_90",
            "selected_episodes": {"3": episodes},
            "seed": 42,
        },
        "run_name": run,
        "epoch_tag": tag,
        "levels": [3, 3, 3],
        "train_codebook_used": 9,
        "completed": True,
        "records": {
            f"{tag}-a": {
                "token": 1,
                "task_id": 3,
                "episode_id": episodes[0],
                "frame_start": 0,
                "start_image_path": "images/a_start.png",
                "final_image_path": "images/a_final.png",
            }
        },
    }
    if model_name:
        manifest["model_name"] = model_name
    if title:
        manifest["report_title"] = title
    path = root / "checkpoints" / tag / "metrics" / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest))


def test_compare_payload_prefixes_media_and_flags_mismatch(tmp_path: Path) -> None:
    _write_collection_manifest(
        tmp_path / "a", "run_a", 100, [33], model_name="Model A", title="Comparison"
    )
    _write_collection_manifest(
        tmp_path / "b", "run_b", 200, [44], model_name="Model B", title="Comparison"
    )

    payload = REPORT.compare_payload(
        [tmp_path / "a", tmp_path / "b"], output_dir=tmp_path / "compare"
    )

    assert payload["title"] == "Comparison"
    assert [model["name"] for model in payload["models"]] == ["Model A", "Model B"]
    occurrence = payload["models"][0]["checkpoints"][0]["skills"][0]["occurrences"][0]
    assert occurrence["start_image_path"] == (
        "../a/checkpoints/epoch0100/images/a_start.png"
    )
    assert "mismatched" not in payload["models"][0]
    assert payload["models"][1]["mismatched"] is True


def test_maybe_build_compare_waits_for_every_collection(tmp_path: Path) -> None:
    _write_collection_manifest(tmp_path / "a", "run_a", 100, [33])
    _write_collection_manifest(tmp_path / "b", "run_b", 100, [33])
    dirs = [tmp_path / "a", tmp_path / "b"]
    compare_dir = tmp_path / "compare"

    (tmp_path / "a" / "metrics").mkdir()
    (tmp_path / "a" / "metrics" / "collection.json").write_text("{}")
    assert REPORT.maybe_build_compare(dirs, output_dir=compare_dir) is None

    (tmp_path / "b" / "metrics").mkdir()
    (tmp_path / "b" / "metrics" / "collection.json").write_text("{}")
    path = REPORT.maybe_build_compare(dirs, output_dir=compare_dir)
    assert path == compare_dir / "index.html"
    assert path.is_file()
    linked_path = compare_dir / "linked_codebooks.html"
    assert linked_path.is_file()
    payload = json.loads((compare_dir / "metrics" / "compare.json").read_text())
    assert [model["name"] for model in payload["models"]] == ["a", "b"]
    standard_html = path.read_text(encoding="utf-8")
    linked_html = linked_path.read_text(encoding="utf-8")
    assert 'href="linked_codebooks.html"' in standard_html
    assert 'id="checkpoint"' in linked_html
    assert 'id="codebooks"' in linked_html
    assert 'id="colorPanel"' in linked_html
    assert "function selectCode(modelIndex,token)" in linked_html
    assert "function sampleColorMap(keys)" in linked_html
    assert "function renderColorPanel(keys,colorMap)" in linked_html
    assert "const sampleKey=o=>" in linked_html
    assert "memberKeys(skill)" in linked_html
    assert "Color samples by" in linked_html
    assert "color-model" in linked_html
    assert "color-legend" in linked_html
    assert "color-code" in linked_html
    assert "clicked code" in linked_html
    assert "linked code(s)" in linked_html
    assert "GT start" in linked_html
    assert "GT end" in linked_html
    assert "task_description" in linked_html
    assert "Cohesion" not in linked_html
    assert "Skill variety" not in linked_html
    assert "Codebook usage" not in linked_html


def test_categorization_partition_metrics_and_neighbor_adjustment() -> None:
    labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    permuted = np.asarray([2, 2, 0, 0, 1, 1], dtype=np.int64)
    assert CATEGORIZATION._normalized_mutual_info(labels, permuted) == pytest.approx(1.0)
    assert CATEGORIZATION._adjusted_rand(labels, permuted) == pytest.approx(1.0)

    tokens = np.asarray([0, 0, 1, 1], dtype=np.int64)
    neighbors = np.asarray([[1], [0], [3], [2]], dtype=np.int64)
    assert CATEGORIZATION._motion_neighbor_consistency(tokens, neighbors) == pytest.approx(1.0)
    collapsed = np.zeros(4, dtype=np.int64)
    assert CATEGORIZATION._motion_neighbor_consistency(collapsed, neighbors) == 0.0


def test_categorization_infers_global_dataset_root_and_builds_bundle(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    config_dir = repository / "lerobot/examples/libero/configs"
    config_dir.mkdir(parents=True)
    (config_dir / "global_config.yaml").write_text("dataset_root: dataset\n")

    model_path = repository / "outputs/FSQ/run/FSQ_epoch1.pt"
    model_path.parent.mkdir(parents=True)
    meta = {
        "fsq_dataset_root": "FSQ_dataset",
        "target_dataset": "langgap_ext_full_full",
        "fsq_inputs_name": "FSQ_inputs",
        "skillset_seg_name": "seg_test",
        "skillset_name": "skillset",
    }
    (model_path.parent / "fsq_meta.json").write_text(json.dumps(meta))
    skills_dir = (
        repository
        / "dataset/FSQ_dataset/langgap_ext_full_full/FSQ_inputs/seg_test/skillset/skills"
    )
    skills_dir.mkdir(parents=True)
    np.savez(
        skills_dir / "ep000_skill0.npz",
        actions=np.zeros((3, 7), dtype=np.float32),
        states=np.zeros((3, 8), dtype=np.float32),
        episode_id=np.int64(0),
        task_id=np.int64(0),
        skill_index=np.int64(0),
        frame_start=np.int64(0),
        frame_end=np.int64(3),
    )

    bundle, inferred_meta = CATEGORIZATION._infer_bundle(
        model_path, repository=repository
    )

    assert bundle == skills_dir.parent / "skills_bundle.npz"
    assert bundle.is_file()
    assert inferred_meta == meta


def test_categorization_checkpoint_scalars_read_only_pickle_metadata(tmp_path: Path) -> None:
    import pickle
    import zipfile

    checkpoint = tmp_path / "checkpoint.pt"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr(
            "checkpoint/data.pkl",
            pickle.dumps({"val_loss": 0.125, "val_select": 0.0625}, protocol=2),
        )

    assert CATEGORIZATION._checkpoint_scalars(checkpoint) == {
        "validation_total": 0.125,
        "validation_reconstruction": 0.0625,
    }


def test_categorization_report_has_checkpoint_driven_views(tmp_path: Path) -> None:
    checkpoint = {
        "epoch_tag": "epoch0100",
        "sample_count": 4,
        "metrics": {
            "motion_neighbor_consistency": 0.5,
            "motion_cohesion": 0.4,
            "direction_nmi": 0.3,
            "direction_coherence": 0.6,
            "opposite_adjacent_collision": 0.1,
            "validation_reconstruction": 0.05,
            "used_codes": 2,
            "effective_codes": 1.8,
            "largest_code_share": 0.6,
            "task_nmi": 0.2,
            "skill_index_nmi": 0.1,
            "gripper_nmi": 0.15,
            "adjacent_same_code": 0.25,
        },
        "group_predictability": {"Relative translation": 0.7},
        "correlation_features": ["disp_x"],
        "axis_correlations": [[0.5], [0.0], [-0.5]],
        "axis_strengths": [[0.4], [0.0], [0.4]],
        "code_features": ["disp_x"],
        "code_feature_means": [[1.0], [-1.0]],
        "direction_labels": list(CATEGORIZATION.DIRECTION_LABELS),
        "direction_colors": list(CATEGORIZATION.DIRECTION_COLORS),
        "direction_composition": [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]],
        "counts": [2, 2],
    }
    payload = {
        "format": CATEGORIZATION.FORMAT,
        "title": "analysis",
        "sample_count": 4,
        "feature_labels": CATEGORIZATION.FEATURE_LABELS,
        "models": [
            {
                "name": "model",
                "mode": "zero",
                "pair_loss": "js",
                "route_loss": True,
                "checkpoints": [checkpoint],
            }
        ],
        "common_checkpoints": ["epoch0100"],
        "pairwise": {"epoch0100": {"nmi": [[1.0]], "ari": [[1.0]]}},
    }

    path = CATEGORIZATION.write_report(tmp_path, payload)

    document = path.read_text(encoding="utf-8")
    assert 'id="model"' in document
    assert 'id="checkpoint"' in document
    assert "Motion-neighbor consistency" in document
    assert "FSQ axis ↔ trajectory feature map" in document
    assert "Code × semantic feature heatmap" in document
    assert "function renderTrend()" in document
    assert "function trendColor(index)" in document
    assert "function bindTrendInteractions()" in document
    assert 'class="trend-legend-item"' in document
    assert 'id="trendStatus"' in document
    assert (tmp_path / "categorization-data.js").is_file()
    assert (tmp_path / "metrics" / CATEGORIZATION.DATA_NAME).is_file()


def test_backfill_train_codebook_used_from_latents(tmp_path: Path) -> None:
    latents = tmp_path / "skill_latents.npz"
    np.savez(latents, tokens=np.asarray([0, 4, 4, 7], dtype=np.int32))
    manifest_path = tmp_path / "manifest.json"
    manifest = {"signature": {"latents_path": str(latents)}, "completed": True}
    manifest_path.write_text(json.dumps(manifest))

    REPORT._backfill_train_codebook_used(manifest_path, manifest)

    assert manifest["train_codebook_used"] == 3
    assert manifest["train_codebook_effective"] == pytest.approx(2.8284, abs=1e-3)
    assert manifest["train_codebook_counts"] == [1, 0, 0, 0, 2, 0, 0, 1]
    saved = json.loads(manifest_path.read_text())
    assert saved["train_codebook_used"] == 3
    assert saved["train_codebook_effective"] == pytest.approx(2.8284, abs=1e-3)
    assert saved["train_codebook_counts"] == [1, 0, 0, 0, 2, 0, 0, 1]


def test_backfill_adds_histogram_when_usage_summary_already_exists(tmp_path: Path) -> None:
    latents = tmp_path / "skill_latents.npz"
    np.savez(latents, tokens=np.asarray([1, 1, 2], dtype=np.int32))
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "signature": {"latents_path": str(latents)},
        "levels": [2, 2],
        "train_codebook_used": 2,
        "train_codebook_effective": 1.8899,
        "completed": True,
    }
    manifest_path.write_text(json.dumps(manifest))

    REPORT._backfill_train_codebook_used(manifest_path, manifest)

    assert manifest["train_codebook_counts"] == [0, 2, 1, 0]


def test_collection_keeps_checkpoint_codebooks_and_prefixes_media() -> None:
    def manifest(epoch: int, token: int) -> dict:
        epoch_tag = f"epoch{epoch:04d}"
        return {
            "levels": [3, 3, 3],
            "run_name": "fsq333",
            "epoch_tag": epoch_tag,
            "signature": {
                "target_task": "libero_90",
                "selected_episodes": {"1": [11], "3": [33]},
            },
            "records": {
                f"{epoch_tag}-sample": {
                    "token": token,
                    "task_id": 3,
                    "episode_id": 33,
                    "frame_start": 0,
                    "start_image_path": "images/sample_start.png",
                    "final_image_path": "images/sample_final.png",
                }
            },
        }

    payload = REPORT.collection_payload([manifest(125, 2), manifest(300, 7)])

    assert [item["epoch_tag"] for item in payload["checkpoints"]] == [
        "epoch0125",
        "epoch0300",
    ]
    assert [item["skills"][0]["token"] for item in payload["checkpoints"]] == [
        2,
        7,
    ]
    occurrence = payload["checkpoints"][1]["skills"][0]["occurrences"][0]
    assert occurrence["start_image_path"] == (
        "checkpoints/epoch0300/images/sample_start.png"
    )
    assert occurrence["final_image_path"] == (
        "checkpoints/epoch0300/images/sample_final.png"
    )
