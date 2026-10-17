"""Run with .venv_molmoact2: real data/stats, spawn reopening, and pi05 LR parity."""
from __future__ import annotations

import ast
import dataclasses
import multiprocessing
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("olmo", reason="Requires the separate MolmoAct2 environment")
PROJECT = Path(__file__).resolve().parents[3]
SRC = PROJECT / "lerobot/examples/libero/configs/train_molmoact2/PT/src"
sys.path.insert(0, str(SRC))
from molmoact2_config import build_settings, checkpoint_grounding, load_config
from train import prepare_environment, register_mixture


@pytest.fixture(scope="module")
def runtime():
    root = PROJECT / "dataset_filtered/libero_10_full_1"
    if not (root / "meta/info.json").is_file():
        pytest.skip("Local LIBERO dataset is required for the runtime checks")
    settings = build_settings(load_config(SRC.parent / "molmoact2_PT_config.yaml"))
    settings.update(dataset="libero_10_full_1", dataset_dir=str(root), proprio_grounding="episode_start_xyz")
    prepare_environment(settings)
    import grounding_adapter
    grounding_adapter.install("episode_start_xyz")
    register_mixture(settings)
    return settings, grounding_adapter


def test_grounded_data_and_stats_reach_the_original_processor(runtime, tmp_path):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from launch_scripts.lerobot_utils.train_plan import get_lerobot_training_data_plan
    from launch_scripts.lerobot_utils import stats as mixture_stats
    from olmo.data.robot_processing import RobotProcessorConfig
    from launch_scripts.data_mixtures import TAG_METADATA_BY_TAG
    settings, adapter = runtime
    kwargs = dict(repo_id=settings["dataset"], root=Path(settings["dataset_dir"]), episodes=[0], video_backend="pyav",
                  delta_timestamps={"action": [i / 20 for i in range(10)]})
    raw = adapter._BaseDataset(**kwargs)
    grounded = adapter.GroundedMolmoDataset(**kwargs)
    reference = raw[0]["observation.state"][:3]
    for index in [0, 5]:
        original, transformed = raw[index], grounded[index]
        torch.testing.assert_close(transformed["observation.state"][:3], original["observation.state"][:3] - reference)
        torch.testing.assert_close(transformed["observation.state"][3:], original["observation.state"][3:])
        torch.testing.assert_close(transformed["action"], original["action"])
    plan = get_lerobot_training_data_plan("sbd_libero", style_robot_action=1.0, style_robot_depth=0.0, style_robot_depth_action=0.0)
    stats, repo_to_tag, _ = mixture_stats._collect_tagged_stats(
        plan.robot_mixture, root_base=settings["dataset_root"], tag_metadata_by_tag=TAG_METADATA_BY_TAG,
    )
    artifact = adapter.artifact_for_metadata(LeRobotDatasetMetadata(settings["dataset"], root=Path(settings["dataset_dir"])))
    for key in ["mean", "std", "q01", "q99"]:
        np.testing.assert_allclose(stats["libero"]["observation.state"][key], artifact.state_stats[key])
    processor = RobotProcessorConfig.from_stats(
        stats, tag_metadata={"libero": TAG_METADATA_BY_TAG["lerobot:libero"]}, repo_to_tag=repo_to_tag,
        norm_mode="q01_q99",
    )
    saved = {"model": {"robot_processor": dataclasses.asdict(processor)}}
    (tmp_path / "config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(saved)))
    assert checkpoint_grounding(tmp_path) == "episode_start_xyz"
    # Raw stats on disk/in a new raw metadata view remain absolute-coordinate stats.
    assert not np.allclose(LeRobotDatasetMetadata(settings["dataset"], root=Path(settings["dataset_dir"])).stats["observation.state"]["mean"][:3],
                           artifact.state_stats["mean"][:3])


def _read_after_spawn(wrapper, connection):
    try:
        import grounding_adapter
        grounding_adapter.worker_init(0)
        was_closed = wrapper.dataset is None
        wrapper._ensure_dataset_loaded()
        first_xyz = wrapper.dataset[0]["observation.state"][:3].tolist()
        connection.send((was_closed, first_xyz, type(wrapper.dataset).__name__))
    finally:
        connection.close()


def test_spawn_worker_reopens_a_grounded_dataset(runtime):
    from olmo.data.lerobot_wrapper import LeRobotDatasetWrapper
    settings, adapter = runtime
    kwargs = dict(repo_id=settings["dataset"], root=Path(settings["dataset_dir"]), episodes=[0], video_backend="pyav",
                  delta_timestamps={"action": [i / 20 for i in range(10)]})
    ds = adapter.GroundedMolmoDataset(**kwargs)
    wrapper = LeRobotDatasetWrapper(
        ds, split="train", camera_keys=["observation.images.image", "observation.images.wrist_image"],
        camera_keys_alternative=None, question_key="task", state_keys=["observation.state"], action_keys=["action"],
        observation_indices=[0], action_indices=list(range(10)), tag_action_horizon=10, tag_n_action_steps=10,
        max_action_horizon=10, max_action_dim=32, dataset_reopen_kwargs=kwargs,
    )
    ctx = multiprocessing.get_context("spawn")
    receiver, sender = ctx.Pipe(duplex=False)
    worker = ctx.Process(target=_read_after_spawn, args=(wrapper, sender))
    worker.start()
    sender.close()
    try:
        assert receiver.poll(120), "Spawned loader did not return a sample"
        closed, xyz, cls = receiver.recv()
        assert closed is True
        assert xyz == pytest.approx([0, 0, 0])
        assert cls == "GroundedMolmoDataset"
        worker.join(20)
        assert worker.exitcode == 0
    finally:
        receiver.close()
        if worker.is_alive():
            worker.terminate()
            worker.join(10)


@pytest.mark.parametrize("warmup", [0, 2, 1000])
def test_lr_matches_the_actual_pi05_scheduler_at_each_optimizer_step(runtime, warmup):
    import torch
    from olmo.train.optim import SchedulerConfig
    # Execute the real pi05 build method without importing its separate LeRobot package.
    tree = ast.parse((PROJECT / "lerobot/src/lerobot/optim/schedulers.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "WarmupConstantSchedulerConfig")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "build")
    namespace = {"Optimizer": torch.optim.Optimizer, "LambdaLR": torch.optim.lr_scheduler.LambdaLR}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "pi05_warmup_constant", "exec"), namespace)
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=2.5e-5)
    pi05 = namespace["build"](SimpleNamespace(num_warmup_steps=warmup), optimizer, 200000)
    cfg = SchedulerConfig(alpha_f=1.0, warmup_min_lr=0.0, connector_t_warmup=warmup + 1,
                          vit_t_warmup=warmup + 1, llm_t_warmup=warmup + 1, action_expert_t_warmup=warmup + 1)
    molmo = cfg.build()
    for k in range(1, warmup + 20):
        for group in ["connector", "vit", "llm", "action_expert"]:
            assert molmo.get_lr(2.5e-5, k, 200000, group) == pytest.approx(pi05.get_last_lr()[0], rel=1e-12)
        optimizer.step()
        pi05.step()
    assert molmo.get_lr(2.5e-5, 200000, 200000, "llm") == 2.5e-5
