"""Checkpoint uploads stay disabled without disabling metrics or touching files."""
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import Mock

import pytest

from lerobot.configs.default import WandBConfig
from lerobot.rl.wandb_utils import WandBLogger


@pytest.fixture
def logger_factory(monkeypatch, tmp_path):
    def make(disable_artifact=True, resume=False):
        run = Mock(id="test-run")
        run.get_url.return_value = "https://wandb.invalid/test-run"
        wandb = SimpleNamespace(init=Mock(return_value=run), Settings=Mock(),
                                Artifact=Mock(), log_artifact=Mock(), Video=Mock())
        monkeypatch.setitem(sys.modules, "wandb", wandb)
        cfg = SimpleNamespace(
            wandb=WandBConfig(enable=True, disable_artifact=disable_artifact,
                             run_id="test-run" if resume else None),
            output_dir=tmp_path, job_name="test", env=None, seed=1000,
            policy=SimpleNamespace(type="skill_expert"), dataset=None, resume=resume,
        )
        cfg.to_dict = lambda: {"wandb": {"disable_artifact": cfg.wandb.disable_artifact}}
        return WandBLogger(cfg), cfg, wandb, run
    return make


@pytest.mark.parametrize("disable_artifact", [True, False])
@pytest.mark.parametrize("weights", ["model.safetensors", "adapter_model.safetensors"])
@pytest.mark.parametrize("resume", [True, False])
def test_checkpoint_and_adapter_uploads_cannot_be_reenabled(
    logger_factory, tmp_path, disable_artifact, weights, resume,
):
    checkpoint = tmp_path / "checkpoints/001000"
    model = checkpoint / "pretrained_model" / weights
    model.parent.mkdir(parents=True)
    model.write_bytes(b"test weights")
    state = checkpoint / "training_state/optimizer_state.safetensors"
    state.parent.mkdir()
    state.write_bytes(b"test optimizer")
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    logger, cfg, wandb, run = logger_factory(disable_artifact, resume)
    assert cfg.wandb.disable_artifact is True
    assert wandb.init.call_args.kwargs["config"]["wandb"]["disable_artifact"] is True
    assert wandb.init.call_args.kwargs["resume"] == ("must" if resume else None)

    # Also guard an old caller changing the flag after constructing its logger.
    logger.cfg.disable_artifact = disable_artifact
    for path in (checkpoint, Path("/missing/checkpoints/000000")):
        logger.log_policy(path)
    wandb.Artifact.assert_not_called()
    wandb.log_artifact.assert_not_called()
    run.log_artifact.assert_not_called()
    assert {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_metrics_auxiliary_sections_and_videos_still_log(logger_factory):
    logger, _, wandb, run = logger_factory(False)
    logger.log_dict({"action_loss": 0.25}, step=100)
    logger.log_dict({"loss": 0.5}, step=100, mode="train_skill_flow")
    run.log.assert_any_call(data={"train/action_loss": 0.25}, step=100)
    run.log.assert_any_call(data={"train_skill_flow/loss": 0.5}, step=100)
    logger.log_video("example.mp4", step=100, mode="eval")
    wandb.Video.assert_called_once_with("example.mp4", fps=None, format="mp4")
    run.log.assert_any_call({"eval/video": wandb.Video.return_value}, step=100)
    logger.finish()
    run.finish.assert_called_once_with()
