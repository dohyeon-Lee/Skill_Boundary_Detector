from lerobot.scripts.lerobot_train import _pi05_wandb_train_metrics, _policy_kind


def test_pi05_logs_only_stage1_comparable_curves() -> None:
    logged = _pi05_wandb_train_metrics(
        {"loss": 0.5, "grad_norm": 1.0, "lr": 2.5e-5, "update_s": 0.3, "dataloading_s": 0.1,
         "steps": 100, "samples": 1600, "episodes": 3.0, "epochs": 0.01}
    )
    assert logged == {"action_loss": 0.5, "grad_norm": 1.0, "lr": 2.5e-5}


def test_pi05_action_loss_is_the_window_average_not_the_last_batch() -> None:
    """Regression: pi05.forward reports its own ``loss`` (last batch) which used to overwrite the
    tracker average in the merged W&B payload, so the curve was an unsmoothed single-batch loss."""
    from pathlib import Path

    import lerobot.scripts.lerobot_train as train_script

    source = Path(train_script.__file__).read_text()
    call = "main_metrics = _pi05_wandb_train_metrics("
    following = source[source.index(call) + len(call):][:120]
    assert "train_tracker.to_dict()" in following


def test_pi05_config_takes_the_pi05_logging_branch() -> None:
    """Regression: the branch checked only ``model_type``, which PI05Config does not have, so pi05 runs
    fell through and logged the last batch's ``train/loss`` (plus steps/samples/...) instead."""
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.skill_expert.configuration_skill_expert import SkillExpertConfig

    assert _policy_kind(PI05Config()) == "pi05"
    assert _policy_kind(SkillExpertConfig()) == "skill_expert"

    from pathlib import Path

    import lerobot.scripts.lerobot_train as train_script

    source = Path(train_script.__file__).read_text()
    assert 'getattr(cfg.policy, "model_type", None) == "pi05"' not in source
    assert "policy_model_type = _policy_kind(cfg.policy)" in source
