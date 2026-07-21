from lerobot.scripts.lerobot_train import _WindowedPolicyMetrics


def test_windowed_policy_metrics_keeps_regime_means_separate() -> None:
    metrics = _WindowedPolicyMetrics()
    metrics.update({
        "action_loss": 2.0,
        "regime/A_active": 1.0,
        "regime/B_active": 0.0,
        "regime/A_flow_loss": 2.0,
    })
    metrics.update({
        "action_loss": 6.0,
        "regime/A_active": 0.0,
        "regime/B_active": 1.0,
        "regime/B_anchor_loss": 4.0,
        "regime/B_image_free_action_loss": 6.0,
    })

    assert metrics.averages() == {
        "action_loss": 4.0,
        "regime/A_active": 0.5,
        "regime/B_active": 0.5,
        "regime/A_flow_loss": 2.0,
        "regime/B_anchor_loss": 4.0,
        "regime/B_image_free_action_loss": 6.0,
    }


def test_windowed_policy_metrics_tracks_stage0_pretrain_ar_values() -> None:
    metrics = _WindowedPolicyMetrics()
    metrics.update({"ar/skill_ce": 2.0, "ar/skill_exact_acc": 0.0})
    metrics.update({"ar/skill_ce": 1.0, "ar/skill_exact_acc": 0.5})

    assert metrics.averages() == {
        "ar/skill_ce": 1.5,
        "ar/skill_exact_acc": 0.25,
    }
