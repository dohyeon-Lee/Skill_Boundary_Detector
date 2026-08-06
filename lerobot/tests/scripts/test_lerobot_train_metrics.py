import math

import torch

from lerobot.scripts.lerobot_train import (
    _WINDOWED_POLICY_MODEL_TYPES,
    _WindowedPolicyMetrics,
    _finite_scalar_metrics,
    _split_namespaced_metrics,
    _sparse_debug_metric_groups,
)


def test_skill_vla_uses_windowed_policy_metrics() -> None:
    assert "skill_vla" in _WINDOWED_POLICY_MODEL_TYPES


def test_windowed_policy_metrics_keeps_regime_means_separate() -> None:
    metrics = _WindowedPolicyMetrics()
    metrics.update({
        "action_loss": 2.0,
        "endpoint_xyz_loss": 4.0,
        "action_objective": 3.0,
        "future_metric_needing_no_allowlist": 10.0,
        "loss_flow": 3.0,
        "regime/A_active": 1.0,
        "regime/B_active": 0.0,
        "regime/A_flow_loss": 2.0,
    })
    metrics.update({
        "action_loss": 6.0,
        "endpoint_xyz_loss": 8.0,
        "action_objective": 7.0,
        "future_metric_needing_no_allowlist": 14.0,
        "loss_flow": 9.0,
        "regime/A_active": 0.0,
        "regime/B_active": 1.0,
        "regime/B_anchor_loss": 4.0,
        "regime/B_image_free_action_loss": 6.0,
    })

    assert metrics.averages() == {
        "action_loss": 4.0,
        "endpoint_xyz_loss": 6.0,
        "action_objective": 5.0,
        "future_metric_needing_no_allowlist": 12.0,
        "loss_flow": 6.0,
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


def test_windowed_policy_metrics_does_not_average_sparse_vsa_debug_values() -> None:
    metrics = _WindowedPolicyMetrics()
    metrics.update(
        {
            "action_loss": 2.0,
            "vsa_debug/visual/top_latents/effective_rank_fraction": 0.45,
        }
    )

    assert metrics.averages() == {"action_loss": 2.0}


def test_wandb_metric_filter_keeps_all_finite_scalars_without_name_allowlist() -> None:
    filtered = _finite_scalar_metrics(
        {
            "brand_new_metric": 3,
            "scalar_tensor": torch.tensor(4.0),
            "loss_per_dim": [1.0, 2.0],
            "vector_tensor": torch.tensor([1.0, 2.0]),
            "not_finite": math.nan,
            "label": "ignored",
        }
    )

    assert filtered == {"brand_new_metric": 3.0, "scalar_tensor": 4.0}


def test_skill_aux_metric_namespaces_need_no_prefix_allowlist() -> None:
    main, groups = _split_namespaced_metrics(
        {
            "steps": 100.0,
            "start_comparison_terminator/loss": 0.4,
            "future_auxiliary_head/end_f1": 0.7,
        }
    )

    assert main == {"steps": 100.0}
    assert groups == {
        "start_comparison_terminator": {"loss": 0.4},
        "future_auxiliary_head": {"end_f1": 0.7},
    }


def test_input_influence_is_removed_from_vsa_debug_namespace() -> None:
    vsa_debug, input_influence = _sparse_debug_metric_groups(
        {
            "vsa_debug/visual/top_latents/effective_rank_fraction": 0.5,
            "vsa_debug/sensitivity/state_shuffle/relative_output_delta": 0.2,
            "action_loss": 1.0,
        }
    )

    assert vsa_debug == {"visual/top_latents/effective_rank_fraction": 0.5}
    assert input_influence == {"state_shuffle/relative_output_delta": 0.2}
