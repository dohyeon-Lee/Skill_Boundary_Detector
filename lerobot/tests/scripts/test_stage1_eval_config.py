import json
import sys
from pathlib import Path

import pytest


_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/stage1_eval/src"
)
sys.path.insert(0, str(_SRC))
from stage1_eval_config import (  # noqa: E402
    _checkpoint_contract,
    _effective_latent_source,
    _model_entries,
)


def _checkpoint(
    tmp_path: Path,
    *,
    label: str = "arch0",
    architecture: str = "cond_gemma",
    revision: str = "skillvla_real_v1",
) -> tuple[Path, Path]:
    project = tmp_path / "project"
    policy_path = project / "outputs/skillVLA_stage1/run/checkpoints/000100/pretrained_model"
    policy_path.mkdir(parents=True)
    dataset = project / "dataset/skillvla_dataset/source/run/skillvla"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text(
        json.dumps({"proprio_grounding": "none"})
    )
    dino = project / "models/dinov3-vitl16"
    dino.mkdir(parents=True)
    skill_flow_enabled = label not in {"arch0", "arch1", "arch2"}
    policy = {
        "type": "skill_expert",
        "architecture": architecture,
        "architecture_label": label,
        "architecture_revision": revision,
        "vision_conditioning_mode": (
            "fixed_bottleneck_cross_attention"
            if label.startswith(("arch1", "arch2"))
            else "interleaved_cross_attention"
        ),
        "conditioning_route": "state_cond",
        "action_loss_mode": "flow",
        "skill_flow_enabled": skill_flow_enabled,
        "skill_flow_target": (
            "extended_chunk" if label.endswith("_skill_chunk") else "canonical"
        ),
        "skill_flow_state_conditioned": False,
        "proprio_grounding": "none",
        "dino_model_path": "models/dinov3-vitl16",
        "train_skill_predictor": False,
        "training_skill_source": "gt",
        "train_terminator": False,
        "fsq_path": "dataset/skillvla_dataset/source/run/FSQ.pt",
    }
    (policy_path / "config.json").write_text(json.dumps(policy))
    (policy_path / "train_config.json").write_text(
        json.dumps(
            {"dataset": {"root": "dataset/skillvla_dataset/source/run/skillvla"}}
        )
    )
    for name in (
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        (policy_path / name).touch()
    return project, policy_path


@pytest.mark.parametrize(
    ("label", "architecture", "revision"),
    [
        ("arch0", "cond_gemma", "skillvla_real_v1"),
        ("arch0_skill", "cond_gemma", "skillvla_real_v1"),
        ("arch0_skill_chunk", "cond_gemma", "skillvla_real_v1"),
        ("arch1", "fixed_visual_bottleneck", "fixed_visual_bottleneck_v1"),
        ("arch1_skill", "fixed_visual_bottleneck", "fixed_visual_bottleneck_v1"),
        (
            "arch1_skill_chunk",
            "fixed_visual_bottleneck",
            "fixed_visual_bottleneck_v1",
        ),
        ("arch2", "fixed_visual_bottleneck", "late_visual_bottleneck_v1"),
        (
            "arch2_skill",
            "fixed_visual_bottleneck",
            "late_visual_bottleneck_v1",
        ),
        (
            "arch2_skill_chunk",
            "fixed_visual_bottleneck",
            "late_visual_bottleneck_v1",
        ),
    ],
)
def test_checkpoint_contract_accepts_retained_modes(
    tmp_path: Path, label: str, architecture: str, revision: str
) -> None:
    project, policy_path = _checkpoint(
        tmp_path,
        label=label,
        architecture=architecture,
        revision=revision,
    )
    contract = _checkpoint_contract(policy_path, project)

    assert contract["architecture"] == architecture
    assert contract["architecture_revision"] == revision
    assert contract["architecture_label"] == label
    assert contract["conditioning_route"] == "state_cond"
    assert contract["vision_conditioning_mode"] == (
        "fixed_bottleneck_cross_attention"
        if label.startswith(("arch1", "arch2"))
        else "interleaved_cross_attention"
    )


def test_checkpoint_contract_rejects_removed_modes(tmp_path: Path) -> None:
    project, policy_path = _checkpoint(tmp_path, label="removed_mode")
    with pytest.raises(ValueError, match="supports only arch0"):
        _checkpoint_contract(policy_path, project)


def test_checkpoint_contract_rejects_old_model_implementation(tmp_path: Path) -> None:
    project, policy_path = _checkpoint(
        tmp_path,
        label="removed_mode",
        architecture="removed_architecture",
        revision="removed_revision",
    )
    with pytest.raises(ValueError, match="supports only arch0"):
        _checkpoint_contract(policy_path, project)


def test_checkpoint_contract_rejects_arch1_with_arch0_implementation(
    tmp_path: Path,
) -> None:
    project, policy_path = _checkpoint(tmp_path, label="arch1")
    with pytest.raises(ValueError, match="requires architecture='fixed_visual_bottleneck'"):
        _checkpoint_contract(policy_path, project)


def test_checkpoint_label_must_match_auxiliary_objective(tmp_path: Path) -> None:
    project, policy_path = _checkpoint(tmp_path, label="arch0_skill")
    config_path = policy_path / "config.json"
    config = json.loads(config_path.read_text())
    config["skill_flow_enabled"] = False
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="skill_flow_enabled=True"):
        _checkpoint_contract(policy_path, project)


def test_foveated_checkpoint_requires_portable_gt_focus_artifact(
    tmp_path: Path,
) -> None:
    project, policy_path = _checkpoint(
        tmp_path,
        label="arch1",
        architecture="fixed_visual_bottleneck",
        revision="fixed_visual_bottleneck_v1",
    )
    config_path = policy_path / "config.json"
    config = json.loads(config_path.read_text())
    config["foveated_vision_enabled"] = True
    config_path.write_text(json.dumps(config))
    dataset = project / "dataset/skillvla_dataset/source/run/skillvla"
    info_path = dataset / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["skill_focus_uv_path"] = "/old/server/run/skill_focus_uv.npz"
    info_path.write_text(json.dumps(info))

    with pytest.raises(FileNotFoundError, match="focus artifact"):
        _checkpoint_contract(policy_path, project)

    portable = dataset.parent / "skill_focus_uv.npz"
    portable.touch()
    contract = _checkpoint_contract(policy_path, project)

    assert contract["foveated_vision_enabled"] is True
    assert contract["focus_uv_path"] == portable


def test_model_defaults_are_inherited_and_model_values_override_them() -> None:
    entries = _model_entries(
        {
            "model_defaults": {
                "checkpoint": "015000",
                "skill_source": "gt",
                "advance_mode": "external",
                "external_skill_model": "outputs/shared/ckpt",
            },
            "models": [
                {"model_dir": "first", "label": "first"},
                {
                    "model_dir": "second",
                    "label": "second",
                    "checkpoint": "030000",
                    "skill_source": "own",
                    "advance_mode": "gt",
                },
            ],
        }
    )

    assert entries[0]["checkpoint"] == "015000"
    assert entries[0]["skill_source"] == "gt"
    assert entries[0]["advance_mode"] == "external"
    assert entries[1]["checkpoint"] == "030000"
    assert entries[1]["skill_source"] == "own"
    assert entries[1]["advance_mode"] == "gt"
    assert "previous_checkpoint" not in entries[0]


def test_previous_historical_selector_is_removed() -> None:
    with pytest.raises(ValueError, match="Unknown model_defaults"):
        _model_entries(
            {
                "model_defaults": {"previous": True},
                "models": [{"model_dir": "old", "label": "old"}],
            }
        )


def test_oracle_latent_is_ignored_for_latent_free_checkpoint() -> None:
    assert (
        _effective_latent_source(
            "oracle", {"skill_flow_latent_best_of_n_enabled": False}
        )
        == "random"
    )
    assert (
        _effective_latent_source(
            "oracle", {"skill_flow_latent_best_of_n_enabled": True}
        )
        == "oracle"
    )
