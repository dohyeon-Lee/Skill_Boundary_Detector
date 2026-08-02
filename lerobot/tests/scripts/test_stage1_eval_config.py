import json
import sys
from pathlib import Path

import pytest


_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/stage1_eval/src"
)
sys.path.insert(0, str(_SRC))
from stage1_eval_config import build_settings  # noqa: E402


def _config(
    tmp_path: Path,
    *,
    architecture: str = "vsa_perceiver_crossattn",
    architecture_revision: str | None = "residual_sa18_v2",
    vision_conditioning_mode: str | None = None,
    include_state: bool | None = None,
    include_skill: bool | None = None,
) -> dict:
    project = tmp_path / "project"
    model_name = "new-vsa"
    policy_path = (
        project
        / "outputs/skillVLA_stage1"
        / model_name
        / "checkpoints/000100/pretrained_model"
    )
    policy_path.mkdir(parents=True)
    dataset = project / "dataset/skillvla_dataset/source/run/skillvla"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text("{}")
    dino = project / "models/dinov3-vitl16"
    dino.mkdir(parents=True)
    policy = {
        "type": "skill_expert",
        "architecture": architecture,
        "action_loss_mode": "flow",
        "dino_model_path": "models/dinov3-vitl16",
        "train_skill_predictor": False,
        "training_skill_source": "gt",
        "train_terminator": False,
        "skill_fsq_levels": [3, 3, 3],
        "chunk_size": 10,
        "n_action_steps": 5,
    }
    if architecture_revision is not None:
        policy["architecture_revision"] = architecture_revision
    if vision_conditioning_mode is not None:
        policy["vision_conditioning_mode"] = vision_conditioning_mode
    if include_state is not None:
        policy["include_state_in_visual_crossattn"] = include_state
    if include_skill is not None:
        policy["include_skill_in_visual_crossattn"] = include_skill
    (policy_path / "config.json").write_text(json.dumps(policy))
    (policy_path / "train_config.json").write_text(
        json.dumps({"dataset": {"root": "dataset/skillvla_dataset/source/run/skillvla"}})
    )
    for name in (
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        (policy_path / name).touch()
    return {
        "project_root": str(project),
        "outputs_root": "outputs",
        "models": [
            {
                "model_dir": model_name,
                "checkpoint": "000100",
                "skill_source": "gt",
                "advance_mode": "gt",
                "label": "new-vsa",
            }
        ],
        "output_name": "test-output",
        "n_action_steps": 5,
        "task_ids": [0],
        "logging": {"wandb": {"enable": False}},
    }


def test_eval_accepts_only_new_architecture_and_exports_it(tmp_path: Path) -> None:
    settings = build_settings(_config(tmp_path))
    models = json.loads(settings["models_json"])

    assert settings["architecture"] == "vsa_perceiver_crossattn"
    assert settings["n_action_steps"] == 5
    assert models[0]["architecture"] == "vsa_perceiver_crossattn"
    assert models[0]["architecture_revision"] == "residual_sa18_v2"
    assert settings["vision_conditioning_mode"] == "residual_cross_attention"
    assert models[0]["vision_conditioning_mode"] == "residual_cross_attention"
    assert models[0]["eval_legacy_vsa"] is False
    assert models[0]["num_visual_latents_per_camera"] == 32
    assert "conditioning_route" not in models[0]
    assert settings["include_state_in_visual_crossattn"] is False
    assert settings["include_skill_in_visual_crossattn"] is False
    assert settings["visual_crossattn_queries"] == "action-only"
    assert models[0]["visual_crossattn_queries"] == "action-only"


def test_eval_recognizes_pre_residual_vsa_checkpoint(tmp_path: Path) -> None:
    settings = build_settings(_config(tmp_path, architecture_revision=None))
    model = json.loads(settings["models_json"])[0]

    assert model["architecture_revision"] == "legacy_alternating_v1"
    assert model["eval_legacy_vsa"] is True
    assert model["vision_conditioning_mode"] == "legacy_alternating"
    assert model["num_visual_latents_per_camera"] == 8


@pytest.mark.parametrize(
    ("include_state", "include_skill", "expected"),
    [
        (True, False, "state + action"),
        (False, True, "skill + action"),
        (True, True, "state + skill + action"),
    ],
)
def test_eval_preserves_checkpoint_visual_query_contract(
    tmp_path: Path,
    include_state: bool,
    include_skill: bool,
    expected: str,
) -> None:
    settings = build_settings(
        _config(tmp_path, include_state=include_state, include_skill=include_skill)
    )
    model = json.loads(settings["models_json"])[0]

    assert model["include_state_in_visual_crossattn"] is include_state
    assert model["include_skill_in_visual_crossattn"] is include_skill
    assert model["visual_crossattn_queries"] == expected


@pytest.mark.parametrize(
    "mode", ["in_context_tokens", "global_visual_adarms"]
)
def test_eval_selects_non_residual_mode_from_checkpoint_metadata(
    tmp_path: Path, mode: str
) -> None:
    settings = build_settings(
        _config(
            tmp_path,
            vision_conditioning_mode=mode,
            include_state=True,
            include_skill=True,
        )
    )
    model = json.loads(settings["models_json"])[0]

    assert model["vision_conditioning_mode"] == mode
    assert model["visual_crossattn_queries"] == "ignored"


def test_eval_rejects_unknown_checkpoint_vision_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported vision_conditioning_mode"):
        build_settings(
            _config(tmp_path, vision_conditioning_mode="name-derived-fallback")
        )


def test_eval_accepts_auxiliaries_from_legacy_main_architecture(tmp_path: Path) -> None:
    config = _config(tmp_path, include_state=True, include_skill=True)
    project = Path(config["project_root"])
    target = (
        project
        / config["outputs_root"]
        / "skillVLA_stage1/new-vsa/checkpoints/000100/pretrained_model"
    )
    target_policy = json.loads((target / "config.json").read_text())
    fsq = project / "dataset/skillvla_dataset/source/run/FSQ.pt"
    fsq.touch()
    target_policy["fsq_path"] = str(fsq)
    (target / "config.json").write_text(json.dumps(target_policy))

    external = project / "external/legacy-stage1/checkpoints/030000/pretrained_model"
    external.mkdir(parents=True)
    tokenizer = project / "models/tokenizer"
    tokenizer.mkdir(parents=True)
    source_policy = {
        **target_policy,
        "architecture": "legacy-state-skill-main",
        "conditioning_route": "state_skill_cond",
        "train_skill_predictor": True,
        "train_terminator": True,
        "tokenizer_path": str(tokenizer),
        "terminator_dino_model_path": target_policy["dino_model_path"],
    }
    (external / "config.json").write_text(json.dumps(source_policy))
    (external / "model.safetensors").touch()

    config["external_skill_model"] = str(external)
    config["models"][0]["skill_source"] = "external"
    config["models"][0]["advance_mode"] = "external"
    model = json.loads(build_settings(config)["models_json"])[0]

    assert model["architecture"] == "vsa_perceiver_crossattn"
    assert model["skill_source"] == "external"
    assert model["advance_mode"] == "external"
    assert Path(model["external_skill_model"]) == external


def test_eval_rejects_legacy_stage1_checkpoint(tmp_path: Path) -> None:
    config = _config(tmp_path, architecture="state_skill_cond")

    with pytest.raises(ValueError, match="legacy Stage-1 checkpoints"):
        build_settings(config)


def test_eval_rejects_replanning_beyond_chunk(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["n_action_steps"] = 11

    with pytest.raises(ValueError, match="exceeds"):
        build_settings(config)
