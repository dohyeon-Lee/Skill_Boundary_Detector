import json
import shutil
import sys
from pathlib import Path

import pytest


_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/stage1_eval/src"
)
sys.path.insert(0, str(_SRC))
from stage1_eval_config import _model_entries, build_settings  # noqa: E402


def _config(
    tmp_path: Path,
    *,
    architecture: str | None = "vsa_perceiver_crossattn",
    architecture_revision: str | None = "interleaved_direct1024_v3",
    vision_conditioning_mode: str | None = None,
    conditioning_route: str | None = None,
    include_state: bool | None = True,
    include_skill: bool | None = True,
    model_name: str = "new-vsa",
    visual_perceiver_width: int | None = None,
    previous: bool = False,
) -> dict:
    project = tmp_path / "project"
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
        "action_loss_mode": "flow",
        "dino_model_path": "models/dinov3-vitl16",
        "train_skill_predictor": False,
        "training_skill_source": "gt",
        "train_terminator": False,
        "skill_fsq_levels": [3, 3, 3],
        "chunk_size": 10,
        "n_action_steps": 5,
    }
    if architecture is not None:
        policy["architecture"] = architecture
    if architecture_revision is not None:
        policy["architecture_revision"] = architecture_revision
    if vision_conditioning_mode is not None:
        policy["vision_conditioning_mode"] = vision_conditioning_mode
    if conditioning_route is not None:
        policy["conditioning_route"] = conditioning_route
    if include_state is not None:
        policy["include_state_in_visual_crossattn"] = include_state
    if include_skill is not None:
        policy["include_skill_in_visual_crossattn"] = include_skill
    if visual_perceiver_width is not None:
        policy["visual_perceiver_width"] = visual_perceiver_width
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
                "previous": previous,
            }
        ],
        "output_name": "test-output",
        "n_action_steps": 5,
        "task_ids": [0],
        "logging": {"wandb": {"enable": False}},
    }


def test_model_defaults_are_inherited_and_model_values_override_them() -> None:
    entries = _model_entries(
        {
            "model_defaults": {
                "previous": True,
                "checkpoint": "015000",
                "skill_source": "gt",
                "advance_mode": "external",
                "external_skill_model": "outputs/shared/ckpt",
            },
            "models": [
                {"model_dir": "historical", "label": "historical"},
                {
                    "model_dir": "current",
                    "label": "current",
                    "previous": False,
                    "checkpoint": "030000",
                    "skill_source": "own",
                    "advance_mode": "gt",
                    "external_skill_model": "outputs/own/ckpt",
                },
            ],
        }
    )

    assert entries[0] == {
        "model_dir": "historical",
        "checkpoint": "015000",
        "skill_source": "gt",
        "advance_mode": "external",
        "terminator_variant": "state_image",
        "label": "historical",
        "model_label": "historical",
        "model_index": 0,
        "checkpoint_index": 0,
        "previous_checkpoint": True,
        # Inherited from model_defaults; one value covers both overlays.
        "external_predictor_model_value": "outputs/shared/ckpt",
        "external_predictor_checkpoint": "last",
        "external_terminator_model_value": "outputs/shared/ckpt",
        "external_terminator_checkpoint": "last",
    }
    assert entries[1] == {
        "model_dir": "current",
        "checkpoint": "030000",
        "skill_source": "own",
        "advance_mode": "gt",
        "terminator_variant": "state_image",
        "label": "current",
        "model_label": "current",
        "model_index": 1,
        "checkpoint_index": 0,
        "previous_checkpoint": False,
        # A per-entry value wins, so each target can use its own terminator.
        "external_predictor_model_value": "outputs/own/ckpt",
        "external_predictor_checkpoint": "last",
        "external_terminator_model_value": "outputs/own/ckpt",
        "external_terminator_checkpoint": "last",
    }


def test_checkpoint_list_expands_checkpoint_major_for_automatic_grid() -> None:
    entries = _model_entries(
        {
            "model_defaults": {
                "checkpoint": ["050000", "070000", "090000"],
                "skill_source": "gt",
                "advance_mode": "gt",
            },
            "models": [
                {"model_dir": "arch3", "label": "arch3"},
                {"model_dir": "arch4", "label": "arch4"},
            ],
        }
    )

    assert [entry["checkpoint"] for entry in entries] == [
        "050000",
        "050000",
        "070000",
        "070000",
        "090000",
        "090000",
    ]
    assert [entry["model_index"] for entry in entries] == [0, 1, 0, 1, 0, 1]
    assert [entry["checkpoint_index"] for entry in entries] == [0, 0, 1, 1, 2, 2]
    assert [entry["label"] for entry in entries] == [
        "arch3 | ckpt 050000",
        "arch4 | ckpt 050000",
        "arch3 | ckpt 070000",
        "arch4 | ckpt 070000",
        "arch3 | ckpt 090000",
        "arch4 | ckpt 090000",
    ]


def test_grid_uses_models_as_columns_and_checkpoints_as_rows(tmp_path: Path) -> None:
    config = _config(tmp_path)
    project = Path(config["project_root"])
    checkpoint_root = project / "outputs/skillVLA_stage1/new-vsa/checkpoints"
    shutil.copytree(checkpoint_root / "000100", checkpoint_root / "000200")
    config["model_defaults"] = {
        "checkpoint": ["000100", "000200"],
        "skill_source": "gt",
        "advance_mode": "gt",
    }
    config["models"] = [
        {"model_dir": "new-vsa", "label": "model-a"},
        {"model_dir": "new-vsa", "label": "model-b"},
    ]

    settings = build_settings(config)
    models = json.loads(settings["models_json"])

    assert settings["model_count"] == 2
    assert settings["checkpoint_count"] == 2
    assert settings["grid_columns"] == 2
    assert [(model["checkpoint_index"], model["model_index"]) for model in models] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]


@pytest.mark.parametrize("workers", [0, 5])
def test_eval_workers_per_gpu_is_bounded(tmp_path: Path, workers: int) -> None:
    config = _config(tmp_path)
    config["eval_max_workers_per_gpu"] = workers

    with pytest.raises(ValueError, match="between 1 and 4"):
        build_settings(config)


def test_checkpoint_grid_rejects_different_columns_per_model() -> None:
    with pytest.raises(ValueError, match="same ordered checkpoint list"):
        _model_entries(
            {
                "models": [
                    {
                        "model_dir": "arch3",
                        "label": "arch3",
                        "checkpoint": ["050000", "070000"],
                    },
                    {
                        "model_dir": "arch4",
                        "label": "arch4",
                        "checkpoint": ["050000", "090000"],
                    },
                ]
            }
        )


def test_eval_accepts_new_architecture_and_exports_it(tmp_path: Path) -> None:
    settings = build_settings(_config(tmp_path))
    models = json.loads(settings["models_json"])

    assert settings["architecture"] == "vsa_perceiver_crossattn"
    assert settings["model_count"] == 1
    assert settings["checkpoint_count"] == 1
    assert settings["panel_count"] == 1
    assert settings["grid_columns"] == 1
    assert settings["n_action_steps"] == 5
    assert settings["eval_expected_tasks"] == 1
    assert settings["immediate_replan_on_skill_end"] is False
    assert models[0]["architecture"] == "vsa_perceiver_crossattn"
    assert models[0]["architecture_revision"] == "interleaved_direct1024_v3"
    assert models[0]["architecture_label"] == "arch2_2"
    assert settings["vision_conditioning_mode"] == "interleaved_cross_attention"
    assert models[0]["vision_conditioning_mode"] == "interleaved_cross_attention"
    assert models[0]["eval_legacy_vsa"] is False
    assert models[0]["eval_vsa_revision"] == ""
    assert models[0]["num_visual_latents_per_camera"] == 32
    assert models[0]["visual_perceiver_width"] == 1024
    assert models[0]["previous_checkpoint"] is False
    assert "conditioning_route" not in models[0]
    assert settings["include_state_in_visual_crossattn"] is True
    assert settings["include_skill_in_visual_crossattn"] is True
    assert settings["visual_crossattn_queries"] == "state + skill + action"
    assert models[0]["visual_crossattn_queries"] == "state + skill + action"


@pytest.mark.parametrize(
    ("revision", "mode", "label", "tokens"),
    [
        (
            "visual_kv_uncompressed_v1",
            "uncompressed_visual_kv_self_attention",
            "arch1_3",
            197,
        ),
        (
            "visual_kv_perceiver_v1",
            "compressed_visual_kv_self_attention",
            "arch2_1",
            32,
        ),
    ],
)
def test_eval_resolves_visual_kv_self_attention_revisions(
    tmp_path: Path, revision: str, mode: str, label: str, tokens: int
) -> None:
    settings = build_settings(
        _config(
            tmp_path,
            architecture_revision=revision,
            vision_conditioning_mode=mode,
        )
    )
    model = json.loads(settings["models_json"])[0]

    assert model["architecture_label"] == label
    assert model["architecture_revision"] == revision
    assert model["vision_conditioning_mode"] == mode
    assert model["num_visual_latents_per_camera"] == tokens
    assert model["visual_crossattn_queries"] == "expert queries; visual fixed KV"


def test_eval_recognizes_pre_residual_vsa_checkpoint(tmp_path: Path) -> None:
    settings = build_settings(_config(tmp_path, architecture_revision=None))
    model = json.loads(settings["models_json"])[0]

    assert model["architecture_revision"] == "legacy_alternating_v1"
    assert model["eval_legacy_vsa"] is True
    assert model["eval_vsa_revision"] == "legacy_alternating_v1"
    assert model["vision_conditioning_mode"] == "legacy_alternating"
    assert model["num_visual_latents_per_camera"] == 8


def test_eval_recognizes_previous_vsa_folder_and_restores_width(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        model_name="old-vsa",
        architecture_revision="residual_sa18_v2",
        previous=True,
    )
    project = Path(config["project_root"])
    source = project / "outputs/skillVLA_stage1/old-vsa"
    destination = project / "outputs/skillVLA_stage1/previous/old-vsa"
    destination.parent.mkdir(parents=True)
    source.rename(destination)
    settings = build_settings(config)
    model = json.loads(settings["models_json"])[0]

    assert model["previous_checkpoint"] is True
    assert model["visual_perceiver_width"] == 384
    assert model["architecture_label"] == "arch2_2"
    assert model["eval_vsa_revision"] == "residual_sa18_v2"


def test_eval_rejects_previous_folder_on_nonhistorical_width(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        model_name="not-old",
        architecture_revision="residual_sa18_v2",
        visual_perceiver_width=1024,
        previous=True,
    )
    project = Path(config["project_root"])
    source = project / "outputs/skillVLA_stage1/not-old"
    destination = project / "outputs/skillVLA_stage1/previous/not-old"
    destination.parent.mkdir(parents=True)
    source.rename(destination)
    with pytest.raises(ValueError, match="visual_perceiver_width=384"):
        build_settings(config)


def test_eval_accepts_current_explicit_cond_checkpoint(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        architecture="cond_gemma",
        architecture_revision="skillvla_real_v1",
        conditioning_route="state_skill_cond",
    )
    settings = build_settings(config)
    model = json.loads(settings["models_json"])[0]

    assert settings["architecture"] == "cond_gemma"
    assert settings["conditioning_route"] == "state_skill_cond"
    assert model["architecture_revision"] == "skillvla_real_v1"
    assert model["architecture_label"] == "arch0"
    assert model["architecture_inferred"] is False
    assert model["conditioning_route"] == "state_skill_cond"
    assert model["visual_crossattn_queries"] == "not_applicable"


def test_eval_preserves_arch0_skill_training_only_alias(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        architecture="cond_gemma",
        architecture_revision="skillvla_real_v1",
        conditioning_route="state_cond",
    )
    project = Path(config["project_root"])
    policy_path = (
        project
        / "outputs/skillVLA_stage1/new-vsa/checkpoints/000100/pretrained_model/config.json"
    )
    policy = json.loads(policy_path.read_text())
    policy.update(
        {
            "architecture_label": "arch0_skill",
            "skill_flow_enabled": True,
            "skill_flow_weight": 1.0,
            "skill_flow_max_length": 171,
        }
    )
    policy_path.write_text(json.dumps(policy))

    settings = build_settings(config)
    model = json.loads(settings["models_json"])[0]

    assert model["architecture_revision"] == "skillvla_real_v1"
    assert model["architecture_label"] == "arch0_skill"
    assert model["conditioning_route"] == "state_cond"


def test_eval_accepts_implicit_skillvla_real_checkpoint(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        architecture=None,
        architecture_revision=None,
        conditioning_route="skill_cond",
    )
    settings = build_settings(config)
    model = json.loads(settings["models_json"])[0]

    assert model["architecture"] == "cond_gemma"
    assert model["architecture_revision"] == "skillvla_real_v1"
    assert model["architecture_inferred"] is True
    assert model["conditioning_route"] == "skillonly_cond"
    assert model["eval_legacy_vsa"] is False


@pytest.mark.parametrize(
    ("revision", "label", "route", "visual_tokens", "width"),
    [
        ("expert_state_adarms_v1", "arch0_1", "state_cond", 0, 0),
        ("cond_expert_state_adarms_v1", "arch0_2", "state_cond", 0, 0),
        (
            "cond_expert_separate_state_adarms_v1",
            "arch0_2_sep",
            "state_cond",
            0,
            0,
        ),
        ("wrist_cond_expert_state_adarms_v1", "arch0_3", "state_cond", 0, 0),
        ("expert_tokens_uncompressed_v1", "arch1_1", "state_skill_cond", 0, 0),
        ("expert_tokens_perceiver_v1", "arch1_2", "state_skill_cond", 32, 1024),
    ],
)
def test_eval_resolves_cond_ablation_revision(
    tmp_path: Path,
    revision: str,
    label: str,
    route: str,
    visual_tokens: int,
    width: int,
) -> None:
    settings = build_settings(
        _config(
            tmp_path,
            architecture="cond_gemma",
            architecture_revision=revision,
            conditioning_route=route,
        )
    )
    model = json.loads(settings["models_json"])[0]

    assert model["architecture_revision"] == revision
    assert model["architecture_label"] == label
    assert model["conditioning_route"] == route
    assert model["num_visual_latents_per_camera"] == visual_tokens
    assert model["visual_perceiver_width"] == width


def test_eval_accepts_mixed_cond_vsa_and_previous_models(tmp_path: Path) -> None:
    config = _config(tmp_path)
    project = Path(config["project_root"])
    source = (
        project
        / "outputs/skillVLA_stage1/new-vsa/checkpoints/000100/pretrained_model"
    )

    cond = (
        project
        / "outputs/skillVLA_stage1/old-cond/checkpoints/000100/pretrained_model"
    )
    cond.mkdir(parents=True)
    cond_policy = json.loads((source / "config.json").read_text())
    cond_policy.pop("architecture")
    cond_policy.pop("architecture_revision")
    cond_policy["conditioning_route"] = "state_skill_cond"
    (cond / "config.json").write_text(json.dumps(cond_policy))
    for name in (
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "train_config.json",
    ):
        (cond / name).write_bytes((source / name).read_bytes())

    previous_policy = json.loads((source / "config.json").read_text())
    previous_policy["architecture_revision"] = "residual_sa18_v2"
    (source / "config.json").write_text(json.dumps(previous_policy))
    previous = project / "outputs/skillVLA_stage1/previous/old-vsa"
    current = project / "outputs/skillVLA_stage1/new-vsa"
    previous.parent.mkdir(parents=True)
    current.rename(previous)
    config["models"] = [
        {
            "model_dir": "old-cond",
            "checkpoint": "000100",
            "skill_source": "gt",
            "advance_mode": "gt",
            "label": "arch0",
        },
        {
            "model_dir": "old-vsa",
            "previous": True,
            "checkpoint": "000100",
            "skill_source": "gt",
            "advance_mode": "gt",
            "label": "previous-arch2",
        },
    ]

    settings = build_settings(config)
    models = json.loads(settings["models_json"])

    assert [model["architecture"] for model in models] == [
        "cond_gemma",
        "vsa_perceiver_crossattn",
    ]
    assert [model["architecture_label"] for model in models] == ["arch0", "arch2_2"]
    assert models[0]["architecture_inferred"] is True
    assert models[1]["previous_checkpoint"] is True
    assert settings["model_architectures"] == (
        "arch0=arch0, previous-arch2=arch2_2[previous]"
    )


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


def test_eval_accepts_external_image_only_terminator(tmp_path: Path) -> None:
    config = _config(tmp_path)
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

    external = project / "external/image-aux/checkpoints/030000/pretrained_model"
    external.mkdir(parents=True)
    (external / "config.json").write_text(
        json.dumps(
            {
                "type": "skill_aux",
                "train_terminator": False,
                "train_image_only_terminator": True,
                "skill_fsq_levels": [3, 3, 3],
            }
        )
    )
    (external / "model.safetensors").touch()

    config["external_skill_model"] = str(external)
    config["models"][0]["advance_mode"] = "external"
    config["models"][0]["terminator_variant"] = "image_only"
    model = json.loads(build_settings(config)["models_json"])[0]

    assert model["advance_mode"] == "external"
    assert model["terminator_variant"] == "image_only"


def test_eval_accepts_original_fsq_terminator_without_trained_terminator(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
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
    assert target_policy["train_terminator"] is False
    (target / "config.json").write_text(json.dumps(target_policy))

    config["models"][0]["advance_mode"] = "original"
    model = json.loads(build_settings(config)["models_json"])[0]

    assert model["advance_mode"] == "original"
    assert Path(model["fsq_path"]) == fsq


def test_eval_original_terminator_requires_fsq_checkpoint(tmp_path: Path) -> None:
    config = _config(tmp_path)
    project = Path(config["project_root"])
    target = (
        project
        / config["outputs_root"]
        / "skillVLA_stage1/new-vsa/checkpoints/000100/pretrained_model"
    )
    target_policy = json.loads((target / "config.json").read_text())
    target_policy["fsq_path"] = str(project / "missing/FSQ.pt")
    (target / "config.json").write_text(json.dumps(target_policy))
    config["models"][0]["advance_mode"] = "original"

    with pytest.raises(FileNotFoundError, match="advance_mode=original"):
        build_settings(config)


@pytest.mark.parametrize("advance_mode", ["own", "original"])
def test_eval_rejects_image_only_variant_for_non_external_terminator(
    tmp_path: Path, advance_mode: str
) -> None:
    config = _config(tmp_path)
    config["models"][0]["advance_mode"] = advance_mode
    config["models"][0]["terminator_variant"] = "image_only"

    with pytest.raises(ValueError, match="requires advance_mode=external"):
        build_settings(config)


def test_eval_rejects_legacy_stage1_checkpoint(tmp_path: Path) -> None:
    config = _config(tmp_path, architecture="state_skill_cond")

    with pytest.raises(ValueError, match="Unsupported Stage-1 checkpoint architecture"):
        build_settings(config)


def test_eval_rejects_replanning_beyond_chunk(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["n_action_steps"] = 11

    with pytest.raises(ValueError, match="exceeds"):
        build_settings(config)


def test_eval_exports_immediate_skill_end_replanning(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["terminator"] = {"immediate_replan_on_skill_end": True}

    settings = build_settings(config)

    assert settings["immediate_replan_on_skill_end"] is True


def test_predictor_and_terminator_externals_can_name_different_checkpoints() -> None:
    """skill_source and advance_mode may overlay from separate checkpoints."""
    entries = _model_entries(
        {
            "external_skill_model": "outputs/top/ckpt",
            "model_defaults": {
                "checkpoint": "100000",
                "skill_source": "external",
                "advance_mode": "external",
            },
            "models": [
                # Inherits the shared top-level value for both roles.
                {"model_dir": "shared", "label": "shared"},
                # Splits the two roles apart.
                {
                    "model_dir": "split",
                    "label": "split",
                    "external_predictor_model": "outputs/pred/ckpt",
                    "external_terminator_model": "outputs/term/ckpt",
                },
                # A per-entry external_skill_model still covers both roles.
                {
                    "model_dir": "both",
                    "label": "both",
                    "external_skill_model": "outputs/both/ckpt",
                },
                # One role overridden, the other left on the entry-wide value.
                {
                    "model_dir": "mixed",
                    "label": "mixed",
                    "external_skill_model": "outputs/both/ckpt",
                    "external_terminator_model": "outputs/term/ckpt",
                },
            ],
        }
    )

    roles = {
        entry["label"]: (
            entry["external_predictor_model_value"],
            entry["external_terminator_model_value"],
        )
        for entry in entries
    }

    assert roles == {
        "shared": ("outputs/top/ckpt", "outputs/top/ckpt"),
        "split": ("outputs/pred/ckpt", "outputs/term/ckpt"),
        "both": ("outputs/both/ckpt", "outputs/both/ckpt"),
        "mixed": ("outputs/both/ckpt", "outputs/term/ckpt"),
    }


def test_role_specific_model_fields_infer_their_modes() -> None:
    entries = _model_entries(
        {
            "model_defaults": {
                "checkpoint": "100000",
                "skill_source": "gt",
                "advance_mode": "gt",
            },
            "models": [
                {
                    "model_dir": "predictor",
                    "external_predictor_model": "predictor-run",
                },
                {
                    "model_dir": "terminator",
                    "external_terminator_model": "terminator-run",
                },
                {
                    "model_dir": "original",
                    "external_terminator_model": "original",
                },
                {
                    "model_dir": "explicit-gt",
                    "skill_source": "gt",
                    "external_predictor_model": "unused-predictor",
                },
            ],
        }
    )
    by_model = {entry["model_dir"]: entry for entry in entries}

    assert by_model["predictor"]["skill_source"] == "external"
    assert by_model["terminator"]["advance_mode"] == "external"
    assert by_model["original"]["advance_mode"] == "original"
    assert by_model["original"]["external_terminator_model_value"] == ""
    assert by_model["explicit-gt"]["skill_source"] == "gt"


def test_original_terminator_selector_rejects_a_conflicting_mode() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        _model_entries(
            {
                "models": [
                    {
                        "model_dir": "model",
                        "advance_mode": "external",
                        "external_terminator_model": "original",
                    }
                ]
            }
        )


def test_external_terminator_run_name_and_checkpoint_expand_under_outputs_root(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    project = Path(config["project_root"])
    checkpoint = (
        project
        / "outputs/skillVLA_terminator/aux-run/checkpoints/004000/pretrained_model"
    )
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "skill_aux",
                "train_terminator": True,
                "skill_fsq_levels": [3, 3, 3],
            }
        )
    )
    (checkpoint / "model.safetensors").touch()
    config["models"][0].update(
        {
            "advance_mode": "external",
            "external_terminator_model": "aux-run",
            "external_terminator_checkpoint": "004000",
        }
    )

    model = json.loads(build_settings(config)["models_json"])[0]

    assert Path(model["external_terminator_model"]) == checkpoint
    assert model["external_terminator_checkpoint"] == "004000"


def test_external_predictor_run_name_and_checkpoint_expand_under_outputs_root(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    project = Path(config["project_root"])
    checkpoint = (
        project
        / "outputs/skillVLA_terminator/predictor-run/checkpoints/005000/pretrained_model"
    )
    checkpoint.mkdir(parents=True)
    tokenizer = project / "models/tokenizer"
    tokenizer.mkdir(parents=True)
    target = (
        project
        / "outputs/skillVLA_stage1/new-vsa/checkpoints/000100/pretrained_model"
    )
    target_policy = json.loads((target / "config.json").read_text())
    fsq = project / "dataset/skillvla_dataset/source/run/FSQ.pt"
    fsq.touch()
    target_policy["fsq_path"] = str(fsq)
    (target / "config.json").write_text(json.dumps(target_policy))
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                **target_policy,
                "type": "skill_aux",
                "train_skill_predictor": True,
                "tokenizer_path": str(tokenizer),
            }
        )
    )
    (checkpoint / "model.safetensors").touch()
    config["models"][0].update(
        {
            "skill_source": "external",
            "external_predictor_model": "predictor-run",
            "external_predictor_checkpoint": "005000",
        }
    )

    model = json.loads(build_settings(config)["models_json"])[0]

    assert Path(model["external_predictor_model"]) == checkpoint
    assert model["external_predictor_checkpoint"] == "005000"


@pytest.mark.parametrize("target_has_predictor", [False, True])
def test_predictor_overlay_contract_depends_on_whether_target_owns_a_predictor(
    tmp_path: Path, target_has_predictor: bool
) -> None:
    """A predictor-free target rebuilds the module from the source checkpoint.

    Its own skill_predictor_* values are unused defaults in that case, so only the
    skill geometry must agree. A target that trained a predictor keeps the strict
    module contract, because the overlay loads into its existing module.
    """
    config = _config(tmp_path)
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
    target_policy["train_skill_predictor"] = target_has_predictor
    # The target's own module shape: LoRA-free.
    target_policy["skill_predictor_lora"] = False
    target_policy["skill_predictor_all_layers"] = False
    tokenizer = project / "models/tokenizer"
    tokenizer.mkdir(parents=True)
    if target_has_predictor:
        # A target owning a predictor must carry its own tokenizer.
        target_policy["tokenizer_path"] = str(tokenizer)
    (target / "config.json").write_text(json.dumps(target_policy))

    external = project / "external/predictor/checkpoints/030000/pretrained_model"
    external.mkdir(parents=True)
    source_policy = {
        **target_policy,
        "train_skill_predictor": True,
        "tokenizer_path": str(tokenizer),
        # Differs from the target only in module shape, not skill geometry.
        "skill_predictor_lora": True,
        "skill_predictor_all_layers": True,
    }
    (external / "config.json").write_text(json.dumps(source_policy))
    (external / "model.safetensors").touch()

    config["external_skill_model"] = str(external)
    config["models"][0]["skill_source"] = "external"

    if target_has_predictor:
        with pytest.raises(ValueError, match="module contract mismatch"):
            build_settings(config)
        return

    model = json.loads(build_settings(config)["models_json"])[0]
    assert Path(model["external_predictor_model"]) == external

    # Skill geometry is still enforced on the relaxed path.
    source_policy["skill_fsq_levels"] = [3, 4, 5]
    (external / "config.json").write_text(json.dumps(source_policy))
    with pytest.raises(ValueError, match="skill geometry mismatch"):
        build_settings(config)
