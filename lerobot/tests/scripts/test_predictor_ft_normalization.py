"""Predictor FT preserves PT state-token semantics; resume preserves its own scale."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.skill_aux.configuration_skill_aux import SkillAuxConfig
from lerobot.policies.skill_expert.processor_skill_expert import (
    load_skill_expert_normalization_stats,
    make_skill_expert_pre_post_processors,
)
from lerobot.policies.skillVLA.dataset_skillVLA import SKILL_END_STATE, SKILL_START_STATE
from lerobot.policies.skillVLA.processor_skillVLA import SkillVLAPrepareStateTokenizerProcessorStep
from lerobot.processor.tokenizer_processor import AutoTokenizer
from lerobot.scripts.lerobot_train import _predictor_processor_dataset_stats
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_STATE


def _stats(low: float, high: float) -> dict:
    return {
        feature: {"q01": torch.full((width,), low), "q99": torch.full((width,), high)}
        for feature, width in ((OBS_STATE, 8), (ACTION, 2))
    }


def _checkpoint(path: Path, stats: dict, *, registry="normalizer_processor") -> Path:
    path.mkdir(parents=True)
    save_file(
        {f"{feature}.{name}": value for feature, values in stats.items() for name, value in values.items()},
        path / "normalizer.safetensors",
    )
    (path / "policy_preprocessor.json").write_text(
        json.dumps({"steps": [{"registry_name": registry, "state_file": "normalizer.safetensors"}]})
    )
    return path


def _config(source, *, resume=False, checkpoint=None, **overrides):
    policy = SimpleNamespace(
        type="skill_aux",
        train_skill_predictor=True,
        skill_predictor_checkpoint_path=str(source) if source else "",
        pretrained_path=checkpoint,
    )
    for key, value in overrides.items():
        setattr(policy, key, value)
    return SimpleNamespace(policy=policy, resume=resume)


@pytest.mark.parametrize("freeze_vlm", [True, False])
@pytest.mark.parametrize("registry", ["normalizer_processor", "skill_expert_normalizer_processor_step"])
def test_new_ft_inherits_checkpoint_stats_and_does_not_mutate_target(tmp_path, freeze_vlm, registry):
    pt = _checkpoint(tmp_path / "pt", _stats(0.0, 10.0), registry=registry)
    target = _stats(100.0, 200.0)
    target["new_dataset_only"] = {"mean": torch.tensor([3.0])}
    original = copy.deepcopy(target)
    actual = _predictor_processor_dataset_stats(_config(pt, skill_predictor_freeze_vlm=freeze_vlm), target)
    for feature in (OBS_STATE, ACTION):
        for name in ("q01", "q99"):
            torch.testing.assert_close(actual[feature][name], _stats(0.0, 10.0)[feature][name])
            torch.testing.assert_close(target[feature][name], original[feature][name])
    torch.testing.assert_close(actual["new_dataset_only"]["mean"], torch.tensor([3.0]))
    assert actual is not target


def test_resume_uses_saved_ft_stats_even_if_original_pt_is_unavailable(tmp_path):
    resumed = _checkpoint(tmp_path / "ft", _stats(-3.0, 4.0))
    actual = _predictor_processor_dataset_stats(
        _config(tmp_path / "unavailable_pt", resume=True, checkpoint=resumed), _stats(100.0, 200.0)
    )
    torch.testing.assert_close(actual[OBS_STATE]["q01"], torch.full((8,), -3.0))
    torch.testing.assert_close(actual[OBS_STATE]["q99"], torch.full((8,), 4.0))


def test_resume_requires_saved_checkpoint_instead_of_changing_scale(tmp_path):
    with pytest.raises(ValueError, match="saved pretrained_path"):
        _predictor_processor_dataset_stats(_config(tmp_path / "pt", resume=True), _stats(0.0, 1.0))


@pytest.mark.parametrize(
    "overrides",
    [{"train_skill_predictor": False}, {"type": "skill_expert"}, {"skill_predictor_checkpoint_path": ""}],
)
def test_scratch_pt_terminator_and_vsa_keep_existing_behavior(tmp_path, overrides):
    target = _stats(100.0, 200.0)
    assert _predictor_processor_dataset_stats(_config(tmp_path / "missing", **overrides), target) is target


def test_missing_pt_statistics_fail_instead_of_using_target_stats(tmp_path):
    with pytest.raises(FileNotFoundError, match="preprocessor config not found"):
        _predictor_processor_dataset_stats(_config(tmp_path / "missing"), _stats(100.0, 200.0))


def test_legacy_warm_start_matches_the_predictor_weight_source(tmp_path):
    pt = _checkpoint(tmp_path / "pt", _stats(0.0, 10.0))
    actual = _predictor_processor_dataset_stats(
        _config(None, auxiliary_checkpoint_path=str(pt)), _stats(100.0, 200.0)
    )
    torch.testing.assert_close(actual[OBS_STATE]["q99"], torch.full((8,), 10.0))


def test_ft_processors_match_pt_prompts_save_stats_and_resume_unchanged(tmp_path, monkeypatch):
    class Tokenizer:
        def __call__(self, texts, max_length, **kwargs):
            ids = torch.zeros((len(texts), max_length), dtype=torch.long)
            mask = torch.zeros_like(ids)
            for index, text in enumerate(texts):
                values = [ord(c) for c in text[:max_length]]
                ids[index, : len(values)] = torch.tensor(values)
                mask[index, : len(values)] = 1
            return {"input_ids": ids, "attention_mask": mask}

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    policy = SkillAuxConfig(
        device="cpu",
        dtype="float32",
        train_terminator=False,
        train_skill_predictor=True,
        max_state_dim=8,
        max_action_dim=2,
        tokenizer_path="test-tokenizer-no-download",
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
    )
    pt_stats = _stats(0.0, 10.0)
    pt_pre, _ = make_skill_expert_pre_post_processors(policy, dataset_stats=pt_stats)
    pt = tmp_path / "pt"
    pt_pre.save_pretrained(pt)
    inherited = _predictor_processor_dataset_stats(_config(pt), _stats(100.0, 200.0))
    ft_pre, ft_post = make_skill_expert_pre_post_processors(policy, dataset_stats=inherited)
    batch = {
        OBS_STATE: torch.full((1, 8), 5.0),
        SKILL_START_STATE: torch.full((1, 8), 5.0),
        SKILL_END_STATE: torch.full((1, 8), 9.0),
        "task": ["pick object"],
    }
    expected, actual = pt_pre(copy.deepcopy(batch)), ft_pre(copy.deepcopy(batch))
    torch.testing.assert_close(actual[OBS_STATE], torch.zeros((1, 8)))
    torch.testing.assert_close(actual[OBS_LANGUAGE_TOKENS], expected[OBS_LANGUAGE_TOKENS])
    # Spatial targets remain physical/grounded values, not quantile-normalized XYZ.
    torch.testing.assert_close(actual[SKILL_END_STATE], batch[SKILL_END_STATE])
    ft = tmp_path / "ft"
    ft_pre.save_pretrained(ft)
    ft_post.save_pretrained(ft)
    saved_stats = load_skill_expert_normalization_stats(ft)
    for name in ("q01", "q99"):
        torch.testing.assert_close(saved_stats[OBS_STATE][name], pt_stats[OBS_STATE][name])
    tokenizer = next(s for s in ft_pre.steps if isinstance(s, SkillVLAPrepareStateTokenizerProcessorStep))
    torch.testing.assert_close(torch.as_tensor(tokenizer.state_q01), pt_stats[OBS_STATE]["q01"])
    torch.testing.assert_close(torch.as_tensor(tokenizer.state_q99), pt_stats[OBS_STATE]["q99"])
    resumed_stats = _predictor_processor_dataset_stats(
        _config(tmp_path / "missing_pt", resume=True, checkpoint=ft), _stats(300.0, 400.0)
    )
    from lerobot.policies.factory import make_pre_post_processors

    # This is the resume branch used by lerobot_train: load the saved pipeline,
    # with its own normalizer stats, and no new dataset_stats/tokenizer bounds.
    resumed_pre, _ = make_pre_post_processors(
        policy,
        pretrained_path=ft,
        preprocessor_overrides={"normalizer_processor": {"stats": resumed_stats}},
    )
    resumed = resumed_pre(copy.deepcopy(batch))
    torch.testing.assert_close(resumed[OBS_LANGUAGE_TOKENS], actual[OBS_LANGUAGE_TOKENS])
    torch.testing.assert_close(resumed[OBS_STATE], actual[OBS_STATE])
    # Eval has only the current boundary observation, not skill_start_state.
    evaluation = resumed_pre({OBS_STATE: batch[OBS_STATE].clone(), "task": ["pick object"]})
    torch.testing.assert_close(evaluation[OBS_LANGUAGE_TOKENS], actual[OBS_LANGUAGE_TOKENS])
    torch.testing.assert_close(evaluation[OBS_STATE], actual[OBS_STATE])


def test_current_ft_yaml_uses_fresh_suffix_to_avoid_auto_resuming_old_stats():
    from pathlib import Path

    import yaml

    path = (
        Path(__file__).resolve().parents[2]
        / "examples/libero/configs/train_skillVLA/FT/Predictor/predictor_ft_config.yaml"
    )
    assert yaml.safe_load(path.read_text())["run"]["suffix"] == "base_ptstats"
