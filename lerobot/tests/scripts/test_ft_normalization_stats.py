"""The FT probe must preserve both input scale and physical action decoding."""
import ast
import copy
import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "lerobot/src"))
from lerobot.utils.ft_normalization import (
    checkpoint_normalization_stats, normalization_stats_mode,
    validate_resume_normalization, with_checkpoint_normalization,
)


def checkpoint(path, kind="standard", shift=0):
    path.mkdir(parents=True)
    stats = {feature: {stat: np.full(dim, shift + value, dtype=np.float32)
                      for stat, value in (("q01", -2), ("q99", 4), ("mean", 1), ("std", 2))}
             for feature, dim in (("observation.state", 8), ("action", 7))}
    for file, registry, subset in (
        ("policy_preprocessor", "molmoact2_masked_normalizer" if kind == "molmo" else "normalizer_processor", stats),
        ("policy_postprocessor", "molmoact2_masked_unnormalizer" if kind == "molmo" else "unnormalizer_processor", {"action": stats["action"]}),
    ):
        tensor_file = file + ".safetensors"
        save_file({f"{feature}.{stat}": value for feature, block in subset.items() for stat, value in block.items()}, str(path / tensor_file))
        (path / (file + ".json")).write_text(json.dumps({"steps": [{"registry_name": registry, "state_file": tensor_file}]}))
    return stats


@pytest.mark.parametrize("kind", ["standard", "molmo"])
def test_pre_and_post_share_exact_checkpoint_stats_and_leave_dataset_unchanged(tmp_path, kind):
    saved = checkpoint(tmp_path / kind, kind)
    target = {key: {stat: value + 20 for stat, value in block.items()} for key, block in saved.items()}
    before = copy.deepcopy(target)
    chosen = with_checkpoint_normalization(target, tmp_path / kind)
    for feature, block in saved.items():
        for stat, value in block.items():
            np.testing.assert_array_equal(chosen[feature][stat], value)
            np.testing.assert_array_equal(target[feature][stat], before[feature][stat])
    action = np.full((1, 10, 7), .3)
    low, high = chosen["action"]["q01"], chosen["action"]["q99"]
    normalized = 2 * (action - low) / (high - low) - 1
    np.testing.assert_allclose((normalized + 1) * .5 * (high - low) + low, action)


def test_groot_groups_use_explicit_modality_order_not_json_order(tmp_path):
    path = tmp_path / "groot";path.mkdir()
    names = {"state": ["x", "y", "z", "roll", "pitch", "yaw", "left_finger", "right_finger"],
             "action": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]}
    raw = {group: {name: {"q01": [float(i)], "q99": [float(i + 10)]}
                   for i, name in reversed(list(enumerate(order)))} for group, order in names.items()}
    for file, registry in [("policy_preprocessor", "groot_n1_7_pack_inputs_v1"), ("policy_postprocessor", "groot_n1_7_action_decode_v1")]:
        (path / (file + ".json")).write_text(json.dumps({"steps": [{"registry_name": registry, "config": {"raw_stats": raw}}]}))
    stats = checkpoint_normalization_stats(path)
    np.testing.assert_array_equal(stats["observation.state"]["q01"], np.arange(8))
    np.testing.assert_array_equal(stats["action"]["q01"], np.arange(7))
    raw["action"]["x"]["q99"] = [999]
    (path / "policy_postprocessor.json").write_text(json.dumps({"steps": [{"registry_name": "groot_n1_7_action_decode_v1", "config": {"raw_stats": raw}}]}))
    with pytest.raises(ValueError, match="disagree"):
        checkpoint_normalization_stats(path)


def test_missing_or_inconsistent_stats_never_fall_back_to_ft(tmp_path):
    path = tmp_path / "saved"; checkpoint(path)
    data = {"action.q01": np.full(7, -200., dtype=np.float32), "action.q99": np.full(7, 4., dtype=np.float32)}
    save_file(data, str(path / "policy_postprocessor.safetensors"))
    with pytest.raises(ValueError, match="disagree"):
        checkpoint_normalization_stats(path)
    (path / "policy_postprocessor.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        checkpoint_normalization_stats(path)


@pytest.mark.parametrize("value", [None, True, "PT", "dataset", "unknown", 0])
def test_invalid_choice_rejected(value):
    with pytest.raises(ValueError, match="normalization_stats"):
        normalization_stats_mode(value)


def test_resume_keeps_saved_choice_and_accepts_legacy_ft(tmp_path):
    metadata = tmp_path / "pretrained_model/sbd_molmoact2.json"
    metadata.parent.mkdir();metadata.write_text("{}")
    validate_resume_normalization(tmp_path, "ft", metadata.name)
    with pytest.raises(ValueError, match="Cannot change"):
        validate_resume_normalization(tmp_path, "pt", metadata.name)
    metadata.write_text(json.dumps({"normalization_stats": "pt"}))
    validate_resume_normalization(tmp_path, "pt", metadata.name)


def pi05_helper():
    script = ROOT / "lerobot/src/lerobot/scripts/lerobot_train.py"
    node = next(n for n in ast.parse(script.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_pi05_processor_dataset_stats")
    scope = {"copy": copy, "json": json, "logging": logging, "Path": Path}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(script), "exec"), scope)
    return scope[node.name]


@pytest.mark.parametrize("mode,resume", [("ft", False), ("pt", False), ("ft", True), ("pt", True)])
def test_pi05_stats_probe_and_resume_read_checkpoint_not_new_target(tmp_path, mode, resume):
    path = tmp_path / "saved"; saved = checkpoint(path)
    (path / "train_config.json").write_text(json.dumps({"normalization_stats": mode}))
    target = {"action": {"q01": np.full(7, -100.), "q99": np.full(7, 100.)}}
    cfg = SimpleNamespace(policy=SimpleNamespace(type="pi05", pretrained_path=path), normalization_stats=mode, resume=resume)
    chosen = pi05_helper()(cfg, target, is_main_process=False)
    np.testing.assert_array_equal(chosen["action"]["q01"], saved["action"]["q01"] if mode == "pt" or resume else target["action"]["q01"])


def test_pi05_cannot_switch_stats_while_resuming(tmp_path):
    path = tmp_path / "saved";checkpoint(path)
    (path / "train_config.json").write_text(json.dumps({"normalization_stats": "pt"}))
    cfg = SimpleNamespace(policy=SimpleNamespace(type="pi05", pretrained_path=path), normalization_stats="ft", resume=True)
    with pytest.raises(ValueError, match="Cannot change"):
        pi05_helper()(cfg, {})


@pytest.mark.parametrize("mode", ["ft", "pt"])
def test_pi05_selected_stats_drive_real_state_and_action_processors(tmp_path, mode):
    import torch
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.processor import NormalizerProcessorStep, TransitionKey, UnnormalizerProcessorStep
    from lerobot.processor.converters import create_transition

    path = tmp_path / "pt"
    saved = checkpoint(path)
    target = {feature: {stat: value * 2 for stat, value in block.items()}
              for feature, block in saved.items()}
    cfg = SimpleNamespace(policy=SimpleNamespace(type="pi05", pretrained_path=path),
                          normalization_stats=mode, resume=False)
    chosen = pi05_helper()(cfg, target, is_main_process=False)
    features = {"observation.state": PolicyFeature(FeatureType.STATE, (8,)),
                "action": PolicyFeature(FeatureType.ACTION, (7,))}
    norm_map = {FeatureType.STATE: NormalizationMode.QUANTILES,
                FeatureType.ACTION: NormalizationMode.QUANTILES}
    encoder = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=chosen)
    decoder = UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats=chosen)
    state, action = torch.full((1, 8), .3), torch.full((1, 10, 7), .3)
    encoded = encoder(create_transition(observation={"observation.state": state}, action=action))
    expected = -.2333333333 if mode == "pt" else -.2833333333
    torch.testing.assert_close(encoded[TransitionKey.ACTION], torch.full_like(action, expected))
    torch.testing.assert_close(encoded[TransitionKey.OBSERVATION]["observation.state"],
                               torch.full_like(state, expected))
    torch.testing.assert_close(decoder(encoded)[TransitionKey.ACTION], action, atol=1e-6, rtol=1e-6)

