import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.optim.schedulers import (  # noqa: E402
    CosineDecayWithWarmupSchedulerConfig,
    WarmupConstantSchedulerConfig,
)
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: E402
from lerobot.policies.pi05.processor_pi05 import (  # noqa: E402
    make_pi05_pre_post_processors,
    with_pi05_eef_relative_action_stats,
)
from lerobot.processor import NormalizerProcessorStep  # noqa: E402
from lerobot.processor.eef_relative_action_processor import (  # noqa: E402
    EefRelativeActionsProcessorStep,
    EefRelativeToOscActionsProcessorStep,
    osc_actions_to_absolute_eef,
)
from lerobot.processor.converters import create_transition  # noqa: E402
from lerobot.types import TransitionKey  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402


def _write_stats(path: Path, *, chunk_size: int = 3) -> None:
    payload = {
        "representation": "eef_anchor_relative_so3",
        "storage_representation": "absolute_eef_command",
        "rotation_representation": "axis_angle_rotation_vector",
        "rotation_composition": "left_world",
        "chunk_size": chunk_size,
        "osc_position_scale": 0.05,
        "osc_rotation_scale": 0.5,
        "action": {
            "min": [-2.0] * 7,
            "max": [2.0] * 7,
            "mean": [0.0] * 7,
            "std": [1.0] * 7,
            "q01": [-1.0] * 7,
            "q99": [1.0] * 7,
        },
    }
    path.write_text(json.dumps(payload))


def _config(tmp_path: Path) -> PI05Config:
    stats_path = tmp_path / "relative_action_stats.json"
    _write_stats(stats_path)
    return PI05Config(
        chunk_size=3,
        n_action_steps=1,
        use_eef_relative_actions=True,
        eef_relative_stats_path=str(stats_path),
        eef_position_scale=0.05,
        eef_rotation_scale=0.5,
        tokenizer_path="unused-local-tokenizer",
        device="cpu",
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        },
    )


def _dataset_stats() -> dict[str, dict[str, torch.Tensor]]:
    return {
        OBS_STATE: {
            "min": torch.full((8,), -10.0),
            "max": torch.full((8,), 10.0),
            "q01": torch.full((8,), -10.0),
            "q99": torch.full((8,), 10.0),
        },
        ACTION: {
            "min": torch.full((7,), -100.0),
            "max": torch.full((7,), 100.0),
            "q01": torch.full((7,), -100.0),
            "q99": torch.full((7,), 100.0),
        },
    }


def test_pi05_eef_relative_processor_uses_relative_stats_and_reloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: object(),
    )
    config = _config(tmp_path)
    preprocessor, postprocessor = make_pi05_pre_post_processors(config, _dataset_stats())

    # Fresh PT has a pretrained weight path, but its ordinary base processor
    # must not replace the newly constructed EEF-relative processor.
    fresh_preprocessor, fresh_postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(tmp_path / "ordinary_pi05_base_without_processors"),
        dataset_stats=_dataset_stats(),
    )
    assert any(
        isinstance(step, EefRelativeActionsProcessorStep)
        for step in fresh_preprocessor.steps
    )
    assert any(
        isinstance(step, EefRelativeToOscActionsProcessorStep)
        for step in fresh_postprocessor.steps
    )

    relative_step = next(
        step for step in preprocessor.steps if isinstance(step, EefRelativeActionsProcessorStep)
    )
    osc_step = next(
        step for step in postprocessor.steps if isinstance(step, EefRelativeToOscActionsProcessorStep)
    )
    normalizer = next(
        step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep)
    )
    assert osc_step.relative_step is relative_step
    assert torch.allclose(normalizer._tensor_stats[ACTION]["q99"], torch.ones(7))

    state = torch.tensor([[0.3, 0.05, 1.0, 0.2, -0.15, 2.9, 0.02, -0.02]])
    osc = torch.tensor([[[0.4, -0.5, 0.2, 0.2, -0.3, 0.1, -1.0]]])
    absolute = osc_actions_to_absolute_eef(osc, state)
    transition = create_transition({OBS_STATE: state}, absolute)
    relative = relative_step(transition)
    restored = osc_step(relative)
    assert torch.allclose(restored[TransitionKey.ACTION], osc, atol=2e-5, rtol=2e-5)

    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)
    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(tmp_path),
    )
    loaded_relative = next(
        step
        for step in loaded_preprocessor.steps
        if isinstance(step, EefRelativeActionsProcessorStep)
    )
    loaded_osc = next(
        step
        for step in loaded_postprocessor.steps
        if isinstance(step, EefRelativeToOscActionsProcessorStep)
    )
    assert loaded_osc.relative_step is loaded_relative


def test_pi05_eef_relative_config_requires_single_action_execution():
    with pytest.raises(ValueError, match="n_action_steps=1"):
        PI05Config(use_eef_relative_actions=True, chunk_size=3, n_action_steps=2)


def test_pi05_default_processor_does_not_enable_eef_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: object(),
    )
    config = _config(tmp_path)
    config.use_eef_relative_actions = False
    config.eef_relative_stats_path = None
    preprocessor, postprocessor = make_pi05_pre_post_processors(config, _dataset_stats())
    assert not any(
        isinstance(step, EefRelativeActionsProcessorStep) for step in preprocessor.steps
    )
    assert not any(
        isinstance(step, EefRelativeToOscActionsProcessorStep)
        for step in postprocessor.steps
    )


def test_pi05_eef_relative_stats_must_cover_prediction_chunk(tmp_path: Path):
    config = _config(tmp_path)
    _write_stats(Path(config.eef_relative_stats_path), chunk_size=2)
    with pytest.raises(ValueError, match="chunk_size=2 < policy chunk_size=3"):
        with_pi05_eef_relative_action_stats(config, _dataset_stats())


def test_pi05_scheduler_supports_warmup_constant_and_cosine():
    constant = PI05Config(
        scheduler_mode="warmup_constant",
        scheduler_warmup_steps=123,
    ).get_scheduler_preset()
    assert isinstance(constant, WarmupConstantSchedulerConfig)
    assert constant.num_warmup_steps == 123

    cosine = PI05Config(scheduler_mode="cosine_decay").get_scheduler_preset()
    assert isinstance(cosine, CosineDecayWithWarmupSchedulerConfig)

    with pytest.raises(ValueError, match="scheduler_mode"):
        PI05Config(scheduler_mode="constant")
