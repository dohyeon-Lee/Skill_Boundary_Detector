"""Run separately in each pinned baseline venv, with SBD_BASELINE_FAMILY set."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

FAMILY = os.environ.get("SBD_BASELINE_FAMILY")
if FAMILY not in ("molmoact2", "groot"):
    pytest.skip("Select a pinned baseline environment with SBD_BASELINE_FAMILY.", allow_module_level=True)
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / f"lerobot/examples/libero/configs/train_{FAMILY}/PT/src"
sys.path.insert(0, str(SRC))
config_module = __import__(f"{FAMILY}_config")
spec = importlib.util.spec_from_file_location("stats_probe_training_hooks", SRC / "training_adapter.py")
hooks = importlib.util.module_from_spec(spec);sys.modules[spec.name] = hooks;spec.loader.exec_module(hooks)


def stats(dim, width, state=False):
    low, high = torch.full((dim,), -width), torch.full((dim,), width)
    if state:
        low[:6], high[:6] = width, width + 1
        low[-2:], high[-2:] = 0., 1.
    else:
        low[-1], high[-1] = -1., 1.
    return {"q01": low, "q99": high, "min": low.clone(), "max": high.clone(),
            "mean": (low + high) / 2, "std": torch.ones(dim), "count": torch.tensor([10])}


@pytest.mark.parametrize("mode", ["ft", "pt"])
def test_native_ft_processor_choice_save_reload_and_resume(tmp_path, monkeypatch, mode):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.lerobot_types import TransitionKey as K
    from lerobot.policies.factory import make_pre_post_processors
    if FAMILY == "groot":
        from lerobot.policies.groot.configuration_groot import GrootConfig as PolicyConfig
        from lerobot.policies.groot import processor_groot as pg
        monkeypatch.setattr(pg, "_load_n1_7_checkpoint_processor_assets", lambda _: None)
        cfg = PolicyConfig(device="cpu", embodiment_tag="libero_sim", chunk_size=10, n_action_steps=10,
                           base_model_path=str(tmp_path / "base"), action_decode_transform="none",
                           output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))})
        from processor_adapter import make_dataset_processors
        create = lambda saved: make_dataset_processors(cfg, saved, SimpleNamespace(), horizon=10)
    else:
        from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config as PolicyConfig
        from lerobot.policies.molmoact2 import processor_molmoact2 as pg
        # Tokenizer assets are unrelated to the real normalization pipelines under test.
        monkeypatch.setattr(pg.MolmoAct2PackInputsProcessorStep, "__post_init__", lambda self: None)
        cfg = PolicyConfig(device="cpu", checkpoint_path=str(tmp_path / "base"), chunk_size=10, n_action_steps=10,
                           normalize_gripper=False, input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,))},
                           output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))})
        cfg.dataset_feature_names = {
            "observation.state": ["x", "y", "z", "rx", "ry", "rz", "left_finger", "right_finger"],
            "action": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
        }
        create = lambda saved: make_pre_post_processors(policy_cfg=cfg, dataset_stats=saved)
    cfg._runtime_dataset_meta = SimpleNamespace()
    pt_stats = {"observation.state": stats(8, 0., state=True), "action": stats(7, 1.)}
    ft_stats = {"observation.state": stats(8, 2., state=True), "action": stats(7, 2.)}
    settings = config_module.build_settings({
        "stage": "ft", "project_root": str(tmp_path),
        "warm_start": {"run_name": "our_pt", "checkpoint": 100000},
        "training": {"normalization_stats": mode},
    })
    source = Path(settings["initial_checkpoint"]) / "pretrained_model"
    source.mkdir(parents=True)
    pt_pre, pt_post = create(pt_stats)
    pt_pre.save_pretrained(source);pt_post.save_pretrained(source)
    monkeypatch.setattr(config_module, "latest_checkpoint", lambda _: None)
    noop = lambda *a, **k: None
    native = SimpleNamespace(make_train_eval_datasets=noop, make_accelerator=noop, update_last_checkpoint=noop,
                             make_pre_post_processors=make_pre_post_processors, make_policy=noop, WandBLogger=object)
    original_preset = PolicyConfig.get_scheduler_preset
    try:
        hooks.install(native, settings, noop)
        pre, post = native.make_pre_post_processors(policy_cfg=cfg, pretrained_path=source, dataset_stats=ft_stats)
        action = torch.full((1, 10, 7), .1);action[..., -1] = torch.tensor([-1., 1.] * 5)
        def encode(pipeline):
            transition = {K.OBSERVATION: {"observation.state": torch.full((1, 8), .5),
                                          "observation.images.image": torch.zeros(1, 3, 16, 16),
                                          "observation.images.wrist_image": torch.ones(1, 3, 16, 16)},
                          K.ACTION: action.clone(), K.COMPLEMENTARY_DATA: {"task": ["test"]}}
            if FAMILY == "groot":
                step = next(s for s in pipeline.steps if isinstance(s, pg.GrootN17PackInputsStep))
            else:
                step = next(s for s in pipeline.steps if isinstance(s, pg.MolmoAct2MaskedNormalizerProcessorStep))
            return step(transition)[K.ACTION]
        encoded = encode(pre)
        expected = .1 if mode == "pt" else .05
        torch.testing.assert_close(encoded[..., :10, :6], torch.full_like(encoded[..., :10, :6], expected))
        decoded = post(encoded)
        torch.testing.assert_close(decoded, action, atol=1e-6, rtol=1e-6)
        saved = tmp_path / "own_ft/pretrained_model";saved.mkdir(parents=True)
        pre.save_pretrained(saved);post.save_pretrained(saved)
        monkeypatch.setattr(config_module, "latest_checkpoint", lambda _: saved.parent)
        # Resuming must not reread the original PT statistics.
        for p in source.iterdir(): p.unlink()
        resumed_native = SimpleNamespace(make_train_eval_datasets=noop, make_accelerator=noop,
            update_last_checkpoint=noop, make_pre_post_processors=make_pre_post_processors, make_policy=noop, WandBLogger=object)
        hooks.install(resumed_native, settings, noop)
        resumed_pre, resumed_post = resumed_native.make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=saved, dataset_stats=ft_stats,
            preprocessor_overrides={"normalizer_processor": {"stats": ft_stats}},
            postprocessor_overrides={"unnormalizer_processor": {"stats": ft_stats}},
        )
        torch.testing.assert_close(encode(resumed_pre), encoded, atol=0, rtol=0)
        torch.testing.assert_close(resumed_post(encoded), action, atol=1e-6, rtol=1e-6)
    finally:
        PolicyConfig.get_scheduler_preset = original_preset

