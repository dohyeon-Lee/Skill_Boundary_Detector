"""Load the pinned N1.7 implementation without importing a second LeRobot package."""
from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types
from pathlib import Path


def native_modules():
    root = next(p for p in Path(__file__).resolve().parents if (p / "third_party/lerobot_groot").is_dir())
    policies = root / "third_party/lerobot_groot/src/lerobot/policies"
    for name, path in (("lerobot.policies.common", policies / "common"),
                       ("lerobot.policies._skill_groot_native", policies / "groot")):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
            sys.modules[name] = module
    # N1.7 only needs these constants from its LeRobot policy configuration.
    # Do not register the vendor GrootConfig over the main package's GrootConfig.
    name = "lerobot.policies._skill_groot_native.configuration_groot"
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.N1_7_DEFAULT_IMAGE_CROP_SIZE = (230, 230)
        module.N1_7_DEFAULT_IMAGE_TARGET_SIZE = (256, 256)
        sys.modules[name] = module
    try:
        model = importlib.import_module("lerobot.policies._skill_groot_native.groot_n1_7")
    except ImportError as error:
        raise ImportError("GR00T SkillVLA requires .venv_groot_lerobot; run bash setup_env.sh --groot-only") from error
    flow = importlib.import_module("lerobot.policies.common.flow_matching")
    return model, flow


def load_native(path):
    module, _ = native_modules()
    # Request HF's loading report before applying the vendor's tuning switches.
    # Refuse partially initialized backbones/experts rather than silently
    # training random weights when a base checkpoint is incomplete.
    model, report = super(module.GR00TN17, module.GR00TN17).from_pretrained(
        str(path), load_backbone_weights=False, use_flash_attention=False,
        transformers_loading_kwargs={"trust_remote_code": True}, output_loading_info=True,
    )
    problems = {key: report.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                if report.get(key)}
    if problems:
        raise RuntimeError(f"GR00T pretrained weights did not load completely: {problems}")
    model.backbone.set_trainable_parameters(tune_llm=True, tune_visual=True)
    model.action_head.set_trainable_parameters(tune_projector=True, tune_diffusion_model=True, tune_vlln=True)
    return model
